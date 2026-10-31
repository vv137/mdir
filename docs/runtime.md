# Runtime and Execution for Milestone M0: Proposal

Status: written as the proposal for M0 (2026-09-29); Section 7 and the
table of ops below are kept current (last checked 2026-10-21). Elsewhere,
nothing is decided except where a decision is cited.

The ops of the `mdrt` dialect:

| Op | Purpose |
|---|---|
| `mdrt.from_buffer`, `mdrt.to_buffer` | Connect code that works on buffers with code that works on fields; on a device, the copy to and from the host. |
| `mdrt.host_call` | Calls a function of the host that reads fields while it runs and keeps nothing of them: the writing of a frame, for example. On a device, the field is copied to a buffer of the host for the call. |
| `mdrt.halo_copy_start`, `mdrt.halo_sum_start`, `mdrt.halo_wait` | The start of a halo copy $G$ or a halo sum $G^\mathsf T$ of `md_dist` as an event, and its completion, which yields the destination; the verifier proves the buffers' lifetimes, and `--lower-mdrt-halo-events` lowers them to the synchronous in-process transport (D310, [distributed-ml2.md](distributed-ml2.md)). |

This document proposes how the M0 program is executed: what the compiler
generates, what the runtime provides, and how the two meet. It is the basis
for the `mdrt` ABI and for the lowering of `md_exec`.

Scope of M0: one process, one node, CPU threads or one GPU, a fixed number
of particles, and an orthorhombic periodic cell.

## 1. What runs where

```text
driver (host)
  │  owns the system, the plan, input and output
  │
  ├── compile: md, dyn → md_exec → code          once, before the run
  │
  └── loop over segments
        ├── run_segment(context, n)               compiled code
        │     ├── rebuild neighbor structure      generated loops
        │     │     └── sort, scan, allocate      runtime primitives
        │     ├── forces                          generated kernels
        │     └── kick, drift                     generated kernels
        └── output, checkpoint, retuning          driver
```

Neighbor build kernels are generated from the first milestone (D8, D27).
The runtime provides only the generic primitives they need (P9).

## 2. Neighbor build

### 2.1 Algorithm

M0 uses one algorithm on both targets: binning into cells
[[Quentrec1973]](references.md#quentrec1973), then a neighbor matrix. [neighbors.md](neighbors.md) describes the method as it is
implemented; the table is the proposal.

| Step | Work | Provided by |
|---|---|---|
| 1 | Compute the cell of each particle | Generated particle loop |
| 2 | Count particles per cell | Generated particle loop |
| 3 | Turn counts into cell offsets | Runtime: exclusive scan |
| 4 | Fill the list of particles, ordered by cell | Generated particle loop |
| 5 | Reorder the fields by cell | Generated particle loop |
| 6 | For each particle, test the particles of the 27 surrounding cells and store those within `r_c + skin` | Generated loop with a generated predicate |
| 7 | If a row of the matrix is full, report it | Runtime: overflow report |

Steps 2 to 4 are a counting sort. On a GPU, steps 2 and 4 use atomic
additions, and the particles of each cell are sorted by index afterward, so
that the result does not depend on the order of the threads.

### 2.2 Neighbor matrix

```text
count[i]            number of neighbors of particle i
index[i][0 .. W)    their indices; entries from count[i] on are unused
```

`W` is the row width, a numeric plan parameter. The list is directed: if `j`
is in the row of `i`, then `i` is in the row of `j`.

| Property | Neighbor matrix | Compressed rows |
|---|---|---|
| Parallel build | Each row is written independently | Needs a scan over row lengths first |
| Inner loop | Fixed bound, no indirection for the row start | Variable bound |
| Memory | $N \times W$, sized for the densest particle | Exact |
| Overflow | Possible; the build is repeated with a larger `W` | Not possible |

The matrix is proposed for M0 because the target is homogeneous systems
(C7), where rows have nearly equal length. PPMD [[Saunders2018]](references.md#saunders2018) uses the same
structure on GPUs.

### 2.3 In the IR

```mlir
%order = md_exec.spatial_order %x, %cell, %ids { width = 1.4 }
           : !mdrt.permutation
%xs    = md_exec.permute %x, %order : !vec
%cells = md_exec.build_cells %xs, %cell { width = 2.8 } : !mdrt.cells
%nl    = md_exec.build_neighbors %cells, %xs, %cell
           { cutoff = 2.5, skin = 0.3, kind = matrix, width = 96 }
           : !mdrt.neighbors
```

Each of these ops lowers to generated loops and calls to runtime primitives.
None lowers to a single opaque runtime call.

## 3. Storage

| Item | Proposal |
|---|---|
| Field storage | A contiguous buffer, lowered to `memref` |
| Vector fields | `memref<?x3xT>` or `memref<3x?xT>`, chosen by the plan |
| Element type | From the precision policy. Decided (D32): the driver allocates the state in the types of the policy, and the buffers state those types to the compiled program. |
| Owner | The runtime allocates and frees; compiled code receives the buffers |
| Host access | The driver reads and writes buffers between segments and increments the version counter of what it writes (P16) |

The context passed to `run_segment` holds the buffers of the state, the
neighbor structure, and the numeric plan parameters.

## 4. CPU threading

| Option | For | Against |
|---|---|---|
| OpenMP, through the upstream `omp` dialect | The lowering exists upstream. Reductions are supported. | Needs the LLVM OpenMP runtime at run time. A host process that already loaded another OpenMP runtime can conflict with it. |
| Thread pool in the runtime | No dependency. No conflict with the host. | The pool, the partitioning, and the reductions must be written. |

The LLVM OpenMP runtime is not part of the LLVM build that
`scripts/build-llvm.sh` produces. It builds on its own from the same source
tree in a few minutes, without rebuilding LLVM. A trial build worked, and a
parallel loop lowered through the `omp` dialect ran on 16 threads with the
expected result.

The conflict matters for the Python library (C5): NumPy and PyTorch load
OpenMP runtimes of their own.

Decided (D28): OpenMP for M0. The choice is confined to the lowering of
parallel loops, so it can be replaced.

Threading is a structural plan parameter.

| Value | Lowering | OpenMP runtime |
|---|---|---|
| `openmp` | Parallel loops through the `omp` dialect | Loaded |
| `none` | Sequential loops | Not loaded |

Because compilation happens before each run, the value is chosen per run.
`none` is the way to avoid a conflict when the process hosts a framework
with an OpenMP runtime of its own, as a learned potential evaluated by an
external framework does. It is also the natural value for a GPU target,
where compiled CPU loops do little work.

## 5. GPU execution

| Option | For | Against |
|---|---|---|
| Upstream `gpu` dialect, with its lowering to runtime calls | Kernel outlining, serialization to PTX, and launch code exist upstream. | The runtime functions it calls must be provided. |
| Project-owned launch path | Full control | Everything must be written. |

The upstream lowering emits calls to a small set of functions (module load,
kernel launch, memory, streams). Upstream implements them in a wrapper
library over the CUDA driver API, which was not built. It can be built with
the CUDA toolkit on this machine, or reimplemented in `mdrt`.

Embedding the kernels as PTX text needs no CUDA toolkit at compile time: the
driver compiles PTX when the module is loaded.

Proposal: the upstream `gpu` dialect and its lowering, with the runtime
functions provided by `mdrt`.

### 5.1 What was tried

On this machine: four RTX 3090, driver 595.84, CUDA toolkit 11.2, and LLVM
23.1.2 built without any knowledge of CUDA.

| Step | Result |
|---|---|
| `gpu.launch` to a kernel in PTX text, with `gpu-lower-to-nvvm-pipeline` and the format `isa` | Works. The toolkit is not needed for this step unless the kernel calls a math function. |
| Math functions in a kernel | Work once `CUDA_ROOT` names the toolkit: `libdevice` is linked as bitcode. |
| Loading and launching | The driver compiles the PTX text when the module is loaded. |
| The upstream wrapper library over the CUDA driver API | Its sparse part needs a newer toolkit. Without that part it builds against toolkit 11.2 and works. It is about 300 lines. |
| `gpu.alloc`, `gpu.memcpy` | Lower only in their asynchronous form, with tokens. |
| A kernel of the shape of a loop over pairs: one thread per particle, a loop over the others, vectors of `f64`, `roundeven`, `fpowi`, `exp`, `erfc` | Ran on a GPU. The sum over 1000 particles agrees with the host. |
| Atomic operations: `memref.atomic_rmw` in a kernel | Work. |
| A counting sort of 100000 keys on the device: a histogram and a fill with atomic operations, a scan, and a sort of each cell by index | The order is the one that the host computes, and the same in every run. |
| A buffer of the type `memref<?xf32, 1>`, with a memory space | Works through the whole lowering. Device buffers can differ in type from host buffers. |

So the proposal is feasible as it stands. It was decided (D34) and
implemented.

### 5.2 Plan

*The plan of M0, kept as it was.* The lowering as it is now is described
in [ops.md](ops.md), Section 10.8, and in the description of
`convert-md-exec-to-gpu` (`include/mdir/Conversion/Passes.td`): loops over
pairs take 16 lanes a particle or groups of 16, global sums are added by
blocks, and the test of validity is a flag rather than a global maximum.

The GPU back end is a second lowering of the storage form
(ops.md, Section 10.7), next to `convert-md-exec-to-loops`.

| Part | Plan |
|---|---|
| Loop over particles or pairs | One `gpu.launch`, with one thread per particle. With the policy `owner_only` a thread writes only to its own particle, so no atomic operation is needed. |
| Buffers | Device memory, from `gpu.alloc`. The state is copied to the device before the first step of a segment and back after the last. |
| Global sums | The kernel writes the contribution of each particle to a device buffer. A second kernel adds them up in blocks, and the host adds up the blocks. The order is fixed, so the sum is reproducible. |
| Neighbor build | On the device, binning included. The particles of a cell are sorted by index after the fill, because the order in which threads take their slots is not fixed. |
| Where a buffer is | Decided by `md-exec-assign-storage`. A device buffer has a memory space in its type, so that host code cannot use it by mistake. `mdrt.from_buffer` uploads and `mdrt.to_buffer` downloads. |
| Buffers that a lowering needs for itself | Given to the loop by `md-exec-assign-storage`, from the pool of its region. A lowering does not allocate in the step loop (D18, B10). The scratch of the reciprocal sums of PME is allocated once in the body of the function and shared by every sum that needs a buffer of the same shape (D315, [pme.md](pme.md), Section 3.4). The partial sums of a global sum take 1024 rows, allocated once in the body of the function and shared by its sums by type, not a row a particle (D[reduction-partials], #711). |
| Test of validity | On the device: the largest displacement is a global maximum, computed like a global sum. |
| Runtime functions | The ones the upstream lowering calls, in `libmdrt`, built only if the CUDA toolkit is found. |

Open questions:

| # | Question | Proposal |
|---|---|---|
| 1 | Neighbor build on the host first | Decided: no. The build runs on the device from the start. A build on the host would cost about as much per step as the forces. |
| 2 | Size of a block of threads | 128, a numeric plan parameter |
| 3 | Which GPU | The first one that `CUDA_VISIBLE_DEVICES` leaves visible |

### 5.3 Transfer readiness and lifetime (design for #567)

Storage assignment currently emits an asynchronous `gpu.memcpy` token
between `gpu.wait async` and an immediate `gpu.wait`. Thus each copy is
complete before the next host access, while the blocking wait is a
synchronization point. In `runtime/mdrt_cuda.c`, `mgpuMemcpy` also calls
`finish()` for a host-device copy; `finish()` synchronizes both runtime
streams when work is pending. An IR token alone does not remove that
runtime-wide wait.

The first #567 slice checks the existing behavior with a read-only
transfer record (`md-exec-check-transfers`, D358). The
record associates the completion token with
the exact source and destination storage roots and slices, their allocation
generations, and the visibility domain. The source is read-borrowed and the
destination write-borrowed from initiation until completion. A consumer
may read the destination only after a completion dependency with the right
visibility; a conflicting write, free, or reuse of either storage waits for
the borrow to end. An SSA token definition dominating a use is not itself
completion. Proved disjoint work may run between start and wait.

| Event | What it establishes | What it does not establish |
|---|---|---|
| Transfer start | An in-flight copy and source/destination borrows | Ready destination data |
| Transfer completion/dependency | Data visible to the stated consumer domain; transfer borrows end | Initialization of unwritten elements or release of a later host writer's borrow |
| Host writer release (D336) | Staging storage may be reused | Checkpoint file committed |
| File commit (D336) | Checkpoint visibility under its existing rename protocol | Permission to skip a required copy dependency |

The verifier and lowering use the same provenance, alias, and effect
queries from D334 and D338. D337 supplies storage-set
identity, D342 supplies checked loop effects, and D355 supplies declared
function and program resource effects; a buffer's set tag does not prove
that aliases are disjoint. The initial verifier is bounded to recognized
same-block transfer/wait pairs and checked loop/function effects. It refuses
a known conflicting use without completion, and diagnoses an unsupported
contract where required alias or completion facts remain unknown. It does
not turn an unresolved dependency into a presumed safe serial path.

The checker runs after storage assignment and GPU conversion, including
reduction readbacks. GPU conversion may remove
readbacks with `keepReadbacksOnDevice` or move and combine their copy/wait
groups with `deferReadbacks`; the final groups are checked after those
rewrites, after stack-release insertion. The same query also checks the
input to GPU conversion. While `md_exec` loops exist, the checker uses
D342's exact operand
and row summary; after conversion it uses the conservative projection to
MLIR memory effects and generic alias tracking. A token operand or a marker
on a copy is not a substitute for a checked completion path. Positive cases
include the emitted immediate-wait
sequence and the existing deferred readback sequence; negative cases put a
read, conflicting write, or deallocation between a copy and its wait,
including through a recognized alias.

The record is analysis state, not printed IR. Its source and destination
slices retain their SSA descriptors, with casts and recognized views traced
to a storage root. The root SSA identity denotes its generation for this
activation or loop iteration; rejecting every intervening free prevents
reuse of that generation. Static byte envelopes prove disjoint slices only
when their offsets, extents, strides, and element sizes are known. An
unknown slice is not a disjoint slice, and a free concerns the whole root,
even through a disjoint view. Distinct allocations and D87's checked root
provenance can prove separation; different set tags cannot.

Only a blocking same-block `gpu.wait` whose dependencies include the copy
token, directly or through asynchronous waits or dependent copies, ends its
host-side borrows. A copy dependency orders that copy's accesses after its
predecessor, and its token carries completion to a later blocking wait. An async
wait alone does not establish host visibility. Synchronous copies complete
at the copy operation. The destination memory space determines host or
device visibility; an unsupported memory-space representation is diagnosed.
The checker uses D342 loop summaries only when complete. It checks defined
functions with actual-argument substitution and rejects recursion and
unrecognized control flow. External calls need a valid D355 `dyn.effects`
summary with no default-memory effect. The parser and named resource singletons are shared through `MDIREffects`,
so execution checks do not depend on the `dyn` dialect library.
Named resources remain distinct from
buffer storage. Loop-carried aliases inside a substituted call remain
unsupported where their relation to the caller's buffers is unresolved.

The first implementation retains the existing waits, copy placement,
checkpoint release/commit points, and failure propagation. It shares the
readiness and borrow contract with `mdrt.halo_copy_start`,
`mdrt.halo_sum_start`, and `mdrt.halo_wait` (#481), whose current in-process
transport and same-block, one-wait restriction remain. Moving waits,
merging readbacks, pinned-memory transport, CUDA graph capture, and actual
MPI/GPU halo overlap are separate measured strategies with explicit
preconditions. A change that removes `mgpuMemcpy`'s runtime-wide `finish()`
must establish scoped stream dependencies and host visibility before it may
claim overlap. No control key, output format, or result changes in this
first slice.

The opt-in [device-resident FFT contract](fft-device-resident.md)
(#580, D360) uses completion events scoped to its producer
queue at host boundaries. Existing `mgpuMemcpy` callers retain their waits;
the new path must establish host visibility and release before reuse instead
of relying on the runtime-wide `finish()`.

## 6. Reference interpreter

V1 decided against an interpreter. The options that were considered:

| Option | For | Against |
|---|---|---|
| A C++ tool, `mdir-interpret` | Uses the IR data structures directly. No new dependency. | Array code is more verbose than in Python. |
| Python with NumPy | Concise | Needs the MLIR Python bindings, which were not built, or a second parser for the IR. |

The interpreter evaluates `md` and `dyn` ops from their definitions, in
double precision. A neighborhood is found by testing all pairs.

Proposal: the C++ tool.

## 7. What is implemented

| Item | State |
|---|---|
| Neighbor build as a template in IR, `lib/Runtime/Templates/NeighborsMatrix.mlir` | Implemented. The compiler adds it to the module, where it is lowered with the rest of the code. See [neighbors.md](neighbors.md). On a device the templates are `NeighborsMatrixGPU.mlir` and, for groups of 16 with each pair once, `NeighborsGroupsGPU.mlir` ([neighbor-groups.md](neighbor-groups.md)); particle mesh Ewald has `PME.mlir` and `PMEGPU.mlir` ([pme.md](pme.md)). |
| Neighbor matrix | Implemented |
| Overflow of a row | The runtime holds the rows (`mdrtMatrixCreate`, `mdrtHostMatrixCreate`) and makes them a quarter wider than needed (`mdrtMatrixGrow`); the build is made again. Compiled code takes the rows where it uses them (`mdrtMatrixEntries`). |
| Storage in `memref<?x3xT>` | Implemented, for `f32` and `f64` |
| Neighbor build for positions of `f32` | Implemented, as an instance of the template with the type replaced |
| `mdrt.from_buffer`, `mdrt.to_buffer` | Implemented. They connect code that works on buffers with code that works on fields. |
| OpenMP | Works through the upstream lowering of `scf.parallel`. Reductions work. |
| Storage form of the loops | Implemented |
| Threading as a plan parameter | Not implemented. The choice is made by the passes that are run after lowering. |
| Runtime library `libmdrt` | The host side of a run (`runtime/mdrt.c`): the counts of builds, of prunings of inner lists, of those done by the loop over pairs (D284), and of late builds (`mdrtCountBuild`, `mdrtCountPrune`, `mdrtCountFusedPrune`, `mdrtCountLateBuild`) and the statistics of the lists of groups (`mdrtNoteGroups`, of complete builds, with their growths after the first build of their structure, D331) that the log reports; the stop of a run whose positions are not numbers or are far outside the cell (`mdrtStopNotNumbers`, D107, D225) and of one whose constraints failed in a step, with the step and the set (`mdrtStopConstraintFailure`, D278), before which, in a step with constraints, the first stops are recorded (`mdrtDeferStops`, `mdrtStopPending`); the generator Philox 4×32-10 and the factors of the thermostat and the barostat drawn from it (`mdrtPhilox4x32`, `mdrtBussiFactor`, `mdrtBarostatStrain`); the FFT of the host, from pocketfft, for blocks of the lines of a grid in f64 or f32 (`mdrtFFTPrepare`, `mdrtFFTColumnBlocks`, `mdrtFFT64RowsForward`, `mdrtFFT64RowsBackward`, `mdrtFFT64Columns` and their `32` forms, `runtime/mdrt_fft.c`, D271); and the neighbor matrix of the host (`mdrtHostMatrixCreate`, `mdrtHostMatrixGrow`, `mdrtHostMatrixEntries`) |
| Spatial reordering (step 5) | Implemented, where a run and where a segment begins (D44); see [neighbors.md](neighbors.md), Section 5 |
| GPU | Implemented for NVIDIA. See [ops.md](ops.md), Section 10.8. |
| Runtime library `libmdrt_cuda` | The functions that the lowering of the `gpu` dialect calls (`mgpu*`), on the CUDA driver API: modules of PTX compiled by the driver at load, launches that skip empty grids, and a caching allocator whose frees do not wait. One stream serves all launches, and an optional second one the reciprocal sum (`mdrtSideBegin`, `mdrtSideEnd`, `mdrtSideJoin`, D81, D87). The host waits only where it reads what the device has computed; a flag of a test is read through pinned memory and an event (`mdrtFlagStart`, `mdrtFlagFinish`, D113), and the other flags that the kernel of the event set as they are (`mdrtFlagTake`, D278). The buffers of the neighbor matrix and of the lists of groups, grown when a build overflows (`mdrtMatrixCreate`, `mdrtMatrixGrow`, `mdrtGroupsCreate`, `mdrtGroupsGrow`, `mdrtGroupsBuffer`); the lists of groups start with one block a group, so that their first build sizes them, and `mdrtGroupsBuilt` marks a structure built and hands the log its growths since (D331); the transforms of cuFFT in f32 and f64 with cached plans, one for each grid, direction, and precision of the process, in a table that grows (`mdrtCudaFFTForward3D`, `mdrtCudaFFTBackward3D` and their `F32` forms); a failure of cuFFT goes to the handler of the embedding program (`mdrtCudaSetStopHandler`, as `mdrtSetStopHandler` of `libmdrt`), or ends the process with status 1 (#323). |
| Memory of the device | A caching allocator: a block that a program frees is kept for a later request of the same size, without waiting, since the work it is handed to runs after the work that used it on the one stream. The driver's own use outside it (the context, modules, cuFFT's plans, cuBLAS, the store of the tuples) is reported with the pool's blocks by `MDIR_DEVICE_MEMORY=2` (D311, [debugging.md](debugging.md)); #471 measures where it goes on large systems. The buffers of a neighbor structure (`mdrtGroupsCreate`, `mdrtMatrixCreate`) are freed when the program ends, or when an activation closes (D215). The structure of the start and the structure of the steps are one storage on a device (D330, [ops.md](ops.md), Section 10.5). The registry of the blocks grows with them, so that every block is taken back when its activation closes; the idle blocks are bounded in bytes (`MDIR_DEVICE_POOL_BYTES`), and so are the workspaces of the cached plans of cuFFT (`MDIR_FFT_PLAN_BYTES`, the least recently taken evicted). A request that the driver refuses for want of memory gives the idle blocks and the plans back and is made again, and `mdir.trim_device_memory()` gives them back between simulations (D348, #622). |
| Products of learned stages | `mdrtGemmF32`, cuBLAS's f32 GEMM and strided-batched GEMM on the stream of the kernels, without TF32; the weights that the products read are copied to the device when first read and freed with the activation or the program that made them (`mdrtDeviceConstant`, D302). |
| Tuples on a device | The renumbered tuples and their incidence structures, in ragged rows (D314), are a store of the program that runs the code, built on the device (D253, D257, [device-tuples.md](device-tuples.md)). |
| Transport of domains | `runtime/mdrt_dist.c`: the halo copy and the halo sum of the in-process executor (`mdrtDistHaloCopy`, `mdrtDistHaloSum`), synchronous ([distributed-ml2.md](distributed-ml2.md)). |
| Kernels | Cubins for the device's architecture, which the driver loads without compiling them; PTX where the device cannot be asked (D214). |

The runtime library for devices reads these variables of the environment:

| Variable | Meaning |
|---|---|
| `MDRT_DEVICE` | The device to run on, among those that `CUDA_VISIBLE_DEVICES` leaves. The default is 0. |
| `MDRT_PROFILE` | If set, the library reports the number and the time of its calls when the program ends. |
| `MDRT_WAIT` | If set, the host waits after every launch and checks it, so that a failure names the kernel that failed. With `MDRT_PROFILE`, the report has the time of each kernel. |
| `MDRT_TRACE` | If set, the library prints each module that it loads, the kernels of each module, and each launch with its grid. |

The template is sequential in its counting sort and parallel in its search
for neighbors.

## 8. The storage form

D17 decided that `md_exec` ops have a value form and a storage form. Both
exist. The pass `md-exec-assign-storage` converts the value form to the
storage form, and `convert-md-exec-to-loops` turns the storage form into
loops on the CPU. See [ops.md](ops.md), Sections 8.5 and 10.

The first implementation lowered the value form directly to loops. The two
steps were separated before the GPU back end, so that the back ends share
the assignment of buffers and differ only in how they turn a loop over
particles or pairs into code.

The generated code did not change: the run times are the same as before.

## 9. Questions

| # | Question | Proposal |
|---|---|---|
| 1 | Neighbor structure for M0 | Decided: neighbor matrix (D29) |
| 2 | CPU threading | Decided: OpenMP for M0 (D28) |
| 3 | GPU execution | Decided and implemented (D34) |
| 4 | Reference interpreter | Decided: none (V1) |
| 5 | Vector field layout for M0 | Decided: `memref<?x3xT>` first (D30) |
| 6 | Storage form of `md_exec` | Decided and implemented (D17, D33, A12) |

Neighbor-group buffers use the shared slot descriptors in `mdir/Runtime/GroupsABI.h` (D350). Their explicit numbers, element kinds, shapes, and roles form one compiler/runtime contract. The runtime validates each generic slot before indexing storage; the host compile cache includes the contract version. Allocation ownership and teardown are unchanged.
