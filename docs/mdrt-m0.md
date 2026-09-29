# Runtime and Execution for Milestone M0: Proposal

Status: proposal (2026-09-29). Nothing here is decided except where a
decision is cited.

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

M0 uses one algorithm on both targets: binning into cells, then a neighbor
matrix. [neighbors-m0.md](neighbors-m0.md) describes the method as it is
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
| Memory | `N × W`, sized for the densest particle | Exact |
| Overflow | Possible; the build is repeated with a larger `W` | Not possible |

The matrix is proposed for M0 because the target is homogeneous systems
(C7), where rows have nearly equal length. PPMD uses the same structure on
GPUs.

### 2.3 In the IR

```mlir
%cells = md_exec.build_cells %x, %cell { width = 2.8 } : !mdrt.cells
%order = md_exec.spatial_order %cells : !mdrt.permutation
%xs    = md_exec.permute %x, %order : !vec
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

The GPU back end is a second lowering of the storage form
(ops-m0.md, Section 10.7), next to `convert-md-exec-to-loops`.

| Part | Plan |
|---|---|
| Loop over particles or pairs | One `gpu.launch`, with one thread per particle. With the policy `owner_only` a thread writes only to its own particle, so no atomic operation is needed. |
| Buffers | Device memory, from `gpu.alloc`. The state is copied to the device before the first step of a segment and back after the last. |
| Global sums | The kernel writes the contribution of each particle to a device buffer. A second kernel adds them up in blocks, and the host adds up the blocks. The order is fixed, so the sum is reproducible. |
| Neighbor build | On the device, binning included. The particles of a cell are sorted by index after the fill, because the order in which threads take their slots is not fixed. |
| Where a buffer is | Decided by `md-exec-assign-storage`. A device buffer has a memory space in its type, so that host code cannot use it by mistake. `mdrt.from_buffer` uploads and `mdrt.to_buffer` downloads. |
| Buffers that a lowering needs for itself | Given to the loop by `md-exec-assign-storage`, from the pool of its region. A lowering does not allocate in the step loop (D18, B10). |
| Test of validity | On the device: the largest displacement is a global maximum, computed like a global sum. |
| Runtime functions | The ones the upstream lowering calls, in `libmdrt`, built only if the CUDA toolkit is found. |

Open questions:

| # | Question | Proposal |
|---|---|---|
| 1 | Neighbor build on the host first | Decided: no. The build runs on the device from the start. A build on the host would cost about as much per step as the forces. |
| 2 | Size of a block of threads | 128, a numeric plan parameter |
| 3 | Which GPU | The first one that `CUDA_VISIBLE_DEVICES` leaves visible |

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
| Neighbor build as a template in IR, `lib/Runtime/Templates/NeighborsMatrix.mlir` | Implemented. The compiler adds it to the module, where it is lowered with the rest of the code. See [neighbors-m0.md](neighbors-m0.md). |
| Neighbor matrix | Implemented |
| Overflow of a row | The runtime reports it and stops the run. Rebuilding with wider rows is not implemented. |
| Storage in `memref<?x3xT>` | Implemented, for `f32` and `f64` |
| Neighbor build for positions of `f32` | Implemented, as an instance of the template with the type replaced |
| `mdrt.from_buffer`, `mdrt.to_buffer` | Implemented. They connect code that works on buffers with code that works on fields. |
| OpenMP | Works through the upstream lowering of `scf.parallel`. Reductions work. |
| Storage form of the loops | Implemented |
| Threading as a plan parameter | Not implemented. The choice is made by the passes that are run after lowering. |
| Runtime library `libmdrt` | One function, the overflow report |
| Spatial reordering (step 5) | Not implemented |
| GPU | Implemented for NVIDIA. See [ops-m0.md](ops-m0.md), Section 10.8. |
| Runtime library `libmdrt_cuda` | The functions that the lowering of the `gpu` dialect calls, on the CUDA driver API. One stream serves all launches. The host waits only where it reads what the device has computed. |

The runtime library for devices reads these variables of the environment:

| Variable | Meaning |
|---|---|
| `MDRT_DEVICE` | The device to run on, among those that `CUDA_VISIBLE_DEVICES` leaves. The default is 0. |
| `MDRT_PROFILE` | If set, the library reports the number and the time of its calls when the program ends. |
| `MDRT_WAIT` | If set, the host waits after every launch. With `MDRT_PROFILE`, the report has the time of each kernel. |

The template is sequential in its counting sort and parallel in its search
for neighbors.

## 8. The storage form

D17 decided that `md_exec` ops have a value form and a storage form. Both
exist. The pass `md-exec-assign-storage` converts the value form to the
storage form, and `convert-md-exec-to-loops` turns the storage form into
loops on the CPU. See [ops-m0.md](ops-m0.md), Sections 8.5 and 10.

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
