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
matrix.

| Step | Work | Provided by |
|---|---|---|
| 1 | Compute the cell of each particle | Generated particle loop |
| 2 | Count particles per cell | Generated particle loop |
| 3 | Turn counts into cell offsets | Runtime: exclusive scan |
| 4 | Fill the list of particles, ordered by cell | Generated particle loop |
| 5 | Reorder the fields by cell | Generated particle loop |
| 6 | For each particle, test the particles of the 27 surrounding cells and store those within `r_c + skin` | Generated loop with a generated predicate |
| 7 | If a row of the matrix is full, report it | Runtime: overflow report |

Steps 2 to 4 are a counting sort. On a GPU, step 2 needs atomic increments
or a sort by key; the runtime provides a sort by key for that case.

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
| Element type | From the precision policy |
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

Proposal: OpenMP for M0, because it works today. The choice is confined to
the lowering of parallel loops, so it can be replaced.

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

## 6. Reference interpreter

| Option | For | Against |
|---|---|---|
| A C++ tool, `mdir-interpret` | Uses the IR data structures directly. No new dependency. | Array code is more verbose than in Python. |
| Python with NumPy | Concise | Needs the MLIR Python bindings, which were not built, or a second parser for the IR. |

The interpreter evaluates `md` and `dyn` ops from their definitions, in
double precision. A neighborhood is found by testing all pairs.

Proposal: the C++ tool.

## 7. Questions

| # | Question | Proposal |
|---|---|---|
| 1 | Neighbor structure for M0 | Neighbor matrix on both targets |
| 2 | CPU threading | OpenMP for M0 |
| 3 | GPU execution | Upstream `gpu` dialect; runtime functions in `mdrt` |
| 4 | Reference interpreter | C++ tool |
| 5 | Vector field layout for M0 | `memref<?x3xT>`; the other layout later |
