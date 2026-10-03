# Fixed-layout MPI, OpenMP, and SIMD Lennard-Jones execution

D[cpu-hybrid-lj]. Experimental build-tree tool; not a `mdir run` backend.

## Implementation boundary

`tools/mdir-cpu-lj/main.cpp` owns input, decomposition, MPI transport, JIT
construction, and output. It generates a storage-form `md_exec.pair_for` for
Lennard-Jones energy, force, and virial. `convert-md-exec-to-loops` lowers that
kernel; fixed-order reductions, OpenMP conversion, and LLVM conversion follow.
The executable does not implement a second C++ force kernel.

MPI is called directly by this reference executable. There is no generated
`mpi` dialect, distributed planner, general field-version verifier, or
`md_dist` execution in this implementation. The particle path does not use
`shard`: its ghost list is an irregular subset of particle IDs, not a dense
tensor halo. No claim of multi-node scalability or faster production simulation
follows from this prototype.

| Source | Responsibility |
|---|---|
| `include/mdir/Dialect/MDExec/MDExecOps.td`, `lib/Dialect/MDExec/MDExecOps.cpp` | Imported neighbor syntax and local verifier |
| `lib/Conversion/MDExecToLoops/MDExecToLoops.cpp` | Owned/local extent checks, imported-row runtime validation, outer CPU loop |
| `lib/Conversion/MDExecToLoops/CPUVectorPairs.cpp` | Restricted neighbor-lane widening, guarded tails, lane sums |
| `lib/Conversion/MDExecKernels/MDExecKernels.cpp` | Existing scalar path and shared cutoff policy |
| `tools/mdir-cpu-lj/main.cpp` | Snapshot, MPI map/transport, MLIR generation, JIT and output |
| `scripts/validation/cpu-hybrid-lj.py` | Independent numerical oracle and launch matrix |

## Build and invocation

Configure a separate build as for MDIR and add `-DMDIR_ENABLE_MPI=ON`.
The tested toolchain uses LLVM/MLIR 23.1.2, GNU C++ 9.5.0, and Open MPI
4.1.2. CMake's `FindMPI` chooses `MPI::MPI_CXX`. Use the launcher from that same MPI
installation. Open MPI and MPICH are alternative builds of this backend, not
interchangeable runtime libraries for an existing executable.

```sh
cmake -S /path/to/mdir -B /path/to/build [usual MDIR options] -DMDIR_ENABLE_MPI=ON
cmake --build /path/to/build --target mdir-cpu-lj mdir-opt
mpiexec -n 2 /path/to/build/bin/mdir-cpu-lj snapshot.txt \
  --threads=2 --simd-width=4 --precision=double
```

| Option | Default | Meaning |
|---|---|---|
| Positional snapshot path | Required | Read-only text input; rank zero parses it |
| `--precision=double\|mixed` | `double` | Pair arithmetic precision |
| `--threads=N` | `1` | Positive OpenMP thread count, assigned to `OMP_NUM_THREADS` |
| `--simd-width=1\|4\|8` | `4` | Neighbor-lane vector width; 1 uses the original scalar lowering |
| `--emit=source\|loops\|llvm` | Absent | Print generated IR instead of evaluating the kernel |

`--emit` still parses and distributes the snapshot, then prints only on rank
zero. `loops` stops after fixed-order reductions, before OpenMP conversion;
`llvm` prints LLVM dialect MLIR, not textual LLVM IR. The tool creates no output
files. Shell redirection controls overwrite behavior. It is not installed as
part of the production driver and introduces no TOML keys.

The input header is `N Lx Ly Lz cutoff sigma epsilon`, followed by exactly `N`
rows `id x y z`. Units are consistent reduced LJ units. IDs are unique signed
64-bit integers, with no contiguity requirement. Coordinates are finite and
canonical: zero inclusive to the corresponding positive edge exclusive.
Sigma and epsilon are strictly positive; the cutoff is positive and strictly
less than half every edge. The potential is truncated without an energy shift,
switch, or long-range correction. Empty systems and empty ranks are valid.
Malformed input, unsupported options, duplicate IDs, coincident particles, and
constants that cannot be represented in the selected precision, and
nonfinite numerical outputs fail through `MPI_Abort`. There is no recoverable
capacity retry. MPI counts and local indices are limited to signed 32-bit
ranges; allocation capacity remains bounded by host memory.

Output consists of `energy E`, `virial Wxx Wxy Wxz Wyx Wyy Wyz Wzx Wzy Wzz`,
and `force ID Fx Fy Fz` rows. Force rows follow rank ownership order and then
input order within each rank; compare by ID, not line position. The virial is
the sum of displacement outer force for each unordered pair, with the same
sign convention as MDIR. It is not stress or pressure and includes no kinetic
term.

## Ownership and transport, in execution order

1. Initialize MPI with `MPI_THREAD_FUNNELED`; reject a weaker returned level.
   Only the initializing thread calls MPI. OpenMP regions are confined to the
   generated kernel, which completes before results are gathered.
2. Rank zero assigns each particle to the periodic x slab containing it,
   orders IDs and coordinates by destination, and scatters them. Metadata and
   owned counts are broadcast. All ranks participate, including empty ranks.
3. Each owner selects particles for every other destination slab using the
   minimum periodic x distance to that slab. The threshold is cutoff plus
   `32 * epsilon_float * Lx`, a conservative transport margin. This margin
   changes available ghost storage, not the interaction cutoff.
4. Exchange per-destination counts with `MPI_Alltoall`. Pack IDs and f64
   coordinates in independent vectors, then exchange each with blocking
   `MPI_Alltoallv`. Direct peer selection supports a cutoff spanning more
   than one slab; it is not restricted to two immediate neighbors.
5. Append ghosts after owned coordinates. The owner is unique and each
   destination receives at most one canonical copy of an ID. Minimum images
   are evaluated in the generated kernel; payload coordinates are not shifted.
6. Construct an all-pairs candidate row for every owned center: all local
   indices in increasing order except the center. The physical cutoff is
   applied inside the kernel. This costs quadratic local candidate storage
   and enumeration and deliberately avoids a spatial neighbor implementation.
7. Evaluate owned rows, reduce energy and virial with `MPI_Reduce`, and gather
   owned force arrays with `MPI_Gatherv`. Directed owner-computes needs no
   reverse force communication. No particle is integrated or migrated.

For a separation vector from the neighbor to the center, the generated kernel
forms $(\sigma/r)^2$, then its third and sixth powers. The pair energy is
$4\epsilon[(\sigma/r)^{12}-(\sigma/r)^6]$. Differentiating this radial energy
gives the center force $24\epsilon[2(\sigma/r)^{12}-(\sigma/r)^6]/r^2$
times that separation vector. The nine
virial components are its outer product with the force. This is the ordinary
Lennard-Jones energy derivative, with no force modification at the cutoff.

Each owned unordered pair is evaluated twice globally. Energy and virial have
weights 1/2; each center's force has weight 1. IDs support result assembly and
routing inspection; the kernel accesses local indices. The halo buffers,
candidate arrays, coordinate arrays, force arrays, and JIT engine remain alive
through the evaluation. Communication is complete before the kernel starts.
There are no pending requests, buffer borrows, or overlap in this baseline.

## Compiler changes

`md_exec.neighbor_view` imports host i32 counts and entries plus an index
`local_size`, producing the existing `!mdrt.neighbors` type. Its rows specify
owned centers, independent of the length of the position buffer. The checked import declares memory reads, so it cannot be treated as a pure
descriptor and hoisted across writes that initialize its input arrays. The op
verifier checks ranks, element types, host memory space, and compatible static
row counts. CPU lowering emits runtime assertions for matching dynamic rows,
`local_size >= owned`, row counts within capacity, indices within range,
strictly increasing indices, and exclusion of each center. Pair lowering
checks coordinate and input extents against `local_size`, and output capacity
against the owned count. Imported views support only `pair_for`; refresh,
reference positions, other neighbor consumers, and GPU lowering are refused.

These checks do not prove geometric coverage, field freshness, or immutability
under arbitrary writes through aliases. Those are caller obligations. The
reference executable constructs the buffers and makes no intervening writes.
The descriptor is not a replacement for the planned distributed field contract.

The new `simd-width` pass option defaults to 1, preserving the production
pipeline. Width 4 or 8 opts into `CPUVectorPairs.cpp`. The restricted kernel
supports scalar floating constants; floating add, subtract, multiply, divide,
negate, extend and truncate; vector broadcast, constant one-dimensional
extract, and construction from elements. Constants must be inside the kernel;
`ins`, arbitrary regions, table lookups, and unsupported operations are rejected.
This is a restricted compiler transformation, not general loop vectorization.

For each owned row, the lowering advances by the chosen width. Each lane
performs guarded index and position loads; the geometry is assembled into
vectors across neighbors. Original vector components become separate lane
vectors. Arithmetic then operates on these vectors. Invalid tail lanes never
load an entry; excluded or out-of-cutoff lanes use a safe squared distance of
one before division, and their contributions are masked to zero. Lane sums
are reduced at the end of the row, then written only to the owned force slot.
The outer center loop becomes OpenMP worksharing. Fixed-order reductions sum
center contributions independently of the OpenMP scheduling order.

## Precision and reproducibility

Coordinates, displacement subtraction before conversion, force storage, and
MPI payloads are f64 in both modes. Double mode computes pair arithmetic in
f64. Mixed mode converts displacements to f32, computes minimum images and LJ
arithmetic in f32, then explicitly widens yielded force, energy, and virial.
Width 4/8 accumulates these widened values in f64 lane accumulators. Width 1
uses the existing scalar path, which retains f32 row sums for widened global
energy/virial yields. This difference is intentional and tested numerically;
SIMD widths do not promise identical bits.

The SIMD path shares the existing `createCutoff2` helper, including the D159
inward f32 cutoff margin. The transport margin does not undo this numerical
policy. Oracle fixtures avoid the rounding neighborhood of the cutoff.
Fixed rank count, layout, precision and width are checked for identical output
with one and two OpenMP threads. Different ranks and widths may regroup sums;
MPI reduction order is not a bitwise portability guarantee.

## Validation and reproduction

```sh
python3 scripts/validation/cpu-hybrid-lj.py /path/to/build/bin/mdir-cpu-lj
# A six-configuration MPI smoke test is also part of lit when MPI is enabled.
```

The standard-library Python oracle enumerates each unordered pair once and
explicitly searches periodic images, independently of the kernel's minimum
image implementation and directed traversal. It compares every force component,
energy, and all nine virial components. Four malformed snapshots additionally check duplicate IDs, coincident
particles, an invalid cutoff, and noncanonical coordinates, with bounded
process timeouts. Its own force convention is checked
against 24 central energy differences with step `1e-6` and absolute tolerance
`2e-7`. Numerical comparison uses `abs(error) <= atol + rtol * abs(reference)`:
`atol=rtol=2e-11` for double and `3e-5` for mixed.

The matrix has six fixtures (periodic, empty ranks, empty system, reversed ID
order, nonunit sigma/epsilon, and a pair spanning two slabs), two precisions, ranks 1/2/5, threads 1/2, and widths 1/4/8: 216 runs.
It includes noncontiguous IDs, a cutoff larger than a rank's slab width, and
partial SIMD vectors. In the two-slab fixture, particles at x=2.3 and
x=4.85 interact across rank 1 when five ranks partition the length-12 box;
exchanging only immediately adjacent ranks would omit that pair. It does not test migration, skin reuse, trajectories,
production potential files, EAM, or distributed GPU execution. Detailed results
and regression status are recorded in the PR; numerical success is not a
performance benchmark.

On the tested host, all 216 numerical configurations (180 initial and
36 additional nonadjacent-rank configurations) and the four rejected-input
checks passed. The periodic fixture reference energy is
$-4.141983094297806$; its particle 101 force is
$(0.2513201688046216, 0.41962282931214345, 0.13421382654072408)$.
The nonunit-parameter fixture reference energy is $-1.8554057299171132$.
Maxima over all configurations, rounded to three significant digits:

| Quantity | Double absolute error | Mixed absolute error |
|---|---:|---:|
| Energy | 8.88e-16 | 7.41e-7 |
| Force component | 4.33e-15 | 1.03e-5 |
| Force component RMS | 1.24e-15 | 5.80e-6 |
| Virial component | 5.33e-15 | 1.20e-5 |

An earlier validation launch was interrupted by relinking its executable;
no numerical comparison failed in that launch. The completed matrix used a
separate executable, held unchanged throughout its run. The assertion-output
regression initially needed unbuffered stdout to observe the diagnostic before
an intentional abort; the corrected test checks both the abort and its message.

Generated IR can be inspected with `--emit=loops`. To inspect a specified CPU
target independently of the JIT's host selection:

```sh
mpiexec -n 1 /path/to/build/bin/mdir-cpu-lj snapshot.txt --emit=llvm > kernel.mlir
mlir-translate --mlir-to-llvmir kernel.mlir > kernel.ll
llc -O3 -mattr=+avx2,+fma kernel.ll -o kernel.s
```

Representative default CPU storage and GPU kernel fixtures produce identical
lowered IR to the clean main build at `d1def6b6b38b9e059e3b982e878158cb2b50253e`.
That comparison is a code-generation regression check, not a timing result.

This inspection produced packed-double `vdivpd` and `vmulpd` instructions for
width 4. It verifies vector code generation for that target, not a measured
speedup or a guarantee about every host's instruction selection.

## Follow-up contract

The roadmap remains [cpu-hybrid-plan.md](cpu-hybrid-plan.md): semantic
requirements and contribution completion, staged EAM, temporal validity and
migration, then asynchronous transport. Reuse `mdrt.event` for logical
completion rather than equating it with `MPI_Request`. Source release,
consumer readiness, memory visibility, and execution location are separate
facts. UCX and NVSHMEM are future backend candidates; neither is implemented.
