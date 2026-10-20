# The primitives of the device lowering

This document describes the parallel primitives that the lowering for a
device composes kernels from, in the form they take today: what each does,
what must hold for it, where that is checked, how to turn it off, which
kernels use it, what it gains, and where the idea comes from. MDIR
generates its device code through MLIR and does not link CUB. It reads
CUB's block and warp algorithms (part of CCCL, which ships with CUDA) and
implements them as lowering strategies with stated preconditions and
switches ([principles.md](principles.md), principles 2 and 3; issue #415).
Which part of MDIR uses which primitive, with the primitive's transpose
and the properties the part relies on, is in
[primitives-map.md](primitives-map.md) (D292); this document
describes how a device implements a primitive.

## 1. Status

| Primitive | CUB's counterpart | Issue | Status | Decision |
|---|---|---|---|---|
| Privatized scatter-accumulate: `tile`, `hash` (Section 2) | `BlockHistogram`, atomic specialization | #416 | Built. One attribute, one pass, and one set of preconditions (step 1). The table `hash` is one emitter that every site calls (`ScatterEmitter.h`, step 3). Its sites are the tuples, the op `md_exec.scatter_add`, and the loops over pairs that add at both ends (over groups, and `(directed, atomic)` over a matrix), where it was measured slower and is off by default (Section 2.7). The tile of PME keeps its template. Fixed point in the deterministic mode (step 2) is planned | D282 (`tile`), D283 (`hash`), D285 (the attribute and the pass), D291 (the op, the emitter, the groups) |
| Sorted scatter: runs reduced without atomics | `BlockHistogram`, sort specialization; `BlockDiscontinuity` | #417 | Planned. `sorted` is reserved in the attribute and refused | — |
| Warp-aggregated atomics | `WarpReduce`, matching by key | #418 | Planned | — |
| Vectorized and transposed gathers | `BlockLoad`, `WarpLoad`, `BlockExchange` | #419 | Next, on the gathering of PME (tiled gathering at orders 6 and 8 is #411) | — |
| Single-pass prefix sums | `BlockScan`; device scan with decoupled look-back | #420 | Planned | — |
| Radix sorts: tuples by first member, binning by key | `BlockRadixSort`; device radix sort (onesweep) | #421 | Next | — |
| Reductions chosen by the algebra of the operator (Section 5) | `BlockReduce` (raking, commutative only) | #422 | Step 1 built, behavior-neutral: the algebra of every sum stated in the IR (`any_order_exact`, `any_order`, `fixed_order`), read by the chooser and the lowerings instead of the mode, with the same PTX; the warp and block reductions one emitter. The strategies it licenses follow only where measured | D301 |
| Compaction, scans, and counting sorts as shared emitters (Section 4) | `WarpScan` over flags (a ballot), `WarpScanShfl`, `BlockScan` (`BLOCK_SCAN_WARP_SCANS`); the scans of counting sorts | #432 | Built, behavior-neutral: one emitter for each, which the lowering calls and the templates call by name; every site of #432's inventory moved, with the same PTX | D296 |
| L2 access policy window: persisting lines for the buffers a step reads again (Section 3) | none: the CUDA programming guide's L2 cache control [[CUDAGuideL2]](references.md#cudaguidel2) | #434 | Measured, not adopted: no gain on either device, up to 14% slower on the RTX 4090 | — |

The order follows #415: #421 first, then #419 on the gathering of PME,
then #417 and #422 for the deterministic mode, and #418 and #420 where
the profile points.

## 2. The privatized scatter-accumulate

### 2.1 What it does

A scatter-accumulate adds values $v_k$ into a field at indices $i_k$ that
the threads share, $a_i \leftarrow a_i + \sum_{k : i_k = i} v_k$. The
plain form gives each value an atomic addition of the device. The
privatized form gives each block of threads a copy of the part of the
field that its values touch, in block memory. The block adds its values
there, with atomics of block memory, and then adds each touched index to
the field once, with one atomic of the device. The atomics of the device
fall from one a value to one a distinct index a block. This is the atomic
specialization of CUB's `BlockHistogram` [[CUB]](references.md#cub), the
privatized histogram whose cost [[GomezLuna2013]](references.md#gomezluna2013)
models, with a value in place of a count.

Two strategies differ in how a block finds its copy of an index. Here $P$
is the order of the stencil of PME, and the slots that a search of a
table tries are written `P`, as in the attribute:

| Strategy | The copy of a block | For a footprint that |
|---|---|---|
| `tile(E)` | A dense tile: the bin of $E^3$ points of the grid and the $P - 1$ points of the stencil past it on each axis, $(E + P - 1)^3$ values | the IR bounds (the stencil of order $P$ and the bin) |
| `hash(S, P, L)` | A table of $S$ slots keyed by the index, found by a multiplicative hash and the next slots in turn [[Knuth1998]](references.md#knuth1998), claimed by a compare-and-swap. A search gives up after `P` slots, and the value goes to the field with an atomic of its own. The host takes the table from $L$ items of a launch on | has no bound (the members of tuples) |
| `atomic` | None: an atomic of the device for each value | — |

`sorted` (#417) is reserved.

### 2.2 In the IR

The strategy is the attribute `#md_exec.scatter`, written `scatter(...)` on
the ops whose lowering scatters (`include/mdir/Dialect/MDExec/MDExecAttrs.td`):

```mlir
%u, %w, %f = md_exec.reciprocal %x, %q, %cell, %moduli
               grid([256, 128, 128]) order(4) beta(0.39) coulomb(332.06)
               scatter(tile(8)) : ...
%f1 = md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1))
        outs(%f0 : !md.field<@atoms, 3 x f32>) arity(2)
        scatter(hash(256, 8, 100000)) { ... }
%f2 = md_exec.pair_for %nl, %x, %cell outs(%f1 : !md.field<@atoms, 3 x f32>)
        cutoff(8.0) exchange [antisymmetric] policy(unique, atomic)
        scatter(hash(512, 8, 0)) { ... }
%a1 = md_exec.scatter_add %v at %idx into %a0 scatter(hash(128, 8, 100000))
        : !md.field<@ghosts, 3 x f64>, !md.field<@ghosts, i32>
          -> !md.field<@atoms, 3 x f64>
```

`md_exec.scatter_add` is the scatter-accumulate on its own (D291,
ruled on #445). It computes $a'_i = a_i + \sum_{k : \mathrm{idx}_k = i} v_k$
over the rows $k$ of the values and the indices:
- The values and the indices are fields of one set, the indices of `i32`.
- The destination has the components and the type of the values.
- In the storage form the op updates the destination where it is, and
  `overwrite(true)` starts it from zeros.

The op is for values that no loop computes in registers. Its first site
is the sum of the ghost rows into their owners ($G^\mathsf T$,
`halo_sum`, #443), then a stored sum from edges to nodes (#377, step 4).
Where a loop computes the values, the loop's destinations take the
strategy instead (`tuple_for`, `pair_for`): a buffer of one value a pair
would add a write and a read of every value.

On the CPU the op is a loop with plain additions on one thread, or a
parallel loop with atomic ones (`convert-md-exec-to-loops{atomic-grid}`).
A combiner other than addition is to come as `combine(...)`. The op
states the algebra of its sum (`md_exec.algebra`, Section 5): the device
and the host add with atomics only where it is `any_order`.
`md-exec-assign-precision` does not take the op yet. A program in mixed
precision keeps it out of the fields it narrows; #443's transport stays in
f64.

Without the attribute, or with `scatter(atomic)`, a site lowers as it does
without privatization. The verifiers refuse a strategy that the op's
lowering does not have:
- `tile` on `md_exec.tuple_for`, `md_exec.pair_for`, and
  `md_exec.scatter_add`: the members of a tuple, the entries of a list,
  and the indices may be anywhere, so no footprint is bounded;
- `hash` on `md_exec.reciprocal`: the template of the spreading has the
  kernels of tiles alone;
- `hash` on a `pair_for` whose conflict is not `atomic`: a loop that
  writes only at its centers scatters nothing. Both `(unique, atomic)`
  and `(directed, atomic)` (D290) take it, over groups and over the rows
  of a matrix.

The pass `md-exec-choose-scatter` sets the attribute on programs for a
device, after the precision is assigned and before the storage, which
sizes the bins of the tiles from it. `md-exec-assign-storage` carries the
attribute onto the storage form. A loop that the storage makes overwrite a
destination writes the rows of its particles and loses it.

Why PME keeps its template and is not a `scatter_add`: the spreading of
PME is a kernel of the runtime template
(`lib/Runtime/Templates/PMEGPU.mlir`, written by
`scripts/generate-pme-gpu-template.py`), in which the compiler instantiates
the addition to the tile. Moving it onto the op would mean generating the
spreading instead of the template, and storing a value for every point of
every stencil.

### 2.3 Preconditions, and where they are checked

Each precondition is one function of `include/mdir/Dialect/MDExec/Scatter.h`.
The pass calls it to choose, and the lowering calls it again to apply.

| Precondition | Strategies | Function | The pass | The lowering |
|---|---|---|---|---|
| The values may be added in another order: integers in any order, floating-point numbers outside the deterministic mode only, since a sum of a block first rounds otherwise. Fixed point is not privatized yet (step 2), so only f32 and f64 are | `tile`, `hash` | `checkReorder` | refuses | fails, naming the reason |
| The footprint of a tile is bounded and wraps around the grid once at most: every edge of the grid has at least $E + P - 1$ points. The template has kernels of $E$ = 4, 8, and 16 | `tile` | `checkTile` | refuses | fails |
| The copy fits 48 KiB of block memory, the static limit of a block from compute capability 7.0 on, which neither kernel exceeds: $(s^3 + 4T)\,b + 12T$ bytes for a tile ($s = E + P - 1$, $T$ = 128 threads, or 512 for $E$ = 16, $b$ the bytes of a value, with the particles of a round); $4S + \sum_d S\,c_d\,b_d$ for a table of $S$ slots and destinations of $c_d$ components | `tile`, `hash` | `getTileBytes`, `getHashBytes` | refuses | a tile fails. The table of a *run* of loops, launched as one kernel with the union of their destinations, is checked with the same function and keeps the atomics where it does not fit: runs are formed in the lowering, after the pass |
| A fallback exists: every destination of a table is a number or a row of a static length of f32 or f64, which have an atomic of the device | `hash` | `getHashDestination` | refuses | fails |
| The site scatters: a `pair_for` whose conflict is `atomic`; a `scatter_add` always | `hash` | the verifier, the pass | refuses, with the reason | — |
| Occupancy: bounded by the 48 KiB above. Neither strategy asks for dynamic block memory, and the measurements below were taken at that bound | `tile`, `hash` | — | — | — |

The strategy applies only where the lowering scatters at all. A loop over
tuples takes its table only where it evaluates each tuple once: not in the
deterministic mode (where the rows of the particles take the tuples), not
for a disjoint set, not with global sums (whose kernel takes several
tuples a thread for the rows of its sums, D103), and not in a run that
holds the forces of a term over centers (D194). The pass leaves the
attribute off the first three, and the lowering keeps the atomics in the
last.

### 2.4 The choice

Where the preconditions hold, the pass chooses by cost:

- **`tile`** for a sum in f32 whose grid has at least 500,000 points
  (`tile-min-points`). A smaller grid has too few bins to occupy a device:
  at 512 bins (JAC, 64³ points) the tiles took 1.004 to 1.008 times the
  step of the bricks of D108 at order 4 on an RTX 4090 and 1.05 on an RTX
  PRO 6000; at 27 to 112 bins (the prolate kernel at ε = 10⁻³ on JAC and
  Factor IX), 1.98 and 1.17. From 1,008 bins up they were faster on both
  devices. A grid in f64 spreads faster without tiles on the devices
  measured, whose f64 is slow and whose atomics of 64 bits in block memory
  are a loop of compare-and-swap. The threshold is the same for every
  device; a rule by the number of multiprocessors would tile more of the
  RTX 4090's systems.
- **`hash`** for every loop that can take it, with $S$ = 128 × the arity
  rounded up to a power of two (room for every member of a block's
  tuples; 512 slots and 8 KB for a run with dihedrals and forces in f32),
  `P` = 8 slots tried, and $L$ = 100,000. Below 100,000 tuples the device holds every
  block at once, and the table lengthens each one: with the table JAC
  (55,000 tuples) ran 1.5% slower on an RTX 4090 and 3% slower on an RTX
  PRO 6000, Factor IX (165,000) 2.3% and 1% faster. The count is known on
  the host only, so the lowering emits both kernels and the host compares.
  Without the bound of 8 slots a block whose tuples share few particles
  fills its table: one block of JAC with 494 particles in 512 slots took
  the kernel from 12 to 42 µs; with the bound, 13.2.

- **`hash` on `scatter_add`**: $S$ = 128 slots (one value a thread) and
  $L$ = 100,000 (`hash-least`), the threshold of the tuples, since no site
  of the op has been measured yet; below $L$ values a launch keeps the
  atomics.
- **`hash` on the loops over pairs that add at both ends** (over groups, and
  `(directed, atomic)` over a matrix): off (`pair-hash=off`), and `auto`
  chooses the atomics too, measured faster on every system (Section 2.7). The slots would be
  $\mathrm{PowerOf2Ceil}(3T)$: one particle of the group and two entries
  a thread for each unit of work.

With `explain`, the pass puts a remark on each op that states the choice
or the reason for a refusal.

### 2.5 Switches

| Switch | Effect |
|---|---|
| `md-exec-choose-scatter{tile=auto\|on\|off}` | `auto` the rule above; `on` every sum that can be tiled; `off` none |
| `tile-edge`, `tile-min-points` | The points of an edge of a bin (4, 8, 16; 8 by default) and the threshold of `auto` |
| `hash=auto\|on\|off` | `on` takes the table at any count ($L$ = 0) |
| `hash-least`, `hash-probes`, `block-size` | $L$, the slots tried `P`, and the threads of a block, as `convert-md-exec-to-gpu` launches them |
| `pair-hash=auto\|on\|off`, `pair-hash-least` | The table of the loops over pairs that add at both ends: `off` by default. `auto` chooses the atomics at every such loop, with a remark that cites the measurement (Section 2.7), so that `auto` and the default agree; `on` takes the table for comparison. $L$ counts units of work |
| `scatter-hash=auto\|on\|off` | The table of `md_exec.scatter_add`. `auto`, the default, sets `hash(128, 8, L)` with $L$ = `hash-least` (100,000): a launch of fewer values keeps the atomics, chosen on the host by the count. No site of the op has been measured yet, so $L$ is the threshold of the tuples (D283), whose values per index are similar for `halo_sum`; it is to be measured with #443's site. `on` takes the table at any count, `off` never |
| `deterministic`, `explain` | The mode, which privatizes no floating-point sum; the remarks |
| `MDIR_PME_SPREAD` | `global` is `tile=off`; `tiles` is `tile=on`; `tiles4`, `tiles8`, `tiles16` set the edge as well; `auto` is the default ([environment.md](environment.md)) |
| `MDIR_TUPLE_SUMS` | `atomics` is `hash=off`; `blocks` is `hash=on`, which the tests of small systems use |
| `MDIR_PAIR_SUMS` | `atomics` is `pair-hash=off`; `blocks` is `pair-hash=on` |

The tests compare each strategy with the atomics:
`pme-tiles-gpu.test` (every edge, orders 4, 6, and 8, the prolate kernel,
a triclinic cell, a slab whose bins overflow, LJ-PME, mixed and double)
and `tuple-block-sums-gpu.test` (mixed against atomics and against
double), both also under compute-sanitizer. `choose-scatter.mlir` checks
each choice and refusal of the pass, `Dialect/MDExec/scatter-invalid.mlir`
the verifiers, and `Conversion/MDExecToGPU/scatter-invalid.mlir` the
lowering's refusals. The tests of D291:
- `choose-scatter-pairs.mlir` and `choose-scatter-add.mlir` check the
  pass on the new sites.
- `Conversion/MDExecToGPU/group-scatter.mlir` and `scatter-add.mlir`
  check the emitter's output: the attributions of the table, the
  clearing, the compare-and-swap, the additions in block memory, the
  fallback, the barriers, the flush, and the host's choice by the count.
- `Integration/eam.mlir` (CPU) and `eam-groups-gpu.test` run EAM over
  groups with each strategy, in double and mixed precision, against
  `Inputs/eam_reference.py`.
- `Integration/dist-exec-scatter-gpu.test` runs the EAM of
  `dist-eam-exec.mlir` on 3x2x1 domains with the table in its two-ended
  loops over a matrix, in double and mixed precision, against the
  references of that test.
- `Integration/scatter-add.mlir` and `scatter-add-gpu.test` run the op
  sequentially, under OpenMP, and on a device with each strategy. They
  compare exactly against the sum written out.
- `Sanitizer/group-scatter-gpu.test` runs the groups' table under
  memcheck, initcheck, and racecheck. `scatter-add-gpu.test` runs the
  op's table under memcheck and racecheck (with `-Dsanitize=1`).

### 2.6 Sites

| Site | Strategy | Kernel | Described in |
|---|---|---|---|
| The spreading of PME on a device: the charges, the coefficients of LJ-PME (D162), the prolate kernel (D266), orthorhombic and triclinic cells | `tile(8)` | `mdrt_gpu_pme_bin<E>`, `mdrt_gpu_pme_spread_tiles<E>` of the template; the addition to the tile is a relaxed `atomicrmw fadd` at the scope of the block (`PME_SHARED_ADD_E`, `MDExecKernels.cpp`) | [pme-m1.md](pme-m1.md#33-spreading-through-tiles), Section 3.3 |
| The kernel of the tuples evaluated once: the bonded terms, the 1-4 pairs, the excluded pairs of the reciprocal sum | `hash(512, 8, 100000)` for a run with dihedrals | `launchTupleBlocks` in `MDExecToGPU.cpp`, through the emitter | [device-tuples.md](device-tuples.md), Section 11 |
| `md_exec.scatter_add`: no site on `main` yet; `halo_sum` of #443 is the first | `hash(128, 8, 100000)` | `lowerScatterAdd` in `MDExecToGPU.cpp`, through the emitter | Section 2.2 |
| The loops over groups, each pair once (D89): the nonbonded forces, EAM's density and its forces | none by default; `hash(512, 8, 0)` with `pair-hash=on` | `lowerGroupPairFor` and `emitGroupPairKernel`, through the emitter | [groups-m1.md](groups-m1.md) |
| The two-ended loops over the rows of a matrix, `(directed, atomic)` (D290): the domains of `md_dist`, the edge scatter of #377 step 4 | none by default; `hash(512, 8, 0)` with `pair-hash=on` | `lowerPairFor`, `launchRows`, and `emitPairKernel`, through the emitter | [distributed-ml2.md](distributed-ml2.md), Section 7.2 |

**The emitter** (D291). The table is one class,
`kernels::BlockHashTable` (`include/mdir/Conversion/ScatterEmitter.h`, in
the library of the kernels). It comes with the helpers of block memory
that it uses: `getSharedAddress`, `emitSharedAdd` (`red.shared.add.f32`
and `.f64`), and `emitSharedCompareSwap` (`atom.shared.cas.b32`). A site
uses it in four steps:
1. It constructs the table in its launch, with the workgroup attributions
   that `getHashAttributions` gives.
2. It calls `emitClear`, which ends with a barrier.
3. It calls `add(out, index, value)` for each value. The slot is found
   once for an index in a block of the IR, and the value goes to the
   device atomic where the search fails.
4. It calls `emitFlush`: a barrier, then one device atomic for each index
   that the table holds.

The site keeps what is its own: how its threads take their items, and the
host's choice by $L$.

Moving the tuples onto the emitter changed no generated code. Two outputs
are byte-identical to those of `main` (1690b051) on JAC NVE, JAC NPT,
Factor IX NVE, and Cellulose NVE: the module after
`convert-md-exec-to-gpu`, and the program as executed with the PTX of
every kernel (`mdir emit --stage=lowered`, `MDIR_GPU_BINARY=ptx`). The
tile of PME keeps its own addition in its template, a relaxed `atomicrmw
fadd` at the scope of the block.

### 2.7 Measured effect

The step with the strategy against without, mixed precision, the settings
of the Amber suite, alternated within one job on one node (details and
spreads in the documents of the sites):

| System | `tile(8)`, order 4: RTX 4090 | RTX PRO 6000 | `hash`: RTX 4090 | RTX PRO 6000 |
|---|---:|---:|---:|---:|
| JAC (not chosen: below both thresholds) | 1.007 | 1.050 | 1.000 | 0.998 |
| Factor IX NVE | 0.960 | 0.995 | 0.974 | 0.989 |
| Cellulose NVE | 0.947 | 0.983 | 0.907 | 0.964 |
| STMV NPT 4 fs | 0.943 | 0.970 | 0.973 | 0.980 |

These are ratios of time a step: below 1 is faster. The `tile` column is
from the timings of D282 (µs a step). The `hash` column is the inverse of
the ns/day ratios of D283. At orders 6 and 8 and with the prolate kernel
the tiles take 0.77 to 0.87 times the step on the RTX 4090 and 0.91 to
0.97 on the RTX PRO 6000. The atomics of the device fall, on Cellulose,
from 24 million a step to at most 11 million for the spreading, and from
three a member to about a quarter of that for the tuples (a block of 128
tuples touches 0.24 distinct particles a member).

**One attribute and one pass (D285).** Moving both sites
onto the attribute changed no kernel. The module after
`convert-md-exec-to-gpu` is identical to that of `main` on JAC NVE and
NPT, Factor IX NVE, Cellulose NVE, and STMV NPT 4 fs, apart from the line
that prints the options of the pass, and the PTX of JAC NVE and NPT (245
and 326 kernels) is identical. The Amber suite, alternated against `main`
in one job on an RTX 4090, three rounds, the median ns/day ratio:

| System | Ratio | System | Ratio |
|---|---:|---|---:|
| JAC NVE | 0.997 | Factor IX NVE | 1.001 |
| JAC NVE 4 fs | 1.000 | Factor IX NPT | 1.001 |
| JAC NPT | 0.997 | Cellulose NVE | 0.999 |
| JAC NPT 4 fs | 1.002 | Cellulose NPT | 1.000 |
| STMV NPT 4 fs | 1.001 | | |

JAC NVE and NPT, again in five rounds on another node, gave 1.001 and
0.999.

**The loops over groups (D291).** The table at the $i$ and $j$
ends of the loops over groups was measured on an RTX 4090 at 450 W
(ada-003), in one job, with the variants alternated. The ratio is the
time of a step or of an evaluation with the table, divided by the time
with the atomics; above 1 is slower.

| System | Atomics | Table | Ratio |
|---|---:|---:|---:|
| JAC NVE, mixed (ns/day, 2 rounds) | 1308 | 1113 | 1.18 |
| Factor IX NVE, mixed | 586 | 473 | 1.24 |
| Cellulose NVE, mixed | 151.0 | 116.6 | 1.30 |
| JAC NVE, every pruning in the loop (`MDIR_FUSED_PRUNE=always`, a warp a group) | 1266 | 1032 | 1.23 |
| Cellulose NVE, the same | 149.4 | 113.3 | 1.32 |
| EAM, 262,144 particles, mixed (ms an evaluation, 3 rounds) | 0.415 | 0.616 | 1.49 |
| EAM, the same, double | 6.19 | 6.49 | 1.05 |
| EAM, mixed, blocks of 512 threads (2048 slots) | 0.586 | 0.931 | 1.59 |
| EAM on one domain through `md_dist` (D290), 110,592 particles: the two loops `(directed, atomic)` over the rows of a matrix, mixed (µs a call, the two kernels, Nsight Systems, 11 calls) | 423 | 873 | 2.06 |
| The same, double | 908 | 1522 | 1.68 |

EAM is the potential of `Integration/eam.mlir` on $64^3$ sites of a
lattice of spacing 1.0, each moved by up to 0.1 along each axis, which
gives about 33 neighbors a particle. Its density and its forces are loops
over groups. With the table, the energy, the virial, and the sum of the
squared forces agree with those of the atomics to $10^{-10}$ relative in
mixed precision, which is the spread between two runs with atomics, and
to $3\times10^{-16}$ in double.

The table gains nothing at this site because a block's values have
almost no index in common:
- The four warps of a block of 128 threads take consecutive units of
  work. These are blocks of 64 entries of one group's list, and a list
  holds each particle once, so the $j$ values of a block are 128 distinct
  particles a round.
- The $i$ values are already summed over the half-warps before their one
  atomic a unit.

With nothing to merge, the table only adds its compare-and-swap. The
compare-and-swap returns a value that the thread waits for, while the
device's `red` would have been issued without waiting. Blocks of 512
threads, which span more groups, were slower still.

The two-ended loop over the rows of a matrix (#443's `(directed,
atomic)`, the form of #377 step 4's two-ended scatter) does merge: the 8
rows of a block are neighbors in the order of the cells and share most of
their entries. It was slower all the same, 2.06 times in mixed precision
and 1.68 times in double. Reuse alone does not make a table pay. The
measurements point to contention as the condition: the entries of the
rows send the atomics of a warp to different addresses, which the device
handles cheaply, and the table's lookup is paid on every value. The
tuples, where the table pays, have the members of a block's tuples
named by several threads at once (above). Contention itself was not
measured: Nsight Compute's counters are refused on the cluster's nodes.
The switch stays for comparison and is off by default at both loops over
pairs.

**The emitter and the op against `main` (D291).** By default
the branch generates the code of `main`. The Amber suite was run with
`main` (0f6da67d) and the branch alternated in one job on an RTX 4090,
three rounds. The table gives the median ns/day of the second half of
each run, and the ratio of the time a step to that of `main`:

| System | `main` | Branch | Ratio |
|---|---:|---:|---:|
| JAC NVE | 1310.1 | 1310.2 | 1.000 |
| Factor IX NVE | 586.7 | 586.5 | 1.000 |
| Cellulose NVE | 152.0 | 152.0 | 1.000 |
| STMV NPT 4 fs | 89.2 | 89.3 | 0.999 |

### 2.8 Results and the deterministic mode

A privatized sum adds the same values in another order. In the default
mode the results agree with the atomics to the rounding of the sums, as
two runs with atomics agree with each other: $10^{-7}$ to $10^{-6}$ of the
RMS force in mixed precision, $10^{-16}$ to $10^{-15}$ in double (the
tables of D282 and D283). The deterministic mode privatizes nothing and
gives the same bits as before. Its grid of PME and its forces are in fixed
point, added by a thread for each particle (D70, D84), and its tuples are
taken by the rows of their particles. Fixed-point additions are exact in
any order, so a privatized fixed-point sum would keep the mode bitwise;
that is step 2 of #416.

## 3. The L2 access policy window (measured, not adopted)

### 3.1 What was tried

From compute capability 8.0 on, a part of the L2 cache can be set aside for
lines marked *persisting*. A stream can carry one window of addresses. A
fraction of the accesses in the window, the hit ratio, is marked
persisting, and the rest is marked *streaming*, which is evicted first
[[CUDAGuideL2]](references.md#cudaguidel2). Step 1 of #434 asked whether
keeping one buffer that a step reads many times in L2 speeds up the step.
It changed no kernel.

For the measurement, the runtime set a window on its two streams over one
buffer that an environment variable chose:

- the call `cuStreamSetAttribute` with
  `CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW`, with persisting hits and
  streaming misses;
- the set-aside `CU_LIMIT_PERSISTING_L2_CACHE_SIZE`, equal to the window
  and at most the device's `MAX_PERSISTING_L2_CACHE_SIZE`;
- the window, at most `MAX_ACCESS_POLICY_WINDOW_SIZE`, optionally capped to
  the first megabytes of the buffer;
- the hit ratio, the set-aside divided by the window.

The buffers windowed:

| Choice | Buffer |
|---|---|
| `outer` | The entries of the outer list of the groups (D114) |
| `outer@32`, `inner@32` | The first 32 MB of the entries of the outer or of the inner list |
| `placed` | The positions of the places in the frames of their groups in f32, which a loop over pairs after a pruning reads (D268) |
| `hot` | The block of the runtime's pool that the most launches took, counted from the launches' arguments over the first 2000 launches: on every system one of the blocks of $3N$ numbers in f64 |

The switch is not kept. The code is described on #434.

### 3.2 Measured effect

The settings:

- the Amber suite in mixed precision, the dual list of groups of 3 / 0.6 Å
  beyond the cutoff at 2 fs (in proportion to the step), and twice the
  suite's steps;
- the rate of the second half of each run;
- the six variants rotated over 5 rounds, the first round dropped;
- one frozen binary, with only the variable differing;
- one job on one node for each device, on 2026-10-18:
  - an RTX 4090 at 450 W (ada-003): 72 MB of L2, at most 49.5 MB set
    aside, a window of at most 128 MB;
  - an RTX PRO 6000 Blackwell Max-Q at 300 W (ada-004): 128 MB of L2, at
    most 80 MB set aside, a window of at most 128 MB.

The tables give the time of a step with the window divided by the time
without it; below 1 is faster.

**RTX 4090.**

| System | No window (µs/step) | `hot` | `placed` | `outer` | `outer@32` | `inner@32` |
|---|---:|---:|---:|---:|---:|---:|
| JAC NVE | 139.6 ± 0.5 | 0.998 | 0.999 | 0.999 | 0.998 | 1.000 |
| Factor IX NVE | 310.2 ± 0.2 | 0.999 | 1.003 | 1.001 | 0.999 | 1.005 |
| Cellulose NVE | 1178.1 ± 1.4 | 0.995 | 1.011 | 1.077 | 1.064 | 1.057 |
| STMV NPT 4 fs | 3997.7 ± 2.7 | 1.014 | 1.019 | 1.139 | 1.076 | 1.059 |

**RTX PRO 6000.**

| System | No window (µs/step) | `hot` | `placed` | `outer` | `outer@32` | `inner@32` |
|---|---:|---:|---:|---:|---:|---:|
| JAC NVE | 116.7 ± 0.1 | 0.999 | 0.999 | 0.999 | 1.000 | 0.999 |
| Factor IX NVE | 238.3 ± 0.2 | 0.999 | 1.001 | 0.999 | 0.999 | 0.998 |
| Cellulose NVE | 1033.6 ± 1.6 | 1.000 | 1.000 | 1.001 | 1.002 | 0.997 |
| STMV NPT 4 fs | 3311.1 ± 3.0 | 1.002 | 1.003 | 1.015 | 1.007 | 1.004 |

The windows covered the following:

| Buffer | Size on JAC and Factor IX | Window on Cellulose and STMV |
|---|---|---|
| The outer list | 5.75 and 23.6 MB, whole | 118 MB at a hit ratio of 0.42 on the RTX 4090 and 0.68 on the RTX PRO 6000; 128 MB of the 277 MB buffer at 0.39 and 0.63 |
| `placed` | 0.34 and 1.30 MB | 5.85 and 15.3 MB |
| `hot` | 0.54 and 2.08 MB | 9.35 and 24.4 MB |

On Cellulose, a build uses about half of the outer list's buffer, from its
start.

The 0.995 of `hot` on Cellulose on the RTX 4090 is within noise: in another
job, the step took 1.143 and 1.145 ms with the window and 1.140 and 1.143
without it.

### 3.3 Why it loses on the RTX 4090 and is neutral on the RTX PRO 6000

Nsight Systems timed the kernels of Cellulose on the RTX 4090: 600 steps,
two alternated rounds, µs a call.

| Kernel | No window | `outer@32` | `outer` |
|---|---:|---:|---:|
| The loop over pairs that prunes (D284), which reads the outer list | 563.5 | 556.1 | 559.7 |
| The loop over pairs | 534.6 | 536.5 | 535.0 |
| An integration kernel (`particle_for ... integration3`) | 70.3 | 125.4 | 120.5 |
| The gathering of PME | 43.3 | 46.3 | 46.0 |
| All kernels, 600 steps (ms) | 682.8 | 735.3 | 741.9 |

The window gains about 1% where it is read. The set-aside, though, takes L2
from the memory-bound kernels of the rest of the step, which on Cellulose
and STMV use all 72 MB of the RTX 4090. The integration kernel takes 70
→ 120–125 µs, and the step loses more than the reader gains. The RTX PRO
6000's 128 MB of L2 still hold the working set beside a set-aside of 80
MB, and the windows change its step by at most 0.3% on Cellulose and 1.5%
on STMV.

### 3.4 Limits of the measurement

- **The positions.** The runtime cannot follow them. Compiled code
  allocates them as any other buffer. Also, at each build, `md_exec.permute`
  writes the fields of the particles into new buffers, and the runtime's
  pool hands the freed blocks to the next permutation of the same size. The
  positions, the velocities, and the target of a permutation therefore
  take turns in three or four blocks that lie far apart: four blocks of 9.8
  MB span 2.3 GB on Cellulose. `hot` thus holds the positions for part of
  the run only. A window that follows them needs a mark from the compiler
  on the field where it is allocated.
- **Kernel-side hints were not measured.** Streaming loads of data that is
  read once, PTX's `createpolicy` with `ld.global.L2::cache_hint`, and
  `discard.global.L2` on scratch at the end of its life (the later part of
  #434) set no part of L2 aside and are a different mechanism.
- **No hit rates.** Nsight Compute's counters are refused on the cluster's
  nodes (`ERR_NVGPUCTRPERM`; the driver's `RmProfilingAdminOnly` is 1),
  so the measurement has kernel times, not L2 hit rates.

## 4. Compaction, scans, and counting sorts

### 4.1 What they are, and why they are shared

The builds of the neighbor structures and the pruning of the dual list
pack items, sum counts, and sort items by keys, each with code written for
its site. Issue #432 counted these sites and measured them: none reaches 3%
of a step, and they already use the cheapest form at the level of a warp,
so no faster strategy was found. The primitives are shared all the same, for
reuse (the maintainer, 2026-10-19; #444's rule of two sites): the code of a
primitive is written once, tested on its own, and taken by the sites to
come, the migration of a device (#38) and the groups on domains first.
Moving the sites changed no kernel (Section 4.5).

The emitters are in `include/mdir/Conversion/CompactionEmitter.h`
(`kernels::`, in the library of the kernels, beside `ScatterEmitter.h`):

| Emitter | What it emits | Preconditions | CUB's counterpart |
|---|---|---|---|
| `emitLanesBelow`, `emitWarpBallot`, `emitWarpRank`, `emitWarpCount` | The warp compaction: the vote of the warp on a predicate, $b = \mathrm{ballot}(p)$; the rank of lane $\ell$ among the lanes that keep, $r_\ell = \mathrm{popc}(b \wedge (2^\ell - 1))$; the count, $\mathrm{popc}(b)$. A kept lane writes its item at the count of the round plus $r_\ell$, so the items keep the order of the lanes | Every lane of the warp reaches the ballot; warps of 32 | a `WarpScan` over flags, done in the bits of one vote [[CUB]](references.md#cub) |
| `emitWarpScan` | The inclusive sum over segments of 2 to 32 lanes, in $\log_2$ rounds of `shuffle up` by $1, 2, 4, \ldots$ [[HillisSteele1986]](references.md#hillissteele1986) | Every lane reaches it. The lane is compared with the offset as a signed or an unsigned number, as the site has it | `WarpScanShfl` |
| `emitBlockScan` | The exclusive sum over a block, and its total in every thread: a warp scan in each warp, the sums of the warps through block memory, a warp scan of those by the first warp | A buffer of one i32 a warp and one for the total in block memory (`memref<9xi32>` for 256 threads); every thread reaches its two barriers | `BlockScan`, `BLOCK_SCAN_WARP_SCANS` |
| `emitDeviceScan` | The exclusive sums of $n$ counts into `start` ($n + 1$, the total last) and a cursor: blocks of 256 threads with 4 counts a thread sum their 1024 counts; one block scans the sums of the blocks; each block scans its counts from its offset. Three launches | The caller gives the buffer of the sums of the blocks, $\max(\lceil n/1024 \rceil, 1)$ of them | `DeviceScan` is one pass with decoupled look-back, #420; this is the reduce-then-scan form |
| `emitChunkedScan`, `emitSerialScan` | The same sums for a few thousand counts: a thread sums each chunk of 256, one thread scans the sums of the chunks (`emitSerialScan`), a thread writes the offsets of each chunk | — | — |
| `emitCursorScatter` | The scatter of a counting sort: each item takes the next slot of its key from the cursor with a device atomic and writes its number there; an item with a negative key has no bin, unless every key is one | The cursor holds the offsets of the keys (a scan above) | `BlockHistogram`'s atomic histogram, by key |
| `emitSortBins` | The items of each bin sorted by insertion, by the item or by a value of the item; a thread a bin | Bins of a few items | — |
| `emitRankInBins` | The stable rank: a thread for each item counts the smaller items of its bin and writes its item at that place | The items of a bin are distinct | — |
| `emitDeviceAtomicAdd` | A relaxed atomic addition of an i32 at the scope of the device, with its old value | — | — |

A counting sort is the site's histogram (one device atomic for each item
where the site computes its key), a scan of the counts, the scatter, and an
order within each bin, since the order of the atomics is not fixed: the
insertion sort for bins of a few items, the rank for bins of many (#231)
and for the partitions to come (#38, with the ghost flag in the key).

### 4.2 How the sites call them

The lowering calls the emitters where it builds its kernels. The templates
(`NeighborsGroupsGPU.mlir`, written by
`scripts/generate-groups-gpu-template.py`, and `NeighborsMatrixGPU.mlir`)
call them by name, with a declaration in the template:

```mlir
func.func private @mdir.primitive.warp_rank(i32, i32) -> i32
...
%rank = func.call @mdir.primitive.warp_rank(%kept, %below) : (i32, i32) -> i32
func.call @mdir.primitive.sort_bins(%key_start, %sorted, %keys, %grid_of_keys,
    %block, %c1) : (memref<?xi32, 1>, memref<?xi32, 1>, index, index, index,
    index) -> ()
```

`convert-md-exec-to-gpu` replaces each call with the code of the emitter
where it stands, when it parses the template (`expandPrimitiveCalls`), and
removes the declarations; the pass `expand-device-primitives` does the same
on a module. A primitive that does not exist, or a call whose operands or
results the primitive does not have, is an error that names it. The
parameters that are not values are attributes of the call: `lanes` and
`unsigned` of `warp_scan`, `every_key` of `cursor_scatter`. The names are
those of the emitters: `warp_lanes_below`, `warp_ballot`, `warp_rank`,
`warp_count`, `warp_scan`, `block_scan`, `device_scan`,
`device_scan_cursor`, `chunked_scan`, `cursor_scatter`, `sort_bins`,
`sort_bins_by`, `rank_in_bins`.

### 4.3 Sites

The inventory of #432, by the primitive each site takes:

| Site | Primitives | Where |
|---|---|---|
| The pruning, fused into the loop over pairs and in the refresh (D284, D114) | warp compaction | `emitPruneGroup` (lowering) |
| The active units (D268) | warp compaction | `emitActiveUnits` (lowering) |
| The groups build: the queue of near candidates, the kept entries, and those of the build of all images | warp compaction | `lists_body`, `images_body` (the groups template) |
| The groups build: the excluded partners of a group (D106) | warp scan, 16 lanes, unsigned | `lists_body` |
| The groups build: particles by column and z-bin key; places by cell of the search grid; the chunks of the columns | device scan (three kernels, block scan within), scatter, insertion sort; the third a scan alone | `build` (the groups template) |
| The matrix build: the rows, orthorhombic, triclinic of one pass, and of all images (D241) | warp compaction | `mdrt_gpu_build_neighbors_matrix*` (the matrix template; the images part generated by `scripts/generate-matrix-images-template.py`) |
| The matrix build: particles by cell | chunked scan, scatter, stable rank | `mdrt_gpu_matrix_sort` |
| The spatial order (D44, D86) | chunked scan, scatter (every key), insertion sort by the numbers of the input | `mdrt_gpu_spatial_order` |
| The listed particles of PME (D262, `md_exec.nonzero`) | the serial scan of the counts of the chunks | `lowerNonzero` (lowering) |

Left as they are: PME's binning for tiles (`match.any.sync` and a rank by
key, one atomic for a key: a warp-aggregated atomic, #418), the serial
partition within a particle's slots of `lowerBuildEdges` (D277), the host's
scan of `emitEdgeSlots`. The warp sums by `shuffle xor` are the reduction
emitters of Section 5 (D301).

### 4.4 Tests

- `Integration/device-primitives.mlir` calls every emitter from kernels
  written for the test; `device-primitives-gpu.test` runs it on a device
  and compares exactly: the compaction of 2 warps and the warp scans with
  results written out, and the block scan of 256 threads, the device scan
  of 5000 counts with and without the cursor, the chunked scan of 1000, and
  the counting sort of 3000 items by 50 keys (with the insertion sort, the
  stable rank, and the order by another buffer) with results computed on
  the host by the definition. With `-Dsanitize=1` it runs under memcheck
  and racecheck.
- `Conversion/MDExecToGPU/expand-primitives.mlir` checks the code of the
  warp compaction and of the warp scans; `expand-primitives-invalid.mlir`
  the refusals.
- The sites' own tests run them as before (the builds of groups and of the
  matrix, the dual list, the listed particles of PME).

### 4.5 The same code

The programs as executed (`mdir emit --stage=lowered` with
`MDIR_GPU_BINARY=ptx`, the PTX of every kernel and the host's code in the
LLVM dialect) are byte-identical to those of `main` (a6fef8b9, and again
at 075b9411 after a rebase) on:
- the nine systems of the Amber suite, each with the pruning fused (the
  default) and in the refresh (`MDIR_FUSED_PRUNE=off`): 245 to 501 kernels
  a program;
- JAC NVE with the matrix (193 kernels, the matrix sort and the spatial
  order among them), the rhombic dodecahedron of `triclinic-gpu.test`
  with the groups and the dual list (133 and 137 kernels), and the OPC
  dipeptide of `pme-skip-zero.test` on a device (457 kernels, with
  `md_exec.nonzero`).

The modules after `convert-md-exec-to-gpu` differ from those of `main` in
the place of one constant at each of three sites, which an emitter creates
in its own order: before or after a load in the scatter of the matrix sort
and of the groups build's grid cells, and in the serial scan of
`lowerNonzero`. The constants leave no trace in the PTX, as the identical
programs show. No kernel changes its name: the expansion puts each
primitive's code where the call stands, in the function of the site.

The step was timed against `main` on an RTX 4090 (Section 4.6).

### 4.6 Measured effect

None is expected, since the programs are the same. The Amber suite was run
with the binaries of `main` (a6fef8b9) and of the branch alternated in one
job on an RTX 4090 at 450 W (ada-001), three rounds. The table gives the
median ns/day of the second half of each run, and the ratio of the time
of a step to that of `main`:

| System | `main` | Branch | Ratio |
|---|---:|---:|---:|
| JAC NVE | 1264.7 | 1255.3 | 1.007 |
| JAC NPT | 1143.9 | 1142.6 | 1.001 |
| JAC NVE 4 fs | 2356.2 | 2359.0 | 0.999 |
| JAC NPT 4 fs | 2208.0 | 2205.8 | 1.001 |
| Factor IX NVE | 569.0 | 568.0 | 1.002 |
| Factor IX NPT | 530.3 | 529.9 | 1.001 |
| Cellulose NVE | 148.8 | 148.7 | 1.001 |
| Cellulose NPT | 141.0 | 140.6 | 1.003 |
| STMV NPT 4 fs | 87.5 | 87.6 | 0.999 |

No system is slower beyond the spread of the runs: the rounds of JAC NVE
took 1259.7 to 1272.4 ns/day with `main` and 1253.6 to 1260.3 with the
branch, whose program is byte for byte that of `main`.

## 5. Reductions by their algebra

### 5.1 What is stated

A reduction or a scatter-accumulate combines values in an order that its
lowering chooses: a tree, the rows of a matrix, the atomics of the
threads, a table in block memory. Whether another order is acceptable is a
property of the sum, not of the lowering, and MDIR states it in the IR for
every value that an op reduces (D301, #422, step 1;
[principles.md](principles.md), principle 1). Each value is a licence to
the lowering about the order of evaluation, in the words of the
maintainer's ruling on #470:

| Algebra | Licence |
|---|---|
| `any_order_exact` | The lowering may evaluate in any order and grouping, and every order gives the same bits (integers, `or`, fixed point). |
| `any_order` | The lowering may evaluate in any order and grouping. For floating-point values the result's rounding then depends on that order. This is the floating-point sums of the default mode. |
| `fixed_order` | The lowering must use the order the IR states, or an exact form of the sum, so that the result is the same on every run of the same build. For `md_dist` this holds for a given decomposition (#409). |

`fixed_order` may be met through an `any_order_exact` form: the
fixed-point spreading of PME in the deterministic mode (D70, D84) is the
case.

**The licence is about order only** (the maintainer's guards on #470).
`any_order` permits any order and grouping. It does not permit changing
the accumulator's precision (an f64 sum taken in f32) or the order of
growth of its rounding error (a tree replaced by a serial chain of
atomics). Such a change is a separate decision, with its own measurement.

The names are licences, not algebraic claims: floating-point
addition is commutative but not associative, and a sum of $m$ terms in
another order and grouping differs from the exact sum by at most
$(m - 1)\,\mathsf u\sum_k\lvert v_k\rvert + O(\mathsf u^2)$
[[Higham2002]](references.md#higham2002), which bounds the difference
between two orders and does not make it zero. Exact, by type, and in any
order:
- a sum of integers, modulo $2^{32}$ or $2^{64}$ (the fixed point of D70
  is a sum of integers);
- a logical or (the flags of D107, D118, D278).

A maximum of floating-point numbers is exact in any order for numbers; the
lowering's form, `select(cmpf ogt)`, is not where one operand is not a
number (NaN), since `ogt` with a NaN selects the second operand. The
maximum of the validity of a structure (D80) is internal to the lowering
and not stated as an attribute; a position that is not a number is
reported by the flags of D107.

### 5.2 In the IR

The algebras are the discardable attribute `md_exec.algebra`, one entry for
each reduced value, in the order of `md_exec::algebra::getReducedTypes`
(`include/mdir/Dialect/MDExec/Algebra.h`):

| Op | Reduced values |
|---|---|
| `md_exec.particle_for` | its `reduce` |
| `md_exec.pair_for`, `md_exec.tuple_for` | its destinations (each a sum over the pairs or tuples of a particle), then its `reduce` |
| `md_exec.scatter_add` | its destination |
| `md_exec.reciprocal` | the spreading into its grid, its energy, its virial |
| `md_dist.all_reduce`, `md_dist.halo_sum` | the sum |

```mlir
%k, %bad = md_exec.particle_for ins(%v : !md.field<@atoms, 3 x f32>)
             reduce(%zero, %no : f64, i1) { ... }
             {md_exec.algebra = #md_exec.algebra<any_order, any_order_exact>}
             -> f64, i1
%u = md_dist.all_reduce %p {md_exec.algebra = #md_exec.algebra<fixed_order>}
       : !md_dist.partial<f64> -> f64
```

- **The verifier** (the dialect's hook for `md_exec.algebra`) refuses an
  attribute on an op that reduces nothing, a count other than that of the
  reduced values, and `any_order_exact` on a floating-point value.
  `fixed_order` on an integer is allowed: a stronger licence than needed.
  A program built for the deterministic mode carries the unit attribute
  `md_exec.deterministic` on its module, whose verifier refuses
  `any_order` on a floating-point value anywhere in it; the lowerings
  remove the mark with the ops it verifies. That verifier walks the module
  at each verification, a pass over its ops, which is cheap beside the
  passes that run between two verifications.
- **`md-exec-assign-algebra{deterministic}`** is the only pass that
  states an algebra; every other pass carries it unchanged. It states it
  on the ops of `md_exec` from the mode and the type: `any_order_exact` for an integer;
  for a floating-point number `fixed_order` in the deterministic mode and
  `any_order` otherwise. The driver runs it after the precision is
  assigned (and after the passes that rebuild loops: fusion, the
  accumulation of destinations, the narrowing of sums), before
  `md-exec-choose-scatter` on a device and before the storage on the CPU
  (`getPipeline`, `lib/Compiler/Compile.cpp`). `md-exec-assign-storage`
  carries it onto the storage form (`algebra::carry`).
- **The sums across domains** are `fixed_order` in every mode: their
  lowering, `lowerAllReduce` and `lowerHaloSum`
  (`lib/Conversion/MDDistToLoops/MDDistToLoops.cpp`), adds the domains,
  then the rows, in their order. That gives the same bits on every run for
  the same decomposition and build, not for another decomposition (#409).
  `md-exec-assign-algebra` states it on every such sum, in either mode
  (`algebra::stateDistributed`); the passes that create them
  (`md-distribute`, `md-differentiate`, the transpose of
  `md_dist.halo_copy`) state nothing.
- **The check.** `--md-exec-check-algebra` refuses an op that reduces
  values without stating their algebra, as `--md-check-transposes` (D295)
  refuses an op without a transpose. `convert-md-exec-to-gpu`,
  `convert-md-exec-to-loops`, and `convert-md-dist-to-loops` refuse such an
  op as well, before they lower anything: a lowering reads the algebra and
  never assumes one.

Why a discardable attribute and not an argument of each op: the ops are
created at about 140 places with their generated builders, which an argument
would change at every one; a pass that rebuilds an op without the
attribute loses the fact, and the lowerings then refuse the op, so a lost
fact is an error and never a silent default. An op interface was not
needed for the list of reduced values: one function,
`algebra::getReducedTypes`, gives it for the seven ops, as
`isOfTheStructure` lists the ops for `--md-check-transposes`.

### 5.3 Where it is read

The chooser and the lowerings read the algebra through the same functions
(`Algebra.h`, and `checkReorder`, `checkTile`, `checkHash` of `Scatter.h`,
which now take the algebras in place of the mode; principle 2). The mode
still decides what is not a sum: the joined integration kernels'
displacements (`integration.storedPositions`, #102), `explicit-fma`, and
`contract=false`.

| Decision | Before | Now | Function |
|---|---|---|---|
| A table or a tile adds a block's values first | refused in the deterministic mode | refused unless every destination allows any order | `checkReorder` (`Scatter.h`), called by `md-exec-choose-scatter` and the lowering |
| A loop over tuples that may share particles evaluates each tuple once, with atomics, or the rows of the particles take its tuples | rows in the deterministic mode | rows unless every value the loop reduces allows any order | `evaluatesOnce` (`MDExecToGPU.cpp`) |
| The renumbered members of the tuples go to the device (for the loops above) | not in the deterministic mode | not where a loop over tuples of the module states `fixed_order` | `tuplesMayReorder` (`MDExecToGPU.cpp`), at three places |
| A loop over pairs that adds at both ends, `(directed, atomic)` | refused in the deterministic mode | refused where a destination is `fixed_order` | `lowerPairFor` |
| A loop over groups, each pair once (D89) | refused in the deterministic mode | refused where any reduced value is `fixed_order` (its global sums come from warps over lists built with atomics) | `lowerGroupPairFor` |
| The spreading of PME in fixed point, added with integer atomics | in the deterministic mode | where the spreading is `fixed_order` | `lowerReciprocal`, `lowerProlate` |
| `md_exec.scatter_add` on a device | atomics in either mode | refused where it is `fixed_order`: no fixed-order form exists on a device yet (fixed point is #416, step 2) | `lowerScatterAdd` (`MDExecToGPU.cpp`) |
| `md_exec.scatter_add` on the host with `atomic-grid` | atomics in either mode | a loop on one thread in the order of the values where it is `fixed_order` | `lowerScatterAdd` (`MDExecToLoops.cpp`) |
| The global sums of the loops | block trees whose order depends on the count alone (D150) | the same: a fixed order meets every algebra | `emitReductions`, `finishSums` |

`md-exec-choose-scatter` lost its option `deterministic`: it reads the
algebra of each op.

**Where the result changed outside the pipelines.** Three rows of the
table decide otherwise than the mode did for an input that no pipeline of
the driver produces, and none of the compared programs (Section 5.6)
reaches them:
- a loop over tuples that may share particles whose only reduced values
  are flags (`any_order_exact`) evaluates each tuple once in the
  deterministic mode too, where it was refused
  (`Conversion/MDExecToGPU/tuple-flags-invalid.mlir` now states
  `fixed_order` on the flag to keep the refusal);
- a deterministic module in which no loop over tuples states `fixed_order`
  copies the members to the device;
- `md_exec.scatter_add` in the deterministic mode is refused on a device
  and runs on one thread on the host (no pipeline creates the op on
  `main`; #443's `halo_sum` is its first site).

### 5.4 The emitters

The reductions of a device are one emitter each,
`include/mdir/Conversion/ReductionEmitter.h` (`kernels::`, beside
`ScatterEmitter.h` and `CompactionEmitter.h`):

| Emitter | What it emits | CUB's counterpart |
|---|---|---|
| `emitShuffleXor` | a value from the lane whose number differs in the bits of `offset`: one `gpu.shuffle xor` for 32 bits, two halves for 64, element by element for a vector | — |
| `emitCombine` | a sum (`addf`, or `addi` for integers) or a maximum (`cmpf ogt` and a select) | the reduction operator |
| `emitWarpReduce` | the butterfly over groups of 2 to 32 lanes: offsets `lanes/2`, ..., 1, every lane with the result | `WarpReduceShfl` [[CUB]](references.md#cub) |
| `emitBlockReduce` | the warps' butterflies, the first lane of each writing to block memory, one barrier, every thread combining the warps in their order (D150) | `BlockReduce` with `BLOCK_REDUCE_WARP_REDUCTIONS` |

Every one has a fixed order (the lanes and the warps decide it, not the
timing of the threads), and so meets each of the three algebras. Sites:
`reduceInBlock` (the global sums and the maximum of validity of
`emitReductions`), `RowLanes::combine` (the lanes of a row of a matrix),
the warp totals of the global sums of a loop over groups, the counts
of the active units (D268, integers), and the sums of rows of the batched
bodies of neural stages, a warp a row (D[ml4-body-reductions],
`warpRowSums` in `StageBodies.cpp`). Left as it is: the half-warp sum of
`emitGroupPairKernel` (`MDExecKernels.cpp`), which shuffles a 64-bit value
with `gpu.shuffle` itself rather than as two halves, so that moving it would
change the code of double precision.

### 5.5 Tests

- `Dialect/MDExec/Transforms/assign-algebra.mlir`,
  `assign-algebra-loops.mlir`: the pass on every op, in both modes, carried
  onto the storage form, printed and parsed again.
- `Dialect/MDExec/Transforms/algebra-distributed.mlir`: the sums across
  domains, `fixed_order` in both modes.
- `Dialect/MDExec/algebra-invalid.mlir`: the verifier, with `any_order`
  refused in a program built for the deterministic mode.
- `Dialect/MDExec/Transforms/check-algebra.mlir`: the check, and the
  refusals of `convert-md-exec-to-loops` and `convert-md-exec-to-gpu`.
- The algebra, not the mode, decides: `Conversion/MDExecToGPU/kept-tuples.mlir`
  (the rows with a lowering not in the deterministic mode),
  `Conversion/MDDistToMDExec/deterministic-invalid.mlir` (the two-ended
  loop refused), `Conversion/MDExecToLoops/scatter-add-algebra.mlir` (the
  host's scatter on one thread), and the refusals of
  `Conversion/MDExecToGPU/scatter-invalid.mlir`, `scatter-add.mlir`, and
  `Dialect/MDExec/Transforms/choose-scatter.mlir`.
- `Conversion/MDExecToGPU/reductions.mlir`: the code of the block
  reduction (the butterfly at 16 to 1 in two halves of 32 bits, the
  barrier, the warps in order), the same in both modes.

The lowering tests run `md-exec-assign-algebra` before the chooser or the
lowering, and before `convert-md-dist-to-*` (`test/lit.cfg.py` adds it to
the substitutions of the lowerings). The bitwise tests of the
deterministic mode stay in the suite unchanged.

### 5.6 The same code

The programs as executed (`mdir emit --stage=lowered` with
`MDIR_GPU_BINARY=ptx`: the PTX of every kernel and the host's code in the
LLVM dialect) are byte-identical to those of `main` (24447fb2) for 32
inputs, 32 of 32, at the branch's head after the rebase (f302e2ed). Before
the rebase, against `main` at 0a336ff9, all 32 were identical at 965573ba
and at cfaa839d; at 2ed9f8d7 the 25 inputs of the default mode were, and
the 7 of the deterministic mode differed only in the mark
`md_exec.deterministic` on the module, which the lowerings remove since
cfaa839d:

| Inputs | Kernels a program |
|---|---:|
| The nine systems of the Amber suite (default mode, groups), with the pruning fused and in the refresh (`MDIR_FUSED_PRUNE=off`): 18 programs | 245 to 501 |
| The deterministic mode on a device (the matrix, PME in fixed point, the tuples by rows): JAC NVE, JAC NPT, Factor IX NVE, Cellulose NVE | 211 to 406 |
| JAC NVE on the CPU in double precision, in the default and the deterministic mode | — |
| JAC NVE with the matrix; with the tables of the tuples (`MDIR_TUPLE_SUMS=blocks`) and of the loops over groups (`MDIR_PAIR_SUMS=blocks`); with the tiles of PME (`MDIR_PME_SPREAD=tiles`) | 193 to 260 |
| The prolate kernel (the three OPC waters of `prolate.test`) on a device, default and deterministic | 165, 173 |
| The listed particles of PME (the OPC dipeptide of `pme-skip-zero.test`, LJ-PME, NPT) on a device, default and deterministic | 418, 439 |

The modules after `convert-md-exec-to-gpu` were not compared; the
programs as executed are the criterion of D285, D291, and D296.

### 5.7 Measured effect

None is expected, since the programs are the same. The Amber suite was
run with the binaries of `main` (0a336ff9) and of the branch before the
rebase (965573ba, whose programs were those of that `main`) alternated in one job
on an RTX 4090 at 450 W (ada-002, `--idle`), three rounds. The table gives
the median ns/day of the second half of each run, and the ratio of the
time of a step to that of `main`:

| System | `main` | Branch | Ratio |
|---|---:|---:|---:|
| JAC NVE | 1274.8 | 1275.2 | 1.000 |
| JAC NPT | 1158.0 | 1154.8 | 1.003 |
| JAC NVE 4 fs | 2378.6 | 2374.5 | 1.002 |
| JAC NPT 4 fs | 2222.4 | 2225.7 | 0.999 |
| Factor IX NVE | 571.1 | 570.3 | 1.001 |
| Factor IX NPT | 532.9 | 531.0 | 1.004 |
| Cellulose NVE | 147.9 | 147.2 | 1.005 |
| Cellulose NPT | 140.3 | 140.1 | 1.001 |
| STMV NPT 4 fs | 87.0 | 86.6 | 1.005 |

No system is slower beyond the spread of the runs: the rounds of STMV
took 86.2 to 87.3 ns/day with `main` and 85.0 to 87.8 with the branch, of
Cellulose NVE 147.7 to 147.9 and 141.9 to 148.1, for programs that are
byte for byte the same.

### 5.8 Next

Strategies that the stated algebra licenses come as later steps of #422,
each only on a measured gain, `main` and the branch in one job:
- `fixed_order` met through `any_order_exact`: fixed-point sums
  privatized in the deterministic mode (#416, step 2), the tuples
  evaluated once in fixed point instead of the rows, the loops over
  groups in fixed point;
- raking (`BlockReduce`'s commutative-only form) for the global sums
  where they are `any_order`.
