# The primitives of the device lowering

This document describes the parallel primitives that the lowering for a
device composes kernels from, in the form they take today: what each does,
what must hold for it, where that is checked, how to turn it off, which
kernels use it, what it gains, and where the idea comes from. MDIR
generates its device code through MLIR and does not link CUB. It reads
CUB's block and warp algorithms (part of CCCL, which ships with CUDA) and
implements them as lowering strategies with stated preconditions and
switches ([principles.md](principles.md), principles 2 and 3; issue #415).

## 1. Status

| Primitive | CUB's counterpart | Issue | Status | Decision |
|---|---|---|---|---|
| Privatized scatter-accumulate: `tile`, `hash` (Section 2) | `BlockHistogram`, atomic specialization | #416 | Built. One attribute, one pass, and one set of preconditions (step 1). The table `hash` is one emitter that every site calls (`ScatterEmitter.h`, step 3). Its sites are the tuples, the op `md_exec.scatter_add`, and the loops over pairs that add at both ends (over groups, and `(directed, atomic)` over a matrix), where it was measured slower and is off by default (Section 2.7). The tile of PME keeps its template. Fixed point in the deterministic mode (step 2) is planned | D282 (`tile`), D283 (`hash`), D285 (the attribute and the pass), D[scatter-op] (the op, the emitter, the groups) |
| Sorted scatter: runs reduced without atomics | `BlockHistogram`, sort specialization; `BlockDiscontinuity` | #417 | Planned. `sorted` is reserved in the attribute and refused | — |
| Warp-aggregated atomics | `WarpReduce`, matching by key | #418 | Planned | — |
| Vectorized and transposed gathers | `BlockLoad`, `WarpLoad`, `BlockExchange` | #419 | Next, on the gathering of PME (tiled gathering at orders 6 and 8 is #411) | — |
| Single-pass prefix sums | `BlockScan`; device scan with decoupled look-back | #420 | Planned | — |
| Radix sorts: tuples by first member, binning by key | `BlockRadixSort`; device radix sort (onesweep) | #421 | Next | — |
| Reductions chosen by the algebra of the operator | `BlockReduce` (raking, commutative only) | #422 | Planned | — |
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

`md_exec.scatter_add` is the scatter-accumulate on its own (D[scatter-op],
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
A combiner other than addition is to come as `combine(...)`, with #422.
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
lowering's refusals. The tests of D[scatter-op]:
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

**The emitter** (D[scatter-op]). The table is one class,
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

**The loops over groups (D[scatter-op]).** The table at the $i$ and $j$
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

**The emitter and the op against `main` (D[scatter-op]).** By default
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
