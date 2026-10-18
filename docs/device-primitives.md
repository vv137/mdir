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
| Privatized scatter-accumulate: `tile`, `hash` (Section 2) | `BlockHistogram`, atomic specialization | #416 | Built. Step 1 put both strategies on one attribute and one pass. Fixed point in the deterministic mode (step 2) and further sites (step 3) are planned | D282 (`tile`), D283 (`hash`), D[privatize-scatter] (the attribute and the pass) |
| Sorted scatter: runs reduced without atomics | `BlockHistogram`, sort specialization; `BlockDiscontinuity` | #417 | Planned. `sorted` is reserved in the attribute and refused | — |
| Warp-aggregated atomics | `WarpReduce`, matching by key | #418 | Planned | — |
| Vectorized and transposed gathers | `BlockLoad`, `WarpLoad`, `BlockExchange` | #419 | Next, on the gathering of PME (tiled gathering at orders 6 and 8 is #411) | — |
| Single-pass prefix sums | `BlockScan`; device scan with decoupled look-back | #420 | Planned | — |
| Radix sorts: tuples by first member, binning by key | `BlockRadixSort`; device radix sort (onesweep) | #421 | Next | — |
| Reductions chosen by the algebra of the operator | `BlockReduce` (raking, commutative only) | #422 | Planned | — |

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

Two strategies differ in how a block finds its copy of an index:

| Strategy | The copy of a block | For a footprint that |
|---|---|---|
| `tile(E)` | A dense tile: the bin of $E^3$ points of the grid and the $P - 1$ points of the stencil past it on each axis, $(E + P - 1)^3$ values | the IR bounds (the stencil of order $P$ and the bin) |
| `hash(S, P, L)` | A table of $S$ slots keyed by the index, found by a multiplicative hash and the next slots in turn [[Knuth1998]](references.md#knuth1998), claimed by a compare-and-swap. A search gives up after $P$ slots, and the value goes to the field with an atomic of its own. The host takes the table from $L$ items of a launch on | has no bound (the members of tuples) |
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
```

Without the attribute, or with `scatter(atomic)`, a site lowers as it does
without privatization. The verifiers refuse a strategy that the op's
lowering does not have: `tile` on `md_exec.tuple_for` (the members of a
tuple may be anywhere, so no footprint is bounded) and `hash` on
`md_exec.reciprocal` (the template of the spreading has the kernels of
tiles alone).

The pass `md-exec-choose-scatter` sets the attribute on programs for a
device, after the precision is assigned and before the storage, which
sizes the bins of the tiles from it. `md-exec-assign-storage` carries the
attribute onto the storage form. A loop that the storage makes overwrite a
destination writes the rows of its particles and loses it.

Why an attribute, not one scatter op that both sites lower through: the
spreading of PME is a kernel of the runtime template
(`lib/Runtime/Templates/PMEGPU.mlir`, written by
`scripts/generate-pme-gpu-template.py`), in which the compiler instantiates
the addition to the tile. The table of the tuples is generated by
`MDExecToGPU.cpp`. One op would have meant generating the spreading
instead of the template. A scatter op is the form for the sites still to
come (step 3 of #416).

### 2.3 Preconditions, and where they are checked

Each precondition is one function of `include/mdir/Dialect/MDExec/Scatter.h`.
The pass calls it to choose, and the lowering calls it again to apply.

| Precondition | Strategies | Function | The pass | The lowering |
|---|---|---|---|---|
| The values may be added in another order: integers in any order, floating-point numbers outside the deterministic mode only, since a sum of a block first rounds otherwise. Fixed point is not privatized yet (step 2), so only f32 and f64 are | `tile`, `hash` | `checkReorder` | refuses | fails, naming the reason |
| The footprint of a tile is bounded and wraps around the grid once at most: every edge of the grid has at least $E + P - 1$ points. The template has kernels of $E$ = 4, 8, and 16 | `tile` | `checkTile` | refuses | fails |
| The copy fits 48 KiB of block memory, the static limit of a block from compute capability 7.0 on, which neither kernel exceeds: $(s^3 + 4T)\,b + 12T$ bytes for a tile ($s = E + P - 1$, $T$ = 128 threads, or 512 for $E$ = 16, $b$ the bytes of a value, with the particles of a round); $4S + \sum_d S\,c_d\,b_d$ for a table of $S$ slots and destinations of $c_d$ components | `tile`, `hash` | `getTileBytes`, `getHashBytes` | refuses | a tile fails. The table of a *run* of loops, launched as one kernel with the union of their destinations, is checked with the same function and keeps the atomics where it does not fit: runs are formed in the lowering, after the pass |
| A fallback exists: every destination of a table is a number or a row of a static length of f32 or f64, which have an atomic of the device | `hash` | `getHashDestination` | refuses | fails |
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
  $P$ = 8, and $L$ = 100,000. Below 100,000 tuples the device holds every
  block at once, and the table lengthens each one: with the table JAC
  (55,000 tuples) ran 1.5% slower on an RTX 4090 and 3% slower on an RTX
  PRO 6000, Factor IX (165,000) 2.3% and 1% faster. The count is known on
  the host only, so the lowering emits both kernels and the host compares.
  Without the bound of 8 slots a block whose tuples share few particles
  fills its table: one block of JAC with 494 particles in 512 slots took
  the kernel from 12 to 42 µs; with the bound, 13.2.

With `explain`, the pass puts a remark on each op that states the choice
or the reason for a refusal.

### 2.5 Switches

| Switch | Effect |
|---|---|
| `md-exec-choose-scatter{tile=auto\|on\|off}` | `auto` the rule above; `on` every sum that can be tiled; `off` none |
| `tile-edge`, `tile-min-points` | The points of an edge of a bin (4, 8, 16; 8 by default) and the threshold of `auto` |
| `hash=auto\|on\|off` | `on` takes the table at any count ($L$ = 0) |
| `hash-least`, `hash-probes`, `block-size` | $L$, $P$, and the threads of a block, as `convert-md-exec-to-gpu` launches them |
| `deterministic`, `explain` | The mode, which privatizes no floating-point sum; the remarks |
| `MDIR_PME_SPREAD` | `global` is `tile=off`; `tiles` is `tile=on`; `tiles4`, `tiles8`, `tiles16` set the edge as well; `auto` is the default ([environment.md](environment.md)) |
| `MDIR_TUPLE_SUMS` | `atomics` is `hash=off`; `blocks` is `hash=on`, which the tests of small systems use |

The tests compare each strategy with the atomics:
`pme-tiles-gpu.test` (every edge, orders 4, 6, and 8, the prolate kernel,
a triclinic cell, a slab whose bins overflow, LJ-PME, mixed and double)
and `tuple-block-sums-gpu.test` (mixed against atomics and against
double), both also under compute-sanitizer. `choose-scatter.mlir` checks
each choice and refusal of the pass, `Dialect/MDExec/scatter-invalid.mlir`
the verifiers, and `Conversion/MDExecToGPU/scatter-invalid.mlir` the
lowering's refusals.

### 2.6 Sites

| Site | Strategy | Kernel | Described in |
|---|---|---|---|
| The spreading of PME on a device: the charges, the coefficients of LJ-PME (D162), the prolate kernel (D266), orthorhombic and triclinic cells | `tile(8)` | `mdrt_gpu_pme_bin<E>`, `mdrt_gpu_pme_spread_tiles<E>` of the template; the addition to the tile is a relaxed `atomicrmw fadd` at the scope of the block (`PME_SHARED_ADD_E`, `MDExecKernels.cpp`) | [pme-m1.md](pme-m1.md#33-spreading-through-tiles), Section 3.3 |
| The kernel of the tuples evaluated once: the bonded terms, the 1-4 pairs, the excluded pairs of the reciprocal sum | `hash(512, 8, 100000)` for a run with dihedrals | `launchTupleBlocks` in `MDExecToGPU.cpp`; `red.shared.add` | [device-tuples.md](device-tuples.md), Section 11 |

The two sites emit their additions in block memory differently. One
emitter for both would change the kernels, so step 1 left them as they
were.

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

**One attribute and one pass (D[privatize-scatter]).** Moving both sites
onto the attribute changed no kernel. The module after
`convert-md-exec-to-gpu` is identical to that of `main` on JAC NVE and
NPT, Factor IX NVE, Cellulose NVE, and STMV NPT 4 fs, apart from the line
that prints the options of the pass. The Amber suite alternated against
`main` on an RTX 4090 is in the decision row of D[privatize-scatter].

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
