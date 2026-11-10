# MDIR internals: a guide for contributors

This page tells a contributor which design patterns the code uses and what
to use when adding something. It summarizes and links; the documents it
links are the statement. Start with [overview.md](overview.md) for what
MDIR computes and [principles.md](principles.md) for the rules a change is
reviewed against; the objects a developer meets at every level are in
[Table 3.4 of the white paper](paper/03-design.md#37-the-objects-of-mdir-for-developers).

## 1. The structure

A program is lowered in stages. Each stage states less about the physics
and more about the execution:

1. `md` and `dyn`: what is computed and how the state advances (potentials,
   relations, tuple sets, sums, requests, step programs).
2. `md_exec`: over which set, with which access (`particle_for`,
   `pair_for`, `tuple_for`, neighbor builds and refreshes, `scatter_add`),
   first in a value form, then in a storage form after
   `md-exec-assign-storage`.
3. `scf` loops (CPU, OpenMP) or `gpu` kernels (device), with the shared
   per-thread code of `lib/Conversion/MDExecKernels`.
4. NVVM and LLVM: cubins or PTX for the device, host code in MLIR's
   `ExecutionEngine`, linked with the runtime `mdrt`.

`md_dist` (domains, halo copy and halo sum) sits between 1 and 2 and is
taken through `mdir-opt` only today. The dialects are described in
[overview.md, Section 2.2](overview.md#22-the-dialects), the passes in
order in [Section 2.3](overview.md#23-the-main-passes), the lowerings in
[Section 2.4](overview.md#24-the-lowerings). The order of the passes of a
run is built by `getPipeline` in `lib/Compiler/Compile.cpp`;
`mdir emit FILE --stage=pipeline` prints it.

| Directory | What lives there |
|---|---|
| `include/mdir/Dialect/{MD,Dyn,MDExec,MDDist,MDRT}`, `lib/Dialect/...` | The dialects: types and ops in TableGen, verifiers, op interfaces |
| `lib/Dialect/MD/Transforms` | Passes over `md`: differentiation, exchange checks, truncation, inlining, the couplings |
| `lib/Dialect/MDExec/Transforms` | Passes over `md_exec`: neighbor reuse, fusion, precision, algebra, the choosers, storage, streams |
| `lib/Conversion/MDToMDExec` | From sums over relations to loops over sets |
| `lib/Conversion/MDExecToLoops`, `lib/Conversion/MDExecToGPU` | The CPU and device lowerings |
| `lib/Conversion/MDExecKernels`, `include/mdir/Conversion/*Emitter.h` | What one thread does, shared by both lowerings, and the emitters of the device primitives |
| `lib/Conversion/MDDistToLoops` | The lowering of `md_dist` |
| `lib/Conversion/GPUToBinary` | The NVIDIA back end: NVVM, `ptxas`, cubins |
| `lib/Runtime/Templates`, `scripts/generate-*-template.py` | Structures that are not loops over a set, written in MLIR: neighbor builds, the spatial sort, PME on a device |
| `runtime/` | The runtime in C: memory pool, streams, launches, cuFFT, cuBLAS, the log |
| `lib/Driver`, `lib/Compiler`, `tools/mdir` | The control file, the readers, the builder of the program, the pipeline, the command `mdir` |

## 2. The patterns

**Progressive lowering.** Each stage keeps only the facts the next one
needs, and a pass works at the highest stage where its facts are visible:
differentiation and exchange contracts on `md`, fusion and precision on
`md_exec`, launch shapes in the device lowering. Use it when deciding where
a change goes: a change that needs the physics of a term belongs above
`md_exec`; a change that needs a buffer or a launch belongs below. Example:
`md-differentiate` (`lib/Dialect/MD/Transforms/Differentiate.cpp`) writes
the forces as `md` before any loop exists. Detail:
[white paper, Section 3.4](paper/03-design.md#34-from-sums-to-loops-and-the-passes-over-loops)
and [Section 3.5](paper/03-design.md#35-lowerings-and-templates).

**Emitters.** One C++ function builds the IR of one primitive, and callers
compose them: `kernels::emitWarpBallot`, `emitBlockScan`, `emitDeviceScan`,
`emitSortBins` (`CompactionEmitter.h`), `emitSinglePassScan`
(`SinglePassScan.cpp`), `emitRaggedRow` and `emitDeviceRaggedRows`
(`RaggedRows.h`), the block hash table (`ScatterEmitter.h`), the warp and
block reductions (`ReductionEmitter.h`), the grid gathers
(`GatherEmitter.h`). Implementations are in
`lib/Conversion/MDExecKernels`. Use one whenever a kernel packs, scans,
sorts, reduces, or scatters. Each emitter has preconditions, many of them
not checked by the compiler: the caller must respect them. They are
gathered, with where each is checked, in
[device-primitives.md, Section 1.1](device-primitives.md#11-preconditions-at-a-glance).

**Named primitives in templates.** A template in `lib/Runtime/Templates`
does not copy an emitter's code; it declares and calls the primitive by
name, `func.call @mdir.primitive.<name>(...)` (for example `warp_rank`,
`block_scan`, `sort_bins`, `ragged_row`). `kernels::expandPrimitiveCalls`
(`lib/Conversion/MDExecKernels/ExpandPrimitives.cpp`) replaces each call
with the emitter's code when `convert-md-exec-to-gpu` parses the template;
the pass `expand-device-primitives` does the same on a module, for tests.
An unknown name or a wrong signature is an error. The group templates
(four `NeighborsGroups*GPU*.mlir` files) are generated by
`scripts/generate-groups-gpu-template.py`: edit the generator and
regenerate them together; `test/Runtime/generated-templates.test` compares
them. Detail: [device-primitives.md, Section 4.2](device-primitives.md#42-how-the-sites-call-them).

**Op interfaces.** A pass that must handle several ops reads an interface,
not a list of ops. `DerivativeOpInterface` and `AdjointOpInterface`
(`include/mdir/Dialect/MD/Transforms/{Derivative,Adjoint}Interface.td`)
give each op its tangent, its reverse rule, and its declared transpose;
`md-differentiate` and the checks of transposes read them.
`LoopEffectOpInterface` (`include/mdir/Dialect/MDExec/LoopEffects.td`)
gives the accesses of a loop of `md_exec` by operand, row scope, and set.
Write a new pass against an interface rather than against one op: today
`SimplifyDistance.cpp`, `ExpandRadial.cpp`, and `ShareDivisions.cpp` match
`PairForOp` only, so a new pair-like loop misses the radial table unless
each pass is patched again (#817, item 5; an interface of pair-kernel
bodies is in progress in #795). Detail:
[primitives-map.md, Section 1.3](primitives-map.md#13-what-the-derivative-uses-and-how-it-is-declared).

**Strategies chosen by preconditions, then by measured policy.** A
strategy states its precondition in terms of the IR; one function checks
it, and both the chooser pass and the lowering call that function. Among
the legal strategies, a policy measured on named devices chooses; each
strategy has a switch, a test that compares it with the baseline, and,
with `explain`, a remark that states the choice or the reason for a
refusal. The example is `md-exec-choose-scatter`
(`lib/Dialect/MDExec/Transforms/ChooseScatter.cpp`): `scatter::checkReorder`
(`include/mdir/Dialect/MDExec/Scatter.h`) is read by the pass and by the
lowering, the thresholds come from measurements, and `tile`, `hash`, and
`pair-hash` are its switches. Detail:
[principles 2 and 3](principles.md#2-an-optimization-states-its-precondition-and-the-precondition-is-checked),
[device-primitives.md, Sections 2.4 and 2.5](device-primitives.md#24-the-choice).

**The TermBuilder registry.** A term of the potential is a class derived
from `TermBuilder` (`lib/Driver/BuilderImpl.h`) in a file of its own
(`lib/Driver/Term*.cpp`), listed by an explicit factory in
`registerTermBuilders()` (`lib/Driver/TermBuilders.cpp`). The central
builder iterates the registered adapters and does not branch on the kind
of a term. A term builds native ops in the shared potential; forces and
virial follow by differentiation. Detail:
[driver.md, Section 2.12](driver.md#212-the-files-of-the-builder).

**Facts in the IR.** A pass relies only on what the IR states: types,
attributes read through a named dialect helper or an op accessor, and the
structure of ops. A fact kept in a pass's map or in a comment is a defect
waiting for an input. Invariants go in types or static attributes (a
buffer's set, `#md_exec.set_layout`); state that changes during a run goes
in SSA values; a required condition is proved, guarded at run time, or the
path is refused with a diagnostic. Detail:
[principle 1](principles.md#1-a-fact-that-a-pass-relies-on-is-in-the-ir),
[principle 10](principles.md#10-execution-contracts),
[execution-contracts.md](execution-contracts.md).

**Primitives closed under transposition.** MDIR is a composition of a
small set of primitives (map, gather, scatter-accumulate, halo copy and
sum, reduction, broadcast, scan, permutation, compaction, linear
transforms), each with its properties and its transpose in the set, so
that a derivative is a composition of the same primitives and a strategy
for one direction serves the other. A new part is placed in that map; a
part that does not fit says why. Detail:
[principle 9](principles.md#9-the-primitives-are-closed-under-transposition),
[primitives-map.md](primitives-map.md).

**The target boundary.** The `gpu` dialect's ops are target-neutral; NVVM
ops and PTX assembly belong in the NVIDIA implementation of an emitter or
in the back end (`buildGpuLowerToNVVM`,
`lib/Conversion/GPUToBinary/GPUToBinary.cpp`). Capabilities (subgroup
width, block memory, f64 rate, atomics) and measured policy are to be one
target record (#558, #433), not thresholds spread through the lowerings.
Detail: [device-primitives.md, Section 6](device-primitives.md#6-portability-one-target-contract-several-implementations).

## 3. To add something

| To add | Use |
|---|---|
| A term of the potential | A `TermBuilder` in its own `lib/Driver/Term<Name>.cpp` and a factory in `registerTermBuilders()`; native ops in the shared potential, parameters as tuple fields, forces by differentiation; check that the printed module and lowered program of the existing corpus are unchanged ([driver.md, Section 2.12](driver.md#212-the-files-of-the-builder), [Appendix B.2](paper/B-contributing.md#b2-where-a-change-goes)) |
| A pass | Its definition in the `Passes.td` of its stage (`include/mdir/Dialect/MD/Transforms`, `include/mdir/Dialect/MDExec/Transforms`, or `include/mdir/Conversion`), its code beside the others of that stage, a test of its IR under `test/Dialect` or `test/Conversion`, and its place in `getPipeline` (`lib/Compiler/Compile.cpp`). Match an op interface rather than one op |
| A device kernel, or device code in a lowering or a template | The part's row of [primitives-map.md](primitives-map.md#3-the-map), then the shared emitters within their preconditions ([device-primitives.md, Section 1.1](device-primitives.md#11-preconditions-at-a-glance)); a template calls them as `@mdir.primitive.<name>`. If none fits, add an emitter: its section in device-primitives.md, its row in Section 1.1, its row in primitives-map.md, and its name in `ExpandPrimitives.cpp` if a template calls it. A hand-written copy where a shared primitive exists states its reason in the PR ([workflow.md](workflow.md#a-piece-of-work)) |
| An optimization or a strategy | A precondition checked by one function that the chooser and the lowering both call; a switch; a test comparing it with the baseline (bitwise where the sums have a fixed order); an `explain` remark for the choice and each refusal; a default from measurements against `main` ([principles 2 and 3](principles.md#3-every-optimization-can-be-turned-off-and-the-two-are-compared), [workflow.md, Performance](workflow.md#performance)). One that reorders or overlaps work carries the argument of [principle 2](principles.md#optimizations-that-reorder-or-overlap-work) in its decision |
| A structure kept across steps, like a neighbor structure | Today the neighbor structures' reference positions, test of validity, and rebuild (`md-exec-reuse-neighbors`, `md-exec-expose-validity`, [neighbors.md](neighbors.md)), and builds counted through `mdrtCountBuild` so that the log reports them. A shared primitive for kept structures (the test, the rebuild and prune flags, growth, the counts) is in progress (#817, item 3) |
| A GPU target | The contract of [device-primitives.md, Section 6.2](device-primitives.md#62-contract-and-choice): an implementation of each emitter for the target, its capabilities and measured policy in the target record, its own compiler boundary beside `GPUToBinary.cpp`; no subgroup width or threshold assumed (#558, #433) |

## 4. GPU practice

Short rules, each with the case that taught it. Most are from #795, the
template kernel for rigid water.

- **A decision that skips work must be uniform across the warp.** Lanes of
  a warp run together; a lane that rejects its item after enumeration
  waits for the others and saves nothing. In #795 a bounding-box test of
  partner groups was first placed per lane, inside the loop over
  candidates, and gained nothing; the correction moves it to the level of
  the group, where the whole warp skips a rejected group before
  enumerating its units, with the ballot and count emitters selecting the
  survivors.
- **f64 is slow on consumer GPUs.** A consumer GPU such as the RTX 3090
  runs f64 arithmetic at 1/64 of the rate of f32. In mixed precision, work per pair
  is f32 and f64 is kept where accuracy needs it: the state, the sums, a
  covering predicate that must not miss a pair
  ([white paper, Section 7](paper/07-precision.md)). Where an f64 test is
  needed, do it once per warp-uniform unit, not per lane: #795's per-lane
  box test did f64 arithmetic for every candidate.
- **Scattered loads are not coalesced.** A warp whose lanes read
  addresses far apart issues a transaction per lane. Read contiguous data
  in lane order (the state is sorted in space before the first step) and stage what a
  warp shares in block memory once. In #795 each lane of the per-lane test
  loaded six f64 values of a scattered partner box; it was worst on
  ubiquitin, whose solute partners are scattered.
- **Do not assume a subgroup width of 32.** Several existing layouts do
  (shuffle masks, warp layouts of the groups and tuple tables, the PME
  gathering layout); they are listed for audit in
  [device-primitives.md, Section 6.2](device-primitives.md#62-contract-and-choice).
  A new layout that needs 32 lanes states it as a precondition.
- **Respect the size preconditions of shared primitives.** Many are not
  checked. `emitSortBins` gives a bin to one thread and sorts it by
  insertion: it is for bins of a few items, and `emitRankInBins` is for
  bins of many. In #795 a refresh path sized its spatial grid with a
  constant, gave `emitSortBins` thousands of items a bin, and made each
  refresh take 18 to 188 ms; the step was 20 to 48 times slower than
  `main` ([workflow.md](workflow.md#a-piece-of-work); a shared grid plan
  and a size contract are #817, items 1 and 2).
- **Time against the prototype early.** A kernel with a prototype or a
  reference reports its time per kernel next to the prototype's from its
  first integration, and a large gap stops the work until its cause is
  known ([workflow.md, Performance](workflow.md#performance)).
