# MDIR internals: a guide for contributors

This page tells a contributor which design patterns the code uses and what
to use when adding something. It summarizes and links; the documents it
links are the statement. Start with [overview.md](overview.md) for what
MDIR computes and [principles.md](principles.md) for the rules a change is
reviewed against; the objects a developer meets at every level are in
[Table 3.4 of the white paper](paper/03-design.md#37-the-objects-of-mdir-for-developers). Every
directory of the repository, with when to edit it, is in
[Section 5](#5-where-things-are).

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

The table lists the directories of the stages only; every directory, with
when to edit it and its document, is in [Section 5](#5-where-things-are).

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

## 5. Where things are

Every directory of the repository, what it holds, when a change edits it,
and the document that describes it. The stage table of
[Section 1](#1-the-structure) gives the order of the lowering; this map is
the full list. The rules that hold across directories (generated files,
GPU tests, who runs which script) follow the tables, in
[Section 5.5](#55-conventions).

### 5.1 `lib` and `include`

A library under `lib/` has its public headers and TableGen files under the
same path in `include/mdir/`; a row names both.

| Directory | What is in it | When to edit it | Detailed doc |
|---|---|---|---|
| `lib/Dialect/MD`, `include/mdir/Dialect/MD` | The `md` dialect (what is computed): types, ops, verifiers, internal coordinates, the exchange-contract proof, grids and stencils | Adding or changing an `md` op, type, or verifier | [ops.md](ops.md), [tuples.md](tuples.md), [dialects/md.md](dialects/md.md) |
| `lib/Dialect/MD/Transforms`, `include/mdir/Dialect/MD/Transforms` | The passes over `md` and `dyn`: differentiation and its checks, truncation, inlining, couplings, reciprocal and stencil expansions, the rewrite to `md_dist`; `Passes.td` and the derivative and adjoint interfaces | Changing one of these passes or a derivative or adjoint rule | [primitives-map.md, Section 1.3](primitives-map.md#13-what-the-derivative-uses-and-how-it-is-declared), [white paper, Section 3](paper/03-design.md) |
| `lib/Dialect/Dyn`, `include/mdir/Dialect/Dyn` | The `dyn` dialect (time evolution): `dyn.program`, `dyn.step`, kicks, drifts, and the coupling ops | Adding or changing a `dyn` op | [dynamics.md](dynamics.md), [dialects/dyn.md](dialects/dyn.md) |
| `lib/Dialect/MDExec`, `include/mdir/Dialect/MDExec` | The `md_exec` dialect (how it is organized): loops over sets, neighbor builds, permutations, scatters, tables, value and storage forms; its shared analyses (algebra, independence, loop effects, scatter strategies, transfers) | Adding or changing an `md_exec` op or attribute, or one of its analyses | [ops.md](ops.md), [execution-contracts.md](execution-contracts.md), [dialects/md_exec.md](dialects/md_exec.md) |
| `lib/Dialect/MDExec/Transforms`, `include/mdir/Dialect/MDExec/Transforms` | The passes over `md_exec`: neighbor reuse and choice, fusion, sums and tables, precision, algebra, the scatter chooser, storage, streams, transfer checks | Adding or changing a pass between `convert-md-to-md-exec` and the final lowering ([Section 3](#3-to-add-something)) | [overview.md, Section 2.3](overview.md#23-the-main-passes), [neighbors.md](neighbors.md) |
| `lib/Dialect/MDDist`, `include/mdir/Dialect/MDDist` | The `md_dist` dialect (where it is computed): decompositions, halos, migration, and the proof of the halo schedule | Changing an `md_dist` op, type, verifier, or the halo schedule; reached through `mdir-opt` only | [distributed-ml2.md](distributed-ml2.md), [dialects/md_dist.md](dialects/md_dist.md) |
| `lib/Dialect/MDRT`, `include/mdir/Dialect/MDRT` | The `mdrt` dialect, the runtime as the IR sees it: opaque handles, buffer conversions, host calls, halo events | Adding a type or op that reaches the C runtime | [runtime.md](runtime.md), [dialects/mdrt.md](dialects/mdrt.md) |
| `lib/Dialect/Program` | No dialect: the one set of dialects a program may contain, shared by the driver and the compiler | A program may contain a new dialect; extend the list and its libraries together | |
| `lib/Conversion/MDToMDExec` | `convert-md-to-md-exec` (sums over relations become loops over sets) and the passes that place semantic linear transforms | A new `md` or `dyn` op needs a lowering, or the mapping to loops changes | [ops.md](ops.md), [pme-semantic-transform.md](pme-semantic-transform.md) |
| `lib/Conversion/MDExecKernels` | What one thread does, shared by both lowerings, and the device-primitive emitters (compaction, scans, sorts, reductions, scatters, gathers, ragged rows); `ExpandPrimitives.cpp` expands `@mdir.primitive.<name>` | Changing per-thread code common to both lowerings, or adding or changing an emitter | [device-primitives.md](device-primitives.md), [primitives-map.md](primitives-map.md) |
| `lib/Conversion/MDExecToLoops` | The CPU lowering to `scf` loops, fixed-order reductions, the read-only proof; embeds the host templates | Changing how `md_exec` runs on the CPU or with OpenMP | [ops.md](ops.md), [overview.md, Section 2.4](overview.md#24-the-lowerings) |
| `lib/Conversion/MDExecToGPU` | The device lowering to `gpu` kernels and the lowering of neural-stage bodies; embeds the device templates | Changing how `md_exec` becomes kernels or how they are launched | [ops.md](ops.md), [device-primitives.md](device-primitives.md), [ml4-ir.md](ml4-ir.md) |
| `lib/Conversion/GPUToBinary` | The NVIDIA back end: NVVM lowering, `ptxas`, one binary per module, cached | Changing NVVM options, the target architecture, or the caching of device code | [compile-cache.md](compile-cache.md) |
| `lib/Conversion/MDDistToLoops` | The in-process executors of `md_dist`, the lowering of halo events, the printed schedule | Changing how `md_dist` executes; reached through `mdir-opt` only | [distributed-ml2.md](distributed-ml2.md) |
| `include/mdir/Conversion` | `Passes.td` of every conversion pass, and the headers of the emitters and the back end | Declaring a conversion pass or option, or changing an emitter's interface or preconditions | [device-primitives.md, Section 1.1](device-primitives.md#11-preconditions-at-a-glance) |
| `lib/Runtime/Templates` | Structures written in MLIR: neighbor builds, the spatial sort, PME on the host and on a device; most are generated | Changing a neighbor build or the PME code of a template (through its generator where there is one) | [white paper, Section 3.5](paper/03-design.md#35-lowerings-and-templates), [pme.md](pme.md), [neighbor-groups.md](neighbor-groups.md) |
| `lib/Compiler`, `include/mdir/Compiler` | The pipeline of a run (`getPipeline`), the host optimizer, the JIT and its memory, the compile cache, the registry of passes, the persistent simulation, the choice of skins | Adding or moving a pass in the pipeline, or changing the JIT, the cache key, or the simulation | [overview.md](overview.md), [compile-cache.md](compile-cache.md), [jit-invariants.md](jit-invariants.md), [python-segments.md](python-segments.md) |
| `lib/Driver`, `include/mdir/Driver` | The control file, the topology readers, the model, the builder of the program and its `Term*.cpp` builders, checkpoints, outputs, tunables | A control-file key, a reader, a term, the dynamics, an output, or a checkpoint changes | [driver.md, Section 2.12](driver.md#212-the-files-of-the-builder), [Appendix B.2](paper/B-contributing.md#b2-where-a-change-goes) |
| `lib/Effects`, `include/mdir/Effects`, `include/mdir/Analysis` | The declared effect summaries of programs and host calls, with the singleton resources; the access facts behind the loop effects of `md_exec` | Adding a declared resource, or changing an effect or access summary | [execution-contracts.md](execution-contracts.md), [dynamics.md](dynamics.md) |
| `lib/MLIP`, `include/mdir/MLIP` | The optional metatomic plugin, loaded at run time, and its C boundary | Changing the learned-potential plugin or its ABI (bump its version) | [metatomic-ml1.md](metatomic-ml1.md) |
| `include/mdir/Runtime` | C headers shared by the compiler and the runtime: the slots of neighbor-group buffers and the copy audit | Changing their layout; bump `MDRT_GROUPS_ABI_VERSION`, which is in the compile-cache key | [runtime.md](runtime.md), [neighbor-groups.md](neighbor-groups.md) |
| `include/mdir/Support` | `PhysicalConstants.h`, the constants and unit factors of C, C++, and Python | Adding or changing a constant; regenerate the Python module | [conventions.md](conventions.md) |

### 5.2 `test`

| Directory | What is in it | When to edit it | Detailed doc |
|---|---|---|---|
| `test` | `lit.cfg.py` (substitutions, features), `mdir_lit.py` (the GPU group, the list of expected unsupported tests), `expected-unsupported.txt`, and `CMakeLists.txt`, which builds the unit-test executables | Adding a substitution, tool, or feature, or a test that `REQUIRES` an optional feature | [building.md, Section 6](building.md#6-testing), [workflow.md, Validation](workflow.md#validation) |
| `test/Dialect` | IR tests of the dialects and their passes through `mdir-opt`; errors in `*-invalid.mlir` | Changing an op, a verifier, or a pass under `lib/Dialect` | [ops.md](ops.md) |
| `test/Conversion` | IR tests of each lowering, one subdirectory per pass; the device lowering is checked as IR, without a device | Changing a lowering under `lib/Conversion` | [ops.md](ops.md) |
| `test/Driver` | Most of the suite: `mdir run`, `emit`, and `check` on control files, the Python package, and the unit-test sources; oracles and systems in `Inputs/` | A key, a term, an integrator, a coupling, an output, or the Python API changes; a feature adds its CPU test and its `*-gpu.test` | [driver.md](driver.md), [validation.md](validation.md) |
| `test/Integration` | Whole MLIR programs lowered and run on the host, checked against references | A lowering or runtime path that needs a whole program, or a kernel checked at the level of the IR | [building.md, Section 7](building.md#7-tools) |
| `test/Integration/GPU` | The device forms of those programs, mostly wrappers of the parent's `.mlir` files | Changing the device lowering or the CUDA runtime, or a parent program it wraps | [building.md, Section 7](building.md#7-tools) |
| `test/Runtime` | Tests of `mdrt` and of the templates; `generated-templates.test` compares each generated template with its generator | Changing the runtime, a template, or a generator | [runtime.md](runtime.md) |
| `test/Sanitizer` | Driver and Python runs under compute-sanitizer; they run only with `lit -Dsanitize=1` | Device code that could read uninitialized memory or race, or a regression test of such a defect | [debugging.md, Tiers of tests](debugging.md#tiers-of-tests) |
| `test/Scale` | `amber-suite.test`: short runs of every system of the Amber benchmark suite, when `MDIR_BENCH_DIR` holds it | Anything only large systems reach | [amber/README.md](../scripts/benchmarks/amber/README.md) |
| `test/Docs` | The link check of all Markdown, the comparison of `docs/dialects/` with TableGen, the anchors after decision numbering | Any Markdown, a heading, or a dialect `.td` file changes | [workflow.md, Reusing a suite](workflow.md#reusing-a-suite) |
| `test/Lit` | Tests of the test setup and of the repository's scripts (release, changelog, builder equivalence, environment), on the CPU | Changing `lit.cfg.py`, `mdir_lit.py`, a script, `docs/environment.md`, or the rules for test outputs | [workflow.md, Validation](workflow.md#validation) |
| `test/lib` | Test-only passes linked into `mdir-opt`, such as `--test-md-exec-loop-effects`; not tests | A pass that exists only to expose an analysis to FileCheck | |

### 5.3 `scripts`

| Directory | What is in it | When to edit it | Detailed doc |
|---|---|---|---|
| `scripts`: `check-before-push.sh`, `check-markdown-links.py`, `check-physical-constants.py`; `scripts/hooks` | The static checks of CI's `checks` job (links, decision numbers, changelog fragments, expected unsupported tests, the release path, characters); `hooks/pre-push` runs them | Adding a repository-wide check that CI and the hook both run | [workflow.md, Continuous integration](workflow.md#continuous-integration) |
| `scripts`: `generate-*`, `mdir_physical_constants.py`, `fit-erfc.py` | The generators of the device templates, the dialect references, and the Python constants; the fit of the erfc polynomial of `Approximate.cpp` | Changing a generated template, a dialect `.td` file, a constant, or the erfc fit; edit the generator and commit its output with it | [device-primitives.md](device-primitives.md), [pme.md](pme.md), [conventions.md](conventions.md) |
| `scripts`: `decision-number.sh`, `decisions-table.py`, `decision-number-anchors.py` | Numbering of `D[<label>]` decisions, the table of `docs/decisions.md`, anchors after numbering | Changing how decisions are numbered or indexed | [workflow.md, Decision numbers](workflow.md#decision-numbers) |
| `scripts`: `build-*.sh`, `check-sdist.sh`, `environment.sh`, `free-energy.py` | Builds of LLVM, HDF5, and a sanitized tree; the sdist check; the helpers that bash scripts source; free energies by TI and MBAR | Changing toolchain versions, the sanitizer build, the sdist, or the variables scripts need | [building.md](building.md), [environment.md](environment.md) |
| `scripts/builder-equivalence` | The check that two builds print the same module and lowered program over a fixed corpus | A staged move of the builder or a change that must leave the program unchanged | [driver.md, Section 2.11](driver.md#211-the-equivalence-check-of-the-builder) |
| `scripts/paper` | The PDF build, references, the appendix check, and the figures and tables of the white paper | Changing the paper's build, references, or measured figures | [Appendix B](paper/B-contributing.md) |
| `scripts/release`, `scripts/packaging` | Preparing, checking, and publishing a release; the changelog tool; `build.sh`, the one entry point of image, tarball, and wheel builds | Changing the release procedure, its checks, the changelog format, or a build target | [workflow.md, Releases](workflow.md#releases), [packaging/README.md](../packaging/README.md) |
| `scripts/mlip` | Converters of learned potentials, the export of neural stages, water boxes, checks, and pinned requirements | Working on the model term, neural stages, or the ML4 baselines | [metatomic-ml1.md](metatomic-ml1.md), [ml4-baselines.md](ml4-baselines.md) |
| `scripts/render` | Movies of trajectories, and the SVGs of `docs/figures/primitives` from their `.dot` sources | Changing those figures | [primitives-map.md](primitives-map.md) |
| `scripts/validation/{forces,gromacs,charmm,protein,suite,nep,p3m}` | Energies and forces against sander, pmemd, GROMACS, CHARMM, and independent oracles of NEP and of PME's influence functions | Changing a force term, a reader of force fields, PME, or NEP | [validation.md](validation.md), [white paper, Section 9](paper/09-correctness.md) |
| `scripts/validation/{nve,ensembles,kinetic-step,pressure-step,barostat,barostat-semi,gpu-cpu}` | Conservation, ensemble statistics, dependence on the time step, the barostats, and the GPU against the CPU | Changing the integrator, a thermostat, a barostat, the virial, or device code that must agree with the CPU | [validation.md](validation.md), [dynamics.md](dynamics.md) |
| `scripts/validation/{free-energy,frames}` | The hydration free energy of ethanol against OpenMM and GROMACS; oracles of the stored-frame evaluator | Changing free-energy perturbation or the frame evaluator | [python-frames.md](python-frames.md), [white paper, Section 6](paper/06-dynamics.md) |
| `scripts/benchmarks/amber` | `bench.py`: prepares the Amber benchmark suite and runs MDIR, pmemd, and GROMACS on it | Changing the performance protocol, or timing a change against `main` | [amber/README.md](../scripts/benchmarks/amber/README.md), [white paper, Section 10](paper/10-performance.md) |
| `scripts/benchmarks/{cpu-breakdown,constraints}` | The CPU target against other engines with recorded baselines; the constraint solver with and without the analytic solve | Changing CPU-target performance or the constraint solver | [cpu-breakdown/README.md](../scripts/benchmarks/cpu-breakdown/README.md), [constraints/README.md](../scripts/benchmarks/constraints/README.md) |
| `scripts/benchmarks/{fused-prune,rolling-prune,device-scan,pme-tiles,prolate-ewald}`, `scripts/experiments/neighbor-structures` | Studies of neighbor lists and PME: pruning, group scans, spreading layouts, prolate Ewald, and standalone CUDA prototypes | Changing or measuring a neighbor structure or a PME layout | [neighbor-groups.md](neighbor-groups.md), [pme.md](pme.md), [neighbor-structures/README.md](../scripts/experiments/neighbor-structures/README.md), [prolate-ewald/README.md](../scripts/benchmarks/prolate-ewald/README.md) |
| `scripts/benchmarks/{activation,frames,analysis,mdbench}`, `scripts/benchmarks/fft-resident.py` | Times of Python activations and frame evaluation, of analysis tools, of engines on MDBench, and of host against device FFTs | Changing what one of them measures | [python-segments.md](python-segments.md), [python-frames.md](python-frames.md), [fft-device-resident.md](fft-device-resident.md) |

### 5.4 The rest

| Directory | What is in it | When to edit it | Detailed doc |
|---|---|---|---|
| `tools/mdir` | The command `mdir`: one file per subcommand (run and emit, check, checkpoint, doctor, version and bug report) and the run manifest; the logic is in `lib/Driver` | A subcommand, a flag, or the user-facing output changes | [driver.md, Section 2.9](driver.md#29-the-command-line), [debugging.md](debugging.md) |
| `tools/mdir-opt` | `mlir-opt` with every MDIR dialect and pass and the test passes; built only with the tests | A dialect, a group of passes, or a test pass is added | [debugging.md, Failures of the passes](debugging.md#failures-of-the-passes) |
| `runtime` | The runtime in C: `libmdrt` (host helpers, the PME FFT on pocketfft, the transport of `md_dist`) and `libmdrt_cuda` (driver API, cuFFT, cuBLAS); install and bundled notices | Compiled code needs a new entry point, or device memory, streams, copies, or the install layout change | [runtime.md](runtime.md), [dialects/mdrt.md](dialects/mdrt.md) |
| `python` | The pybind11 extension `mdir._core`: bindings, DLPack, NumPy interchange, units, the XLA FFI handlers | The Python API or the wheel's contents change | [python.md](python.md), [python-package.md](python-package.md) |
| `python/mdir` | The pure-Python package: re-exports, the frame evaluator, the JAX and PyTorch adapters | The frame evaluator or an adapter changes | [python-frames.md](python-frames.md) |
| `cmake` | `BuildInfo.cmake` (the commit for `mdir version`) and the probe of `std::from_chars` | Adding a probe at configure time or a value of the build info | [building.md](building.md) |
| `packaging` | The Dockerfiles of the image, the CI image, the manylinux tarball and wheels; the Apptainer recipe; license texts | The base image, versions, bundled libraries, or notices change | [packaging/README.md](../packaging/README.md), [licenses/README.md](../packaging/licenses/README.md) |
| `third_party` | Vendored, unchanged: pocketfft, toml++, the XLA FFI headers | Upgrading or adding a vendored library, with its license, origin, and notice | [pocketfft/README.md](../third_party/pocketfft/README.md), [xla_ffi/README.md](../third_party/xla_ffi/README.md) |
| `examples/ala3` | Tri-alanine in OPC: the four stages, which are also the templates that `mdir template` prints | Keys of the stages or the standard pipeline change; reconfigure, since the stages are embedded | [ala3/README.md](../examples/ala3/README.md), [driver.md, Section 1.3](driver.md#13-standard-pipeline-templates) |
| `examples/argon` | Liquid argon at constant energy, as a control file and as a hand-written `md` module; many tests use it | The `md` syntax or the minimal control file changes | [driver.md](driver.md), [ops.md](ops.md) |
| `examples/{ubiquitin,popc-bilayer,kcl-osmotic}` | Ubiquitin in OPC, a POPC bilayer, and the osmotic pressure of KCl by reweighting; inputs are built, not committed | The keys they use change; a new example follows the four stages | [ubiquitin](../examples/ubiquitin/README.md), [popc-bilayer](../examples/popc-bilayer/README.md), [kcl-osmotic](../examples/kcl-osmotic/README.md) |
| `docs` | The design and user documents, one per topic, kept as the current state | Any implementation change, with the roadmap and the paper | [workflow.md](workflow.md), [principles.md](principles.md) |
| `docs/paper` | The white paper as Markdown sections and appendices; `02-notation.md` defines the symbols | A method, a result, or the control file changes | [paper/README.md](paper/README.md), [Appendix B](paper/B-contributing.md) |
| `docs/decisions` | One file per decision; a PR adds `<label>.md` | A PR records a decision | [decisions.md](decisions.md), [workflow.md, Decision numbers](workflow.md#decision-numbers) |
| `docs/dialects` | The dialect references, generated from TableGen | Never by hand: regenerate after editing a `.td` file | [ops.md](ops.md) |
| `docs/release-notes`, `docs/figures` | The notes of each release; the figures of the design documents with their sources | A release is prepared; a design figure changes | [workflow.md, Releases](workflow.md#releases), [primitives-map.md](primitives-map.md) |
| `changelog.d` | One changelog fragment per PR | Every PR adds its own | [changelog.d/README.md](../changelog.d/README.md) |
| `.github` | The CI workflows (the CPU checks and build, the CI image, decisions, the release path and tag check, PyPI) and the issue templates | Changing CI or the release automation; a pre-merge check goes in `check-before-push.sh` | [workflow.md, Continuous integration](workflow.md#continuous-integration) |
| Files at the root | `CMakeLists.txt` (options, the version), `pyproject.toml` (the wheel), `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md` | An option or the wheel changes; the version and `CHANGELOG.md` change only at a release | [building.md](building.md), [python-package.md](python-package.md) |

### 5.5 Conventions

- **Generated files are not edited by hand.** Edit the source and
  regenerate; a test compares them. `docs/dialects/*.md` come from the
  `*Ops.td` files (`ninja mdir-dialect-docs`; `test/Docs/dialect-docs.test`).
  The PME and group templates, and the part between `BEGIN GENERATED` and
  `END GENERATED` of the matrix templates, come from
  `scripts/generate-*-template.py` (`test/Runtime/generated-templates.test`).
  `scripts/mdir_physical_constants.py` comes from
  `scripts/generate-physical-constants.py`
  (`test/Driver/physical-constants.test`). `test/Integration/GPU/tuples*.mlir`
  come from `test/Integration/Inputs/generate_tuples_tests.py`. The table of
  `docs/decisions.md` and `CHANGELOG.md` are written by the merger's and the
  release's scripts.
- **Embedded at configure time.** The templates of `lib/Runtime/Templates`
  and the stages of `examples/ala3` are read into `Templates.cpp` from
  `Templates.cpp.in` when CMake configures; after editing them, configure
  again. The files under `python/mdir` are copied into the build tree, so
  rebuild after editing them.
- **GPU tests.** The line `REQUIRES: cuda` makes a test a GPU test: it puts
  it in the group that runs 16 at once and makes it unsupported without
  CUDA. A GPU test is named `*-gpu.test`; the exceptions are the files of
  `test/Integration/GPU`, `test/Scale/amber-suite.test`, and
  `test/Driver/analytic-bonds-triclinic.test`. Release tests are
  `*-release.test` and `*-release-gpu.test` (`-Drelease=1`).
- **Expected unsupported tests.** `test/expected-unsupported.txt` lists them
  by the feature that decides it. A PR that adds or changes a `REQUIRES` or
  `UNSUPPORTED` line changes the list with it; `check-before-push.sh` and
  `test/Lit/expected-unsupported.test` check it.
- **Outputs of tests.** A test writes only inside an emptied `%t`
  (`test/Lit/test-outputs.test`). lit never collects files under a
  directory named `Inputs` or `lib`.
- **Who runs which script.** The build runs `cmake/BuildInfo.cmake` and
  embeds the templates. Tests run the link checker, the generators of
  templates and dialect references, the builder-equivalence check, the
  appendix check, `scripts/benchmarks/amber/bench.py` (through
  `test/Scale`), and the `scripts/mlip` checks. CI and the pre-push hook run
  `scripts/check-before-push.sh`; enable the hook once per clone with
  `git config core.hooksPath scripts/hooks`. Everything else (generators,
  validation, benchmarks, the paper) is run by hand. The maintainer alone
  runs the publishing scripts and workflows, and the merger alone runs
  `scripts/decision-number.sh assign`.
- **Scripts name no machine paths.** A value of a machine comes from the
  environment or from `MDIR_ENV_FILE`, and a new variable adds its row to
  [environment.md](environment.md) (`test/Lit/script-paths.test`,
  `test/Lit/script-environment.test`).
- **Vendored code** under `third_party` is not edited; a change is an
  upgrade from upstream with its origin recorded.
