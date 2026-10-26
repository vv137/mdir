# Roadmap

Status: each row says where its item stands and what comes next, with the
issues that hold the work (#603); the history of an item is its decisions,
in [decisions.md](decisions.md), whose rows hold the measurements. The
results measured as the first milestone was built are in
[validation.md](validation.md), Section 1, and the principles that the work
follows are in [principles.md](principles.md).

**Milestones** (D169; M2 split into M2a and M2b by D195):

| Milestone | Scope | Status |
|---|---|---|
| M1 | An all-atom protein in water with an Amber force field, on one node, CPU and GPU; at least the rate of pmemd.cuda on every system of the Amber suite | Done (D114, 2026-10-02); released as 0.1.0. Under one protocol for both programs (D327) the target holds on every system ([white paper, Section 10](paper/10-performance.md)) |
| M2a | The Python API (Section 6) | Done: items 1 to 8 of [python.md](python.md), Section 4; the last was `observe` in the Python model (D232, #188) |
| M2b | Differentiable simulation: tunable parameters, a frame evaluator with derivative rules, PyTorch then JAX adapters, and trajectory reweighting [[ThalerZavadlav2021]](references.md#thalerzavadlav2021) (Section 6.1) | In progress: the derivative in the tunables (D230, #203), the frame evaluator and its PyTorch adapter. Next: JAX, force matching (#138) |
| M3 | Learned potentials on one GPU: ML1 steps 1 to 3, the IR of ML4, PaiNN in stages, and MACE in stages (Section 7, D287) | In progress: ML1 (D270, D273, D277), ML4's baselines (D286), its IR (D288, D289), PaiNN in stages (D294, D293) and step 4b (D299, D302, D303, D318, D329). Next: the rest of step 4b (#462), MACE in stages (#377) |
| M4 | Distributed execution: ML2, ML3, ML5, and distributed particle mesh Ewald (Section 7) | In progress, ML2 beside M3: `md_dist` in the compiler on an in-process executor (D275, D276, D279, D290, D307, D308, D310, D313, D332). Next: migration and the first `[distribute]` run (#38), MPI on CPU ranks; ML3, ML5, and distributed PME are planned |

The size sweeps of M3 take boxes of more than 99,999 waters (D339, #550).
Features of classical MD (Section 5) continue between milestones.

## 1. Robustness (under way)

| Item | State and next |
|---|---|
| Robust differentiation (#15) | Done: activity analysis (D182), rule coverage (D183), the f64 checker (D184), singular-point conventions (D185), term names in diagnostics (D186), and declared transposes of `md` with `--md-check-transposes` (D295, #451) |
| Principles of development | Done ([principles.md](principles.md)) |
| Tests under compute-sanitizer; short runs of the Amber suite | Done (`test/Sanitizer`, `test/Scale`) |
| Full-suite wall time (#485) | Done in part: two Slurm shards (D309), the release tier for the long Python matrices (D324, #530). Next: profile the other long Python driver tests |
| Portable device primitive boundary (#558, #433) | In progress: the inventory of NVVM, PTX, and GPU operations with their emitters (D340, [device-primitives.md](device-primitives.md#6-portability-one-target-contract-several-implementations)). Next: one operation group per PR behind emitters, one capability and measured-policy record, then the runtime's device calls behind an interface; NVIDIA output stays identical |
| Declared semantic attributes (#607) | In progress: differentiation uses the typed accessors of tuple coordinate kinds and member places (D343), the storage-form module marker has one checked helper (D349, #636), and batched stage rows use a checked function attribute (D[stage-rows-attr], #642). Next: the other stage, storage, transfer, and attribute families, and the ops without a verifier, each a slice of its own |
| Kernels named after their op and line | Done |
| Reports of defects that can be reproduced | Done: `mdir bug-report`, crash reproducers of the passes, versions, an issue template ([debugging.md](debugging.md)) |
| The set of each buffer in the type of the storage form; memory planning as a pass of its own | In progress: field, scratch, tuple-member, and incidence buffers carry their set (D337, #565). Next: a lifetime-based memory planner and set types for the other structural buffers |
| A summary of the effects of each loop op, from which fusion decides | In progress: the loop effect interface (D342, #566). Next: checked summaries shared by fusion, streams, and distribution |
| Copies between host and device ordered by async tokens in the IR | After the effects |
| A debug lowering that checks bounds and the residuals of iterative solvers | With the storage types |
| The barostat with positional restraints: a defect | Fixed 2026-10-02 (D124; `reference_scaling`, [validation.md](validation.md#the-barostat-with-positional-restraints-d124)) |
| The conserved energy of the Trotter count under a fast compression | Closed 2026-10-02, not a defect (D92); a run that changes its volume fast is checked with `work = "EXACT"` |
| A build of the driver and runtime under AddressSanitizer and UndefinedBehaviorSanitizer | Done (`scripts/build-sanitized.sh`) |
| The declared external-input set boundary (#607, #625) | The unit marker on `memref.cast` has a named writer and reader; the dialect checks its placement, value, and set provenance (D345). Other semantic attribute families remain in #607. |

## 2. The rest of the first milestone

| Stage | State and next |
|---|---|
| M1e | Done (2026-10-02): the readers and renumbering, Amber's formats and conventions (D133), flexible water against sander ([validation.md](validation.md), Section 1) |
| M1f | Done: the terms and dynamics against AmberTools, GROMACS, and CHARMM (white paper, Section 9) |
| M1k | Done (2026-10-03): the Amber suite term by term against pmemd 26 and sander (D133), conservation, volumes at constant pressure, and rates as one model (D134; white paper, Sections 9 and 10) |
| Integrators | Done: leapfrog with everything velocity Verlet has (D76) |
| Comparison on a protein | Done (2026-10-02, `scripts/validation/protein`): ubiquitin in OPC against GROMACS 2026.3 (white paper, Section 10.6) |
| The rest of the step against GROMACS | In part (D118). Next: the fused integration kernel and the kernel of the tuples across warps (with Nsight Compute counters), and CUDA graphs for the launches (#60) |
| The CPU target against GROMACS (D267, #359) | In part: the host code at O3 (D269), PME on the host in parallel (D271), and a recorded baseline (D333, #552, #549). Next: the host `vector(W)` pair-loop strategy (#554, D267 step 4), then the builds of the list and the rest of the step |
| Performance | Done: D78, D79, D80, D81, D83, D87, D262, D283, D285 (#416, #415), D291, D296 (#432), D301 (#422) ([device-primitives.md](device-primitives.md)). Next: one kernel for the kick, drift, and constraints of each group; host synchronization on the device; CUDA graphs; the tile structure (D82, [neighbor-tiles.md](neighbor-tiles.md), stages T1 to T4) |
| Device scan strategies (#420) | In part: the single-pass scan (D321) is opt-in; the three-launch scan stays the default. Next: further sites only with a measured critical path ([device-primitives.md](device-primitives.md)) |
| Device gather strategies (#419) | In part: shared gather emitters (D325, #539). Next: vectorized or transposed loads where measured; tiled gathering at orders 6 and 8 (#411) |
| Barostat integrators | Done: the exact work (D77), the Trotter integrator (D92), the virial of the groups (D116), the ensemble test of two pressures (white paper, Section 9.5). Next: the effective energy of [[Bernetti2020]](references.md#bernetti2020), eq. S11, as a diagnostic |
| CHARMM force fields | Done (D121, D122, [charmm.md](charmm.md)). Next: CHARMM-GUI's triclinic cells and cell files, VSWITCH (done since as D175, F3), the lone pairs of CGenFF, and a bilayer at constant pressure against GROMACS and CHARMM |
| PME grid axes | TODO: the longest edge as the contiguous axis of the grid, with the tables of D104 and the other stages on the permuted axes ([pme.md](pme.md#the-axes-of-the-grid)) |
| Ewald summation with prolates | Done (D266, #334; measured again with tiles, D282, #336; the direct sum as a table, D280, #380; [pme.md](pme.md#7-the-prolate-kernel-d266), Section 7) |
| Influence functions of P3M (#468) | In part: step 1, `[pme] influence = "P3M"` (D304; a triclinic cell is #475). Next: the self-force subtraction, the a priori error estimate and a choice of $(\beta, K, P)$, interlacing, and the prolate kernel with an optimal $G$ (#467) |
| Dual pair lists | Done (D114; the default since D261; D268, #357; skins by measurement, D281, #406; fused pruning, D284, #362) |
| Device memory on large systems (#471) | In part: D311, D312, D314, D315, D316, D320 (#517), D330, D331; STMV ×15 (16.01 M atoms) runs on one RTX 4090. Next: ×16, which runs out of memory after step 0 |
| Large-system startup (#520) | In part: D316 and D323 (#525). Next: topology reading, validation, and the rest of the setup (#520) |
| Physical validity ([[Merz2018]](references.md#merz2018)) | Done: the kinetic energy and the ensembles at two temperatures and two pressures, isotropic and semi-isotropic ([validation.md](validation.md#physical-validity), white paper, Section 9.5). Next, in order: two tensions on a bilayer; equipartition between solute and solvent and among motions; $\sigma_E \propto \Delta t^2$ at constant energy |
| Periodic reordering of the state | TODO: the loops over pairs read the order of the last build (D86), the gathers and scatters that of the state; measure the loss over 10 ns of a large system first ([neighbors.md](neighbors.md#the-order-of-the-state)); reorder every N builds if it exceeds about 1% |
| Pairs and lists, next | Measure first, then a box test of a group in the pruning and forces of the groups in fixed point, as the grid of PME (D70) ([neighbor-groups.md](neighbor-groups.md#next)) |
| Other engines on MDBench | Preliminary (2026-10-02, [validation.md](validation.md#other-engines-on-mdbench)). Next: three repeats on GPU 0 alone, 9naw, runs with other seeds |
| Semi-isotropic barostat | Done (D119). Next: the area per lipid, which needs runs of 100 ns or more |

The goal of the first milestone, its history, and the rates of the first
comparison are in [validation.md](validation.md#the-goal-of-the-first-milestone)
(D114, D118, D133, D134).

## Documentation

| Item | State |
|---|---|
| Equations in the Markdown documents written in TeX (`$...$`, `$$...$$`) instead of Unicode text | Done (2026-10-02) |
| A guide for contributors | Done: Appendix B of the white paper, and `CONTRIBUTING.md` |

## 3. White paper, after the first milestone

Written 2026-10-02 ([docs/paper/](paper/README.md)), built as a PDF with
pandoc and tectonic (`scripts/paper/build-pdf.sh`); a change to a method,
an algorithm, or an implementation updates it. The performance of the
suite is measured against pmemd.cuda (the user, 2026-10-01), and the
comparison on a protein (Section 10.6) against GROMACS. Every citation is
checked against its source (docs/references.md), and the figures are
generated from the logs by scripts in the repository.

| Part | State |
|---|---|
| Notation (Section 2) | Done |
| Every method as implemented | TODO: keep the design documents complete in TeX as features land |
| Usage (Appendix C, [C-usage.md](paper/C-usage.md)) | Done (2026-10-03); its section on outputs follows U12 |
| The design, the neighbor algorithms (Section 4: D80, D82, and D89 to D106, among them D95, D97, D99, D100, D105), correctness (the target of D65), performance, precision (Section 7, D75) | Done |
| Related work (Section 11) | Done, every source read on 2026-10-02 ([paper/README.md](paper/README.md#related-work)) |
| The principles of development; limitations and the next milestones | Done |
| A manual of the control file (Appendix A) | Done, checked by `scripts/paper/check-appendix.sh` |
| How to take part (Appendix B) | Done |

## 4. Usability (after the first milestone)

From a review of how a researcher meets MDIR (2026-10-02). The order is
that of the work; P0 is what a production run on a cluster needs.

| # | Item | Priority | State |
|---|---|---|---|
| U1 | `steps` as the total of a run, and `mdir run --continue` | P0 | Done (D129, D172; [driver.md](driver.md), Section 2.7) |
| U2 | Output that continues with the run | P0 | Done (D130, D149) |
| U3 | A stop on SIGTERM, SIGINT, or `--max-walltime` | P0 | Done (D131; exit status 75) |
| U4 | Keep the previous checkpoint | P0 | Done (D132, D173: `<checkpoint>.prev`, `fsync`, SHA-256) |
| U5 | A directory for a run: the log in a file, protection against overwriting, a manifest | P1 | Done (D149, D168; [driver.md](driver.md), Section 2.8) |
| U6 | `mdir check` as a preflight | P1 | Done (2026-10-03, D151) |
| U7 | Errors that say what to do | P1 | Done (2026-10-03, D146; the constraints' step and set, D278, #349) |
| U8 | A quickstart first in the README, and `mdir doctor` | P1 | Done (D155) |
| U9 | Control files of the standard pipeline from `mdir template` | P2 | Done (2026-10-03, D147) |
| U10 | Distribution | P2 | Done in part: the container recipes in `packaging/`, whose kernels are cubins (a driver older than the toolkit cannot compile PTX; D214). 0.2.0 is source only |
| U11 | Many runs of one plan (as GROMACS's `-multidir`) | Later | With `ensemble` |
| U12 | Outputs as one system | P1 | Done (2026-10-03, D149, D151). Not taken: intervals of their own for the files of columns |
| U13 | The cost of a checkpoint on a large system | P1 | Done (#548, D335, D336). Next: the DCD frame between the steps, and the order on the device with a copy into pinned memory (#567) |

Not taken: `target = "AUTO"` (a check would pass on one machine and fail
on another, and a large run could fall back to the CPU without a word),
runs without a number of steps (the counts of the schedule are constants of
the program), and checkpoints by the clock (they would break the exact
continuation).

## 5. Features for the purpose of MDIR

MDIR is for general MD, all-atom and coarse-grained, on workstations and
clusters ([decisions.md](decisions.md), Section 1, C1 and C2). What runs of
that kind still lack, in the order of how often they are needed:

| # | Feature | State and next |
|---|---|---|
| F1 | Langevin dynamics (BAOAB) | Done (D135; g-BAOAB with constraints, D265, #268; D263, D264). Next: a thermostat consistent with the shadow Hamiltonian (#118), and the energy exchanged with the bath for a conserved energy |
| F2 | Triclinic cells | Done but the rates (D123, D125 to D127, D241, D242, #258, #263; [triclinic.md](triclinic.md)). Next: P5, pmemd.cuda and GROMACS on the same systems |
| F3 | CHARMM force fields | In part (D121, D122, D127, LJ-PME D162, VSWITCH D175). Missing: lone pairs, the Drude model |
| F4 | Outputs for analysis | In part: XTC (D141), the cell area (D170), H5MD frames (D239, [python-h5md.md](python-h5md.md)). Missing: velocities in the frames of `mdir run`, and the pressure tensor |
| F5 | Restraints beyond positions | Done (D136, D74, D139, D145, D148) |
| F6 | Coarse-grained models: tabulated potentials, Martini from GROMACS topologies, DPD | Missing; part of the purpose from the start |
| F7 | Free energy and enhanced sampling | In part: decoupling with $\partial U/\partial\lambda$ and MBAR (D161, D2), and `observe` (D189, #16). Next: relative transformations, annihilation, Coulomb soft-core, $\lambda(t)$, replica exchange (P4), one pass over the selection's pairs; for #16 profiles, RDF, RMSD, adaptive biases, and a bridge to PLUMED |
| F8 | Learned potentials | Section 7 |
| F9 | The custom forces of OpenMM beyond tuples | In progress (D137, D138, D139, D140, D142, D143, D144, D150, D152, D160, D165, D178, D188, #67). Next: the triplets' GPU kernel, selections, topology, and bond order (D160); the first reduction inside the loop of the sums (D150); GROMACS's seven-column tables for F6 |
| F10 | Thermostats, integrators, and barostats beyond M1 | In progress (D163; D163a with D206, D163b, D163c). Next, as designs: multiple time steps ([[Tuckerman1992]](references.md#tuckerman1992), [dynamics.md](dynamics.md#multiple-time-steps)), a flexible cell with the pressure tensor (F4), and the couplings as ops of `dyn` (#541, D326, [dynamics.md](dynamics.md), Section 2.2) for M2b and the fusion of #356 |

## 6. Python API (M2a)

D195 splits M2 in two: M2a is this section, the Python API as planned in
[python.md](python.md) (D187, with the maintainer's rulings of 2026-10-04);
M2b is Section 6.1. M2a is done; each part is a decision and a document:

| Part | Decisions and documents |
|---|---|
| Preparation, loading, compiling, host arrays | D191 ([python-model.md](python-model.md)), D192 ([python-compile.md](python-compile.md)), D193 (#78, [python-arrays.md](python-arrays.md)) |
| Persistent simulations and their ownership | D196 (#85, [python-segments.md](python-segments.md)), D199 (#89, [jit-invariants.md](jit-invariants.md)), D211 (#147), D215 (#135) |
| Velocities, restraints, units, minimization | D198 (#92, [python-velocities-restraints.md](python-velocities-restraints.md)), D200 (#95, [python-units.md](python-units.md)), D202 (#91, [python-minimize.md](python-minimize.md)), D219 (#106), D274 (#369) |
| Reporters and the example | D207 (#109, [python-reporters.md](python-reporters.md)); `examples/ala3/run.py` (#82) |
| The compile cache | D212 (#142, [compile-cache.md](compile-cache.md)), D214 (#148), D217 (#163), D227 (#162), D234 (#196), D236 (#236) |
| Tunable parameters | D213 (#130, [python-tunable.md](python-tunable.md)), D226 (#160) |
| The model's terms and settings | D222 (#161), D232 and D233 (#188, [python-observe.md](python-observe.md)), D245 (#270; the default `AUTO` since D261, #350), D246 (#279), D247 (#275), D248 (#281), D249 (#278) |
| Views, borrows, topology, checkpoints, the package | D220 (#131, [python-dlpack.md](python-dlpack.md)), D229 (#136), D238 (#206), D244 (#254), D221 (#120, [python-topology.md](python-topology.md)), D223 (#132, [python-checkpoints.md](python-checkpoints.md)), D228 (#133, [python-package.md](python-package.md)) |

M2 covers a documented classical subset; conda follows later. The design
follows a reading of OpenMM's Python layer
([python.md](python.md#a-reading-of-openmms-python-layer)): both front ends,
the control file and Python, produce the same IR and share one validation.

| Item | Design |
|---|---|
| Loading | `mdir.load_amber`, `mdir.load_gromacs`: topology, parameters, positions, cell; nothing about the run. Later an import of an OpenMM `System` for the supported subset |
| Topology | `loaded.topology`, `system.topology`, `program.topology`: read-only copies, `select(mask)`, `to_openmm()` (done, D221) |
| Physics apart from execution | `System`, the integrator and the ensemble, and `Execution` as typed objects, not strings |
| An explicit compile | `mdir.compile(...)` returns an immutable program and its plan; a change afterwards marks it stale. Tunable parameters do not recompile |
| Runs and reporters | `sim.run(n)` runs segments to the next report of any reporter, with the writers of the driver in C++ and the GIL released |
| State | `state()` (host copies in the order of the input), `view()` (DLPack, leases, P16), and `borrow()` with `commit()` (D229) |
| Errors | Typed: input, compile, unsupported, simulation |
| Checkpoints | The H5MD checkpoint of the driver, with the hashes of the model and the plan; one format |

Future MLIP work integrates PyTorch/JAX automatic differentiation
(maintainer, 2026-10-04): evaluation and parameters stay apart from the
ownership of a simulation, and framework adapters need explicit derivative
rules and validated derivatives. This does not claim training or
trajectory differentiation in the initial M3 inference deliverable.

### 6.1 Differentiable simulation (M2b)

Fitting the parameters of a potential to measured averages, as Differentiable
Trajectory Reweighting does [[ThalerZavadlav2021]](references.md#thalerzavadlav2021),
needs no derivative through the trajectory. Frames $S_i$ sampled at
$\hat\theta$ are reweighted to $\theta$ by
$w_i = \mathrm{softmax}_i[-\beta(U_\theta(S_i) - U_{\hat\theta}(S_i))]$,
and $\langle O\rangle_\theta = \sum_i w_i O_i$. Automatic differentiation of
that expression in the framework gives the gradient,
$\langle\partial O/\partial\theta\rangle - \beta(\langle O\,\partial U/\partial\theta\rangle - \langle O\rangle\langle\partial U/\partial\theta\rangle)$,
so MDIR need provide only two things: a sampler, which is not
differentiated, and an evaluator of $U_\theta$, the virial, and observables
at stored frames, with a derivative rule in $\theta$.

| Item | State and next |
|---|---|
| Tunable parameters | Done in M2a (D213, D226); their derivatives in M2b (D230, #203, [python-gradient.md](python-gradient.md); the tails' derivatives D209; the charges under PME, D231): the host applies the chain rule, not an IR of the derived quantities (maintainer, PR #205) |
| Frame evaluator | Done ([python-frames.md](python-frames.md)): D240 (#249, the multi-point evaluation of #87, with D161 and D230), D238, D239, D243 (#264), D251 (#291), D252 (#313); its starts made cheap by D253 (#294), D255 (#316), D254 (#324), D256 (#328), D257 (#295), D258 (#326) ([python-segments.md](python-segments.md), [device-tuples.md](device-tuples.md)). Next: the metatomic shape of #138, and force matching (stage 4 of #138), refused until then |
| Frames on disk | Done (D239, #251, [python-h5md.md](python-h5md.md)); XTC frames for reweighting: #264 |
| Adapters | PyTorch done: `mdir.torch.evaluate`, a `torch.library.custom_op` with `register_autograd` (maintainer, PR #250; first written here as `torch.autograd.Function`). Next: JAX (`jax.custom_vjp` over `jax.ffi`), one framework-neutral contract with DLPack buffers (maintainer, 2026-10-05) |
| Reweighting | Next, in the framework, with the effective sample size $\exp(-\sum_i w_i\ln w_i)$ deciding when to sample again; an example fits the radial distribution function and the density of water |
| Further targets | Free energies through `[free_energy]` ($\partial\Delta G/\partial\theta$ from $\langle\partial U/\partial\theta\rangle$ or MBAR weights), the pressure through the derivative of the virial, the relative entropy of coarse-grained models |

Learned potentials (M3) keep their parameters in the framework: MDIR
samples, and the framework model evaluates $U_\theta$ at the frames. M2b
depends on M2a's segments, device views, and tunable parameters (D213),
and on buffers that stay on the device (D215).

## 7. Learned potentials (M3) and distributed execution (M4)

D169 splits this section between two milestones, as D287 (the rulings on
#377 of 2026-10-16 and 2026-10-17) sets them: ML1 steps 1 to 3, the IR of
ML4, PaiNN in stages, and MACE in stages close M3 (MACE is required); ML2,
ML3, PaiNN and MACE on two GPUs, ML5, and distributed particle mesh Ewald
close M4. A milestone orders completion, not the start of work: ML2 runs
in parallel with M3. The direction is D166 and D167 (2026-10-03);
[the architecture plan](future-architecture-plan.md#adopted-scope-and-first-deliverable)
holds the contracts, the sequence (Allegro, then PaiNN, then MACE), the
scope of the first deliverable (one local MLIP on one GPU, then on two GPUs
sharing one system), and the [layers and external AD](future-architecture-plan.md#external-ad-and-execution-layers).
Classical MD keeps its features, validation, and baselines, with no MLIP
backend required.

| Milestone | Work | Completion evidence |
|---|---|---|
| ML1 | One compatible Allegro artifact through the optional metatomic adapter and versioned potential/neighbor contract; retain external AD and tensor backend | On one GPU, energy, per-particle forces, and stress/virial agree with the original backend under declared tolerances; coordinate/cell finite differences and short NVE run; record artifact/backend identity, units, species, precision, copies, and synchronization; unsupported outputs rejected; classical builds still work without the adapter |
| ML2 | Dependency graph, ownership and periodic images, coverage and freshness; fixed two-domain LJ, then EAM and reverse contribution routing; CPU ranks before GPUs | Energy, forces, and virial agree with one domain; migration, rebuilds, boundaries, uneven/empty domains, and exactly-once accumulation tested; inspectable schedules and negative tests for missing transfers, stale intermediates, and invalid completion |
| ML3 | The ML1 local model on two GPUs using the ML2 ownership and transfer contract | Same artifact, weights, precision, and outputs as ML1; one/two-GPU agreement against the original backend including migration and short-run conservation; peak memory and full-step time recorded, without requiring a speedup; completes the first deliverable |
| ML4 | PaiNN with an opaque path and declared stages/differentiable communication, then MACE using the same contracts; external AD generates backward; semantic import is optional | Communication adjoint checks and same-model comparison of enlarged coordinate halos and per-layer feature exchange, including reverse derivatives and saved-value lifetimes; energy/force/virial agreement and measurements of bytes, messages, redundant work, memory, and full-step time |
| ML5 | Bounded legal-plan selection and measured overlap | Fixed-plan baselines and tuning cost reported; one/two/four-GPU strong and weak scaling; multiple nodes before a cluster claim; checked preconditions and an off switch for each optimization |

| Track | State and next |
|---|---|
| ML1 | Done: the TorchScript backend with chunks (D270, [metatomic-ml1.md](metatomic-ml1.md#9-what-the-first-pr-built-d270)), minimization and constant pressure (D272), the AOTInductor backend (D273), the pair list from MDIR's structures (D277). Next: the CPU target |
| ML4 | In progress: the baselines (D286, [ml4-baselines.md](ml4-baselines.md)), the IR (D288, D289, [ml4-ir.md](ml4-ir.md)), PaiNN in stages, step 4a (D294, D293), and step 4b's fusion, cuBLAS products, row sums, short reductions, and f64 fields (D299, D302, D303, D318, D329). Next: AOTInductor bodies, store against recompute, and the two-ended scatter (#462); MACE in stages (#377) |
| ML2 | In progress ([distributed-ml2.md](distributed-ml2.md)): the dialect and the in-process oracle (D275), the program from the global potential (D276), EAM (D279), loops through `md_exec` (D290), the factor grid (D307, #476), the schedule proof (D308, #483), halo events (D310, #490), migration primitives (D313, #494), center partitions and filters (D317, #496, #535, #499), and the collective skin predicate (D332, #542). Next: the witness in candidate-map reuse, the live state at retirement, and the first `[distribute]` run (#38); interior work before the wait (#536); asynchronous transport (#481); MPI on CPU ranks, several GPUs, then the classical track of #31 |
| Distributed PME (#452) | In progress: the execution-level transform (D306, #480), the semantic transform (D319, #502), and its storage (D341, #526). Next: its use by `md-differentiate` in the reciprocal chain, the spread/influence/gather decomposition, and the transfer of grids (#452) |
| ML3, ML5 | Planned. A four-GPU measurement follows ML3's gate. Distributed PME, global attention or charge solvers, constraints across domains, training, arbitrary PyTorch/JAX import, and trajectory differentiation remain outside ML5 |
| Companion work | A persistent kernel cache and runnable compiled artifacts, with keys of the model, the backend, and the plan choices that change code; same-plan continuation first |

To bring into line with the detailed plan: [decisions.md](decisions.md),
Section 7 still tags the dependency interface M2b, which A14 moved into M1.
(The first design's description of the plan as an ordered list of stages,
and of `mlff`'s op families, was removed from
[architecture.md](architecture.md) by #587.)

## 8. Analysis

Simple observables run at the speed of reading a file; what went wrong in
the comparisons of this work was never the arithmetic, but images,
frame indices, column orders, and definitions
([validation.md](validation.md#analysis-tools-on-mdbench)).

| # | Item | State |
|---|---|---|
| A1 | Frames correct by construction: molecules whole and the solute in one image, from the graph of bonds and constraints that MDIR holds; the step, the time, and the cell in each frame; the count of images of each molecule for an exact unwrapping (mean square displacements); velocities on request (F4) | Missing; with U2 |
| A2 | Observables of the run, compiled into the program and sharing its loops: radial distributions from the pair list, density profiles along an axis, the pressure tensor and the lateral pressure profile, order parameters, the interaction energy between two selections (protein and RNA), the distribution of the kinetic energy for the tests of [[Merz2018]](references.md#merz2018); summed in `f64`, with block averages and their errors | Missing. The interaction energy and the lateral pressure need the forces of the run, which a program of analysis has only through a rerun |
| A3 | `mdir analyze`: the same kernels over frames read from files: the energies of frames term by term (a rerun), observables of pairs on the device, the matrix of RMSDs between frames by QCP [[Theobald2005]](references.md#theobald2005) on the device | Missing |
| A4 | Exchange rather than reimplementation: frames to MDTraj, MDAnalysis, and PyTorch through the DLPack views of the Python API (Section 6); each observable of A2 and A3 validated against cpptraj and GROMACS as the terms are, its definition written down | Missing |

Not taken: reimplementing the actions of cpptraj. A device pays where pairs
are involved, where all pairs of frames are (clustering), or where the
frames never reach a file.

The first runtime-interface slice of #558 stage 2 is #623: a shared, versioned neighbor-group slot contract, with descriptor-driven extents and range checks (D[groups-buffer-abi]).
