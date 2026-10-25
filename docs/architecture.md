# MDIR Architecture: the first design, and where it went

Status: superseded on 2026-10-25 (#587). The current picture of MDIR, what
is built and what is planned, is [overview.md](overview.md). This page
replaces revision 5 of the first design (2026-09-29), which described the
plan of milestone M0. The design's choices are rows of
[decisions.md](decisions.md) (the C, D, P, A, B, and S series), which keep
their text; this page says, for each section of that revision, where its
content is now, and what in it no longer holds. Revision 5 is in the
history of this file.

## Where each section went

| Section of revision 5 | Now |
|---|---|
| Status | [overview.md](overview.md), Sections 1 and 2.2: `md`, `dyn`, `md_exec`, `mdrt`, and `md_dist` are built; `md_dist` is not yet taken by a run |
| 1. Goal, and the table of components | [overview.md](overview.md), Section 1, and the table of dialects in Section 2.2 |
| 2. A graph and a pipeline | The pipeline: [overview.md](overview.md), Section 2.1. `md_dist` and `md_exec` as peers (A8): [distributed-ml2.md](distributed-ml2.md) |
| 3. Compilation model | [overview.md](overview.md), Section 2.4 (what is fixed at compilation and what is an argument), from P1, D227, and [driver.md](driver.md), Section 2.5; the cache of compiled code: [compile-cache.md](compile-cache.md) |
| 4.1 State | D3, S1, S2, P17, D15; [ops.md](ops.md), Sections 2.3 and 2.4 |
| 4.2 `md` | [ops.md](ops.md); relations and orientation A2, exchange contracts B3, truncation B4, units S4, exclusions D14 and P8, [tuples.md](tuples.md) |
| 4.3 `mlff` | Not built; learned potentials are neural stages in `md` (D288, [ml4-ir.md](ml4-ir.md)) and the plugin of ML1 ([metatomic-ml1.md](metatomic-ml1.md)) |
| 4.4 `dyn` | [overview.md](overview.md), Section 2.2; requirements and capabilities D6; counter-based random numbers P5, A5, A13, and [dynamics.md](dynamics.md); couplings as ops of `dyn`, designed: D326 |
| 4.5 `ensemble` | Not built; the ensemble is the `[ensemble]`, `[thermostat]`, and `[barostat]` tables of the control file and the corresponding objects of the Python package |
| 4.6 Step loop | [driver.md](driver.md), Section 2.2 (the schedule of a run is compiled as one function); the segments of a Python simulation: [python-segments.md](python-segments.md) |
| 5. Semantic differentiation | D2, D182, D183, D295; [primitives-map.md](primitives-map.md), Section 1.3; [white paper, Section 3.3](paper/03-design.md#33-differentiation-and-exchange-contracts) |
| 6. Particle dependency interface | Not built (D4, A3); the staged dependencies are planned in [future-architecture-plan.md](future-architecture-plan.md) and [distributed-ml2.md](distributed-ml2.md) |
| 7. Joint planner | Not built (D9, A4). Its parts today: the options of the passes, the skins chosen by measurement (D281), and the chooser passes with stated preconditions (D285, #444); the planned record of choices and reasons is #433 |
| 7.2 Pair execution policy | D11, P13; on a device, groups compute each pair once (D89) |
| 7.3 Reproducibility | D84, D171, D301; [overview.md](overview.md), Section 4 |
| 7.4 Precision | D23, S8, D31, D32; [white paper, Section 7](paper/07-precision.md) |
| 8.1 `md_dist` | [distributed-ml2.md](distributed-ml2.md) (D275), which replaced the sketch of revision 5 |
| 8.2, 8.3 `md_exec` | [ops.md](ops.md), [neighbors.md](neighbors.md), and [white paper, Section 3.4](paper/03-design.md#34-from-sums-to-loops-and-the-passes-over-loops); neighbor validity P15, B1, A11, D80, D88; value and storage forms D17, D18, B10, D337 |
| 8.4 Dependency token | `!mdrt.event` and the halo events of D310 ([runtime.md](runtime.md)); the order of a block carries dependencies (A12); copies ordered by tokens are planned (#567, [execution-contracts.md](execution-contracts.md), Section 11) |
| 9. Runtime | [overview.md](overview.md), Section 2.5; the boundary between compiler and runtime, D8 and P9 ([decisions.md](decisions.md)); [runtime.md](runtime.md) |
| 10. Lowering to upstream MLIR | [overview.md](overview.md), Section 2.4 |
| 11. Input and output | [driver.md](driver.md), Sections 1, 2.6, and 2.8; D24, D26, D40, D44, D141, D239 |
| 12. Roadmap | [roadmap.md](roadmap.md); milestones D169 and D195 |

## What no longer holds

These statements of revision 5 were not true of `main` on 2026-10-25 and
were not carried over:

- The status table listed `md_dist` and the events of `mdrt` as not
  implemented. Both exist (D275 to D332; D310), though no run takes
  `md_dist` yet.
- `dyn` was to have `dyn.thermostat`, `dyn.barostat`,
  `dyn.constraint_position`, `dyn.constraint_velocity`, `dyn.random`, and
  `dyn.metropolis`. It has `dyn.program`, `dyn.step`, `dyn.kick`, and
  `dyn.drift`; constraints and noise are written by the builder in `md`
  ops, and couplings as ops are the design of D326.
- `md_exec` was to have `pack`, `unpack`, `launch`, and `combine`. It has
  none of them; it has `tuple_for`, `refresh_neighbors`, `reciprocal`,
  `linear_transform`, `scatter_add`, and the other ops of
  [ops.md](ops.md).
- `mlff`, `ensemble`, the joint planner with `ExecutionPlan` and
  `ExecutionTuningState`, and `ParticleDependencyInterface` were presented
  as components. None is built.
- The lowering table named AMD GPUs (`amdgpu`, `rocdl`), MPI, and NVSHMEM
  as targets. Only the CPU and NVIDIA GPUs are targets, and `md_dist` runs
  in one process.
- The trajectory was "DCD first, then XTC". DCD, XTC, and H5MD are all
  written (D141, D239).
- The milestone table predated M1's completion and the split of M2
  (D195); [roadmap.md](roadmap.md) holds the current one.
