# MDIR Decision Log

Last updated: 2026-10-03.

This log records what has been decided. Every entry is accepted unless it
is marked otherwise.
[overview.md](overview.md) describes the design that results, and
[architecture.md](architecture.md) maps the first design to it.
[design-review.md](design-review.md) and [prior-art.md](prior-art.md) hold
the analysis that led to the decisions.

Identifiers are stable. C*n* are project constraints. D*n* and P*n* are
decisions; the P series was first recorded as proposals and accepted on
2026-09-29, and keeps its numbering so that existing references stay valid.
A D number is given when the change that makes the decision is merged,
in the order of the merges; until then the branch writes a label,
`D[<label>]` (CONTRIBUTING.md).

An entry is not rewritten when a later entry changes it. It says
"Amended by" and names the later entry, which holds where the two differ.

From D20 on, each decision is a file of its own, `decisions/Dnnn.md`,
which holds its full text; the table of Section 5 gives its statement, the
bold sentence that begins it, and the date on which it was recorded
(#602). A branch adds `decisions/<label>.md` for its decision and does not
edit the table, which `scripts/decisions-table.py` writes from the files
([workflow.md](workflow.md#decision-numbers)).

## 1. Project constraints

| # | Constraint |
|---|---|
| C1 | MDIR targets general-purpose MD. Both all-atom (AA) and coarse-grained (CG) simulation must be supported. |
| C2 | The same stack runs on workstations and on HPC clusters. |
| C3 | Compilation is just-in-time: the program is compiled before the run starts. How much is compiled ahead of time is negotiable. |
| C4 | CPU and GPU are both first-class targets from the first milestone. |
| C5 | There are two entry points: a standalone front end that reads a declarative input format, and a Python library in the style of OpenMM (objects are constructed and composed through an API). Both produce the same `md` IR. |
| C6 | Determinism is a separate, opt-in execution mode, not the default. |
| C7 | The v0 performance target is homogeneous systems at finite density. Sparse and strongly inhomogeneous systems are not a v0 performance goal. |

## 2. Decisions, D series

### D1. Development order differs from lowering order

Development order:

```text
1. md
2. md_exec, with the CPU back end
3. minimal dyn
4. GPU back end
5. md_dist
6. mlff
7. ensemble
```

Lowering order in the finished compiler remains
`semantic IR → joint planning → md_dist → md_exec`. On a single rank
`md_dist` is the identity, so the pipeline is `md → md_exec`.

The GPU back end precedes `md_dist` because C4 puts both CPU and GPU in the
first milestone, and because GPU constraints shape `md_exec` more than
distribution does.

*Amended by D166:* an optional opaque MLIP adapter may precede `md_dist`;
the distributed semantic MLIP path still follows the ownership and
communication foundations. The lowering order is unchanged.

### D2. Differentiation is a semantic-level transformation

Forces, virial, and $dH/d\lambda$ are produced by an `md`-to-`md` transformation
that runs before locality analysis and distributed planning. Its output is
again expressed in `md` and `mlff` ops.

- Closed-form pair and bonded terms: symbolic differentiation.
- `mlff`: reverse-mode, generating an explicit derivative graph.
- LLVM-level AD (Enzyme [[Moses2020]](references.md#moses2020)) is not the default path. It remains an option for
  opaque external kernels and back-end experiments.

### D3. State has value semantics

Semantic-level ops take and return state as SSA values:

```mlir
%s1 = dyn.kick  %s0, %f0, %half_dt
%s2 = dyn.drift %s1, %dt
%f1 = md.evaluate @H(%s2) request [#md.forces]
%s3 = dyn.kick  %s2, %f1, %half_dt
```

Value semantics does not imply copying. In-place updates are chosen later,
in the manner of bufferization.

### D4. Locality is an op interface with stages

Interaction ops implement `LocalityInterface`, which returns an ordered list
of stages. Each stage reports:

| Field | Meaning |
|---|---|
| `support` | Spatial or topological extent the stage reads from. |
| `reads` | Per-particle fields consumed. |
| `writes` | Per-particle fields produced. |
| `accumulation` | Whether and how contributions to non-owned particles are returned to their owners. |
| `freshness_requirement` | How current the inputs must be when the stage runs. |

The distribution planner consumes only this interface. It compares an
expanded halo against per-stage communication with a cost model.

*Amended by A3.*

### D5. The semantic dialects form an object graph, not a lowering chain

`md`, `mlff`, `dyn`, and `ensemble` are peers that reference each other.
Documentation distinguishes the **semantic dialect graph** from the
**lowering pipeline**.

### D6. `dyn` programs declare requirements and capabilities

A `dyn` program does not claim to sample a named ensemble. It declares what
it requires (for example a temperature) and what it provides (for example
thermostatting, or approximate energy conservation). A protocol verifier
detects inconsistent combinations, such as a canonical ensemble requested
with no thermostat or stochastic sampler.

Reason: whether an integrator samples a given ensemble exactly is a numerical
property too subtle to encode as a type-system fact.

*Amended by B9.*

### D7. The step loop gets a driver, not a dialect

v0 provides a single driver construct (`sim.run`) or a C++ runtime driver.
A `sim` dialect is created only if the need is demonstrated. See P4.

### D8. Compiler and runtime boundary

| Runtime owns | Compiler owns |
|---|---|
| Memory and storage lifetime | Neighbor algorithm selection |
| Neighbor-list objects | Bin and cluster parameters |
| Cell buffers | Build and prune kernels |
| Migration buffers | Pair traversal |
| Communication resources | Fusion |
| MPI and NVSHMEM contexts | Schedule |

`md_exec.build_neighbors` lowers to a runtime storage allocation plus a
generated build kernel. The neighbor algorithm is not hidden in an opaque
runtime.

### D9. A joint planner produces an `ExecutionPlan`

Distribution and execution decisions are made together, because parameters
such as the skin affect both. The planner is a compiler component, not a
dialect. Example plan contents: partition, skin, neighbor strategy,
Newton's-third-law usage, force accumulation strategy, halo mode.

*Amended by A4.*

### D10. One dependency token

Ordering constraints are expressed with a single project-owned SSA token
type, `!mdrt.event`. It lowers to an MPI request, a CUDA event, or an NVSHMEM
signal depending on the back end. `md_exec.task` and `md_exec.depend` are
dropped.

*Amended by A9 and A12.*

### D11. Pair execution policy is decided in `md_exec`

The semantic level says only that a pair contributes forces to both
particles. The plan carries a policy per pair computation:

```text
PairExecutionPolicy {
    traversal
    newton3
    conflict_strategy
    reduction_order
}
```

### D12. Mesh electrostatics are excluded from v0

v0 supports Lennard-Jones and cutoff-based electrostatics. The architecture
states that distributed fields and grids may exist alongside particle
domains. The upstream `shard` dialect is a reference for that part.

### D13. Project-owned runtime ABI is the default lowering target

Communication and scheduling lower to a project-owned runtime ABI. The
upstream `mpi` and `async` dialects are optional targets.

### D14. Exclusions and reproducibility are major design items

- `md.neighborhood` needs a way to combine a geometric relation with an
  exclusion relation and a special-pair relation.
- Execution contracts for reproducibility are decided early. See P6.

### D15. Per-particle fields are a custom type

At the semantic level, per-particle fields and the simulation state are
custom types, not builtin tensors. Owned/ghost structure and packed layouts
do not fit the tensor model. In-place analysis is project-owned as a
consequence; upstream bufferization is not reused for these types.

### D16. Energy expressions are strings

The Python library accepts energy expressions as strings, as OpenMM does.
The declarative input format uses the same expression syntax, so one
expression parser serves both entry points.

### D17. Value semantics ends inside `md_exec`

`md_dist` and the upper part of `md_exec` operate on field values. A halo
exchange returns a new version of a field whose ghost region is current; it
does not overwrite a buffer. A pass inside `md_exec` then assigns storage:
it decides in-place updates and the data layout, and converts ops to their
storage form.

```mlir
%x_h   = md_dist.forward_halo %x
%f_int = md_exec.pair_for interior(%x),   %nl { ... }
%f_bnd = md_exec.pair_for boundary(%x_h), %nl { ... }
%f     = md_exec.combine %f_int, %f_bnd
```

`md_exec` ops exist in one set with two forms, value and storage, in the
manner of upstream `linalg`. The destination of a result is passed as an
operand. Access modes map onto that shape:

| Access | Form |
|---|---|
| Read | Input operand |
| Accumulate into existing | Destination operand carrying the old value, plus result |
| Accumulate from zero | Destination operand initialized to zero, plus result |

Consequences:

- Halo placement and redundant-exchange elimination are ordinary SSA
  dataflow and common-subexpression elimination.
- In value form, `!mdrt.event` is needed only for ordering constraints that
  have no data dependency. After storage assignment, the dependencies that
  field values carried are carried by event tokens.

*Amended by A12 and D33.*

### D18. No implicit copies in the step loop

The storage assignment pass must not insert a copy of a per-particle field
inside the step loop silently. When a copy is required, the compiler reports
it with the reason. A copy is legitimate only when one version of a field has
more than one consumer, as in a Metropolis rejection.

*Amended by B10.*

### D19. Shared types: `md` and a new `mdrt` dialect

| Types | Dialect |
|---|---|
| State, fields, relations, neighborhoods, quantities with units | `md` |
| Event token, domains and partitions, storage, physical neighbor structures | `mdrt` |

`mdrt` is the IR representation of the runtime ABI (D13). Runtime calls are
`mdrt` ops with verified operands, lowered to function calls in the last
stage. `md_dist` and `md_exec` both depend on `mdrt`; neither depends on the
other.

## 3. Decisions, P series

| # | Decision | Precedent |
|---|---|---|
| P1 | **JIT binding times.** Structure is a compile-time constant: terms, functional forms, precision, target. Particle count, box, time step, temperature, and $\lambda$ are run-time values. Force-field parameter tables may be made constant per parameter, opt-in. Compile once, distribute to all ranks, and cache by a hash of IR and target. | OpenMM kernel cache |
| P2 | **Full-shell halo first.** The communication transport is abstracted, with in-process, MPI, and NVSHMEM implementations, so that one `md_dist` serves workstations and clusters. | LAMMPS; GROMACS thread-MPI |
| P3 | **Milestones.** M0: Lennard-Jones fluid, NVE. M1: Martini CG membrane and water. M2: AA protein and water with PME and constraints. M3: MLFF. *Amended by A10.* | — |
| P4 | **Compiled segments.** The JIT compiles `run_segment(state, n) -> state`, which contains the step loop, rebuild checks, and migration. The driver owns events between segments: output, checkpoints, replica exchange, Python callbacks. | OpenMM `step(n)`; HOOMD-blue `run(n)` |
| P5 | **Counter-based random numbers.** `dyn.random` is a pure function of seed, step, particle global ID, and stream ID. No generator state is carried in the simulation state. *Amended by A5 and A13.* | HOOMD-blue (Random123 [[Salmon2011]](references.md#salmon2011)) |
| P6 | **Reproducibility levels.** *Fast*: no guarantee. *Deterministic*: same binary, hardware, and decomposition give the same bits. *Decomposition-independent*: the same bits for any rank count. Bitwise agreement between different hardware is a non-goal. *Amended by A6.* | GROMACS `-reprod`; OpenMM deterministic forces; Desmond and Anton fixed-point accumulation |
| P7 | **Plan in the IR.** The `ExecutionPlan` is serializable, user-overridable, and attached to the IR, so each lowering can be tested with a fixed plan. Structural parameters (cluster size, conflict strategy) require recompilation; numeric parameters (skin, rebuild interval, domain boundaries) are run-time values that can be tuned during the run. *Amended by A4.* | GROMACS run-time tuning of list interval and load balance |
| P8 | **Exclusions as a first-class relation.** Nonbonded pairs are "within cutoff, minus excluded." Scaled 1-4 pairs are a separate topological relation with their own parameters. Terms may also sum over the exclusion relation, which Ewald exclusion corrections require. | OpenMM exceptions; GROMACS pair interactions |
| P9 | **Runtime primitives versus generated predicates.** Refinement of D8: generic parallel primitives (sort, scan, compaction) live in the runtime. MD-specific predicates and kernels (distance tests, exclusion filters, type-pair cutoffs) are generated. | — |
| P10 | **Trusted sampling metadata.** Refinement of D6: library-provided thermostat and barostat kinds carry metadata stating whether they preserve the target distribution. The verifier warns, for example, when replica exchange is combined with a thermostat that does not. | — |

P11 to P18 follow from the review of PPMD (Saunders et al. 2018 [[Saunders2018]](references.md#saunders2018)). See
[prior-art.md](prior-art.md). D17 makes P11 concrete.

| # | Decision |
|---|---|
| P11 | **`md_exec` adopts the loop and access model of PPMD.** Core ops are a particle loop, a pair loop, and a reduction. With value semantics, access modes are the op signature: a field that is read is an operand, a field that is produced is a result, and a field that is updated is both. Explicit access descriptors are declared only for external, opaque kernels. For generated kernels they are derived and verified. |
| P12 | **Layer boundary wording.** `md`: what is computed. `md_exec`: over which set, with which access pattern. Back end: how. |
| P13 | **Baseline pair execution.** Pair terms are executed over directed pairs, writing only to the central particle. This removes scatter races, atomics, and reverse accumulation from the first back ends. Half-list execution is added later as a planner optimization. Scope: two-body terms only. Bonded terms (M1) require a scatter strategy and MLFF (M3) requires reverse accumulation, so both remain in the architecture. | *Amended by D49, which has the strategy for bonded terms.*
| P14 | **`md` has a generic relational core.** Particles, fields, relations, neighborhoods, and reductions are generic. A Hamiltonian is one kind of region built from them; analyses and collective variables are others. Whether a region can be differentiated depends on the ops it contains. |
| P15 | **Neighbor structure validity is explicit.** A neighbor structure is a value built from a reference configuration, with a validity condition. The default rebuild policy is a fixed interval with a buffer sized for that interval, which needs no per-step global reduction. A displacement check is optional. *Amended by A11 and B1: the default policy checks validity in every step.* |
| P16 | **Version counters at the segment boundary.** Inside a compiled segment, halo exchange placement is decided statically. State modified from the host between segments is detected with run-time version counters. |
| P17 | **Particle identity is separate from storage index.** Topology, exclusions, and random number streams refer to global IDs. Particles are spatially reordered when neighbor structures are rebuilt. *Amended by D44: the particles are put in order where a run and where a segment begins.* |
| P18 | **The cutoff predicate is explicit on the pair loop.** It is not buried in the kernel body, so the lowering can choose between a branch and a mask. |

## 4. Resolved questions

| Question | Resolved by |
|---|---|
| Q1. Position of the GPU back end | D1 |
| Q2. Form of the standalone front end | C5 |
| Q3. Field type representation | D15 |
| Q4. Location of shared types | D19 |
| Q5. Form of energy expressions | D16 |
| Q6. Where value semantics ends | D17 |

## 5. Decisions made with the M0 specification

From D20 on, in the order of their numbers; the decisions made after the M0
specification follow those made with it.

<!-- decisions-table: written by scripts/decisions-table.py from docs/decisions/ -->
| # | Decision | Recorded |
|---|---|---|
| [D20](decisions/D20.md) | LLVM 23.1.2 is the pinned release. | 2026-09-29 |
| [D21](decisions/D21.md) | M0 integrators are velocity Verlet [[Swope1982]](references.md#swope1982) and leapfrog [[HockneyEastwood1988]](references.md#hockneyeastwood1988). | 2026-09-29 |
| [D22](decisions/D22.md) | Energy expressions use the syntax of OpenMM custom forces [[Eastman2017]](references.md#eastman2017). | 2026-09-29 |
| [D23](decisions/D23.md) | Three precision modes are supported: single, mixed, and double. | 2026-09-29 |
| [D24](decisions/D24.md) | The input format is TOML. | 2026-09-29 |
| [D25](decisions/D25.md) | Trajectories are written as XTC. | 2026-09-29 |
| [D26](decisions/D26.md) | Checkpoints are written as H5MD [[deBuyl2014]](references.md#debuyl2014), in 64-bit floating point. | 2026-09-29 |
| [D27](decisions/D27.md) | D8 applies from M0. | 2026-09-29 |
| [D28](decisions/D28.md) | CPU threading uses OpenMP in M0. | 2026-09-29 |
| [D29](decisions/D29.md) | The neighbor structure of M0 is the neighbor matrix. | 2026-09-29 |
| [D30](decisions/D30.md) | Vector fields are stored as `memref<?x3xT>` first. | 2026-09-29 |
| [D31](decisions/D31.md) | Precision is assigned by a pass on `md_exec`. | 2026-09-29 |
| [D32](decisions/D32.md) | Buffers state the type that the state is stored in. | 2026-09-29 |
| [D33](decisions/D33.md) | The storage handle of a field is a `memref`. | 2026-09-29 |
| [D34](decisions/D34.md) | GPU execution uses the upstream `gpu` dialect, with the runtime functions in `libmdrt_cuda`. | 2026-09-29 |
| [D35](decisions/D35.md) | The control file has one table for each concern of a run | 2026-09-29 |
| [D36](decisions/D36.md) | The control file is written in Å, kcal/mol, ps, amu, K, and atm. | 2026-09-29 |
| [D37](decisions/D37.md) | Trajectories are written as DCD first. | 2026-09-29 |
| [D38](decisions/D38.md) | The control file is parsed with toml++ | 2026-09-29 |
| [D39](decisions/D39.md) | The schedule of a run is compiled. | 2026-09-29 |
| [D40](decisions/D40.md) | A checkpoint holds the state as the next step needs it | 2026-09-29 |
| [D41](decisions/D41.md) | The test of validity of a neighbor structure is made where the positions are written. | 2026-09-29 |
| [D42](decisions/D42.md) | The minimum image [[AllenTildesley2017]](references.md#allentildesley2017) is found with a multiplication | 2026-09-29 |
| [D43](decisions/D43.md) | A neighbor build searches a copy of the positions that is made for it | 2026-09-29 |
| [D44](decisions/D44.md) | The particles are put in the order of their positions where a run begins and where a segment begins | 2026-09-29 |
| [D45](decisions/D45.md) | The log takes the temperature and the pressure from the kinetic energies at the half steps as well | 2026-09-29 |
| [D46](decisions/D46.md) | The target of M1 is a bilayer of DPPC in water with Martini 2.2 | 2026-09-29 |
| [D47](decisions/D47.md) | The tuples of a topology are a set of entities | 2026-09-29 |
| [D48](decisions/D48.md) | A kernel over a relation of a topology takes internal coordinates | 2026-09-29 |
| [D49](decisions/D49.md) | A term over tuples is executed by the particles | 2026-09-29 |
| [D50](decisions/D50.md) | The thermostat of M1 is stochastic velocity rescaling | 2026-09-29 |
| [D51](decisions/D51.md) | A neighbor structure is built after every change of the cell | 2026-09-29 |
| [D52](decisions/D52.md) | A topology is read in the format of GROMACS | 2026-09-29 |
| [D53](decisions/D53.md) | The target of M1 is all-atom | 2026-09-29 |
| [D54](decisions/D54.md) | M1 has an intermediate stage | 2026-09-29 |
| [D55](decisions/D55.md) | Particle mesh Ewald on one device or one process is in M1. | 2026-09-29 |
| [D56](decisions/D56.md) | Constraints are in M1 | 2026-09-29 |
| [D57](decisions/D57.md) | Lennard-Jones takes its parameters from a table for each pair of types | 2026-09-29 |
| [D58](decisions/D58.md) | Topologies are read in the formats of Amber (`prmtop`, `inpcrd`) and of GROMACS (`.top`, `.itp`, `.gro`) | 2026-09-29 |
| [D59](decisions/D59.md) | MDIR has one program with subcommands | 2026-09-29 |
| [D60](decisions/D60.md) | The linear momentum is removed every `center_of_mass_interval` steps | 2026-09-29 |
| [D61](decisions/D61.md) | The members of a relation of a topology are the places of the particles in their fields | 2026-09-29 |
| [D62](decisions/D62.md) | MDIR has one form for each term of the potential and one meaning for each parameter | 2026-09-29 |
| [D63](decisions/D63.md) | The molecules that SETTLE constrains can be chosen. | 2026-09-29 |
| [D64](decisions/D64.md) | MDIR stays under the MIT license. | 2026-09-30 |
| [D65](decisions/D65.md) | The target of M1 is ff19SB with OPC | 2026-09-30 |
| [D66](decisions/D66.md) | A step runs on the device without waiting for the host | 2026-09-30 |
| [D67](decisions/D67.md) | The velocities are coupled at the end of a step, every `interval` steps | 2026-09-30 |
| [D68](decisions/D68.md) | A virtual site is a particle of mass 0 in the particle set | 2026-09-30 |
| [D69](decisions/D69.md) | The reciprocal sum of particle mesh Ewald is one op, `md.reciprocal` | 2026-09-30 |
| [D70](decisions/D70.md) | The grid of particle mesh Ewald is accumulated in fixed point | 2026-09-30 |
| [D71](decisions/D71.md) | $\beta$, the grid, and the order of particle mesh Ewald can each be given | 2026-09-30 |
| [D72](decisions/D72.md) | A barostat changes the cell at the end of a period of coupling | 2026-09-30 |
| [D73](decisions/D73.md) | Minimization is steepest descent in the metric of the masses, on the surface of the constraints | 2026-09-30 |
| [D74](decisions/D74.md) | Restraints are applied after the evaluation of the potential, not in it | 2026-09-30 |
| [D75](decisions/D75.md) | In a loop over pairs or tuples, only the difference of two positions is taken in their type; the minimum image, the squared distance, and the test of the cutoff are in the type of the kernel. | 2026-09-30 |
| [D76](decisions/D76.md) | Leapfrog takes the steps of velocity Verlet and differs in what it stores | 2026-09-30 |
| [D77](decisions/D77.md) | The barostat counts the work of a scaling exactly, from the energy of the scaled positions, and the next step takes their forces. | 2026-09-30 |
| [D78](decisions/D78.md) | On a device, the reciprocal sum of PME computes in the type of the forces | 2026-09-30 |
| [D79](decisions/D79.md) | The terms of the potential take the positions in the type of the kernel, converted once per evaluation; the constraints keep theirs. | 2026-09-30 |
| [D80](decisions/D80.md) | A neighbor structure stays valid when a barostat scales the cell; structures that differ only in their cells share one storage. | 2026-09-30 |
| [D81](decisions/D81.md) | The reciprocal sum of PME runs on a second stream | 2026-09-30 |
| [D82](decisions/D82.md) | On a device, the loops over pairs get a tile structure: the cluster pair list of GROMACS, with the exact validity of MDIR. | 2026-09-30 |
| [D83](decisions/D83.md) | The groups of the constraints are a disjoint union of tuple sets, which the IR states, and the loops over one group read the fields before the updates of the others. | 2026-09-30 |
| [D84](decisions/D84.md) | The deterministic level is a mode that a run chooses, not a constraint on every run | 2026-09-30 |
| [D85](decisions/D85.md) | Device code counts in 32 bits: the numbers of particles, cells, tuples, entries of a row, and loops; only a flattened element number and a byte address are 64 bits | 2026-09-30 |
| [D86](decisions/D86.md) | On a device, the loops over pairs run in the order of the cells of the last build, and the state keeps its order | 2026-09-30 |
| [D87](decisions/D87.md) | An op runs on a second stream only beside ops that are proven independent of it, and the proof is checked where it is lowered. | 2026-09-30 |
| [D88](decisions/D88.md) | A fixed interval of rebuilds of the neighbor structures is an opt-in policy, never a default, and a run that takes it is warned and told what it risked. | 2026-09-30 |
| [D89](decisions/D89.md) | On a device, the loops over pairs of large systems run over groups of 16 particles that share a list of neighbors, each pair once. | 2026-10-01 |
| [D90](decisions/D90.md) | Under `fast_math`, the f32 kernels of loops may approximate divisions and erfc, within a stated bound, and share the exponential of erfc with its derivative. | 2026-10-01 |
| [D91](decisions/D91.md) | What a kernel derives from tables alone is computed once per entry of the tables, in f64. | 2026-10-01 |
| [D92](decisions/D92.md) | The barostat scales the cell within the drift of the last step of a period by default (the integrator of Trotter type of [[Bernetti2020]](references.md#bernetti2020), SI Sec. V.C), and counts the energy of the scaling from the virials before and after it. | 2026-10-01 |
| [D93](decisions/D93.md) | A loop sums only the elements of a vector that are used, and the virial of pairs is the outer product of $\mathbf d$ and the vector of the forces. | 2026-10-01 |
| [D94](decisions/D94.md) | Under `fast_math`, what a pair kernel computes of the distance alone beyond roots and powers is a table in f32. | 2026-10-01 |
| [D95](decisions/D95.md) | The loop over groups takes no minimum image: the build moves each place and each entry by whole cells into the frame of its group. | 2026-10-01 |
| [D96](decisions/D96.md) | The tables that a kernel reads at the same entry are one table of vectors. | 2026-10-01 |
| [D97](decisions/D97.md) | In the loop over groups the mask of an entry does not turn with it: each lane takes its bits by ballots. | 2026-10-01 |
| [D98](decisions/D98.md) | On a device an approximate division in f32 is a product with an approximate reciprocal that flushes subnormal numbers. | 2026-10-01 |
| [D99](decisions/D99.md) | The build of the lists of groups queues the candidates near the box of a group and tests them 32 at a time. | 2026-10-01 |
| [D100](decisions/D100.md) | A candidate outside the range of the places of the partners of a group is not searched among its excluded pairs. | 2026-10-01 |
| [D101](decisions/D101.md) | A loop over groups takes the positions as they are stored; its gather moves them to the frames of the groups in f64 and then converts them. | 2026-10-01 |
| [D102](decisions/D102.md) | The gathering of PME of order 4 on a device takes the places of the particles from the weights. | 2026-10-01 |
| [D103](decisions/D103.md) | On a device, a loop over tuples with global sums also evaluates each tuple once. | 2026-10-01 |
| [D104](decisions/D104.md) | The influence function of PME on a device comes from factors along each edge, computed in f64; the energy and the virial are summed by blocks on the device. | 2026-10-01 |
| [D105](decisions/D105.md) | The build of the lists gathers the partners of the excluded pairs of a group a particle a lane, and sorts them with the network of the fewest values that hold them. | 2026-10-01 |
| [D106](decisions/D106.md) | A group whose excluded partners do not fit the memory of its warp takes its excluded pairs from their rows. | 2026-10-01 |
| [D107](decisions/D107.md) | A build of a neighbor structure on a device gives a position that is not a number no cell and no place, counts it, and the run then stops with a word. | 2026-10-01 |
| [D108](decisions/D108.md) | The bricks of the spreading of PME are a buffer of their own, which the copy into the grid leaves zero. | 2026-10-01 |
| [D109](decisions/D109.md) | The spreading and the gathering of PME of order 4 take the places of the particles, six values each, and compute the weights of the splines in closed form where they use them. | 2026-10-01 |
| [D110](decisions/D110.md) | On a device, a loop over particles, the loops over the tuples of a disjoint union that read it, and the loop over particles that reads them are one kernel, with a second for the tuples whose members lie in two warps. | 2026-10-01 |
| [D111](decisions/D111.md) | A warp of the first kernel of an integration run takes whole groups, and the second kernel is one block. | 2026-10-01 |
| [D112](decisions/D112.md) | Below double precision the waters are constrained by M-SHAKE, and `md-exec-approximate` leaves the kernels of the integrators and of the constraints as they are. | 2026-10-01 |
| [D113](decisions/D113.md) | The host reads a flag of the device without waiting for the work issued after it, and the reciprocal sum goes before the test of the neighbor structure. | 2026-10-01 |
| [D114](decisions/D114.md) | A structure of groups may keep a dual list: the loops over pairs take an inner list pruned from the outer one with a shorter reach, whenever its own test of validity asks. | 2026-10-01 |
| [D115](decisions/D115.md) | The lists of groups keep an entry for each image of a candidate within the reach, not only the image nearest to the center of the group. | 2026-10-02 |
| [D116](decisions/D116.md) | The count of the work of a scaling of Trotter type takes the virial of the groups from the forces of the evaluations, $\mathcal W = \mathrm{tr}\,\mathsf W_\text{eval} - \sum_g\sum_{j\in g}(\mathbf x_j - \mathbf X_g)\cdot\mathbf F_j$, not the virial of the step with twice the internal kinetic energy. | 2026-10-02 |
| [D117](decisions/D117.md) | On the CPU with OpenMP the variables of the reductions are allocated once, at the entry of the function: `hoist-static-allocas` runs after `convert-scf-to-openmp`. | 2026-10-02 |
| [D118](decisions/D118.md) | The flags that kernels set are memory of the host mapped for the device, which the runtime reads after an event and clears as it reads; each loop that sets a flag has one of its own. | 2026-10-02 |
| [D119](decisions/D119.md) | With `coupling = "SEMI_ISOTROPIC"` the barostat scales x and y together by the strain of the area and z by that of the height, each from its own pressure and its own noise (eqs. 9a and 9b of [[Bernetti2020]](references.md#bernetti2020)); the strain, the virials, and the kinetic energy are kept axis by axis. | 2026-10-02 |
| [D120](decisions/D120.md) | A minimization begins from the positions of the file taken onto the surface of the constraints. | 2026-10-02 |
| [D121](decisions/D121.md) | A topology of GROMACS may carry the terms of CHARMM force fields, and the Lennard-Jones of a topology takes a modifier of its own. | 2026-10-02 |
| [D122](decisions/D122.md) | MDIR reads the files of CHARMM: a PSF of the XPLOR kind, its CRD, and the files of topology, parameters, and streams that give its parameters | 2026-10-02 |
| [D123](decisions/D123.md) | Triclinic cells run on the CPU | 2026-10-02 |
| [D124](decisions/D124.md) | Under a barostat the reference of a restraint is a center that follows the cell plus an offset that stays: $\mathbf x^\text{ref} = \mathbf s \odot \mathbf c + \mathbf o$, $\mathbf s$ the edges of the cell over those of the file. | 2026-10-02 |
| [D125](decisions/D125.md) | Triclinic cells run on a device, with the neighbor matrix | 2026-10-02 |
| [D126](decisions/D126.md) | Groups of 16 run in triclinic cells, built in the brick of the cell with lattice shifts | 2026-10-02 |
| [D127](decisions/D127.md) | The barostat scales a triclinic cell as $H\,\mathrm{diag}(\boldsymbol\mu)$, and CHARMM's hexagonal cells run | 2026-10-02 |
| [D128](decisions/D128.md) | An opt-in analytic projection handles isolated one-bond constraint groups. | 2026-10-02 |
| [D129](decisions/D129.md) | `steps` is the length of a run, and `mdir run --continue` continues the run from its own checkpoint until it has taken them; a run that begins from the checkpoint of another begins at its step. | 2026-10-02 |
| [D130](decisions/D130.md) | A continued run appends its frames to the trajectory that its checkpoint counts them in, cut to those frames. | 2026-10-02 |
| [D131](decisions/D131.md) | SIGTERM, SIGINT, and `--max-walltime` stop a run of dynamics at its next checkpoint, with the exit status 75. | 2026-10-02 |
| [D132](decisions/D132.md) | The checkpoint before is kept as `<checkpoint>.prev`. | 2026-10-02 |
| [D133](decisions/D133.md) | The reader of Amber takes the format before Amber 7 and the convention of a negative periodicity, and the suite runs from its own files. | 2026-10-02 |
| [D134](decisions/D134.md) | The comparisons with pmemd.cuda run one model: its $\beta$, its grid, and its influence function. | 2026-10-02 |
| [D135](decisions/D135.md) | Langevin dynamics by the middle scheme is the second thermostat: `method = "LANGEVIN"` of `[thermostat]` with `friction` in 1/ps. | 2026-10-03 |
| [D136](decisions/D136.md) | Terms over tuples are given by expressions, as the custom forces of OpenMM over bonds, angles, and torsions: [[energy.bond]] in `r`, [[energy.angle]] and [[energy.dihedral]] in `theta`. | 2026-10-03 |
| [D137](decisions/D137.md) | With a topology, [[energy.pair]] adds a pair term given by an expression over the pairs of the topology, in its charges and its Lennard-Jones parameters, and `groups` restricts it to the pairs between two masks: the custom nonbonded force of OpenMM with its interaction groups. | 2026-10-03 |
| [D138](decisions/D138.md) | [[energy.function]] gives a function of one argument by its values at evenly spaced points, which every expression of the control file calls by its name: the continuous tabulated function of OpenMM. | 2026-10-03 |
| [D139](decisions/D139.md) | A term over tuples may take the centers of groups as its members: [[energy.bond]], [[energy.angle]], and [[energy.dihedral]] with `groups`, masks of Amber, in place of `particles`, the custom centroid bond force of OpenMM and a restraint between pull groups. | 2026-10-03 |
| [D140](decisions/D140.md) | `electrostatics = "REACTION_FIELD"` takes a dielectric continuum beyond the cutoff, of the relative permittivity `reaction_field_dielectric` (0 for a conductor), whose field acts on every pair of charges within the cutoff, those of one molecule as well, as GROMACS has it. | 2026-10-03 |
| [D141](decisions/D141.md) | A trajectory is in DCD or in the XTC of GROMACS, by the extension of its name (`trajectory_format = "AUTO"`, the default) or as `trajectory_format` says. | 2026-10-03 |
| [D142](decisions/D142.md) | `type = "NONE"` in [boundary] runs without a periodic cell: the run places the particles in a cell that keeps every image beyond the reach of the neighbor structures, and stops if they spread so far that one could come within it. | 2026-10-03 |
| [D143](decisions/D143.md) | A potential may compute fields from the positions and sum them into its energy, and differentiation carries the derivative of the energy back through them: intermediate fields, the first being the Born radii of generalized Born. | 2026-10-03 |
| [D144](decisions/D144.md) | `implicit_solvent = "OBC2"` (or `"OBC1"`) in [energy] adds generalized Born of Onufriev, Bashford, and Case to a topology of Amber, from its sections RADII and SCREEN, with `solvent_dielectric` (78.5), `solute_dielectric` (1), and `surface_area_energy` in kcal/mol/Å² for the nonpolar term (0, none, by default). | 2026-10-03 |
| [D145](decisions/D145.md) | The time `t` in ps may enter the expressions of terms over tuples, over the centers of groups, and over the pairs of a topology, and `pull_coordinates` in [output] names a file of the coordinates of the terms over centers, with their energies and forces, at every energy of the log. | 2026-10-03 |
| [D146](decisions/D146.md) | Unknown control keywords explain how to repair the input | 2026-10-03 |
| [D147](decisions/D147.md) | The standard pipeline is available as kinds of `mdir template` | 2026-10-03 |
| [D148](decisions/D148.md) | `[[energy.external]]` adds a term of the absolute positions of single particles, given by an expression in `x`, `y`, `z` in Å, the charge `q` in e, the time `t` in ps, and parameters, over the particles of a mask of Amber (`selection`) or of their numbers (`particles`), as the custom external force of OpenMM [[Eastman2017]](references.md#eastman2017). | 2026-10-03 |
| [D149](decisions/D149.md) | The outputs of a run form one system: `[output]` names every file and its interval, the files of columns share one form, every file continues with the run, and a run that is not continued keeps the outputs of an earlier one as backups ([driver.md](driver.md), Section 2.8). | 2026-10-03 |
| [D150](decisions/D150.md) | The sums over few tuples cost less: a block adds all the sums of its kernel in one pass, a reduction of one part gives the results without a second kernel, a loop that evaluates each tuple once launches no more rows than it has tuples, the destinations of one loop that a loop over particles adds one after the other merge, loops over tuples fuse past the pure ops that use the first, and the host code is scheduled by the fast list scheduler of LLVM. | 2026-10-03 |
| [D151](decisions/D151.md) | `mdir check` is a read-only preflight, with a JSON report | 2026-10-03 |
| [D152](decisions/D152.md) | Generalized Born takes the model of Hawkins, Cramer, and Truhlar (`implicit_solvent = "HCT"`), a salt (`salt_concentration` in mol/L), a cutoff of the descreening (`born_radius_cutoff` in Å), and the radii and screening of mbondi2 by element (`born_radii = "MBONDI2"`) for a topology without its own. | 2026-10-03 |
| [D153](decisions/D153.md) | What the run does that its user may not intend is a warning of the system, which `mdir run` prints and `mdir check` lists by its code; the first is the waters of a topology of Amber or CHARMM that run flexible because [constraints] does not say how (`flexible_water`). Without a periodic cell the spread of the particles is checked at every row of the log as well as at the frames, the checkpoints, and the end (amends D142). | 2026-10-03 |
| [D154](decisions/D154.md) | The virial of a term of the absolute positions is the derivative of its energy when the barostat scales the positions and the cell, $\mathsf W_{aa} = -\partial U/\partial\varepsilon_a$, and `scaling` in [[energy.external]] says how the term follows the cell: `"NONE"` keeps it fixed in space, `"CELL"` takes its positions in the frame of the cell, $\mathbf x \odot \mathbf L_0/\mathbf L$ with $\mathbf L_0$ the cell of the input; a run at constant pressure must say which (amends D148). | 2026-10-03 |
| [D155](decisions/D155.md) | The README begins with a runnable quickstart, and `mdir doctor` checks an installation by compiling and running an embedded system on each built target. | 2026-10-03 |
| [D156](decisions/D156.md) | A kernel that takes a value the host computes from global sums computes it itself from the device's copy of the sums, and the host no longer waits for those sums. | 2026-10-03 |
| [D157](decisions/D157.md) | Two destinations of one loop over pairs that a loop over particles adds merge into one when the pair gives them by the same contract of exchange, as `md-exec-accumulate-destinations` merged those of a loop over tuples (D150). | 2026-10-03 |
| [D158](decisions/D158.md) | More warnings of the system (D153), and a selection of nothing is one of them rather than an error. | 2026-10-03 |
| [D159](decisions/D159.md) | A function of the distance that no table holds stays in its kernel, and the square of the cutoff in f32 is pulled in so that the square root of every square that passes is four units in the last place below the cutoff in f32. | 2026-10-03 |
| [D160](decisions/D160.md) | Terms over the triplets centered on each particle are a relation of arity 3 that `md.triplets` derives from a neighborhood, summed by the tuple ops; on the CPU its members are written at every evaluation into a buffer as long as the triplets, and the loops over tuples run on it unchanged. | 2026-10-03 |
| [D161](decisions/D161.md) | `[free_energy]` decouples a selection of whole molecules through states of $\boldsymbol\lambda$, every component a parameter of the expressions, and writes $\partial U/\partial\lambda$ and the energy of every state at every energy, for thermodynamic integration and MBAR. | 2026-10-04 |
| [D162](decisions/D162.md) | Particle mesh Ewald for the dispersion: `lennard_jones = "PME"` sums $-c_ic_j/r^6$ over all pairs and images on a grid of its own, with $c_i = 2\sqrt{\varepsilon_i}\,\sigma_i^3$ of the type of each particle with itself, and the direct terms give each pair within the cutoff its own Lennard-Jones. | 2026-10-03 |
| [D163a](decisions/D163a.md) | A Nosé–Hoover chain is the deterministic thermostat: `method = "NOSE-HOOVER"` of `[thermostat]`, with `time_constant` $\tau_T$ in ps and `chain_length` $M$ (3 by default), acting at the end of each period of coupling as velocity rescaling does (D67). | 2026-10-03 |
| [D163b](decisions/D163b.md) | Brownian dynamics is an integrator: `integrator = "BROWNIAN"` with `friction` $\gamma$ of `[dynamics]` in 1/ps, at the temperature of `[ensemble]`, which must be `NVT`, and with no `[thermostat]`. | 2026-10-03 |
| [D163c](decisions/D163c.md) | Anisotropic cell rescaling: `coupling = "ANISOTROPIC"` of `[barostat]`, each axis scaled by the strain of its own edge from its own pressure and noise, $\Delta\varepsilon_a = -\tfrac{f_a}{3}(P_0 - P_{aa}) + \sqrt{2k_BTcf_a/(3V)}\,R_a$, the step of z of eq. (9b) of [[Bernetti2020]](references.md#bernetti2020) on each axis, with `compressibility` one number or three (x, y, z; 0 keeps an edge to the bit, all 0 is an error). | 2026-10-03 |
| [D165](decisions/D165.md) | Parameters of each particle are entries of the control file resolved by the driver into a value for every particle; a pair term gathers them as a field of the particles, and a term over tuples or of the positions takes them as parameters of its tuples or particles. | 2026-10-03 |
| [D166](decisions/D166.md) | MLIP support starts with one existing model through an optional metatomic adapter on one GPU, then validates that same local model on two GPUs partitioning one physical system; staged message-passing optimization follows. | 2026-10-03 |
| [D167](decisions/D167.md) | Neural-network reverse-mode AD and tensor execution belong to an external PyTorch/JAX backend; MDIR verifies and schedules particle dependencies and communication. The model progression is Allegro → PaiNN → MACE. | 2026-10-03 |
| [D168](decisions/D168.md) | An optional `[output].manifest` records execution provenance as versioned JSON Lines. | 2026-10-03 |
| [D169](decisions/D169.md) | After the first milestone, the milestones are M2, the Python API; M3, learned potentials (MLIP); and M4, distributed execution. | 2026-10-03 |
| [D170](decisions/D170.md) | Barostat outputs report the area of the xy face spanned by the first two cell vectors beside the volume. | 2026-10-03 |
| [D171](decisions/D171.md) | On the CPU every reduction of a parallel loop is added over a fixed partition of its iterations into chunks, each folded in order, and the chunks folded in their order; the result does not depend on the threads or their number, in every mode. | 2026-10-03 |
| [D172](decisions/D172.md) | A checkpoint records what defined its run, and a run that takes it compares: `--continue` refuses a change of the physics or of the coupling and names it; a run that begins from another run's checkpoint evaluates its forces and the state of its barostat again when anything differs. | 2026-10-03 |
| [D173](decisions/D173.md) | Format 1 of the checkpoint is the contract of release 0.1.0: the reader checks `format`, refuses a newer one, and refuses a file of a development build; the state carries its SHA-256, and the file is on stable storage before it takes its name. | 2026-10-03 |
| [D174](decisions/D174.md) | A release is an installed tree, a container, and a tagged source, and its version comes from one place. | 2026-10-03 |
| [D175](decisions/D175.md) | A topology Lennard-Jones potential can be multiplied by the cubic switch in squared distance of CHARMM VSWITCH. | 2026-10-03 |
| [D176](decisions/D176.md) | A build of groups on a device also counts as beyond any cell a position whose wrap does not land in the cell, and clamps the cell of every key on both sides. | 2026-10-04 |
| [D177](decisions/D177.md) | The binary tarball of a release is built for manylinux_2_28 and needs no CUDA toolkit: it is built on the PyPA image of glibc 2.28 (`packaging/Dockerfile.manylinux`), links libstdc++ and libgcc statically, and carries the runtime, the OpenMP runtime, HDF5, cuFFT, and libdevice; without `CUDA_ROOT`, `CUDA_HOME`, or `CUDA_PATH` the kernels link the libdevice in `share/mdir/cuda`. | 2026-10-04 |
| [D178](decisions/D178.md) | Tabulated functions may read their values from a text file. | 2026-10-04 |
| [D179](decisions/D179.md) | Moving a GPU reciprocal sum before a neighbor refresh requires independence from every intervening op, as well as dominance of its operands. | 2026-10-04 |
| [D180](decisions/D180.md) | An approximate f32 quotient rounds before subsequent arithmetic on the GPU. | 2026-10-04 |
| [D181](decisions/D181.md) | On a device, a reduction of one part computes the contributions of its rows in its own block, and adjacent loops that evaluate each tuple once without global sums are launched as one kernel; a term over the centers of groups then costs two kernels fewer. | 2026-10-04 |
| [D182](decisions/D182.md) | Activity is a separate analysis shared by scalar derivatives, positional fields, sum weights, and parameter derivatives. | 2026-10-04 |
| [D183](decisions/D183.md) | Scalar rules are dispatched by an op interface, with external models for arithmetic, math, vectors, and table lookup. | 2026-10-04 |
| [D184](decisions/D184.md) | Requested derivatives can be checked against central differences of energy at an evaluation's actual inputs. | 2026-10-04 |
| [D185](decisions/D185.md) | Singular points use explicit reference conventions, without inventing finite derivatives. | 2026-10-04 |
| [D186](decisions/D186.md) | Custom expression operations retain their control-file term names in MLIR locations, and an unknown activity proof retains its first unknown operation for diagnostic attribution. | 2026-10-04 |
| [D187](decisions/D187.md) | M2 preparation records the Python API prerequisites, seven sequential implementation PRs, and validation gates under D169, with maintainer rulings approved 2026-10-04. | 2026-10-04 |
| [D188](decisions/D188.md) | Term parameters cannot shadow supplied names, and compound expressions accept declared lambda components. | 2026-10-04 |
| [D189](decisions/D189.md) | A term given by an expression observes its energy and its generalized forces along its own constants, written at every energy; the selection lives in the term (`observe = [constants]`) and the columns in one file, `[output] observables`. | 2026-10-04 |
| [D190](decisions/D190.md) | With one state, `[free_energy]` writes $\partial U/\partial\lambda$ alone. | 2026-10-04 |
| [D191](decisions/D191.md) | The first M2 implementation supplies a native value model and shares preparation with file input. | 2026-10-04 |
| [D192](decisions/D192.md) | M2 exposes an optional Python lowering interface with owned, versioned inputs and immutable compiled IR. | 2026-10-04 |
| [D193](decisions/D193.md) | The M2 host boundary uses shaped arrays before segments. | 2026-10-05 |
| [D194](decisions/D194.md) | On a device, the sums of a loop over tuples evaluated once that reach only a later such loop, through arithmetic, are computed by a block of the kernel of the run of the later loop; a term over the centers of groups then takes no kernel of its own. | 2026-10-05 |
| [D195](decisions/D195.md) | M2 is split into M2a, the Python API of D187, and M2b, differentiable simulation; M2b adapts PyTorch first and JAX second. | 2026-10-05 |
| [D196](decisions/D196.md) | A Python simulation persists across runs of any number of steps: its programs take the counts of their loops as arguments, the first begins as `mdir run` does and the second continues the state of the last segment as a run continues its checkpoint, and the periods of coupling are counted from the first step. | 2026-10-06 |
| [D197](decisions/D197.md) | The host code of a program is compiled at LLVM's default level of code generation, by one machine that both front ends make (#90). | 2026-10-06 |
| [D198](decisions/D198.md) | Python draws initial velocities by the CLI's code and takes positional restraints as typed `[[restraints]]` tables. | 2026-10-06 |
| [D199](decisions/D199.md) | Persistent simulations own the final object memory and unwind boundary. | 2026-10-06 |
| [D200](decisions/D200.md) | Every setter of the Python model whose value has a unit also takes an `openmm.unit.Quantity`, converted at the boundary to the unit the setter declares; outputs stay plain floats and NumPy arrays. | 2026-10-06 |
| [D201](decisions/D201.md) | In the deterministic mode a step that writes energies moves the particles as a step that does not, where the compiler can guarantee it: device kernels contract no product into a fused multiply-add, and the virial of the constraints reads the velocity changes that the step applies. | 2026-10-06 |
| [D202](decisions/D202.md) | A Python simulation of a program whose integrator minimizes runs the minimizer of `mdir run`, `Simulation.minimize(steps=None)`, in parts as `run(n)` runs steps. | 2026-10-06 |
| [D203](decisions/D203.md) | The log reports the optimal temperature of Jung, Kobayashi, and Sugita (2019) [[Jung2019]](references.md#jung2019), $N_fk_BT = \tfrac43K_\text{half} + \tfrac23K$, and its pressure takes $K_\text{half}$, with and without constraints; with constraints the steps of the rows of the log measure $K_\text{half}$ from the velocities of the half steps after the constraints of the positions. The thermostats and the barostats keep the kinetic energy $K$ of the velocities they scale. | 2026-10-06 |
| [D204](decisions/D204.md) | In the deterministic mode a step that writes energies moves the particles as a step that does not, with constraints and PME, on the CPU and on devices, and a Python simulation of a model gives the state of `mdir run` within a part. | 2026-10-06 |
| [D205](decisions/D205.md) | The Python model takes the control file's Coulomb modifier as `System.coulomb_modifier`. | 2026-10-06 |
| [D206](decisions/D206.md) | A Nosé–Hoover chain whose action the canonical factorization no longer follows stops the run: after each action, a scaling factor or a chain variable that is not finite, or a part that met $\lvert s\,v_j\rvert > 3$ ($s = w_kh/n_c$ a part of a Suzuki–Yoshida weight, $v_j = p_{\eta_j}/Q_j$), ends the run with an error before the velocities are scaled. This is a guard on the scheme of D163a, not a change of the integrator: $n_c = \lceil 50h/\tau_T\rceil$ and the order stay fixed before the run, and every run that the guard does not stop moves as before, bit for bit. | 2026-10-06 |
| [D207](decisions/D207.md) | A Python simulation's reporters write `mdir run`'s energy file and trajectory inside the parts of a run, at times that are arguments of the compiled program, and call Python functions at their steps. | 2026-10-06 |
| [D208](decisions/D208.md) | The test suite runs at most sixteen tests that need a GPU at once by default (`gpu_workers` of `test/mdir_lit.py`), and no Python test runs all its scenarios in one file. | 2026-10-06 |
| [D209](decisions/D209.md) | With a topology, the correction for the dispersion includes the tail beyond the cutoff of every `[[energy.pair]]` term, over the pairs its `groups` select and the topology does not exclude, in the energy, the virial, the λ-state energies, `dH/dλ`, and the columns of `observe`. | 2026-10-06 |
| [D210](decisions/D210.md) | Under a plain cutoff the free-energy file and the observables file differentiate the potential that the forces sample, shifted to 0 at the cutoff; with a topology the correction for the dispersion is allowed under `POTENTIAL_SHIFT` and adds the estimate of what the shift takes within the cutoff. | 2026-10-06 |
| [D211](decisions/D211.md) | A Python simulation compiles one program of segments, whose entry does the work of the start only on its first call, and every lowering of the process takes MLIR's threads from one pool of the process. | 2026-10-06 |
| [D212](decisions/D212.md) | A Python simulation's JIT engine takes its host object from an on-disk cache keyed by the content of its LLVM module and of the code generator, not by the build of MDIR; the cache is enabled by `MDIR_COMPILE_CACHE_DIR`, disabled by `MDIR_COMPILE_CACHE=off`, and bounded by `MDIR_COMPILE_CACHE_MAX_MB` (2048, least recently used first), and `Simulation.compile_stats` reports it. | 2026-10-07 |
| [D213](decisions/D213.md) | A Python model declares tunable parameters, whose values a simulation takes without compiling: each a named vector $\boldsymbol\theta$ with a map $\iota$ from its sites (particles for `"charge"`, Lennard-Jones types for `"sigma"` and `"epsilon"` with the combining rule, one site for a constant of a pair term, tuples for a parameter of a tuple term), and an update puts the values into the model and rebuilds the program's values by the code that compiles them, refusing any value that would change the program's text. | 2026-10-07 |
| [D214](decisions/D214.md) | The GPU modules of a program are serialized in parallel on the process's compile threads, their PTX and cubins are kept in the compile cache, and they are loaded as cubins compiled by the toolkit's `ptxas` for the architecture of the device that NVML reports. `MDIR_GPU_BINARY` (`auto`, `cubin`, `ptx`) and `MDIR_GPU_ARCH` choose otherwise. | 2026-10-07 |
| [D215](decisions/D215.md) | A Python simulation runs every part in one activation of the entry of its program of segments, which waits at the end of each part on a stack of its own and hands the host its state where it is; its buffers, order of the particles, and neighbor structures persist from part to part, a run in parts equals one run to the bit in the deterministic mode, and the host copies the state only where it is asked for. | 2026-10-07 |
| [D216](decisions/D216.md) | The control structure holds the constants of restraints in kJ/mol/nm², converted from the control file's kcal/mol/Å² once where `[[restraints]]` is read; a typed Python restraint passes its constant through unchanged. | 2026-10-07 |
| [D217](decisions/D217.md) | `mdir.clear_compile_cache(directory=None)` removes the entries of this format from the directory given or from `MDIR_COMPILE_CACHE_DIR` and returns what it removed; `cache=False` on `mdir.compile` (keyword only, recorded in the program) and on `Simulation(program, cache=None)` (keyword only, `None` takes the program's choice) bypasses the cache for one compile, and `compile_stats["cache_bypassed"]` reports it. | 2026-10-07 |
| [D218](decisions/D218.md) | The evaluation at the start of a run is a segment of its own in every program: the steps begin with neighbor structures of their own, built at step 1, in `mdir run` with and without checkpoints and in a Python simulation. | 2026-10-07 |
| [D219](decisions/D219.md) | A minimization stops at the first row of the energies whose largest force, without its parts along the constraints, is below a tolerance: `[minimize] force_tolerance` (kcal/mol/Å) and `Simulation.minimize(steps=None, tolerance=None)` (kJ/mol/nm). | 2026-10-07 |
| [D220](decisions/D220.md) | A Python simulation exposes its state where its program keeps it as read-only DLPack views: `Simulation.view()` returns the positions, velocities, and forces of the last boundary of its activation in the order of the program and the types it stores, the input index of each row, and the host values of its tunables; the view and every managed tensor taken from it hold a lease, and while one is alive the operations that write or free the buffers are refused. | 2026-10-07 |
| [D221](decisions/D221.md) | The Python model gives read-only views of a topology, copies taken when they are read, and evaluates the masks of Amber by the parser of the control file. | 2026-10-07 |
| [D222](decisions/D222.md) | A Python pair term takes `dispersion`, the control file's `dispersion_correction` of a pair term with a topology: Python's `None` (the default) follows the system, `DispersionCorrection.None_` leaves the term out of the correction for the dispersion, `EnergyPressure` asks for its tail; and setting `System.dispersion` makes the system's correction explicit, as giving the key of `[energy]`, with `None` restoring the default. | 2026-10-07 |
| [D223](decisions/D223.md) | A Python simulation writes the checkpoint of `mdir run` and continues one written by either front end, keeping D172's distinction between the same run and a new stage; the fingerprint of a Python model holds the entries that the control file of the same model writes, and format 1 gains entries that earlier readers ignore. | 2026-10-07 |
| [D224](decisions/D224.md) | `mdir.compile` builds a program and sets up its pipeline without running it; `Program.lowered_ir` lowers on its first read. | 2026-10-07 |
| [D225](decisions/D225.md) | A part of a Python simulation that fails runs to its end without any access outside a buffer, whatever its positions or cell, and a barostat that scales an edge of the cell by a factor outside [1/2, 2] at one coupling, or makes it not finite, fails the run. | 2026-10-07 |
| [D226](decisions/D226.md) | A Python model may declare the Lennard-Jones table tunable by pairs of types, `mdir.Tunable(name, "sigma_pair")` and `"epsilon_pair"`, whose sites are the unordered pairs of types $(a,b)$, $a \le b$, in the order of the flat upper triangle, $s(a,b) = aT - a(a-1)/2 + b - a$, listed by `System.type_pairs`; a pair that a map takes has its entry's value after the combining rule of per-type tunables. | 2026-10-07 |
| [D227](decisions/D227.md) | The values of a program that depend on the state it starts from are arguments of its entry, not constants of its text: the tilts of a triclinic cell, the edges of the cell that the restraints' reference positions belong to under a barostat, the barostat's virial and energy of the dispersion correction and the PME background times the volume, and the barostat state of a continued run that scales every step (`Program::startValues`). The PME grid and the neighbor capacity stay structural; the estimate of the capacity is rounded up to four significant bits, and `Execution.neighbor_capacity` gives it from Python. | 2026-10-07 |
| [D228](decisions/D228.md) | The Python interface is the package `mdir`, a pip wheel for CPython 3.10–3.13 on manylinux_2_28 (x86-64) built by scikit-build-core from `pyproject.toml`, whose version is read from `project(mdir VERSION ...)`; the extension `mdir._core` finds its runtime (`mdir/lib`) and libdevice (`mdir/cuda`) beside itself, HDF5 comes grafted by auditwheel, cuFFT and ptxas from NVIDIA's wheels `nvidia-cufft` and `nvidia-cuda-nvcc` through the extra `mdir[cuda]`, and the `mdir` command stays out of the wheel. | 2026-10-08 |
| [D229](decisions/D229.md) | A Python simulation lends its state for writing through DLPack as a tracked borrow with an explicit commit: `Simulation.borrow()` returns the buffers of the positions and the velocities where the program keeps them, without the read-only flag, and buffers of the host with the edges of the cell and the values of the tunables; `Borrow.commit()` checks everything, advances the versions of the changed fields, and begins an activation from the committed state as an update of tunables does, so that the order, the neighbor structures, the derived fields, and the forces are built anew; a borrow excludes every other operation, and one that ends without a commit is undone from the copy of the last boundary. | 2026-10-08 |
| [D230](decisions/D230.md) | A program compiled with `System.tunable_gradient` evaluates, when `Simulation.tunables.gradient()` asks, the derivative of the potential that the forces sample in every entry of every tunable, $\partial U/\partial\boldsymbol\theta$ with the named structure of $\boldsymbol\theta$: `md-differentiate` gives the derivative of a potential in an argument that is a field of the particles, every site derivative leaves the program as such a field through a seed, a field of zeros added with weights where a kernel reads a parameter, and the host applies the chain rule beside `applyTunables` and adds the derivatives of the constants of the energy. | 2026-10-08 |
| [D231](decisions/D231.md) | The derivative of the energy in the charges with particle mesh Ewald: `md.reciprocal` yields, as an optional fourth result, the potential of the grid at the particles, $\partial E_\text{rec}/\partial q_i = \sum_{\mathbf k}\theta_i(\mathbf k)\phi(\mathbf k)$, gathered with the weights of the B-splines in the three templates of PME, only in a program compiled with the derivative; the host adds the derivatives of the self term and of the background. | 2026-10-08 |
| [D232](decisions/D232.md) | A term of the Python model given by an expression takes `observe`, the control file's key (D189); a Python simulation writes the file of `[output] observables` through `ObservablesReporter` and gives the same values to a script in `State.observables` at every step of energy; tunables and `observe` share a program, and an observed constant may be a tunable with one entry. | 2026-10-08 |
| [D233](decisions/D233.md) | The Python model takes terms of the absolute positions, `mdir.ExternalTerm` in `System.external_terms`: the control file's `[[energy.external]]` (D148) with `x`, `y`, `z` in nm and the energy in kJ/mol, over a mask or an array of particle indices, with constants. | 2026-10-08 |
| [D234](decisions/D234.md) | The compile cache keeps the total of its entries in a file, `<dir>/size`, which a compile that stored entries updates under a lock of the file, and lists the directory to evict only when that total exceeds the bound, when no total is known, or when the last listing is a day old. | 2026-10-08 |
| [D235](decisions/D235.md) | On a device the particles of a cell of the neighbor matrix take their places by rank: each particle counts the particles of its cell with a smaller index, one thread per particle, in place of a sort by insertion in one thread per cell. | 2026-10-08 |
| [D236](decisions/D236.md) | A `Program` keeps the code of its simulations in memory: the first `mdir.Simulation(program)` leaves the bitcode of the LLVM module of its program of segments and the host object of that module with the `Program`, and a later simulation with the same key parses the one and links the other, without the MLIR pipeline and without host code generation. | 2026-10-08 |
| [D237](decisions/D237.md) | The potential of an output reads the neighbor structure of the steps and does not refresh it: `md-exec-reuse-neighbors` removes a refresh that repeats the last refresh of its structure (the same positions and cell, as values, and the same parameters), through a loop that only repeats it as well; `output-independence*.test` hold that a run with an output ends in the state of the run without it, bit for bit in the deterministic mode. | 2026-10-08 |
| [D238](decisions/D238.md) | A writable borrow lends the tilts of a triclinic cell: `Borrow.tilt` is a buffer of the host with $b_x, c_x, c_y$ beside `Borrow.cell`, which stays the diagonal $a_x, b_y, c_z$ for every program, the two arrays of `mdir.Cell`; a commit checks the cell (finite, the diagonal at least twice the cutoff, reduced, the pairlist distance at most half of the least of the diagonal), refuses tilts that are all zero as a structural change, and begins the activation with the committed tilts, which are start values of the entry (D227) taken from the state of the host; the grid of PME and the neighbor capacity stay the program's, and no program is built. | 2026-10-09 |
| [D239](decisions/D239.md) | A trajectory that holds the frames of a run without loss is a third trajectory format, H5MD 1.1 [[deBuyl2014]](references.md#debuyl2014): `[output] trajectory = "x.h5md"` (by the extension `.h5md`, or `trajectory_format = "H5MD"`; `trajectory_precision = "DOUBLE"` or `"SINGLE"`, `trajectory_strings = "FIXED"` or `"VARIABLE"`) and `mdir.H5MDReporter(file, period, positions="f64", velocities=False, forces=False, strings="fixed")`; `mdir.read_h5md` reads it one frame at a time. | 2026-10-09 |
| [D240](decisions/D240.md) | `mdir.FrameEvaluator(program)` evaluates the potential that the forces sample (D210), the virial, and the columns of `observe` at $K$ stored frames given in the order of the input, each with its cell, in a simulation of its own, with one start of an activation per frame, and gives the vector-Jacobian product in the tunables, $\mathbf g \mapsto \sum_n g_n\,\partial U_{\boldsymbol\theta}(S_n)/\partial\boldsymbol\theta$; `mdir.torch.evaluate` makes it an operation of PyTorch (`torch.library.custom_op` with `register_autograd`). | 2026-10-09 |
| [D241](decisions/D241.md) | The neighbor matrix of a triclinic cell takes a pair if any of its images is within the reach; the build, a commit, and the runtime ask I2 alone of the cell | 2026-10-09 |
| [D242](decisions/D242.md) | The lists of the groups hold every image of every pair within a reach below the least edge of the cell, in an orthorhombic and in a triclinic cell; the builder, a commit, and the runtime refuse a longer one | 2026-10-09 |
| [D243](decisions/D243.md) | A frame evaluator made with `terms="dependent"` evaluates the terms that a tunable enters alone, $U_\text{dep}$ of $U = U_\text{fixed} + U_\text{dep}(\boldsymbol\theta)$, in a second program that it compiles from the model of the program it is given; its energy is named `dependent_energy`, and `energy` and `virial` raise in that mode. | 2026-10-10 |
| [D244](decisions/D244.md) | A Python simulation runs at constant pressure in a triclinic cell: the tilts that the barostat scales with the cell (D127) and reports at each scaling become the tilts of the state of the host at the end of a part, through which `State.cell`, a checkpoint, the frames of the reporters, `Borrow.tilt`, and the next activation (D238) have them; a part that fails restores the cell of before with its tilts. | 2026-10-10 |
| [D245](decisions/D245.md) | The Python model selects the groups of 16 and the dual list with the keys of the control file: `Execution.neighbor_structure` (`NeighborStructure.Matrix`, the default, or `Groups`) and `System.pruned_distance` (nm; 0 keeps one list), refused as `[execution] neighbor_structure` and `[energy] pruned_distance` are, with the words of `mdir check`; with one structure a Python simulation runs at the rate of `mdir run`. | 2026-10-10 |
| [D246](decisions/D246.md) | The Python model has `[pme] influence` and `[constraints] analytic_bonds` as `System.pme_influence` (`PMEInfluence.SPME`, the default, or `Optimal`) and `System.analytic_bonds`, and `System.pme_order` takes 6 and 8 as the control file does. | 2026-10-11 |
| [D247](decisions/D247.md) | The Python model selects the coupling and the work of the barostat with the keys of the control file: `Ensemble.barostat_coupling` (`BarostatCoupling.Isotropic`, the default, `SemiIsotropic`, `Anisotropic`) and `Ensemble.barostat_work` (`BarostatWork.Trotter`, the default, `TrotterFirstOrder`, `Exact`, `FirstOrder`), with the refusal of `mdir check`; each runs to the bit of `mdir run`. | 2026-10-11 |
| [D248](decisions/D248.md) | Where the Python model and the control file had different defaults for one setting, the Python model takes the control file's: a plain cutoff, `switch_distance` and `pairlist_distance` from the cutoff, the motion of the center of mass removed with the thermostat, and, for a state without velocities, those that `mdir run` draws; a system given to both front ends with nothing beyond its inputs is one model, to the bit. | 2026-10-11 |
| [D249](decisions/D249.md) | The Python model selects the thermostat with the keys of the control file, and takes Brownian dynamics: `Ensemble.thermostat` (`ThermostatMethod.VRescale`, the default, `Langevin`, `NoseHoover`), `Ensemble.friction` (1/ps, needed by Langevin dynamics and refused otherwise), `Ensemble.chain_length` (3 by default, for a Nosé–Hoover chain), and `Integrator.method = IntegratorMethod.Brownian` with `Integrator.friction` under `EnsembleKind.NVT`; `Ensemble.coupling_period` is `None` until set, the control file without `interval`. | 2026-10-11 |
| [D250](decisions/D250.md) | For a device, the loop over particles that closes a run of loops over disjoint tuples is not fused with the one that opens the next, so that each of the two runs of a leapfrog step under Langevin dynamics with constraints is one integration kernel | 2026-10-12 |
| [D251](decisions/D251.md) | The frames of a frame evaluator are evaluated inside one activation of a program of frames, whose entry is a loop over frames that takes the positions and the cell of each at a boundary; the program keeps the particles in the order of the input, and the energy of a frame is a function of the frame alone. | 2026-10-12 |
| [D252](decisions/D252.md) | A frame evaluator gives, where they are asked, the gradients of the energy of a frame in the positions and in the strain, $\partial U/\partial\mathbf x_i = -\mathbf F_i$ and $\partial U/\partial\varepsilon_{ab} = -\mathsf W_{ba}$ under $\mathbf x \to (\mathsf 1 + \boldsymbol\varepsilon)\mathbf x$ of the positions and the cell vectors; one argument, `gradients=`, a tuple of names among `"tunables"`, `"positions"`, and `"strain"`, replaces `gradient=`; `mdir.torch.evaluate` differentiates the energy in the positions and the cell, and every second derivative is refused. | 2026-10-13 |
| [D253](decisions/D253.md) | A simulation keeps the renumbered members of its tuples and their incidence structures from one activation of its entry to the next; a start takes them only where the identities of the particles and the members are equal, element for element, to those they were built from, and builds them in the memory of the last where they are not. | 2026-10-13 |
| [D254](decisions/D254.md) | An update of the tunables writes its values, in place, into a model that the simulation keeps for its updates; whether a tuple set is disjoint is found in one pass over its members; and a build computes each integral of the tail of a pair term once. | 2026-10-14 |
| [D255](decisions/D255.md) | Every program takes the renumbered members of its tuples and their incidence structures from the runtime, and `mdir run` opens a store of the process for its one call of the entry, so that each segment of checkpoints, which begins in a new order of the particles, builds them in the memory of the segment before. | 2026-10-14 |
| [D256](decisions/D256.md) | An update of the tunables keeps the arrays of the program that no value of a tunable enters, found by a build from the kept model with every tunable site set to not-a-number and compared to the bit with the compiled program's, and shares them with copy on write; it makes the others by the functions that make them at a compile; the first update of each set of tunables that changed is compared with a build from a copy of the model of the compile, and a difference is never a silent fallback. | 2026-10-14 |
| [D257](decisions/D257.md) | On a GPU, the renumbered members of the tuples and their incidence structures are built on the device by kernels of the program, as a gather of the structure in the order of the files: the row of place $p$ is the row of the particle $\pi(p)$ in that order with its members relabeled by $\pi^{-1}$; the identities are compared with the store's last order on the device, and only a flag of 4 bytes reaches the host. | 2026-10-14 |
| [D258](decisions/D258.md) | The entry of a program declares the buffers of the model that it takes read-only (`mdrt.read_only` on the masses, the fields of the particles, the tables, and the members and fields of the tuple sets); `md-exec-assign-storage` never takes such a buffer for another field or for a copy back from a device; and `check-read-only-arguments`, where the lowering to loops or to a device ends, proves that the code writes none of them or fails the compile. A start of an activation then hands the entry the arrays of the program without a copy, keeps their conversions into the type of the entry where it is given the same array, and keeps the buffers of the state, written anew at every start. | 2026-10-15 |
| [D259](decisions/D259.md) | Steepest descent takes a trial only if the constraints took every group back to its shape: a trial whose constrained lengths are off their targets, $\lvert r^2/d^2 - 1\rvert > 10^{-4}$ below double precision and $10^{-8}$ in double, or are not numbers, is evaluated where the step began and given the energy $+\infty$, so that the step shrinks. | 2026-10-16 |
| [D260](decisions/D260.md) | The step of steepest descent grows to at most 0.1 Å, not 1 Å. | 2026-10-16 |
| [D261](decisions/D261.md) | `neighbor_structure = "AUTO"` is the default: on a GPU, outside the deterministic mode, a system of at least 23,000 particles whose loops over pairs all take each pair once takes the groups of 16 with a dual list, `pairlist_distance` = cutoff + 3 Å and `pruned_distance` = cutoff + 0.6 Å × max(1, Δt / 2 fs) where they are not given; every other run takes the matrix with its reach of cutoff + 1.5 Å. `MATRIX` and `GROUPS` given are unchanged. | 2026-10-16 |
| [D262](decisions/D262.md) | The reciprocal sums of PME spread only the particles whose coefficient can be nonzero, from a list of their places that a field of the particles holds and that is made again wherever the particles take a new order; a device gathers over the list where at most half the particles are in it, and every particle otherwise (`gather_all`). | 2026-10-16 |
| [D263](decisions/D263.md) | Stochastic velocity rescaling and the Nosé–Hoover chain control the optimal temperature of Jung, Kobayashi, and Sugita (2019) [[Jung2019]](references.md#jung2019), $N_fk_BT_\text{opt} = \tfrac43K_\text{half} + \tfrac23K$, not the kinetic energy $K$ of the velocities they scale. | 2026-10-16 |
| [D264](decisions/D264.md) | The barostat and the log take the pressure of the groups that the barostat scales, with the kinetic energy of their centers at the half steps [[Jung2020]](references.md#jung2020): on axis $a$, $P_a = (2K_{\text{half},g,a} + W_{g,aa} + C/3V)/V$, with $W_g = \sum_g\mathbf R_g\otimes\mathbf F_g$ the virial of the groups and $K_{\text{half},g,a} = \sum_g\tfrac{1}{2M_g}P_{g,a}^2 + \tfrac{\Delta t^2}{8}\sum_gF_{g,a}^2/M_g$. | 2026-10-16 |
| [D265](decisions/D265.md) | Langevin dynamics with constraints takes the g-BAOAB step of Leimkuhler and Matthews (2016) [[Leimkuhler2016]](references.md#leimkuhler2016), with one drift in each A ($K_r = 1$), and under Langevin dynamics the log's temperature is that of $K_\text{half}$, the kinetic energy of the half steps around each step. | 2026-10-16 |
| [D266](decisions/D266.md) | Particle mesh Ewald may take the prolate spheroidal wave function of order zero (PSWF) as its splitting and spreading kernel, "Ewald summation with prolates" (ESP), as an option beside the Gaussian splitting with B-splines, which stays the default | 2026-10-16 |
| [D267](decisions/D267.md) | The CPU target closes its gap to GROMACS in five steps, in this order, each its own PR: (1) LLVM's module pipeline on the host code, at O2 or O3 as the rate and the host compile time measured in a job decide, the level in the key of the compile cache; (2) PME on the host in parallel, the spreading by atomic adds to the fixed-point grid, so that a result is the same to the bit for any number of threads (D70), f32 in the mixed mode where the mixed mode of a device takes f32, and the energy and virial of the convolution only where a step writes them; (3) the builds of the list, whose structure on the CPU is decided once step 4 is measured; (4) the loop over pairs vectorized in the compiler's lowering, not in hand-written intrinsics, which supersedes the plan of #33 for its kernels (its MPI part stays on hold for M4); (5) the rest of the step. | 2026-10-16 |
| [D268](decisions/D268.md) | The pruning of the inner list of the groups (D114) stays global and exact; the loop over groups takes only the active units, the blocks that a pruning left with entries, listed by the pruning in the order of the blocks with their number on the device; the index of the blocks of each group is made at the build; the kernel of the frames of a pruning copies its configuration; a loop over pairs that follows a pruning at the same positions in f32 takes its frames. | 2026-10-16 |
| [D269](decisions/D269.md) | The host code of a program takes LLVM's module pipeline at O3 before the code generator, in `mdir run` (a transformer of MLIR's `ExecutionEngine`) and in a simulation (`JITEngine`); the level is in the key of the compile cache, whose hash of the module is taken before the pipeline, so that a hit costs neither the pipeline nor the code generation. | 2026-10-16 |
| [D270](decisions/D270.md) | Learned potentials enter MDIR through an optional plugin, `libmdir_metatomic` (`-DMDIR_ENABLE_METATOMIC`), a C++20 library with RTTI and exceptions that links libtorch and metatomic-torch and is reached from the core only through a versioned C ABI; ML1 evaluates the Allegro model `mir-group/Allegro-OAM-L:0.1` (CC-BY-4.0), made a metatomic artifact by `scripts/mlip/nequip-to-metatomic.py`, against the `nequip` package's ASE calculator (`NequIPCalculator`), evaluating the Allegro model, and the device memory per pair is its primary measure. | 2026-10-16 |
| [D271](decisions/D271.md) | Particle mesh Ewald on the host runs in parallel loops of the compiled code, and its result is the same to the bit for any number of threads, in every mode (#359, step 2 of D267). | 2026-10-16 |
| [D272](decisions/D272.md) | `[minimize]` and stochastic cell rescaling take a term of `[[energy.model]]`: the model's call follows every evaluation of the minimizer and of the barostat (the exact work in the scaled cell, and the start of a continued run that scales every step), its energy and virial enter as those of any other term, and the plugin builds its pair list from the cell of each evaluation. | 2026-10-16 |
| [D273](decisions/D273.md) | A second backend of the plugin of D270, `backend = "AOTI"`, loads an AOTInductor package of the `nequip` package's `ase` target (`.nequip.pt2`) with `torch::inductor::AOTIModelPackageLoader` and runs it on MDIR's stream with the plugin's own pair list (edges, cell shifts, and cell in float64, as the package takes them whatever `model_dtype` is); the forces and the stress are the package's own (the `nequip` package's `ForceStressOutput`, inside the compiled graph), the virial $\mathsf W = -V\sigma$; chunks of centers pass the pairs of the chunk and sum `atomic_energy` over its centers, for a package that declares itself local (`mdir::local_centers`, written by `scripts/mlip/nequip-to-aoti.py` for Allegro); a package without kernels for the device (`AOTI_COMPUTE_CAPABILITY`, or the list `mdir::archs` of a package of several architectures) is refused. | 2026-10-16 |
| [D274](decisions/D274.md) | Steepest descent ends at the first row of the log whose next step has shrunk below a floor: $10^{-6}$ of the cap of 0.1 Å below double precision ($10^{-7}$ Å) and $10^{-10}$ of it in double ($10^{-11}$ Å), with a line at the end that gives the step, the floor, and the largest force; the exit status is 0, and the checkpoint is written at that row. | 2026-10-16 |
| [D275](decisions/D275.md) | Distribution is a dialect, `md_dist`, whose values are global views of distributed fields: ownership is an SSA value (a layout), a halo is a value built from it with its reach in the type, a ghost value records the owned value it copies, and the owner-to-ghost copy $G$ (`halo_copy`) and the sum of the ghosts' contributions at the owner, $\mathbf a + G^\mathsf T\mathbf b$ (`halo_sum`), are ops declared to each other as adjoints. | 2026-10-17 |
| [D276](decisions/D276.md) | A global potential runs on domains through the compiler: `md-distribute` rewrites its body into `md_dist` before `md-differentiate`, which carries the adjoints of the transfers back by transposes declared through an op interface (the transpose of $G$ is $G^\mathsf T$, of $G^\mathsf T$ is $G$), and the directed traversal is an optimization of the derivative's form, `md-dist-owner-computes`, with a switch. | 2026-10-17 |
| [D277](decisions/D277.md) | The pair list of a term of `[[energy.model]]` is a value of the program, `md.edges` of an `md.neighborhood` at the model's cutoff (`!md.edges`, the directed pairs of every image within the cutoff, grouped by center in the order of the particles), built on the device from the rows of MDIR's neighbor matrix with the skin of the run: each build counts and writes the candidates of each particle, every image of each entry within the reach (so cells narrower than the cutoff need no ghosts), in slots fixed until the next build, and each step writes those within the cutoff, in f64, to the front of each particle's slots; the plugin takes it through version 2 of the C boundary, by default exactly (one copy of the counts per call, then a gather on the device), or padded (every slot, no wait, `MDIR_MLIP_PAIRS=padded`), and its own search of D270 stays behind `MDIR_MLIP_PAIRS=search`. | 2026-10-17 |
| [D278](decisions/D278.md) | A constraint failure in dynamics stops the run, as a position that is not a number does: after the constraints of the positions, each set (SETTLE, then each set of SHAKE) is tested by a loop of its own that tells whether a constrained length is off its target, $\lvert r^2/d^2 - 1\rvert > 10^{-4}$ below double precision and $10^{-8}$ in double, or is not a number (the test of D259); the host reads the tests after the step's evaluation and stops with the step and the set (`settles`, `shake1` to `shake3`). In a step with constraints a build that finds positions that are not numbers records its stop until the tests are read, and reports it only if every set passed. | 2026-10-17 |
| [D279](decisions/D279.md) | The derivative of a potential on domains goes through the fields it computes from the positions: `md-differentiate` keeps the adjoint of each such field on domains, carries it back through maps, sums over particles, $G$, and $G^\mathsf T$ by their rules, and a gather of a density gives its forces through a copy of the density's adjoint to the ghosts, of exactly the version computed in the derivative, and a two-ended gather of a kernel antisymmetric by construction. | 2026-10-17 |
| [D280](decisions/D280.md) | The direct sum of the prolate kernel in a loop over pairs is a function of $r^2$ that a table takes as it takes the Gaussian's (D94): $f q_i q_j (1 - \Phi(x))/r = f q_i q_j (1 - x) \bar\chi(x)/r$, with $\bar\chi(x) = (1 - \Phi(x))/(1 - x)$, the mean of $\chi$ over $[x, 1]$, a Chebyshev series in $2x - 1$, $x = r/r_c$ (`md.chebyshev`, a new op of the md dialect whose derivative is the series of the derivatives), which `md-exec-simplify-distance` takes into a function of $r^2$ and `md-exec-expand-radial` makes a table of cubics in f32, under `fast_math`, in kernels of f32. | 2026-10-17 |
| [D281](decisions/D281.md) | The skins of the neighbor lists are chosen by the run where they are `"AUTO"`, which a key not given is: the defaults of D261, the choice an earlier run of the same key cached, or, for a long run of `mdir run` with the groups on a GPU outside the deterministic mode, candidates of a cost model measured on the system, each compiled anew, of which the fastest is kept unless it is within the noise of the defaults. | 2026-10-18 |
| [D282](decisions/D282.md) | On a device, the reciprocal sums of PME may spread their charges through tiles of the memory of a block, one for each bin of $8^3$ points of the grid, into which the particles are binned by their first point at each evaluation, as cuFINUFFT spreads [[Shih2021]](references.md#shih2021); `md_exec.reciprocal ... tiles(E)` states the choice, which `md-exec-choose-spreading` makes: every sum in f32 on a grid of at least 500,000 points is tiled; a smaller grid, a grid in f64, and the deterministic mode spread as before (the bricks of D108 for the B-splines of order 4 in f32). `MDIR_PME_SPREAD` overrides the choice. | 2026-10-18 |
| [D283](decisions/D283.md) | On a device, a kernel of loops over tuples evaluated once adds what the tuples of a block give each particle in a table in block memory, and each particle of the table to the destinations once a block. | 2026-10-18 |
| [D284](decisions/D284.md) | Where a refresh of a dual list of groups prunes, the first loop over pairs that takes its result, at the same positions and in the same cell, prunes for it in its own kernel, a warp for each group, from 4096 groups of the last build (20,480 on `sm_12x`): the warp prunes its group as the refresh did, waits for the stores of its lanes, and computes the pairs of the entries it kept as one stream of the group; the kernel of the active units follows. The refresh keeps the kernel of the frames. | 2026-10-18 |
| [D285](decisions/D285.md) | The tiles of PME (D282) and the tables of the tuple kernel (D283) are two strategies of one privatized scatter-accumulate: the attribute `#md_exec.scatter` states the strategy of an op, the pass `md-exec-choose-scatter` chooses it, and the pass and the lowerings check each precondition with the same functions (`Scatter.h`). | 2026-10-18 |
| [D286](decisions/D286.md) | The baselines of ML4's staged models (#377, steps 1 and 2) are two pretrained checkpoints run as one whole-system call of the model term (`backend = "METATOMIC"`, D270, with MDIR's pair list, D277): nablaDFT 2.0's PaiNN of the split `train_large` (SchNetPack 2; MIT; 6 layers, 5 Å, 128 features, float32, energy in Hartree) and MACE-OFF23 medium (mace-torch; Academic Software License, academic and non-commercial; 2 layers, 5 Å, 128x0e+128x1o, float64), each pinned by an immutable URL and its SHA-256 in `scripts/mlip/fetch-ml4-artifacts.py`, with MACE-MP-0 medium (MIT) in its list as the alternative; weights and artifacts are never committed or redistributed. `scripts/mlip/painn-to-metatomic.py` and `mace-to-metatomic.py` make metatomic artifacts of them without changing a weight, and declare no `mdir::local_centers`: a message-passing atom energy depends on atoms within $L\,r_c$, so the whole system goes in one call. | 2026-10-18 |
| [D287](decisions/D287.md) | M3, learned potentials on one GPU, is ML1 steps 1 to 3, the IR of ML4, PaiNN in stages, and MACE in stages; M4, distributed execution, is ML2, ML3, PaiNN and MACE on two GPUs (an expanded halo against per-layer exchange), ML5, and distributed particle mesh Ewald. A milestone orders completion, not the start of work. StableHLO is a form of the bodies of the neural stages, beside AOTInductor. | 2026-10-19 |
| [D288](decisions/D288.md) | A neural stage is a symbol of the module, `md.stage @s forward(@f) vjp(@b) : (...) -> (...)`, a pure function of one element (one directed pair, one particle) whose bodies, its forward and its vector-Jacobian product, are functions that the symbols name; `md.apply_stage @s(...)`, or `md.apply_stage @s vjp(...)`, applies it in a kernel of `md.gather_relation`, `md.map_particles`, or `md.sum_particles`, and `md-differentiate` carries the adjoints of the features back through the stages by their products, as rules of the structure: the adjoint of a gather over a relation is a gather over the same relation that evaluates the product from both ends of each pair, and of a sum or a map a map. | 2026-10-19 |
| [D289](decisions/D289.md) | A field of features is an `!md.field` with a feature dimension: any number of components, `!md.field<@atoms, 16 x f64>`, and the representation of O(3) that they carry in the type, `!md.field<@atoms, irreps<4x0e + 4x1o>, f64>`, in e3nn's notation [[Geiger2022]](references.md#geiger2022), blocks in the order written and, within a block, `mul_ir` (the copies of each irreducible representation together, e3nn's and MACE's layout; the default) or `ir_mul` (the copies of each component together, SchNetPack's PaiNN), with the components of e3nn 0.4's real spherical harmonics, $(x, y, z)$ for $l = 1$. | 2026-10-19 |
| [D290](decisions/D290.md) | A domain's work runs through `md_exec`: `--convert-md-dist-to-md-exec`, the second executor of `md_dist`, gives the loops of each domain to `md_exec` over the domain's local rows and keeps the transfers of the reference executor, so that the CPU path and the device path per domain are the existing ones; `md_exec.build_neighbors`, `refresh_neighbors`, and `pair_for` take a center extent, `centers(%n)`, apart from the rows of the set, and `images(explicit)`, plain displacements for rows that carry their images; and `pair_for` takes the policy `(directed, atomic)`, the kernel's value at the center and $\sigma$ times it at the neighbor, both added atomically. | 2026-10-19 |
| [D291](decisions/D291.md) | A scatter-accumulate of its own in `md_exec`, `md_exec.scatter_add %values at %indices into %dest` with `overwrite(...)` and `scatter(...)`, for values that no loop computes in registers; the destinations of a loop over pairs that adds at both ends of a pair take `scatter(...)` as those of a loop over tuples do; and the table of a block, `hash`, is one emitter that every site calls. | 2026-10-19 |
| [D292](decisions/D292.md) | Closure under transposition is a design goal of MDIR: every part is written as a composition of a small set of primitives (map; gather and scatter-accumulate; halo copy $G$ and halo sum $G^\mathsf T$; reduction and broadcast; scan and reverse scan; permutation and its inverse; compaction and expansion; a linear transform with a declared adjoint, the FFT first, whose adjoint is the unnormalized backward transform (the maintainer's ruling on #452, 2026-10-19); the structural sorts and builds, which have no transpose), each with its transpose in the set and its stated properties (associative, commutative, exact, stable, ordered), and [primitives-map.md](primitives-map.md) maps every part to its primitives, the properties it relies on, its strategies with their measurements, and the state of its code. | 2026-10-19 |
| [D293](decisions/D293.md) | A stage's body may be batched: a function of `linalg` on tensors with a static leading dimension, the rows of a tile, which may compute in f32 where the declaration says f64. StableHLO is legalized to `linalg` when a model is exported (StableHLO's `stablehlo-legalize-to-linalg` in `stablehlo-opt`, pinned in `scripts/mlip/painn-stages.py`), so that MDIR's build takes no dependency on StableHLO. `md-inline` leaves the applications of batched stages in the kernels; `md-exec-lower-stage-bodies` lowers the bodies to functions of buffers (elementwise fusion, one-shot bufferization, a kernel for each parallel loop on a device, the bodies' buffers from the runtime's pool); `convert-md-exec-to-gpu` runs a loop whose kernel applies them as a host loop over tiles: for each level of the applications, a kernel packs the operands of the tile's rows and the bodies are called; a last kernel unpacks the results and runs the rest of the kernel. The rows of a loop over pairs are the slots of the pairs of images (D277), and a center adds up its rows in one thread: a gather, without atomics. Wide values (features) go component by component through the kernels. | 2026-10-19 |
| [D294](decisions/D294.md) | nablaDFT's PaiNN (D286) in stages: per layer a context map per atom, an edge gather of the context and the features, and a node map, with an embedding map and a readout sum, the features `irreps<128x0e + 128x1o, ir_mul>`; its stages are re-implemented in JAX from SchNetPack's weights and checked equal to SchNetPack (each stage, and the composed energy, forces, and virial against its calculator in float64), and exported by `scripts/mlip/painn-stages.py` as one MLIR module of the declarations, the bodies, and the potential of the positions, the cell, and the atomic numbers. `[[energy.model]]` and `mdir.ModelTerm` take it with `backend = "STAGES"` and `path` naming the module: a potential of the program, added to the energy and differentiated by `md-differentiate`, on a GPU in a periodic cell over a neighbor matrix (AUTO chooses it), without `centers_per_call`, `[free_energy]`, tunables, or a model of the plugin beside it; the rows of a tile are fixed when the model is exported. | 2026-10-19 |
| [D295](decisions/D295.md) | The ops of `md` with a kernel declare the transpose of their structure through `AdjointOpInterface`, and `md-differentiate` takes it from them instead of building the op itself; a check, `--md-check-transposes`, refuses an op of the structure of a potential without one. | 2026-10-19 |
| [D296](decisions/D296.md) | Compaction, scans, and counting sorts are one device emitter each, `CompactionEmitter.h` (`kernels::`, beside `ScatterEmitter.h`), which the lowering calls and the templates call by name, `func.call @mdir.primitive.<name>(...)` with a declaration in the template, expanded by the emitter where the call stands when the lowering parses the template (`expandPrimitiveCalls`; the pass `expand-device-primitives` for the tests). | 2026-10-19 |
| [D297](decisions/D297.md) | The loops of `md_dist` declare the transpose of their structure through `AdjointOpInterface`, as the ops of `md` do (D295), and the cotangent of `createTranspose` becomes a range. | 2026-10-19 |
| [D298](decisions/D298.md) | The spreading of the force on a virtual site can be derived as the transpose of its placement: with `MDIR_SITE_SPREAD=derived` the builder writes, for each set of sites, a potential $\sum_s \mathbf f_s\cdot\mathbf p_s$ of the force on each site against its place from its owner, and `md-differentiate` gives its forces through the declared transpose of `md.sum_tuples` (D295) and the reverse derivative of the placement; `written`, the transpose written out (`emitSpreadSites`), stays the default. | 2026-10-19 |
| [D299](decisions/D299.md) | `md-exec-lower-stage-bodies` fuses a batched body before it bufferizes it: the unit dimensions are folded, a pad of a constant becomes an elementwise gather of its source, a reduction from a `linalg.fill` of a constant starts from that constant at its first reduced iteration, other fills become generics, an elementwise producer is fused into a consumer that reads it through a permutation of its loops (or, cheap, into a consumer without reductions), a producer of several uses by evaluating it again in each such consumer, never into a contraction; and each result is written in the caller's buffer. The option `fuse=false` keeps D293's lowering. | 2026-10-19 |
| [D300](decisions/D300.md) | In a loop over pairs in tiles of rows (D293), the sum of a tile's rows into their centers runs over the rows of the tile, not over the particles: the row of a center's first slot in the tile leads the center's sum, and a wide destination is added up by a thread for each row and component. The sums and their order are D293's. | 2026-10-19 |
| [D301](decisions/D301.md) | The algebra of every sum is a fact of the IR, a licence to the lowering about the order of evaluation, and the chooser and the lowerings read it instead of the mode. | 2026-10-20 |
| [D302](decisions/D302.md) | On a device, `md-exec-lower-stage-bodies` lowers every op of a batched body of the shape of a GEMM in f32 to a call of cuBLAS's `cublasSgemm` through the runtime (`mdrtGemmF32`), on the stream of the kernels and in their order, in cuBLAS's default math mode (no TF32), in both precision modes: one reduced loop, the result's last dimension the columns and the others its rows (several are one), the operands read by the rows and the reduced loop and by the reduced loop and the columns, either of them transposed (A only with a single rows dimension), a payload of multiply and add, buffers of contiguous rows. A product that starts from a fill of zero calls with beta 0 instead of the fill's kernel; the weights that a product reads are copied to the device by the runtime when first read (`mdrtDeviceConstant`), each copy owned by the activation that made it (or by none outside one) and freed with it: when the activation closes, when a program unloads its modules (for those made outside an activation), and, with the handle of cuBLAS, when the device changes; a host address names the weights only while their program is loaded (the review of #472). The option `contractions=gemm\|loops` of the pass, `gemm` by default, and `MDIR_STAGE_CONTRACTIONS` choose the strategy (#444); `mdrt_cuda` links cuBLAS. | 2026-10-20 |
| [D303](decisions/D303.md) | On a device, `md-exec-lower-stage-bodies` lowers each sum of rows of a batched body to a kernel of a warp for each element of the result: a generic of one result in f32 or f64 whose one reduced loop has at least 32 iterations and runs along the contiguous dimension of every operand it indexes, and whose payload yields the accumulator (or a constant at the first reduced iteration) plus a term that does not read it. Lane $l$ adds the terms $l, l+32, \dots$ in order, the warp combines its lanes by D301's butterfly (`kernels::emitWarpReduce`), and the first lane stores the start plus the sum, in the type of the result. A dot product of rows is such a sum, not a GEMM. | 2026-10-20 |
| [D304](decisions/D304.md) | `[pme] influence = "P3M"` (Python `PMEInfluence.P3M`) takes eq. 2.28 of [[Ballenegger2012]](references.md#ballenegger2012), the influence function that minimizes the error of the forces of analytic differentiation, with the Gaussian and B-splines, the sums over the aliases at $\max_a\lvert J_a\rvert \le 2$; opt-in, `"SPME"` stays the default; a rectangular cell only (the maintainer's ruling on PR #474; a triclinic one is #475). | 2026-10-20 |
| [D305](decisions/D305.md) | The two-program staged-PaiNN GPU test pins its independent JAX reference to the CPU backend and compares energies within 0.013 kJ/mol for its fixed synthetic weights and water box. | 2026-10-21 |
| [D306](decisions/D306.md) | A linear transform is an execution primitive with its adjoint and pairing declared in the IR. | 2026-10-21 |
| [D307](decisions/D307.md) | `md-distribute` selects one static Cartesian factor grid for `md_dist.decomposition` from `domains=N box=Lx,Ly,Lz`, or takes the explicit `grid=gx,gy,gz` baseline. | 2026-10-21 |
| [D308](decisions/D308.md) | A reusable proof records the SSA dependencies of each `md_dist.halo_copy` and `md_dist.halo_sum` before physical lowering | 2026-10-21 |
| [D309](decisions/D309.md) | A full-suite validation on Slurm may use two disjoint lit shards on independent build trees of one commit when two GPU allocations are available. | 2026-10-21 |
| [D310](decisions/D310.md) | The physical lowering of `md_dist.halo_copy` and `md_dist.halo_sum` may expose a `mdrt.halo_copy_start` or `mdrt.halo_sum_start` event and a matching `mdrt.halo_wait` that yields the completed destination buffer version. | 2026-10-21 |
| [D311](decisions/D311.md) | `MDIR_DEVICE_MEMORY=2` breaks the device memory of a run down by where it goes | 2026-10-21 |
| [D312](decisions/D312.md) | The device pool does not grow with the number of builds: a loop over the groups gathers its positions and fields (and a pruning check the frames of its places) into buffers of the room of the structure's places, the length of its `order` buffer, not of the number of places that the last build occupied; the kernels still run over the occupied places. A buffer of the runtime's neighbor structures that a growth replaces (the groups' buffers and `blockOf`, the rows of a matrix, the slots of the pairs of images) goes back to the driver, not to the pool. The pool still reuses a block only for a request of the same size. | 2026-10-21 |
| [D313](decisions/D313.md) | `md_dist.migrate` creates a new ownership epoch from current owned positions, and `md_dist.redistribute` carries a complete owned field to that epoch by global particle identity. | 2026-10-21 |
| [D314](decisions/D314.md) | The incidence structure of a tuple set is stored in ragged rows (CSR [[Saad2003]](references.md#saad2003)): one buffer of i32 holding the number of rows $N$, the offsets $o_0 = 0 \le \dots \le o_N = M$ in entries, the $M$ entries of $a + 2$ values (the tuple, the place of the particle in it, its members), and one entry of zeros; row $i$ is the entries $o_i$ to $o_{i+1}$, $N + 2 + (a + 2)(M + 1)$ elements in all. The storage type of `!mdrt.incidence` is `memref<?xi32>`. Code reads a row only through one accessor (`include/mdir/Conversion/RaggedRows.h`: `emitRaggedRow`, `emitRaggedEntry`), which the templates of the neighbor builds call as the device primitives `ragged_row` and `ragged_entry`. On a device the structure is built from the members by the primitives of D296 (`emitDeviceRaggedRows`): a count of the memberships of each place with a device atomic, the device scan, a cursor scatter, and an insertion sort of each row; the index that the host built for each tuple set and uploaded (D257) is gone. The CPU takes the same layout. | 2026-10-21 |
| [D315](decisions/D315.md) | The reciprocal sums of a function share their scratch: `md-exec-assign-storage` allocates each buffer of the scratch of `md_exec.reciprocal` once in the body of the function, and every later sum of the function that needs a buffer of the same shape takes it. The shape is the type, the formula of the length with its count (and the number of particles, for the places and the bins), and whether the buffer holds zeros between evaluations (the bricks of D108, shared only with bricks and set to zero once). No sum takes one buffer twice. | 2026-10-22 |
| [D316](decisions/D316.md) | The most neighbors of a particle at the start, from which the driver sizes the rows of a neighbor structure without `neighbor_capacity` (D227), are counted exactly by cells, over the threads of the process, in place of the serial search over particles; the count is the same integer, so the width of the lowered program does not change. | 2026-10-22 |
| [D317](decisions/D317.md) | A semantic partition of the owned centers of one neighborhood relation identifies exactly the rows whose relation pairs read no ghost neighbor. | 2026-10-22 |
| [D318](decisions/D318.md) | A short, static sum in a batched neural stage body can opt into one GPU thread per result, accumulating its 2 to 31 terms in a register in source order and fusing an eligible adjacent pointwise operation and duplicate result write; the loop and separate pointwise launches remain the default. | 2026-10-22 |
| [D319](decisions/D319.md) | The pure semantic `md.linear_transform` describes an unnormalized three-dimensional real FFT over a static grid and its packed Hermitian half-spectrum as SSA tensor values. | 2026-10-22 |
| [D320](decisions/D320.md) | A tiled reciprocal sum allocates bins and no unused spreading bricks; its order-4 f32 weighted gathering is selected for tiles even without bricks, and for a non-tiled sum with bricks. | 2026-10-22 |
| [D321](decisions/D321.md) | The group neighbor build may use an opt-in single-pass integer scan with decoupled look-back, while its three-launch scan remains the default. | 2026-10-22 |
| [D322](decisions/D322.md) | Assigning a decision number also updates links to Markdown headings whose GitHub anchors contain its label. | 2026-10-22 |
| [D323](decisions/D323.md) | Read common finite decimal `.gro` coordinate and velocity fields with direct binary64 conversion; retain the general LLVM reader when it cannot consume a complete finite field. | 2026-10-22 |
| [D324](decisions/D324.md) | The default Python `observe` tests run representative precision and modifier combinations, while release tests run their full matrix | 2026-10-22 |
| [D325](decisions/D325.md) | The Gaussian and prolate PME gathers use shared scalar emitters for the wrapped z coordinate and the grid read. | 2026-10-22 |
| [D326](decisions/D326.md) | A design decision; nothing is built yet: the couplings become ops of `dyn`, written in the step program and lowered to what the driver writes today. | 2026-10-22 |
| [D327](decisions/D327.md) | The Amber suite is measured with MDIR and pmemd.cuda under one protocol, that of each system's `mdin.GPU`: the same steps (`nstlim`), the same output (energies every `ntpr` steps, positions every `ntwx`, restarts or checkpoints every `ntwr`), and the rate over the same window, from the first report to the last step; MDIR with the searched and with the default reaches; and the whole wall time of each run. | 2026-10-22 |
| [D328](decisions/D328.md) | A distributed pair traversal may select a proved interior or boundary center set from its exact relation. | 2026-10-22 |
| [D329](decisions/D329.md) | Offer opt-in f64 storage for neural stage feature fields and their adjoints in mixed precision while preserving the f32-field default and the exported bodies' declared f32 arithmetic. | 2026-10-22 |
| [D330](decisions/D330.md) | On a device, `md-exec-assign-storage` gives a neighbor structure the storage of an earlier structure of the same shape that nothing uses any more, with a `md_exec.reset_neighbors` where the later one begins. The shape is the type of the structure, its kind and width, the number of particles, and the type of the positions. In every program of dynamics, the structure of the evaluation at the start and the structure of the steps now share one storage. The steps still begin with a structure of their own in the sense of D218 and build at step 1. | 2026-10-22 |
| [D331](decisions/D331.md) | On a device, a structure of groups of neighbors starts with one block of 64 entries a group, and its first build sizes the pool of blocks: the build counts the blocks it takes even where they do not fit, and the existing growth (`mdrtGroupsGrow`) gives the pool a quarter more than that count, after which the build is made again. A later build that needs more grows the pool the same way, never leaving out a pair, and `mdir run` logs the growths after the first build of a structure. | 2026-10-22 |
| [D332](decisions/D332.md) | `md_dist.covered` makes the skin test for a distributed halo a collective predicate over every owned identity in its ownership epoch. | 2026-10-22 |
| [D333](decisions/D333.md) | Use a matched, reproducible NVE CPU baseline before selecting host primitive strategies | 2026-10-22 |
| [D334](decisions/D334.md) | Verified execution uses one shared contract for provenance, effects, validity, and state-specific witnesses across `md`, `dyn`, `md_exec`, and `md_dist`. | 2026-10-22 |
| [D335](decisions/D335.md) | A checkpoint of `mdir run` keeps ending its segment: the next segment begins in a new order of the particles and builds its structures anew, on every target and in every mode. Keeping the order and the structures across a checkpoint was measured and not taken. | 2026-10-22 |
| [D336](decisions/D336.md) | `mdir run` writes each checkpoint on a thread of its own while the steps go on: the call of the host copies the state as the program holds it and the rest of the file, and the thread orders the state into the order of the input, hashes it, and writes, synchronizes, and renames the file. The next checkpoint, a stop on a signal or the wall time, the end of the run, and the exit of the process wait for the write; a write that fails stops the run at the next wait, with its error and the status 1. | 2026-10-22 |
| [D337](decisions/D337.md) | Storage-form field, loop-reduction scratch, tuple-member, and incidence buffers carry their row set in an identity memref layout. | 2026-10-25 |
| [D338](decisions/D338.md) | The verified execution design binds requirements and evidence to exact subjects, versions, effects, and numerical policy across `md`, `dyn`, `md_exec`, and `md_dist`. | 2026-10-25 |
| [D339](decisions/D339.md) | The `water-box.py` generator writes `.gro` residue numbers modulo 100,000, as it already writes atom numbers. | 2026-10-25 |
| [D340](decisions/D340.md) | A device primitive has one operation and correctness contract, with backend-specific instruction emission and a single target record that separates capabilities from measured strategy policy. | 2026-10-25 |
| [D341](decisions/D341.md) | Bridge semantic FFT values to execution buffers with private storage for each result and each backward input. | 2026-10-25 |
| [D342](decisions/D342.md) | Execution loops expose a derived, conservative access interface. | 2026-10-25 |
| [D343](decisions/D343.md) | Tuple coordinate kinds and member places are read and extended through their declared operation accessors. | 2026-10-26 |
| [D344](decisions/D344.md) | The shards of a suite are taken in the lexical order of the tests (`--order=lexical`), and `lit.cfg.py` refuses a shard without it; the check of the expected unsupported tests compares a shard's own tests. | 2026-10-25 |
| [D345](decisions/D345.md) | An external buffer acquires a storage set only at a verified input cast. | 2026-10-25 |
| [D346](decisions/D346.md) | The readers of the tunables read a snapshot that a successful operation publishes whole, and an export of a buffer takes its own lease before it releases the GIL | 2026-10-25 |
| [D347](decisions/D347.md) | **An integer of a fingerprint is written exactly, and the text of an integer below $2^{53}$ in magnitude does not change** (#620). | 2026-10-25 |
| [D348](decisions/D348.md) | The registry of the device's blocks grows with them, the idle blocks of the pool and the cached plans of cuFFT are bounded in bytes, a request that the driver refuses for want of memory is made again after they are given back, and an embedding program may give them back itself | 2026-10-25 |
| [D349](decisions/D349.md) | The module's checked storage form has one declared marker contract. | 2026-10-26 |
| [D350](decisions/D350.md) | The compiler and runtime share one versioned neighbor-group buffer slot contract. | 2026-10-26 |
| [D351](decisions/D351.md) | A batched stage body retains a verified static row declaration through bufferization. | 2026-10-26 |
| [D352](decisions/D352.md) | The module of a program is the builder's text parsed and printed again by MLIR's printer, with the locations of its ops, and `mdir emit --stage=module` and `Program.ir` print it without the positions in the builder's text | 2026-10-26 |
| [D353](decisions/D353.md) | An outlined batched body is identified by a checked function marker. | 2026-10-26 |
| [D354](decisions/D354.md) | A kernel is named after its function, the op of the loop it comes from, what a fused kernel holds, and its ordinal in the lowering, not after the line of that op in the text of the module | 2026-10-26 |
| [D355](decisions/D355.md) | A `dyn.program` states the effects of its body in `effects`, which its verifier checks against the body and `dyn.step` takes; the functions of the host declare theirs in `dyn.effects`. | 2026-10-26 |
| [D356](decisions/D356.md) | A NEP4 model of GPUMD is evaluated as the shape of EAM: one gather over the pairs into a field of radial descriptors and angular moments, then a neural stage of each atom for the invariants and the network, with forces and virial from `md-differentiate`; no triplet loop and no new op. | 2026-10-26 |
| [D357](decisions/D357.md) | A pair loop must prove its required neighbor cutoff, exact centers, and image convention through SSA provenance at recognized boundaries; unresolved proofs are rejected. | 2026-10-27 |
| [D358](decisions/D358.md) | Host-device copies have checked completion dependencies and source/destination borrows that forbid a dependent read, conflicting write, free, or reuse while the copy is in flight. | 2026-10-27 |
| [D359](decisions/D359.md) | An explicit registry of term builders owns the dispatch for each migrated kind of energy term, beginning with external terms. | 2026-10-27 |
| [D360](decisions/D360.md) | Keep a bounded semantic FFT component resident on the device with checked layout, private backward inputs, and scoped host-boundary completion. | 2026-10-27 |
<!-- end decisions-table -->






### 5.1 Amendments to earlier decisions

A1 to A10 come from an external review of revision 3 of the architecture,
A13 from one of revision 4.

| # | Amendment | Amends |
|---|---|---|
| A1 | **`md` expresses the potential energy $U(\mathbf x; \theta)$, not a Hamiltonian.** The op is `md.potential`. \"Hamiltonian\" means $K + U$ and is used only where that is meant, as in Hamiltonian replica exchange. | Architecture wording |
| A2 | **A relation is a set of tuples with an arity and an orientation.** The logical relation is distinct from its physical traversal. A directed traversal of an unordered relation sums with weight one half. | New |
| A3 | **`LocalityInterface` is renamed `ParticleDependencyInterface`.** Any semantic computation that needs data of other particles implements it: potentials, `mlff`, pairwise thermostats, constraints, virtual sites, collective variables. | D4 |
| A4 | **The plan is split in two.** `ExecutionPlan` holds structural decisions; it is immutable, hashable, and cacheable. `ExecutionTuningState` holds numeric parameters and measured costs; it is updated only at segment boundaries or other declared safe points. A structural change means replanning and possibly compiling a new variant. | D9, P7 |
| A5 | **The random number key is seed, step, stream ID, and entity key.** The entity key is a particle global ID, a canonical pair of global IDs, a fixed key for global moves, or a replica ID. | P5 |
| A6 | **Decomposition-independent reproducibility is a future mode.** It is not required for M0 to M3. | P6 |
| A7 | **`ensemble.state` replaces `thermo.state`.** There is one dialect. | Architecture examples |
| A8 | **`md_dist` and `md_exec` are peer dialects.** Distributed lowering is performed first, so that `md_exec` lowering can specialize the regions that distribution creates. | Architecture wording |
| A9 | **`!mdrt.event` is an opaque runtime completion object.** It is not tied one-to-one to a transport primitive. Its implementation may be one or more MPI requests, a CUDA event, or an NVSHMEM signal. | D10 |
| A10 | **M2 has three parts.** M2a: constraints and virtual sites. M2b: PME on one node. M2c: distributed PME. | P3 *Amended by A14.* |
| A12 | **In the storage form, the ops of a block run in the order of the block.** The ops declare which buffers they read and write, so the upstream analyses keep that order where it matters. Event tokens (D10) are added when ops run asynchronously; until then nothing needs them. | D17 |
| A11 | **Rebuilds are checked after the fact.** With the fixed-interval policy, the maximum displacement since the previous rebuild is measured at each rebuild. Violations of the validity condition are counted and reported. *Amended by B1.* | P15 |
| A13 | **The random number key has a draw index**: seed, step, stream ID, entity key, and draw index. A stream ID names a purpose, such as a thermostat or a kind of move. The draw index counts the numbers that one entity takes from one stream in one step, from 0, and is a constant in the kernel that draws. The numbers then do not depend on the order in which kernels run or in which a kernel is evaluated. | P5, A5 |
| A14 | **Constraints and particle mesh Ewald on one node move to M1** (D53). M2a keeps virtual sites; distributed particle mesh Ewald stays in M2c. | A10 |

### 5.2 Choices made in the M0 specification

S3 was withdrawn; B2 replaces it.

| # | Choice | Section of ops.md |
|---|---|---|
| S1 | There is no aggregate state type. The state is the set of loop-carried SSA values. | 2.3 |
| S2 | The state holds velocities, not momenta. | 2.4 |
| S4 | The IR carries plain numbers in one internal unit system, declared per module. | 2.6 |
| S5 | The cutoff of a neighborhood is a compile-time constant in M0. | 4.4 |
| S6 | Reductions and accumulation are clauses of the loop ops, not separate ops. | 8.1 |
| S7 | Parameter binding is not an attribute. A parameter is static when the caller passes a constant. | 4.3 |
| S8 | Precision is assigned per role. A mode is a default assignment of types to roles. | 7 |

### 5.3 Amendments made with draft 2 of the M0 specification

B1 to B3 and B5 to B10 come from an external review of draft 1.

| # | Item | Amends |
|---|---|---|
| B1 | **The default rebuild policy checks validity every step.** A fixed interval with no check must be selected explicitly, because a violation found at the next rebuild cannot be repaired. | P15, A11 |
| B2 | **`f64` is the reference precision of the semantic program.** It is ordinary floating point, not an abstract real. Lowering to single or mixed precision deliberately relaxes the numerical semantics. No separate real type is introduced. | S3 |
| B3 | **`exchange` is a semantic contract.** It applies to `md.sum_relation` and `md.gather_relation`. The compiler verifies it when it can prove it; otherwise the front end must assert it. | New |
| B4 | **Truncation is an attribute of a relation sum.** The kinds are `none`, `shift`, `force_shift` [[Stoddard1973]](references.md#stoddard1973), `switch`, and `force_switch`, the cubic polynomial added to the force of the GROMACS manual [[GromacsManual2025]](references.md#gromacsmanual2025), added for parity with GROMACS. A pass expands it into the kernel before differentiation. Energy conservation is validated with `force_shift` or `switch`. *Amended by D121:* `force_switch` is not the force switch of Steinbach and Brooks [[Steinbach1994]](references.md#steinbach1994), which CHARMM uses and which the driver writes into the Lennard-Jones of a topology as `POWER_FORCE_SWITCH`. | New |
| B5 | **Derivatives of functions that are not smooth have fixed conventions**, including the branch taken at a tie. | New |
| B6 | **Comparing velocity Verlet with leapfrog maps the initial velocities**: $\mathbf v(-\Delta t/2) = \mathbf v(0) - \tfrac12 \Delta t\, \mathbf F(0) / m$. | New |
| B7 | **Both relation kernels receive the distance and the displacement vector.** | New |
| B8 | **The virial is $\mathsf W = \sum \mathbf d_{ij} \otimes \mathbf K(i, j)$** [[Louwerse2006]](references.md#louwerse2006), positive for repulsion, with $P = (2 E_\text{kin} + \mathrm{tr}\,\mathsf W) / (3V)$. *Amended by D45, which says which kinetic energy the pressure takes.* | New |
| B9 | **Integrators declare `symplectic` and `time_reversible`.** They do not declare energy conservation. | D6 examples |
| B10 | **A value that is live across an overwrite gets its own buffer.** The criterion is liveness, not the number of consumers. D18 covers extra buffers as well as copies. | D18 |

### 5.4 Restart

| # | Decision |
|---|---|
| R1 | **Neighbor structures are rebuilt at the start of every segment.** A run that is restarted from a checkpoint then performs the same rebuilds as a run that was not interrupted, provided both use the same segment schedule. Without this rule the two runs sum forces in different orders. |

### 5.5 Validation

| # | Decision |
|---|---|
| V1 | **No reference interpreter.** Validation rests on three things instead: tests that run single kernels against closed-form values; tests that run whole programs against scripts that evaluate all pairs; and, from M1 on, comparison of energies and forces with an established MD engine. An interpreter is reconsidered if a class of errors shows up that these do not catch. This withdraws the interpreter from D23 and from the validation plan of the M0 specification. |

### 5.6 Keeping neighbor structures valid

D41 keeps one structure for the whole system and the criterion of
Section 8.2 of the M0 specification. These schemes were considered with
it.

| Scheme | Assessment | State |
|---|---|---|
| A flag that the threads of the integrator set, in place of a global maximum | No reduction, and no kernel for the test. The result is the same. | Adopted (D41) |
| The decision to build on the device, so that the host reads nothing in a step | The host reads four bytes in a step and waits for the device there. The device does not wait for the host, because the kernels of the step are queued. What could be saved is the copy, a few microseconds. | Not planned |
| A test every n steps, with a reserve in the skin | The reserve is an assumption about velocities. It is the policy `interval` with the diagnostic of A11. | The policy is planned; it stays off by default |
| Lists for pairs of cells, each with a test of its own, rebuilt where they fail | In a liquid the displacements are alike everywhere, and the gain is that of a maximum over fewer particles. By an estimate from the statistics of the largest of n displacements, which was not measured, that is nothing for a small system and at most a factor of two in the time between builds for a million particles. A row of a list is then valid since a time of its own, which a checkpoint would have to hold for a run to continue exactly (R1). Of interest for systems with a fast region, and with domain decomposition. | Open, for M2 |
| A second list that is built while the first is in use | On one device the build and the forces compete for the same device. The list in use needs a reserve in the skin. | Not planned |
| A prediction of the step in which the structure fails, from bounds on velocities | With the test in the drift, a step without a test saves nothing. | Not needed |
| A skin that is tuned during the run for the shortest time per step | The skin is a plan parameter that decides correctness in no way and speed much. The driver can measure the time of the forces and of the builds. | Open, for M1 |
| No list: a search of the cells in every evaluation of the forces | A search visits 27 cells, which hold 6 times the particles that a list holds for the same reach. Of interest for cutoffs that differ much between particles. | Not planned |

## 7. Not yet designed

These items follow from the decisions above but have no design yet.

| Item | Needed for | Status |
|---|---|---|
| `md` dialect: types, ops, truncation, differentiation, exchange check | M0 | Implemented |
| `dyn` dialect | M0 | Implemented |
| `md_exec` dialect in the value form, and the conversion into it | M0 | Implemented |
| Lowering of `md_exec` to executable code on the CPU | M0 | Implemented |
| Reuse of neighbor structures across steps, with the policy `check` | M0 | Implemented |
| Rebuild policy `interval`, with the diagnostic of A11 | M0 | Decided 2026-09-30: an option only, never the default. A list kept for a fixed number of steps with a buffer estimated from a tolerance of the energy drift, as GROMACS does [[Pall2020]](references.md#pall2020), may miss pairs within the cutoff; the default stays the exact test of D41 and D80. A run that chooses it must say so in its log and its checkpoint, with the tolerance. Not implemented. |
| Storage form of the `md_exec` ops, as D17 decided | M0 | Implemented: `md-exec-assign-storage` |
| Precision policy | M0 | Implemented: single, mixed, and double, on the CPU and on a GPU |
| Declaring the role of a field that a function takes | M0 | |
| Vectorization of loops over pairs across pairs | M0 | Without it, single precision is no faster than double on the CPU |
| Fusion of loops over the same neighbor structure | M0 | Implemented |
| Fusion of loops over particles | M0 | Implemented |
| The test of validity in the loop that writes the positions (D41) | M0 | Implemented: `md-exec-expose-validity` |
| Fusion of the loop over pairs with the kick that follows it | M0 | |
| The particles in the order of their positions (P17, D44) | M0 | Implemented: `md_exec.spatial_order`, `md_exec.permute`, and the keyword `spatial_order` |
| A skin that is tuned during the run | M1 | |
| Removal of the square root from kernels that do not need it | M0 | Implemented |
| Freeing of buffers | M0 | |
| Driver, TOML input, XTC and H5MD output | M0 | Driver, TOML input, and DCD output are implemented; see [driver.md](driver.md), Section 4 |
| Checkpoints in H5MD, restart from a checkpoint | M0 | Implemented; a run that continues is exact |
| Trajectory in XTC | M0 | |
| GPU back end | M0 | Implemented for NVIDIA: `convert-md-exec-to-gpu`, `libmdrt_cuda` |
| Global sums of vectors on a device, for the virial | M0 | Implemented. The driver writes the virial and the pressure to the log. |
| GPU back end for AMD | M1 | |
| Check that particle set symbols in types are declared | M0 | |
| Regression test for the numerical values of derivatives | M0 | Implemented with a kernel runner |
| `mdrt` ABI: storage, neighbor structures, events | M0 | Proposal in [runtime.md](runtime.md) |
| Schema of the TOML input | M0 | |
| Layout of the MDIR group inside an H5MD checkpoint | M0 | Decided (D40); see driver.md, Section 2.6 |
| HDF5 development files | M0 | Installed by `scripts/build-hdf5.sh` |
| Storage assignment pass | M0 | Specified in ops.md, Section 10 |
| Lowering of transcendental functions on GPU targets | M1 | Works on NVIDIA through `libdevice`; see ops.md, Section 3.3. Open for AMD. |
| Syntax for combining relations | M1 | |
| Streams and draw indices of the random numbers of a thermostat (A13) | M1 | |
| The deterministic level as a constraint on the plan | With the planner | Note: the level says what the planner may choose, not what it prefers. It excludes a sum whose order threads decide. The lowerings of M0 add up in a fixed order on a device. On the CPU the order of a reduction is that of the OpenMP runtime: the tests find the same bits from run to run with the same number of threads, which the OpenMP standard does not promise. |
| The dependency interface on domains other than particles | M2b | Note: the planner should reason about stages on a domain, of which the particles of a set are one and a mesh is another, so that PME does not need another planner. `ParticleDependencyInterface` (D4, A3) is not implemented yet. |
| Scatter strategy for bonded terms | M1 | Decided (D49) |
| Long-range dispersion correction | M1 | |
| Distributed fields and grids | M2c | |
