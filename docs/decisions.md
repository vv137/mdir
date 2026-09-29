# MDIR Decision Log

Last updated: 2026-09-29.

This log records what has been decided. Every entry is accepted unless it
is marked otherwise.
[architecture.md](architecture.md) describes the design that results.
[design-review.md](design-review.md) and [prior-art.md](prior-art.md) hold
the analysis that led to the decisions.

Identifiers are stable. C*n* are project constraints. D*n* and P*n* are
decisions; the P series was first recorded as proposals and accepted on
2026-09-29, and keeps its numbering so that existing references stay valid.

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

### D2. Differentiation is a semantic-level transformation

Forces, virial, and `dH/dλ` are produced by an `md`-to-`md` transformation
that runs before locality analysis and distributed planning. Its output is
again expressed in `md` and `mlff` ops.

- Closed-form pair and bonded terms: symbolic differentiation.
- `mlff`: reverse-mode, generating an explicit derivative graph.
- LLVM-level AD (Enzyme) is not the default path. It remains an option for
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

### D10. One dependency token

Ordering constraints are expressed with a single project-owned SSA token
type, `!mdrt.event`. It lowers to an MPI request, a CUDA event, or an NVSHMEM
signal depending on the back end. `md_exec.task` and `md_exec.depend` are
dropped.

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

### D18. No implicit copies in the step loop

The storage assignment pass must not insert a copy of a per-particle field
inside the step loop silently. When a copy is required, the compiler reports
it with the reason. A copy is legitimate only when one version of a field has
more than one consumer, as in a Metropolis rejection.

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
| P1 | **JIT binding times.** Structure is a compile-time constant: terms, functional forms, precision, target. Particle count, box, time step, temperature, and λ are run-time values. Force-field parameter tables may be made constant per parameter, opt-in. Compile once, distribute to all ranks, and cache by a hash of IR and target. | OpenMM kernel cache |
| P2 | **Full-shell halo first.** The communication transport is abstracted, with in-process, MPI, and NVSHMEM implementations, so that one `md_dist` serves workstations and clusters. | LAMMPS; GROMACS thread-MPI |
| P3 | **Milestones.** M0: Lennard-Jones fluid, NVE. M1: Martini CG membrane and water. M2: AA protein and water with PME and constraints. M3: MLFF. | — |
| P4 | **Compiled segments.** The JIT compiles `run_segment(state, n) -> state`, which contains the step loop, rebuild checks, and migration. The driver owns events between segments: output, checkpoints, replica exchange, Python callbacks. | OpenMM `step(n)`; HOOMD-blue `run(n)` |
| P5 | **Counter-based random numbers.** `dyn.random` is a pure function of seed, step, particle global ID, and stream ID. No generator state is carried in the simulation state. | HOOMD-blue (Random123) |
| P6 | **Reproducibility levels.** *Fast*: no guarantee. *Deterministic*: same binary, hardware, and decomposition give the same bits. *Decomposition-independent*: the same bits for any rank count. Bitwise agreement between different hardware is a non-goal. | GROMACS `-reprod`; OpenMM deterministic forces; Desmond and Anton fixed-point accumulation |
| P7 | **Plan in the IR.** The `ExecutionPlan` is serializable, user-overridable, and attached to the IR, so each lowering can be tested with a fixed plan. Structural parameters (cluster size, conflict strategy) require recompilation; numeric parameters (skin, rebuild interval, domain boundaries) are run-time values that can be tuned during the run. | GROMACS run-time tuning of list interval and load balance |
| P8 | **Exclusions as a first-class relation.** Nonbonded pairs are "within cutoff, minus excluded." Scaled 1-4 pairs are a separate topological relation with their own parameters. Terms may also sum over the exclusion relation, which Ewald exclusion corrections require. | OpenMM exceptions; GROMACS pair interactions |
| P9 | **Runtime primitives versus generated predicates.** Refinement of D8: generic parallel primitives (sort, scan, compaction) live in the runtime. MD-specific predicates and kernels (distance tests, exclusion filters, type-pair cutoffs) are generated. | — |
| P10 | **Trusted sampling metadata.** Refinement of D6: library-provided thermostat and barostat kinds carry metadata stating whether they preserve the target distribution. The verifier warns, for example, when replica exchange is combined with a thermostat that does not. | — |

P11 to P18 follow from the review of PPMD (Saunders et al. 2018). See
[prior-art.md](prior-art.md). D17 makes P11 concrete.

| # | Decision |
|---|---|
| P11 | **`md_exec` adopts the loop and access model of PPMD.** Core ops are a particle loop, a pair loop, and a reduction. With value semantics, access modes are the op signature: a field that is read is an operand, a field that is produced is a result, and a field that is updated is both. Explicit access descriptors are declared only for external, opaque kernels. For generated kernels they are derived and verified. |
| P12 | **Layer boundary wording.** `md`: what is computed. `md_exec`: over which set, with which access pattern. Back end: how. |
| P13 | **Baseline pair execution.** Pair terms are executed over directed pairs, writing only to the central particle. This removes scatter races, atomics, and reverse accumulation from the first back ends. Half-list execution is added later as a planner optimization. Scope: two-body terms only. Bonded terms (M1) require a scatter strategy and MLFF (M3) requires reverse accumulation, so both remain in the architecture. |
| P14 | **`md` has a generic relational core.** Particles, fields, relations, neighborhoods, and reductions are generic. A Hamiltonian is one kind of region built from them; analyses and collective variables are others. Whether a region can be differentiated depends on the ops it contains. |
| P15 | **Neighbor structure validity is explicit.** A neighbor structure is a value built from a reference configuration, with a validity condition. The default rebuild policy is a fixed interval with a buffer sized for that interval, which needs no per-step global reduction. A displacement check is optional. |
| P16 | **Version counters at the segment boundary.** Inside a compiled segment, halo exchange placement is decided statically. State modified from the host between segments is detected with run-time version counters. |
| P17 | **Particle identity is separate from storage index.** Topology, exclusions, and random number streams refer to global IDs. Particles are spatially reordered when neighbor structures are rebuilt. |
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

| # | Decision |
|---|---|
| D20 | **LLVM 23.1.2 is the pinned release.** It was the latest stable release on 2026-09-29. `scripts/build-llvm.sh` builds it. |
| D21 | **M0 integrators are velocity Verlet and leapfrog.** |
| D22 | **Energy expressions use the syntax of OpenMM custom forces.** This makes D16 concrete. |
| D23 | **Three precision modes are supported: single, mixed, and double.** The reference interpreter computes in double precision. |
| D24 | **The input format is TOML.** Its schema starts small and is expected to change. |
| D25 | **Trajectories are written as XTC.** XTC holds positions only, in reduced precision, so it does not serve as a checkpoint. |
| D26 | **Checkpoints are written as H5MD, in 64-bit floating point.** A checkpoint holds positions and velocities, together with everything else an exact restart needs. H5MD is an HDF5-based format with standard places for positions, velocities, periodic images, particle IDs, and the box, and it allows application-specific groups. |

### 5.1 Amendments to earlier decisions

A1 to A10 come from an external review of revision 3 of the architecture.

| # | Amendment | Amends |
|---|---|---|
| A1 | **`md` expresses the potential energy `U(x; θ)`, not a Hamiltonian.** The op is `md.potential`. "Hamiltonian" means `K + U` and is used only where that is meant, as in Hamiltonian replica exchange. | Architecture wording |
| A2 | **A relation is a set of tuples with an arity and an orientation.** The logical relation is distinct from its physical traversal. A directed traversal of an unordered relation sums with weight one half. | New |
| A3 | **`LocalityInterface` is renamed `ParticleDependencyInterface`.** Any semantic computation that needs data of other particles implements it: potentials, `mlff`, pairwise thermostats, constraints, virtual sites, collective variables. | D4 |
| A4 | **The plan is split in two.** `ExecutionPlan` holds structural decisions; it is immutable, hashable, and cacheable. `ExecutionTuningState` holds numeric parameters and measured costs; it is updated only at segment boundaries or other declared safe points. A structural change means replanning and possibly compiling a new variant. | D9, P7 |
| A5 | **The random number key is seed, step, stream ID, and entity key.** The entity key is a particle global ID, a canonical pair of global IDs, a fixed key for global moves, or a replica ID. | P5 |
| A6 | **Decomposition-independent reproducibility is a future mode.** It is not required for M0 to M3. | P6 |
| A7 | **`ensemble.state` replaces `thermo.state`.** There is one dialect. | Architecture examples |
| A8 | **`md_dist` and `md_exec` are peer dialects.** Distributed lowering is performed first, so that `md_exec` lowering can specialize the regions that distribution creates. | Architecture wording |
| A9 | **`!mdrt.event` is an opaque runtime completion object.** It is not tied one-to-one to a transport primitive. Its implementation may be one or more MPI requests, a CUDA event, or an NVSHMEM signal. | D10 |
| A10 | **M2 has three parts.** M2a: constraints and virtual sites. M2b: PME on one node. M2c: distributed PME. | P3 |
| A11 | **Rebuilds are checked after the fact.** With the fixed-interval policy, the maximum displacement since the previous rebuild is measured at each rebuild. Violations of the validity condition are counted and reported. | P15 |

### 5.2 Choices made in the M0 specification

S3 was withdrawn; B2 replaces it.

| # | Choice | Section of ops-m0.md |
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
| B4 | **Truncation is an attribute of a relation sum.** The kinds are `none`, `shift`, `force_shift`, `switch`, and `force_switch`; the last was added for parity with GROMACS. A pass expands it into the kernel before differentiation. Energy conservation is validated with `force_shift` or `switch`. | New |
| B5 | **Derivatives of functions that are not smooth have fixed conventions**, including the branch taken at a tie. | New |
| B6 | **Comparing velocity Verlet with leapfrog maps the initial velocities**: `v(−dt/2) = v(0) − (dt/2) · F(0) / m`. | New |
| B7 | **Both relation kernels receive the distance and the displacement vector.** | New |
| B8 | **The virial is `W = Σ d_ij ⊗ K(i, j)`**, positive for repulsion, with `P = (2 E_kin + tr W) / (3V)`. | New |
| B9 | **Integrators declare `symplectic` and `time_reversible`.** They do not declare energy conservation. | D6 examples |
| B10 | **A value that is live across an overwrite gets its own buffer.** The criterion is liveness, not the number of consumers. D18 covers extra buffers as well as copies. | D18 |

### 5.4 Restart

| # | Decision |
|---|---|
| R1 | **Neighbor structures are rebuilt at the start of every segment.** A run that is restarted from a checkpoint then performs the same rebuilds as a run that was not interrupted, provided both use the same segment schedule. Without this rule the two runs sum forces in different orders. |

## 6. Not yet designed

These items follow from the decisions above but have no design yet.

| Item | Needed for | Status |
|---|---|---|
| `md` dialect: types, ops, truncation, differentiation, exchange check | M0 | Implemented |
| `dyn` and `md_exec` dialects | M0 | Next step |
| Check that particle set symbols in types are declared | M0 | |
| Regression test for the numerical values of derivatives | M0 | Implemented with a kernel runner |
| `mdrt` ABI: storage, neighbor structures, events | M0 | Under discussion; requirements in ops-m0.md, Section 11 |
| Reference interpreter for the semantic dialects | M0 | |
| Schema of the TOML input | M0 | |
| Layout of the MDIR group inside an H5MD checkpoint | M0 | |
| HDF5 development files | M0 | The machine has the HDF5 runtime library but not its headers |
| Storage assignment pass | M0 | Specified in ops-m0.md, Section 10 |
| Lowering of transcendental functions on GPU targets | M1 | See ops-m0.md, Section 3.3 |
| Syntax for combining relations | M1 | |
| Scatter strategy for bonded terms | M1 | |
| Long-range dispersion correction | M1 | |
| Distributed fields and grids | M2c | |
