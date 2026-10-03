# MDIR Architecture

Status: draft, revision 5 (2026-09-29).

| Part | State |
|---|---|
| `md`, `dyn`, `md_exec`, the lowerings to the CPU and to NVIDIA GPUs, the driver | Implemented for milestone M0. [ops-m0.md](ops-m0.md), [neighbors-m0.md](neighbors-m0.md), and [driver-m0.md](driver-m0.md) describe what is implemented and have the actual syntax. |
| The planner, `md_dist`, `ensemble`, `mlff`, the events of `mdrt`, `ParticleDependencyInterface` | Not implemented. In M0 the options of the passes and the driver stand for the plan, and the ops of a block run in the order of the block (A12). |

The IR snippets of this document are illustrative; where they differ from
ops-m0.md, that document holds.

Related documents:

- [decisions.md](decisions.md): the decisions this document follows.
  References such as D3 or P4 point there.
- [ops-m0.md](ops-m0.md): types and ops for milestone M0.
- [prior-art.md](prior-art.md): earlier work and what is taken from it.
- [design-review.md](design-review.md): review of revision 1.
- [references.md](references.md): the literature that the documents cite.
- [conventions.md](conventions.md): the forms of the terms of the potential,
  the meaning of their parameters, and the units.

## 1. Goal

MDIR is an MLIR-based [[Lattner2021]](references.md#lattner2021) compiler stack for general-purpose molecular dynamics
(MD).

- It supports all-atom and coarse-grained simulation.
- It runs on workstations and on HPC clusters, on CPUs and GPUs.
- It compiles just in time, before the run starts.
- It is used through a standalone front end that reads a declarative input
  format, or through a Python library in the style of OpenMM. Both produce
  the same `md` IR and share one parser for energy expressions.

The design favors a small number of components with sharply separated roles.

| Component | Kind | Question it answers |
|---|---|---|
| `md` | Semantic dialect | What is computed? |
| `mlff` | Semantic dialect | How is a learned energy model expressed? |
| `dyn` | Semantic dialect | How is the state advanced in time? |
| `ensemble` | Semantic dialect | Which thermodynamic states are sampled, and by what protocol? |
| Joint planner | Compiler component | Which distribution and execution strategy is used? |
| `md_dist` | Execution dialect | Where does computation happen? |
| `md_exec` | Execution dialect | Over which set, and with which access pattern? |
| `mdrt` | Runtime dialect | Which runtime services and resources are used? |
| Runtime library | Library | Who owns storage and communication resources? |
| Upstream MLIR dialects | Back end | How does the hardware execute it? |

## 2. Two structures: a graph and a pipeline

The semantic dialects are not stacked. They form an **object graph** in which
ops reference each other. Lowering starts only below them, in the
**lowering pipeline**.

```text
       ┌────────────────────────────┐
       │   Semantic dialect graph   │
       │                            │
       │ md     dyn     ensemble    │
       │  \      |        /         │
       │       mlff                 │
       └────────────┬───────────────┘
                    │
          semantic differentiation
                    │
                    ▼
       ┌────────────────────────────┐
       │      Semantic analyses     │
       │                            │
       │ locality                   │
       │ field dependencies         │
       │ state and effects          │
       │ parameter dependencies     │
       └────────────┬───────────────┘
                    │
                    ▼
       ┌────────────────────────────┐
       │        Joint planner       │
       │                            │
       │ decomposition              │
       │ neighbor strategy          │
       │ skin                       │
       │ force conflict strategy    │
       │ precision                  │
       │ communication schedule     │
       └────────────┬───────────────┘
                    │
              ┌─────┴─────┐
              ▼           ▼
          md_dist       md_exec
              └─────┬─────┘
                    ▼
            mdrt (runtime ABI)
                    +
        scf / vector / gpu / linalg
                    │
           ┌────────┴────────┐
           ▼                 ▼
       LLVM (CPU)       NVVM / ROCDL
```

`md_dist` and `md_exec` are peer dialects (A8). Distributed lowering is
performed first, so that `md_exec` lowering can specialize the regions that
distribution creates. On a single rank distributed lowering is the identity,
and the pipeline reduces to `md → md_exec`.

How the semantic objects reference each other:

```text
ensemble state
      │
      ├─────────┐
      ▼         ▼
 potential     dynamics program
      │         │
      └────┬────┘
           ▼
      simulation
```

## 3. Compilation model

Compilation happens once, before the run (C3). What is fixed at compile time
and what stays a run-time value (P1):

| Compile-time constant | Run-time value |
|---|---|
| Terms of the potential and their functional forms | Particle count |
| Structure of the dynamics program | Simulation cell |
| Precision | Time step, temperature, $\lambda$ |
| Target hardware | Force-field parameter tables, by default |
| Structural plan parameters | Numeric plan parameters |

Binding is not an attribute in the IR. A parameter is static when the front
end passes it as a constant, which it may do per parameter when the
specialization pays off. Temperature and $\lambda$ are always run-time values, so
that all replicas share one compiled kernel.

The program is compiled once and distributed to all ranks. Compiled code is
cached under a hash of the IR and the target.

## 4. Semantic dialects

### 4.1 State

State has value semantics (D3). The state is not an aggregate type (S1). It
is the set of SSA values carried from one step to the next: positions,
velocities, forces, the simulation cell, and auxiliary thermostat and
barostat variables. The state holds velocities, not momenta (S2).

An op that changes part of the state takes the old value and returns a new
one. Analyses read invalidation off the use-def graph: an op that returns new
positions invalidates neighbor structures, and an op that returns a new cell
invalidates the partition.

The state and its per-particle fields are custom types, not builtin tensors
(D15). Value semantics does not mean copying; see Section 8.3.

**Particle identity is separate from storage index** (P17). Topology,
exclusions, and random number streams refer to global IDs. The storage order
of particles may change whenever neighbor structures are rebuilt.

### 4.2 `md` — particles, relations, and potentials

Earlier documents call this dialect MDIR-H.

`md` has a generic relational core (P14): particles, fields, relations,
neighborhoods, and reductions. A potential is one kind of function built from
that core. Analyses and collective variables are others. Whether a function
can be differentiated depends on the ops it contains.

A potential expresses the potential energy of a configuration, $U(\mathbf x; \theta)$, and
nothing about how or where it is computed (A1). Kinetic energy is not part of
it. "Hamiltonian" means `K + U` and is used only where that is meant, as in
Hamiltonian replica exchange.

```mlir
md.potential @lj(%x: !vec, %cell: !md.cell, %eps: f64, %sigma: f64) -> f64 {
  %n = md.neighborhood %x, %cell { cutoff = 1.0 } : !pairs
  %u = md.sum_relation %n, %x, %cell { ... } : f64
  md.return %u : f64
}

%f = md.evaluate @lj(%x, %cell, %eps, %sigma) request [forces] : !vec
```

Evaluation requests are energy, forces, virial, and derivatives with respect
to parameters.

**A relation is a set of tuples** with an arity and an orientation (A2). The
logical relation is distinct from its physical traversal. A potential sums
over each unordered pair once; an execution policy may traverse both
directions and weight the sum by one half.

**Exchange behavior is a contract** (B3). A kernel over pairs states whether
it is symmetric or antisymmetric under exchange of the two particles. The
compiler verifies the statement when it can prove it; otherwise the front end
must assert it.

**Truncation is an attribute** of a sum over a relation (B4): none, shift,
force shift, or switch. A pass expands it into the kernel before
differentiation.

**Quantities are plain numbers** in one internal unit system, declared per
module (S4). Front ends convert.

Concepts that must not appear here: MPI, GPU, neighbor-list implementation,
ghost atoms, thermostats, integrators, replica exchange. Whether a pair is
computed once or twice is also not an `md` concern; `md` records only how a
kernel behaves under exchange of the two particles.

**Exclusions are a first-class relation** (D14, P8).

| Relation | Definition |
|---|---|
| Nonbonded pairs | Within the cutoff, minus excluded pairs |
| Scaled 1-4 pairs | A separate topological relation with its own parameters |
| Excluded pairs | A relation that terms may sum over |

The last row exists because Ewald methods [[Ewald1921]](references.md#ewald1921) compute a correction term over
excluded pairs. Exclusion is therefore not the same as removal.

**Neighborhoods are not exclusive to potentials.** Pairwise thermostats
such as DPD [[Hoogerbrugge1992]](references.md#hoogerbrugge1992) are velocity-dependent and not part of the potential, yet they
need a neighborhood. `dyn` ops may consume neighborhoods too.

### 4.3 `mlff` — machine-learned force fields

`mlff` extends `md`. Its ops consume `md` neighborhoods and produce energies
that compose with other `md` terms.

Planned op families: embedding, radial basis, spherical harmonics, message,
tensor product, readout.

```mlir
%n  = md.neighborhood %x, %cell { cutoff = 0.5 } : !pairs
%h0 = mlff.embed %species
%h1 = mlff.message %n, %h0
%h2 = mlff.message %n, %h1
%e  = mlff.readout %h2
```

Until `mlff` exists, learned potentials attach through
`md.external_potential`. An opaque potential cannot expose per-layer stages,
so its halo must cover the full receptive field.

### 4.4 `dyn` — Dynamics IR

Expresses how one time step advances the state given a potential.

Ops: `dyn.kick`, `dyn.drift`, `dyn.thermostat`, `dyn.barostat`,
`dyn.constraint_position`, `dyn.constraint_velocity`, `dyn.random`,
`dyn.metropolis`.

Velocity Verlet [[Swope1982]](references.md#swope1982):

```mlir
%v1 = dyn.kick  %v,  %f,  %m, %half_dt
%x1 = dyn.drift %x,  %v1, %dt
%f1 = md.evaluate @lj(%x1, %cell, %eps, %sigma) request [forces]
%v2 = dyn.kick  %v1, %f1, %m, %half_dt
```

**Random numbers are counter-based** [[Salmon2011]](references.md#salmon2011) (P5, A5, A13). `dyn.random` is a pure
function of seed, step, stream ID, an entity key, and a draw index. The
state carries no generator. The draw index counts the numbers that one
entity takes from one stream in one step, and is a constant in the kernel
that draws.

| Entity | Key |
|---|---|
| Particle | Global ID |
| Pair | Canonical pair of global IDs |
| Global move | A fixed key |
| Replica | Replica ID |

**Requirements and capabilities** (D6). A `dyn` program declares what it
requires and what it provides. It does not claim to sample a named ensemble.

```mlir
dyn.program @baoab {
    requires.temperature
    provides.thermostatting
    ...
}
```

Library-provided thermostat and barostat kinds also carry metadata stating
whether they preserve the target distribution (P10).

### 4.5 `ensemble` — Ensemble / Protocol IR

Expresses which thermodynamic states are sampled and by which protocol: NVT,
NPT, temperature [[Sugita1999]](references.md#sugita1999) and Hamiltonian [[Fukunishi2002]](references.md#fukunishi2002) replica
exchange, REST2 [[Wang2011]](references.md#wang2011), expanded ensembles [[Lyubartsev1992]](references.md#lyubartsev1992),
alchemical schedules.

```mlir
ensemble.state @s0 {
    temperature = 300 K
    parameters = { lambda_vdw = 0.0 }
}

ensemble.replicas {
    states = [@s0, @s1, @s2, @s3]
    exchange { interval = 1000, topology = nearest_neighbor }
}
```

A thermodynamic state is not a dynamics algorithm. A protocol verifier checks
that the state, the protocol, and the declarations of the `dyn` program are
consistent. It issues warnings, not errors. Examples: a canonical ensemble
with no thermostat, or replica exchange with a thermostat that does not
preserve the canonical distribution.

Until `ensemble` exists as a dialect, it is a C++ runtime object.

### 4.6 Step loop

No `sim` dialect is planned for v0 (D7). The loop is split in two (P4):

| Part | Owner | Contents |
|---|---|---|
| Segment | Compiled code | The step loop, neighbor rebuilds, migration |
| Between segments | Driver | Output, checkpoints, replica exchange, Python callbacks |

The JIT compiles `run_segment(state, n) -> state`.

Inside a segment, halo exchange placement is decided statically. Host code
may modify the state between segments, which the compiler cannot see. That
case is detected with run-time version counters on the state (P16).

Segment boundaries are also where the tuning state is updated (Section 7.1).

## 5. Semantic differentiation

Forces, the virial, and parameter derivatives are derivatives of the
potential energy. They are produced
by an `md`-to-`md` transformation (D2):

```text
md / mlff forward graph
        ↓  semantic differentiation
md / mlff derivative graph
        ↓  dependency analysis
distributed planning
```

- Closed-form pair and bonded terms are differentiated symbolically.
- `mlff` models are differentiated in reverse mode.

The derivative graph keeps its locality information. That matters because
reverse mode transposes communication: where the forward pass exchanges halo
features before a message stage, the backward pass accumulates adjoints on
ghosts and returns them to their owners.

## 6. Particle dependency interface

Any semantic computation that needs data of other particles implements
`ParticleDependencyInterface` (D4, A3). That covers potentials, `mlff`
stages, pairwise thermostats, constraints, virtual sites, and collective
variables. The interface returns an ordered list of stages. Each stage
reports `support`, `reads`, `writes`, `accumulation`, and
`freshness_requirement`.

| Computation | Stages |
|---|---|
| Lennard-Jones | One stage: reads position and species within the cutoff, writes force. |
| EAM [[Daw1984]](references.md#daw1984) | Three stages: neighbors to electron density; density to embedding; neighbors and density to force. |
| Message passing, L layers | L stages, each reading the previous layer's features within the cutoff. |

Bonded terms, constraints, and virtual sites report topological support
instead of a radius.

## 7. Joint planner

Distribution and execution decisions are coupled, so one planner makes them
together (D9). For example, a larger skin widens the halo and raises
communication cost, but lowers the neighbor rebuild frequency.

The planner is a compiler component. Its result has two parts (A4).

### 7.1 Plan and tuning state

| | `ExecutionPlan` | `ExecutionTuningState` |
|---|---|---|
| Contents | Structural decisions | Numeric parameters and measured costs |
| Examples | Neighbor strategy, cluster shape, conflict strategy, precision | Skin, rebuild interval, domain boundaries |
| Mutability | Immutable; hashable and cacheable | Updated at segment boundaries and other declared safe points |
| Changing it | Replanning, and possibly compiling a new variant | No recompilation |

```text
ExecutionPlan                         ExecutionTuningState
  partition_kind     = regular          partition = (4, 4, 2)
  neighbor           = cluster(8x4)     skin      = 0.15
  newton3            = false            rebuild_interval = 20
  force_accumulation = owner_only
```

The run alternates between segments and retuning:

```text
run_segment → profiling feedback → retune numeric parameters → run_segment
```

The plan is serializable, can be overridden by the user, and is attached to
the IR (P7). Each lowering can then be tested against a fixed plan.

### 7.2 Pair execution policy

Each pair computation carries a `PairExecutionPolicy` (D11) with fields
`traversal`, `newton3`, `conflict_strategy`, and `reduction_order`.

| Conflict strategy | Cost |
|---|---|
| Directed pairs, write to the central particle only | Every pair computed twice |
| Each pair once, atomic adds | Atomic overhead; summation order varies |
| Each pair once, privatization or coloring | Extra memory or preprocessing |

The baseline for two-body terms is directed pairs with owner-only writes
(P13). It needs no atomics and no reverse accumulation. Computing each pair
once is a later planner optimization.

The baseline does not extend to other terms. A bonded term contributes to
several particles, so it needs a scatter strategy. Reverse-mode `mlff`
scatters adjoints to neighbors, so it needs reverse accumulation.

### 7.3 Reproducibility

Determinism is an opt-in execution mode (C6). The plan selects one level
(P6):

| Level | Guarantee |
|---|---|
| Fast | None |
| Deterministic | Same binary, hardware, and decomposition give the same bits |
| Decomposition-independent | The same bits for any rank count |

A level says what the planner may choose, not what it prefers: at the
deterministic level a sum whose order threads decide is not allowed.

The decomposition-independent level is a future mode and is not required for
M0 to M3 (A6). It needs fixed-point [[LeGrand2013]](references.md#legrand2013) or exact accumulation. Bitwise agreement
between different hardware is a non-goal.

### 7.4 Precision

Three modes are supported: single, mixed, and double (D23). Precision is a
structural plan parameter, assigned per role (S8).

| Mode | Positions, velocities, integration | Forces, kernel arithmetic | Global sums |
|---|---|---|---|
| Single | `f32` | `f32` | `f64` |
| Mixed | `f64` | `f32` | `f64` |
| Double | `f64` | `f64` | `f64` |

The semantic program is a floating-point program in `f64`, and that is the
reference (B2). Lowering to single or mixed precision deliberately relaxes
the numerical semantics, and its result agrees with the reference only
within a tolerance.

Precision is assigned by a pass on `md_exec`, after the transformations of
that level and before storage assignment (D31). The buffers that hold the
state state the type it is stored in; the driver allocates them according
to the policy, and the compiled program follows them (D32).

## 8. Execution dialects

### 8.1 `md_dist` — distributed execution plan

Expresses how the computation is split across ranks and GPUs. Concepts that
first appear here: domains, partitions, migration, owned and ghost particles,
forward halo exchange, reverse accumulation, interior and boundary
computation, replica communicators.

```mlir
%x_h   = md_dist.forward_halo %x { radius = 0.5 }

%f_int = md_dist.compute_interior %x   { ... }
%f_bnd = md_dist.compute_boundary %x_h { ... }

%f_loc = md_dist.combine %f_int, %f_bnd
%f     = md_dist.reverse_accumulate %f_loc
```

`md_dist` ops take and return field values (Section 8.3).

The first halo scheme is a full shell: each domain holds ghost copies of
every particle within the halo distance (P2).

`md_dist` does not know about `MPI_Isend`, NVSHMEM puts, or CUDA streams. It
records only that some data must reach another domain before a computation
runs. The transport is chosen below it:

| Transport | Use |
|---|---|
| In-process | Workstation with several GPUs, no MPI library needed |
| MPI | Cluster |
| NVSHMEM | NVIDIA cluster |

Particle domains are not the only distributed object. Mesh-based methods such
as PME [[Darden1993]](references.md#darden1993), [[Essmann1995]](references.md#essmann1995) need distributed fields and grids. They are excluded from v0 (D12) but
the dialect must not assume that all communication is particle halo exchange.

### 8.2 `md_exec` — MD-specific execution IR

Turns abstract `md` constructs into concrete computational structure. Its
core follows PPMD [[Saunders2018]](references.md#saunders2018) (P11): a particle loop, a pair loop, and a reduction, each
with a kernel region.

Initial op set:

```text
md_exec.build_cells        md_exec.build_neighbors
md_exec.spatial_order      md_exec.permute
md_exec.particle_for       md_exec.pair_for
md_exec.pack               md_exec.unpack
md_exec.launch
```

Reductions and accumulation are clauses of the two loop ops (S6).

**Access modes are the op signature.**

| Access | Form |
|---|---|
| Read | Input operand |
| Accumulate into existing | Destination operand carrying the old value, plus result |
| Accumulate from zero | Destination operand initialized to zero, plus result |

For generated kernels the access modes are derived from the kernel region and
verified. They are declared by hand only for external, opaque kernels.

**The cutoff predicate is explicit on the pair loop** (P18). It is not part
of the kernel body, so the lowering can choose between a branch and a mask.

**Physical neighbor representation** is chosen here: dense cell list
[[Quentrec1973]](references.md#quentrec1973), Verlet list [[Verlet1967]](references.md#verlet1967), cluster list [[Pall2013]](references.md#pall2013), BVH, or sparse cell map. The same pair loop may use a
different structure on each target.

**Neighbor structure validity is explicit** (P15). A neighbor structure is a
value built from a reference configuration, with a validity condition.

| Rebuild policy | Property |
|---|---|
| Check every step | Default (B1). Exact. Needs a global reduction each step when distributed. |
| Fixed interval | Must be selected explicitly. No check between rebuilds. |

With the fixed interval, the maximum displacement since the previous rebuild
is measured at each rebuild. Violations of the validity condition are counted
and reported (A11). A violation cannot be repaired afterward, which is why
the fixed interval is not the default.

Particles are put in the order of their positions where a run begins and
where a segment begins (D44).

**Data layout** is also chosen here: SoA, AoSoA, or cluster-packed, lowered
to `memref`.

`md_exec` is the last domain-specific IR.

### 8.3 Where value semantics ends

`md_dist` and the upper part of `md_exec` operate on field values (D17). A
halo exchange returns a new version of a field whose ghost region is current.

```mlir
%x_h   = md_dist.forward_halo %x
%f_int = md_exec.pair_for interior(%x),   %nl { ... }
%f_bnd = md_exec.pair_for boundary(%x_h), %nl { ... }
%f     = md_exec.combine %f_int, %f_bnd
```

Here `%f_int` does not depend on `%x_h`, so the interior computation can
overlap the halo exchange. No separate analysis is needed to see that.

A storage assignment pass inside `md_exec` then decides in-place updates and
the data layout. `md_exec` ops have a value form and a storage form, in the
manner of upstream `linalg`.

A value needs a buffer of its own when it is still live after the point where
its buffer would be overwritten (B10). In a Metropolis step [[Metropolis1953]](references.md#metropolis1953), the proposed
positions go to a second buffer while the old positions stay in place.

The pass introduces neither a copy nor an extra buffer inside the step loop
silently (D18). It reports the value, the overwrite, and the later consumer.

### 8.4 Dependency token

In value form, ordering follows from data dependencies between field values.
Once storage is assigned, operations act on buffers. While they run one
after another, the order of the ops in their block carries the dependencies
(A12). For asynchronous execution the dependencies are carried by one SSA
token type, `!mdrt.event` (D10):

```mlir
%halo_done     = mdrt.halo_exchange %x_buf ...
%interior_done = md_exec.launch ... async
md_exec.launch depends_on [%halo_done] ...
```

The token also expresses ordering constraints that never had a data
dependency.

`!mdrt.event` is an opaque runtime completion object (A9). It is not tied
one-to-one to a transport primitive: one halo exchange may involve several
sends and receives. Its implementation may be one or more MPI requests, a
CUDA event, or an NVSHMEM signal.

The schedule the IR must be able to express:

```text
interior kernel ─────────────┐
                             │
halo exchange ───────────────┤ overlap
                             │
                      halo arrived
                             │
                      boundary kernel
```

## 9. Runtime: `mdrt`

Communication, scheduling, and storage lower to a project-owned runtime ABI
(D13). The `mdrt` dialect is its representation in the IR (D19). Runtime
calls are `mdrt` ops with verified operands, lowered to function calls in the
last stage.

Shared types are split between two dialects:

| Types | Dialect |
|---|---|
| State, fields, relations, neighborhoods, quantities with units | `md` |
| Event token, domains and partitions, storage, physical neighbor structures | `mdrt` |

`md_dist` and `md_exec` both depend on `mdrt`; neither depends on the other.

The boundary between compiler and runtime (D8, P9):

| Runtime owns | Compiler owns |
|---|---|
| Memory and storage lifetime | Neighbor algorithm selection |
| Neighbor-list objects | Bin and cluster parameters |
| Cell buffers | Build and prune kernels |
| Migration buffers | Pair traversal |
| Communication resources | Fusion |
| MPI and NVSHMEM contexts | Schedule |
| Generic parallel primitives: sort, scan, compaction | MD-specific predicates: distance tests, exclusion filters, type-pair cutoffs |

`md_exec.build_neighbors` becomes a runtime call that allocates storage plus
a generated kernel that fills it.

## 10. Lowering to upstream MLIR

| Concern | Target | Notes |
|---|---|---|
| Control flow | `scf` | Parallel loops only after the conflict strategy has removed write conflicts. |
| CPU SIMD | `scf` → `vector` → `llvm` | Instruction set selection happens below `vector`. |
| GPU | `gpu` | No `md_gpu` dialect. |
| NVIDIA | `gpu` → `nvgpu` → `nvvm` → LLVM NVPTX | Hot patterns may specialize directly to `nvgpu`/`nvvm`. |
| AMD | `gpu` → `amdgpu` → `rocdl` → LLVM AMDGPU | |
| Communication | `mdrt` | The upstream `mpi` dialect is an optional target. It is unstable and has no `mpi.waitall`. |
| Asynchrony | `mdrt`, via `!mdrt.event` | The upstream `async` dialect targets CPU coroutines and is not used as the general mechanism. |
| MLFF dense math | `linalg` | Irregular neighbor gathers go through `md_exec` instead. |
| Data layout | `memref` | |
| Target metadata | `dlti` | MD-specific tuning comes from the planner. |
| Distributed grids | — | Not in v0. The upstream `shard` dialect is a reference. |

## 11. Input and output

| Purpose | Format | Contents |
|---|---|---|
| Input | TOML (D24, D35) | System, potential, dynamics, and run settings |
| Trajectory | DCD first, then XTC (D25, D37) | Positions, in reduced precision |
| Checkpoint | H5MD [[deBuyl2014]](references.md#debuyl2014), 64-bit floating point (D26) | Everything an exact restart needs |

A checkpoint holds:

| Item | Place in H5MD |
|---|---|
| Positions | `position` |
| Velocities | `velocity` |
| Forces, with an integrator that begins a step with them | `force` |
| Periodic image counters | `image` |
| Particle global IDs | `id` |
| Cell | `box` |
| Step and time | `step`, `time` |
| Velocity time offset, precision mode, unit system, random seed | MDIR group |
| Thermostat and barostat variables | MDIR group |

Random number generators have no state to save, because random numbers are
a function of seed, step, and entity (P5).

With velocity Verlet the forces are saved (D40). A step begins with the
forces of the step before, and forces that are computed again from the
positions differ in their last bits: a neighbor structure that is built
again has another order.

The particles of a file are in the order of the input, whatever order the
run keeps them in (D44).

## 12. Roadmap

Development order (D1):

```text
1. md
2. md_exec, with the CPU back end
3. minimal dyn
4. GPU back end
5. md_dist
6. mlff
7. ensemble
```

The first deliverable is a vertical slice that runs and is validated end to
end on a single node:

```text
md → semantic differentiation → md_exec → scf / vector → llvm
```

Milestones (P3, as amended by D169) and what each one adds:

| Milestone | System | Adds |
|---|---|---|
| M0 | Lennard-Jones fluid, NVE | Whole pipeline on CPU and GPU, JIT, validation |
| M1 | AA protein and water with an Amber force field | Bonded terms executed by the particles, exclusions and scaled pairs, tables of pairs of types, PME on one node, constraints, removal of the motion of the center of mass, thermostat, barostat, readers of Amber and GROMACS topologies, the `mdir` command (D53) |
| M2 | Python API | An object API over the same IR as the control file (D169); virtual sites, formerly M2a, shipped with M1 |
| M3 | Learned potentials on one GPU | External AD and tensor execution (D167), the potential and neighbor contract |
| M4 | Distributed execution | Ownership, halos, reverse accumulation, distributed PME (formerly M2c) |
| Later | Martini [[Marrink2007]](references.md#marrink2007) CG membrane and water | Deferred (D53) |

The v0 performance target is homogeneous systems at finite density (C7).

### Experimental CPU distribution reference

D[cpu-hybrid-lj] provides a fixed-layout LJ snapshot executable, separate from
`mdir run`: synchronous C++ MPI decomposes ownership and materializes ghosts,
then generated `md_exec` CPU loops use OpenMP and optional neighbor SIMD.
`md_exec.neighbor_view` distinguishes owned rows from local gather extent.
This is a reference execution path, not implementation of the proposed
`md_dist` field-version and contribution verifier. Exact interfaces and limits
are documented in [cpu-hybrid-lj.md](cpu-hybrid-lj.md).
