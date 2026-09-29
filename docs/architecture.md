# MDIR Architecture

Status: draft, revision 3 (2026-09-29). Nothing described here is implemented
yet. All IR snippets are illustrative; the syntax is not final.

Related documents:

- [decisions.md](decisions.md): the decisions this document follows.
  References such as D3 or P4 point there.
- [prior-art.md](prior-art.md): earlier work and what is taken from it.
- [design-review.md](design-review.md): review of revision 1.

## 1. Goal

MDIR is an MLIR-based compiler stack for general-purpose molecular dynamics
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
| `md` (MDIR-H) | Semantic dialect | What is computed? |
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

`md_dist` lowers before `md_exec`. On a single rank `md_dist` is the
identity, and the pipeline reduces to `md → md_exec`.

How the semantic objects reference each other:

```text
ensemble state
      │
      ├─────────┐
      ▼         ▼
 Hamiltonian   dynamics program
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
| Terms of the Hamiltonian and their functional forms | Particle count |
| Structure of the dynamics program | Simulation cell |
| Precision | Time step, temperature, λ |
| Target hardware | Force-field parameter tables, by default |
| Structural plan parameters | Numeric plan parameters |

A force-field parameter may be bound statically, per parameter, when the
specialization pays off. Temperature and λ are always run-time values, so
that all replicas share one compiled kernel.

The program is compiled once and distributed to all ranks. Compiled code is
cached under a hash of the IR and the target.

## 4. Semantic dialects

### 4.1 State

State has value semantics (D3). Every op that changes the simulation state
takes the old state and returns a new one. The state has components
`(x, p, h, ξ, ...)`: positions, momenta, simulation cell, and auxiliary
thermostat and barostat variables.

This lets analyses read invalidation off the use-def graph: an op that
redefines positions invalidates neighbor structures, and an op that redefines
the cell invalidates the partition.

The state and its per-particle fields are custom types, not builtin tensors
(D15). Value semantics does not mean copying; see Section 8.3.

**Particle identity is separate from storage index** (P17). Topology,
exclusions, and random number streams refer to global IDs. The storage order
of particles may change whenever neighbor structures are rebuilt.

### 4.2 `md` — Hamiltonian IR (MDIR-H)

`md` has a generic relational core (P14): particles, fields, relations,
neighborhoods, and reductions. A Hamiltonian is one kind of region built from
that core. Analyses and collective variables are others. Whether a region can
be differentiated depends on the ops it contains.

A Hamiltonian expresses the energy of a configuration, `H(x; θ)`, and nothing
about how or where it is computed.

```mlir
%n = md.neighborhood %s {
    support = #md.radius<1.0 nm>
}

%e = md.sum_relation %n {
    ...
}

%f = md.evaluate @H(%s) request [#md.forces]
```

Evaluation requests are energy, forces, virial, and `dH/dλ`.

Concepts that must not appear here: MPI, GPU, neighbor-list implementation,
ghost atoms, thermostats, integrators, replica exchange. Whether a pair is
computed once or twice is also not an `md` concern; `md` records only that an
interaction is symmetric under exchange of the two particles.

**Exclusions are a first-class relation** (D14, P8).

| Relation | Definition |
|---|---|
| Nonbonded pairs | Within the cutoff, minus excluded pairs |
| Scaled 1-4 pairs | A separate topological relation with its own parameters |
| Excluded pairs | A relation that terms may sum over |

The last row exists because Ewald methods compute a correction term over
excluded pairs. Exclusion is therefore not the same as removal.

**Neighborhoods are not exclusive to Hamiltonians.** Pairwise thermostats
such as DPD are velocity-dependent and not part of the Hamiltonian, yet they
need a neighborhood. `dyn` ops may consume neighborhoods too.

### 4.3 `mlff` — machine-learned force fields

`mlff` extends `md`. Its ops consume `md` neighborhoods and produce energies
that compose with other `md` terms.

Planned op families: embedding, radial basis, spherical harmonics, message,
tensor product, readout.

```mlir
%n  = md.neighborhood %s { support = #md.radius<5.0 angstrom> }
%h0 = mlff.embed %species
%h1 = mlff.message %n, %h0
%h2 = mlff.message %n, %h1
%e  = mlff.readout %h2
```

Until `mlff` exists, learned potentials attach through
`md.external_potential`. An opaque potential cannot expose per-layer stages,
so its halo must cover the full receptive field.

### 4.4 `dyn` — Dynamics IR

Expresses how one time step advances the state given a Hamiltonian.

Ops: `dyn.kick`, `dyn.drift`, `dyn.thermostat`, `dyn.barostat`,
`dyn.constraint_position`, `dyn.constraint_velocity`, `dyn.random`,
`dyn.metropolis`.

Velocity Verlet:

```mlir
%s1 = dyn.kick  %s0, %f0, %half_dt
%s2 = dyn.drift %s1, %dt
%f1 = md.evaluate @H(%s2) request [#md.forces]
%s3 = dyn.kick  %s2, %f1, %half_dt
```

**Random numbers are counter-based** (P5). `dyn.random` is a pure function of
seed, step, particle global ID, and stream ID. The state carries no generator.

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
NPT, temperature and Hamiltonian replica exchange, REST2, expanded ensembles,
alchemical schedules.

```mlir
thermo.state @s0 {
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

## 5. Semantic differentiation

Forces, virial, and `dH/dλ` are derivatives of the energy. They are produced
by an `md`-to-`md` transformation (D2):

```text
md / mlff forward graph
        ↓  semantic differentiation
md / mlff derivative graph
        ↓  locality analysis
distributed planning
```

- Closed-form pair and bonded terms are differentiated symbolically.
- `mlff` models are differentiated in reverse mode.

The derivative graph keeps its locality information. That matters because
reverse mode transposes communication: where the forward pass exchanges halo
features before a message stage, the backward pass accumulates adjoints on
ghosts and returns them to their owners.

## 6. Locality interface

Interaction ops implement `LocalityInterface` (D4), which returns an ordered
list of stages. Each stage reports `support`, `reads`, `writes`,
`accumulation`, and `freshness_requirement`.

| Interaction | Stages |
|---|---|
| Lennard-Jones | One stage: reads position and species within the cutoff, writes force. |
| EAM | Three stages: neighbors to electron density; density to embedding; neighbors and density to force. |
| Message passing, L layers | L stages, each reading the previous layer's features within the cutoff. |

Bonded terms, constraints, and virtual sites report topological support
instead of a radius.

## 7. Joint planner

Distribution and execution decisions are coupled, so one planner makes them
together (D9). For example, a larger skin widens the halo and raises
communication cost, but lowers the neighbor rebuild frequency.

The planner is a compiler component. Its result is an `ExecutionPlan`:

```text
partition          = regular(4, 4, 2)
skin               = 0.15 nm
neighbor           = cluster(8x4)
newton3            = false
force_accumulation = owner_only
halo_mode          = ...
```

### 7.1 The plan is part of the IR

The plan is serializable, can be overridden by the user, and is attached to
the IR (P7). Each lowering can then be tested against a fixed plan.

| Kind of parameter | Examples | Changing it |
|---|---|---|
| Structural | Cluster size, conflict strategy | Requires recompilation |
| Numeric | Skin, rebuild interval, domain boundaries | Tuned during the run |

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

Bitwise agreement between different hardware is a non-goal.

## 8. Execution dialects

### 8.1 `md_dist` — distributed execution plan

Expresses how the computation is split across ranks and GPUs. Concepts that
first appear here: domains, partitions, migration, owned and ghost particles,
forward halo exchange, reverse accumulation, interior and boundary
computation, replica communicators.

```mlir
%x_h   = md_dist.forward_halo %x { radius = 5.0 angstrom }

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
as PME need distributed fields and grids. They are excluded from v0 (D12) but
the dialect must not assume that all communication is particle halo exchange.

### 8.2 `md_exec` — MD-specific execution IR

Turns abstract `md` constructs into concrete computational structure. Its
core follows PPMD (P11): a particle loop, a pair loop, and a reduction, each
with a kernel region.

Initial op set:

```text
md_exec.build_cells        md_exec.build_neighbors
md_exec.particle_for       md_exec.pair_for
md_exec.reduce             md_exec.accumulate
md_exec.pack               md_exec.unpack
md_exec.launch
```

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

**Physical neighbor representation** is chosen here: dense cell list, Verlet
list, cluster list, BVH, or sparse cell map. The same pair loop may use a
different structure on each target.

**Neighbor structure validity is explicit** (P15). A neighbor structure is a
value built from a reference configuration, with a validity condition.

| Rebuild policy | Property |
|---|---|
| Fixed interval, buffer sized for that interval | Default. No per-step global reduction. |
| Displacement check | Optional. Needs a global reduction each step when distributed. |

Particles are spatially reordered when neighbor structures are rebuilt.

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

The pass must not silently copy a per-particle field inside the step loop
(D18). A required copy is reported with its reason. A copy is legitimate only
when one version of a field has more than one consumer, as in a Metropolis
rejection.

### 8.4 Dependency token

In value form, ordering follows from data dependencies between field values.
Once storage is assigned, operations act on buffers and those dependencies
are carried by one SSA token type, `!mdrt.event` (D10):

```mlir
%halo_done     = mdrt.halo_exchange %x_buf ...
%interior_done = md_exec.launch ... async
md_exec.launch depends_on [%halo_done] ...
```

The token also expresses ordering constraints that never had a data
dependency. It lowers to an MPI request, a CUDA event, or an NVSHMEM signal.
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

## 11. Roadmap

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

Milestones (P3) and what each one adds:

| Milestone | System | Adds |
|---|---|---|
| M0 | Lennard-Jones fluid, NVE | Whole pipeline on CPU and GPU, JIT, validation |
| M1 | Martini CG membrane and water | Bonded terms with a scatter strategy, exclusions, reaction field, thermostat, barostat |
| M2 | AA protein and water | PME, constraints, virtual sites |
| M3 | MLFF | Reverse mode, reverse accumulation, feature halo exchange |

The v0 performance target is homogeneous systems at finite density (C7).
