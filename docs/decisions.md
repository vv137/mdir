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

An entry is not rewritten when a later entry changes it. It says
"Amended by" and names the later entry, which holds where the two differ.

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
| P1 | **JIT binding times.** Structure is a compile-time constant: terms, functional forms, precision, target. Particle count, box, time step, temperature, and λ are run-time values. Force-field parameter tables may be made constant per parameter, opt-in. Compile once, distribute to all ranks, and cache by a hash of IR and target. | OpenMM kernel cache |
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

| # | Decision |
|---|---|
| D20 | **LLVM 23.1.2 is the pinned release.** It was the latest stable release on 2026-09-29. `scripts/build-llvm.sh` builds it. |
| D21 | **M0 integrators are velocity Verlet [[Swope1982]](references.md#swope1982) and leapfrog [[HockneyEastwood1988]](references.md#hockneyeastwood1988).** |
| D22 | **Energy expressions use the syntax of OpenMM custom forces [[Eastman2017]](references.md#eastman2017).** This makes D16 concrete. |
| D23 | **Three precision modes are supported: single, mixed, and double.** The reference is the semantic program in double precision. V1 withdrew the interpreter that was to execute it. |
| D24 | **The input format is TOML.** Its schema starts small and is expected to change. |
| D25 | **Trajectories are written as XTC.** XTC holds positions only, in reduced precision, so it does not serve as a checkpoint. *Amended by D37: DCD comes first.* |
| D26 | **Checkpoints are written as H5MD [[deBuyl2014]](references.md#debuyl2014), in 64-bit floating point.** A checkpoint holds positions and velocities, together with everything else an exact restart needs. H5MD is an HDF5-based format with standard places for positions, velocities, periodic images, particle IDs, and the box, and it allows application-specific groups. *Amended by D40: with velocity Verlet a checkpoint holds the forces.* |
| D27 | **D8 applies from M0.** The kernels that build cells and neighbor structures are generated from the first milestone. The runtime provides only generic primitives. |
| D28 | **CPU threading uses OpenMP in M0.** Threading is a structural plan parameter with the values `openmp` and `none`. With `none`, loops are lowered sequentially and the OpenMP runtime is not loaded. |
| D29 | **The neighbor structure of M0 is the neighbor matrix.** |
| D30 | **Vector fields are stored as `memref<?x3xT>` first.** The other layout comes later. |
| D31 | **Precision is assigned by a pass on `md_exec`.** `md-exec-assign-precision` runs on the value form, after the transformations of that level and before storage assignment. Everything before it works on the reference program in `f64`. The type `!md.field` admits `f32` for this; the ops of `md` and `dyn` reject it (B2). |
| D32 | **Buffers state the type that the state is stored in.** `mdrt.from_buffer` and `mdrt.to_buffer` accept a buffer of `f32` for a field of `f64`. Whoever allocates the state applies the roles `position` and `velocity`; the compiled program follows the buffers and never converts one (D18). For fields that no buffer holds, the pass derives the role from how the field is used. |
| D33 | **The storage handle of a field is a `memref`.** The storage form of the `md_exec` ops takes `memref<?x3xT>` and `memref<?xT>`. No storage type of its own is introduced: upstream passes and the upstream `gpu` dialect work on `memref`s, and the memory space of a `memref` can tell the device. This narrows D19, which placed a storage type in `mdrt`. |
| D34 | **GPU execution uses the upstream `gpu` dialect, with the runtime functions in `libmdrt_cuda`.** The kernels are embedded as PTX text. The neighbor build runs on the device, binning included; the particles of a cell are sorted by index so that the result does not depend on the order of the threads. A buffer on a device has a memory space in its type. |
| D35 | **The control file has one table for each concern of a run**: `[input]`, `[output]`, `[energy]`, `[dynamics]`, `[ensemble]`, `[boundary]`, and `[execution]`. The keywords are those of [driver-m0.md](driver-m0.md). An unknown keyword is an error. |
| D36 | **The control file is written in Å, kcal/mol, ps, amu, K, and atm.** Inside MDIR the units are nm, kJ/mol, ps, amu, K, and bar, which are consistent (S4). The driver converts. An energy expression is evaluated in the units of the control file: the driver scales the distance that enters it and the energy that leaves it. |
| D37 | **Trajectories are written as DCD first.** XTC (D25) follows. |
| D38 | **The control file is parsed with toml++**, which is added to the repository under `third_party`. |
| D39 | **The schedule of a run is compiled.** The driver builds one function with a loop for each period of output, and the code calls the host to write. A field that the host reads is passed with `mdrt.host_call`; it stays where it is. The loop over checkpoint intervals is marked as a segment, and neighbor structures start empty in each of its iterations (R1). |
| D40 | **A checkpoint holds the state as the next step needs it**: positions and velocities, and with velocity Verlet the forces. What MDIR needs beside the state is in the group `/parameters/mdir`. HDF5 1.14.6 is the pinned release; the releases from 2.0 on need a newer CMake than the machine has. |
| D41 | **The test of validity of a neighbor structure is made where the positions are written.** Each thread of the drift tests the particle that it has moved, and a thread whose particle has moved more than half the skin [[AllenTildesley2017]](references.md#allentildesley2017) sets a flag. There is no global maximum of the displacements: the refresh needs to know whether one particle is beyond the limit, not how far the farthest has moved. The criterion is unchanged, so the structure is built in the same steps. See Section 5.6 for what else was considered. |
| D42 | **The minimum image [[AllenTildesley2017]](references.md#allentildesley2017) is found with a multiplication** by one over the edge lengths, which is computed once. The number of images is a whole number, so the displacement of a pair differs from that of a division only if the pair is within rounding of half an edge apart, which is beyond every cutoff. |
| D43 | **A neighbor build searches a copy of the positions that is made for it**: the positions in the cell, in f32, in the order of the cells. It takes a pair that is within the cutoff plus the skin plus a margin for the rounding, so a row may hold pairs that are a little beyond. The width of the cells is chosen when the structure is built, from the density. On a device the search of a small system has one thread for each row of cells of a particle. See [neighbors-m0.md](neighbors-m0.md). |
| D44 | **The particles are put in the order of their positions where a run begins and where a segment begins**, not with every build of a neighbor structure as P17 has it. The order is by cell and, within a cell, by the number of the particle, which is its place in the input. The run carries the numbers as a field, and the files of a run are in the order of the input. The order does not depend on the order that the particles are in, so a run that continues from a checkpoint is exact. See [neighbors-m0.md](neighbors-m0.md), Section 5. |
| D45 | **The log takes the temperature and the pressure from the kinetic energies at the half steps as well** (Jung, Kobayashi, and Sugita, 2018 and 2019 [[Jung2018]](references.md#jung2018), [[Jung2019]](references.md#jung2019)): the temperature from the mean of the kinetic energies half a step before, at, and half a step after the step, the pressure from the mean of the two half steps. The total energy keeps the kinetic energy of the step, because that sum varies least. Leapfrog and velocity Verlet write the same log. A thermostat and a barostat of M1 take these estimates; with constraints the half steps must be measured, not computed from the forces. See [driver-m0.md](driver-m0.md), Section 2.3. |
| D46 | **The target of M1 is a bilayer of DPPC in water with Martini 2.2** [[Marrink2007]](references.md#marrink2007), [[deJong2013]](references.md#dejong2013), at constant temperature and pressure. M1 has no constraints, no virtual sites, no random numbers for each particle, and no correction for the dispersion beyond the cutoff. See [design-m1.md](design-m1.md). *Amended by D53.* |
| D47 | **The tuples of a topology are a set of entities**, declared with `md.tuple_set`. The members of the tuples are a relation, and a tuple has its parameters as fields of the tuple set, one value for each tuple. The members are the numbers of the particles, not their places in memory (P17). *Amended by D61.* |
| D48 | **A kernel over a relation of a topology takes internal coordinates**: distances, displacements, angles or their cosines, and dihedrals, which the op names. Differentiation takes the derivative of a coordinate from a closed form. |
| D49 | **A term over tuples is executed by the particles**: every particle visits the tuples that it is a member of and takes its own part, so a tuple of `k` particles is evaluated `k` times. A thread writes to its own particle, as with pairs (P13). Other strategies are choices for the planner later. *Amended by D61.* |
| D50 | **The thermostat of M1 is stochastic velocity rescaling** (Bussi, Donadio, and Parrinello 2007 [[Bussi2007]](references.md#bussi2007)), **and the barostat stochastic cell rescaling** (Bernetti and Bussi 2020 [[Bernetti2020]](references.md#bernetti2020)), not the barostat of Bussi, Zykova-Timan, and Parrinello (2009) [[Bussi2009]](references.md#bussi2009), which has a momentum of the cell. Both take a few random numbers for each step, which the host draws. There is one group with a thermostat. The thermostat takes the temperature and the barostat the pressure of D45. |
| D51 | **A neighbor structure is built after every change of the cell**, in M1. A structure that follows the cell is an optimization for later. |
| D52 | **A topology is read in the format of GROMACS**, and GROMACS is the engine that results are compared with (V1). *Amended by D58.* |
| D53 | **The target of M1 is all-atom**: a protein in water with ff14SB [[Maier2015]](references.md#maier2015) and TIP3P [[Jorgensen1983]](references.md#jorgensen1983), with particle mesh Ewald [[Darden1993]](references.md#darden1993), [[Essmann1995]](references.md#essmann1995) and constraints, at constant energy, temperature, and pressure. Martini and other coarse-grained force fields are deferred. The correction for the dispersion beyond the cutoff [[Shirts2007]](references.md#shirts2007) is in M1, on by default. Virtual sites stay in M2, so water with four sites cannot be run. See [design-m1.md](design-m1.md), Section 1. |
| D54 | **M1 has an intermediate stage** before particle mesh Ewald and constraints: flexible water, a Coulomb cutoff, a time step of 0.5 fs, at constant energy, with every term but the reciprocal part of the electrostatics compared with the reference engines. |
| D55 | **Particle mesh Ewald on one device or one process is in M1.** Its design is written when its stage begins. Two questions are open: spreading the charges on a device without breaking the deterministic level, and an FFT under a license that fits MIT (FFTW is under the GPL). |
| D56 | **Constraints are in M1**: SETTLE [[Miyamoto1992]](references.md#miyamoto1992) for water, SHAKE [[Ryckaert1977]](references.md#ryckaert1977) for the bonds of hydrogen, with RATTLE [[Andersen1983]](references.md#andersen1983) under velocity Verlet. A thread takes a cluster of constrained particles and writes to its particles only. The forces of the constraints add to the virial, every constraint removes a degree of freedom, and the kinetic energies at the half steps (D45) are measured, not computed from the forces. |
| D57 | **Lennard-Jones takes its parameters from a table for each pair of types**, which the front end fills from a mixing rule, with the pairs that the topology sets overriding it (NBFIX). No mixing rule is left at run time when a topology is read. Charges combine with the rule `product`. |
| D58 | **Topologies are read in the formats of Amber (`prmtop`, `inpcrd`) and of GROMACS (`.top`, `.itp`, `.gro`)**, by readers of MDIR's own, written from specifications of the formats that were learned from the documentation and the code of the engines, and sharing no code with them. Results are compared with AmberTools and GROMACS. |
| D59 | **MDIR has one program with subcommands**: `mdir run`, `template`, `check`, `emit`, `checkpoint`, `version`. `mdir-opt` stays a program for developers. |
| D60 | **The linear momentum is removed every `center_of_mass_interval` steps**, 100 by default, by a sum over the particles and a subtraction that fuses with the kick. The angular momentum is left alone under periodic boundaries. |
| D61 | **The members of a relation of a topology are the places of the particles in their fields**, and are renumbered where the particles are put in a new order, so a loop needs no table from numbers to places. The incidence structure has rows as wide as the particle with the most tuples needs, found when it is built. A sum over tuples takes the contribution of a tuple from its member at place 0, with no weight. |
| D62 | **MDIR has one form for each term of the potential and one meaning for each parameter**, which [conventions.md](conventions.md) states: harmonic bonds and angles as `½ k (x − x0)²`, periodic dihedrals as `k (1 + cos(n φ − φ0))` with the IUPAC sign of the dihedral, Lennard-Jones as `4 ε ((σ / r)¹² − (σ / r)⁶)` with `σ` and `ε` for each pair of types, Coulomb with the constant of CODATA 2018. Readers convert into these forms, and one library of the front end builds the kernels from them, whatever the format. |
| D63 | **The molecules that SETTLE constrains can be chosen.** From a GROMACS topology, those with `[ settles ]`; from an Amber topology, the residues named in `water_residues` of `[constraints]`, `["WAT"]` by default. A named residue that is not a rigid water of three sites is an error. `rigid_water = false` constrains the same water with SHAKE. |
| D64 | **MDIR stays under the MIT license.** Its readers are written from specifications of the formats, with no code of GROMACS or Amber, and it links no library under the GPL: the FFT of particle mesh Ewald is cuFFT on a device and a library under a permissive license, such as pocketfft, on the host; FFTW is not used. |
| D65 | **The target of M1 is ff19SB with OPC**, besides ff14SB with TIP3P (D53): CMAP and virtual sites, with the extra points of Amber and the virtual sites of GROMACS for water of four sites, move to M1. |
| D66 | **A step runs on the device without waiting for the host**, as far as possible: the test of validity of the neighbor structures, the thermostat, the barostat, and the constraints are decided on the device, so that the host waits only where it writes output, and a step can be one graph of kernels. The work of the host that is left, such as the incidence structures of a segment, moves to the device where it is measured to matter. |
| D67 | **The velocities are coupled at the end of a step, every `interval` steps**: the motion of the center of mass is removed, then stochastic velocity rescaling scales the velocities. The period of coupling is a loop of the schedule, as the intervals between energies are, so a step between couplings has no global sum. The thermostat takes the kinetic energy of the velocities it scales, without that of the center of mass: with velocity Verlet those of the end of the step, with leapfrog those half a step later. `center_of_mass_interval` equals the period of the thermostat, which is 10 by default; without a thermostat the motion is removed only if `center_of_mass_interval` is given, which amends the default of D60. The random numbers are drawn on the host from the key of A13 (stream 0, the step, entity 0), so a trajectory does not depend on how the steps are grouped into loops or on a restart. The energy that the coupling takes is written as the conserved energy. Drawing on the device (D66) waits for random numbers in kernels (Section 11.3 of design-m1.md). |
| D68 | **A virtual site is a particle of mass 0 in the particle set**, placed from its atoms in the programs of the step after every drift, with its force moved to them before the kick, both over a tuple set of the sites. The neighbor structures, the new orders of the particles, and the output take sites as they take atoms. The extra point of Amber keeps the placement of sander, which is not linear in the positions, and GROMACS keeps its linear combination; neither stands in for the other (design-m1.md, Section 19). |
| D69 | **The reciprocal sum of particle mesh Ewald is one op, `md.reciprocal`**, which yields the energy and, when differentiation asks, the forces and the virial from the grid; it is not differentiated through. The direct sum, the excluded pairs, the self term, and the net charge need no new op. The op lowers to templates in IR for spreading, the product with the influence function, and gathering, on the CPU and on a GPU, and to calls of the FFT of the runtime: pocketfft on the host and cuFFT on a device (D64). The driver computes the influence function as a table ([pme-m1.md](pme-m1.md)). |
| D70 | **The grid of particle mesh Ewald is accumulated in fixed point**, 64-bit integers at the scale 2⁴⁰ with integer atomics, on the host and on a device, so that the sum does not depend on the order of the threads (P6) [[LeGrand2013]](references.md#legrand2013). |
| D71 | **β, the grid, and the order of particle mesh Ewald can each be given**, or follow from a tolerance (`erfc(β rc) = 10⁻⁵`) and a largest spacing (1.2 Å), with the smallest even number of points of the factors 2, 3, 5, and 7, so that a run can take the parameters of sander or of GROMACS. The direct sum is not shifted at the cutoff by default, as in sander; `coulomb_modifier` shifts it, as GROMACS does. The influence function is that of Essmann et al. by default, as in GROMACS; `influence = "OPTIMAL"` scales it by the factor of the aliasing of the B-splines that sander applies by default ([pme-m1.md](pme-m1.md), Section 1.1). |
| D72 | **A barostat changes the cell at the end of a period of coupling**, by stochastic cell rescaling [[Bernetti2020]](references.md#bernetti2020) with the pressure of the step, as GROMACS does: the positions of the particles and the cell by `exp(Δε/3)`, the velocities by its inverse. The cell is not a value that the loops carry but a buffer on the host that the barostat writes and each iteration reads, so the rest of the schedule does not change; everything that depends on the cell (the neighbor structures, the influence function of PME, the corrections proportional to 1/V) takes it from there. Groups that the constraints keep rigid move with their centers of mass, so that the constraints do not take the stretched bonds back with velocities that heat the system. Isotropic first ([design-m1.md](design-m1.md), Section 11.4). |
| D73 | **Minimization is steepest descent in the metric of the masses, on the surface of the constraints**, with a step that grows by 1.2 when it lowers the energy and shrinks by 0.2 when it does not. The direction is the force over the mass with its parts along the constraints taken off as RATTLE takes them off the velocities, so that SETTLE and SHAKE, which weigh the particles by their masses, keep the step downhill; the step is scaled by a 16-norm of the direction, no less than its largest element, so that no particle moves farther than the step and no reduction to a maximum is needed. A minimization writes a checkpoint of the positions; a run of dynamics that reads it begins there anew, with drawn velocities, rather than continuing it ([design-m1.md](design-m1.md), Section 21). |
| D74 | **Restraints are applied after the evaluation of the potential, not in it**: a map over the particles adds −2k(x − x_ref) to the forces, and a sum over them gives the energy and the diagonal of the virial, Σ −2k d⊙d. A restraint depends on the absolute position of a particle, which the tuple loops, built on minimum-image displacements, do not provide; the positions are not wrapped, so x − x_ref needs no image. Under a barostat the reference positions are those of the file times the edge of the cell over that of the file, a factor that the loops pass to the steps with the cell, so that the barostat scales them with the positions. The particles are selected by masks of Amber ([design-m1.md](design-m1.md), Section 22). |
| D75 | **In a loop over pairs or tuples, only the difference of two positions is taken in their type; the minimum image, the squared distance, and the test of the cutoff are in the type of the kernel.** The subtraction is where precision is lost, and it stays in `f64` in the mixed mode; the image is then rounded to `f32`, by about 10⁻⁷ of the edge of the cell for a pair across it. A consumer device computes in `f64` at 1/64 of the rate of `f32`: the forces of the target of D65 took 172 microseconds a step on an RTX 3090 with the image in `f64`, 76 with it in `f32`, and 60 with the subtraction in `f32` as well, which is left out for its precision. The test of dynamics in mixed precision against its reference now agrees to 2 × 10⁻⁶ instead of 10⁻⁶; the target keeps its bonds to 3 × 10⁻⁶ Å and its conserved energy at constant temperature as before ([ops-m0.md](ops-m0.md), Section 10). |
| D76 | **Leapfrog takes the steps of velocity Verlet and differs in what it stores**: the velocities half a step behind the positions. It carries the forces of the positions and evaluates once per step, at its end, as velocity Verlet does. Its step with energies begins from the velocities of the time of the positions, `P(v_{n−½} + h f_n/m)`, so that the virial of the constraints has the meaning it has with velocity Verlet, and returns the velocities of the time of the new positions for the kinetic energy. The constraints bring both integrators to the same positions, so their logs agree. The barostat couples the velocities of the time of the positions and stores them half a kick behind, which counts the energy it gives exactly; the thermostat alone scales the stored velocities ([design-m1.md](design-m1.md), Section 23). |
| D77 | **The barostat counts the work of a scaling exactly, from the energy of the scaled positions, and the next step takes their forces.** Counted to first order in the strain, −(μ − 1) tr W, as GROMACS does, the conserved energy drifts by the mean of the second order, which the noise of the strain makes proportional to 1/τ_p and independent of the time step and the period of coupling: on tri-alanine in 1218 OPC waters, 2.13 kcal/mol/ps with τ_p = 2 ps and 0.52 with τ_p = 8, about 7 k_B T per τ_p. Evaluating the scaled positions gives U(x′) − U(x) and the forces of x′ at once, so a step no longer begins with the forces of other positions: the drift becomes 0.011 kcal/mol/ps, below that at constant volume. The strain is a step of λ = √V, eq. (7) and (S7) of [[Bernetti2020]](references.md#bernetti2020), whose noise does not depend on the volume; with the evaluation after the scaling, 1 + 1/N_P evaluations per step (its Table I), this is the paper's reversible integrator (SI Sec. V.B) but for the thermostat, which acts once at the end of the period rather than in halves around the step. The conserved energy of MDIR is the energy of the system and what the thermostat and the barostat exchanged with the bath; its drift measures the integration of the Hamiltonian dynamics and the counting of the work, not the violation of detailed balance by the barostat, which the paper's effective energy measures (Sec. II.C) and leaves undefined for the Euler scheme (Sec. II.D); the volume distribution is what tests the barostat. `work = "FIRST_ORDER"` keeps the approximation, and the tests compare both. The paper's Trotter integrator, which needs no evaluation after the scaling, is the candidate to remove its cost ([design-m1.md](design-m1.md), Section 11.4). |
| D78 | **On a device, the reciprocal sum of PME computes in the type of the forces**: the B-splines, the grid, its transform (cuFFT R2C and C2R in `f32`), the influence function, and the gathered forces are in `f32` in the mixed and single modes, as GROMACS does; the fractions of the positions on the grid and the edges of the cell stay in `f64`, the charges are added in fixed point as before (D70), and the energy and virial of a step of energy are summed in `f64`. A consumer device computes `f64` at 1/64 of the rate of `f32`: on JAC (23,558 atoms, RTX 3090) the gather took 149 µs a step and 38 after, the scaling of the transform 90 and 36, and the rate went from 211 to 265 ns/day. The reciprocal energy at the start moved by 2.4 × 10⁻⁵ kcal/mol of 2150 (10⁻⁸), and the conservation of the total energy over 4000 steps did not get worse. The host keeps `f64` ([pme-m1.md](pme-m1.md)). *Amended 2026-10-01:* the fraction of a position along the cell, x/L less its floor, stays in `f64`, which keeps the positions of a large cell exact where it matters; its product with the number of points and the fraction within a point are in the type of the forces, which resolves a point to K·2⁻²⁴ (1.6 × 10⁻⁵ of a point for K = 270), as a computation of GROMACS in `f32` does. With the splines no longer in local memory, the `f64` of the placement dominated the gathering on Cellulose: 208 to 127 µs a step; the total energy at step 500 moved by 6 × 10⁻⁹ of its value. |
| D79 | **The terms of the potential take the positions in the type of the kernel, converted once per evaluation; the constraints keep theirs.** Where the kernel is narrower than the positions (the mixed mode), `md-exec-assign-precision` puts a loop over particles before the loops over pairs and over tuples that are not disjoint, which converts the positions to `f32`, and those loops read that field: the difference of two positions is then taken in `f32`, as GROMACS does, where D75 took it in `f64`. On a consumer device each `f64` subtraction and conversion runs at 1/64 of the rate; on JAC (RTX 3090, 5 runs each) the rate went from 293 (tables in `f64`) to 344.5 ± 0.7 ns/day with the subtraction in `f32` and to 366.7 ± 0.7 with the converted positions, which also halve the bytes each neighbor loads; the tables that kernels in `f32` look up are copied to the device in `f32` (`md-exec-assign-storage{tables=f32}`), which gave 265 to 293 and the same results to the bit. Positions of a cell of 60 Å are 4 × 10⁻⁶ Å apart in `f32`. The drift of the total energy of `examples/ala3` at constant energy over 100 ps, 2 fs, was −0.0094 kJ/mol/ns per atom before and +0.0004 after; a minimization of 50 steps ends 0.4 kcal/mol from that in double precision, as steps are taken or refused. The loops of SETTLE and SHAKE, over disjoint tuples, keep `f64` positions, which the lengths of the bonds depend on. `kernel-positions=false` restores D75 ([ops-m0.md](ops-m0.md), Section 7). |
| D80 | **A neighbor structure stays valid when a barostat scales the cell; structures that differ only in their cells share one storage.** Built at $\mathbf x^\text{ref}$ in the cell of edges $\mathbf L^\text{ref}$ with the reach $R = r_c + s$, a structure is valid in the cell $\mathbf L$ when, with $\mathbf m = \mathbf L / \mathbf L^\text{ref}$ per axis, $2\max_i \lVert \mathbf x_i - \mathbf m \odot \mathbf x^\text{ref}_i \rVert \le \min(\mathbf m)\,R - r_c$. A pair left out was farther than $R$ apart at the build; the scaled reference puts it farther than $\min(\mathbf m) R$ apart, and each particle is at most $d_\text{max}$ from its scaled reference, so it is farther than $\min(\mathbf m) R - 2 d_\text{max} \ge r_c$ apart now. The condition is exact whatever moves the particles: the rigid groups that the barostat moves with their centers, or the dynamics. In the cell of the build it is the half-skin test. Before, a structure was valid only in the cell of its build, and the loops of a run at constant pressure refreshed their structures in cells of their own values, so `md-exec-reuse-neighbors` kept five structures apart; the tri-alanine example at 1 bar rebuilt every 2.5 steps, and every 5.0 steps once they share one storage, as at constant volume; its rate went from 323.8 to 414.8 ns/day (one run each). `md_exec.reference_cell` and `md_exec.cell_edges` let the test that `md-exec-expose-validity` writes take the cells. |
| D81 | **The reciprocal sum of PME runs on a second stream**, beside the loops that follow it, from the positions as they are where it begins, until an op reads its forces, writes what it reads, or is not a loop, where the first stream waits for it (`convert-md-exec-to-gpu{pme-stream}`, on by default; `mdrtSideBegin`, `mdrtSideEnd`, `mdrtSideJoin` in the runtime, which waits for both streams where the host reads). The results are the same to the bit. On an RTX 3090 (n = 5 for JAC, 3 otherwise), JAC went from 365.6 ± 0.7 to 374.1 ± 0.8 ns/day, FactorIX from 85.2 ± 0.3 to 86.3 ± 0.1, and Cellulose stayed at 15.6: the kernels of pairs and of PME each fill the device, the more so in a large system. The caching allocator still assumes that memory freed by one stream is reused in order; the sum allocates nothing while it runs beside. *Amended 2026-09-30:* off by default. Once the kernels of PME were cheaper (5dec86b and the spreading of D84), the second stream slowed the steps: JAC (RTX 3090, 20,000 steps) ran at 346 to 356 ns/day with it and 447 without, in the default mode, and 386 to 407 against 439 in the deterministic mode; FactorIX at 123 against 137. PME's kernels share the device with the loop over pairs, which is the longest part of a step, and lengthen it. `pme-stream=true` kept the option; D87 replaced it with `md-exec-assign-streams{reciprocal=true}`, which decides where the first stream waits from the effects of the ops. |
| D82 | **On a device, the loops over pairs get a tile structure: the cluster pair list of GROMACS, with the exact validity of MDIR.** Proposed, and on hold: a prototype found the tile kernels slower than the neighbor matrix at equal reach on an RTX 3090, and the reach of the list and the cost of a build to decide the rate ([tiles-m1.md](tiles-m1.md), Section 11). Particles are grouped in tiles of 8, formed and searched as the clusters of [[Pall2013]](references.md#pall2013) are, with a mask of 64 bits per tile pair for the pairs within the reach and not excluded; the dual list with pruning follows [[Pall2020]](references.md#pall2020). MDIR departs in three points: the lifetime of each list is decided by the exact test of D80 (for the pruned inner list, both $2\max_i \lVert \mathbf x_i - \mathbf m \odot \mathbf x^\text{ref}_i \rVert \le \min(\mathbf m) R - r_c$ and the same with the positions, cell, and reach $R_\text{in}$ of the pruning), not by a buffer estimated from a tolerance of the drift; the kernel of a pair stays the semantic kernel, and computing a pair once is legal only with its exchange contract; and the deterministic level (P6), which a half list with floating-point atomics breaks and one in fixed point (D70, [[LeGrand2013]](references.md#legrand2013)) does not, is a mode of a run (D84), not a constraint on the default. The particles are not permuted into the order of the tiles; the conversion of the positions of D79 writes them in that order. The stages T1 (full list), T2 (half list), T3 (dual list), and T4 (refinements) are each measured on JAC, FactorIX, Cellulose, and STMV before the next. *Superseded 2026-10-01 by D89:* groups of 16 particles sharing a list of particles, measured faster than both the tiles and the matrix on Cellulose. |
| D83 | **The groups of the constraints are a disjoint union of tuple sets, which the IR states, and the loops over one group read the fields before the updates of the others.** `md.disjoint_union @constraints on(@atoms) of [@settles, @shake1, ...]` states that no atom is a member of two tuples of the sets, whether of one set or of two; the driver checks it from the members and leaves the union out where it fails. A loop over the tuples of a set reads its fields at their members only, where what a loop over an other set of the union gathered is zero: a field $\mathbf y = \mathbf x + \mathbf c$ with $\mathbf c$ gathered over an other set agrees with $\mathbf x$ there, and `md-bypass-updates` lets the loop read $\mathbf x$ (the same values up to the sign of a zero). The chain that the driver writes for SETTLE and each set of SHAKE, a gather and an addition per group, becomes gathers that depend on no other group and additions that `md-exec-fuse-loops` fuses into one loop over particles with the correction of the velocities; `convert-md-exec-to-gpu` puts consecutive loops over disjoint tuples that read nothing the others write into one kernel, in which a thread evaluates the tuples whose member at place 0 it is. On JAC (RTX 3090, 20,000 steps) the constraints of a step took 8 kernels over tuples and 10 over particles before, 2 and 6 after, and the rate went from 357 to 375 ns/day; the drift of the total energy is the same to the digit. The union is also the unit of ownership of a later domain decomposition, as the update groups of GROMACS are [[GromacsManual2025]](references.md#gromacsmanual2025): a group then lies in one domain, its update needs no communication, and the halo grows by the extent of a group. Constraints coupled over larger structures (all bonds, rings) are no union of small groups and need communication across domains; how to do that is open. The union is not yet a fusion of the kick, drift, and constraint of a group in one thread, which remains to do. |
| D84 | **The deterministic level is a mode that a run chooses, not a constraint on every run** (amends P6; the user, 2026-09-30). By default a lowering may add in an order that the threads decide, such as floating-point atomic additions to the forces of a half list, and so ignore that floating-point addition is not associative; a run in the deterministic mode gets the same bits from the same binary, hardware, and decomposition, with sums in a fixed order or in fixed point (D70). In both modes the arithmetic keeps its care for rounding: sums of many terms in `f64` or in fixed point, differences of positions before their conversion to `f32` (D75, D79), compensated or pairwise sums where they matter. The mode is `deterministic` in `[execution]` (`convert-md-exec-to-gpu{deterministic}`): by default the charges of PME are spread on a device with floating-point atomics, in the deterministic mode in fixed point; every other sum of now is in a fixed order in both. |
| D85 | **Device code counts in 32 bits: the numbers of particles, cells, tuples, entries of a row, and loops; only a flattened element number and a byte address are 64 bits** (the user, 2026-09-30: 64 bits for the index of a particle is too much). MLIR's `index` is 64 bits on NVPTX, and every template and generated kernel used it for everything; the search of the neighbor matrix needed more than 64 registers and spilled, and took 612 µs a build on JAC where it takes 430 in `i32` (f0f5b35). A signed 32-bit number holds 2.1 × 10⁹ particles; what overflows first is the flattened entry of a two-dimensional buffer, the matrix of neighbors at $n W$ (at 800 entries a row, from 2.7 million particles), which stays 64 bits. A run whose counts do not fit stops before it starts, with the limit. The search kernel counts in `i32` now; the lowering of the other kernels (the `index` of `convert-md-exec-to-gpu` and of the templates of PME) follows, where the host and the device must agree on the width of the sizes and strides of a buffer at the launch. |
| D86 | **On a device, the loops over pairs run in the order of the cells of the last build, and the state keeps its order** (implemented 2026-09-30: JAC ran at 505 ns/day over 40 ps and 506 over 200 ps, where it ran at 478 and 383). The particles were put in the order of their positions only where a run or a segment began (D44), and the order decays as they diffuse: over 40 ps of JAC (RTX 3090) the loop over pairs went from 117 to 208 µs a step alone, and from 153 to 189 fused with the bonded terms; a run of 200 ps ran at 383 ns/day where one of 40 ps ran at 478. Putting the whole state in order more often would renumber the members of every tuple set on the host. Instead the build of the neighbor matrix, which sorts the particles by cell anyway, keeps that order: the rows and entries of the matrix are places in it, a kernel of each step gathers the positions and the fields that the loops over pairs read into it, the loops read them there, and each writes the forces of a particle at its place in the state. Loops over tuples that share the kernel of a run take the particle of the same place. The order is new with every build, as GROMACS puts the particles in the order of its grid at every search (its `x_to_nbat` kernel, 7.6 µs a step on JAC). Over the same 40 ps the other kernels kept their times (the constraints 15.0 to 15.4 µs, the spreading of PME 10.8 to 11.1), so the order of the loops over pairs is what matters on one device. Putting the whole state in order at intervals, with the members of the tuples renumbered and the incidence structures rebuilt on the device, belongs to the domain decomposition (M2), where the particles of a node change as they migrate and need it anyway; it can then serve one device too. With the loops over pairs in order, the loops over tuples evaluate each tuple once and add to its members with atomics (`tuples-once`, the default outside the deterministic mode): 526 against 505 ns/day for the rows of the particles, which evaluate a tuple once for each member. In a domain decomposition the two differ as Newton on and off do for pairs: a tuple across two domains is evaluated once and its forces on the other domain's atoms sent back, or on both sides with the halo of positions alone; the choice belongs to a policy of `md_exec.tuple_for` that the plan makes from the cost of communication. |
| D87 | **An op runs on a second stream only beside ops that are proven independent of it, and the proof is checked where it is lowered.** `md-exec-assign-streams` marks an op `md_exec.side`, moves it up its block past the ops it is independent of, and puts an `md_exec.join` before the first op after it that is not; `convert-md-exec-to-gpu` checks the window again (`verifySide`) and rejects a program where it does not hold. Two ops are *independent* when neither uses a value the other gives, both declare all their memory effects (and those of the ops inside them), and nothing one writes or frees may alias what the other reads, writes or frees (`Independence.h`). **Claim.** The device memory and every value the host reads are as in the serial program. **Argument.** The host issues ops in program order; each stream runs its work in the order it was given; `mdrtSideBegin` records an event on the first stream that the second waits for, and `mdrtSideJoin` makes the first wait for an event recorded after the side op. So every op before the side op R runs before R, and every op from the join on runs after R. Only R's order relative to the ops X strictly between R and the join changes, and each such X is independent of R: neither reads what the other changes, so the two give the same memory in either order or at once. Moving R up past an independent op is the same lemma. **Premises, and where each is checked.** (P1) The declared effects cover what the lowering of an op touches: a contract of each op, audited for `md_exec.reciprocal` (positions, charges, moduli, `outs`, `scratch`; the cuFFT plans are used by the reciprocal sums alone, and all of them go to the second stream when any does), and for the neighbor structure, whose hidden buffers the analysis counts as part of the structure: its own buffers and the excluded pairs it keeps (`reset_neighbors`). An op without declared effects, such as a call, is dependent on every op. Two ops declare less than they touch: `md_exec.reference_cell` is pure but reads the cell of the structure's last build, and `md.lookup` in a kernel reads its table buffer and declares no effect. Neither can conflict with the reciprocal sum, which writes only its `outs` and `scratch`; that is why only `md_exec.reciprocal` may be marked, and marking another op needs these contracts fixed first. (P2) Aliasing is traced to allocations and function arguments: distinct live allocations are disjoint, two arguments may alias, a view is its source. Two arguments i and j that a loop carries, whose iterations yield a permutation π of them, are after k iterations the initial values at π^k(i) and π^k(j), which are distinct places as π is a bijection; so they are distinct memory at every iteration if the initial values at every two distinct places of the orbits of i and of j are. All pairs across the two orbits count, as the orbits may differ in length (a first version compared one turn of the shorter only, and missed the case in `assign-streams.mlir`, `@orbits`). The analysis checks that the yield is a permutation; otherwise it knows nothing. (P3) The lowering issues each op's work where the op is: runs of fused loops span only loops and constants, so neither a side op nor a join is inside one, and `deferReadbacks` does not move a readback past a call of `mdrtSide*`. (P4) The runtime sends all work issued between `mdrtSideBegin` and `mdrtSideEnd` to the second stream and no other; the side op's lowering is the only thing issued there, and it allocates and frees no memory of the device (checked by the lowering, through the bodies of the templates it calls; of the functions of the runtime without a body, only the transforms of cuFFT, audited, may be called, since their plans hold their work areas; host memory from `malloc` orders nothing on the device), because the allocator hands a freed block to the next allocation without waiting, as work on one stream in order may. A copy that the host reads waits for both streams. (P5) No marked op is inside another's window (the pass ends a window at the next marked op; the check rejects it). **Evidence.** `test/Driver/pme-gpu.test` runs a water box in the deterministic mode with the sums on the second stream, beside the refresh, the loop over pairs and five loops over tuples, and the state equals the serial one to the bit. That is evidence, not proof; the proof is the argument with P1–P5. **Measured** (RTX 3090 at 300 W, 20,000 steps, two alternating runs each): JAC 534.9 and 534.6 against 527.1 and 527.5 ns/day; Cellulose 26.8 and 26.8 against 27.0 and 27.0. Off by default (D81). A copy that waited for its own stream alone (sound by P1: an op that reads what the side op writes comes after its join) gained nothing and was not kept. Buffers fixed for the whole run, in place of those that the lowering allocates in each step, would remove the premise on the allocator in P4. |
| D88 | **A fixed interval of rebuilds of the neighbor structures is an opt-in policy, never a default, and a run that takes it is warned and told what it risked.** `rebuild_interval = n` in `[energy]` makes the driver run `md-exec-rebuild-at-interval{interval=n}`, which gives the refreshes the policy `interval` of `md_exec.refresh_neighbors` (B1 of the M0 specification): a build at the first refresh and every `n` refreshes after a build, with no test of validity between. Unlike D87 this changes results beyond rounding: a particle that moves more than half the skin before the next build leaves pairs within the cutoff out of the structure, and their forces are missing. It is therefore a policy that a user chooses, not an optimization (Section 2 of `docs/principles.md`). **Safeguards.** The run warns at the start on the standard error, whatever the log is, and writes the warning to the log; the pass, the op, the lowerings and the runtime say in their comments that it is not a default. At each build the structure is tested as `check` would test it, before it is rebuilt, and a build that finds it no longer valid is counted (`mdrtCountLateBuild`); the log reports the count, and the run warns if it is not 0. The test is the sufficient condition of the op: it counts a build as late when one particle has moved more than half the skin, while a pair is left out only when its two particles have come more than the skin closer than they were at the build; a late build permits that but does not imply it. The count bounds the builds after which pairs may have been missed, not the pairs. A particle that went too far and came back within the interval is not seen. **Measured** (JAC NVE, RTX 3090 at 300 W, 20,000 steps, one run each): the default 527.9 ns/day with 2,150 builds (every 9.3 steps); `rebuild_interval = 10`, 547.4 with 2,001 builds of which 1,938 were late; `= 20`, 595.8 with 1,001 builds, all late. The energy changed by 3.2e-5, 3.8e-5 and 2.5e-5 of its value. What GROMACS gains with a thermostat or barostat (a list every 50 steps) comes from a longer list and pruning with a buffer estimated from a tolerance, which is another design. |
| D89 | **On a device, the loops over pairs of large systems run over groups of 16 particles that share a list of neighbors, each pair once.** A second kind of neighbor structure, `groups` ([groups-m1.md](groups-m1.md)): the places of a build in groups of 16, each with a list of entries (a place and a mask of 16 bits over the group), which hold every pair within the reach once; the loop computes each pair once and adds the value of the other particle with an atomic addition (in the type of the destination by default, in fixed point in the deterministic mode, D84). It is legal for kernels with an exchange contract, which `md_exec.pair_for` now carries for each destination and sum; the others keep the matrix. The validity of the list is the exact test of D80, unchanged: the list is a superset of the pairs within the cutoff while the test holds. Measured first (Cellulose, reach 9 Å, RTX 3090, standalone): the loop 1013 to 1034 µs against 1245 to 1267 for the matrix, with a sixteenth of its entries; the build of the matrix took 8.0 ms, of which 5.0 for tests of candidates that the bounding boxes of groups avoid. Layout after [SalomonFerrer2013]. Supersedes D82. Stages G1 to G5. |
| D90 | **Under `fast_math`, the f32 kernels of loops may approximate divisions and erfc, within a stated bound, and share the exponential of erfc with its derivative.** `md-exec-approximate` runs after `md-exec-assign-precision`: an f32 division in the kernel of a loop takes `afn` (on a device `div.approx.f32` or `rcp.approx.f32`, within 2 ulp, instead of the correctly rounded sequence), so does an f32 exponential (`ex2.approx` of x log2(e), within 2 + 1.2 |x| ulp, about 1.5e-6 at x = -(β rc)² ≈ -10, instead of the reduction of the argument of `expf`: 1687 to 1622 µs on Cellulose with groups), and `erfc(sqrt(y) c)` with `c > 0` becomes `exp(-c² y) P(1 / (1 + x / 2))`, P of degree 9 fitted by weighted least squares to erfc(x) exp(x²) on [0, 6] (`scripts/fit-erfc.py`); its relative error in f32 is below 3.3e-7 there (3.05e-7 measured on the host, `test/Integration/erfc-approximation.mlir`), that of `erfcf` a few ulp. The exponential is the one the derivative of erfc already computes, `exp(k y)` with `k` within 4 ulp of `-c²`; the rounding of its argument is common to both. In the force of the direct sum of Ewald the erfc term is a share 1 / (1 + 2x²) of the force factor, so the relative error it adds to the force is at most that of erfc over 1 + 2x². Only a square root times a positive constant matches, so a negative argument never does; f64 is left as it is. Measured (RTX 3090, reach 10 Å): the loop over pairs of Cellulose 2134 to 1856 µs with the matrix, 2097 to 1687 with groups; JAC 113 to 102 and 92. The energies of JAC at the start change by 1e-8 to 6e-8 of their values, the conservation over 20 steps not at all (2.25e-5 against 2.26e-5). |
| D91 | **What a kernel derives from tables alone is computed once per entry of the tables, in f64.** `md-exec-fold-tables` runs before `md-exec-assign-precision`: a value of the kernel of a loop that depends only on tables looked up at the same indices, and on constants (48 ε σ¹² and 24 ε σ⁶ of Lennard-Jones from the tables of σ and ε of a pair of types), becomes a lookup of a new table, `md_exec.tabulate`, which `md-exec-assign-storage` computes on the host and places as the tables it comes from. A table that serves two loops is made once. In mixed precision the kernel reads the value rounded once instead of computing it in f32: the Lennard-Jones energy of JAC at the start is 9123.5245 against 9123.5262 in double precision, from 9123.5324. Measured (RTX 3090, reach 10 Å, with D90): the loop over pairs of Cellulose 1622 to 1548 µs with groups, JAC 92 to 85. |
| D92 | **The barostat scales the cell within the drift of the last step of a period by default (the integrator of Trotter type of [[Bernetti2020]](references.md#bernetti2020), SI Sec. V.C), and counts the energy of the scaling from the virials before and after it.** The exact work of D77 evaluates the scaled positions once more each period, 1 + 1/N_P evaluations a step (the paper's reversible integrator, its Table I); the Trotter type needs none. A period of N_P steps takes N_P − 2 plain steps, then the step of energy whose pressure (its virial and the kinetic energy of its velocities without the center of mass) gives the strain and the new cell, then `step_trotter`, which kicks half, drifts half, scales the positions by μ (each rigid group with its center of mass, as D72) and the velocities by 1/μ, drifts the other half in the new cell, evaluates there with the virial, and kicks half: eqs. (S12a–d). Its eqs. (S13a) and (S15), which write them as one, leave out the scaling of q(t) and have Δt for Δt/2 (with V′ = V they give twice the drift). Leapfrog drifts with the velocities of the middle of the step as velocity Verlet does, so the step is the same. The velocities change the kinetic energy by (1/μ² − 1) K of the velocities scaled, exactly; the positions change the potential energy by −ln μ tr W to first order, with W the virial of the groups (D72) and of the constant terms, taken as the mean of those before and after the step, exact to second order in the strain but not in the offset of half a step between those virials and the scaling: the conserved energy then drifts, growing with N_P as the paper's effective energy does (its Fig. 2c). Measured on the Lennard-Jones mixture of `barostat.test` (5 seeds, 80 ps, τ_P = 1 ps), the drift per step is (0.7 ± 0.6) × 10⁻⁶ k_BT at N_P = 2, (3.6 ± 2.4) × 10⁻⁶ at 10, (2.2 ± 1.2) × 10⁻⁵ at 100; counted exactly it is within 3 × 10⁻⁷ of 0, and to first order about 4 × 10⁻³. In the compression of `barostat-ff19sb-gpu.test` (a fifth of the volume in 10 ps) 1.1 × 10⁻² of the value against 4 × 10⁻³ exactly. The trajectory does not depend on the count. With a period of one step every step scales, and its pressure is that of the state that the step before left: the trace of the virial, that of the rigid groups, and the kinetic energy without the center of mass after the thermostat, three numbers that the run keeps on the host after each step (`%trotter_memory`) and the checkpoints keep (`/parameters/mdir/barostat_state`), so that a run that continues ends in the state of one that does not stop, with rigid water and PME as well (`barostat-pme-restart.test`); from a checkpoint without them the state is evaluated once. `work = "TROTTER_FIRST_ORDER"` counts the energy of a scaling from the virial before it only, to first order, as `"FIRST_ORDER"` does, and the step that scales computes no virial: the same trajectory, a conserved energy that drifts by 2e-3 of its value on the mixture of `barostat.test` instead of 3e-4, and on JAC under NPT (RTX 3090) 527 ns/day against 464 (D93). `work = "EXACT"` and `"FIRST_ORDER"` remain. |
| D93 | **A loop sums only the elements of a vector that are used, and the virial of pairs is the outer product of d and the vector of the forces.** `md-exec-narrow-sums` follows each vector result of a loop back from its users, through elementwise arithmetic and `vector.extract`, and makes the loop sum only the elements used, in the type its kernel computes in; the pressure takes the trace of the virial, three of its nine elements. `md-differentiate` writes the virial of pairs as d ⊗ (f d), whose f d is the vector of the forces that a fused loop computes once, rather than f (d_a d_b), into whose elements the simplification of distances multiplied each term of f. On JAC under NPT (RTX 3090), the loop over pairs of energy and virial takes 80 µs, from 137, against 67 for the forces alone. With these, the barostat of Trotter type (two loops with the virial each period) and the exact work (one with the virial and one with the energy) cost the same: 423 and 428 ns/day, within the noise. |
| D94 | **Under `fast_math`, what a pair kernel computes of the distance alone beyond roots and powers is a table in f32.** `md-exec-simplify-distance{radial=true}` groups the terms of a value of the kernel by their factors free of the distance, and a group whose part that depends on it takes an exponential, erfc, or the like becomes those factors times `md_exec.radial` of a function of the module, `(f64) -> f64`: the screened Coulomb force of Ewald summation is q_i q_j times one function of r², its energy another. `md-exec-expand-radial`, after the precision is assigned, makes it in an f32 kernel a table over r² from 2^-10 of the cutoff squared to it: the intervals that the bits of r² as an f32 number name, 2^b of them for each power of two, and in each a cubic in u = r² − (its start), exact in f32, fitted at four Chebyshev nodes in f64; b is the fewest from 4 to 12 whose cubics are within 3e-8 of the function in f64, relative to its largest value on each interval, and the pass fails unless they are within 3e-7 in f32, as a kernel evaluates them (17 points an interval). A table is a constant of the module, four f32 numbers an interval, read with one load of 16 bytes; tables that are the same are one. In an f64 kernel the function is inlined. On JAC (cutoff 8 Å, b = 7, 1281 intervals, 20 KB a table), the Coulomb energy at the start is −62721.9863 kcal/mol against −62721.9880 in double precision, from −62721.9826; the loop over pairs at a reach of 9 Å takes 65.4 µs, from 73.1, and on Cellulose 1113, from 1201 (RTX 3090). |
| D95 | **The loop over groups takes no minimum image: the build moves each place and each entry by whole cells into the frame of its group.** The frame of a group is its first particle where it is wrapped and the others at their places relative to it, as the build takes them. The box kernel stores for each place P, the cells that move its particle, as it is kept (unwrapped), into the frame, ten bits an axis from −512 (`GROUPS_SHIFTS` of the runtime), and W, the cells between the frame and the wrapped position, −1 to 1, in the unused fourth lane of the wrapped positions. The list kernel stores in bits 16 to 24 of the mask of an entry e = W_g − W_q − k, with k the cells of the minimum image of the candidate from the center of the box, which it computes anyway: −2 to 2 an axis, as e + 2 in three bits, computed in the packed form of W, where the fields cannot carry, and only for a candidate with pairs. The gather of the loop adds P·L in f64 before the positions become f32, so they stay near the cell however far the particles have gone; the kernel adds e·L to the position of the entry and subtracts. A frame anchored on the first particle as it is kept would let e grow with the drift of two groups apart and wrap in its three bits; the test of the template places the particles −3 to 3 cells from the cell to check it. On Cellulose (reach 9 Å, RTX 3090), the loop over pairs takes 35 µs a step less and the build 27 µs more (a build 90 µs more, every 4 steps), 3397 to 3383 µs a step of the device; on JAC, 15 µs less for the loop, 1002 to 966 µs. |
| D96 | **The tables that a kernel reads at the same entry are one table of vectors.** After `md-exec-fold-tables` has made its tables, the lookups of one kernel at the same indices become one `md.lookup` of a table whose entries are `vector<N x f64>` (`vector<N x f32>` once narrowed for the device), its kernel the kernels of those tables one after another, and `vector.extract` takes each value. `md.table` and `md_exec.tabulate` take vectors of f64 or f32 as elements; a table that only merged tables read is dropped. The loop over groups is bound by the pipe of loads and shuffles (92% of its peak on Cellulose, Nsight Compute, RTX 3090), with three loads and nine shuffles a pair: the two coefficients of Lennard-Jones, 48 ε σ¹² and 24 ε σ⁶, were two loads of 4 bytes and are one of 8, as pmemd.cuda reads them (one load of 8 bytes and one of 16 for its spline of erfc a pair, eight shuffles). On Cellulose the loop over groups takes 1040 µs a launch (median), from 1078. Energies at the start are unchanged to the digit. |
| D97 | **In the loop over groups the mask of an entry does not turn with it: each lane takes its bits by ballots.** At each round of 32 entries, a ballot for each of the 16 particles of the group gives the bits of that particle with every entry; the lane of particle u keeps word u, the 16 bits of its half-warp, rotated left by 15 − u, so that at step k bit 15 is the pair with the entry it then holds, and each step shifts it by one. Sixteen ballots and selections a round take the place of sixteen shuffles: the shuffles a pair are 8, from 9, on the pipe of loads and shuffles that bounds the loop (D96). The shifts of the entries (bits 16 to 24 of the mask, D95) are read before, at the load. On Cellulose the loop takes 1006 µs a launch (median), from 1040, and the device 3343 µs a step with D96, from 3405; on JAC the loop 59.4 µs, from 62.9 (RTX 3090). Energies unchanged to the digit. |
| D98 | **On a device an approximate division in f32 is a product with an approximate reciprocal that flushes subnormal numbers.** `convert-md-exec-to-gpu` makes an `arith.divf` in f32 marked `afn` (D90) in a kernel `x * nvvm.rcp.approx.ftz.f(y)`, and `1 / y` the reciprocal alone. The division of the backend, `div.approx.f32`, scales its operands where they are subnormal or beyond 2^126 (two comparisons, two selections, and a product more a pair in the loop over groups); a subnormal number does not occur in the arithmetic of a pair, and the reciprocal is within 1 ulp. The 16 steps of a round of the loop over groups count in i32, not in an index of 64 bits. On Cellulose the loop over groups takes 983 µs a launch (median), from 1006, and the device 3318 µs a step, from 3343 (RTX 3090). |
| D99 | **The build of the lists of groups queues the candidates near the box of a group and tests them 32 at a time.** A lane a candidate, the warp of a group tested each against the 16 particles of the group where it lay within the reach of the box: a third of the lanes on Cellulose (10 of 32, Nsight Compute), on the third of the instructions of the kernel. Each lane now only tests its candidate against the box; those near it go to a queue of the warp in the memory of the block (64 places a warp), in the order of the lanes, and when it holds 32 the warp takes them, a lane each: the particles of the group, the excluded pairs, the shift of D95, and the entries, as before; the rest of the queue moves to its front, and what is left at the end is taken by the lanes it fills. The entries keep an order that does not depend on the threads. On Cellulose the kernel of the lists takes 1508 µs a build (median), from 2098, and the device 3293 µs a step, from 3422 (RTX 3090); on JAC the build 178 µs a step, from 217, and the device 937, from 980. |

### 5.1 Amendments to earlier decisions

A1 to A10 come from an external review of revision 3 of the architecture,
A13 from one of revision 4.

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
| A10 | **M2 has three parts.** M2a: constraints and virtual sites. M2b: PME on one node. M2c: distributed PME. | P3 *Amended by A14.* |
| A12 | **In the storage form, the ops of a block run in the order of the block.** The ops declare which buffers they read and write, so the upstream analyses keep that order where it matters. Event tokens (D10) are added when ops run asynchronously; until then nothing needs them. | D17 |
| A11 | **Rebuilds are checked after the fact.** With the fixed-interval policy, the maximum displacement since the previous rebuild is measured at each rebuild. Violations of the validity condition are counted and reported. *Amended by B1.* | P15 |
| A13 | **The random number key has a draw index**: seed, step, stream ID, entity key, and draw index. A stream ID names a purpose, such as a thermostat or a kind of move. The draw index counts the numbers that one entity takes from one stream in one step, from 0, and is a constant in the kernel that draws. The numbers then do not depend on the order in which kernels run or in which a kernel is evaluated. | P5, A5 |
| A14 | **Constraints and particle mesh Ewald on one node move to M1** (D53). M2a keeps virtual sites; distributed particle mesh Ewald stays in M2c. | A10 |

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
| B4 | **Truncation is an attribute of a relation sum.** The kinds are `none`, `shift`, `force_shift` [[Stoddard1973]](references.md#stoddard1973), `switch`, and `force_switch` [[Steinbach1994]](references.md#steinbach1994); the last was added for parity with GROMACS [[GromacsManual2025]](references.md#gromacsmanual2025). A pass expands it into the kernel before differentiation. Energy conservation is validated with `force_shift` or `switch`. | New |
| B5 | **Derivatives of functions that are not smooth have fixed conventions**, including the branch taken at a tie. | New |
| B6 | **Comparing velocity Verlet with leapfrog maps the initial velocities**: `v(−dt/2) = v(0) − (dt/2) · F(0) / m`. | New |
| B7 | **Both relation kernels receive the distance and the displacement vector.** | New |
| B8 | **The virial is `W = Σ d_ij ⊗ K(i, j)`** [[Louwerse2006]](references.md#louwerse2006), positive for repulsion, with `P = (2 E_kin + tr W) / (3V)`. *Amended by D45, which says which kinetic energy the pressure takes.* | New |
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
| Driver, TOML input, XTC and H5MD output | M0 | Driver, TOML input, and DCD output are implemented; see [driver-m0.md](driver-m0.md), Section 4 |
| Checkpoints in H5MD, restart from a checkpoint | M0 | Implemented; a run that continues is exact |
| Trajectory in XTC | M0 | |
| GPU back end | M0 | Implemented for NVIDIA: `convert-md-exec-to-gpu`, `libmdrt_cuda` |
| Global sums of vectors on a device, for the virial | M0 | Implemented. The driver writes the virial and the pressure to the log. |
| GPU back end for AMD | M1 | |
| Check that particle set symbols in types are declared | M0 | |
| Regression test for the numerical values of derivatives | M0 | Implemented with a kernel runner |
| `mdrt` ABI: storage, neighbor structures, events | M0 | Proposal in [mdrt-m0.md](mdrt-m0.md) |
| Schema of the TOML input | M0 | |
| Layout of the MDIR group inside an H5MD checkpoint | M0 | Decided (D40); see driver-m0.md, Section 2.6 |
| HDF5 development files | M0 | Installed by `scripts/build-hdf5.sh` |
| Storage assignment pass | M0 | Specified in ops-m0.md, Section 10 |
| Lowering of transcendental functions on GPU targets | M1 | Works on NVIDIA through `libdevice`; see ops-m0.md, Section 3.3. Open for AMD. |
| Syntax for combining relations | M1 | |
| Streams and draw indices of the random numbers of a thermostat (A13) | M1 | |
| The deterministic level as a constraint on the plan | With the planner | Note: the level says what the planner may choose, not what it prefers. It excludes a sum whose order threads decide. The lowerings of M0 add up in a fixed order on a device. On the CPU the order of a reduction is that of the OpenMP runtime: the tests find the same bits from run to run with the same number of threads, which the OpenMP standard does not promise. |
| The dependency interface on domains other than particles | M2b | Note: the planner should reason about stages on a domain, of which the particles of a set are one and a mesh is another, so that PME does not need another planner. `ParticleDependencyInterface` (D4, A3) is not implemented yet. |
| Scatter strategy for bonded terms | M1 | Decided (D49) |
| Long-range dispersion correction | M1 | |
| Distributed fields and grids | M2c | |
