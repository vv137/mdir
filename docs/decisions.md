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
| D60 | **The linear momentum is removed every `comm_period` steps**, 100 by default, by a sum over the particles and a subtraction that fuses with the kick. The angular momentum is left alone under periodic boundaries. |
| D61 | **The members of a relation of a topology are the places of the particles in their fields**, and are renumbered where the particles are put in a new order, so a loop needs no table from numbers to places. The incidence structure has rows as wide as the particle with the most tuples needs, found when it is built. A sum over tuples takes the contribution of a tuple from its member at place 0, with no weight. |
| D62 | **MDIR has one form for each term of the potential and one meaning for each parameter**, which [conventions.md](conventions.md) states: harmonic bonds and angles as `½ k (x − x0)²`, periodic dihedrals as `k (1 + cos(n φ − φ0))` with the IUPAC sign of the dihedral, Lennard-Jones as `4 ε ((σ / r)¹² − (σ / r)⁶)` with `σ` and `ε` for each pair of types, Coulomb with the constant of CODATA 2018. Readers convert into these forms, and one library of the front end builds the kernels from them, whatever the format. |
| D63 | **The molecules that SETTLE constrains can be chosen.** From a GROMACS topology, those with `[ settles ]`; from an Amber topology, the residues named in `settle_residues` of `[constraints]`, `["WAT"]` by default. A named residue that is not a rigid water of three sites is an error. `fast_water = false` constrains the same water with SHAKE. |
| D64 | **MDIR stays under the MIT license.** Its readers are written from specifications of the formats, with no code of GROMACS or Amber, and it links no library under the GPL: the FFT of particle mesh Ewald is cuFFT on a device and a library under a permissive license, such as pocketfft, on the host; FFTW is not used. |
| D65 | **The target of M1 is ff19SB with OPC**, besides ff14SB with TIP3P (D53): CMAP and virtual sites, with the extra points of Amber and the virtual sites of GROMACS for water of four sites, move to M1. |
| D66 | **A step runs on the device without waiting for the host**, as far as possible: the test of validity of the neighbor structures, the thermostat, the barostat, and the constraints are decided on the device, so that the host waits only where it writes output, and a step can be one graph of kernels. The work of the host that is left, such as the incidence structures of a segment, moves to the device where it is measured to matter. |
| D67 | **The velocities are coupled at the end of a step, every `thermostat_period` steps**: the motion of the center of mass is removed, then stochastic velocity rescaling scales the velocities. The period of coupling is a loop of the schedule, as the intervals between energies are, so a step between couplings has no global sum. The thermostat takes the kinetic energy of the velocities it scales, without that of the center of mass: with velocity Verlet those of the end of the step, with leapfrog those half a step later. `comm_period` equals the period of the thermostat, which is 10 by default; without a thermostat the motion is removed only if `comm_period` is given, which amends the default of D60. The random numbers are drawn on the host from the key of A13 (stream 0, the step, entity 0), so a trajectory does not depend on how the steps are grouped into loops or on a restart. The energy that the coupling takes is written as the conserved energy. Drawing on the device (D66) waits for random numbers in kernels (Section 11.3 of design-m1.md). |
| D68 | **A virtual site is a particle of mass 0 in the particle set**, placed from its atoms in the programs of the step after every drift, with its force moved to them before the kick, both over a tuple set of the sites. The neighbor structures, the new orders of the particles, and the output take sites as they take atoms. The extra point of Amber keeps the placement of sander, which is not linear in the positions, and GROMACS keeps its linear combination; neither stands in for the other (design-m1.md, Section 19). |
| D69 | **The reciprocal sum of particle mesh Ewald is one op, `md.reciprocal`**, which yields the energy and, when differentiation asks, the forces and the virial from the grid; it is not differentiated through. The direct sum, the excluded pairs, the self term, and the net charge need no new op. The op lowers to templates in IR for spreading, the product with the influence function, and gathering, on the CPU and on a GPU, and to calls of the FFT of the runtime: pocketfft on the host and cuFFT on a device (D64). The driver computes the influence function as a table ([pme-m1.md](pme-m1.md)). |
| D70 | **The grid of particle mesh Ewald is accumulated in fixed point**, 64-bit integers at the scale 2⁴⁰ with integer atomics, on the host and on a device, so that the sum does not depend on the order of the threads (P6) [[LeGrand2013]](references.md#legrand2013). |
| D71 | **β, the grid, and the order of particle mesh Ewald can each be given**, or follow from a tolerance (`erfc(β rc) = 10⁻⁵`) and a largest spacing (1.2 Å), so that a run can take the parameters of sander or of GROMACS. The direct sum is not shifted at the cutoff by default, as in sander; `pme_shift` shifts it, as GROMACS does. The influence function is that of Essmann et al. by default, as in GROMACS; `pme_influence = "OPTIMAL"` scales it by the factor of the aliasing of the B-splines that sander applies by default ([pme-m1.md](pme-m1.md), Section 1.1). |

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
| Rebuild policy `interval`, with the diagnostic of A11 | M0 | |
| Storage form of the `md_exec` ops, as D17 decided | M0 | Implemented: `md-exec-assign-storage` |
| Precision policy | M0 | Implemented: single, mixed, and double, on the CPU and on a GPU |
| Declaring the role of a field that a function takes | M0 | |
| Vectorization of loops over pairs across pairs | M0 | Without it, single precision is no faster than double on the CPU |
| Fusion of loops over the same neighbor structure | M0 | Implemented |
| Fusion of loops over particles | M0 | Implemented |
| The test of validity in the loop that writes the positions (D41) | M0 | Implemented: `md-exec-expose-validity` |
| Fusion of the loop over pairs with the kick that follows it | M0 | |
| The particles in the order of their positions (P17, D44) | M0 | Implemented: `md_exec.spatial_order`, `md_exec.permute`, and the keyword `reorder` |
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
