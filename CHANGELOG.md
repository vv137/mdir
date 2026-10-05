# Changelog

All notable changes to MDIR are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the versions
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before
1.0, a minor version (0.x) may change the control file, the checkpoint
format, or the outputs; every such change is listed under **Changed** or
**Removed** with what a user has to do. The decision numbers (Dnnn) refer to
[docs/decisions.md](docs/decisions.md).

## [Unreleased]

### Changed

- In the deterministic mode (`deterministic = true`) device kernels no longer
  contract products and sums into fused multiply-adds, so that a step that
  writes energies moves the particles as one that does not
  (D201, #97). JAC in mixed precision is about 2.5%
  slower in this mode; the default mode is unchanged.

- Replace the Python preview's flat numeric lists with shaped, read-only
  NumPy copies and strict buffer/CPU DLPack inputs. Assign native float64
  `(N, 3)` coordinate arrays and copy outputs before editing; tuple IDs use
  shaped int32/int64 inputs. Enabling Python now requires NumPy >=1.23
  and checks it at configuration and import (D193, #78).


- `[free_energy]` with one state writes `dHdl.<name>` alone: the `dU.0`
  column, identically 0, is gone, and the state energy is no longer
  evaluated. `--continue` onto a file written before stops at its column
  header; continue with `--no-append` or start a new file.
  `scripts/free-energy.py` refuses a one-state run, since TI and MBAR need
  two or more states (D190, #71).

- `observe` is a reserved key of every term table given by an expression
  (D189); a constant named `observe` must be renamed, with its uses in
  the expression.

- Custom term parameters now reject names supplied by the term, declared
  particle parameters, functions, and lambda components, including unused
  declarations. Control files with colliding parameters must rename the
  parameters and their expression uses (for example, change a bond constant
  `r` to `r0`) (D188, #67).

- On a GPU, adjacent bonded loops are launched as one kernel, and the sums
  over few tuples, as the centers of groups, take one kernel instead of
  two. A bias on the distance between two centers of groups costs JAC
  about 9 µs a step instead of 12.5, as much as OpenMM's
  `CustomCentroidBondForce`, and JAC without it runs 3.7% faster
  (D181, #20).

- On a GPU, outside the deterministic mode, the sums over the centers of
  a term over the centers of groups are computed by a block of the kernel
  of the bonded terms, which then evaluates the term and its forces: the
  term takes no kernel of its own. A bias on the distance between two
  centers costs JAC 2.7 µs a step instead of 9.2 (D194,
  #66).

### Added

- Python: a simulation of a program whose integrator minimizes runs the
  minimizer of `mdir run`, `Simulation.minimize(steps=None)`, in parts
  that `request_stop()` and Ctrl-C end; `State.minimization` holds the row
  of the log at its last step (D202, #91).

- Python: setters of the model whose value has a unit also take an
  `openmm.unit.Quantity` (cutoffs, time step, temperatures, pressure,
  restraint constants, positions, velocities, cell, restraint reference,
  and others), converted to MDIR's units; a list of `Vec3` in a quantity
  becomes a float64 array. OpenMM stays optional. A NumPy array wrapped in
  a quantity is no longer stored with its unit dropped
  (D200, #95).

- Python: `InitialState.draw_velocities(system, temperature, seed)` draws
  the velocities that `mdir run` draws for the same input, bit for bit, in
  a new state; `System.restraints`, a list of `mdir.Restraint` (selection,
  force constant in kJ/mol/nm², `ReferenceScaling`), and
  `System.restraint_reference` map onto `[[restraints]]`
  (D198, #92).

- Roadmap: M2 is split into M2a, the Python API, and M2b, differentiable
  simulation (fitting parameters to ensemble averages by trajectory
  reweighting, PyTorch then JAX) (D195).

- `mdir.Simulation`: a Python simulation that persists across `run(n)` of
  any number of steps, with the coupling at the same steps as an
  uninterrupted run, `state()` as NumPy copies, the energies of a final
  step of energy (`run(n, energy=True)`) equal to the row of `mdir run` at
  that step, stops between parts, and errors that leave Python alive
  (D196, #85). Host JIT code keeps disjoint exception-frame
  ranges across engine destruction, and failed parts finish runtime cleanup
  before their state is discarded. Reporters and checkpoints follow separately.

- Optional Python loaders and typed model inputs, explicit shared compiler
  lowering, immutable IR/plan inspection and tracked stale detection
  (D192, #76). Persistent execution and wheels follow separately.

- Native C++ model preparation for the M2 Python API: owned Amber,
  GROMACS, and CHARMM data, physics apart from initial state, typed
  integrator/ensemble/execution options, MD-unit custom expressions,
  and shared validation and IR construction (D191, #74).
  Python bindings and persistent simulations follow separately.

- `observe = [...]` in a term given by an expression writes the term's
  energy and its derivatives in the listed constants, the generalized
  forces along them, to `[output] observables` at every energy, without
  `[free_energy]`: for example the force on a wall, and so an osmotic
  pressure (D189, part of #16).
- M2 Python API preparation: driver prerequisites, proposed implementation
  sequence, adopted maintainer contracts, and validation gates in
  `docs/python-m2.md`
  (D187); this preparation adds no executable Python API.
- Tabulated functions can read whitespace-separated numeric grids from
  `values_file` with an explicit `shape`, with the existing interpolation
  in one, two, or three dimensions. Manifests hash the files, and
  continuation checks the loaded grid (D178).
- Differentiation diagnostics carry control-file term names, including an
  unknown operation inside a captured kernel whose failure is reported
  through a sum or field (D186, part 5 of #15).

- Explicit differentiation conventions at coincident particles, active and
  inactive `sqrt(0)`, absolute-value and min/max ties, dependent select
  conditions, and table boundaries, with direct singular-point regression
  tests (D185, part 4 of #15).

- `--md-check-derivatives` instruments evaluations with f64 central
  differences at their actual inputs: every requested force component,
  scalar parameter derivative, and six independent cell strains. Failed
  checks print the analytic value, numerical value, error, and tolerance
  before stopping. The checker requires CPU double precision and an
  explicit cell for virials (D184, part 3 of #15).

- Scalar derivative rules use `DerivativeOpInterface`, with external
  arithmetic, math, and vector models and explicit differentiable or
  structural operand roles. `--md-check-derivative-coverage` rejects scalar
  operations without a rule or declared zero (D183, part 2 of #15).

- Shared three-outcome activity analysis for scalar, position, cell, and
  parameter differentiation. `--md-analyze-activity=argument=N` reports
  active, proven inactive, and unknown results; required unknown activity
  fails differentiation (D182, part 1 of #15).

### Fixed

- The virial of SETTLE and SHAKE at a step that writes energies reads the
  velocity changes the step applies instead of solving the constraints again
  (D201, #97).

- The host code of a program is compiled at LLVM's default level, which is
  the level it always had: the requested `Aggressive` level was not applied
  by MLIR's execution engine, and measured no faster (D197,
  #90).

- Persistent simulations check allocated host code and relocated unwind ranges
  before registering exception frames and deregister before freeing object
  storage. Creation and teardown synchronize with execution. Unsupported
  layouts fail compilation with an ownership diagnostic; no control-file keys
  or output formats change (D199, #89).

- Compound terms accept declared `lambda_<name>` components
  (D188, #67).

- GPU PME preserves mapped-charge producers when moving a reciprocal sum
  ahead of a neighbor refresh. Free-energy runs with spatial ordering,
  checkpoints, and continuation now use the mapped charges directly,
  replacing the host-computed charge workaround
  (D179, #26).

- `md-exec-assign-storage` asserted that "the buffers of a loop are
  conserved" when a loop carried a field it did not read, or when an op in
  a loop produced a field that nothing used. Both now free the buffer. The
  Brownian step no longer reads the velocities to avoid it, and a particle
  of mass 0 (a virtual site) gets velocity 0 instead of keeping its old
  velocity (#19).
- GPU mixed-precision free-energy derivatives with `POTENTIAL_SHIFT` no
  longer fuse an approximate quotient into the cutoff subtraction. The
  shifted Lennard-Jones derivative is checked against OpenMM within
  1e-5 kcal/mol, replacing the widened 1e-3 tolerance
  (D180, #48).

## [0.1.0] - 2026-10-04

The first milestone (M1): an all-atom protein in water with an Amber force
field, compiled before the run for one CPU (OpenMP) or one NVIDIA GPU, at
least as fast as pmemd.cuda on every system of the Amber benchmark suite.

### Added

- The binary tarball of a release, `mdir-VERSION-manylinux_2_28_x86_64.tar.gz`,
  runs on Linux x86-64 with glibc 2.28 or later and carries HDF5, cuFFT, and
  libdevice, so a GPU run needs only the NVIDIA driver; `CUDA_ROOT` is
  optional (D177).
- `cmake --install` installs `bin/mdir`, its runtime, and the OpenMP runtime
  under one prefix, and the log of a run begins with the version and the
  commit of the build (D174).
- A container recipe, for Docker and Apptainer, with LLVM/MLIR 23.1.2, HDF5,
  and CUDA (D174).
- Barostat runs report the area of the xy face of the cell, `AREA_XY` in
  the log and `area_xy` in the energy file, in Å² (D170).
- Checkpoints record a fingerprint of the run's physics, coupling, and
  execution; `mdir checkpoint --print=fingerprint` lists it (D172).
- `lennard_jones_modifier = "SQUARED_DISTANCE_SWITCH"`, CHARMM's switch of
  the Lennard-Jones potential in r² (VSWITCH; Brooks et al. 1983), between
  `switch_distance` and `cutoff`, for topology Lennard-Jones and 1-4 pairs;
  agrees with CHARMM 51b1 on two POPC within 3.7e-7 kcal/mol (D175).
- Alchemical free energy (D161). `[free_energy]` decouples a selection of
  whole molecules (`couple`, a mask) through states of λ given in
  `[free_energy.lambdas]` (`coulomb`, `vdw`, and any other component, a
  parameter `lambda_<name>` of the expressions of `[energy]`), with the
  soft-core Lennard-Jones of Beutler et al. (`soft_core_alpha`,
  `soft_core_power`). `[output] free_energy` writes dH/dλ of each
  component and the energy of every state relative to the run's at every
  energy; `scripts/free-energy.py` gives the free energy by thermodynamic
  integration and MBAR. Checkpoints record the states and the run's state.
  The Lennard-Jones takes a plain cutoff or the potential shift; a switch
  (`switch_distance` below `cutoff`, `FORCE_SWITCH`, `POWER_FORCE_SWITCH`,
  `SQUARED_DISTANCE_SWITCH`), implicit solvent, and `lennard_jones = "PME"`
  are refused with `couple`.
- `--md-differentiate=remarks=true` lists the ops that a derivative with
  respect to a parameter takes as independent of it (D161).
- **The compiler.** The `md` dialect states potentials and their particle
  sets, and semantic differentiation derives forces and the virial from
  them. `md_exec` chooses neighbor structures, storage, and precision, and
  lowers to loops on the CPU or to kernels on a GPU. `dyn` describes the
  step. The program is compiled once, just before the run.
- **Inputs.** Amber topologies and coordinates; GROMACS topologies with
  their includes (D121); the files of CHARMM: PSF, CRD, topology,
  parameters, and streams (D122); PDB.
- **Force fields.**
  - Harmonic bonds and angles, proper and improper dihedrals, CMAP,
    Urey–Bradley terms, and exclusions with scaled 1-4 pairs.
  - Tables of Lennard-Jones by pairs of types, with NBFIX.
  - Switching and CHARMM's force switch, and the long-range correction of
    the dispersion.
  - SHAKE and SETTLE, and virtual sites.
- **Electrostatics and solvent.**
  - Particle mesh Ewald in orthorhombic and triclinic cells (D123, D127).
  - Reaction field (D140), and runs without a periodic cell (D142).
  - Generalized Born, OBC and HCT, with salt and the radii of mbondi2
    (D144, D152).
  - Particle mesh Ewald for the dispersion (LJ-PME, D162).
- **Custom terms**, as the custom forces of OpenMM:
  - pair terms (D137);
  - terms over bonds, angles, and dihedrals (D136);
  - terms of absolute positions (D148);
  - terms over the centers of groups, for pulling, with the time `t`
    (D139, D145);
  - tabulated functions of one, two, or three arguments, continuous or
    discrete (D138, D165);
  - parameters of each particle, and compound terms over listed particles
    (D165);
  - terms over triplets, on the CPU (D160).
- **Free energy** (D161): alchemical decoupling of a selection over λ
  states, with soft-core, dH/dλ, and the energies of every state for MBAR.
- **Dynamics.**
  - Integrators: velocity Verlet, leapfrog (D76), Langevin dynamics by the
    middle scheme (D135), and Brownian dynamics (D163b). Steepest-descent
    minimization.
  - Thermostats: velocity rescaling, Nosé–Hoover chains (D163a), and
    Langevin.
  - The stochastic cell-rescaling barostat: isotropic, semi-isotropic
    (D119), or anisotropic (D163c), for orthorhombic and triclinic cells.
  - Positional restraints (D74).
- **Execution.**
  - Precision modes `single`, `mixed`, and `double`, and an opt-in
    deterministic mode.
  - On a GPU: neighbor groups with a dual list (D114).
- **Outputs.**
  - Logs and files of energies and of the coordinates of pulling (D149).
  - Trajectories in DCD or XTC (D141).
  - Checkpoints in H5MD that continue a run on the CPU or on a GPU (D129).
  - Numbered backups of existing files (D149).
  - An optional run manifest of provenance (D168).
- **The `mdir` command.**
  - Subcommands: `run`, `check`, `template`, `checkpoint`, `doctor`,
    `version`, `bug-report`, and `emit`.
  - Warnings of the system for what a run does that its user may not
    intend (D153, D158).

### Changed

- `mdir run --continue` refuses a checkpoint whose physics or coupling
  differ from the control file and names each change. `[execution]`,
  `[output]`, and a larger `steps` may still change. To change a
  temperature, restraints, or another key between stages, start the next
  stage from the checkpoint through `[input] checkpoint` (D172).
- A run that starts from another run's checkpoint with different physics
  evaluates its forces and barostat state at its first step, instead of
  taking the stored ones. A run whose only difference is execution
  continues bit for bit (D172).
- Checkpoint format 1 is the contract of this release. Files from newer
  formats, and files from development builds before it, are refused
  (D173).
- On the CPU every reduction of a parallel loop is summed in a fixed order
  over fixed chunks, so a run gives the same bits with any number of threads
  and across a continuation. The last bits of single-threaded runs change
  once (D171).

### Fixed

- A run whose positions became NaN or far beyond the cell could abort on a
  GPU, with or without a message, instead of stopping with "positions are
  not numbers"; the groups build now counts such positions and the run
  always stops with the message (D176, #22).
- A Lennard-Jones pair of epsilon 0 gave NaN in mixed precision where two
  particles all but meet; it now contributes exactly 0, also under a
  switch (#18).
- A checkpoint carries the SHA-256 of its state, checked on every read, and
  is flushed to stable storage before it replaces the previous one (D173).
- Multithreaded CPU runs with a thermostat or a barostat were not
  reproducible from run to run, even with `deterministic = true` (D171).
- The NPT continuation of a run's own checkpoint no longer warns that the
  cell differs from the input (D172).
- `mdir version` and `mdir doctor` no longer print `CUDA unknown` when the
  toolkit has no `version.json`, as in CUDA's runtime images. They take the
  version from a loaded CUDA runtime or the toolkit's `version.txt`, and
  otherwise say `runtime version not reported`. The toolkit is the one
  `CUDA_ROOT` names, if set. The line also gives the CUDA version that the
  driver supports (`driver API`) (D174).
- A driver too old for the PTX ISA version of the kernels ends the run with
  an error that names the driver's CUDA version and the PTX ISA version and
  asks for a newer driver, instead of a generic failure of
  `cuModuleLoadDataEx` (D174).

### Validated

- Every term against sander or pmemd, GROMACS 2026.3, CHARMM 51b1, or
  OpenMM 8.6.1 on the same systems, and the custom terms against OpenMM's
  custom forces. Each comparison is pinned by a test that names the program
  and its value (white paper, Section 9).
- The ensembles: kinetic-energy distributions against the canonical one,
  and the volume distributions at constant pressure. The conserved energy
  is checked over long runs (Sections 6 and 9).

### Performance

RTX 3090 capped at 300 W, mixed precision, the Amber 24 benchmark suite,
mean ± standard deviation of three runs (white paper, Table 10.2):

| System | Atoms | MDIR, ns/day | pmemd.cuda, ns/day | Ratio |
|---|---|---|---|---|
| JAC NVE | 23,558 | 814.6 | 619.3 | 1.32 |
| JAC NVE 4 fs | 23,558 | 1520.3 | 1153.3 | 1.32 |
| JAC NPT | 23,558 | 745.2 | 587.7 | 1.27 |
| JAC NPT 4 fs | 23,558 | 1437.7 | 1133.3 | 1.27 |
| Factor IX NVE | 90,906 | 290.4 | 264.1 | 1.10 |
| Factor IX NPT | 90,906 | 273.4 | 250.4 | 1.09 |
| Cellulose NVE | 408,609 | 64.0 | 61.3 | 1.04 |
| Cellulose NPT | 408,609 | 61.4 | 58.3 | 1.05 |
| STMV NPT 4 fs | 1,067,095 | 41.6 | 37.5 | 1.11 |

### Known limitations

- **One device per run.** A run uses one CPU or one GPU; distributed
  execution is milestone M4.
- **No polarizable or learned potentials.** There is no Drude, AMOEBA, or
  MLIP support; learned potentials are milestone M3. CHARMM's lone pairs
  are not supported.
- **Triplet terms run on the CPU only.**
- **The deterministic mode does not take neighbor groups.** It keeps the
  neighbor matrix and its lower rate.
- **Free energy is limited to decoupling.** Relative (A to B)
  transformations, annihilation, and replica exchange are not supported.
  A decoupled selection takes a plain Lennard-Jones cutoff or
  `POTENTIAL_SHIFT`; switches are refused. On a GPU in mixed precision with
  the shift, dH/dλ of the Lennard-Jones differs from double precision by
  about 5e-4 kcal/mol at states with λ_V > 0 (#48).
- **A per-step field of the particles feeding the reciprocal sum** gives
  positions that are not numbers on a GPU when the particles are reordered
  (#26). Free energy avoids it in its step program and is tested on its
  output path; no other feature uses that pattern.
- **The cell does not shear at constant pressure.** The barostat scales
  each axis (isotropic, semi-isotropic, or anisotropic, D163c), without
  coupling the angles of the cell.
- **Binaries** are built for manylinux_2_28 (glibc 2.28 or newer: RHEL,
  Rocky, and Alma 8+, Ubuntu 20.04+, Debian 11+) and x86-64 with an NVIDIA
  driver supporting CUDA 13.0 or newer for the GPU (D177).

[Unreleased]: https://github.com/vv137/mdir/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/vv137/mdir/releases/tag/v0.1.0
