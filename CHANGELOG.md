# Changelog

All notable changes to MDIR are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the versions
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before
1.0, a minor version (0.x) may change the control file, the checkpoint
format, or the outputs; every such change is listed under **Changed** or
**Removed** with what a user has to do. The decision numbers (Dnnn) refer to
[docs/decisions.md](docs/decisions.md).

## [Unreleased]

### Added

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

### Changed

- Refined the proposed distributed architecture and roadmap around field
  versions, owner–replica maps, coverage, and contribution completion, with
  reference-execution and verifier gates before transport and asynchronous
  scheduling (D[md-dist-architecture]). Documentation only; no distributed
  implementation or control-file changes.

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

## [0.1.0] - unreleased

The first milestone (M1): an all-atom protein in water with an Amber force
field, compiled before the run for one CPU (OpenMP) or one NVIDIA GPU, at
least as fast as pmemd.cuda on every system of the Amber benchmark suite.

### Added

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
- **The cell keeps its shape at constant pressure.** The barostat does not
  couple the shape of the cell.

[Unreleased]: https://github.com/vv137/mdir/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/vv137/mdir/releases/tag/v0.1.0
