# Changelog

All notable changes to MDIR are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the versions
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before
1.0, a minor version (0.x) may change the control file, the checkpoint
format, or the outputs; every such change is listed under **Changed** or
**Removed** with what a user has to do. The decision numbers (Dnnn) refer to
[docs/decisions.md](docs/decisions.md).

## [Unreleased]

## [0.2.1] - 2026-10-26
### Added

- **The suite checks which tests are unsupported**: `test/expected-unsupported.txt`
  lists them by lit feature, and a run of the whole suite warns on a test
  that is unsupported but not listed, or listed but run
  (`-Dexpected_unsupported=strict` makes it an error) (#608).
- **An equivalence check of the builder**:
  `scripts/builder-equivalence/check.py` shows that two builds of `mdir`
  emit the same program for the examples, the control files of the tests,
  and the Amber suite
  ([docs/driver.md](docs/driver.md#211-the-equivalence-check-of-the-builder);
  #568).
- **The release path is checked before release day**:
  `scripts/release/check-path.sh` runs the edits of a release on a copy of
  the tree and checks the changelog's move, the notes with absolute links,
  and versions pinned in tests or code, and the notes of a GitHub Release
  now link into the repository at the tag (#609).
- **A generated reference of the dialects**: `docs/dialects/` holds the
  ops, types, and attributes of `md`, `dyn`, `md_exec`, `md_dist`, and
  `mdrt`, generated from TableGen and kept in sync by a test, and
  `docs/ops.md` links it (#610).
- **`scripts/ci/build-image.sh`** builds and pushes the image of CI on a
  machine with Docker, under the tag that the workflows compute, with a
  `--dry-run` that prints the tag (#611).
- **CI on GitHub for the CPU tier**: fast checks on every pull request,
  and on a ready pull request and on main a build with Python and the CPU
  tests in an image with LLVM/MLIR 23.1.2 and HDF5 (#611).
- **`MDIR_BUILD_TESTS`** (CMake, on by default) builds the tests and
  `mdir-opt`; the wheel turns it off, so the wheel builds from its sdist,
  which leaves `test/` out ([docs/building.md](docs/building.md), #624).
- **A release builds, checks, and installs its Python wheels**:
  `scripts/release/publish.sh` builds the wheels with the tarball, checks
  them, installs each by pip into a fresh environment of its CPython and
  runs it on the CPU, and one on the GPU
  (`scripts/release/check-wheel-install.sh`), and attaches them to the
  release (#641).

### Changed

- **Tuple coordinate attributes** are accessed through their declared op
  interface during differentiation; emitted programs and results are
  unchanged (D343, #612).
- **Each decision is a file of its own**, `docs/decisions/Dnnn.md`, under a
  short table in [docs/decisions.md](docs/decisions.md), and each PR's
  changelog entry is a fragment in `changelog.d/` that a release moves into
  CHANGELOG.md (#602).
- **The neighbor-group buffer slots are declared once** for the compiler
  and the runtime, and an invalid slot is diagnosed before runtime storage
  is accessed (D350, #626).

### Fixed

- **A pull request with a pending decision passes the checks**: the table of
  `docs/decisions.md` lists numbered decisions only, so a PR that adds
  `docs/decisions/<label>.md` leaves it as it is, and the release path
  (`check-path.sh`) and `decision-number.sh check --pending` pass on it;
  the check of the release path no longer fails now and then while its
  temporary repository is removed (#602, #609).
- On a GPU, a run with more than 4096 device blocks alive at once no longer
  leaves memory allocated after it ends. The memory that the runtime keeps
  for reuse is now bounded in bytes (`MDIR_DEVICE_POOL_BYTES`,
  `MDIR_FFT_PLAN_BYTES`). It is given back to the driver when an
  allocation would otherwise run out of memory, or by
  `mdir.trim_device_memory()` (D348, #622).
- A seed of $2^{53}$ or more is now fingerprinted exactly, so `--continue`
  and a Python continuation refuse a changed large seed. Integers below
  $2^{53}$ keep their text, and their checkpoints continue
  (D347, #620).
- **Invalid input storage-set boundary markers are diagnosed** before lowering; valid inputs run as before (D345, #625).
- Reading `sim.tunables` from another thread during an update now gives the
  values, version, and history of before, whole, and
  `sim.tunables.snapshot()` returns all three at once. A buffer exported to
  a consumer stream keeps the simulation from running even if its view is
  released during the hand-off (D346, #621).
- **The shards of a suite cover it**: a shard needs `--order=lexical`, which
  `lit.cfg.py` now requires, since the default order followed the times
  recorded in each build tree and two trees took overlapping shards; the
  check of the expected unsupported tests compares a shard's own tests
  (D344, #608).
- **Invalid batched stage row declarations are diagnosed** before lowering; valid staged models run as before (D351, #642).
- **The checked storage form uses one declared module marker contract** across assignment, loop verification, and lowering, with unchanged simulation behavior (D349, #636).

### Known limitations

- **One device per run.** Distributed execution (`md_dist`) runs only in
  one process, on the reference executor, and no run of `mdir run` or of
  Python takes it yet (#31, #38).
- **Learned potentials run on one GPU.** A model term of `backend =
  "METATOMIC"` refuses the CPU target, the deterministic mode,
  `[free_energy]`, `observe`, tunables, the frame evaluator, more than one
  model, constraints, and virtual sites, among others
  ([docs/metatomic-ml1.md](docs/metatomic-ml1.md), Section 9.1).
- **No polarizable potentials.** There is no Drude or AMOEBA model, and
  CHARMM's lone pairs are not supported.
- **Triplet terms run on the CPU only.**
- **The deterministic mode does not take neighbor groups.** It keeps the
  neighbor matrix, and `AUTO` takes the matrix there.
- **Free energy is limited to decoupling.** Relative (A to B)
  transformations, annihilation, and replica exchange are not supported,
  and a decoupled selection takes a plain Lennard-Jones cutoff or
  `POTENTIAL_SHIFT`, not a switch.
- **The cell does not shear at constant pressure.** The barostat scales
  each axis (isotropic, semi-isotropic, or anisotropic), without coupling
  the angles of the cell.
- **`[pme] influence = "P3M"` takes a rectangular cell only** (#475).
- **`mdir run` does not cache its host objects** in the compile cache yet;
  its GPU modules are cached (#99).
- **Binaries** are built for manylinux_2_28 (glibc 2.28 or newer: RHEL,
  Rocky, and Alma 8+, Ubuntu 20.04+, Debian 11+) and x86-64, with an NVIDIA
  driver supporting CUDA 13.0 or newer for the GPU (D177); the wheels are
  for CPython 3.10 to 3.13 (D228).

## [0.2.0] - 2026-10-26
### Added

- **Learned potentials through metatomic (ML1).** An optional plugin,
  built with `-DMDIR_ENABLE_METATOMIC=ON`, adds a TorchScript model of
  metatomic to the potential on a GPU through `[[energy.model]]` (`backend
  = "METATOMIC"`) or `mdir.System.models`, with
  `scripts/mlip/nequip-to-metatomic.py` for NequIP and Allegro models
  ([docs/metatomic-ml1.md](docs/metatomic-ml1.md); D270, #137).
- **A model term takes minimization and stochastic cell rescaling**: its
  energy, forces, and virial enter `[minimize]` and every coupling of the
  barostat (D272, #367).
- **Neural models in stages (ML4).** `backend = "STAGES"` runs a model as
  an MLIR module of neural stages that MDIR compiles and differentiates on
  one GPU; pretrained PaiNN in stages ran 196,608 atoms, 32 times the
  opaque call's largest system on 24 GB ([docs/ml4-ir.md](docs/ml4-ir.md),
  Section 8; D294, D293, #377, #447).
- **Pretrained PaiNN and MACE baselines**:
  `scripts/mlip/fetch-ml4-artifacts.py` fetches nablaDFT's PaiNN and
  MACE-OFF23 or MACE-MP-0, checked against pinned hashes, and converts
  them for `backend = "METATOMIC"`
  ([docs/ml4-baselines.md](docs/ml4-baselines.md); D286, #377, #436).
- **`[[energy.model]] stage_field_precision = "DOUBLE"`** keeps the
  feature fields of a model in stages, and their adjoints, in f64 in mixed
  precision; f32 stays the default (D329, #543).
- **An automatic, exact choice of the neighbor-list skins**:
  `pairlist_distance = "AUTO"` and `pruned_distance = "AUTO"` (and the
  keys not given) let a long GPU run with the groups measure and cache the
  fastest skins, with `mdir.Auto` in Python
  ([docs/neighbors.md](docs/neighbors.md#6-the-choice-of-the-skins); D281,
  #406).
- **Ewald summation with prolates**: `[pme] kernel = "PROLATE"` is a
  second PME kernel, on the CPU and a GPU, that needs 6 to 7 times fewer
  grid points at a force error of 10⁻⁵; the Gaussian stays the default
  ([docs/pme.md](docs/pme.md#7-the-prolate-kernel-d266); D266, #334).
- **`[pme] influence = "P3M"`**, the influence function optimal for the
  forces, in a rectangular cell, and `mdir.PMEInfluence.P3M` in Python
  ([docs/pme.md](docs/pme.md#11-the-influence-function); D304, #468,
  #475).
- **`MDIR_SITE_SPREAD=derived`** spreads the forces on virtual sites by
  the derived transpose of their placement; the written-out form stays the
  default (D298, D295, #453, #465).
- **`MDIR_DEVICE_MEMORY=2`** reports at the end of a GPU run what the CUDA
  context, the modules, cuFFT, and the tuple store took, and the pool's
  blocks at its peaks (D311, #471).
- **Lossless frames in H5MD**: `[output] trajectory = "run.h5md"` (or
  `trajectory_format = "H5MD"`) writes positions in f64 or f32 with the
  cell of every frame, and Python has `mdir.H5MDReporter` and
  `mdir.read_h5md` ([docs/python-h5md.md](docs/python-h5md.md); D239,
  #251).
- **The frame evaluator**: `mdir.FrameEvaluator(program)` evaluates
  energies, virials, observables, and their derivative in the tunables at
  stored frames, and `mdir.torch.evaluate` makes it an operation of
  PyTorch for fits by reweighting
  ([docs/python-frames.md](docs/python-frames.md); D240, D238, D239, D210,
  #249).
- **Frame gradients in the positions and the strain**: `evaluate(...,
  gradients=("positions", "strain"))` gives $\partial U/\partial\mathbf x$
  and $\partial U/\partial\boldsymbol\varepsilon$ per frame (D252, #313).
- **`FrameEvaluator(program, terms="dependent")`** evaluates only the
  terms that a tunable enters, 1.5 to 2.5 times faster a frame (D243,
  #264).
- **Tunable parameters in Python**: `System.tunables` declares charges,
  Lennard-Jones σ and ε, pair-term constants, and tuple-term parameters
  that `Simulation.tunables` updates without compiling, and `run(0,
  energy=True)` evaluates a state without a step (D213, D221, #130, #184).
- **Tunable Lennard-Jones by pairs of types**: `mdir.Tunable(name,
  "sigma_pair")` and `"epsilon_pair"` tune the pair table, NBFIX pairs
  included (D226, #160, #191).
- **The derivative of the energy in the tunables**: with
  `System.tunable_gradient`, `Simulation.tunables.gradient()` returns it
  without a step ([docs/python-gradient.md](docs/python-gradient.md);
  D230, D210, #203), the charges under PME included (D231, #203).
- **`observe` in the control file**: `observe = [...]` in a term given by
  an expression writes its energy and its derivatives in the listed
  constants to `[output] observables` at every energy, for example the
  force on a wall (D189, #16).
- **`observe` in Python**: `PairTerm`, `TupleTerm`, and `ExternalTerm`
  take `observe`, `ObservablesReporter` writes the file of `mdir run`, and
  `State.observables` gives the values
  ([docs/python-observe.md](docs/python-observe.md); D232, D189, #188).
- **Terms of the absolute positions in Python**: `mdir.ExternalTerm` and
  `System.external_terms`, the control file's `[[energy.external]]` (D233,
  #188).
- **The control file's keys in the Python model**: the thermostats and
  Brownian dynamics (D249, #278), the groups of 16 and the dual list
  (D245, #270), `System.pme_influence`, `System.analytic_bonds`, and PME
  orders 6 and 8 (D246, #279), the barostat's coupling and work (D247,
  #275), `Execution.neighbor_capacity` (D227, #162),
  `System.coulomb_modifier` (D205, #127), and a pair term's `dispersion`
  (D222, #161); each runs to the bit of `mdir run` in the deterministic
  mode.
- **A Python simulation at constant pressure in a triclinic cell**, bit
  for bit as `mdir run`, its checkpoints continued by either front end
  (D244, D127, #254).
- **Persistent Python simulations**: `mdir.Simulation` keeps a simulation
  across `run(n)` calls, with the coupling at the steps of an
  uninterrupted run and errors that leave Python alive (D196, #85), on the
  foundations of D191 (#74) and D192 (#76).
- **Reporters of a Python simulation**: `EnergyReporter`,
  `TrajectoryReporter` (DCD, XTC), and `CallbackReporter` write the files
  of `mdir run` (D207, #109).
- **Checkpoints of Python simulations**: `Simulation.save_checkpoint`,
  `CheckpointReporter`, `Simulation(program, checkpoint=path)`, and
  `mdir.read_checkpoint` read and write the checkpoint of `mdir run`, and
  either front end continues the other's (D223, #132).
- **Minimization from Python**: `Simulation.minimize(steps=None)` runs the
  minimizer of `mdir run` in parts that `request_stop()` and Ctrl-C end
  (D202, #91).
- **A force tolerance for minimization**: `[minimize] force_tolerance`
  (kcal/mol/Å) ends a minimization once the largest force is below it, and
  `Simulation.minimize(tolerance=...)` takes it in kJ/mol/nm (D219, #106).
- **Velocities and restraints in Python**: `InitialState.draw_velocities`
  draws the velocities of `mdir run` bit for bit, and `System.restraints`
  maps onto `[[restraints]]` (D198, #92); `InitialState.from_state(state,
  velocities=True)` starts a new stage from a reached state (#169).
- **OpenMM quantities in Python**: setters with a unit take an
  `openmm.unit.Quantity`, converted to MDIR's units; OpenMM stays optional
  (D200, #95).
- **DLPack views and borrows**: `Simulation.view()` shares the positions,
  velocities, and forces where the program keeps them, read-only (D220,
  #131), and `Simulation.borrow()` makes them writable with a checked
  `commit()` (D229, #136), triclinic tilts included (D238, #206)
  ([docs/python-dlpack.md](docs/python-dlpack.md)).
- **Read-only topology views**: `LoadedData.topology`, `System.topology`,
  and `Program.topology` return an `mdir.Topology` with `select(mask)` and
  `to_openmm()` (D221, #120).
- **The Python interface as a pip wheel**: `pyproject.toml` builds the
  package `mdir` for CPython 3.10 to 3.13 on manylinux_2_28, and
  `mdir[cuda]` adds NVIDIA's cuFFT and ptxas wheels (D228, #133). No wheel
  is published for 0.2.0 (see Known limitations).
- **A compile cache for Python simulations**: `MDIR_COMPILE_CACHE_DIR`
  keeps the host object of each program on disk (D212, #142, #99), the GPU
  modules become cubins, cached in `<dir>/gpu/` for `mdir run` too, with
  `MDIR_GPU_BINARY` and `MDIR_GPU_ARCH` (D214, #148, #164), and
  `mdir.clear_compile_cache()` and `cache=False` control it from Python
  (D217, #163) ([docs/compile-cache.md](docs/compile-cache.md)).
- **`MDIR_COMPILE_THREADS`** bounds the threads with which a process
  lowers its programs (D211).
- **Grids of tabulated functions from files**: `values_file` with an
  explicit `shape` reads a whitespace-separated grid in one, two, or three
  dimensions (D178).
- **Checks of differentiation**: `--md-analyze-activity` (D182),
  `--md-check-derivative-coverage` (D183), `--md-check-derivatives`
  against f64 central differences (D184), explicit conventions at singular
  points (D185), and diagnostics that name the control file's terms (D186)
  (#15).
- **`examples/ala3/run.py`** runs the four stages of the ala3 example from
  Python (#82).
- **Documentation**: an overview of MDIR,
  [docs/overview.md](docs/overview.md) (#587); the primitives of MDIR and
  the map of its parts, [docs/primitives-map.md](docs/primitives-map.md)
  (D292, #450); the design of the couplings as ops of `dyn` (D326, #541);
  a device portability inventory (D340, #558); the verified-execution
  contract (D338, #574); an L2 access policy window, measured and not
  adopted (#434); and the control-file reference's advice of
  `coulomb_modifier = "POTENTIAL_SHIFT"` at a loose PME tolerance at
  constant energy (#346).

### Changed

- **The default neighbor structure is `neighbor_structure = "AUTO"`**: on
  a GPU outside the deterministic mode, a system of at least 23,000
  particles runs on the groups of 16 with a dual list, about 1.6 to 2
  times as fast; set `neighbor_structure = "MATRIX"` to keep the old
  default ([docs/neighbor-groups.md](docs/neighbor-groups.md); D261,
  #350).
- **The Python model takes the control file's defaults** (plain cutoff,
  pairlist distance from the cutoff, center-of-mass removal with the
  thermostat, drawn velocities); a script that relied on an old default
  sets it explicitly, as the table in
  [docs/python-model.md](docs/python-model.md) shows (D248, #281).
- **`Ensemble.coupling_period` reads `None` until it is set**, where it
  read 10; set `ensemble.coupling_period = 10` to state the old value
  (D249, #278).
- **`FrameEvaluator.evaluate` takes `gradients=` in place of
  `gradient=`**, with no alias: replace `gradient=False` by `gradients=()`
  and drop `gradient=True` (D252, #313, #314).
- **A second derivative through `mdir.torch.evaluate` raises
  `mdir.UnsupportedError`** in place of PyTorch's `RuntimeError` (D252).
- **The frame evaluator keeps the particles in the order of the input**
  and evaluates the frames of a call in one activation, so a frame's
  energy has the same bits in any call and frames cost what their kernels
  cost (D251, #291).
- **Checkpoints are written on a thread beside the steps**; the next
  checkpoint, a stop, and the end of the run wait for the write, and a
  failed write stops the run there with status 1 (D336, D335, #548).
- **A constraint failure in dynamics stops the run** with the step and the
  constraint set, where the run went on with distorted groups (D278, #349,
  #381).
- **Steepest descent ends once its step shrinks below a floor**, with the
  checkpoint and exit status 0, instead of evaluating the same positions
  until `steps` ends (D274, #369).
- **The step of steepest descent grows to at most 0.1 Å**, not 1 Å, so
  minimizations give other rows (D260, D259, #348, #351).
- **With the groups, `pairlist_distance` may be up to 0.999 of the least
  edge of the cell**, in an orthorhombic and a triclinic cell, and a run
  whose barostat takes the cell below that stops (D242, #263).
- **The pair list of a model term is MDIR's**, built on the device at the
  model's cutoff; a plugin takes it through version 2 of the C boundary,
  and a plugin of version 1 is refused (D277, #378).
- **Leapfrog with constraints**: every step leaves the state of the plain
  step, and under Langevin dynamics that step is the documented one of
  velocity Verlet, so trajectories change; control files and checkpoints
  do not (D135, #285).
- **With constraints, the log's temperature is the estimate of Jung,
  Kobayashi, and Sugita (2019)** and its pressure takes the half-step
  kinetic energy, so rows read 1 to 2 K higher at 2 fs; with rigid waters
  `mdir run` ends with the temperatures of solute and solvent (D203,
  #112).
- **Under a plain cutoff, the free-energy and observables files take the
  shifted potential** that the forces sample, which removes a bias in
  reweighting gradients, and `dispersion_correction = "ENERGY_PRESSURE"`
  is allowed with `POTENTIAL_SHIFT` (D210, #140, #144).
- **In the deterministic mode a step that writes energies moves the
  particles as one that does not**, with constraints and PME too: device
  kernels form fused multiply-adds only as the formula states them (D201,
  #97), and in mixed precision the forces of the steps are stored in f64,
  so deterministic trajectories change (D204, #102, #105).
- **Particle mesh Ewald on the CPU runs in parallel**, and in mixed
  precision it computes in f32 as a GPU does, so CPU results in that mode
  change at the rounding of f32 (D271, D267, #359).
- **Python dispersion correction**: setting `System.dispersion` makes the
  correction explicit, as the control file's key does, and with a switch
  the default is off with the warning `dispersion_switched` (D222, #161).
- **A typed Python restraint keeps its force constant in kJ/mol/nm²
  exactly** (D216, #100).
- **`mdir.compile` no longer lowers the program**: `Program.lowered_ir`
  lowers on its first read, and an error that only the MLIR pipeline finds
  is raised by `Simulation(program)` (D224, #151).
- **A Python simulation runs every part in one activation**, so a run in
  parts equals one run to the bit in the deterministic mode, and a live
  simulation holds its memory between runs (D215, #135).
- **The Python extension is `mdir._core`** of the package `mdir`, and
  `MDIR_PYTHON_INSTALL_DIR` names the package's directory; a cache that
  set it appends `/mdir` (D228).
- **Python inputs and outputs are NumPy arrays**: shaped, read-only
  copies, and strict buffer or CPU DLPack inputs; Python needs NumPy 1.23
  or later (D193, #78).
- **`[free_energy]` with one state writes `dHdl.<name>` alone**;
  `--continue` onto an older file stops at its header (D190, #71).
- **`observe` is a reserved key** of every term given by an expression
  (D189), and custom term parameters may not reuse names that the term,
  particle parameters, functions, or λ components supply (D188, #67).
- **The neighbor-capacity estimate** is rounded up to four significant
  bits instead of a multiple of 8 (D227).
- **`md-exec-choose-scatter` replaces `md-exec-choose-spreading`** and the
  options `tuple-block-sums` and `tuple-block-least`; a script that names
  them in `MDIR_PRINT_AFTER` or a pipeline names the new pass and its
  options `tile` and `hash` (D285, #416, #417).
- **The comparison with pmemd.cuda on the Amber suite** runs both programs
  under one protocol with `bench.py run --suite-protocol`; the white
  paper's Section 10 is measured again with it (D327, #534).
- **`scripts/release/publish.sh` can release on a machine whose GPUs
  belong to a batch scheduler and that has no Docker**:
  `MDIR_RELEASE_SUITE` names the machine's command of the suite, and
  `--no-binary` makes a release without the manylinux tree
  ([docs/workflow.md](docs/workflow.md#releases); #593).
- **Scripts take what a machine provides from the environment**
  (`MDIR_ENV_FILE` and the variables in
  [docs/environment.md](docs/environment.md#scripts)) and stop naming what
  is missing (#289); a release build includes the Python interface and
  runs its tests (#297).
- **`scripts/mlip/water-box.py`** wraps the `.gro` residue number after
  99,999, as it does the atom number (D339, #550).
- **The roadmap**: M2 is split into M2a, the Python API, and M2b,
  differentiable simulation (D195); the milestones of learned potentials
  are M3 and M4 (D287, amending D167 and D169); the Python API's
  preparation is in [docs/python.md](docs/python.md) (D187).
- **Design documents are named by topic** (`docs/ops.md`, `runtime.md`,
  `pme.md`, `tuples.md`, `dynamics.md`, and others), and a CPU test checks
  every relative link (#390; the questions of M1 were answered by D64 and
  D70); `README.md` is a short entry point, with building from source in
  [docs/building.md](docs/building.md) (#589).
- **Compiler internals, with no change of results, keys, or files**: loop
  effects (D342, #566), set-bearing storage buffers (D337, #565), semantic
  FFT values and their storage (D319, D341, D306, #452, #480, #502, #526),
  declared transposes (D295, D297, #451, #459, #461), shared device
  emitters for compaction, scans, reductions, grid gathers, and
  scatter-accumulates (D296, D301, D325, D291, D285, #409, #416, #422,
  #432, #445, #460, #470, #539), the IR of neural stages (D288, D289,
  #377, #441), and, in the frame evaluator's program, the forces and
  virial handed to the host only where asked and the rows of a frame
  allocated in the loop over frames (D252, D251, D117, #303, #305, #309).
- **Distributed execution (ML2, `md_dist`)** is built in the compiler and
  runs on the in-process reference executor, which no run takes yet: the
  dialect and its verifier (D275), distribution and its derivative (D276,
  D279), the path through `md_exec` (D290), transposes of its loops
  (D297), factor-grid selection (D307), schedule proof (D308), halo events
  (D310), migration (D313), center partitions and filters (D317, D328),
  and a collective skin predicate (D332)
  ([docs/distributed-ml2.md](docs/distributed-ml2.md); #31, #38, #442,
  #476, #483, #490, #494, #496, #535, #536, #542).
- **Test infrastructure**: at most sixteen GPU tests at once, with the
  slowest files split by scenario (D208, #141); a shorter default Python
  `observe` matrix, the full one in release tests (D324, #530).

### Fixed

- **`.gro` velocities in fields of another width than the coordinates**
  are refused with the line, where they were read as 0 (#365).
- **A run whose positions or cell blew up** stops with a message, and a
  Python simulation raises `SimulationError` and keeps the state from
  before the part, where the process crashed (D225, #168).
- **A barostat that shrank a triclinic cell** below twice the pairlist
  distance lost pairs of the neighbor matrix (D241, #258), and **the
  groups lost pairs** at a pairlist distance close to the edge of an
  orthorhombic cell (D242, #263); both are right at every reach now.
- **Steepest descent on rigid water** stalled or tore waters; a trial that
  the constraints cannot take back is no longer taken (D259, #343).
- **The correction for the dispersion** includes the tail of every
  `[[energy.pair]]` term, so NBFIX corrections have their long-range part
  (D209, #145), a term left out no longer aborts the build of a later one
  (D209, #238), a correction at the force field's own parameters is no
  longer refused (#155), and the estimate of the shift follows the volume
  under a barostat (D210, #224).
- **Outputs and the barostat**: an output no longer counts as a refresh
  for `rebuild_interval` (D237, #233), and the rows of observables,
  free-energy, and pull files at a step where the barostat couples take
  the state of that step (#218).
- **The deterministic mode no longer depends on `energy_interval`** for
  torsions on the CPU (D204, #243) and for terms over the centers of
  groups in mixed precision (D204, #240); a Python simulation and `mdir
  run` give the same state to the bit (D218, #121).
- **A Nose-Hoover chain driven far from equilibrium** stops the run with
  an error instead of turning it to NaN (D206, #117).
- **Memory of long runs**: a run keeps nothing for a row of energies
  (D203, #300), `mdir run` without a log file keeps no text of its log
  (#304), a CPU run takes the stack of its rows once (D117, D215, #305),
  the device pool no longer grows with the number of builds (D312, #471),
  and a persistent Python simulation frees each part (#110).
- **Python simulations**: a continuation from a triclinic checkpoint takes
  its tilts (D238, #257); an open `TrajectoryReporter` takes a committed
  cell (D238); `save_checkpoint` is refused while a view is alive (D220,
  #207); `view()` and `borrow()` say why no activation exists (#220); a
  borrow with the groups checks an orthorhombic cell (D242, D245);
  `draw_velocities` takes a system with a model term (#388); tunable
  charges and pair constants together give both derivatives (D230, #256);
  and `mdir.compile` refuses a derivative without a rule (D230, D224).
- **Minimizations with no steps** are refused by `mdir.compile` (#178) and
  by `[minimize] steps = 0` (#180), where Python or `mdir run` died with a
  floating-point exception.
- **More than 8 cuFFT plans in a process** no longer abort it; a failed
  plan or transform stops the run with an error (#323).
- **HDF5 in threads**: every use of the library holds one mutex of the
  process, so checkpoints and H5MD frames of several simulations may be
  written from threads
  ([docs/python-checkpoints.md](docs/python-checkpoints.md); D239, #262).
- **The compile cache** no longer lists its directory after every store,
  which stalled on a network file system (D234, #196), and a program's
  text no longer depends on what the process built before (D212, D214,
  #152).
- **A program with a dual list that also keeps a neighbor matrix** lowers
  (D245), and **a GPU free-energy run with spatial ordering** uses the
  mapped charges directly (D179, #26).
- **The warnings** `short_thermostat_period` is reported once (#124), and
  a run from a PDB file reports its control file's warnings (D158, #129).
- **Free energy on a GPU in mixed precision with `POTENTIAL_SHIFT`**
  agrees with OpenMM within 1e-5 kcal/mol (D180, #48); compound terms
  accept declared `lambda_<name>` components (D188, #67).
- **SETTLE and SHAKE** read the step's velocity changes for the virial
  (D201, #97); a particle of mass 0 gets velocity 0 in a Brownian step
  (#19).
- **Notices of third parties**: the release tarball (#194), the Python
  wheels, and the container image (#221) carry the licenses of the code
  they distribute, in `share/mdir/licenses`.
- **The documents** no longer say that frames rounded to XTC cannot be
  used for reweighting (D239, D240, #264).
- **Compiler and host code**: passes no longer assert on unregistered ops
  (#383, #395) or on unused loop fields (#19), a refused loop no longer
  aborts the process (#208), the CPU lowering checks its stack allocations
  (D117, #312), the loop over frames is checked when it is built (D251,
  #302), uninitialized device reads reported by compute-sanitizer are gone
  (D110, #347, #413), persistent simulations register their exception
  frames safely (D199, #89), and the host code is compiled at LLVM's
  default level (D197, #90).
- **`scripts/release/changelog.py release`** keeps every entry of a
  subsection that [Unreleased] repeats, and stops before it writes if an
  entry would be lost (#596).
- **Tests and scripts**: intermittent failures of `not-numbers-gpu.test`
  (#425) and of `pme-tiles-gpu.test` (D79, #427, #428, #429), the
  device-memory report test (D311, #559), the JAX reference of
  `model-stages-programs-gpu.test` (D305, #486), the host-memory test of
  the frame evaluator (#306), outputs under `%t` (#423), `CUDA_CACHE_PATH`
  in tests (#223), the template generators (#168, #211), and
  `check-appendix.sh` without `mdir` (#209).

### Validated

- **The white paper's comparison with pmemd.cuda** is measured again on
  the Amber suite with the skins that MDIR chooses (D285, D281, D280,
  #437), and again under one protocol for both programs (D327, #534); the
  ratios are in Section 10.
- **A reproducible CPU baseline** compares the CPU target in mixed and
  double precision on JAC, Cellulose, and TIP3P under matched conditions
  (D333, #552).
- **The prolate kernel against the Gaussian**: on the CPU target (D266,
  D271, #371) and on a GPU with tiled spreading (D266, D282), in
  [docs/pme.md](docs/pme.md#7-the-prolate-kernel-d266), Section 7.

### Performance

- **Device memory**: ragged incidence structures for the tuples (D314),
  one set of PME buffers for every reciprocal sum (D315, D87), shared
  neighbor storage for the start and the steps (D330, D218), group lists
  sized by their first build (D331), and no unused spreading bricks (D320,
  #517); STMV ×15 (16 million atoms) now runs on one 24 GB GPU (#471).
- **The setup counts neighbors by cells and threads**: `mdir run` reaches
  its first row on STMV in 18 s where it took 61 s (D316, #471).
- **PME on a GPU spreads through tiles of block memory** for large f32
  grids, 4% to 21% less time a step; `MDIR_PME_SPREAD` overrides the
  choice (D282, #336).
- **The bonded kernel sums a block's values in block memory first** on
  large systems, 9 to 10% faster on Cellulose (D283, #408).
- **Fused pruning and pair loop** for the dual list: 4 to 6% less time a step on an
  RTX 4090 from about 6,400 groups; `MDIR_FUSED_PRUNE` overrides it (D284,
  D281, #362).
- **The inner loop of a dual list takes only the blocks that hold
  entries**, 12 to 47 µs less a step on an RTX 4090 (D268, #357, #360).
- **A device build of the neighbor matrix sorts a cell by rank**, which
  made a small system with crowded cells 2.5 times as fast (D235, #231).
- **Neural stages on a GPU**: products through cuBLAS (D302), batched
  products and fixed-order row sums (D303, D301), more fusion (D299,
  D293), and per-row sums of a pair tile (D300) each shorten a PaiNN step,
  the pair tiles alone by 1.8 to 4 times; an opt-in register strategy for
  short sums follows (D318) (#462, #477).
- **Python simulations start and update faster**: tuple structures are
  kept across activations (D253, D215, #294, #324) and built on the device
  (D257, #295), an update writes the model in place (D254, #324, #326) and
  keeps the arrays no tunable enters (D256, #328), and a start hands the
  program's arrays without copies (D258, #326); `mdir run` builds its
  segments' tuples in the memory of the segment before (D255, #316).
- **Compiling**: a `Program` keeps the code of its simulations, so a
  second `Simulation` is made in under 0.2 s (D236, #236); one program per
  Python simulation with a shared thread pool (D211, #147); faster
  lowering of f32 pair tables (D94, D159, #236); and state-dependent
  values as entry arguments, so later stages hit the cache (D227, #162).
- **Bonded kernels on a GPU**: adjacent bonded loops share one kernel
  (D181, #20), and the terms over centers of groups run inside the bonded
  kernel (D194, #66).
- **Leapfrog under Langevin dynamics with constraints** is 1.2 to 2.4%
  faster on a GPU (D250, #287).
- **`.gro` coordinates and velocities are read faster**, with a direct
  binary64 conversion (D323, #525).
- **An opt-in single-pass scan for group builds**,
  `single-pass-group-scan=true` of `convert-md-exec-to-gpu`; the
  three-launch scan stays the default (D321, #420, #524).
- **`md_exec.scatter_add` and pair hashing**: a privatized table for
  two-ended pair loops behind `MDIR_PAIR_SUMS`, measured slower and off by
  default (D291, #416, #445).

### Known limitations

- **No binary tarball and no wheel.** This release is source only (the
  maintainer, 2026-10-25): build MDIR, and its Python interface, from
  source as [docs/building.md](docs/building.md) describes.
- **One device per run.** Distributed execution (`md_dist`) runs only in
  one process, on the reference executor, and no run of `mdir run` or of
  Python takes it yet (#31, #38).
- **Learned potentials run on one GPU.** A model term of `backend =
  "METATOMIC"` refuses the CPU target, the deterministic mode,
  `[free_energy]`, `observe`, tunables, the frame evaluator, more than one
  model, constraints, and virtual sites, among others
  ([docs/metatomic-ml1.md](docs/metatomic-ml1.md), Section 9.1).
- **No polarizable potentials.** There is no Drude or AMOEBA model, and
  CHARMM's lone pairs are not supported.
- **Triplet terms run on the CPU only.**
- **The deterministic mode does not take neighbor groups.** It keeps the
  neighbor matrix, and `AUTO` takes the matrix there.
- **Free energy is limited to decoupling.** Relative (A to B)
  transformations, annihilation, and replica exchange are not supported,
  and a decoupled selection takes a plain Lennard-Jones cutoff or
  `POTENTIAL_SHIFT`, not a switch.
- **The cell does not shear at constant pressure.** The barostat scales
  each axis (isotropic, semi-isotropic, or anisotropic), without coupling
  the angles of the cell.
- **`[pme] influence = "P3M"` takes a rectangular cell only** (#475).
- **`mdir run` does not cache its host objects** in the compile cache yet;
  its GPU modules are cached (#99).

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

[Unreleased]: https://github.com/vv137/mdir/compare/v0.2.1...HEAD
[0.1.0]: https://github.com/vv137/mdir/releases/tag/v0.1.0
[0.2.0]: https://github.com/vv137/mdir/releases/tag/v0.2.0
[0.2.1]: https://github.com/vv137/mdir/releases/tag/v0.2.1
