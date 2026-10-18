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

- **Baselines of ML4: a pretrained PaiNN and a pretrained MACE through the
  model term** (D286, #377, PR #436,
  [docs/ml4-baselines.md](docs/ml4-baselines.md)):
  `scripts/mlip/fetch-ml4-artifacts.py` fetches nablaDFT's PaiNN (MIT) and
  MACE-OFF23 medium (Academic Software License: academic, non-commercial
  use), or MACE-MP-0 medium (MIT), each checked against its pinned SHA-256;
  `painn-to-metatomic.py` and `mace-to-metatomic.py` make metatomic
  artifacts of them for `backend = "METATOMIC"`, without changing a weight;
  `model-sweep.py` measures peak device memory and time against size.
  Validated against the packages' calculators; 164 kB per pair for
  MACE-OFF23 (float64) and 61 kB for PaiNN, as the
  packages take. The tests (`model-term-ml4-gpu.test`) take the fetched
  files from `MDIR_ML4_ARTIFACTS` and an environment of
  `scripts/mlip/ml4-requirements.lock` from `MDIR_ML4_PYTHON`, and are
  unsupported without them. No key of the control file changes.
  `water-box.py` now rounds the cell of its reference to the $10^{-7}$ nm
  of the `.gro` file, as it rounded the positions.

- **An exact automatic choice of the skins of the neighbor lists**
  (D281, #406,
  [docs/neighbors-m0.md](docs/neighbors-m0.md#6-the-choice-of-the-skins)):
  `pairlist_distance = "AUTO"` and `pruned_distance = "AUTO"` of
  `[energy]`, and the keys not given, let the run choose the skins. The
  lists stay exact, so the choice changes the speed, not the pairs within
  the cutoff. A run of `mdir run` with the groups on a GPU, outside the
  deterministic mode, of at least $5 \times 10^{12}$ particles times steps,
  measures the defaults of D261, two candidates of a cost model fitted to
  the grid of #406, and a climb from the fastest, each compiled anew from
  the state the run begins with, and keeps the fastest unless it is within
  the noise of the defaults. The choice is cached under a key of the
  system, the device, the precision, the step, and the ensemble
  (`MDIR_SKIN_CACHE_DIR`, by default `~/.cache/mdir/skins`;
  `MDIR_SKIN_CACHE=off`), which a later run and a continuation take.
  `MDIR_SKIN_SEARCH=always|never` overrides the length. Every other run
  keeps the reaches of before. The log (`MDIR: skins:`), the manifest
  (`pairlist_distance_source`, `pruned_distance_source`, `skin_search`),
  `mdir check` (`skins:`), and `Program.plan` say where the reaches came
  from. In Python, `System.pairlist_distance` and `System.pruned_distance`
  take `mdir.Auto`, which takes the cache or the defaults and measures
  nothing.

- The pruning of the inner list of a dual list in the loop over pairs
  (D284, #362, [docs/groups-m1.md](docs/groups-m1.md), Section
  6.2): where a refresh prunes, the loop over pairs that follows it prunes
  and computes in one kernel, a warp for each group, from 4096 groups
  (20,480 on `sm_12x`).
  On an RTX 4090 a step takes 4 to 6% less from about 6,400 groups; on an
  RTX PRO 6000 Max-Q it gains on the largest systems only, by 1 to 2%.
  `MDIR_FUSED_PRUNE=off|always` sets it aside or takes it at any size.
  The log line of the prunings adds how many the loop did. No control-file
  key changes; the forces are those of before to the order of their sums.
  The cache key of the skins (D281) names the setting, so an entry
  cached before misses once.
- ML2, stage 1, part 3 (D279, #31,
  [docs/distributed-ml2.md](docs/distributed-ml2.md), Section 15):
  `md-differentiate` derives the forces and the virial of a potential on
  domains through the fields it computes from the positions, such as the
  density and the embedding of EAM: the adjoint of the density is copied
  to the ghosts between the two passes, and the forces of its gather
  return to the owners by $G^\mathsf T$, or are computed at the owners by
  `md-dist-owner-computes`. No control-file key or output changes.
- ML2, stage 1, part 2 (D276, #31,
  [docs/distributed-ml2.md](docs/distributed-ml2.md), Section 14):
  `md-distribute` rewrites a global potential to run on a grid of domains
  (`grid`, `traversal`), and `md-differentiate` derives its forces and
  virial through the declared transposes of the transfers
  (`AdjointOpInterface`: $G$ and $G^\mathsf T$, distribution and
  collection, all-reduce); `md-dist-owner-computes` turns the derivative's
  form on the directed traversal into the computation at the owners. The
  dialect gains `md_dist.map_particles` and a two-ended `gather_pairs` over
  a directed relation. The proof of exchange contracts moves to the library
  of `md` (`mdir/Dialect/MD/MDExchange.h`). No control-file key or output
  changes.
- ML2, stage 1 of distributed execution, the core (D275, #31,
  [docs/distributed-ml2.md](docs/distributed-ml2.md)): the dialect
  `md_dist`, with layouts (ownership), halos, the owner-to-ghost copy and
  the sum of the ghosts' contributions at the owner as adjoint ops, and a
  verifier that rejects a missing transfer, a stale halo, a halo whose
  reach is below the cutoff, and partial sums read or counted twice; the
  in-process reference executor of several domains,
  `--convert-md-dist-to-loops` with `runtime/mdrt_dist.c`, whose trace
  `MDRT_DIST_TRACE` prints the rows of each domain and of each transfer.
  `md-check-exchange` now also proves kernels through components and
  lanes of vectors (`vector.extract`, `vector.from_elements`,
  `vector.shuffle`), so more kernels pass it. No control-file key or
  output changes.
- The prolate kernel (D266) measured against the Gaussian on the CPU
  target after D271 (#371, [docs/pme-m1.md §7](docs/pme-m1.md#7-the-prolate-kernel-d266)):
  `scripts/benchmarks/prolate-ewald/cpu_cases.py`, `cpu_job.py`, and
  `cpu_table.py` match the kernels by the measured force error at a
  cutoff of 9 Å and time them at 16, 4, and 1 threads with the parts of
  the step; `cpu-breakdown/prof.py` labels the prolate kernel's loop over
  pairs. MDIR's behavior does not change.
- The pair list of a term of `[[energy.model]]` is MDIR's (D277,
  #378): `md.edges` of an `md.neighborhood` at the model's cutoff, every
  image of each pair within it from the rows of the neighbor matrix (cells
  narrower than the cutoff included), built on the device in slots fixed at
  each build of the structure, the pairs within the cutoff first. The plugin
  takes it through version 2 of its C boundary (a plugin of version 1 is
  refused), exactly or padded (`MDIR_MLIP_PAIRS`), and searches its own with
  `MDIR_MLIP_PAIRS=search`; `MDIR_MLIP_CHECK_PAIRS=1` compares the two.
- `[minimize]` and stochastic cell rescaling with a term of
  `[[energy.model]]` (D272, #367): the model's energy,
  forces, and virial enter the steepest descent and the pressure of every
  coupling and count of the work, and its pair list follows the cell of each
  evaluation. `scripts/mlip/virial-strain.py` checks the virial against the
  energy under strain.
- ML1, a learned potential through an optional metatomic plugin
  (D270, #137, [docs/metatomic-ml1.md](docs/metatomic-ml1.md)):
  `-DMDIR_ENABLE_METATOMIC=ON` builds `libmdir_metatomic`, a C++20 library
  linking libtorch and metatomic that MDIR loads through the C header
  `include/mdir/MLIP/Plugin.h` (or `MDIR_MLIP_PLUGIN`); the classical build
  is unchanged. `[[energy.model]]` (`name`, `backend = "METATOMIC"`, `path`,
  `centers_per_call`) and `mdir.System.models` (`mdir.ModelTerm`) add a
  TorchScript artifact of metatomic to the potential on a GPU: its own pair
  list with every image, its forces added into MDIR's buffer of forces by a
  host call that declares the write (`mdrt.host_call` takes fields in `outs`
  and yields them, and numbers of the host), its energy and virial in the
  potential, the log, and the pressure, and chunks of centers for a model that declares
  itself local. `scripts/mlip/nequip-to-metatomic.py` turns a model package of
  the `nequip` package (NequIP or Allegro) into such an artifact; `water-box.py` and
  `check-model-term.py` there make the systems and the checks of the
  validation. ML1 refuses the deterministic mode, the CPU, a barostat,
  `[minimize]`, `[free_energy]`, `observe`, tunables, constraints, and
  virtual sites with a model term.
- Ewald summation with prolates, a second kernel of particle mesh Ewald
  (D266, #334): `[pme] kernel = "PROLATE"` takes the prolate
  spheroidal wave function of order zero as the splitting and as the
  window of the spreading, on the CPU and a GPU, in orthorhombic and
  triclinic cells, in `mixed` and `double` and the deterministic mode. Its
  `tolerance` is $\psi_0^c(1)$, 10⁻⁵ by default, which gives a relative
  force error of 1.0 to 1.2 × 10⁻⁵ over the Coulomb forces where the
  Gaussian's 10⁻⁵ gives 1.4 × 10⁻³;
  `bandwidth` and `window_bandwidth` give $c$ and $c_1$, `order` the points
  of the window (5 to 8, 6 by default), and the grid follows from eq. 15 of
  Liang et al. (2026), refined until its aliasing estimate is at most half
  the tolerance, unless `grid` or `max_spacing` is given; the window is the
  paper's less its value at the edge of its support, so that the energy is
  continuous when a particle crosses a plane of the grid; `beta` and `influence` are refused with it, and
  `bandwidth` and `window_bandwidth` without it. The Python model has
  `System.pme_kernel` (`mdir.PMEKernel`), `pme_bandwidth`, and
  `pme_window_bandwidth`; the manifest of a run names the kernel. The
  Gaussian stays the default: on one GPU the prolate kernel's step takes
  1.2 to 1.4 times as long at MDIR's default accuracy and about as long at
  a force error of 10⁻⁵, where its grid has 6 to 7 times fewer points.
- The control-file reference recommends `coulomb_modifier =
  "POTENTIAL_SHIFT"` at a PME tolerance looser than the default for runs at
  constant energy (#346); `NONE` stays the default, as in Amber.

- The gradients of the energy of a frame in the positions and in the
  strain, from the frame evaluator (D252, #313):
  `evaluator.evaluate(frames, gradients=("tunables", "positions",
  "strain"))` gives `out.gradients["positions"]`, `(K, N, 3)` in
  kJ/mol/nm, $\partial U/\partial\mathbf x = -\mathbf F$ in the order of
  the input and in the type of the forces of the program (float32 in
  `mixed`, float64 in `double` and in `mixed` with the deterministic
  mode), and `out.gradients["strain"]`, `(K, 3, 3)` float64 in kJ/mol,
  $\partial U/\partial\boldsymbol\varepsilon$ under $\mathbf x \to
  (\mathsf 1 + \boldsymbol\varepsilon)\mathbf x$ of the positions and the
  cell vectors: the virial tensor with the opposite sign, nine components
  for orthorhombic and triclinic programs, not symmetrized. Without
  constraints `virial` is $-\mathrm{tr}$ of it; with them `virial`
  has the term of the constraint forces besides, and a pressure takes
  `virial`. For a program with restraints to reference positions
  `"strain"` is refused: their virial is taken about the references and
  is not that derivative. A frame that asks for neither costs what it cost; one that
  asks pays one more evaluation of its forces (a third of a frame on the
  CPU; on one RTX 4090 in mixed precision 13% to 29% of a frame of the
  dipeptide and 20% to 37% of one of JAC, by what is asked). Through
  `mdir.torch.evaluate`, positions `(K, N, 3)`, `cells`, and `tilts` that
  are tensors requiring a gradient are differentiated (the energy at
  fixed Cartesian positions), and `gradients=("positions", "strain")`
  returns the two as tensors. A loss on the forces or the stress can be
  evaluated but cannot yet train the tunables: its derivative in them
  (force matching) and every other second derivative raise
  `mdir.UnsupportedError`, never a zero. Checked against central
  differences of the evaluator's energy, torch-pme (the Coulomb part),
  and OpenMM (the whole potential), on the CPU and a GPU, in double and
  mixed precision. See
  [docs/python-frames.md](docs/python-frames.md#the-gradients-in-the-positions-and-the-strain-d252).
- The thermostats of the control file and Brownian dynamics in the Python
  model (D249, #278): `Ensemble.thermostat`
  (`mdir.ThermostatMethod.VRescale`, the default, `Langevin`,
  `NoseHoover`), `Ensemble.friction` (1/ps, for Langevin dynamics), and
  `Ensemble.chain_length` (3 by default, for a Nosé–Hoover chain), the
  `method`, `friction`, and `chain_length` of `[thermostat]`; and
  `mdir.IntegratorMethod.Brownian`, refused until now, with
  `Integrator.friction`, the `friction` of `[dynamics]`, under
  `EnsembleKind.NVT`. The refusals are those of the control file with its
  words. `Program.plan["thermostat"]` and `["integrator"]`;
  `Checkpoint.thermostat_state`. A checkpoint of a Python simulation holds
  the state of the chain and continues in either front end. Each runs to
  the bit of `mdir run` in the deterministic mode, on the CPU and a GPU,
  in double and mixed precision.
- The groups of 16 and the dual list in the Python model
  (D245, #270): `Execution.neighbor_structure`
  (`mdir.NeighborStructure.Matrix`, the default, or `Groups`) and
  `System.pruned_distance` (nm; 0 keeps one list), the
  `[execution] neighbor_structure` and `[energy] pruned_distance` of the
  control file, refused as they are with the same words;
  `Program.plan["neighbor_structure"]` and `["pruned_distance"]`. With one
  structure a Python simulation runs at the rate of `mdir run`: JAC at
  constant energy in mixed precision, 0.206 ms a step with the dual list
  against 0.259 with the matrix. The matrix stays the default.
- `System.pme_influence` (`mdir.PMEInfluence.SPME`, the default, or
  `Optimal`) and `System.analytic_bonds` in the Python model, the
  `[pme] influence` and `[constraints] analytic_bonds` of the control file,
  and `System.pme_order` of 6 and 8 (D246, #279): the
  settings of the Amber suite's script that a Python run could not state.
  Each runs to the bit of `mdir run` in the deterministic mode.
- The coupling and the work of the barostat in the Python model
  (D247, #275): `Ensemble.barostat_coupling`
  (`mdir.BarostatCoupling.Isotropic`, the default, `SemiIsotropic`,
  `Anisotropic`) and `Ensemble.barostat_work` (`mdir.BarostatWork.Trotter`, the
  default, `TrotterFirstOrder`, `Exact`, `FirstOrder`), the `coupling` and
  `work` of `[barostat]`, with its refusal of the first-order work under
  the semi-isotropic and the anisotropic coupling;
  `Program.plan["barostat"]`. With them, `Ensemble.compressibility` takes
  three numbers for the anisotropic coupling, and
  `Ensemble.compressibility_z`, `surface_tension` (bar nm), and `surfaces`
  are the keys of the semi-isotropic one. Each runs to the bit of `mdir run` in the
  deterministic mode, in an orthorhombic and in a triclinic cell, and a
  checkpoint continues in either front end. With the exact and the
  first-order work a Python simulation takes `coupling_period = 1`.
- The frame evaluator over the terms that depend on the tunables alone
  (D243, #264): `mdir.FrameEvaluator(program,
  terms="dependent")` compiles a second program with only the terms that a
  tunable enters (a tunable constant's pair term; a tunable parameter's
  tuple term; the Lennard-Jones for a tunable $\sigma$ or $\epsilon$; the
  electrostatics, whole, for tunable charges) and evaluates that. Its
  energy is `out.dependent_energy`, whose differences in the tunables and
  whose derivative are those of the potential energy; `out.energy` and
  `out.virial` raise `UnsupportedError` in that mode, the observed columns
  of the terms left out are absent, and `out.unavailable` lists what is not
  given. A frame costs 1.5 to 2.5 times less: on JAC in mixed precision
  2.97 ms against 7.44 ms for a constant of a pair term and 4.12 ms
  against 7.68 ms for the charges. The mode saves time; the accuracy of
  a difference in mixed precision is that of the whole potential. `terms="all"`,
  the default, is the evaluator of before, which now has
  `out.unavailable == ("dependent_energy",)`; `Program.plan["terms"]` is
  new. `docs/python-frames.md` also shows the reweighting of a potential
  that is linear in its tunables from the sums that `observe` records,
  without stored frames.

- The frame evaluator (D240, #249): `mdir.FrameEvaluator(program)`
  evaluates a program compiled with tunables and `System.tunable_gradient`
  at stored frames, in a simulation of its own.
  `evaluate(positions, cells=None)` takes `(K, N, 3)` positions in the
  order of the input (an array, a `numpy.memmap`, or an iterable of states
  or of pairs `(positions, cell)`, read in turn) and returns, per frame,
  the potential energy that the forces sample (D210), the virial, the
  volume, and the columns of `observe`, with `vjp(cotangent)`, the product
  with the derivative in the tunables (the Jacobian is kept up to
  `jacobian_bytes`, 256 MiB, and the frames are evaluated again above it),
  and `depends`, the tunables that each output may read. A frame takes one
  evaluation: 1.05 ms on the dipeptide in water and 6.85 ms on JAC on an
  RTX 3090 in mixed precision, against 2.55 ms and 16.8 ms through a
  borrow, a commit, and `gradient()`. `mdir.torch.evaluate(evaluator,
  theta, positions, cells)` is the same as an operation of PyTorch whose
  backward is that product, for fits by reweighting; an output whose
  derivative in the tunables is not implemented (the virial) raises in a
  backward unless it is detached. `mdir.KB` is Boltzmann's constant in
  kJ/(mol K). Each frame has its cell, the tilts of a triclinic cell
  included (D238), and `evaluator.evaluate(mdir.read_h5md(file))` evaluates
  the frames of an H5MD trajectory (D239) one at a time, to the bit of the
  states for a file in f64. See [docs/python-frames.md](docs/python-frames.md).

- A Python simulation runs at constant pressure in a triclinic cell
  (D244, #254, `docs/python-segments.md`);
  `mdir.Simulation(program)` raised `UnsupportedError` for a barostat in
  such a cell. The tilts of `State.cell`, of a checkpoint, of the frames
  of the reporters, and of `Borrow.tilt` follow the barostat, which scales
  them with the cell as in `mdir run` (D127). In the deterministic mode
  the run is that of `mdir run` with the same settings and checkpoints at
  the same steps, bit for bit, and a checkpoint of either front end is
  continued by the other. The coupling is the isotropic one, the only one
  of the Python model. No control-file key and no file format changes.

- The tilts of a triclinic cell in a writable borrow (D238,
  #206, `docs/python-dlpack.md`): for a simulation whose program was
  compiled from a triclinic cell, `Borrow.cell` is the diagonal
  $a_x, b_y, c_z$ (it raised `UnsupportedError`) and the new `Borrow.tilt`
  is the tilts $b_x, c_x, c_y$, both `(3,)` float64 buffers of the host,
  the arrays of `mdir.Cell`. A commit takes them with the positions and
  begins an activation with the committed cell; it refuses
  (`InputError`, nothing changed) tilts that are all zero, a cell that is
  not reduced, a diagonal below twice the cutoff, and, for a program with
  the groups, a pairlist distance above half of the least of the diagonal.
  The grid of PME and the
  neighbor capacity stay those of the program
  (`Program.plan`). `Borrow.tilt` of a program compiled for an
  orthorhombic cell is a read-only array of three zeros (the read-only
  flag of DLPack; in NumPy an array that cannot be written), and tilts
  written to it in spite of the flag are refused at the commit;
  `Borrow.cell` of such a program is unchanged. No control-file key and
  no file format changes.
- Frames without loss, in H5MD (D239, #251): a third
  trajectory format that holds the positions of the state as the run has
  them, in nm, in f64 (or f32), with the cell of every frame, triclinic
  included, the step, and the time, for the reweighting of stored frames.
  `[output] trajectory = "run.h5md"` selects it by the extension `.h5md`
  (`.h5` stays that of checkpoints) or `trajectory_format = "H5MD"`;
  `trajectory_precision = "DOUBLE"` (the default) or `"SINGLE"` and
  `trajectory_strings = "FIXED"` (the default) or `"VARIABLE"` are new
  keys. In Python, `mdir.H5MDReporter(file, period, positions="f64",
  velocities=False, forces=False, strings="fixed")` takes the place of a
  `TrajectoryReporter` and also writes the potential energy of each frame,
  the version of the tunables, and, if asked, the velocities and the
  forces; `mdir.read_h5md(file)` reads the frames one at a time, and a
  frame unpacks as the pair `(positions, cell)`. The file continues with
  the run, is backed up, and goes to parts as a DCD does, and a run that
  is killed leaves a file that opens with the frames written. The `unit`
  attributes are strings of fixed length, as the units module of H5MD
  gives them, or of variable length with `strings = "variable"`;
  MDAnalysis 2.10.0 with h5py 3.16.0 reads files written with
  `strings = "variable"`, and with `"fixed"` it stops with the message
  "time unit 'b'ps'' is not recognized by H5MDReader". On JAC
  (23,558 atoms) a frame takes 565,642 bytes in f64 and 282,946 in f32
  (DCD: 282,776) and 0.78 ms and 0.57 ms (DCD: 0.62 ms). The manifest's
  `trajectory_format` and the report of `mdir check` may now be `"H5MD"`.
  See [docs/python-h5md.md](docs/python-h5md.md).

- A `Program` keeps the code of its simulations (D236, #236):
  the second and later `mdir.Simulation(program)` of one `Program` in a
  process take the LLVM module and the host object that the first left
  with the `Program`, and neither run the MLIR pipeline nor generate host
  code, with or without `MDIR_COMPILE_CACHE_DIR`. On the dipeptide in
  water in mixed precision such a simulation is made in 0.07 s on the CPU
  and 0.13 s on a GPU, instead of 5.4 s and 9.4 s (3.1 s and 5.6 s on hits
  of the disk cache), and gives the same trajectory to the bit in the
  deterministic mode. `Simulation.compile_stats` has two new keys,
  `program_reused` and `reuse_saved_seconds`; with `program_reused`,
  `pipeline_seconds` and the counts of the caches are 0, which a script
  that reads them for every simulation now sees. `cache=False` compiles
  everything anew, as before. See
  [docs/compile-cache.md](docs/compile-cache.md#reuse-within-a-process).

- `observe` in the Python model (D232, #188): `PairTerm`,
  `TupleTerm`, and `ExternalTerm` take `observe`, `None` or a list of
  constants of the term, as the control file's key (D189);
  `ObservablesReporter(file, period)` writes the file of `[output]
  observables`, equal to that of `mdir run` for the same model, with a
  period of its own, appended across runs and continued from a checkpoint
  like the energy file; `State.observables` gives the values of a step of
  energy to a script, in kJ/mol, also in the state of a `CallbackReporter`;
  `Program.plan["observables"]` lists the columns. Tunables and `observe`
  now share a program, the file's last column is then `tunables_version`,
  and an observed constant may be a tunable with one entry, whose column
  equals `Simulation.tunables.gradient()` of it. A program that minimizes
  refuses `observe`. See [docs/python-observe.md](docs/python-observe.md).
- Terms of the absolute positions in the Python model (D233,
  #188): `mdir.ExternalTerm` and `System.external_terms`, the control
  file's `[[energy.external]]`, with `x`, `y`, `z` in nm and the energy in
  kJ/mol, over a mask or an array of particle indices, with constants, and
  `ExternalScaling` under a barostat. The fingerprint of such a model has
  the entry `[python] external_terms`.
- The derivative of the energy in the tunable parameters of a Python
  model (D230, #203): with `System.tunable_gradient = True`
  a compiled program can evaluate it, and
  `Simulation.tunables.gradient()` returns, without taking a step, a
  read-only mapping of the names of the tunables to arrays of the shape of
  their values, in kJ/mol per unit of each, with `.energy`, `.version`,
  `.step`, and `.zero`. It is the derivative of the potential that the
  forces sample, shifted at the cutoff where the run cuts it (D210). It
  covers constants of pair terms, parameters of tuple terms, per-type
  sigma and epsilon, the table by pairs of types, and the charges under a
  Coulomb cutoff; the charges with particle mesh Ewald are refused at
  compile until that part lands. `Program.plan["tunables"][k]["gradient"]`
  says `"rule"` or `"zero"` for each tunable. Programs compiled without
  the input are unchanged. In the IR, `derivative(n)` of `md.evaluate`
  takes an argument that is a field of the particles and gives a field.
  See [docs/python-gradient.md](docs/python-gradient.md).
- The derivative of the energy in the charges with particle mesh Ewald
  (D231, #203): a program compiled with
  `System.tunable_gradient` and tunable charges under PME is no longer
  refused. `md.reciprocal` takes an optional fourth result, the potential
  of the grid at the particles, and `md_exec.reciprocal` the flag
  `potential`; programs without the derivative are unchanged. The suite
  has the feature `torchpme` for its check against torch-pme.
- `Execution.neighbor_capacity` in Python, the `[execution]
  neighbor_capacity` of the control file (D227,
  #162): 0, the default, estimates it, and
  `Program.plan["neighbor_capacity"]` is the capacity that a compile took.
- The Python interface as a pip wheel (D228, #133): the
  package `mdir` for CPython 3.10–3.13 on Linux x86-64 with glibc 2.28 or
  later (manylinux_2_28), built by `pyproject.toml` (scikit-build-core) and
  in release form by the target `wheels` of `packaging/Dockerfile.manylinux`.
  It carries its runtime, libdevice, and HDF5; `pip install "mdir[cuda]"`
  adds NVIDIA's wheels of cuFFT and of ptxas (`nvidia-cufft`,
  `nvidia-cuda-nvcc`) for GPU runs, which then need only the NVIDIA
  driver. `import mdir` initializes no CUDA. The `mdir` command stays in the
  release tarball. The ala3 Python example declares `mdir[cuda]` of its
  version and runs with `uv run --find-links <wheels>` from a copy of the
  directory.

- Checkpoints of Python simulations (D223, #132):
  `Simulation.save_checkpoint(path)` and `CheckpointReporter(file,
  period)` write the H5MD checkpoint of `mdir run` (format 1, with `.prev`
  and durable replacement); `Simulation(program, checkpoint=path)`
  continues the same run as `mdir run --continue` does, and `stage=True`
  begins a new stage as `[input] checkpoint` does; `append=False` writes
  the reporters' files to a part of their own; `mdir.read_checkpoint(path)`
  reads one. Either front end continues the other's checkpoint: the
  fingerprint of a Python model has the entries that the control file of
  the same model writes. Format 1 is unchanged: the front end, model and
  plan hashes, and tunables are additional entries with a hash of their
  own, which earlier readers ignore; `mdir checkpoint` prints them.

- Tunable Lennard-Jones by pairs of types (D226, #160):
  `mdir.Tunable(name, "sigma_pair")` and `"epsilon_pair"` tune the table of
  the pairs of types, so that a pair the combining rule does not give (an
  NBFIX) or a force field without a rule can change. The sites are the
  unordered pairs $(a, b)$, $a \le b$, in the order of the flat upper
  triangle, which the read-only `Topology.type_pairs` lists and maps are
  built from. Pair and per-type tunables may be declared together: the pair
  tunables set the pairs that their maps take after the rule, and a
  diagonal pair does not redefine the per-type values; the 1-4 pairs keep
  their own parameters (maintainer's decisions on PR #191). An update
  equals a compile with the new values to the bit. No control-file key
  changes.
- Controls of the compile cache from Python (D217,
  #163): `mdir.clear_compile_cache(directory=None)` removes the entries of
  this format from the directory given or from `MDIR_COMPILE_CACHE_DIR`,
  and returns what it removed. `mdir.compile(..., cache=False)` and
  `Simulation(program, cache=False)` bypass the cache for one compile
  without touching the environment; a simulation takes its program's
  choice unless given one. `Simulation.compile_stats["cache_bypassed"]`
  reports it.
- A force tolerance for minimization (D219, #106):
  `[minimize] force_tolerance` (kcal/mol/Å) ends `mdir run`'s minimization
  at the first row of the energies whose largest force (the `MAX_FORCE`
  column, without the parts along the constraints) is below it, with the
  checkpoint at that step and a line in the log saying whether it
  converged. `Simulation.minimize(steps=None, tolerance=None)` takes it in
  kJ/mol/nm (or an OpenMM quantity) and checks every
  `Schedule.energy_period` steps; `state().minimization["converged"]`
  reports the result. Without a tolerance nothing changes.
- Tracked writable DLPack borrows of a Python simulation with a commit
  (D229, #136, `docs/python-dlpack.md`):
  `Simulation.borrow()` returns a `Borrow` whose `positions` and
  `velocities` are the program's buffers, writable through `__dlpack__`
  on the device or the host, in the order of the program (`ids` gives the
  input index of each row), with the edges of an orthorhombic cell
  (`cell`) and the values of the tunables (`tunables`) as writable buffers
  of the host. `Borrow.commit()` takes what was written: it checks
  everything first (finite values, a cell of at least twice the cutoff,
  the checks of an update of tunables with its refusal of structural
  changes; a refusal changes nothing), advances `Simulation.versions` of
  the changed fields, and begins an activation from the committed state,
  whose order, neighbor structures, and forces are built anew, as an
  update of tunables does. While a borrow or a tensor taken from it is
  alive, `run`, `minimize`, evaluations, updates of tunables, `view()`,
  `borrow()`, `state()`, and `save_checkpoint` raise `SimulationError`. A
  borrow that ends without a commit (`abandon()`, the end of a `with`
  block) is undone to the bit. Writes through a read-only `view()` stay
  undefined. A triclinic cell is not taken yet (#206; taken since
  D238, above). No control-file key and no file format changes.
- Read-only DLPack views of a Python simulation (D220, #131,
  `docs/python-dlpack.md`): `Simulation.view()` returns a `View` of the
  positions, velocities, and forces where the program keeps them, on the
  device or the host, in the order of the program and the types it stores
  (forces in f32 in mixed precision), with the input index of each row
  (`ids`) and the values of the tunables, each a `Buffer` with
  `__dlpack__` and `__dlpack_device__`, so that `torch.from_dlpack(view.positions)`
  shares the buffer. A view and every tensor taken from it hold a lease
  (`Simulation.leases`); while one is alive, `run`, `minimize`,
  evaluations, and updates of tunables raise `SimulationError`. On a device
  the consumer's stream waits for the simulation's work, and the next part
  waits for the consumer's. No control-file key changes.

- Read-only topology views of the Python model (D221, #120):
  `LoadedData.topology`, `System.topology`, and `Program.topology` return an
  `mdir.Topology`, a copy of the atoms (`atom_names`, `atomic_numbers`,
  `masses`, `charges`, `particle_types`, `type_names`, `type_pairs`,
  `residue_indices`), residues (`residue_names`, one per residue, and
  `residue_starts`), bonded
  tuples (`bonds`, `angles`, `dihedrals` with `improper_dihedrals`,
  `harmonic_impropers`), and `virtual_sites`, as read-only NumPy arrays and
  lists; a compiled program's view also gives the bonds that SHAKE holds
  (`constraints`) and the waters that SETTLE holds (`rigid_waters`).
  `Topology.select(mask)` returns the particles of a mask of Amber by the
  parser of the control file, an `InputError` with its diagnostic for a mask
  that does not parse. `Topology.to_openmm(cell=None)` returns an
  `openmm.app.Topology` when OpenMM is importable. No control-file key
  changes.
- `mdir.InitialState.from_state(state, velocities=True)` makes the initial
  state of a new stage from the `State` that a simulation reached: copies of
  its positions, cell, and velocities, or without the velocities when asked,
  or when the state has none. The velocities of a leapfrog state, half a
  step behind its positions, are refused. The ala3 example and
  `docs/python-minimize.md` use it in place of setting the fields one by
  one.

- Tunable parameters of a Python model (D213, #130):
  `System.tunables` declares charges per particle, σ and ε per
  Lennard-Jones type (with the combining rule), constants of pair terms, and
  parameters of tuple terms as named vectors with a map from their sites,
  and `Simulation.tunables` takes new values without compiling, as one
  atomic update that advances a value version (`tunables.version`,
  `tunables.history`, `State.tunables_version`, and a last column
  `tunables_version` in the energy file of a simulation with tunables). The
  quantities derived from them (the table of the types by the combining
  rule, whose NBFIX pairs keep their values with a warning; the 1-4 products, the constants of PME and
  of the reaction field, the correction for the dispersion, the tails of
  pair terms) are rebuilt by the code that compiles them, and an update
  equals a compile with the new values to the bit. `run(0, energy=True)`
  evaluates the forces and the energies of the state without a step. Maps
  are built from `System.topology` (D221), the one place of the topology
  data (#184). No control-file key changes.
- `MDIR_COMPILE_THREADS` bounds the threads with which a process lowers
  its programs (D211's process-wide pool otherwise takes every core). The
  test suite sets it to the cores over `gpu_workers`, or to
  `-Dcompile_threads=N`, so that sixteen tests compiling at once do not ask
  for sixteen times the cores.
- A compile cache of host objects for Python simulations
  (D212, #142): `MDIR_COMPILE_CACHE_DIR=<dir>` keeps the
  relocatable object of each compiled program on disk, keyed by the
  content of its LLVM module and the code generator, so that later
  processes and rebuilds of MDIR that generate the same module skip host
  code generation. `MDIR_COMPILE_CACHE=off` disables it and
  `MDIR_COMPILE_CACHE_MAX_MB` bounds it (2048 MiB, least recently used
  first). `Simulation.compile_stats` reports the compile times and the
  hits. The test suite shares a cache in its build tree
  (`-Dcompile_cache=off` turns it off). `mdir run` does not use it yet
  (#99).
- The GPU modules of a program are serialized in parallel, on the threads
  that `MDIR_COMPILE_THREADS` bounds, and become cubins for the device that
  runs them, which the driver loads without compiling
  (D214, #148). The toolkit's `ptxas` compiles them. The
  kernels stay PTX, as before, when there is no device to ask, when LLVM
  does not know the architecture, or when there is no `ptxas`.
- The compile cache keeps the PTX and the cubin of each GPU module in
  `<dir>/gpu/`, for `mdir run` as well as for Python simulations.
  `MDIR_COMPILE_CACHE_MAX_MB` bounds the host and GPU entries together.
- `Simulation.compile_stats` gains `gpu_*` keys.
- `MDIR_GPU_BINARY` (`auto`, `cubin`, or `ptx`) and `MDIR_GPU_ARCH`
  (`sm_XY`) choose the binaries of the kernels and their architecture.
  The device's architecture is asked of NVML, which creates no CUDA state,
  so a process may still fork after `mdir.compile`.
- The Python model's pair terms take `dispersion`, the control file's
  `dispersion_correction` of a pair term (D222, #161):
  `None` follows the system, `DispersionCorrection.None_` leaves the term
  out of the correction for the dispersion, `EnergyPressure` asks for its
  tail. `System.dispersion_given` says whether `System.dispersion` was set;
  a pair term whose tail diverges is then an `InputError`, and is left out
  with the `UserWarning` `pair_tail_left_out` otherwise.
  `Program.plan["dispersion"]` reports which tails are in the correction.

### Changed

- **Documented: an L2 access policy window, measured and not adopted**
  (#434, [docs/device-primitives.md §3](docs/device-primitives.md#3-the-l2-access-policy-window-measured-not-adopted)):
  a persisting window over the outer or inner list of the groups, the
  frames of the positions, or the buffer most launches take gave no gain
  on the Amber suite on an RTX 4090 or an RTX PRO 6000, and made Cellulose
  and STMV up to 14% slower on the RTX 4090. No code, key, or default
  changes.

- **PME on a device spreads through tiles of the memory of a block**
  (D282, #336,
  [docs/pme-m1.md §3.3](docs/pme-m1.md#33-spreading-through-tiles)): the
  particles are binned by their first grid point at each evaluation, a
  block adds the charges of a bin into a tile and the tile into the grid,
  as cuFINUFFT spreads. A new pass, `md-exec-choose-spreading`, sets
  `tiles(8)` on `md_exec.reciprocal` for every sum in f32 whose grid has
  at least 500,000 points; a smaller grid, double precision, and the
  deterministic mode spread as before.
  The prolate kernel, LJ-PME, and triclinic cells take the tiles too.
  `MDIR_PME_SPREAD` (`global`, `tiles`, `tiles4`, `tiles8`, `tiles16`,
  `auto`) overrides the choice. On an RTX 4090 in mixed precision the
  step at order 4 takes 0.94 to 0.96 times as long on Factor IX,
  Cellulose, and STMV, and 0.79 to 0.87 at orders 6 and 8 and with the
  prolate kernel. No control-file key or
  output changes; results agree with the previous spreading to the
  rounding of the atomics.

- **The tiles of PME and the tables of the tuple kernel are strategies of
  one privatized scatter-accumulate** (D285, part of #416,
  [docs/device-primitives.md](docs/device-primitives.md)): the attribute
  `#md_exec.scatter` (`tile(E)` on `md_exec.reciprocal`, replacing
  `tiles(E)`; `hash(S, P, L)` on `md_exec.tuple_for`; `sorted` reserved
  for #417) states the strategy, and the pass `md-exec-choose-scatter`
  chooses it, replacing `md-exec-choose-spreading` and the options
  `tuple-block-sums` and `tuple-block-least` of `convert-md-exec-to-gpu`.
  The pass and the lowerings check each precondition with the same
  functions. `MDIR_PME_SPREAD` and `MDIR_TUPLE_SUMS` keep their values.
  The generated kernels, the results, and the speed are those of before;
  no control-file key or output changes. A script or test that names
  `md-exec-choose-spreading` in `MDIR_PRINT_AFTER` or in a pipeline, or
  the removed options, names `md-exec-choose-scatter` and its options
  `tile` and `hash` instead.
- **The kernel of the bonded terms, the 1-4 pairs, and the excluded pairs
  adds the values of a block in block memory first** (D283,
  #408, [docs/device-tuples.md](docs/device-tuples.md), Section 11): on a
  GPU, outside the deterministic mode, each particle that the tuples of a
  block touch is added to the forces once a block instead of once a
  member; a run of fewer than 100,000 tuples (JAC) keeps the atomics. On
  one RTX 4090 the kernel takes 84 µs a step on Cellulose where it took
  221, and 189 on STMV where it took 439; in the Amber suite Cellulose
  runs 9 to 10% faster, Factor IX 2.5%, STMV 3 to 5%, and JAC as before
  (on an RTX PRO 6000 3.5%, 1%, 2%, and as before). The forces change by
  the rounding of their sums, as between two runs.
  `MDIR_TUPLE_SUMS=atomics` keeps the atomics of every value and `=blocks`
  takes the table at any size, for comparison and tests. No control-file
  key changes.
- **Steepest descent ends once its step has shrunk below a floor**
  (D274, #369): $10^{-7}$ Å below double precision and
  $10^{-11}$ Å in double, checked at each row of the log, rather than
  evaluating the energy at the same positions until `steps` ends. Where
  trials no longer lower the energy by more than its rounding or noise
  (a learned potential's, in #369), the step fell to 0 and stayed there.
  The run ends as at `force_tolerance`, with the checkpoint at that row,
  exit status 0, and a line `MDIR: stopped at step N: the step has shrunk
  to ...` that gives the step, the floor, and the largest force. A Python
  minimization ends there as well, and `state().minimization` has
  `stalled`. A run that reached such a step now ends before `steps`.

- **A constraint failure in dynamics stops the run** (D278,
  #349), as a position that is not a number does: a SETTLE or SHAKE group
  whose constrained lengths are off their targets after a step
  ($\lvert r^2/d^2 - 1\rvert$ above $10^{-4}$ below double precision and
  $10^{-8}$ in double, or not a number) ends `mdir run` with a message
  naming the step and the set (`settles`, `shake1`, `shake2`, `shake3`),
  and fails a part of a Python simulation. Before, such a run went on with
  distorted groups. In a run with constraints this message takes precedence
  over that of positions that are not numbers in the same step (the
  maintainer's ruling on PR #381). The kernels of the constraints test the
  lengths they give (`md.gather_tuples` may yield an `i1`); on one RTX 4090
  the Amber suite runs 0.3% to 1.7% slower. `MDIR_CONSTRAINT_TEST=separate`
  tests them in loops of their own instead, for comparison. No key changes.

- **The loop over the inner list of a dual list of groups takes only the
  blocks that hold entries** (D268, #357; the maintainer's
  ruling on PR #360). Each pruning lists them, the index of the blocks of
  the groups is made at each build instead of each pruning, the pruning's
  kernel of the frames copies its configuration, and a loop that follows a
  pruning at the same positions in mixed precision takes its frames. The
  pruning stays global and exact. On an RTX 4090 a step takes 12 µs less
  on 103,292 TIP3P waters (1130 to 1118), 20 on as many OPC waters, and 47
  on STMV. No key, format, or default changes. A run with
  `MDIR_CHECK_PRUNING` set counts, at every step, the pairs within the
  cutoff in the outer and the inner lists and stops if they differ.

- **Particle mesh Ewald on the CPU runs in parallel** (D271,
  #359, step 2 of D267). The spreading adds the charges to the fixed-point
  grid with integer atomics from a loop over the particles, the FFT
  transforms the lines of each pass in parallel, and the product with the
  influence function is a loop over the rows of the grid that sums the
  energy and the virial only on the steps that write them. The result is
  the same to the bit for any number of threads. In the mixed mode the
  B-splines, the grid, its transform, and the gathering are in f32, as on a
  GPU, so the reciprocal energy, forces, and virial of the CPU in that mode
  change at the rounding of f32. No key of the control file changes.
- **The step of steepest descent grows to at most 0.1 Å, not 1 Å**
  (D260, #351; the maintainer's ruling on PR #348). Near a
  cap of 1 Å many trials turned rigid waters further than the constraints
  can take back and were not taken (D259). On 103,292
  OPC waters the largest force falls below 5 kcal/mol/Å by step 500
  rather than 3300, at the same energy after 5000 steps (within 0.03%).
  `initial_step` is unchanged and its first step is taken as given. A
  minimization now gives other rows: the dipeptide of
  `minimize-tolerance.test` stops at step 90 rather than 100.

- **The default neighbor structure is `neighbor_structure = "AUTO"`**
  (D261, #350, stage G3 of `docs/groups-m1.md`). On a GPU,
  outside the deterministic mode, a system of at least 23,000 particles
  whose loops over pairs all take each pair once runs on the groups of 16
  with a dual list, `pairlist_distance` 3 Å and `pruned_distance` 0.6 Å
  beyond the cutoff at 2 fs (wider in proportion to a longer step), for
  the reaches not given; every other run keeps the matrix and its reach of
  1.5 Å beyond the cutoff, as before. A large system that gave no
  structure runs about 1.6 to 2 times as fast (a TIP3P box of 103,292
  waters, RTX 4090: 1971 to 1140 µs a step). The log (`MDIR: neighbor
  structure: ...`), `mdir check` (`neighbors:`, and `run.neighbors` with
  `--json`), the manifest (`neighbor_structure`,
  `neighbor_structure_reason`, `pairlist_distance_angstrom`,
  `pruned_distance_angstrom`; `neighbor_structure_requested` may be
  `"auto"`), and `Program.plan` state the structure and why.
  `MATRIX` and `GROUPS` given are unchanged. In Python,
  `Execution.neighbor_structure` defaults to `NeighborStructure.Auto`.
  To keep the old default, set `neighbor_structure = "MATRIX"`.
  `pruned_distance` under `AUTO` is refused only on the CPU and in the
  deterministic mode, and is left unused, with a note, where the size
  takes the matrix.
- **`FrameEvaluator.evaluate` takes `gradients=` in place of `gradient=`**
  (D252, #313; the maintainer's ruling on PR #314). The
  argument `gradient=True` or `False`, whether the derivative in the
  tunables is kept, is removed, with no alias. A script names what it
  wants: `gradients=("tunables",)` is the default and what `gradient=True`
  was, `gradients=()` what `gradient=False` was, and `"positions"` and
  `"strain"` are the new outputs (under Added). What a user has to do:
  replace `gradient=False` by `gradients=()` and drop `gradient=True`; a
  call that still writes `gradient=` raises `TypeError`.
- A second derivative through `mdir.torch.evaluate` raises
  `mdir.UnsupportedError`, naming the derivative, where it raised
  PyTorch's `RuntimeError` for an operation without a registered
  derivative (D252). It was an error before and is one now;
  a script that caught `RuntimeError` there catches the other class.
- The program of a frame evaluator hands the forces and the virial tensor
  of a frame to the host where the host asks (D252): its
  boundary takes what is asked of each frame, and two calls stand, with
  an evaluation of their own, in branches that a frame without the
  request does not take. The emitted
  module of a program of frames (private, `Simulation._ir` of an
  evaluator) changes by these lines, so the kept code of an evaluator, in
  memory and in the compile cache on disk, is compiled anew once on each
  target, as after #309. The programs of `mdir run` and of a `Simulation`,
  their text, and their entries of the cache are unchanged, and so is
  every number of an evaluator that asks for nothing new.
- **A Python simulation keeps the structures of its tuples across the
  starts of its activations** (D253, #294).
  Each evaluation after a run, update of the tunables, commit of a borrow,
  and start after a checkpoint begins an activation (D215), which
  renumbered the bonds, angles, dihedrals, 1-4 pairs, and excluded pairs
  to the order of the particles and built an incidence structure for each
  on the host, in new memory, and on a GPU uploaded them: 43 MB on JAC of
  the Amber suite and 696 MB on Cellulose, at a cost that grew faster than
  the system because memory over 32 MiB is mapped and faulted in anew at
  every start. A simulation now keeps them, takes them at a start where
  the particles are in the order of the last one, and builds them in the
  same memory where they are not. On one RTX PRO 6000 Blackwell in mixed
  precision, a start with its evaluation at which the particles have not
  moved (`tunables.gradient()` again) takes 2.0 ms on JAC, 6.1 ms on
  Factor IX, and 30 ms on Cellulose where it took 5.0, 61, and 413 ms; an
  update of the tunables with its evaluation 8.3, 30, and 203 ms where it
  took 15, 135, and 867 ms (what remains of an update is not its starts,
  #324); a start after steps, in a new order, 5.1, 17, and 98 ms where it
  took 5.0, 58, and 387 ms: nothing on JAC, whose structures were under
  32 MiB. The rate of a step is unchanged: over five runs of 20,000
  steps of each build in turn, 0.13% on Cellulose and 0.24% on Factor
  IX, of the size of the spread of the runs (0.26% and 0.2%).
  Nothing of the interface changes and no number: every state, energy,
  and derivative of a sequence of starts equals that of before to the
  bit, on the CPU and a GPU, in double and mixed precision, in the
  deterministic mode (`python-kept-tuples.test`,
  `python-kept-tuples-gpu.test`). What
  changes: the pipeline of the program of a
  simulation and of a frame evaluator names a new option of the lowering,
  so the code that a `Program` keeps for its simulations and the entries
  of the compile cache on disk are made anew once; a simulation holds on
  the host two copies of the members of its tuples and one of the order
  for each tuple set (2 MB on JAC, 48 MB on Cellulose) and on a GPU
  16 MiB in which the structures are written, and holds the
  structures between activations as it held them during one. `mdir run`
  is unchanged: its programs, their pipeline, and their lowered text. New
  for diagnosis: with `MDRT_CHECK_KEPT_TUPLES` set in the environment,
  every kept structure that is taken is also built anew and compared, and
  a difference ends the program
  ([docs/environment.md](docs/environment.md#debugging),
  [docs/jit-invariants.md](docs/jit-invariants.md#what-a-simulation-keeps-across-activations-d253)).
- **An update of the tunables of a Python simulation no longer copies its
  model, sorts the members of its tuples, or repeats the quadratures of a
  tail** (D254, #324). `Simulation.tunables.update()` (and the
  commit of a borrow that changes tunables, a simulation that continues a
  checkpoint with other values, and the update of a frame evaluator) put
  the new values into a copy of the whole model, built the program from
  it, and compared its text with the compiled one. On one RTX 4090 in
  mixed precision, of an update with the `gradient()` after it on JAC,
  Factor IX, and Cellulose of the Amber suite (8.4, 31, and 205 ms with a
  constant of a pair term tunable), the rebuild was 4.2, 18, and 143 ms,
  and of that a sort that asks whether each tuple set is disjoint 2.7, 13,
  and 108 ms and the copies of the model 0.3, 2.0, and 11 ms; where a pair
  term has its tail in the correction for the dispersion the rebuild was
  120, 186, and 199 ms, 114, 164, and 40 ms of it the same quadratures
  repeated for every pair of classes of particles. A simulation now keeps
  one copy of its model for its updates and writes the values into it in
  place, a tuple set is found disjoint in one pass over its members, and a
  build computes each integral of a tail once. An update with its
  `gradient()` now takes 5.4, 16, and 87 ms with a constant of a pair term
  tunable, 8.1, 22, and 112 ms with its tail in the correction, and 13, 43,
  and 212 ms with charges, Lennard-Jones, a pair constant, and a tuple
  parameter all tunable (where it took 127, 216, and 376 ms); a start alone
  and the rate of a step are unchanged. Nothing of the interface
  changes and no number: every state, force, energy, and derivative of a
  sequence of updates of every kind of tunable equals that of before to
  the bit, on the CPU and a GPU, in double and mixed precision, in the
  deterministic mode (`python-update-cost.test`,
  `python-update-cost-gpu.test`), and the text of every program and its
  lowered module are those of before to the byte, so no entry of the
  compile cache is made anew. What changes: from its first update on a
  simulation holds the copy of its model (13 MB on JAC, 150 MB on
  Cellulose), which every update allocated and freed before. With
  `MDRT_CHECK_KEPT_TUPLES` set in the environment every update also builds
  its program from a copy of the model of the compile, as before, and
  compares the two; a difference fails the update
  ([docs/environment.md](docs/environment.md#debugging),
  [docs/jit-invariants.md](docs/jit-invariants.md#what-an-update-of-the-tunables-keeps-d254)).
  What an update still costs is its two starts, the arrays of the program
  built from the topology, and the copies of a start (#326).
- **An update of the tunables keeps the arrays of the program that no value
  of a tunable enters** (D256, #328). After D254 an update
  still made every array of the program from the topology. A simulation
  now keeps those that a build from its model with every tunable site set
  to not-a-number makes the same to the bit, shared with the compiled
  program, and makes only the others. On one RTX 4090 in mixed precision
  an update with its `gradient()` on JAC, Factor IX, and Cellulose takes
  4.6, 13, and 65 ms with a constant of a pair term tunable where it took
  5.3, 16, and 87 ms; 5.3, 16, and 75 ms with a parameter of a tuple term
  where it took 9.4, 33, and 171 ms; and 9.3, 27, and 118 ms with four
  kinds of tunable where it took 13, 44, and 212 ms. A step is unchanged,
  and the host holds less after many updates (791 MB against 853 MB on
  Factor IX after 21). Nothing of the interface changes and no number:
  the update sequence of D254 equals that of `main` to the bit on the CPU
  and a GPU, in double and mixed precision. The first update of each set
  of tunables that changed is compared with a program built from a copy
  of the model of the compile; with `MDRT_CHECK_KEPT_TUPLES` every update
  is, and a difference fails the update; without it a difference, which
  only a defect of mdir can cause, is said once on the standard error and
  the simulation builds every later program from a copy
  ([docs/jit-invariants.md](docs/jit-invariants.md#what-an-update-of-the-tunables-keeps-d254)).
- **The segments of `mdir run` build the structures of their tuples in the
  memory of the segment before** (D255, #316). A run
  orders its particles anew where each segment of checkpoints begins,
  and renumbered the tuples and built their incidence structures there on
  the host, in new memory, and on a GPU uploaded them: on one RTX 4090 in
  mixed precision 4.6 ms a segment on JAC of the Amber suite, 27 ms on
  Factor IX, 152 ms on Cellulose, and 465 ms on STMV. A run now takes them
  from the store of the runtime that a Python simulation keeps (D253),
  opened for the run: 4.5, 15, 78, and 238 ms. A run with a checkpoint
  every 10,000 steps is 0.45% faster on Factor IX, 0.88% on Cellulose, and
  0.81% on STMV, and with one every 1,000 steps 3.5%, 5.0%, and 5.5%; JAC
  and the CPU target do not change. A run without checkpoints, such as
  those of the Amber suite, orders its particles once: its rate does not
  change, and it ends sooner by the build of its start (56 ms on
  Cellulose, 194 ms on STMV). The host holds less at its peak, since no
  whole structure is built there: 1,112 to 786 MB on Cellulose and 2,581
  to 1,495 MB on STMV. Every number of a run is that of before to the bit
  (checkpoints and energies on the four systems on a GPU and on JAC and
  Factor IX on the CPU, in mixed and double precision, a run and its
  continuation). What changes: the lowered text of the host's code of
  every program of `mdir run` with tuples, which calls the runtime where
  it had loops (its kernels are the same text, so their entries in the
  compile cache, which are keyed by the kernels, are still found; a
  Python simulation is not changed). With `MDRT_CHECK_KEPT_TUPLES` set,
  the log of a run says at its end how many structures were built and how
  many taken as they were kept; without it the log is that of before.
- **On a GPU the structures of the tuples are built on the device**
  (D257, #295). Where the particles are ordered anew (a
  segment of `mdir run`, a start of a Python simulation after steps), the
  renumbered members and the incidence structures are now built by
  kernels of the program, from the structure in the order of the files,
  whose rows are gathered and whose members are relabeled; the order is
  compared on the device, and only a flag of 4 bytes reaches the host. On
  one RTX 4090 in mixed precision a start in a new order of a Python
  simulation with its evaluation takes 1.98 ms on JAC of the Amber suite,
  5.67 ms on Factor IX, and 26.2 ms on Cellulose where it took 5.29, 18.1,
  and 96.9 ms; `mdir run` with a checkpoint every 1,000 steps is 2.3% to
  4.5% faster on the four systems and every 10,000 steps 0.2% to 0.6%; a
  run without checkpoints keeps its rate and ends sooner by the build of
  its start. The device holds 6% to 10% more than the incidence structures
  (the members and an index of each tuple set: 71 MB on Cellulose, 152 MB
  on STMV); the host holds less (735 MB at its peak on Cellulose where it
  held 822 MB). Every number of a run is that of before to the bit, and
  the CPU target does not change. The lowered text of the host's code of
  every GPU program with tuples changes once; new kernels are compiled
  once and kept in the compile cache. `MDRT_CHECK_KEPT_TUPLES` now also
  compares every structure built on the device with the host's build, and
  `MDRT_PROFILE` reports the bytes that the store keeps on the device.
- **A start of an activation hands the entry the arrays of the program
  instead of copies** (D258, #326). Every
  start of a Python simulation (an update of the tunables has two, and a
  frame evaluator one at every update) copied the state and the whole
  model into new memory of the host for the entry, one number at a time:
  0.44, 1.76, and 11.5 ms of a start on JAC, Factor IX, and Cellulose of
  the Amber suite. The entry now declares the buffers of the model
  read-only (the masses, the fields of the particles, the tables, and the
  members and fields of the tuple sets), `md-exec-assign-storage` takes
  none of them for another field or for a copy back from a device, and a
  new pass, `check-read-only-arguments`, fails the compile wherever the
  lowered code could write one. A start hands those arrays as they are,
  keeps their conversions into the type of the entry (the fields in f32
  in mixed precision) where it is given the same array, and keeps the
  buffers of the state, written anew at each start. On one RTX 4090 in
  mixed precision `gradient()` again takes 1.63, 4.25, and 20.8 ms where
  it took 2.02, 5.97, and 30.2 ms; an update with its evaluation 3.62,
  8.92, and 42.8 ms where it took 4.38, 12.4, and 62.0 ms; and a start in
  a new order 1.51, 3.46, and 15.7 ms where it took 1.91, 5.19, and 25.3
  ms. A step is unchanged. Nothing of the interface changes and no number:
  the update sequence of D254 and a sequence of starts equal those of
  `main` to the bit on the CPU and a GPU, in double and mixed precision.
  The module text of every program gains the attribute `mdrt.read_only`
  on these arguments, so every entry of the compile cache is compiled
  anew once. The code of `mdir run` is that of before but for one buffer:
  on the CPU the identities in a new order go to a buffer of their own (4
  bytes a particle, 1.6 MB on Cellulose), where they were written into
  the buffer of the types of the particles, and on a GPU their copy at the
  end of the run goes to their own buffer; a step does what it did. With `MDRT_CHECK_KEPT_TUPLES` every conversion that a start
  takes as kept is made anew and compared, and every buffer handed
  without a copy is compared at the end of its activation
  ([docs/jit-invariants.md](docs/jit-invariants.md#what-a-start-gives-the-entry-d258)).
- Leapfrog under Langevin dynamics with constraints is faster on a GPU
  (D250, #287): 2.4% on JAC of the Amber suite
  (1166 to 1194 ns/day on an RTX 4090 in mixed precision) and 1.2% on
  Cellulose (66.0 to 66.8 ns/day). The constraints of the positions are
  now one kernel with the drift before them and the loop after them, as in
  velocity Verlet. In the deterministic mode the trajectory is the same to
  the bit; no key, file, or checkpoint changes.
- The program of a frame evaluator takes the rows of its evaluation in
  its loop over frames, as a run takes them in its loops over steps, and
  no longer before it (#309, amendments of D251 and D117). The builder
  moved the allocations of these rows before the loop by matching its own
  text, and refused any other spelling of an allocation in the loop
  (#303); since #305 the lowerings keep the stack of that entry, which
  never returns, whatever the text holds, so the matching and the stack
  part of the check are removed. For the CPU `hoist-static-allocas` takes
  every allocation of a constant size once, and for a program of frames
  it now refuses a module that would keep any other outside the first
  block of a function (`refuse-rest`; no program has one); for a GPU the
  stack of each frame is given back where the frame ends. Nothing of the
  interface changes and no number: every output of an evaluator equals
  that of before to the bit, on the CPU and a GPU, in double and mixed
  precision, and the stack of the entry is as deep after frame 201 as
  after frame 1, to the byte. What changes: the emitted module of a
  program of frames (private, `Simulation._ir` of an evaluator) has these
  rows in the loop, so the kept code of an evaluator, in memory and in
  the compile cache on disk, is compiled anew once on each target. The
  programs of `mdir run` and of a `Simulation`, their text, their
  pipeline, and their entries of the cache are unchanged
  (`python-frames-loop.test`, `python-frames-loop-gpu.test`,
  `hoist-allocas.mlir`).
- **The frame evaluator evaluates the frames of a call inside one
  activation, in the order of the input** (D251, #291).
  `mdir.FrameEvaluator`, `evaluate`, and `mdir.torch.evaluate` are
  unchanged; a frame costs what its kernels cost: on one RTX 4090 in mixed
  precision 0.69 ms on JAC with the whole potential and 0.45 ms with the
  dependent terms, where it took 5.00 ms and 2.07 ms, and 0.28 ms and 0.16
  ms on the dipeptide in water, where it took 0.84 ms and 0.43 ms. What
  changes for a script:
  - The evaluator's program keeps the particles in the order of the input,
    whatever `Execution.reorder` says. The energy of a frame is a function
    of the frame alone, with the same bits in any call, and equals that of
    a sampler compiled with `Execution.reorder = False` to the bit. Against
    a sampler in the spatial order, the default, it differs by the rounding
    of sums in another order: about $10^{-10}$ kJ/mol in double precision
    and $2\times10^{-4}$ kJ/mol in mixed precision. A reweighting, which
    takes both of its energies from the evaluator, is not affected.
  - `FrameEvaluator(program)` compiles a program of its own with
    `terms="all"` as well, once for a `Program` (about 2 to 3 s; a later
    evaluator of the program 0.07 s), and the program of
    `terms="dependent"` is compiled once for a `Program` too.
  - After a frame whose evaluation fails, the next frame pays one start of
    an activation. Errors and what is refused are as before.
- **The scripts of the repository take what a machine provides from the
  environment** (#289). `scripts/release/publish.sh`,
  `scripts/validation/gromacs/run.sh`, and `examples/argon/run.sh` had
  defaults that existed on one machine. They now stop at their start and
  name what is missing: set `MDIR_RELEASE_DIR`, `MDIR_LLVM`, `HDF5_PREFIX`,
  `CUDA_ROOT`, and `MDIR_RELEASE_GPU` for a release, and `LLVM_PREFIX` for
  the argon example; `mdir`, `gmx`, and `tleap` are found on the path or
  named by `MDIR`, `GMX`, and `TLEAP`. `MDIR_ENV_FILE` names a file with
  the values of a machine. `publish.sh --dry-run` rehearses a release
  without a tag ([docs/environment.md](docs/environment.md#scripts)).
- **The build of a release has the Python interface, and its suite runs
  the Python tests** (#297): `scripts/release/publish.sh` also needs
  `MDIR_PYTHON` and `PYBIND11_DIR`, and gives lit the interpreters that
  `MDIR_TORCH_PYTHON` and `MDIR_H5MD_PYTHON` name. Until now a release was
  tagged on a suite in which every Python test was unsupported.
- **Leapfrog with constraints: every step leaves the state of the plain
  step, and under Langevin dynamics the plain step is the documented one**
  (#285). With SHAKE or SETTLE, a step that writes a row of energies (and
  every other step with the virial) began from the projected velocities of
  the time of the positions, as velocity Verlet does, and the plain step
  from a kick of a whole step. The two agree to rounding only while the
  velocities do not change within the drift. Trajectories change as
  follows; control files, checkpoints, and file formats do not.
  - *Leapfrog, Langevin dynamics, constraints.* The friction acted on the
    part of the kick along the constraints, and the stored velocities of a
    plain step were off the documented step (that of velocity Verlet,
    D135) by $(1 - e^{-\gamma\Delta t})/2$ of that part: 1.3 × 10⁻³ nm/ps
    after one step at 1/ps and 2 fs on a dipeptide in water, 2.9 × 10⁻⁵ nm
    in the positions after 20 steps. Every step now takes the form of
    velocity Verlet, and leapfrog follows velocity Verlet to rounding
    (2.9 × 10⁻¹⁴ nm after 20 steps); a NumPy reference of the step with
    constraints agrees within 2 × 10⁻¹² nm/ps. The plain step has one
    projection of the velocities more: on JAC of the Amber suite (23,558
    atoms, RTX 4090, mixed precision) 0.149 ms a step instead of 0.135,
    9% slower, and 0.160 instead of 0.148 with stochastic cell rescaling.
    Runs with velocity Verlet are not affected.
  - *Leapfrog, constraints, the barostat with `work =
    "TROTTER_FIRST_ORDER"`.* The step that scales the cell had the same
    defect with $1/\mu$ for the friction when it took no energies:
    8.3 × 10⁻⁷ nm after 20 steps at 2000 atm. It now takes the form of
    velocity Verlet with or without them.
  - *Leapfrog, constraints, otherwise* (constant energy, velocity
    rescaling, the Nosé–Hoover chain, the other steps of the barostats).
    The state after a step with the virial is now that of the plain step
    to the bit, where it differed by rounding (2 × 10⁻¹⁴ nm after 20
    steps); the step takes the drift of velocity Verlet on the side for
    the virial of the constraints. The rows of the energy file are the
    same at their six decimals.

  Two runs that differ only in `energy_interval`, and a Python simulation
  in parts that do not follow the rows of `mdir run`, now end in the same
  state to the bit in the deterministic mode. Leapfrog without constraints
  was so already.
- **The defaults of the Python model are those of the control file**
  (D248, #281). A script that relied on a default below gets
  another model than before; the last column gets the old one back.

| Setting | Before | Now, as the control file without the key | To get the old model |
|---|---|---|---|
| `System.truncation` | `Truncation.Switch` | `Truncation.None_`, a plain cutoff | `system.truncation = mdir.Truncation.Switch` |
| `System.switch_distance` | 1.0 nm | the cutoff (no switch); it follows `System.cutoff` until it is set | `system.switch_distance = 1.0` |
| The correction for the dispersion, in effect | off, with the warning `dispersion_switched` (the default switch turned it off) | on (`EnergyPressure`), without a warning | follows from the switch; or `system.dispersion = mdir.DispersionCorrection.None_` |
| `System.pairlist_distance` | 1.35 nm | 0.15 nm beyond the cutoff; it follows `System.cutoff` until it is set (equal at the default cutoff) | `system.pairlist_distance = 1.35` |
| `Ensemble.com_period` | 0: the motion of the center of mass is never removed | `None`: with a thermostat every `coupling_period`, without one never | `ensemble.com_period = 0` |
| A state without velocities | begins at rest | takes the velocities that `mdir run` draws: at `Ensemble.temperature`, with `Ensemble.seed` | `state.velocities = numpy.zeros((N, 3))` |

  A system given to both front ends with nothing beyond its inputs now
  runs to the same bits in the deterministic mode. `mdir run` and the
  control file are unchanged.
- `Ensemble.coupling_period` of the Python model reads `None` until it is
  set, where it read 10 (D249, #278): the control file
  without `interval`, as the defaults above. For stochastic velocity
  rescaling and a Nosé–Hoover chain that is every 10 steps as before (or
  `com_period`, where that is set and not 0, which was refused when it was
  not 10); under Langevin dynamics it is 10 with a barostat and no period
  without one. A script that reads the attribute before setting it gets
  `None`; `ensemble.coupling_period = 10` states the old value.
- With the groups, `pairlist_distance` may be up to 0.999 of the least
  edge of the cell (the least of $a_x, b_y, c_z$ of a triclinic one),
  where a triclinic cell was limited to half of it and an orthorhombic
  cell was not limited (D242, #263). The builder refuses a
  longer one ("exceeds 0.999 of the least edge of the cell"), a commit of a
  writable borrow does, and a run whose barostat takes the cell below
  `pairlist_distance`/0.999 stops, as one does at twice the cutoff.
- A build of the neighbor matrix on a device orders the particles of a
  cell by rank, one thread per particle, in place of an insertion sort in
  one thread per cell (D235, #231). A system of about a
  thousand particles has a few cells of more than a hundred particles
  each, whose sort took most of a build: ethanol in 467 waters at constant
  pressure runs at 0.115 ms a step where it ran at 0.291 (RTX 3090, mixed
  precision). The order, and so every result, is unchanged.
- A program no longer holds the values that depend on the state it starts
  from as constants of its text (D227, #162): the
  tilts of a triclinic cell, the cell of the restraints' reference
  positions under a barostat, the barostat's constants of the dispersion
  correction and the PME background, and the barostat state of a continued
  run that scales every step are arguments of the entry. A stage that
  starts from another equilibrated cell, such as the production stage of
  the ala3 example, therefore hits the compile cache when its PME grid and
  neighbor capacity are the same. Results are unchanged to the bit.
- The estimate of the neighbor capacity (`[execution] neighbor_capacity`
  absent) is rounded up to four significant bits instead of a multiple
  of 8, at most 12.5% more room at first, so that it is the same for the
  starts of continued stages (D227). The capacity
  does not change the results; the Amber suite runs at the same rate.
- The Python extension is the module `mdir._core` of the package `mdir`
  (D228): a build puts `mdir/__init__.py` and the extension in
  `<build>/python/mdir`, and `cmake --install` in `MDIR_PYTHON_INSTALL_DIR`,
  which now names the package's directory (by default
  `lib/pythonX.Y/site-packages/mdir`). `import mdir` and every name under
  it are unchanged; `PYTHONPATH=<build>/python` still imports the build. A
  cache that set `MDIR_PYTHON_INSTALL_DIR` should append `/mdir`.
- Setting `System.dispersion` in Python makes the correction for the
  dispersion explicit, as giving `dispersion_correction` in the control
  file (D222, #161): with a switch it is refused, and a
  divergent tail is an error; assign `None` for the default. Without a
  periodic cell the default is now off, as in the control file; before,
  compiling such a system needed `System.dispersion = DispersionCorrection.None_`.
- With a switch (`Truncation.Switch`, the default, `ForceSwitch`, ...) the
  Python model's default correction for the dispersion is now off, and
  `mdir.compile` warns `dispersion_switched` (D222, #161):
  the energies and pressures of a default switched `System` lose the tail
  of the Lennard-Jones and of its pair terms, which it applied before
  though the switch takes part of the potential below the cutoff. For the
  correction, use `Truncation.None_` or `Truncation.Shift`.
- A typed Python restraint keeps its force constant in kJ/mol/nm² exactly
  (D216, #100). Before, it passed through the control
  file's kcal/mol/Å², and a constant not computed from a control-file value
  could change by one rounding. Runs from a control file are unchanged.

- `mdir.compile` no longer lowers the program (D224, #151): it
  builds it and sets up its pipeline, so `Program.ir`, `pipeline`, and
  `plan` are ready on return, and `Program.lowered_ir` lowers on its first
  read, once. A `Simulation` lowers programs of its own and never used that
  lowering. `mdir.compile` takes 0.3 s instead of 3.5 to 9.2 s a stage of
  the Python ala3 example on a GPU, and the example at a hundredth of its
  steps 62 s instead of 85 s; results are unchanged to the bit. An error
  that only the MLIR pipeline finds is now raised by `Simulation(program)`
  or by the read of `lowered_ir` instead of by `compile`.

- Under a plain cutoff (`lennard_jones_modifier = "NONE"`, a Coulomb
  cutoff, or the direct sum of PME without `coulomb_modifier`), the
  free-energy file (`dHdl.<name>`, `dU.<k>`) and the observables file
  (`<term>.energy` and its derivatives) now take the potential that the
  forces sample: each pair within the cutoff less its energy at the cutoff,
  as under `POTENTIAL_SHIFT`, whose files they now equal on the same
  trajectory. Before, they took the potential cut without a shift, whose
  derivatives differ by a term that fluctuates with the number of pairs
  within the cutoff, and biased reweighting gradients by 18% (#140) and
  thermodynamic integration by $\langle N_\text{pairs}\rangle\,\partial
  u(r_c)/\partial\lambda$. The forces, the trajectories, the energies of the
  log and of the energy file, the virial, and the pressure are unchanged.
  With a topology, `dispersion_correction = "ENERGY_PRESSURE"` is now
  allowed with `lennard_jones_modifier = "POTENTIAL_SHIFT"` (a switch stays
  refused): it adds the estimate of what the shift takes within the cutoff
  at a uniform density, with no virial, to the energy of the log under the
  shift, and to the free-energy and observables files under either modifier
  (D210, #144).

- With constraints, the temperature of the log is now the optimal estimate
  of Jung, Kobayashi, and Sugita (2019), $N_fk_BT = \tfrac43K_\text{half} +
  \tfrac23K$, and its pressure takes $K_\text{half}$, the mean of the
  kinetic energies of the half steps, measured from the velocities after the
  constraints; before, both took the kinetic energy of the step $K$ with
  constraints (without them the log already took these estimates). Rows of
  runs with constraints read 1 to 2 K higher at 2 fs and 4 to 5 K at 4 fs; the
  dynamics, the thermostats, and the barostats are unchanged. With rigid
  waters, `mdir run` ends with the temperatures of the solute and of the
  solvent (D203, #112).
- In the deterministic mode, a step that writes energies moves the particles
  as one that does not with constraints and PME too, and a Python simulation
  of a model equals `mdir run` within a part on the CPU. The trajectories of
  deterministic runs therefore change: in mixed precision the forces that
  the steps carry are stored in f64, and checkpoints and Python states hold
  forces in f64 (`force_dtype` is `float64`); on a device the kernels make
  fused multiply-adds by the formula rather than none (#102, #105,
  D204). Runs outside the deterministic mode are
  unchanged.

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

- The test suite runs at most sixteen tests that need a GPU at once instead
  of four (`-Dgpu_workers=N` still sets the number), and the three Python
  tests that bounded its wall time, with their CPU twins, are split into one
  file per scenario. On one RTX 3090 the suite takes about 260 s instead of
  about 1090 s (D208, #141).
- The free-energy, Python minimization, and velocities-and-restraints tests
  that D208 left whole are split by scenario as well, on the CPU and on a
  GPU; the runs of the restraints' effect stop after their first part. The
  slowest file on a GPU is now a `python-simulation-lifetime` one, which
  runs one process's whole sequence (D208, #141).

- A Python `Simulation` compiles one program, whose entry does the work of
  the start only on its first call, instead of a second program for the
  segments after the first on its first part; every lowering in the
  process uses MLIR's threads, from one pool shared by the process. The four
  stages of the Python ala3 example at a hundredth of their steps take 146 s
  instead of 240 s on a GPU in mixed precision, and 162 s instead of 251 s
  in double, with less CPU time; results are unchanged to the bit in the
  deterministic mode (D211, #147).

- A Python `Simulation` runs every part in one activation of its entry,
  whose buffers, order of the particles, and neighbor structures stay
  where the program keeps them, on the device, from part to part; the host
  copies the state only for `state()`, callbacks, and updates of tunables.
  A run in parts now equals the same steps in one part to the bit in the
  deterministic mode (energies, energy files, and trajectories included);
  before, each part put the particles in order and built its neighbor
  structures anew, and a run in parts agreed with one run only within that
  rounding, so its numbers change in the last bits. A part that fails still
  keeps the state of the last part that succeeded, from a copy on the
  device. A live simulation now holds its memory between runs. Short parts
  are faster: a part of 10 steps of JAC takes 2.75 ms instead of 10.3 on a GPU in mixed precision, and of one step 0.309 ms instead of 8.77; long runs are unchanged (D215, #135).
- Lowering a program whose pair kernels take tables in f32 (D94) is
  faster: the table of a function of the distance is searched for once
  for a function, a cutoff, and a tolerance in a process, instead of three
  times in each lowering, and the function is evaluated through a list of
  steps. On the dipeptide in water with PME (CPU, mixed precision)
  `md-exec-expand-radial` takes 0.025 s instead of 0.38 and
  `md-exec-simplify-distance` 0.013 s instead of 0.21, and
  `pipeline_seconds` of a `Simulation` falls by about 40%; a pair term
  that no table holds (the Stillinger-Weber form, D159) lowers in 0.56 s
  instead of 10.9. The tables and the compiled programs are the same to
  the bit (#236).

### Added

- `examples/ala3/run.py`: the four stages of the ala3 example (minimization,
  NVT, NPT with restraints, production) from Python, with the settings of its
  control files, reporters in production, and a PEP 723 header for
  `uv run` (#82).

- Python: `System.coulomb_modifier` (`CoulombModifier.None_`,
  `PotentialShift`), the control file's `[energy] coulomb_modifier` for PME
  (D205, #127).

- Python: reporters of a simulation, `Simulation.reporters` with
  `EnergyReporter`, `TrajectoryReporter` (DCD, XTC), and `CallbackReporter`;
  the energy file and the trajectory are those of `mdir run`, written inside
  the parts of a run at the periods given (D207, #109).

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

- `pme-tiles-gpu.test` failed in about 1 of 300 runs at order 8 in mixed
  precision, a relative rms of the forces of 2.46e-6 against 2e-6 (#427).
  The spreading was not at fault. The test compares the forces after one
  step of 0.001 ps, and the rounding of the forces of step 0 moved the
  positions by about 1e-12 nm, enough for one coordinate near a midpoint
  between two numbers of f32 to round to the other one when the positions
  are converted for the kernels (D79). Its step is now $10^{-6}$ ps, after
  which the positions of two runs agree to 4e-16 nm, and the forces of the
  two spreadings differ by 0.7 to 1.2e-7. The tolerances are unchanged.
  `test/Sanitizer/pme-tiles-gpu.test` runs the spreading of order 8, with
  and without tiles, under compute-sanitizer, synccheck included.

- `InitialState.draw_velocities` refused a system with a term of
  `[[energy.model]]` (`System.models`): it prepared the system for the
  default execution, on the CPU, which a model term refuses (#388). The
  draw now leaves the model terms out of its preparation; they change
  neither the masses nor the constraints, so the velocities are those of
  the same system without them, bit for bit, and no plugin is loaded.
  `compile` still checks the term and its target.
- `--convert-md-dist-to-loops` asserted on a function that holds ops of
  `md_dist` and an op of no registered dialect, which `--allow-unregistered-dialect` admits (#383).
  It took the dialect of every op it walked and of every user of a value
  of `md_dist`, which is null for such an op. The op is now taken as one
  of another dialect: left in place, or refused with the pass's diagnostic
  when it uses a value of `md_dist`.
- compute-sanitizer's initcheck reported 4.7 million reads of memory that
  nothing had written in a GPU run of 20 steps of NPT on examples/ala3 in
  mixed precision (#347). The kernels of an integration run (D110) read
  the first tuple of every particle in the incidence structure that the
  device builds and keep it only where the particle has a tuple; the slots
  past a particle's count were never written. The build now writes zeros
  there. The values read were never used, so the results do not change.
  `test/Sanitizer/npt-velocity-verlet-gpu.test` runs that case under
  initcheck, memcheck, and racecheck.
- `md-exec-assign-storage`, `md-exec-fold-tables`,
  `convert-md-exec-to-loops`, and `convert-md-exec-to-gpu` asserted on an
  op of no registered dialect, which `--allow-unregistered-dialect` admits
  (#395): they took an op's dialect with `isa<...Dialect>`, which asserts
  on the null dialect of such an op. They now use `isa_and_nonnull` and
  leave the op where it is, as an op of another dialect. The same form is
  used in `md-exec-simplify-distance`, which an unregistered op does not
  reach there: only ops whose values it can evaluate reach the check.
- A process that used more than 8 plans of cuFFT ended with "mdrt: too
  many grids of FFT" and `abort()` (#323). The runtime of a device keeps a
  plan for each grid, direction, and precision of the process, and a
  program in mixed precision with tunable charges takes four a grid, so
  three such grids in one Python process ended it with no Python error.
  The table now grows. A plan of cuFFT that cannot be made, or a transform
  that fails, stops the run: a `SimulationError` of the simulation or the
  frame evaluator in Python, and an exit with status 1 of `mdir run`, in
  place of an abort. `python-frame-gradients-torchpme-gpu.test` runs its
  two systems in one process again.

- A `.gro` file whose velocities do not read in fields as wide as its
  coordinates is refused, with the line, instead of read with the
  velocities that do not read taken as 0 (#365). Velocities written one
  column wider than the coordinates (`%12.7f`, then `%13.8f`) kept the
  first component, cut to the width, and lost the other two: 81 waters
  written at 300 K started at 89 K. The velocities of a `.gro` file are in
  fields as wide as its coordinates, with one more decimal, as GROMACS
  writes and reads them at any precision; a line has three of them or
  none, and every line alike.
- Steepest descent stalled with a step of 0, or failed, on rigid water
  (D259, #343). A trial longer than SETTLE or M-SHAKE can
  take back, which the step reaches as it grows toward its cap of 1 Å,
  left a water torn in mixed precision (an O-H of 8 Å). The trial was
  taken when its energy was lower; every later trial then held the change
  that takes the water back, which does not shrink with the step, so the
  step shrank to 0 and the run of dynamics after it failed. In double
  precision SETTLE gave positions that are not numbers, and on a GPU the
  run failed at a build of the neighbor structure. Such a trial is now not
  taken, and the step shrinks. With `main`, 1000 steps on a GPU in mixed
  precision stalled for 103,292 OPC waters (by step 200) and for 31,808
  (by step 300); with the fix both go 5000 steps, to -14.37 and -14.29
  kcal/mol per water in mixed precision and -14.37 and -14.29 in double,
  with every water at its shape, and NVT from the large box runs (10 ps
  in mixed precision, 2 ps in double).
  `minimize-rigid-water.test` and `minimize-rigid-water-gpu.test` take
  three OPC waters with a first step of 2 Å.

- The test of the memory of the host across the frames of a frame
  evaluator read bytes of its own (#306). It made the `ctypes` structure
  of `mallinfo2` and the handle of the C library anew at every reading,
  which moved the bytes in use by 16 up and down and by 576 once with no
  evaluator at all, and its bound of 4096 bytes over 100 frames let that
  pass. Read with a structure and a handle made once
  (`scripts/validation/frames/heap.py`), the bytes in use of the host do
  not move with the frames or the calls of an evaluator: +0 between
  readings at powers of two of calls to 131,072 frames, on the CPU and a
  GPU, in double and mixed precision, for calls of 1 to 256 frames with
  and without the Jacobian. Nothing of `mdir.FrameEvaluator` changes.
  `python-frames-loop.test` and `python-frames-loop-gpu.test` now bound
  the growth by 64 bytes over 100 frames, in calls of 25 frames and of
  one.
- A run on the CPU takes the rows that it writes once on its stack, not
  again at every row (#305, an amendment of D117). The observed columns
  (`observe`), the coordinates and the terms of pulling (`pull`), and
  dH/dλ with the energies of the states (`free_energy`) are written to
  rows that the program allocates where it writes them, in its loops over
  steps, and memory of the stack taken in a loop is given back only when
  the function returns: 16 bytes for every two numbers of a row, at every
  row. Measured on the CPU: 32 bytes a step of energy of a Python
  simulation with 4 observed columns (48 with 5, 80 with 9), whose entry
  never returns between parts (D215), so that the 64 MiB of its stack
  last 2.1 million steps of energy; and in `mdir run` 80 bytes a row of
  one term of pulling over two centers, 144 of two, 32 of 3 observed
  columns, and 192 of a free energy with 11 states, until the stack of
  the process is full: under the usual 8 MiB a run of pulling with a row
  at every step ended with a segmentation fault at row 100,902. Runs
  without these rows did not grow, and neither did any run on a GPU,
  where the lowering gives back the stack of each iteration of a loop of
  the host. `hoist-static-allocas` now runs a second time where the
  lowering for the CPU ends, for every program, and every allocation of a
  constant size is taken once, where its function begins. No number
  changes: the energies, the observed columns, the rows of pulling and of
  the free energy, and the checkpoints equal those of before, byte for
  byte and bit for bit in the deterministic mode, on the CPU and on a GPU
  in double and mixed precision. What changes: the pipeline of a CPU
  program (`Program.pipeline`, `mdir emit --stage=pipeline`) names the
  pass once more, and the lowered module of a CPU program with a topology
  has its allocations in another place, so its entry of the compile cache
  on disk is compiled anew once; the emitted module and everything of a
  GPU program are unchanged (`python-stack-rows.test`, `stack-rows.test`,
  and their GPU twins).
- A run keeps nothing for a row of its energies (#300), in `mdir run` and
  in a Python simulation. Each row kept its time and, with rigid water,
  the temperatures of the solute and of the solvent, for the two lines of
  the summary at the end of the log of `mdir run` (the rate from the
  middle of the run, and the mean temperatures with the errors of ten
  blocks, D203): 48 bytes a row in vectors that double. Measured: +48
  bytes a row of `mdir run` with a row at every step, which is 24 GB for
  5e8 steps, and +64 bytes a call of `run(1, energy=True)` of a
  simulation, which printed no summary, on the CPU and on a GPU (0 for
  `run(1)`), about 70 MB of resident memory after 1.5 million calls. The
  summary now takes sums in constant memory, in the order of the pass over
  the rows that they replace: its two lines are the same to the digit,
  and the memory of the host is flat across rows (`output-rows.test`,
  `python-memory.test`).
- `mdir run` without a log file keeps no text of its log (#304). The log
  keeps what it prints until its file opens, so that the file holds the
  lines before it; a run without `log` in `[output]` opens none, and
  every line stayed in memory until the run ended, in a string that
  doubles: +154 to +170 bytes a row of the energies between rows 512 and
  65536, in dynamics and in a minimization (a row is about 121 bytes of
  text), which is about 60 GB of text for 5e8 steps with a row at every
  step. The run now tells its log after the set-up that no file will
  open, and the log frees what it holds and keeps nothing; with a file it
  frees the lines before the file once they are written. The standard
  output and the log file are the same bytes, and the memory is flat
  across rows with and without a log file (`output-rows.test`). The log of
  a Python simulation is quiet and never kept text.
- A writable borrow of a simulation with the groups refuses an orthorhombic
  cell whose least edge is below the pairlist distance over 0.999, as it
  refused a triclinic one (D242, D245); the check was skipped
  for orthorhombic cells, which no Python program with the groups could
  reach before.
- A program with a dual list that also keeps a neighbor matrix (the loops
  of the derivative in the tunables, `System.tunable_gradient`) lowers:
  `md-exec-choose-neighbors` takes the reach and the test of the inner
  list from the refreshes of the structures that stay matrices
  (D245). Before, the lowering refused it.
- `Simulation.view()` and `borrow()` of a simulation without an activation
  of its program say why it has none (#220, `docs/python-dlpack.md`). After
  `save_checkpoint` or a `CheckpointReporter` the message was that of a
  simulation that has not run ("before its first run"); it now says that a
  checkpoint ended the activation, at which step, that `state()` copies the
  state of the host, and what brings the buffers back. The other states
  have messages of their own: a simulation created with `checkpoint=` that
  has not run since, a commit of a borrow or an update of tunables that
  failed and was undone, an update of the tunables of a minimization, and a
  failure. The remedy named is one that the simulation accepts: a run, or
  `run(0, energy=True)`; a run alone for leapfrog without tunables after
  its first run, where the evaluation is refused; `minimize(steps)` for a
  minimization, whose message named `run(0, energy=True)` before. A read
  still does not evaluate: `view()` starts no activation.
- With the groups (`neighbor_structure = "GROUPS"`, with or without a dual
  list) in an orthorhombic cell, a pairlist distance of more than about
  the edge of the cell less the half-width of a group lost pairs (#263,
  D242, `docs/groups-m1.md`, Section 5.2): the lists took the
  image of a candidate nearest to the center of a group and the one on the
  other side of the boundary along each axis, which are not every image
  within such a reach, and nothing refused the input. Measured on 216 argon
  atoms: in a cube of 17.2 Å with a pairlist distance of 15 Å a pair was
  missing at 126 of 2000 steps (force errors of 0.2 kJ/mol/nm), with 12 Å
  none; in a cube that a barostat took from 22 to 19.4 Å with 17 Å, the
  potential lacked pairs in 23 of 79 frames. It takes a skin of about half
  the cell: the cells of ordinary runs, wider than twice the pairlist
  distance, are not affected, and v0.1.0 has the defect only for such
  inputs. The lists now take the 27 images a cell away or none along each
  axis and hold every image within a reach below the least edge of the
  cell, in an orthorhombic and in a triclinic cell.
- Every use of the HDF5 library holds one mutex of the process (#262,
  `docs/python-checkpoints.md`, Threads). The library is thread-safe only
  if built so (that of the reference build is not), and the writer and the
  reader of checkpoints took no lock, while those of H5MD (D239) took one
  of their own: in a script with threads, `save_checkpoint` or a
  `CheckpointReporter` of one simulation, which write outside the mutex of
  the runs and without the GIL, could be inside the library together with a
  frame in H5MD or a checkpoint of another simulation, or with
  `read_checkpoint`, `Simulation(program, checkpoint=)`, or `read_h5md` in
  another thread. A test of four such threads crashed or misread sound
  files in 30 of 30 runs without the lock in the checkpoint code, and
  passes with it. `read_checkpoint`, `Simulation(program, checkpoint=)`,
  and `read_h5md` now release the GIL while they read. One simulation, or
  one `mdir run`, was not affected. A module that brings its own HDF5 into
  the process (h5py) is outside the mutex.
- In a triclinic cell the neighbor matrix (the default structure, on the
  CPU and on a device) lost pairs once its reach, `pairlist_distance`, was
  more than half of the least of $a_x, b_y, c_z$ (#258, D241,
  `docs/triclinic-m2.md`, Section 2). The builder refused such an input,
  but a barostat could shrink an accepted cell past the bound, and the run
  went on: its build tested a candidate at the image of the one pass along
  $\mathbf c$, $\mathbf b$, $\mathbf a$, which is not the nearest image
  beyond that distance, left the pair out, and the pair was absent when it
  came within the cutoff. Measured on 216 argon atoms in a cell with every
  tilt on its bound (cutoff 9.2 Å, pairlist distance 11 Å, half of the
  diagonal at the start; 20,000 bar): the potential lacked up to 8 pairs
  (0.019 kcal/mol) in 9 of 79 frames. At constant energy with the reach
  16% past the bound, a pair was missing at 24% to 43% of the steps, with
  force errors up to 0.3 kJ/mol/nm of forces of several hundred. The
  errors are those of pairs at the cutoff, and no pair was counted twice.
  v0.1.0 has the defect: a run at constant pressure in a triclinic cell
  with the matrix whose cell came below twice the pairlist distance, and a
  run continued from its checkpoint (refused), are affected; runs whose
  least of $a_x, b_y, c_z$ stayed at least twice the pairlist distance,
  orthorhombic cells, and the groups are not. The build now takes a pair
  if any of its images is within the reach, so the matrix is right for
  every reach, and the builder and a commit of a writable borrow ask a
  triclinic cell with the matrix only for a diagonal of at least twice the
  cutoff, as the runtime does; `pairlist_distance` is no longer limited to
  half of the least of the diagonal there, and is refused above 127 times
  `cutoff`.
- The documents said that frames rounded to the precision of XTC cannot be
  used for reweighting (D239, D240). They can, for tunables of soft
  nonbonded terms and of the charges: the rise of the energy of a rounded
  frame (about 50 kJ/mol for the dipeptide in flexible water) is of the
  stiff terms and cancels in the difference of two energies evaluated at
  the same rounded frame. It does not cancel for a tunable of a stiff term
  or against reference energies recorded by the run.
  `docs/python-frames.md` has the measured table for f64, f32, and
  $10^{-3}$ nm (#264, `scripts/validation/frames/precision.py`); frames in
  f64 stay the default because they serve any tunable chosen later. No
  code changes.

- In the deterministic mode on the CPU, a run depended on `energy_interval`
  at the last bits of the forces of the torsions (#243, D204): the code
  generator computed the sine of the force of a torsion by `sincos` of the
  C library in a step that writes energies, whose kernel has the cosine of
  the energy as well, and by `sin` in a step that does not, and the two do
  not always round alike (7 arguments in 10,000 with glibc 2.35). Seen in
  double precision under NPT with `work = "TROTTER"` or
  `"TROTTER_FIRST_ORDER"` (a velocity of the dipeptide in water by 8e-17
  nm/ps after 400 steps, or 2e-16 after 40 with a term over centers), it
  could occur in any ensemble in double precision; in mixed precision the
  torsions are in f32, where the two functions of that glibc agree. In the
  deterministic mode a sine and a cosine are now calls of `sin` and `cos`
  each (`separate-sin-cos`). A deterministic run on the CPU may differ
  from before in the last bits where a step of energy met such an
  argument; the default mode and runs on a GPU are unchanged.
- A Python simulation that continues from a checkpoint of a triclinic cell
  runs with the tilts of the checkpoint (#257, D238). It ran
  with the diagonal of the checkpoint and the tilts that its program was
  compiled from, while `State.cell` reported the cell of the checkpoint:
  after an NPT run of `mdir run` in a rhombic dodecahedron, whose tilts
  the barostat had moved by 0.081 nm, a Python stage compiled from the
  coordinate file evaluated +18855 kJ/mol where the state has −12438
  kJ/mol, and failed within 50 steps. A program compiled from the state of
  the checkpoint, and an orthorhombic cell, were not affected.
- The frames of a `TrajectoryReporter` that is open when a writable borrow
  commits a cell take the committed cell (D238). They kept the
  cell of before: after a commit of the edges and the positions scaled by
  1.01, the next frame of a DCD had the edges of before, 0.03 nm short.
- A Python system that declares a tunable of the charges and a tunable
  constant of a pair term gives both derivatives with
  `System.tunable_gradient` (#256, an amendment of D230). It compiled and
  failed at its first `mdir.Simulation` ("'md.sum_relation' takes argument
  2 of 'tunable', a field, within another op"), with PME and under a
  Coulomb cutoff, with the charges free or tied: the entry asks for the
  derivatives in the numbers and then in the fields in one evaluation, and
  `md-differentiate` took the sums that the derivative in a number had
  added, which gather the charges as every pair term does, for uses of the
  charges by the energy. The derivative in a field now follows the ops of
  the potential alone, after the forces and the virial as well. The
  derivative in each tunable equals that of the program that declares it
  alone to the bit (28 combinations of the charges with the other
  tunables, CPU and GPU, double and mixed).
- `mdir.compile` refuses a derivative without a rule, as D230 says: it
  runs `md-check-exchange` and `md-differentiate` on the program and on the
  program of the segments of a simulation, lowering nothing, and raises
  `InputError` with the op and the name of the tunable. Such a program
  compiled and failed at `mdir.Simulation` with `CompileError`, since
  `compile` does not lower (D224).
- The Python wheels and the container image carry the notices of the
  third parties whose code they distribute in binary form, as the release
  tarball does (#221). A wheel held pocketfft, toml++, LLVM, and the
  OpenMP runtime without their notices; it now has `pocketfft-LICENSE`,
  `tomlplusplus-LICENSE`, `LLVM-LICENSE.TXT`, and `OpenMP-LICENSE.TXT` in
  `mdir/licenses`, beside those of MDIR, HDF5, and pybind11. The image
  held no notice of its own: it now has MDIR's license and the same files
  with `HDF5-COPYING` in `/opt/mdir/share/mdir`, and the CUDA EULA for
  libdevice as `/usr/local/cuda/EULA.txt`. Every install of MDIR now puts
  the notices of what its binaries hold into `share/mdir/licenses`.
  `scripts/release/check-wheel.sh` checks each notice of a wheel and
  fails on a grafted library it knows no notice for, and the new
  `scripts/release/check-image.sh` checks the image.
- A pair term that the correction for the dispersion leaves out no longer
  aborts the build of the program when another pair term follows it (#238).
  A term whose tail diverges or that reads `t` is left out with the warning
  `pair_tail_left_out` under the default correction (D209); the collection
  of the tails ended there, so the terms after it had no entry, and
  `mdir emit`, `mdir run`, and `mdir.compile` aborted in the builder, while
  `mdir check` passed. The order with such a term last ran. `groups` played
  no part: a `1/r` term over all pairs, or one of `t`, before another term
  did the same, with `[free_energy]`, `observe`, or a tunable in the later
  term as well, on the CPU and a GPU. Every term now has its tail whatever
  its place; the energies, the forces, the virial, and the correction of
  the inputs that ran are unchanged, and the two orders give the same
  (`pair-term-order.test`, `python-pair-order.test`, and their GPU twins).
- In the deterministic mode and mixed precision, a run with a term over
  the centers of groups (`groups` of `[[energy.bond]]`, `[[energy.angle]]`,
  or `[[energy.dihedral]]`) depended on `energy_interval` (#240, D204): a
  step that writes energies added the forces of the 1-4 pairs to the sum
  of the terms before them in f64, and a step that does not accumulated
  them in f32, so the forces on the atoms with 1-4 pairs were rounded
  otherwise (1e-7 nm in the positions of the dipeptide in water after 40
  steps, on the CPU and on a GPU, at constant energy, at constant
  temperature, and under NPT with `work = "TROTTER"` or
  `"TROTTER_FIRST_ORDER"`). The terms are now added in one order and one
  type in every kind of step. A run with such a term may differ from
  before at the rounding of f32 in every mode; a run without one is
  unchanged.
- An output no longer counts for `rebuild_interval`
  (D237, #233). The potentials of `[output]
  observables`, of the free-energy file, and of a pull file share the
  neighbor structure of the steps, and each of their evaluations was a
  refresh of it: with `rebuild_interval = 4` a run with the three files
  built the structure 33 times in 80 steps and without them 22 times, and
  ended in another state. A refresh that repeats the last one of its
  structure is now removed, so the outputs read the structure of the step.
  Runs without `rebuild_interval` do not change. This is also the cause of
  the trajectories that depended on the free-energy file under the barostat
  of Trotter type before the fix of #218: the refresh of the output, in the
  cell of before the scaling, built the structure where the steps would not
  have, and the steps summed their forces in another order. No pair within
  the cutoff was missing from such a structure in 20,000 instrumented
  steps; the trajectories of those runs are valid.
  `output-independence*.test` compare runs with and without each output.
- The estimate of the shift of the correction for the dispersion (D210)
  under a barostat (#224). It is `c1/V - c2`, and the whole of it was scaled
  with the volume as the tail is, which left `c2 (1 - V0/V)` in the energy
  of the log and the conserved energy under `POTENTIAL_SHIFT`, in the
  columns of `observe` of a pair term, in the free-energy file, and in
  `Simulation.tunables.gradient()` and its energy, once the cell had left
  the volume of the start: 1.2e-3 of what the correction adds at a volume
  6% smaller. Only the part `c1/V` follows the volume now. Nothing changes
  at constant volume, and nothing in the forces or the pressure.
- `scripts/paper/check-appendix.sh` without the `mdir` to ask, or with one
  that cannot be run, prints its usage and exits with 2; it ran an empty
  command and reported the whole example of Appendix A as a difference,
  with the status 1 of an appendix that no longer agrees (#209).
- A program that `md-exec-assign-storage` refuses inside a loop that carries
  fields (the loop of the steps) fails with the diagnostic of the pass, a
  `CompileError` in Python; the process aborted after it with "operation
  destroyed but still has uses" when the body of the loop used a value of
  the code before it, such as a neighbor structure (#208).
- A row of `[output] observables`, of the free-energy file, and of the pull
  file at a step where the barostat couples is now of the state of that
  step (#218). With the barostat of Trotter type the step scales the
  positions, and the potentials of these outputs took the cell of before the
  scaling for their minimum image, so the pairs across the cell were at
  distances off by the change of its edge: on the dipeptide in water the
  energy of an observed pair term and `dH/dlambda` of a pair term were off
  by 5e-6 to 1.4e-5 relative at every such row, under NVE and NVT by
  nothing. The energies of the log, terms over tuples, and terms of the
  positions were right. `State.observables` of a Python simulation takes
  the same fix.
- The release tarball carries the notices of the third parties whose code
  it distributes in binary form, in `share/mdir/licenses`: HDF5's
  `COPYING` beside the bundled `libhdf5` (#194), and the licenses of
  pocketfft (in `libmdrt.so`), toml++ (in `mdir`), LLVM (in `mdir`), and
  LLVM's OpenMP runtime (`lib/libomp.so`). The 0.1.0 tarball had only
  MDIR's license and the CUDA EULA. `scripts/release/check-binary.sh`
  fails on a tree without them;
  [packaging/licenses/README.md](packaging/licenses/README.md) gives the
  clause for each component.
- The test suite hands `CUDA_CACHE_PATH` to its tests, as it hands `HOME`:
  set for a suite, it had no effect, so the kernel cache of the CUDA driver
  of the tests stayed in the home directory (#223).
- `Simulation.save_checkpoint` raises `SimulationError` while a read-only
  view of the simulation or a tensor taken from one is alive (a lease,
  D220), as `run` and the other operations that end the activation do. It
  was accepted, and freed the buffers under the view: `view.valid` became
  false and the arrays and tensors of a consumer pointed to freed memory,
  on a device to blocks returned to the runtime's pool. Release the view
  and delete the tensors before a checkpoint (#207).
- The compile cache no longer lists its directory after stores. Every
  compile that stored an entry listed `host/` and `gpu/` and read the
  status of every entry, once per run of the GPU pass and once per host
  object, so its cost grew with what the shared directory held, and GPU
  tests stalled for minutes when the cache was on a network file system
  (#196). The cache now keeps the total of its entries in the file
  `<dir>/size`, updated under a lock, and lists the directory to evict
  only when the total exceeds `MDIR_COMPILE_CACHE_MAX_MB`, when no total
  is known, or once a day (D234). The bound and the
  order of eviction are unchanged; the temporary files of processes that
  died are removed within a day instead of at the next store. See
  [docs/compile-cache.md](docs/compile-cache.md#the-bound).
- The scripts that generate the GPU templates of particle mesh Ewald and of
  the groups of neighbors (`scripts/generate-pme-gpu-template.py`,
  `scripts/generate-groups-gpu-template.py`) emit the guards of #168 that
  the templates held and they did not, so a run of either no longer takes
  them out; `test/Runtime/generated-templates.test` compares what they
  generate with the four templates, byte for byte (#211). No template
  changes.
- A run whose positions blew up (to about 1e190 nm, or to positions that
  are not numbers with PME) or whose barostat blew up the cell ended the
  process on a GPU with an illegal address or an allocation that failed,
  and a Python run on the CPU could crash the same way. A Python simulation
  now raises `SimulationError` and keeps the state from before the part;
  `mdir run` stops with a word. A coupling of the barostat that scales an
  edge of the cell by a factor outside [1/2, 2], or makes it not finite,
  fails the run; the stop of a build says that a position "is not a number
  or is far outside the cell" (D225, #168).

- The builder numbered the values that store the state of a barostat of
  Trotter type with a count of the process instead of one of the build, so
  the text of a program depended on what the process had built before, and
  two builds on two Python threads could emit the same names. A program
  now has the same text, and therefore the same keys in the compile cache
  (D212, D214), whatever the process built before (#152).
- In the deterministic mode a Python simulation and `mdir run` give the same
  state to the bit for any number of steps, in one part or in several, with
  or without a checkpoint at the end (D218, #121). Before,
  they parted at the rounding of the forces after about 20 steps on the
  dipeptide in water (1.5e-3 kJ/mol/nm in mixed precision, 2.2e-10 in
  double). The neighbor structures of the evaluation at the start carried
  into the steps of a simulation and of `mdir run` without checkpoints, while
  `mdir run` with checkpoints built them anew at step 1, so the programs
  rebuilt at other steps. The evaluation at the start is now a segment of
  its own in every program, and the steps build their structures at step 1.
  No control key or file format changes. The trajectory of `mdir run`
  without checkpoints changes at the rounding of its sums, and a fixed
  `rebuild_interval` counts from step 1, one step later than before.
- `mdir.compile` of a minimization (`Integrator.minimize = True`) with
  `Schedule.energy_period = 0` or `Schedule.steps = 0` raises `InputError`
  instead of killing Python with a floating-point exception (#178). The
  program of a minimization loops over the intervals between its energies,
  as `[minimize]` of a control file does, which refuses
  `energy_interval = 0` the same way. `Simulation.minimize(steps)` still
  takes any nonnegative count.
- `mdir run` of a control file with `[minimize] steps = 0` stops at
  reading it with "a minimization takes steps: 'steps' may not be 0"
  instead of dying with a floating-point exception (#180). `steps` of
  `[minimize]` is a positive count; a minimization of no steps has no
  energies to write.

- An `[[energy.pair]]` correction at or near the force field's own
  parameters, such as an NBFIX at zero offset where a fit begins, is no
  longer refused by the correction for the dispersion as a tail that
  "cannot be integrated" or "diverges" (#155). Its terms cancel to their
  rounding, which the quadrature now measures against the size of the terms
  rather than their difference. The tail is 0, and its derivatives in
  `observe` include their tails.

- With a topology, `dispersion_correction = "ENERGY_PRESSURE"` now includes
  the tail beyond the cutoff of every `[[energy.pair]]` term, over the pairs
  that its `groups` select and the topology does not exclude, in the energy,
  the pressure, the λ-state energies, `dH/dλ`, and the derivatives of
  `observe`. Before, it took the topology's own Lennard-Jones alone, so a
  correction term such as an NBFIX written in the control file was missing
  its long-range part (0.80 kcal/mol on 300 particles), and NPT densities
  and pressures were wrong (1.1% in the volume of an NBFIX mixture in NPT).
  Runs whose pair terms decay faster than $r^{-3}$ change in the line
  `dispersion`. With a topology, `[[energy.pair]]` now accepts its
  `dispersion_correction`: `"NONE"` leaves the term out of the correction,
  which it otherwise follows from `[energy]`. A term whose tail diverges,
  such as a Coulomb-like $1/r$, or that depends on the time `t`, is left
  out with the warning `pair_tail_left_out` under the default correction,
  and is an error when the control file gives `dispersion_correction =
  "ENERGY_PRESSURE"`, until the term gives `dispersion_correction = "NONE"`
  (D209, #145).

- The warning `short_thermostat_period` is reported once for a run from a
  topology. It was added to the system twice, so `mdir check`,
  `mdir check --json`, `mdir run`, and a Python model showed it twice (#124).

- A Nose-Hoover chain driven far from equilibrium no longer turns the run
  to NaN. After each action of the chain, a value that is not finite, or a
  part that moved the chain by |s v_j| above 3, stops the run with an error
  before the velocities are scaled. The error names the step, the chain's
  velocities, and the kinetic energy relative to the bath's, and suggests a
  shorter `interval`, a larger `time_constant`, or another thermostat. The
  factorization is unchanged, and runs the guard does not stop move as before
  (D206, #117).

- A persistent Python simulation no longer keeps the memory of each part:
  every `run(n)` call left its device buffers (about 97 MiB on JAC on a GPU)
  or its host buffers (on the CPU) allocated, until a long run ran out of
  memory. What a part allocates is now freed or returned to the runtime's
  pool when it ends (#110).

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
- A run whose particles come from a PDB file now reports the warnings of
  its control file (`unused_parameter` and `constant_expression`, D158) in
  `mdir check`, `mdir check --json`, and `mdir run`, as a run from a
  topology does; they were dropped before (#129).
- The loop over frames of an evaluator (D251) is checked where it is
  assembled from the emitted text (#302): a program whose loop would take
  memory of the stack at every frame, or the cell or the positions of the
  program in place of those of the frame, is refused when it is built,
  with the line. No program of today is refused, and no number, key, or
  call changes. `python-frames-loop*.test` measure the stack of the entry
  per frame to the byte.
- The tests that write outputs under `%t` remove them first (#423). They
  wrote at fixed paths without removing them, so every run of the suite
  in one build tree left one numbered backup more (`#name.dcd.N#`), and
  after 99 runs `mdir` refused to write and the tests failed. 42 tests
  now start from an empty `%t` (`rm -rf %t && mkdir -p %t`, or
  `rm -rf %t` before `split-file` or a script); those that wrote next to
  `%t` (`%t.dcd`, `%t.toml`) write inside it. `test/Lit/test-outputs.test`
  lists any test that runs `mdir run` or a Python interpreter on a path
  under `%t` that no `rm` removes before its first use.
- The build of groups on a device read memory that nothing had written
  when a chunk of fewer than four particles left a group empty (#413): the
  lists read the cells between the frame and the wrapped position at the
  first place of every group, which the boxes wrote only for a place with a
  particle. compute-sanitizer's initcheck reported it on JAC of the Amber
  suite. The boxes now write zero cells there for an empty place, as they
  write its shift; no entry of an empty group used the value, so no result
  changes. `test/Sanitizer/groups-empty-gpu.test` builds three empty
  groups.

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
