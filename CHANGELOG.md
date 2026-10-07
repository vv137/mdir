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

- `Execution.neighbor_capacity` in Python, the `[execution]
  neighbor_capacity` of the control file (D[cell-runtime-constants],
  #162): 0, the default, estimates it, and
  `Program.plan["neighbor_capacity"]` is the capacity that a compile took.

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

- A program no longer holds the values that depend on the state it starts
  from as constants of its text (D[cell-runtime-constants], #162): the
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
  starts of continued stages (D[cell-runtime-constants]). The capacity
  does not change the results; the Amber suite runs at the same rate.
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
