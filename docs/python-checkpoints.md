# Checkpoints of a Python simulation (D[python-checkpoints])

Issue #132, M2a item 5 ([python-m2.md](python-m2.md), Sections 3–5).
Status: design under review in the draft pull request; the questions put to
the maintainer there are marked below.

A Python simulation writes the checkpoint of `mdir run` (H5MD format 1,
D173) and continues from one, written by either front end. The two ways a
run takes a checkpoint keep D172's distinction: continuing the same run
(`mdir run --continue`) and beginning a new stage from another run's
checkpoint (`[input] checkpoint`).

## Interface

```python
sim = mdir.Simulation(program)
sim.reporters.append(mdir.EnergyReporter("prod.dat", 1000))
sim.reporters.append(mdir.CheckpointReporter("prod.h5", 10_000))
sim.run(50_000)
sim.save_checkpoint("prod.h5")            # at any step, between runs

# Later, in another process: the same run goes on.
sim = mdir.Simulation(program, checkpoint="prod.h5")
sim.reporters.append(mdir.EnergyReporter("prod.dat", 1000))   # appended
sim.run(50_000)

# A new stage from the checkpoint of another run (equilibration with
# restraints, then production without them).
sim = mdir.Simulation(production, checkpoint="npt.h5", stage=True)

mdir.read_checkpoint("prod.h5")           # a read-only view of a checkpoint
```

- `Simulation.save_checkpoint(path)` writes the state after the last run:
  step, time, positions, velocities, the forces the next step begins with,
  the cell, the energy the coupling has taken (`bath`), the fingerprint, and
  the entries below. The file appears under its name only when it is
  complete and on stable storage; the one it replaces stays as
  `path + ".prev"` (D132, D173). It is the writer of `mdir run`, called on
  the host, not through the compiled code. A state before the first run has
  no forces; its checkpoint is refused for `--continue` as `mdir run`'s
  would be, so `save_checkpoint` evaluates the start first (`run(0,
  energy=True)`'s evaluation, D213).
- `CheckpointReporter(file, period)` calls `save_checkpoint(file)` at every
  step that is a multiple of `period`; like a callback, its step ends a part.
  A simulation takes one.
- `Simulation(program, checkpoint=path)` continues the same run, as
  `mdir run --continue` does. It refuses a checkpoint whose physics or
  coupling fingerprint differs from the program's and names each entry with
  its old and new value; another integrator, particle count, or particle
  type is refused as well. A change of the execution (target, precision,
  threads, deterministic mode) is a `UserWarning`, and the bits then follow
  the new mode. The simulation takes the step, the time, the positions,
  velocities and forces, the cell, the bath, and the values of the
  tunables, and its first step takes the forces of the checkpoint, as a
  later part of one simulation does (D215).
- `Simulation(program, checkpoint=path, stage=True)` begins a new stage at
  the checkpoint, as `[input] checkpoint` does: the step and the time
  continue (the random streams are keyed by the absolute step), the bath
  starts at 0, and the forces are taken only if physics and coupling match;
  otherwise a `UserWarning` names what differs and the first step evaluates
  the forces of the new physics. A minimization's checkpoint gives the
  positions and the cell only (as in `mdir run`).
- `read_checkpoint(path)` returns a read-only `Checkpoint` with the step,
  time, positions, velocities, forces, cell, integrator, precision,
  fingerprint as `(group, name, value)` tuples, the tunables, and the
  provenance entries.

### Errors

- A build without HDF5: `UnsupportedError`.
- A file that is not a checkpoint, of a newer format, of a development build
  before 0.1.0, or whose state does not match its hash: `InputError`, with
  the pointer to `.prev` that `mdir run` gives.
- A refused continuation (above): `InputError`. A write that fails:
  `SimulationError`; the simulation is unchanged.
- `checkpoint=` with a program that minimizes, or `save_checkpoint` on a
  simulation that failed: `InputError` and `SimulationError` respectively.

### Reporters across a continuation

As `mdir run --continue` (D129, D130, D149): with `append=True` (the
default) an energy file that exists keeps its rows up to the checkpoint's
step and is appended to; a trajectory whose name the checkpoint records is
cut to the frames the checkpoint counts and appended to. A trajectory of
another name is refused, as `mdir run` refuses it. With `append=False` the
files of the reporters are those of the part, `<name>.partNNNN<ext>`. A
checkpoint written by a Python simulation records the name of its
trajectory reporter's file and its number of frames, so `mdir run
--continue` appends to it in turn. Without a continuation the files are
backed up as before (D207).

## Fingerprint of a Python model

D172's fingerprint lists what defined a run in three groups: physics,
coupling, and execution. A Python model has no control file, so its
fingerprint is built in memory from the model, with the entries that the
control file of the same model writes: the same names (`[energy] cutoff`,
`[[restraints]]`, `[thermostat] interval`, ...), values in the units of the
control file (Å, kcal/mol/Å², atm), and the same canonical text. A setting
is recorded when it was given explicitly (assigned on the Python object),
as D172 records written keys (maintainer ruling 6 of
[python-m2.md](python-m2.md)). The topology files are hashed by their
contents as `mdir run` hashes them (the coordinates file is not one of
them), and so are the masses and the reference of the restraints. A Python
script and a control file that set the same options of the same files have
the same fingerprint, and either continues the other's run.

Settings without a control-file key (custom terms in Python's units,
tunables) have entries of their own: a CLI run never has them, so a
checkpoint with them is a new stage for `mdir run`, never `--continue`.

**Question to the maintainer (needs-decision).** Some Python defaults
differ from what the control file takes when its key is absent:

| Setting | Python default | Control file without the key |
|---|---|---|
| `switch_distance` | 1.0 nm (switch on) | equal to `cutoff` (no switch) |
| `pairlist_distance` (execution group) | 1.35 nm | `cutoff` + 1.5 Å |
| `center_of_mass_interval` under NVT/NPT | 0 (no removal) | the thermostat interval |
| thermostat and barostat methods | implied by `Ensemble.kind` | must be written |

Recording only given settings would let a Python model and a control file
that both leave out `switch_distance` look identical with different
physics. The recommendation is to record a setting when it was given *or*
when the value the Python model resolves differs from the control file's
value for an absent key, and to write the methods that `Ensemble.kind`
implies.

## Additional entries of format 1

Format 1 stays: the new entries are datasets that a reader of release 0.1.0
ignores, outside `state_sha256` (which every reader recomputes), with a hash
of their own that a reader of this release checks when present. No
migration is needed and no changelog entry under Changed.

- `/parameters/mdir/front_end`: `python` (absent: the control file).
- `/parameters/mdir/model_sha256`: SHA-256 of the physics and coupling
  groups of the fingerprint and of the tunable declarations.
- `/parameters/mdir/plan_sha256`: SHA-256 of the canonical text of the
  program's plan (`Program.plan`: target, precision, deterministic, order,
  dtypes, PME grid, entry) and of its semantic IR.
- `/parameters/mdir/tunables`: the declarations as text, one dataset of
  values per tunable, the version and the history (D213).
- `/parameters/mdir/extras_sha256`: SHA-256 of the entries above.

`mdir checkpoint FILE` prints them when present.

## Validation plan

- `mdir run` → Python and Python → `mdir run --continue`: a run of 20 steps
  stopped at 10 against the run of 20, CPU and GPU, mixed and double, NVE,
  NVT and NPT; bitwise where the deterministic mode orders both front ends
  alike (cutoff electrostatics without constraints, D207), within the
  tolerances of [python-segments.md](python-segments.md) otherwise (#121,
  #125).
- Python → Python: bitwise in the deterministic mode, with tunables
  updated before the checkpoint.
- Same run against a changed stage: topology, masses, restraints,
  thermostat and barostat settings, tunables and their values, and the
  fingerprint entries named in the refusal or the warning.
- Reporters: energy rows and frames appended across a continuation, equal
  to those of the uninterrupted run; `append=False` parts.
- Corrupted files (a changed byte of the state, a truncated file, a missing
  `fingerprint`) and the refusals of `mdir run` for the same files. A build
  without HDF5 cannot be made on this machine's recipe; the typed error is
  checked through a test hook of the bindings.
