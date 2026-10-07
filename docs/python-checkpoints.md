# Checkpoints of a Python simulation (D223)

Issue #132, M2a item 5 ([python-m2.md](python-m2.md), Sections 3–5).
Status: implemented (PR #186). The maintainer chose option 1 of the
question below on PR #186.

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
  the host, not through the compiled code. A step begins with forces, so a
  simulation that has not run evaluates its start first (`run(0,
  energy=True)`'s evaluation, D213). The next part begins from the state
  written, with new neighbor structures, as `mdir run` builds them anew at
  each checkpoint: a simulation that goes on after `save_checkpoint` and
  one that continues the file take the same steps to the bit in the
  deterministic mode. This ends the activation of the entry (D215), which
  costs a part boundary and a copy of the state per checkpoint.
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
  otherwise a `UserWarning` names what differs and the first part begins
  with an evaluation of the forces of the new physics (`%first_call = 2`,
  as after an update of tunables, D213). Leapfrog's velocities are half a
  step behind already, so that evaluation takes no half kick back; the
  program of segments now selects the kick on `%first_call` for every
  leapfrog program, not only for one with tunables. Unlike `mdir run` from
  `[input] checkpoint`, the evaluation projects the positions onto the
  constraints once more, which they satisfy already. A minimization's
  checkpoint, or a minimizing program, takes the positions and the cell
  only and begins at step 0 (as in `mdir run`).
- The time continues from that of the checkpoint, whose run may have had
  another time step; the step keys the random numbers (A13), so a stage at
  the step of the checkpoint does not repeat the numbers of the run before
  it.
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

`Execution.neighbor_capacity`, when given and not 0, is the entry
`[execution] neighbor_capacity` that the control file's key writes
(D[cell-runtime-constants]). A run that gives the same capacity to both
front ends continues without a warning; a capacity in one of them only is
a change of the execution, with the warning of one. The estimate, 0 or
the key absent, has no entry.

Settings without a control-file key (custom terms in Python's units,
tunables) have entries of their own: a CLI run never has them, so a
checkpoint with them is a new stage for `mdir run`, never `--continue`.

**Decided (option 1, maintainer on PR #186).** Some Python defaults
differ from what the control file takes when its key is absent:

| Setting | Python default | Control file without the key |
|---|---|---|
| `switch_distance` | 1.0 nm (switch on) | equal to `cutoff` (no switch) |
| `pairlist_distance` (execution group) | 1.35 nm | `cutoff` + 1.5 Å |
| `center_of_mass_interval` under NVT/NPT | 0 (no removal) | the thermostat interval |
| thermostat and barostat methods | implied by `Ensemble.kind` | must be written |

Recording only given settings would let a Python model and a control file
that both leave out `switch_distance` look identical with different
physics. A setting is therefore recorded when it was given *or* when the
value the Python model resolves differs from the control file's value for
an absent key, and the methods that `Ensemble.kind` implies are written.
The truncation is written as the control file writes it: a switch by its
`switch_distance` (no entry when it equals the cutoff, which is no
switch), the other modifiers by `lennard_jones_modifier`, and none by no
entry. A restraint writes `selection` and `force_constant` (kcal/mol/Å²),
and `reference_scaling` only when it is `"ALL"`. The code is
`lib/Driver/ModelFingerprint.cpp`.

## Additional entries of format 1

Format 1 stays: the new entries are datasets that a reader of release 0.1.0
ignores, outside `state_sha256` (which every reader recomputes), with a hash
of their own that a reader of this release checks when present. No
migration is needed and no changelog entry under Changed.

- `/parameters/mdir/front_end`: `python` (absent: the control file).
- `/parameters/mdir/model_sha256`: SHA-256 of the physics and coupling
  groups of the fingerprint and of the tunable declarations.
- `/parameters/mdir/plan_sha256`: SHA-256 of the canonical text of the
  program's plan (target, precision, deterministic, order, dtypes, PME
  grid, entry) and of its semantic IR.

They are provenance: a continuation decides by the fingerprint, which a
reader of release 0.1.0 compares as well.
- `/parameters/mdir/tunables`: the declarations as text, one dataset of
  values per tunable, the version and the history (D213).
- `/parameters/mdir/extras_sha256`: SHA-256 of the entries above.

`mdir checkpoint FILE` prints them when present.

## Evidence

`test/Driver/python-checkpoints-*.test` (`Inputs/python_checkpoints.py`)
on the dipeptide in water (1168 particles), deterministic mode, on the CPU
and on one RTX 3090, in double and mixed precision. Every comparison below
is of the bits of the positions, velocities, and forces at step 20 (zero
differing values in each of the eight combinations of target, precision,
and direction):

| Case | Reference | Result |
|---|---|---|
| `mdir run` stops at step 10, Python continues to 20 (NVE, NVT; NPT with PME, SHAKE and SETTLE) | `mdir run` of 20 steps | equal to the bit; energy file and DCD byte for byte; same fingerprint and bath |
| Python writes step 10, `mdir run --continue` to 20 (same three) | a Python continuation of the same file, and `mdir run` of 20 steps | equal to the bit; files byte for byte |
| Python with tunable charges updated at step 10, checkpoint at 15, continued | the same simulation going on after `save_checkpoint` | equal to the bit; values, version 1, history `[(0, 0), (10, 1)]`, energy rows with `tunables_version` |
| Stages of other physics from step 10 (restraints added, 310 K, NPT, and leapfrog at 310 K) | `mdir run` from `[input] checkpoint` | equal to the bit |
| A stage of the same physics | `mdir run` of 20 steps | equal to the bit |

Before #121 was fixed (PR #173) the Python and CLI programs built their
neighbor structures at different steps. With that difference, Python ->
`mdir run` and the stages agreed within $2.2\times10^{-16}$ nm in double
precision, and the continuation in each direction from one file was
already exact. A simulation that ran on after `save_checkpoint` without
starting a new activation differed from the continued file by
$1.1\times10^{-16}$ nm. This is why the next part now begins from the
state written.

Refusals of the same run name each entry: `[[restraints]]` and "the
reference of the restraints", `[ensemble] temperature`, `[ensemble]
ensemble` and `[barostat] method`, `[energy] electrostatics`, and "the
masses" and "the files of the topology" for a topology whose first mass was
changed. Another integrator is refused in both modes. Another precision is
a warning naming `[execution] precision`. `mdir run --continue` refuses a
Python checkpoint with tunables (`[python] tunables`) and begins a stage
from it.

Corrupted files are refused by the bindings (with the pointer to `.prev`),
by `read_checkpoint`, and by `mdir checkpoint` (exit status 2). The cases
are a flipped bit in the positions (the state hash), a changed character
of `model_sha256` (the hash of the additional entries), a truncated file,
and a file that is not HDF5. The error without HDF5
(`python-checkpoints-no-hdf5.test`) runs only in a build without HDF5,
which the standard build does not make.

The GPU scenarios take about 10 minutes together on one RTX 3090, most
of it compiling the programs.
