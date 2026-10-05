# Persistent simulations for M2 (D[python-segments])

Issue #85 is item 3 of the M2 sequence ([python-m2.md](python-m2.md)): a
simulation that persists across calls, `run(n)` for any number of steps,
status and errors that return to Python, and the ownership of the runtime
and the device. Reporters, checkpoints, DLPack, and packaging follow as
items 4 to 7.

## Interface

```python
program = mdir.compile(system, state, integrator, ensemble, execution, schedule)
sim = mdir.Simulation(program)
taken = sim.run(1000)        # steps taken: 1000 unless a stop was requested
snapshot = sim.state()       # owned host copies, in input order
```

`Simulation(program)` refuses a stale program (`StaleProgramError`) and
compiles what it runs from the program's prepared model: a program for the
first segment, which begins as `mdir run` begins, and on the first `run`
after it a program that continues from the state the last segment left, as
`mdir run --continue` continues from a checkpoint. Later edits of the
compile inputs do not change a simulation; compile again and create a new
one. `Schedule.steps` and `Schedule.energy_period` are not used:
`run(n)` sets the number of steps, and the energies of the log belong to
the reporters (item 4).

`run(n)` advances `n` more steps, `n >= 0`. It returns the number of
steps taken, which is `n` unless `request_stop()` was called during the
run, from any thread; the run then stops at the end of the part under way.
`step` and `time` are absolute: the number of the last step taken and its
time in ps. A KeyboardInterrupt also ends the run at the end of a part, with
`step` recording the steps taken, and is raised again.

`state()` returns a `State` of owned, read-only NumPy copies that survive
later runs and the simulation:

| Attribute | Shape and dtype | Unit |
|---|---|---|
| `step`, `time` | int, float | step, ps |
| `positions` | `(N, 3)` float64 | nm |
| `velocities` | `(N, 3)` float64 | nm/ps, at `time + velocity_offset * timestep` |
| `velocity_offset` | float | 0 for velocity Verlet, -0.5 for leapfrog |
| `forces` | `(N, 3)` float64 | kJ/mol/nm, at `positions`, as the next step takes them |
| `cell` | `Cell` | nm |

Forces are those that the next step begins with: in mixed precision they
were computed in f32. Energies, virials, and pressures are not available
from a simulation yet; they come with the reporters.

## Segments and the phase of the coupling

The entry of a program takes the counts of its loops at run time. With a
coupling period C, a period is C - k plain steps and k closing steps that
couple: k = 1, or k = 2 with the barostat of Trotter type (D92), whose
first closing step gives the strain that its second applies. `run(n)`
splits the steps into calls of the entry:

- from inside a period, the rest of its plain steps and its closing steps;
- whole periods;
- the plain steps of the period in which the run ends.

The periods are counted from the first step of the simulation, and the
random numbers of the thermostat and the barostat are keyed by the
absolute step, so a segmented run couples at the same steps with the same
random numbers as an uninterrupted one. A run may not end between the two
closing steps of a Trotter period: `run(n)` refuses such an `n` with
`InputError` before it takes any step.

A call of the entry is one part. Parts are at most about 0.5 s long: the
length of the next part follows the measured time per step, in whole
periods. Each part begins as a continued run does: the particles are put in
the order of their positions again (unless `Execution.reorder` is off) and
the neighbor structures are built anew. A segmented run therefore sums its
forces in another order than an uninterrupted one wherever the order of
the particles or of the pairs differs, and agrees with it within the
rounding of that order rather than bit for bit.

## Errors

| Exception | When |
|---|---|
| `StaleProgramError` | The program's inputs changed after compilation |
| `InputError` | `n < 0`, or a run that ends between the two closing steps of a Trotter period |
| `UnsupportedError` | NPT with a coupling period of 1 (scaling every step); NPT in a triclinic cell; a second GPU device in one process |
| `SimulationError` | A failure during a run: positions that are not numbers at a build of the neighbor structures, a barostat that takes the cell below twice the cutoff; or another operation under way on the same simulation |

After a `SimulationError` raised by a failure, the simulation keeps the
state from before the failed part, `failed` is true, and `run` refuses to
continue; `state()` still returns that state. Failures of the device
runtime itself (a CUDA error) still end the process, as they do for
`mdir run`.

## Ownership

A simulation owns its compiled programs, its JIT, its buffers, and the
state between runs. The driver keeps the output of the run under way and
the runtime keeps its streams and allocation cache in process-wide state,
so one simulation runs at a time in a process: a run waits for another
simulation's run to end. Simulations created one after another are
independent. The GIL is released while a part runs.

The device is resolved when the first GPU simulation in a process runs:
`Execution.device` is an index among the devices that `CUDA_VISIBLE_DEVICES`
leaves visible. The CUDA context is process-wide, so a later GPU simulation
on another device is refused with `UnsupportedError`.

The runtime libraries are found next to the module: in `lib` of the build
tree, or of the prefix the module is installed under, or in the directory
that `MDIR_RUNTIME_DIR` names.

## Not in this item

Drawing velocities at a temperature and positional restraints, which the
four-stage example (#82) needs, are not part of this item; issue #85 records
where they go.
A state without velocities starts at rest.
