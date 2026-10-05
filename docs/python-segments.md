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
sim.run(10, energy=True)     # the last of these steps is a step of energy
snapshot = sim.state()       # owned host copies, in input order
snapshot.energies["potential"]
```

`Simulation(program)` refuses a stale program (`StaleProgramError`) and
compiles what it runs from the program's prepared model: a program for the
first segment, which begins as `mdir run` begins, and on the first `run`
after it a program that continues from the state the last segment left, as
`mdir run --continue` continues from a checkpoint. Both are built from the
system at its first step, so their constants and neighbor structures are
the same whatever state the run reaches. Later edits of the
compile inputs do not change a simulation; compile again and create a new
one. `Schedule.steps` and `Schedule.energy_period` are not used:
`run(n)` sets the number of steps.

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
were computed in f32.

## Steps of energy

What a report can ask for is compiled; when reports happen is chosen at
run time (maintainer ruling on PR #86). The programs of a simulation keep
the step of energy that `mdir run` takes at a row of its log, and the entry
takes, besides the counts of its loops, whether the last step of the call
is such a step. `run(n, energy=True)` makes the last of its `n` steps a step
of energy, which evaluates the energy and the virial with that step's
forces. A report therefore costs no evaluation of its own.

`state().energies` then holds a dict of the row that `mdir run` writes at
that step, in MD units: `potential`, `kinetic`, `total` (their sum),
`conserved` (with the energy the coupling has taken), `temperature` (K),
`virial` (kJ/mol), `pressure` (bar), and `volume` (nm^3). It is `None` if the
last run did not end with a step of energy, or was stopped before its end.
A run that ends where a period of coupling closes takes the step of energy
that closes the period, with its coupling, as `mdir run` does; a run that
ends inside a period takes a plain step of energy.

These energies are those of the rows of `mdir run` at the same step, to
every printed digit, when the simulation reaches the step in one part; after
a boundary they agree within the rounding of the new order (Section
"Segments and the phase of the coupling"). Reporters, which schedule steps
of energy over several reports, frames, and files, are item 4.

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
rounding of that order rather than bit for bit, as `mdir run` does across
its checkpoints. Repeated runs of one partition are identical bit for bit
in the deterministic mode.

On the dipeptide in water with PME (`test/Driver/python-segments.test`),
1 + 7 + 13 steps against 21, with coupling every 10 steps:

| Precision | Positions (nm) | Velocities (nm/ps) | Forces (kJ/mol/nm) |
|---|---|---|---|
| double, CPU | at most 9e-16 | 7e-13 | 6e-10 |
| double, GPU | at most 9e-16 | 6e-13 | 6e-10 |
| mixed, CPU and GPU | 2e-7 to 6e-7 | 2e-4 to 4e-4 | 0.3 to 0.8 |

The thermostat moves the velocities by 5e-2 nm/ps over these steps, so a
coupling at other steps would show. In mixed precision the difference is
of the size of the error of mixed precision itself (the difference of a
mixed run from a double one): the forces are rounded to f32 after a sum in
another order.

## Errors

| Exception | When |
|---|---|
| `StaleProgramError` | The program's inputs changed after compilation |
| `InputError` | `n < 0`, or a run that ends between the two closing steps of a Trotter period |
| `UnsupportedError` | A minimization; NPT with a coupling period of 1 (scaling every step); NPT in a triclinic cell; a second GPU device in one process |
| `SimulationError` | A failure during a run: positions that are not numbers at a build of the neighbor structures on a device, a state that is not numbers at the end of a part, a barostat that takes the cell below twice the cutoff; or another operation under way on the same simulation |

After a `SimulationError` raised by a failure, the simulation keeps the
state from before the failed part, `failed` is true, and `run` refuses to
continue; `state()` still returns that state. On a device the runtime
stops the part where a build of the neighbor structures finds positions
that are not numbers (D107), which ends `mdir run`; on the CPU the part
runs to its end and its state is checked there. Failures of the device
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
four-stage example (#82) needs, follow in a small model PR after this one
(maintainer ruling on PR #86): `InitialState.draw_velocities(temperature,
seed)`, and a typed `System.restraints` list mapped to `[[restraints]]`
(D74, D124). Until then a state without velocities starts at rest.
