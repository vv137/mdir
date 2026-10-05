# Energy minimization in a Python simulation (D[python-minimize])

Issue #91. A Python simulation (D196) runs dynamics; this decision lets it
run the minimization of `mdir run` as well, so that the first stage of the
standard pipeline (minimization, NVT equilibration, NPT with restraints,
production; #82) runs from Python.

## Interface

```python
integrator = mdir.Integrator()
integrator.minimize = True              # [minimize] in place of [dynamics]
integrator.minimize_step = 0.01         # nm, [minimize] initial_step
schedule = mdir.Schedule()
schedule.steps = 2000                   # [minimize] steps
program = mdir.compile(system, state, integrator, mdir.Ensemble(), execution, schedule)
simulation = mdir.Simulation(program)
taken = simulation.minimize()           # 2000 steps; or minimize(steps)
minimized = simulation.state()
minimized.minimization                  # the row of the log at the last step
```

A program whose `Integrator.minimize` is true is a minimization, as
`[minimize]` in place of `[dynamics]` is in a control file; its ensemble is
NVE, as before (D191). `Simulation(program)` accepts it.
`Simulation.minimize(steps=None)` takes `steps` more steps of the
minimizer, the program's `Schedule.steps` when it is `None`, and returns
the steps taken. Any nonnegative count is accepted: a Python simulation
writes no log, so `steps` need not be a multiple of an energy interval.
`run(n)` on a minimization and `minimize()` on a simulation of dynamics
raise `InputError`.

The minimizer is that of `mdir run`, not a second implementation: the
same program (`emitMinimization`), with steepest descent whose first step
moves no particle farther than `minimize_step`, a step that lowers the
energy accepted and the next one 1.2 times longer (at most 0.1 nm), a step
that does not rejected and the next one 0.2 times as long, the positions
first put on the surface of the constraints (SHAKE, SETTLE) and every
trial put back on it, virtual sites placed, and the restraints of the
system. Like `mdir run`, it takes a fixed number of steps and has no
convergence criterion.

## Parts, stops, and failures

A minimization runs in parts of about `part_seconds`, as `run(n)` does
(D196): between parts `request_stop()` and Ctrl-C are polled, with the GIL
released while a part runs. The program of the parts after the first
continues the last: it takes the positions and the length of the next
step that the last part left, and does not put the positions on the
constraints again. A part that fails (positions that are not numbers)
raises `SimulationError` and leaves the state from before it.

## The state afterwards

`state()` holds the positions in input order (a minimization never
reorders), the velocities as given (zero when the state had none), the
forces at the final positions, and `step` the number of steps taken; `time`
is 0. `energies` is `None`; `minimization` is the row that `mdir run`
writes for the last step: `energy` (kJ/mol, with the constant parts of the
dispersion correction and of the Ewald sum, as the log), `rms_force` and
`max_force` (kJ/mol/nm, without the parts along the constraints),
`max_force_particle` (zero-based, in input order; the log prints it
one-based), and `step_size` (nm), the length of the next step.

## The next stage

A later stage starts from the minimized positions through a new initial
state, as D196 and D198 build one:

```python
nvt_state = mdir.InitialState()
nvt_state.positions = minimized.positions
nvt_state.cell = minimized.cell
nvt_state = nvt_state.draw_velocities(system, 300.0, seed)
```

## Validation

To follow: the final row against `mdir run` on the same input, in several
parts, on CPU and GPU, mixed and double, in deterministic mode; a
continuation into an NVT stage; refusals.
