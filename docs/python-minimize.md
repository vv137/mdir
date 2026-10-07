# Energy minimization in a Python simulation (D202)

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

With a tolerance (D[minimize-tolerance], see [Convergence](#convergence)):

```python
schedule.energy_period = 10             # [output] energy_interval: the checks
taken = simulation.minimize(tolerance=41.84)  # kJ/mol/nm (1 kcal/mol/Å)
simulation.state().minimization["converged"]  # True if it stopped there
```

A program whose `Integrator.minimize` is true is a minimization, as
`[minimize]` in place of `[dynamics]` is in a control file; its ensemble is
NVE, as before (D191). `Simulation(program)` accepts it.
`Simulation.minimize(steps=None)` takes `steps` more steps of the
minimizer, the program's `Schedule.steps` when it is `None`, and returns
the steps taken. Any nonnegative count is accepted: a Python simulation
writes no log, so `steps` need not be a multiple of an energy interval.
The program that `compile` builds still loops over the intervals between
its energies, as `[minimize]` of a control file does, so `compile` refuses
a minimization whose `Schedule.energy_period` or `Schedule.steps` is 0
with `InputError` (#178), as the control file refuses
`energy_interval = 0`.
`run(n)` on a minimization and `minimize()` on a simulation of dynamics
raise `InputError`.

The minimizer is that of `mdir run`, not a second implementation: the
same program (`emitMinimization`), with steepest descent whose first step
moves no particle farther than `minimize_step`, a step that lowers the
energy accepted and the next one 1.2 times longer (at most 0.1 nm), a step
that does not rejected and the next one 0.2 times as long, the positions
first put on the surface of the constraints (SHAKE, SETTLE) and every
trial put back on it, virtual sites placed, and the restraints of the
system. Like `mdir run`, it takes a fixed number of steps unless it is
given a tolerance on the force (below).

## Convergence

`Simulation.minimize(steps=None, tolerance=None)` with a `tolerance` in
kJ/mol/nm (a number, or an OpenMM quantity of force, as D200 converts)
stops at the first check whose largest force is below it, or after
`steps`, whichever comes first, as `[minimize] force_tolerance` stops
`mdir run` (D[minimize-tolerance]). The force is the `max_force` of the
row, the largest $\lVert m_i\mathbf g_i\rVert$ over the particles with
mass, without the parts along the constraints, and the test is strict
(`<`). A check is made where the call begins (with no step if the state is
already converged), every `Schedule.energy_period` steps (the
`energy_interval` of the control file), and where the call ends: the parts
of the call end at those steps, and the host compares the row of the last
part with the tolerance between parts. The program is that without a
tolerance; since a part continues the last to the bit (D215), the
simulation stops at the step where `mdir run` with the same tolerance and
energy interval stops, with the same row. A tolerance must be positive and
finite, and the program's `energy_period` positive.

`minimize()` returns the steps taken, as before. The row of the last step,
`state().minimization`, has `converged`: `True` if its largest force is
below the tolerance of the call that wrote it, `False` if not, and `None`
for a call without one.

GROMACS's `emtol` is likewise a tolerance on the largest force
[[GromacsManual2025]](references.md#gromacsmanual2025). OpenMM 8.6's
`LocalEnergyMinimizer.minimize(tolerance=...)` is on the root mean square
of all force components (its documentation), which is the `rms_force` of
the row over $\sqrt3$: an average over thousands of waters can be below a
tolerance that a strained residue is far above.

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
nvt_state = mdir.InitialState.from_state(minimized, velocities=False)
nvt_state = nvt_state.draw_velocities(system, 300.0, seed)
```

Without `System.restraint_reference`, the restraints of that stage hold
the particles at the positions of the state it is compiled with, here the
minimized ones (D198). To restrain toward the coordinates file, as
`examples/ala3/2-nvt.toml` does, set the reference to the loaded positions.

## Boundaries of parts

A part after the first continues the last in the same activation of the
entry (D215): it keeps the positions, forces, energy, and
length of the next step that the last part left, the order of the
particles, and the neighbor structures, so a minimization in parts takes
the steps of a minimization in one part to the bit in the deterministic
mode (3 + 17 + 20 steps against 40 on the dipeptide, double and mixed,
CPU and GPU). Before D215 a part built its neighbor
structures anew and evaluated the energy and the forces at its first
positions, and a minimization, whose choice of each step depends on a
comparison of two energies, could drift from one with boundaries at other
steps where the energy surface is flat. Within a part the program is that
of `mdir run`.

## Validation

`test/Driver/python-minimize-*.test` (CPU), one file per scenario of
`test/Driver/Inputs/python_minimize.py` (`double`, `mixed`,
`cutoff-refusals`), and their `-gpu` twins run it on the dipeptide in water of
`test/Driver/Inputs/dipeptide` (1168 particles; cutoff 8 Å, pair list 9 Å,
PME; SHAKE and SETTLE; a restraint of 10 kcal/mol/Å² on the heavy atoms of
the peptide; deterministic mode) against `mdir run` with the same control
file, its log written at every step. The energy of the minimization falls
from −2747.3 to −3551.8 kcal/mol in 100 steps.

| Check | Double, CPU | Double, GPU | Mixed, CPU | Mixed, GPU |
|---|---|---|---|---|
| Row at step 100, one part | every printed digit | every printed digit | within 3 E | within 3 E |
| Positions at step 100 against the checkpoint | 2.4e-12 nm | 3.2e-12 nm | 1.2e-4 nm | 1.1e-5 nm |
| Row at step 110 after 10 parts of one step | every printed digit | every printed digit | 7.8e-3 relative | 7.0e-4 relative |
| Row at step 30 after 30 parts of one step | every printed digit | every printed digit | 1.8e-4 relative | 6.2e-4 relative |
| NVT from the minimized state, row at step 10 | every printed digit | every printed digit | within 3 E | within 3 E |

The row is the energy, the RMS and the largest force, its particle, and
the step length, as `%.6f` of kcal/mol and Å in the columns file. The
tolerance of the positions in double is 1e-11 nm: the minimization of
`mdir run` and that of the simulation are programs of loops of another
shape (counts taken by the entry, rather than constants), whose rounding
differs by an ulp from the second step, below every printed digit. In
mixed precision E is the difference of `mdir run` in mixed and in double
at that row (for positions, the largest difference of their checkpoints),
the error of mixed precision itself, as D196 bounds segments: with PME the
model and `mdir run` already differ in mixed precision at the first
evaluation, in dynamics as well (#105, outside this decision). With cutoff
electrostatics and no constraints, mixed precision is free of that
difference: one part of 20 steps gives every printed digit of `mdir run`,
and 20 parts of one step agree within 1.2e-8 (CPU) and 4.6e-8 (GPU) of each
value (tolerance 1e-6), while the energy falls from −5736.3 to
−5931.2 kcal/mol.

The NVT stage starts at the positions and the cell of the minimized state,
with velocities from `draw_velocities` at 300 K and the restraints of the
control file (reference: the coordinates file), as `mdir run` begins anew
at the checkpoint of a minimization. The test also checks the refusals
(`run(n)` on a minimization, `minimize()` on dynamics, a negative count),
that `minimize(0)` takes no step and leaves no row, and that
`minimize()` takes the steps of the schedule. Stops and Ctrl-C share the
loop of parts with `run(n)`, which `python-segments-errors-stops.test`
checks.

The tolerance (D[minimize-tolerance]) is checked by
`test/Driver/python-minimize-tolerance-*.test` and their `-gpu` twins
(`Inputs/python_minimize_tolerance.py`), in double and mixed precision, on
the dipeptide with PME, SHAKE, SETTLE, and the restraint, with rows every
10 steps. The reference is `mdir run` without the key: the criterion
applied in NumPy to its rows gives step 100 in every mode (largest force
60.9 kcal/mol/Å at step 90, at least 25 at the rows before, and 9.6 in
double, 6.9 in mixed on the CPU, 8.4 in mixed on the GPU at step 100) for
a tolerance of 15 kcal/mol/Å. `mdir run` with
`force_tolerance = 15.0` stops there, with the rows of the reference up to
step 100, and its energy file, trajectory, and checkpoint equal those of a
run of 100 steps without the key, byte for byte. The simulation stops at
step 100 with `converged` true and the row of `mdir run` to every printed
digit, in both precisions on the CPU and the GPU. The test also checks the
log of a run that does not converge in 20 steps (1 kcal/mol/Å) and of one
that converges at step 0 (30 kcal/mol/Å), and in Python a call that is
converged at its start (no step), one that does not converge, and one
without a tolerance (`converged` is `None`).
