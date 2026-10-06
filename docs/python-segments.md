# Persistent simulations for M2 (D196)

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
compiles what it runs from the program's prepared model, once: one program
of segments, whose entry begins as `mdir run` begins on its first call and
continues from the state the last segment left on the others, as
`mdir run --continue` continues from a checkpoint
(D211, [Compiles](#compiles)). It is built from the
system at its first step, so its constants and neighbor structures are
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

On the dipeptide in water with PME (`test/Driver/python-segments-*.test`,
one file per scenario of `Inputs/python_segments.py`),
1 + 7 + 13 steps against 21, with coupling every 10 steps:

| Precision | Positions (nm) | Velocities (nm/ps) | Forces (kJ/mol/nm) |
|---|---|---|---|
| double, CPU | at most 9e-16 | 7e-13 | 6e-10 |
| double, GPU | at most 9e-16 | 6e-13 | 6e-10 |
| mixed, CPU and GPU | 2e-7 to 6e-7 | 2e-4 to 4e-4 | 0.3 to 0.8 |

The thermostat moves the velocities by 5e-2 nm/ps over these steps, so a
coupling at other steps would show. The double comparison uses
$T_q = c_q \epsilon_{64} \max_i \lvert q_i^{\mathrm{whole}}\rvert$,
with $\epsilon_{64}=2^{-52}$. For the 1168-particle fixture and PME order 4,
$M=(N-1)+4^3=1231$ counts a conservative ceiling of direct neighbors plus
interpolation terms. Forces and velocities use $c_q=4M=4924$: two different
summation orders and a factor-two guard for other reductions and propagation.
Positions use $c_x=2(21)=42$, counting an accumulated update in each trajectory
at each step. These are regression budgets for this fixture, checked against
observed residuals; they are not forward-error bounds for arbitrary nonlinear
trajectories or cancellation. The least observed margins are 29.8, 7.59, and
2.33 for positions, velocities, and forces, respectively.

Mixed precision uses $T_q=3E_q$, where $E_q$ is the maximum difference between
the uninterrupted mixed and double references. Two mixed paths with errors of
size $E_q$ can differ by $2E_q$ by the triangle inequality; the factor 3 adds a
50% guard for variation between their error sizes when the reduction order
changes. The observed difference divided by $E_q$ is at most 1.05, leaving
at least a factor 2.85 of margin. This self-calibrated allowance is not a
universal mixed-precision error bound.

The table selects the least-margin case for each target, precision, and
quantity; other cases have more margin. Margin is tolerance divided by the
observed difference. Each test run prints all three values for every case.

| Target | Precision | Quantity | Case with least margin | Observed difference | Tolerance | Margin |
|---|---|---|---|---:|---:|---:|
| CPU | double | positions (nm) | NVE VelocityVerlet | 8.881784e-16 | 2.655975e-14 | 29.904 |
| CPU | double | velocities (nm/ps) | NVE VelocityVerlet | 6.522560e-13 | 4.952198e-12 | 7.592 |
| CPU | double | forces (kJ/mol/nm) | NVE VelocityVerlet | 5.580318e-10 | 1.302357e-09 | 2.334 |
| CPU | mixed | positions (nm) | NVT VelocityVerlet | 3.064083e-07 | 1.532936e-06 | 5.003 |
| CPU | mixed | velocities (nm/ps) | NVE VelocityVerlet | 2.372057e-04 | 1.488703e-03 | 6.276 |
| CPU | mixed | forces (kJ/mol/nm) | NVT VelocityVerlet | 4.191208e-01 | 1.939384e+00 | 4.627 |
| GPU | double | positions (nm) | NPT VelocityVerlet | 8.881784e-16 | 2.654271e-14 | 29.884 |
| GPU | double | velocities (nm/ps) | NVT Leapfrog | 6.141754e-13 | 4.937995e-12 | 8.040 |
| GPU | double | forces (kJ/mol/nm) | NVT Leapfrog | 5.636593e-10 | 1.314916e-09 | 2.333 |
| GPU | mixed | positions (nm) | NVT VelocityVerlet | 6.081129e-07 | 1.768247e-06 | 2.908 |
| GPU | mixed | velocities (nm/ps) | NVT VelocityVerlet | 4.108488e-04 | 1.420102e-03 | 3.457 |
| GPU | mixed | forces (kJ/mol/nm) | NVT VelocityVerlet | 8.018188e-01 | 2.312715e+00 | 2.884 |

## Errors

| Exception | When |
|---|---|
| `StaleProgramError` | The program's inputs changed after compilation |
| `InputError` | `n < 0`, or a run that ends between the two closing steps of a Trotter period |
| `UnsupportedError` | NPT with a coupling period of 1 (scaling every step); NPT in a triclinic cell; a second GPU device in one process |
| `SimulationError` | A failure during a run: positions that are not numbers at a build of the neighbor structures on a device, a state that is not numbers at the end of a part, a barostat that takes the cell below twice the cutoff; or another operation under way on the same simulation |

After a `SimulationError` raised by a failure, the simulation keeps the
state from before the failed part, `failed` is true, and `run` refuses to
continue; `state()` still returns that state. On a device a build of the
neighbor structures that finds positions that are not numbers (D107),
which ends `mdir run`, reports to the simulation through a handler of the
runtime instead; the part runs to its end, as it does on the CPU, and its
state is checked and discarded there. The runtime is never left in the
middle of its work. Failures of the device
runtime itself (a CUDA error) still end the process, as they do for
`mdir run`.

## Ownership

A simulation owns its compiled programs, its JIT, its buffers, and the
state between runs. The driver keeps the output of the run under way and
the runtime keeps its streams and allocation cache in process-wide state,
so one simulation runs at a time in a process: a run waits for another
simulation's run to end. Simulations created one after another are
independent. The GIL is released while a part runs.

D199 enforces the host object contract after code generation,
including ORC's synthesized entries. The owned memory manager reserves space
for each object together, checks its actual executable sections and relocated
exception-frame descriptions before registering them, and deregisters before
freeing any storage. Creation, execution, and
teardown share the process runtime mutex; independent simulations may be held
simultaneously, but operations using the runtime wait for that mutex. No call
holds it while polling Python. See [JIT ownership](jit-invariants.md) for the
state transitions, rejected layouts, tests, and dependency assumptions.

The original D196 section placement is retained for locality: `.ltext` for
x86-64's large code model, `.text` otherwise. Its diagnostic run observed
52 overlapping registrations before the correction and none afterward,
with all 144 registrations paired with deregistration. D199
replaces reliance on that exercised layout with checks of every final object.

Each call of an entry is a call of its own in the runtime: what the call
allocates (the memory of the host that compiled code takes with `malloc`,
the neighbor structures of the runtime, and the blocks of device memory) is
freed when it returns, the device blocks into the runtime's pool, where the
next part takes the blocks of the same sizes. The memory of a simulation
therefore stays that of one part however many parts it runs (#110);
`mdir run` calls its entry once and is unchanged.

The device is resolved when the first GPU simulation in a process runs:
`Execution.device` is an index among the devices that `CUDA_VISIBLE_DEVICES`
leaves visible. The CUDA context is process-wide, so a later GPU simulation
on another device is refused with `UnsupportedError`.

The runtime libraries are found next to the module: in `lib` of the build
tree, or of the prefix the module is installed under, or in the directory
that `MDIR_RUNTIME_DIR` names.

## Compiles

A simulation compiles one program (D211, #147).
Before, it compiled a program for the first segment at creation and, on the
first part after it, a second program that continued the state, a full
lowering and JIT of a nearly identical program on the path of the run. The
entry of a program of segments now takes `%first_call`, nonzero on the first
call. The work of the start is a branch on it:

- for dynamics, the forces and the energy at the start, the energies and the
  terms of the row of step 0 with the virial, kinetic energy, and the outputs
  of pulls, free energy, and observables at the start, and the half kick
  back of leapfrog;
- for a minimization (D202), the projection of the positions onto the
  constraints and the terms at the start.

A later call takes the forces and the velocities given (for a minimization,
the positions as given), as the continuation program did. The signature is
the union of the two: the first call of dynamics is given the forces as
well, zeros, and does not read them. Everything after the start is one code for both
calls.

The branch is a loop of one iteration on the first call and none on the
others, carrying the values given: the passes that place fields and their
buffers know the loops that carry fields, while `md-exec-assign-precision`
and `md-exec-assign-storage` refuse a field-valued `scf.if`. A program of segments with a barostat that scales the
cell every step, which a simulation refuses already, is refused by the
builder too: the state that its first scaling takes is not an argument.

Two entries that share the steps would not save the compile: `md-inline`
inlines the loop of steps into each, so nothing is shared without calls
that are not inlined and deduplicated kernels, and calls would cut
optimizations between the start and the first step.

Every MLIR context of a lowering in the process, the simulation's and that
of `mdir.compile`, lowers with the threads of one process-wide
`llvm::DefaultThreadPool`, given to it through `MLIRContext::setThreadPool`,
so a simulation keeps no pool of its own. The pool is never destroyed. Its
threads do not exist in a child forked after a lowering; a child that lowers
makes a pool of its own. It has as many threads as the host has cores,
unless the environment variable `MDIR_COMPILE_THREADS` gives a smaller
positive number. Processes that compile side by side take it to share the
cores: the test suite sets it to the cores over `gpu_workers` (8 on a
128-core host), where a suite took 285, 252, and 262 s with 128, 8, and 1
threads to a process (one RTX 3090, a shared host).

In the deterministic mode the Python ala3 example gives the states, energy
files, and trajectory of the two programs to the bit, on the CPU and a GPU,
in mixed and double precision; `python-one-program-gpu.test` checks that no
kernels are loaded after a simulation's creation. At a hundredth of its
steps on one RTX 3090, the example takes 146 s instead of 240 s in mixed
precision and 162 s instead of 251 s in double, with less CPU time; the
runs that compiled the second program take a second or so instead of 20 to
34 s. `mdir.compile` still lowers a program of its own for
`Program.lowered_ir` (#151).

## Not in this item

Drawing velocities at a temperature and positional restraints, which the
four-stage example (#82) needs, followed in a small model PR after this one
(maintainer ruling on PR #86): `InitialState.draw_velocities(system,
temperature, seed)`, and a typed `System.restraints` list mapped to
`[[restraints]]` (D74, D124), in
[python-velocities-restraints.md](python-velocities-restraints.md)
(D198). A state without velocities still starts
at rest; drawing them is explicit.

A program that minimizes is a simulation of its own, which takes
`minimize(steps)` in parts as `run(n)` takes steps (D202,
[python-minimize.md](python-minimize.md)).

The program of segments also takes report intervals (D207,
[python-reporters.md](python-reporters.md)): its second nest runs a number
of intervals that end with a step of energy, and the frame period that the
entry takes decides where a frame is copied.
