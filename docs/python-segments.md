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
every printed digit, whether the simulation reaches the step in one part or
in several (Section "Segments and the phase of the coupling"). Reporters,
which schedule steps of energy over several reports, frames, and files, are
item 4.

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

A part is the steps that the host asks for at once. Parts are at most
about 0.5 s long: the length of the next part follows the measured time
per step, in whole periods. Every part of a simulation runs in one
activation of the entry ([Resident buffers](#resident-buffers)), which
keeps the order that the start of the run gave the particles, its neighbor
structures, and its buffers: a part continues the uninterrupted run. A run
in parts therefore equals one run to the bit in the deterministic mode,
the cell and the energies included. The particles are put in order and the
neighbor structures built only where an activation begins: at the first
part, and after an update of tunables or an evaluation after the first run,
which begin from the state of the host (D213).

Before D215 each part began as a continued run does, with a
new order and new structures, and a run in parts agreed with one run only
within the rounding of the new order (in double precision, 9e-16 nm for the
positions and 6e-10 kJ/mol/nm for the forces on the fixture below), as
`mdir run` does across its checkpoints.

The evaluation at the start of a run is a segment of its own in every
program (D[front-end-divergence], #121). The neighbor structures that it
builds do not carry into the steps. The steps build their own at step 1 and
rebuild them where the test of displacement asks. Before, the structures of
the start carried into the steps of a simulation and of `mdir run` without
checkpoints. `mdir run` with checkpoints built anew at step 1, the first
step of its first segment between checkpoints. The test of displacement
then compared the positions with those of another step, so the programs
rebuilt at other steps and summed the forces of one configuration in
another order. On the dipeptide with cutoff electrostatics this showed from
step 20 (forces 1.5e-3 kJ/mol/nm in mixed precision and 2.2e-10 in double).
In the deterministic mode, a simulation of N steps, in one part or in
several, now equals `mdir run` of N steps to the bit, with or without a
checkpoint at step N, on the CPU and on a GPU, in mixed and double
precision, with cutoff electrostatics and PME (N = 1 to 50, parts of 20 and
of 7 steps; `python-deterministic-run-*.test`).
`mdir run` with checkpoints before step N still differs at the rounding.
Each of its segments after the first puts the particles in order and builds
the structures anew where it begins, so that a run continued from a
checkpoint is exact (#125).

On the dipeptide in water with PME (`test/Driver/python-segments-*.test`,
one file per scenario of `Inputs/python_segments.py`), parts of 1 + 7 + 13
and of 1×5 + 3×2 + 10 steps against 21, with coupling every 10 steps, at
constant energy, temperature (velocity Verlet and leapfrog), and pressure,
in double and mixed precision, on the CPU and a GPU, give the positions,
velocities, forces, and cell of the run in one part to the bit. The
thermostat moves the velocities by 5e-2 nm/ps over these steps, so a
coupling at other steps would show.

## Errors

| Exception | When |
|---|---|
| `StaleProgramError` | The program's inputs changed after compilation |
| `InputError` | `n < 0`, or a run that ends between the two closing steps of a Trotter period |
| `UnsupportedError` | NPT with a coupling period of 1 (scaling every step); NPT in a triclinic cell; a second GPU device in one process |
| `SimulationError` | A failure during a run: positions that are not numbers at a build of the neighbor structures on a device, a state that is not numbers at the end of a part, a barostat that takes the cell below twice the cutoff; or another operation under way on the same simulation |

After a `SimulationError` raised by a failure, the simulation keeps the
state from before the failed part: the state at the end of the last part
that succeeded, which a copy made there holds
([Resident buffers](#resident-buffers)), or for a part that began an
activation the state it began from. `failed` is true, and `run` refuses
to continue; `state()` still returns that state. On a device a build of the
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

The parts of a simulation run in one activation of its entry
(D215). What the activation allocates (the memory of the
host that compiled code takes with `malloc`, the neighbor structures of the
runtime, and the blocks of device memory) is its own and is held between
runs; the runtimes record it by activation, so that a simulation that
waits between parts keeps its memory while another runs. It is freed when
the activation ends: at the end of the simulation, at a failure, and where
an update of tunables or an evaluation after the first run begins another;
the device blocks return to the runtime's pool, where later activations
take the blocks of the same sizes. The memory of a simulation therefore
stays that of one activation however many parts it runs (#110), and a live
simulation holds it between runs, so that live simulations hold the sum of
theirs. `mdir run` calls its entry once and opens no activation.

The device is resolved when the first GPU simulation in a process runs:
`Execution.device` is an index among the devices that `CUDA_VISIBLE_DEVICES`
leaves visible. The CUDA context is process-wide, so a later GPU simulation
on another device is refused with `UnsupportedError`.

The runtime libraries are found next to the module: in `lib` of the build
tree, or of the prefix the module is installed under, or in the directory
that `MDIR_RUNTIME_DIR` names.

## Resident buffers

Issue #135, `D215`: the buffers of a simulation stay where
the program keeps them, on a device or on the host, from part to part, and
copies of the host are made where they are asked for. It is the base of
views of device buffers through DLPack (#131) and of the frame evaluator of
M2b (#138).

### One activation runs the parts

Before, every part was a call of the entry that built everything anew from
the state of the host: it copied the state and every buffer of parameters
to the device, put the particles in order, renumbered the tuples and built
their incidence structures on the host, built the neighbor structures, and
copied the state back at its end. On JAC (23,558 atoms, mixed precision,
PME on a grid of 64³, one RTX 3090) a part of 10 steps took 10.3 ms, of
which 2.7 ms were steps: 50 MB were copied to the device in every part,
from memory of the host that is not pinned.

Now one activation of the entry runs every part. The program of segments
wraps its loops of steps in a loop over parts, and each iteration begins at
the end of a part, or of the start of the run, with a call of the host:

```mlir
mdrt.host_call @mdrtPartBoundary(%x, %v, %f, %id, %not_numbers, %part_counts) {in_place}
    : (!vec, !vec, !vec, !ids, f64, memref<9xi64>)
```

The host is given the positions, velocities, and forces where they are
(addresses of the device on a GPU), in the order of the program, with the
particle at each place (`%id`) and the number of their components that are
not numbers. It keeps the activation waiting there, on a stack of its own,
until the next part, whose step and counts it writes into `%part_counts`;
the activation then runs the steps of that part. What the start of the run
built stays where it is:

- the state, in the buffers of the steps; the cell and the state of the
  barostat in the memory of the host that the loops read, the bath on the
  host as before;
- the parameters: the fields of the particles, the tables and their f32
  copies, the tables that `md-exec-fold-tables` derives, the fields of
  tuples, and the values of tunables;
- the order of the particles, the renumbered tuples and their incidence
  structures, and the neighbor structures, whose test of validity goes on
  from their last build.

`in_place` hands a function of the host the buffers of the fields where
they are, instead of copies in memory of the host:
`md-exec-assign-storage` passes them through a `memref.memory_space_cast`,
and in the deterministic mode `md-exec-assign-precision` does not give them
copies of their own, so that the host sees the buffers of the steps
themselves. The loops rotate their buffers through the values they carry,
so the addresses can change from one part to the next; a view of them
is taken again after every part, and while it is alive no part runs
(D[python-dlpack], [python-dlpack.md](python-dlpack.md)).

**The entry.** It takes the buffers of the host once, when an activation
begins, and the scalars of the start: the cell, the time step, the step,
`%first_call`, the first length of a minimization's step, and the constant
terms of a barostat with tunables. The counts of the loops, the period of
the frames, and the step of each part come through `%part_counts`. The
program allocates all of its buffers, as before; the simulation borrows
them at the boundary. An entry that takes buffers of the device that it
does not allocate was considered and not taken: it would take out of the
program everything the start of the run builds (the f32 and folded tables,
the incidence structures, which the lowering to the device frees at the end
of their block, and the neighbor structures, a dozen buffers, scalars of
the host, and handles that the runtime grows) across
`md-exec-assign-storage`, both lowerings, and the runtime, and the state
would still rotate through the buffers of the loops, so that a part would
still end with a copy.

**Lifetimes.** An activation never returns. It ends at a boundary when the
simulation ends, fails, or begins another activation: what it allocated,
which the records of the runtimes (`mdrtActivationOpen`, `Enter`, `Leave`,
`Close`, and `mdrtDeviceActivation...`) and of the memory of the host that
its code takes list, is freed, and its stack is unmapped
([JIT ownership](jit-invariants.md#activations-that-outlive-a-call-dresident-buffers)).
`mdrtBeginCall` and `mdrtDeviceEndCall`, which freed what one call made
when another simulation could not run between two calls, are gone.

### Failures: a copy at each boundary

A part that fails leaves the state from before it (D196). After each part
that succeeds, the state at the boundary is copied into memory that the
simulation holds: on a device, on the stream of the kernels, before the
work of the next part and without a wait of the host. A part fails where
the count of components that are not numbers is not 0 (a sum over the
particles of $x - x$ unordered with itself, which an infinity fails too),
where a build of the neighbor structures on a device finds positions that
are not numbers (D107), where a barostat takes the cell below twice the
cutoff, or, without a periodic cell, where the particles have spread too
far (D142). (On a device, positions that are huge or not numbers can still
end the process inside PME or a loop over particles before the part ends,
as on main, #168.) The simulation then takes the copy, or, for a
part that began an activation, the state of the host it began from, and
ends the activation. Two sets of buffers that the parts alternate between
were not taken: the loops already rotate the buffers of the state, and the
copy costs microseconds (below).

### Copies of the host

`state()`, the callbacks of reporters (through `state()`), and the updates
of tunables copy the state from the buffers of the activation where they
are asked for, once per part at most; the energy file and the trajectory
are written inside the parts as before. An update of tunables, and an
evaluation after the first run (`run(0, energy=True)`), copy the state, end
the activation, and begin another from the state of the host with
`%first_call = 2`: the new values are uploaded, the particles put in order,
and the forces evaluated at the state, as a simulation compiled with the new
values from that state does, so that D213's comparison with such a
simulation holds to the bit, and the value version advances with that
upload. A future write of the state from the host (#136) does the same. An
upload in place that kept the order would make an update cheaper, at the
cost of that comparison; it can follow if M2b needs it.

**The CPU** has the same ownership: the activation keeps its buffers of the
host, which are those that the entry was given, and the views are their
addresses.

### Validation

On the dipeptide in water with PME, in the deterministic mode, on the CPU
and a GPU, in double and mixed precision:

- runs in parts against one run, at constant energy, temperature
  (velocity Verlet and leapfrog), and pressure, the energy file and the
  trajectory of reporters every 10 steps over runs of 7, and a
  minimization of 3 + 17 + 20 steps against 40 followed by 21 steps of
  dynamics: equal to the bit (`python-segments-*.test`), and each run in
  one part equal to main's to the bit, energy files and trajectories
  included;
- a part that fails after parts that succeeded keeps the state of the last
  of them, read between the parts or not, to the bit against a simulation
  that stops there; a first part that fails keeps the state it began from
  (`python-resident-failures*.test`);
- after an update of tunables and an evaluation, parts of 12, 5 + 7, and
  1 + 1 + 10 steps give one state to the bit (`python-resident-updates*.test`);
- an activation resumed on other Python threads, with four threads of
  execution on the CPU, gives the state of one thread to the bit
  (`python-resident-threads*.test`);
- the memory of the device or of the host stays flat over 30 runs, 30
  evaluations that each begin an activation, and 30 simulations that end
  (`python-memory*.test`).

### Performance

`run(n)` repeated, in ms per run, against main (cb575dd) on the same
build options; one RTX 3090 (GPU 0, 300 W cap, idle), the host's CPU for
the CPU rows; NVE, velocity Verlet; JAC with PME on a grid of 64³, rigid
bonds to hydrogens and water, 2 fs; the dipeptide with PME, 1 fs. One long
run per cell.

| System, target, precision | Parts of 1 | Parts of 10 | Parts of 100 | One run (ms/step) |
|---|---|---|---|---|
| JAC, GPU, mixed | 8.77 → 0.309 | 10.3 → 2.75 | 34.0 → 27.2 | 0.274 → 0.273 |
| JAC, GPU, double | 16.0 → 8.08 | 88.2 → 80.6 | 815 → 806 | 8.12 → 8.06 |
| dipeptide, GPU, mixed | 0.902 → 0.132 | 1.48 → 0.908 | 9.43 → 9.25 | 0.0885 → 0.0932 |
| dipeptide, GPU, double | 1.15 → 0.393 | 4.30 → 3.59 | 37.2 → 36.0 | 0.371 → 0.362 |
| dipeptide, CPU, mixed | 14.5 → 7.91 | 75.5 → 72.9 | 789 → 752 | 7.82 → 7.42 |
| dipeptide, CPU, double | 21.3 → 12.4 | 138 → 137 | 1504 → 1415 | 13.4 → 14.1 |

A part of JAC in mixed precision now costs 36 µs beyond its steps (0.309
ms for a part of one step against 0.273 ms per step of a long run): the
boundary, the count of components that are not numbers, the copy of the
state on the device (1.4 MB), and Python. Main copied 50 MB to the device
in each part, built the incidence structures on the host, and loaded a
module of cuFFT. Reading the state after each part of 10 steps adds 0.39
ms on JAC (100 runs: 3.13 ms against 2.75), the copy to the host and the
order of the input. Long runs are unchanged: alternated with main, twice
each, one run of 20,000 steps took 0.2754 and 0.2755 ms per step on main
and 0.2721 and 0.2721 on this branch for JAC, and 0.0943 and 0.0955 against
0.0899 and 0.0924 for the dipeptide (the single dipeptide cell of the table
is within that spread).

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
34 s.

`mdir.compile` builds the program and its pipeline but does not lower it
(D[compile-once], #151); `Program.lowered_ir` lowers on its first read
([python-compile.md](python-compile.md)). Before, every stage lowered the
program of `compile`, NVPTX code generation included, for a text that the
simulation does not use: the simulation lowers a program of segments of
its own, built from the same prepared model with an entry of
`%first_call`, whose text differs, so it cannot take that lowering. At a
hundredth of its steps on one RTX 3090 in mixed precision, `mdir.compile`
takes 0.3 s instead of 3.5, 5.4, 9.2, and 8.4 s for the four stages, which
compile in 5.4, 13.0, 21.2, and 20.4 s instead of 8.6, 17.5, 29.6, and
27.5 s; the example takes 62 s instead of 85 s. The states and outputs are
those of before to the bit in the deterministic mode, on the CPU and a
GPU, in mixed and double precision.

The host object of a simulation's program can be kept on disk across
processes by the compile cache (D212,
[compile-cache.md](compile-cache.md)), which `MDIR_COMPILE_CACHE_DIR`
enables; `Simulation.compile_stats` reports the times of the pipeline and
of the engine, and the hits. `Simulation(program, cache=False)` compiles
without it, and `mdir.clear_compile_cache()` empties it
(D[compile-cache-controls],
[compile-cache.md](compile-cache.md#controls-from-python)).

## Evaluations without a step

`run(0, energy=True)` evaluates the state without taking a step
(D213, [python-tunable.md](python-tunable.md)): a call of the
entry whose loops take no steps. Before the first run it is the first call,
the start of the run (the row of step 0 goes to a reporter's energy file,
and leapfrog kicks its velocities back by half a step); after it, the call
takes `%first_call = 2` and evaluates the forces of the state anew, writing
no row, so that `state().forces` and, with velocity Verlet,
`state().energies` are those of the current values of the program. An
update of tunable parameters makes the same call. Only a program with
tunables leaves leapfrog's velocities alone on such a call (its half kick
back is a select on `%first_call`); a program without them refuses leapfrog
after the first run, and leapfrog's energies stay unset, its velocities
being half a step behind the positions. `run(0)` without `energy` does
nothing, as before.

An evaluation after the first run begins a new activation of the entry
from the state ([Resident buffers](#resident-buffers)), whose order of the
particles and neighbor structures are those of that state: an update of the
values during a run, followed by steps, equals to the bit a simulation
compiled with the new values from the same state that evaluates it first
and then takes the same steps (velocity Verlet, deterministic mode).
Before the first run, an evaluation followed by steps equals the run that
takes its steps at once, to the bit, since the start ends at a boundary in
both (before D215, the two calls differed from one in the
last bits, the second call putting the particles in order anew).

## A new stage from a reached state

`InitialState.from_state(state, velocities=True)` makes a new initial state
from the `State` that a simulation reached. It copies the positions, the
cell, and the velocities. With `velocities=False`, or for a state without
velocities, it leaves the velocities out, so that `draw_velocities` can
follow, as after a minimization:

```python
start = mdir.InitialState.from_state(state)            # the next stage
start = mdir.InitialState.from_state(minimized, velocities=False).draw_velocities(
    system, 300.0, seed)
```

A program compiled from it is a new stage: its baths, its step count, and
its random streams start anew, and the first step evaluates the forces. A
continuation that keeps those is a checkpoint's ([python-checkpoints.md](python-checkpoints.md)). The velocities of a
leapfrog state are half a step behind its positions (`velocity_offset =
-0.5`), and an initial state takes velocities at the time of its
positions. `from_state` therefore refuses them and asks for
`velocities=False`. A stage from the state is bit for bit one from the same
state built field by field (`python-initial-state.test`).

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
