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
program (D218, #121). The neighbor structures that it
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
| `UnsupportedError` | NPT with a coupling period of 1 (scaling every step); a second GPU device in one process |
| `SimulationError` | A failure during a run: positions that are not numbers or are far outside the cell at a build of the neighbor structures on a device, a state that is not numbers at the end of a part, a barostat that takes the cell below twice the cutoff, makes it not finite, or scales an edge by a factor outside [1/2, 2] at one coupling (D225); or another operation under way on the same simulation |

After a `SimulationError` raised by a failure, the simulation keeps the
state from before the failed part: the state at the end of the last part
that succeeded, which a copy made there holds
([Resident buffers](#resident-buffers)), or for a part that began an
activation the state it began from. `failed` is true, and `run` refuses
to continue; `state()` still returns that state. On a device a build of the
neighbor structures that finds positions that are not numbers or are far
outside the cell (D107, D176), which ends `mdir run`, reports to the
simulation through a handler of the runtime instead; the part runs to its
end, as it does on the CPU, and its state is checked and discarded there.
The runtime is never left in the middle of its work. The rest of the part
runs on whatever the failure left, so no kernel addresses outside its
buffers for any positions or cell (D225): the loops over
the rows of a matrix skip the places that the build left empty, the points
of PME are defined for coordinates that are not numbers, and every build
takes at most 256 cells along an axis. Failures of the device
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
(D220, [python-dlpack.md](python-dlpack.md)). A writable borrow hands a
consumer the same buffers and takes what it wrote at a commit
(D229).

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
far (D142). A barostat that makes the cell not finite or scales an edge by
a factor outside [1/2, 2] at one coupling fails the part as well, and the
part that has failed runs to its end without any access outside a buffer
(D225, #168). The simulation then takes the copy, or, for a
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
upload. The commit of a writable borrow (D229,
[python-dlpack.md](python-dlpack.md#writable-borrows-d229))
does the same with the state that a consumer wrote. An
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
  (`python-memory*.test`). The memory of the host is read as the bytes in
  use of the allocator (`mallinfo2`, with the thread cache off) and as the
  address space mapped outside the heap, bounded by 0.5 MiB from step 5 to
  step 30 and by a median of 16 KiB per step: the resident memory read +3
  to +4 MiB alone and up to +8.8 MiB beside the other tests of a suite for
  the same 25 simulations, which keep nothing (#228).

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
(D224, #151); `Program.lowered_ir` lowers on its first read
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
(D217,
[compile-cache.md](compile-cache.md#controls-from-python)).

Within a process, the second and later simulations of one `Program` do
not compile: the `Program` keeps the LLVM module and the host object of
the program of segments that its first simulation compiled, and a later
simulation links a copy of its own (D236,
[compile-cache.md](compile-cache.md#reuse-within-a-process)).
On the dipeptide in water, `mdir.Simulation(program)` then takes 0.07 s on
the CPU and 0.13 s on a GPU instead of 5.4 s and 9.4 s;
`compile_stats["program_reused"]` says so. `cache=False` compiles anew.

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

## NPT in a triclinic cell (D244)

Issue [#254](https://github.com/vv137/mdir/issues/254). A Python simulation
refused a barostat in a triclinic cell, which `mdir run` runs (D127). The
tilts are values that the entry takes when an activation begins (D227), and
a simulation kept those of the build as the tilts of its state: after a
part in which the barostat had scaled them, `State.cell.tilt`, a
checkpoint, and the next activation would have had the tilts of before.

Since D238 an activation begins with the tilts of the state of the host,
and the cell of the host is set in one place
([python-dlpack.md](python-dlpack.md#tilts-of-a-triclinic-cell-d238)).
What this adds: the tilts that the barostat reports at each scaling
(`mdrtSetTilt`, D127) become the tilts of the host's state at the end of a
part; a part that fails restores the cell of before with its tilts through
that one place, as a commit that is undone does; and the cell that the
runtime writes into a checkpoint is that of the host from the start, so
that a checkpoint written before the first scaling has the tilts. The
refusal is lifted: `State.cell`, `save_checkpoint` and a
`CheckpointReporter`, the frames of the reporters, and `Borrow.tilt`
follow the barostat. The program, its kernels, and the path of a step are
those of before: the barostat of a triclinic cell is that of `mdir run`,
validated by D127.

**The couplings and the works.** When this was written the Python model
had the isotropic coupling and the default work alone. It has the
semi-isotropic and the anisotropic coupling and the four works since
D[python-barostat] ([below](#the-coupling-and-the-work-of-the-barostat-dpython-barostat)),
in a triclinic cell as in an orthorhombic one, each to the bit of
`mdir run`. With the neighbor
matrix, the default, every image within the reach is held whatever the
barostat does to the cell (D241). With the groups
(`Execution.neighbor_structure`, D245,
[python-model.md](python-model.md#the-groups-and-the-dual-list)) the
runtime fails the part in which the barostat takes the cell below the
pairlist distance over 0.999 (D242), as it stops `mdir run`
(`python-groups-reach-gpu.test`). The stop of a cell below twice the
cutoff is the runtime's, for both front ends.

**Validation against `mdir run`.** `python-triclinic-npt-cli-double.test`,
`-mixed`, and their `-gpu` twins (`Inputs/python_triclinic_npt.py`): the
rhombic dodecahedron of `test/Driver/Inputs/triclinic` with 403 rigid
waters (SETTLE), PME on $28^3$, velocity Verlet at 2 fs, stochastic
velocity rescaling and isotropic stochastic cell rescaling every 10 steps
at 300 K and 1 atm, the velocities drawn by both front ends from one seed,
in the deterministic mode. The input is not equilibrated (its pressure is
$5.5\times10^4$ atm at the start), so the barostat moves the tilts by
$2.0\times10^{-2}$ nm in 200 steps; the comparison is one of the two
front ends, not of the liquid. The two front ends give the same run bit
for bit when their checkpoints are at the same steps: a checkpoint ends a
segment of `mdir run` as it ends the activation of a simulation (D215,
D218), the neighbor structures are built anew after it, and a run with a
checkpoint at step 100 differs from one without by the rounding of the
sums of the forces ($5\times10^{-11}$ nm in the positions at step 200 in
double precision). That is the condition of an orthorhombic cell as well
(`python-deterministic-run-*.test`, `python-checkpoints-*.test`). Each row
below holds on the CPU and on an RTX 3090, in double and in mixed
precision; the differences that are not 0 are those of the CPU in double
precision.

| Quantity | Reference | Difference | Tolerance |
|---|---|---|---|
| Positions, velocities, forces, $a_x, b_y, c_z$, and $b_x, c_x, c_y$ after 200 steps in one part, in 20 parts of 10, and in parts of 30, 70, and 100 | The checkpoint of `mdir run` of 200 steps | 0 | 0 |
| The energy file (21 rows) and the DCD trajectory (20 frames, each with its cell) of those runs | The files of `mdir run` | 0 bytes | 0 |
| The fingerprint of the checkpoint | That of `mdir run` | equal | equal |
| The shape of the cell at step 200: $\lvert b_x\rvert$, $\lvert c_x/a_x - 1/2\rvert$, $\lvert c_y/b_y - 1/2\rvert$ | The rhombic dodecahedron | 0 | $10^{-14}$ |
| `mdir run` to step 100, a Python simulation from its checkpoint to step 200: the state, the cell, the energy file, and the trajectory | `mdir run` of 200 steps with a checkpoint at step 100 | 0 | 0 |
| A Python simulation to step 100 with a `CheckpointReporter`: the cell and the tilts of its checkpoint | The checkpoint of `mdir run` at step 100 | 0 | 0 |
| That checkpoint continued by a second Python simulation, by the one that wrote it, and by `mdir run --continue`: the state, the cell, and the files at step 200 | `mdir run` of 200 steps with a checkpoint at step 100 | 0 | 0 |
| The cell of each of 20 frames in DCD (lengths and cosines in f64) | `State.cell` at the step of the frame | $7\times10^{-16}$ nm | $10^{-12}$ nm |
| The same in XTC (vectors in f32) | The same | $1\times10^{-7}$ nm | $5\times10^{-7}$ nm |
| The same in H5MD (`box/edges`, $(3, 3)$ for each frame, D239), and the positions | The same | 0 | 0 |

**Validation of the state of the host.** `python-triclinic-npt.test` and
its `-gpu` twin (`Inputs/python_dlpack_tilts.py npt`): the same cell with
403 flexible waters at 0.5 fs, in double and in mixed precision.

| Quantity | Reference | Difference | Tolerance |
|---|---|---|---|
| The tilts of `State.cell` after 60 steps, and of `Borrow.tilt` | The start: they move by $3.1\times10^{-3}$ nm; $c_x/a_x$, $c_y/b_y$ stay 1/2 and $b_x$ 0 | shape 0 | $10^{-14}$ |
| The state after three parts of 20 steps | One run of 60 steps | 0 | 0 |
| 20 steps continued from a checkpoint written at step 40, by another simulation of the program | The simulation that wrote it, 20 steps on | 0 | 0 |
| The same | The run that did not stop, which kept its neighbor structures | $8.9\times10^{-16}$ nm in double, $5.6\times10^{-8}$ nm in mixed precision (CPU) | $10^{-12}$, $10^{-6}$ nm |
| A commit of the tilts (0.03, −1.28, −1.29) nm through a writable borrow before the first step, then 40 steps under the barostat, which moves the tilts by $1.8\times10^{-3}$ nm | A simulation compiled from the committed state | 0 | 0 |
| A frame of step 20 committed into the simulation that ran it, at step 40: `Borrow.tilt` before and after the commit; the forces, the potential, the virial, the pressure, and the volume of the evaluation | The tilts of the state and of the frame; a simulation compiled from the frame | 0 | 0 |
| A barostat at $2\times10^5$ bar with a cutoff of 0.88 nm, which takes $c_z$ below twice the cutoff after two or three periods of coupling (the tilts $3.5\times10^{-2}$ nm from the start): the positions, the velocities, and the cell with its tilts that the failed run keeps | A simulation that runs to the step kept and stops | 0 | 0 |
| 20 frames of 200 steps of a Python run at constant pressure ($c_x$ from 1.30091 to 1.30780 nm) put into a second simulation: energies and forces of each | A simulation compiled from the frame | 0 | 0 |

**The frames of a Python run.** `python-triclinic-npt-frames.test` and its
`-gpu` twin: 20 frames of 400 steps of the run of rigid water ($c_x$ from
1.3020 to 1.3303 nm, volumes 12.49 to 13.32 nm³), written by an
`H5MDReporter` and read by the frame evaluator (D240): the energy, its
derivative in the tunables, the virial, and the volume of frames 0, 6, 13,
and 19 equal those of a simulation compiled from the frame to the bit, in
double and in mixed precision; and the same frames in DCD committed one by
one into a second simulation (D238), the potential, the virial, the
volume, and the forces of each equal to those of a simulation compiled
from it. The tests of D238 and D240 took these frames from `mdir run`.

**A cell that the barostat shrinks.** `python-triclinic-reach.test` and its
`-gpu` twin: the system of `triclinic-reach.test` (D241) in a Python
simulation, whose barostat has the default work (`TROTTER`; the Python
model has no key for it, and the control file of that test has
`FIRST_ORDER`), 216 argon atoms in a cell with every tilt on its bound that
stochastic cell rescaling at 20,000 atm takes from 2.2 to 1.92 nm, with a
pairlist distance of 1.1 nm, beyond half of the least of the diagonal at
each of the 79 frames. The potential of each row, which with that work is
of the configuration and the cell of the frame of its step, equals the sum in NumPy
over every image of every pair within the cutoff to $10^{-5}$ kcal/mol (0
frames of 79 differ), on the CPU and on the device.

## The coupling and the work of the barostat (D[python-barostat])

Issue [#275](https://github.com/vv137/mdir/issues/275). `mdir.Ensemble`
had `pressure`, `tau_p`, `compressibility`, and `coupling_period`, and
nothing for the `coupling` and the `work` of `[barostat]`: a Python
simulation ran isotropic stochastic cell rescaling with the default work
alone.

| Python | Control file | Values |
|---|---|---|
| `Ensemble.coupling` | `[barostat] coupling` | `mdir.BarostatCoupling.Isotropic` (the default), `SemiIsotropic` (x and y together, z by its own; D119), `Anisotropic` (each axis by its own; D163c) |
| `Ensemble.work` | `[barostat] work` | `mdir.BarostatWork.Trotter` (the default; D92), `TrotterFirstOrder`, `Exact`, `FirstOrder` (D77) |

`Ensemble.compressibility` is one number, that of every axis and of z,
and there is no surface tension: what a control file with one
`compressibility` and none of `compressibility_z`, `surface_tension`, and
`surfaces` runs. Without a barostat (`EnsembleKind.NVE`, `NVT`) the two
fields are not read, as `tau_p` is not. `Program.plan["barostat"]` is
`{"coupling": ..., "work": ...}` with the two values, or `None` without a
barostat. The fingerprint of a checkpoint (D223) has `[barostat] coupling`
and `[barostat] work`, in the group `coupling`, when the field was set or
is not the default of an absent key: `mdir run --continue` takes the
checkpoint of a Python simulation with the same keys, and refuses another
coupling or work as it does between two control files.

| Case | Where | Message |
|---|---|---|
| `FirstOrder` with `SemiIsotropic` or `Anisotropic` | `mdir.compile`, `InputError` | that of `mdir check`, with `model` for the path: `model: 'work = "FIRST_ORDER"' counts the work from the trace of the virial of the step with twice the internal kinetic energy, which holds for the trace only; with 'coupling = "SEMI_ISOTROPIC"' or "ANISOTROPIC" use "TROTTER", "TROTTER_FIRST_ORDER", or "EXACT"` |
| a value that is not of the enumeration | the assignment, `TypeError` | |
| `Trotter` or `TrotterFirstOrder` with `coupling_period = 1` | `mdir.Simulation`, `UnsupportedError`, as before | `a simulation with a barostat that scales the cell every step (coupling period 1) is not supported yet` |

The program, its kernels, and the path of a step are those of `mdir run`:
the model hands the builder the control structure of the control file. A
part may not end between the two steps that close a period of a work of
Trotter type ([Segments and the phase of the
coupling](#segments-and-the-phase-of-the-coupling)), which holds for
`TrotterFirstOrder` as for `Trotter`; `Exact` and `FirstOrder` close a
period with one step, and with them a Python simulation takes
`coupling_period = 1`, a scaling at every step.

**Validation against `mdir run`** (`python-barostat*.test`,
`Inputs/python_barostat.py`), in the deterministic mode: for each coupling
and each work that it takes, 10 cases (the three couplings with `TROTTER`,
`TROTTER_FIRST_ORDER`, and `EXACT`, and the isotropic one with
`FIRST_ORDER`), in an orthorhombic cell (the dipeptide in 382 waters, PME
on $32^3$, SHAKE and SETTLE) and in a triclinic one (403 rigid waters in
the rhombic dodecahedron, PME on $28^3$), on the CPU and on a GPU, in
double and in mixed precision: 80 runs of 40 steps at 2 fs with a coupling
every 10 steps at 2000 atm, each with a checkpoint at step 20.

| Quantity | Reference | Difference | Tolerance |
|---|---|---|---|
| Positions, velocities, forces, $a_x, b_y, c_z$, and $b_x, c_x, c_y$ after 40 steps in parts of 10 and 30 | The checkpoint of `mdir run` of the same control | 0 in each of the 80 | 0 |
| The energy file (5 rows) and the DCD trajectory (4 frames, each with its cell) | The files of `mdir run` | 0 bytes | 0 |
| The fingerprint of the checkpoint | That of `mdir run` | equal | equal |
| `mdir run` to step 20, a Python simulation from its checkpoint to step 40: the state, the cell, and the files | `mdir run` of 40 steps | 0 | 0 |
| A Python simulation to step 20, `mdir run --continue` from its checkpoint: the same | The same | 0 | 0 |
| The cell of the last frame of the DCD trajectory | `State.cell` | below $10^{-12}$ nm | $10^{-12}$ nm |
| The strains $\ln(L_k/L_k^0)$ of the three axes: isotropic | each other | below $10^{-12}$ | $10^{-12}$ |
| The same, semi-isotropic | x and y equal, z another: in the orthorhombic cell $-9.2\times10^{-4}$, $-9.2\times10^{-4}$, $4.9\times10^{-4}$ | x and y within $10^{-12}$ | $10^{-12}$; z apart by more than $10^{-7}$ |
| The same, anisotropic | all apart: in the orthorhombic cell $-1.09\times10^{-3}$, $3.7\times10^{-4}$, $-2.0\times10^{-4}$ | | apart by more than $10^{-7}$ |
| A coupling period of 1 with `EXACT` (the three couplings) and `FIRST_ORDER` (isotropic), 40 steps in parts of 13 and 27, orthorhombic, CPU double and GPU mixed: the state and the cell | `mdir run` | 0 | 0 |

The oracle is `mdir run`, whose couplings and works are validated against
their own references (D77, D92, D119, D127, D163c): this item adds the way
to them from Python, not physics.

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
