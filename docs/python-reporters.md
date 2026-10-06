# Reporters of a Python simulation (D[python-reporters])

Issue #109, M2a item 4 of [python-m2.md](python-m2.md). A persistent
simulation (D196) writes the outputs of `mdir run` (the energy file and the
trajectory) and calls Python functions at chosen periods, at the cost of
`mdir run`'s own outputs. This document is the design of the draft; the
measurements follow the implementation.

## Interface

```python
sim = mdir.Simulation(program)
sim.reporters.append(mdir.EnergyReporter("prod.dat", period=1000))
sim.reporters.append(mdir.TrajectoryReporter("prod.dcd", period=5000))  # or .xtc
sim.reporters.append(mdir.CallbackReporter(lambda s, state: ..., period=1000))
sim.run(500_000)
sim.close_reporters()          # or the simulation's end; files are complete then
```

- `EnergyReporter(path, period)` writes the columns file of `[output]
  energy` (D149): the same columns, units (kcal/mol, K, bar, Å³), and
  `%.6f` digits.
- `TrajectoryReporter(path, period, format=None)` writes DCD or XTC (from the
  extension unless `format` is given), the frames of `[output] trajectory`:
  positions in input order, the cell, the same headers.
- `CallbackReporter(function, period)` calls `function(simulation, state)`
  with the `state()` of the due step (read-only NumPy copies, D193), with
  the GIL held. An exception it raises ends `run(n)` there and propagates;
  the steps taken so far count (`simulation.step`).
- A report is due at step `s` when `s % period == 0` and `s` is after the
  step the reporter was added at; the step at which `run(n)` ends gets a
  report only if it is due (OpenMM's scheduling model). Reporters due at one
  step share one step of energy and one copy.
- Files are opened at the first `run` after the reporter is added. An
  existing file is backed up as `mdir run` backs up its outputs (D149);
  rows and frames are appended across `run` calls; `close_reporters()` (or
  the end of the simulation) closes the files and empties the list. No
  control-file key changes.
- A simulation takes one `EnergyReporter` and one `TrajectoryReporter` (the
  outputs of `mdir run` it has one of each); callbacks are any number.
- The energy file of a reporter present at the simulation's first run holds
  the row of step 0, as the file of `mdir run` does; the callbacks follow
  OpenMM and are not called at the start.
- With the barostat of Trotter type, a built-in reporter's period must be a
  multiple of the coupling period: a step of energy cannot fall between the
  two steps that close a period (D92).
- A part that fails (D196) may have written rows and frames before its
  failure; the state returns to the step before the part, the files do not.
- A minimization takes no reporters.

Checkpoints are M2a item 5 (`python-checkpoints`) and are not reporters
here. The log of `mdir run` is not a reporter: the energy file holds its
numbers.

## Performance: reports inside a call

A call of the entry is a part of D196: each part builds its buffers and
neighbor structures anew (about 1% of a step on JAC with parts of 0.5 s).
Reports must not cut parts:

1. **The times are arguments of the entry.** The program of segments gains
   a nest of report intervals with run-time counts, the loop structure that
   `mdir run` compiles with constants for its `energy_interval`: K intervals
   of R steps, each ending with the step of energy that `mdir run` takes at
   a row of its log (D196, D201: in the deterministic mode it moves the
   particles as any other step). R is the greatest common divisor of the
   periods of the built-in reporters; with a thermostat or barostat, R must
   be a multiple of the coupling period (as `energy_interval` must), or the
   reports fall back to ending parts at their steps (correct, slower). A
   run that does not begin at a multiple of R first takes the steps up to
   the next report in a part of its own (the existing D196 nest), and its
   last steps after the last report likewise. A step of energy at a
   multiple of R where no reporter is due (periods 6 and 10 give R = 2)
   costs an evaluation of the energy and writes nothing.
2. **What is due is decided at run time.** At each step of energy the row
   goes to the energy file if an `EnergyReporter` is due (the host decides,
   no copy beyond the scalars), and the positions are copied and written as
   a frame only where a `TrajectoryReporter` is due: the frame period is an
   argument of the entry and guards the copy (a loop of one iteration or none, which the passes that
   place the buffers carry). Reporters due at
   one step share the step of energy and the copy.
3. **The writers are those of `mdir run`.** Rows and frames go through the
   same host functions (`mdrtWriteEnergies`, `mdrtWriteFrame`) and writers
   (`ColumnFile`, `TrajectoryWriter`) as the CLI's, so the files are equal
   and the cost is `mdir run`'s.
4. **Python callbacks are the slow path.** A callback's due step ends the
   part (`run(n, energy=True)` at that step), the GIL is taken, `state()`
   copies everything, and the next part builds its structures anew. The
   cost is that of a part boundary plus the copy plus the function, measured
   below.

### Asynchronous writers (#113)

A frame copies the positions to the host and writes the file while the
device waits, in `mdir run` as here. Issue #113 proposes copies to pinned
buffers on a stream of their own and a writer thread with a bounded ring.
Since the reporters write through the CLI's own host functions, making those
asynchronous changes both front ends at once and needs nothing more from
this design. **Proposal: #113 follows this PR**, on the shared path: it
changes what `mdir run` writes at every frame and must show byte-identical
files, continuation, signals, and checkpoints (D129–D132, D149), a
validation of its own; folding it in would double this PR. If the
maintainer prefers, the order can be reversed.

## Determinism

In the deterministic mode, where D201 guarantees that a step of energy
moves the particles as a plain step (cutoff electrostatics, no constraints,
#102), the reporters' schedule does not change the state: a run with
reporters equals one without, to the bit. With constraints or PME on a GPU
(#102 open) the schedule may change the last bits, as `energy_interval`
does in `mdir run`.

## Validation (planned)

- Periods 7 and 11 over 25 steps: due steps, counts, shared reports, the
  final remainder; repeated runs; reporters added between runs.
- Files equal to those of `mdir run` with the same `energy_interval` and
  `trajectory_interval`: rows to the printed digits, frames to the bit;
  CPU and GPU, mixed and double; backups; append across runs.
- Callback errors, copy independence, GIL release while a part runs.
- Deterministic mode: positions with and without reporters, to the bit,
  where D201 holds.
- JAC on GPU 0, ms/step against `mdir run` with the same outputs: no
  reporters; energies every 100 and 1000 steps; DCD every 1000 steps; a
  callback every 1000 steps.
