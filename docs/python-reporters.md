# Reporters of a Python simulation (D207)

Issue #109, M2a item 4 of [python.md](python.md). A persistent
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
- `H5MDReporter(path, period, positions="f64", velocities=False, forces=False, strings="fixed")`
  (D239, [python-h5md.md](python-h5md.md)) writes the frames of
  the state without loss in H5MD, in place of a `TrajectoryReporter`.
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
- A simulation takes one `EnergyReporter` and one `TrajectoryReporter` or
  `H5MDReporter` (the
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

`ObservablesReporter(file, period)` (D232,
[python-observe.md](python-observe.md)) writes the file of `[output]
observables` (D189) for a program whose terms observe, with a period of its
own: it is a built-in reporter like the two above (one per simulation, the
row of step 0, backups, appending, the rule of the Trotter barostat), its
period enters the greatest common divisor R, and the host writes its rows
where they are due. A callback's state has the same values in
`State.observables`.

Checkpoints (D223, [python-checkpoints.md](python-checkpoints.md))
add `CheckpointReporter(file, period)`, and the files of these reporters
continue across a checkpoint as those of `mdir run --continue` do. The log of `mdir run` is not a reporter: the energy file holds its
numbers.

## Performance: reports inside a call

A call of the entry was a part of D196, which built its buffers and
neighbor structures anew (about 1% of a step on JAC with parts of 0.5 s);
since D215 a part continues the activation of the entry
that the last one left, and a boundary costs a copy of the state on the
device and a reduction ([python-segments.md](python-segments.md#resident-buffers)).
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

### Asynchronous writers (#113, D400)

The frames of the reporters are written as those of `mdir run`: the call of
each frame stages copies of the positions and the numbers of the particles
(on a GPU, on a stream of copies of its own, into pinned memory), and one
thread writes them while the steps go on ([driver.md](driver.md),
Section 2.8). The end of each part (`mdrtPartBoundary`) waits for the frames
of the part, so a part returns to Python with its frames in the file, as
before. That is the one wait: the simulation gets control back only at the
end of a part, so the changes of the reporters, the cell set from Python, a
checkpoint, a frame written from the state, and the end of the simulation
find nothing queued. A failed write of a frame raises its error from the
part, which keeps the state of before it.

## Determinism

In the deterministic mode, where D201 guarantees that a step of energy
moves the particles as a plain step (cutoff electrostatics, no constraints,
#102), the reporters' schedule does not change the state: a run with
reporters equals one without, to the bit. With constraints or PME on a GPU
(#102 open) the schedule may change the last bits, as `energy_interval`
does in `mdir run`.

## Validation

`test/Driver/python-reporters.test` (CPU) and `python-reporters-gpu.test`
run `test/Driver/Inputs/python_reporters.py` on the dipeptide in water
(1168 particles, cutoff electrostatics, deterministic mode) against
`mdir run` with the same `energy_interval` (10) and `trajectory_interval`
(50), 100 steps in one `run` call, which takes one part of ten report
intervals:

| | Double, CPU | Double, GPU | Mixed, CPU | Mixed, GPU |
|---|---|---|---|---|
| NVE, energy file and DCD/XTC | byte for byte | byte for byte | byte for byte | byte for byte |
| NVT (V-rescale every 10 steps), energy file and DCD/XTC | byte for byte | byte for byte | see below | see below |

In NVT and mixed precision the program of segments and that of `mdir run`
round differently in single precision (loops of another shape; #105), and
the trajectories part. The first row is equal to the byte; the kinetic
energy, the temperature, and the DCD frames agree within 3 E, three times
the difference of `mdir run` in mixed and in double, with a floor of
$10^{-5}$ of each value (and $10^{-5}$ Å for the frames); a row of the
interval before is rejected at every row. The potential, the total, the
virial, the pressure, and the conserved energy are not compared after the
first row (#992): under the plain cutoff of the Coulomb energy, a pair of
charges within the cutoff in one trajectory and beyond it in the other
moves the potential by $332.06\,q_iq_j/r_c$, 7.218 kcal/mol for two
hydrogens of TIP3P at $r_c = 8$ Å and 14.44 for an oxygen and a hydrogen.
About $N^2/2\cdot4\pi r_c^2/V = 2.7\times10^4$ pairs lie within 1 Å of the
cutoff in this cell, so once the trajectories are some $4\times10^{-5}$ Å
apart one such pair is expected, and the rows then differ by whole steps
of this size, which no bound from the rounding holds. Measured: on a V100
the rows at step 60 differ by 7.218 kcal/mol between the two programs,
against $1.6\times10^{-3}$ between `mdir run` in mixed and in double; on
an RTX 4090 by 14.47 kcal/mol at step 100, within 3 E there only because
`mdir run` in mixed and in double were 16.2 kcal/mol apart by such steps
too. The kinetic energy and the temperature do not jump at such a crossing, but their gap grows by kicks: the pair differs in force by $332.06\,q_iq_j/r_c^2$, 0.90 kcal/mol/Å for two hydrogens, while it is on opposite sides, and 3 E holds as long as the baseline takes kicks of the same kind. Measured at the worst row (step 80 of NVT): the kinetic energy at 0.75 of its bound on a V100, 0.65 on an RTX 4090, and 0.01 on the CPU; the frames at 0.18, 0.17, and below 0.001. The double-precision rows check every column, byte for byte.

With SHAKE and
SETTLE, where the rows take the optimal temperature from the half steps
(D203), the rows equal those of `mdir run` in double precision, with
energies every 10 steps and frames every 50, and with energies every 20
and frames every 10 (steps of energy that are not rows, which keep no
temperatures of the solvent, as in `mdir run`). The test
also checks the schedule (callbacks of periods 7 and 11 over 25 steps at
steps 7, 11, 14, 21, 22, and 28 after three more steps; energy rows at 0,
7, 14, 21, 28), a callback error that ends the run at its step, the
state with and without reporters equal to the bit in the deterministic
mode (NVT, cutoff, no constraints), backups, and refusals.

### Performance

JAC (23,558 atoms, PME 64³, SHAKE and SETTLE, NVE, mixed precision) on
one RTX 3090 (GPU 0, alone), ms per step over 5,000 steps after 5,000.
When this was measured the Python model had no pruned lists and no
groups of neighbors (it has since D245,
[python-model.md](python-model.md#the-groups-and-the-dual-list), with
which the two front ends run at one rate), so its rate without reporters
differs from that of `mdir run`'s control file; the overheads are the
comparison:

| Outputs | Python | Overhead | `mdir run` | Overhead |
|---|---|---|---|---|
| None | 0.289 | | 0.215 | |
| Energies every 100 steps | 0.292 | +1.1% | 0.210 | within noise |
| Energies every 1000 steps | 0.291 | +0.8% | 0.207 | within noise |
| DCD every 1000 steps | 0.293 | +1.2% | 0.212 | within noise |
| Python callback every 1000 steps | 0.291 | +0.8% | | |

The resolution of `mdir run`'s timing is 1 µs per step (its run time is
printed to 0.01 s). The callback's cost is a part boundary and a copy of
the state: about 2 ms each here. It needs the memory of a part to be
reused (#110, PR #115); without that fix, every part allocated its device
memory anew and a callback every 1000 steps cost +46%.
