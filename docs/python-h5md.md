# Frames without loss: the H5MD trajectory (D[h5md-reporter])

Issue #251. Status: implemented. What was decided in the design and what
the maintainer decided on the pull request is listed under
[Decisions](#decisions).

The frame evaluator of M2b (#249) evaluates the potential at stored
frames. The frames have to be those the
dynamics visited, to rounding: positions rounded to $10^{-3}$ nm, the
precision of XTC, raise the energy of a frame of the dipeptide by
$48 \pm 6$ kJ/mol, and DCD holds f32 in Å, which is not the f32 of the
state in nm. MDIR gains a third trajectory format that holds a frame as the
run has it: H5MD [[deBuyl2014]](references.md#debuyl2014), version 1.1 of
the specification, the layout that the checkpoints already use (D173).

## Interface

### Python

```python
sim = mdir.Simulation(program)
sim.reporters.append(mdir.H5MDReporter("prod.h5md", period=1000))
sim.run(1_000_000)
sim.close_reporters()

frames = mdir.read_h5md("prod.h5md")      # nothing but the steps is read
len(frames); frames.steps; frames.times
frame = frames[10]                         # one frame is read
frame.step, frame.time, frame.positions, frame.cell
for positions, cell in frames:             # one frame at a time
    ...
```

`H5MDReporter(file, period, positions="f64", velocities=False, forces=False, strings="fixed")`:

| Argument | Meaning |
|---|---|
| `file` | The name of the file; any extension. |
| `period` | Steps between frames; the schedule of the other reporters (D207): a frame at every step that is a multiple of `period` after the step the reporter was added at. |
| `positions` | `"f64"` (the default) or `"f32"`: the type of the positions in the file, and of the velocities and the forces if they are written. The positions of the state are f64 in the mixed and the double mode, so `"f64"` rounds nothing; `"f32"` holds the state rounded to the nearest f32, in half the space. |
| `velocities`, `forces` | Whether the frames hold them: those of `state()` at the step of the frame. |
| `strings` | `"fixed"` (the default) or `"variable"`: the form of the `unit` attributes, strings of fixed length as the units module of H5MD prescribes, or of variable length ([Strings](#strings)). |

- It is the trajectory of the simulation: a simulation takes one
  `TrajectoryReporter` or one `H5MDReporter`, not both (decision 7).
  `TrajectoryReporter("x.h5md")` is refused with the name of this class.
- Frames of positions alone are written inside the parts of a run, through
  the host function that writes the frames of DCD and XTC (D207): the cost
  is a copy of the positions and a write, and no part ends. That function
  is given the buffer of the state before the conversion to Å and f32 that
  DCD takes, so nothing is rounded.
- With `velocities` or `forces`, a frame ends a part at its step, as a
  `CheckpointReporter` does (D223), and is written from the state: the
  program is compiled before the reporters are known, and its frame call
  carries the positions only. The cost is a part boundary and a copy of the
  state per frame (measured below).
- Files are opened at the first `run`, backed up as the other outputs
  (`#name.n#`, D149), appended across `run` calls, and closed by
  `close_reporters()`.
- A minimization takes no reporters, as before.
- A build without HDF5 raises `UnsupportedError` when the reporter is made
  and from `read_h5md`, as it does for checkpoints.

`read_h5md(file, group=None)` returns a read-only `H5MDTrajectory`:

| Member | Value |
|---|---|
| `len(t)`, `t.particles` | The number of frames and of particles. |
| `t.steps`, `t.times` | `(K,)` int64 and float64 (ps): read when the file is opened. |
| `t[i]`, iteration | A `Frame`, read from the file when asked for. Negative indices count from the end; slices are not taken (use `range`). |
| `t.has_velocities`, `t.has_forces`, `t.has_potential_energy`, `t.has_tunables_version` | What each frame holds. |
| `t.units` | The `unit` attributes of the file by element (`"position": "nm"`, ...). |
| `t.creator`, `t.dtype` | The program that wrote the file, and `float64` or `float32`. |
| `t.close()` | Closes the file; also at the end of a `with` block. |

A `Frame` has `step`, `time`, `positions` (`(N, 3)`, the type of the file),
`cell` (`mdir.Cell`, as `State.cell`; `None` without a periodic cell),
`velocities`, `forces`, `potential_energy`, and `tunables_version` (`None`
where the file has none). It unpacks as the pair `(positions, edges)` of
the input contract of the frame evaluator (#249, its design document):
the positions in the order of the input and the `(3,)` edges of the cell in
nm, so `evaluator.evaluate(mdir.read_h5md(path))` reads one frame at a
time. A triclinic frame unpacks with the `(3, 3)` matrix of the cell
vectors in rows; the evaluator takes such frames once it takes the tilts
(#206, question Q10 of #250).

The reader reads the files of this writer and any H5MD file with
`particles/<group>/position` and a `box` (explicit or fixed step storage,
`edges` with time or without). It does not convert units: `t.units` says
what the file holds, and frames for the evaluator are in nm.

### Control file

```toml
[output]
trajectory           = "run.h5md"   # H5MD by the extension .h5md
trajectory_format    = "H5MD"       # or by name, for another extension
trajectory_precision = "DOUBLE"     # DOUBLE (the default) or SINGLE; H5MD only
trajectory_strings   = "FIXED"      # FIXED (the default) or VARIABLE; H5MD only
trajectory_interval  = 1000
```

- The format is chosen as D141 chooses DCD and XTC: by the extension with
  `trajectory_format = "AUTO"`, or by name. Only `.h5md` selects it by
  extension: `.h5` is the extension of checkpoints in the documents, and a
  trajectory named `x.h5` needs `trajectory_format = "H5MD"`.
- `trajectory_precision` and `trajectory_strings` are refused with DCD and
  XTC.
- The control file writes positions and the cell; velocities and forces in
  the frames of `mdir run` are left to a later item (decision 8).
- `--continue`, `--no-append`, backups, the manifest, and `mdir check`
  treat the file as they treat a DCD (D130, D149).

## Layout of the file

```
/h5md                                   version = [1, 1]
    /author                             name
    /creator                            name = "MDIR", version
    /modules/units                      version = [1, 0], system = "SI"
/particles/all
    /box                                dimension = 3, boundary = 3 x "periodic" or "none"
        /edges
            step, time                  hard links to position/step, position/time
            value   f64 [K][3] or [K][3][3]     unit "nm"
    /position
        step    i64 [K]
        time    f64 [K]                 unit "ps"
        value   f64 or f32 [K][N][3]    unit "nm"
    /velocity   (if asked)              step: link; time: link, or its own with leapfrog
        value   [K][N][3]               unit "nm ps-1"
    /force      (if asked)              step, time: links
        value   [K][N][3]               unit "kJ mol-1 nm-1"
    mass        f64 [N]                 unit "g mol-1"
    species     i32 [N]
/observables
    /potential_energy   (see below)     step, time: links; value f64 [K], unit "kJ mol-1"
    /tunables_version   (with tunables) step, time: links; value i64 [K]
/parameters/mdir
    trajectory_format = 1, precision, timestep (ps), period,
    velocity_offset, front_end
```

- **Order.** Row $i$ of every element is particle $i$ of the input, the
  order of `state()`, of DCD and XTC, and of the frame evaluator. No `id`
  element is written: by the specification the identity of a particle is
  then its index.
- **Units.** nm, ps, kJ/mol, and g/mol (the atomic mass unit): the units
  inside MDIR and of the Python interface. The control file and DCD are in
  Å, but a conversion would round, so the file holds the numbers of the
  state and labels them with the `unit` attribute of the units module of
  H5MD (version 1.0, system `SI`): `nm`, `ps`, `nm ps-1`, `kJ mol-1 nm-1`,
  `kJ mol-1`, `g mol-1`.
- **Cell.** `box/edges` is time-dependent in every ensemble, one row per
  frame, so that a reader need not know the ensemble: `[K][3]` for an
  orthorhombic cell, `[K][3][3]` for a triclinic one with the cell vectors
  in rows, $(L_x,0,0)$, $(b_x,L_y,0)$, $(c_x,c_y,L_z)$
  ([triclinic-m2.md](triclinic-m2.md)). The shape is that of the cell at
  the start; a barostat scales tilts and does not create them. Its `step`
  and `time` are hard links to those of `position`, as the specification
  requires. Without a periodic cell (D142), `boundary` is `none` and there
  is no `edges`: the cell the run places around the particles is not one of
  the system, as in DCD and XTC.
- **Step and time.** Explicit storage: `step` is the step of the run and
  `time` its time in ps, both continuing across a checkpoint. Elements
  sampled together share one `step` and one `time` through hard links.
  Leapfrog's velocities are half a step behind the positions: `velocity`
  then has a `time` dataset of its own with the times of the velocities,
  and `/parameters/mdir/velocity_offset` is $-0.5$.
- **Observables.** `tunables_version` is the version of the values of the
  tunables (D213) that the forces of the frame were computed with; it is
  written for a program with tunables and costs nothing.
  `potential_energy` is the potential energy of the row of the energy file
  at the step of the frame (kJ/mol, with the constant terms). It is
  written when every frame is a step of energy, which costs nothing more:
  always in a Python simulation, whose reports are steps of energy (D207),
  and in `mdir run` when `trajectory_interval` is a multiple of
  `energy_interval`; otherwise the element is absent. It is the energy the
  run reports, not the potential the forces sample under a plain cutoff
  (D210; decision 9).
- **Types.** Little-endian IEEE f64 and f32, i64, i32; strings of fixed
  length in ASCII, as the specification asks, except the `unit` attributes
  of a file written with `strings = "variable"`.

### Strings

The units module of H5MD gives the `unit` attribute as a string of fixed
length in ASCII, and that is what a file holds by default. With
`strings = "variable"` (`trajectory_strings = "VARIABLE"`) the `unit`
attributes, and only they, are strings of variable length in ASCII, a form
that the specification does not give. Everything else is the same in both
forms: the other strings of the file (the author, the creator, `boundary`,
the `system` of the units module, and those of `/parameters/mdir`) stay of
fixed length, and the datasets are equal to the bit.
`/parameters/mdir/strings` is `fixed` or `variable`; a file without it has
fixed-length strings. A continued run that asks for the other form than
the file has is refused, naming both; nothing is converted.
`read_h5md` reads both forms.

What was found, on the files of the test (`python-h5md-independent.test`):

| Reader | `strings = "fixed"` | `strings = "variable"` |
|---|---|---|
| `mdir.read_h5md` | reads | reads |
| h5py 3.16.0 (HDF5 2.0.0) | reads; `attrs["unit"]` is a `bytes` object | reads; `attrs["unit"]` is a `str` object |
| MDAnalysis 2.10.0 with h5py 3.16.0, `H5MDReader` | stops with the message "time unit 'b'ps'' is not recognized by H5MDReader", with and without `convert_units=False` | reads the files as written: positions, velocities, forces, the cell, the step, the time, and `potential_energy`, with and without its unit conversion |

Only the `unit` attributes decide this: the files that MDAnalysis 2.10.0
reads have every other string of fixed length.

### Chunks, compression, and flushing

- A chunk of `value` holds whole frames: one frame when a frame is 64 KiB
  or more (2,731 particles in f64), otherwise as many as fill 64 KiB. The
  datasets of one number per frame take chunks of 1,024.
- No compression: the mantissas of positions in f64 do not compress, and
  the HDF5 of the reference build has no deflate filter. The gain of
  gzip with and without the shuffle filter on JAC is measured
  [below](#size-and-cost) with h5py (decision 10).
- The writer extends and writes the values of a frame, then `step` and
  `time`, then flushes the library's buffers to the operating system
  (`H5Fflush`), once per frame. A run that is killed leaves a file that
  opens and holds the frames flushed. The file is written in the earliest
  format of the library (superblock 0), which has no mark of an open
  writer, so a killed file needs no repair tool. The reader counts the
  frames by the shortest dataset, and a continuation cuts every dataset to
  that count. There is no `fsync` per frame: as for DCD, a crash of the
  machine may lose frames that the checkpoint counts, and `--continue`
  then refuses the file by name.
- The library of the reference build is not thread-safe. The writer and
  the reader of this format take one mutex of the process around every
  call of it, so frames written by a run and files read in other threads
  take their turns; asynchronous writers (#113) have to keep that. The
  checkpoint code does not take that mutex: #262 lists what a script with
  threads can and cannot overlap.

### Continuation and overwriting

As DCD and XTC (D130, D149):

- A run that is not continued keeps a file of the same name as
  `#name.n#`.
- The checkpoint records the name of the trajectory and its number of
  frames. `mdir run --continue` and `Simulation(program, checkpoint=...)`
  open the file, check that it is a trajectory of this format with the
  same particles, type, elements, and shape of the cell, refuse it if it
  holds fewer frames than the checkpoint counts, cut every dataset to that
  count (the frames a run wrote past its last checkpoint are removed), and
  append. The step of the last frame kept must not be past the step of the
  checkpoint. The space of removed frames is reused by those that follow.
- `--no-append` and `append=False` write `<name>.partNNNN<ext>`.
- What a file holds is fixed when it is made. A Python simulation always
  writes the potential energy, and `mdir run` writes it only where
  `trajectory_interval` is a multiple of `energy_interval`: one front end
  continues the file of the other only where both would write the same
  elements, and is refused by name otherwise (`append=False` and
  `--no-append` go on in a part of their own).

## Validation

`test/Driver/python-h5md.test` and `python-h5md-gpu.test`
(`Inputs/python_h5md.py`) on the alanine dipeptide in water (1168
particles) and on water in a rhombic dodecahedron (1209 particles), in the
deterministic mode, on the CPU and on one RTX 3090, in double and in
mixed precision. Every comparison is of bits: the number of differing
values is 0 in each of the four combinations of target and precision.

| Quantity | Reference | Result |
|---|---|---|
| Positions and cell of 5 frames, `H5MDReporter` in f64, NVE | `state()` of a `CallbackReporter` at the same steps | equal to the bit |
| The same in f32 | the same states converted to f32 by NumPy | equal to the bit |
| Positions, velocities, forces, and cell (frames from the state) | the same | equal to the bit; the positions and velocities also equal those of the run without them |
| NPT (cell rescaling, PME): 5 frames with 5 cells, without and with velocities and forces | the same | equal to the bit |
| Triclinic cell, NVE: cell vectors in rows | `state().cell.vectors` | equal to the bit |
| Leapfrog: velocities of the half step and their times | `state()` | equal to the bit |
| Potential energy of each frame | `state().energies["potential"]` | equal |
| Version of the tunables after an update at step 20 | the simulation's history | `[0, 0, 1, 1]` at steps 10 to 40 |
| `mdir run`: the last frame, orthorhombic, triclinic NVE, and triclinic with a barostat | the checkpoint of the same step (`read_checkpoint`) | positions and cell equal to the bit |
| `mdir run` with `trajectory_precision = "SINGLE"` | the f64 file of the same run | its f32 rounding, to the bit |
| `mdir run`: potential energy of each frame | the row of the energy file | equal to the six decimals of the file |
| The DCD of the same `mdir run` | f32 of the H5MD positions times 10 | equal to the bit: the two writers are handed the same buffer |

A Python simulation takes no barostat in a triclinic cell yet, so that
case is covered by `mdir run` alone.

Continuation and a killed run:

| Case | Reference | Result |
|---|---|---|
| Python: a checkpoint at step 20, the run ends at 30, `Simulation(checkpoint=)` goes on to 60; positions alone, and with velocities and forces | the file of a run with the same checkpoints that did not stop | 6 frames equal to the bit; the frame of step 30 was removed and written again |
| The same continuation with another type, other elements, or another period | | refused, naming what differs |
| A file of the name that the checkpoint records whose first frames are of later steps | | refused: "holds a frame of step 40 among the 2 that the checkpoint counts, past its step, 20" |
| `append=False` | | the frames that follow in `<name>.part0002.h5md` |
| `mdir run --continue` stopped at every checkpoint (4 stops), then once more from the checkpoint before the last | the file of the run that did not stop | 5 frames equal to the bit; "removed 1 frames past the checkpoint" |
| SIGKILL in a callback at step 35, frames every 10 | the states of an unbroken run | the file opens and holds 3 frames, equal to the bit, every dataset of the same length |

A run that continues from a checkpoint begins a part anew at it (D223),
which moves the last bits of later steps by about $10^{-16}$ nm relative
to a run without that checkpoint; the reference runs therefore write the
same checkpoints.

Independent readers (`python-h5md-independent.test` and its `-gpu` twin,
`Inputs/h5md_independent.py`, which does not import `mdir`; run with
`-Dh5md_python=<interpreter>` since the Python of the tests has no h5py,
and unsupported otherwise):

| Reader | What it checks | Result |
|---|---|---|
| h5py 3.16.0 (HDF5 2.0.0), the paths of the specification written out by hand | `/h5md` version, author, creator, and the units module; fixed-length strings of the author and the creator; `box` dimension and boundary; shapes, types, and `unit` attributes of every element; `step` and `time` of `box/edges`, `velocity`, `force`, and the observables are the same objects as those of `position`; no `id`; superblock 0; positions, velocities, forces, cells, potential energies, steps, and times against the states that the run saved with NumPy | passes for 8 files (f64, f32, with velocities and forces, NPT, triclinic, and three with `strings = "variable"`), CPU and GPU; the killed file opens with 3 whole frames |
| MDAnalysis 2.10.0 with h5py 3.16.0, `H5MDReader`, the files written with `strings = "variable"`, as written | positions, velocities, forces (the f32 of the states, to the bit, with `convert_units=False`; within $10^{-6}$ after its conversion to Å), the cell as lengths and angles, the step, the time, and `potential_energy` in `ts.data` | passes for 3 files (with velocities and forces, NPT, triclinic), CPU and GPU |
| The same reader, the files written with `strings = "fixed"` | whether it opens them; the test prints the outcome and passes with either | stops with the message "time unit 'b'ps'' is not recognized by H5MDReader" |

The h5py script also checks the form of every string of the eight files
(the `unit` attributes of the form asked for, every other string of fixed
length, `/parameters/mdir/strings`) and that the files of the two forms
hold the same datasets, to the bit. `h5dump` and `h5ls` are not in the HDF5 of the reference
build, so the structure was listed with h5py.

The test without HDF5 (`python-h5md-no-hdf5.test`) runs only in a build
without it, which the standard build does not make; the branch of the
writer and the reader without HDF5 was compiled by hand.

### Size and cost

JAC (23,558 atoms, PME 64³, SHAKE and SETTLE, NVE, mixed precision) on one
RTX 3090 (GPU 0, alone). `mdir run`, 20,000 steps, ms per step over steps
10,000 to 20,000, two runs each (they agree within 6 µs per step); the
cost of a frame is the difference to no trajectory with a frame every 10
steps, where 1 µs per step is 0.01 ms per frame:

| Trajectory | ms/step, frames every 100 | ms/step, frames every 10 | ms per frame | Bytes per frame |
|---|---|---|---|---|
| None | 0.205 | 0.205 | | |
| DCD | 0.210 | 0.266 to 0.269 | 0.62 | 282,776 |
| H5MD, f64 | 0.212 | 0.280 to 0.286 | 0.78 | 565,642 |
| H5MD, f32 | 0.210 | 0.262 | 0.57 | 282,946 |
| XTC | 0.213 to 0.214 | | 0.85 (from every 100 steps, ±0.1) | 85,761 |

A frame in f64 is $24N$ bytes of positions and 250 bytes of everything
else (the step, the time, the cell, and the indices of the chunks); in
f32 it is 170 bytes more than a frame of DCD. Its cost in f32 is that of
DCD, and in f64 0.16 ms more for twice the bytes; at a frame every 1,000
steps that is 0.4% of the run.

A Python simulation of the same system, 10,000 steps after 2,000 with a
frame every 100 steps, the better of two runs (they agree within 2 µs per
step). Its frames are steps of energy (D207), which a frame of `mdir run`
above is not:

| Reporter | ms/step | ms per frame | Bytes per frame |
|---|---|---|---|
| None | 0.2690 | | |
| `TrajectoryReporter`, DCD | 0.2766 | 0.76 | 282,778 |
| `H5MDReporter`, f64 | 0.2778 | 0.88 | 569,066 |
| `H5MDReporter`, f32 | 0.2765 | 0.75 | 286,370 |
| `H5MDReporter`, f64, velocities and forces | 0.2872 | 1.82 | 1,700,064 |
| `CallbackReporter` that does nothing | 0.2767 | 0.77 | |

(The bytes per frame here are the file over 100 frames, with the fixed
part of the file, the masses and the species among it.) A frame with
velocities and forces ends a part and copies the whole state: 1.8 ms, 1 ms
more than a frame inside a part, for three times the bytes.

**Compression** (not implemented; measured with h5py on 50 frames of the
positions above, one chunk per frame as in the file):

| Positions | Filter | Stored | ms per frame |
|---|---|---|---|
| f64 | gzip 1 | 95.7% | 19.1 |
| f64 | shuffle, gzip 1 | 86.0% | 15.6 |
| f64 | shuffle, gzip 4 | 85.5% | 16.7 |
| f32 | shuffle, gzip 1 | 79.6% | 6.8 |

It would save 14% in f64 at twenty times the cost of the frame, so there
is no option for it (decision 10).

**Runs without the reporter.** The compiled program is unchanged: the
builder, the dialects, the conversions, and the runtime are not touched,
and a frame in H5MD goes through the host call that a frame of DCD goes
through. On the host, a run with a trajectory copies six more numbers
when a barostat sets the cell, and the frame function tests the kind of
its writer once per frame. JAC without a trajectory runs at 0.205 ms per
step with this change.

## Decisions

Decided in the design:

1. H5MD is a third trajectory format behind the writer interface of DCD and
   XTC, so that continuation, the frame count of the checkpoint, backups,
   and parts are those of D130 and D149.
2. Only `.h5md` selects it by extension; `.h5` stays the checkpoints'.
3. The file is in nm, ps, kJ/mol: the numbers of the state, labeled.
4. `box/edges` is time-dependent in every ensemble; no `id` element.
5. The first argument after the file is `period`, the name the other
   reporters use.
6. One type for positions, velocities, and forces of a file.

Decided by the maintainer on the pull request:

7. A simulation writes one trajectory, as `[output]` has one: a DCD or XTC
   and an H5MD at once would need a second slot with its own count in the
   checkpoint.
8. Velocities and forces are written from a Python simulation only, at
   part boundaries. From `mdir run` they would need a frame call of the
   compiled program that carries them.
9. The potential energy of a frame is the one the run reports, which is
   free at a step of energy. The shifted energy of D210 is formed only in
   the evaluation of the derivative in the tunables (a part boundary and
   an evaluation per frame); the frame evaluator computes
   $U_{\hat\theta}(S_n)$ itself, and the stored value is a check.
10. No compression: 14% of an f64 frame for twenty times its cost
    ([Size and cost](#size-and-cost)).
11. The `unit` attributes are strings of fixed length by default, as the
    specification has them, and an option writes them as strings of
    variable length ([Strings](#strings)).
