# Frames without loss: the H5MD trajectory (D[h5md-reporter])

Issue #251. Status: **design; the implementation and its measurements
follow in the same pull request.** What is decided here and what is put to
the maintainer is listed under [Decisions and questions](#decisions-and-questions).

The frame evaluator of M2b (#249, [python-frames.md](python-frames.md))
evaluates the potential at stored frames. The frames have to be those the
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

`H5MDReporter(file, period, positions="f64", velocities=False, forces=False)`:

| Argument | Meaning |
|---|---|
| `file` | The name of the file; any extension. |
| `period` | Steps between frames; the schedule of the other reporters (D207): a frame at every step that is a multiple of `period` after the step the reporter was added at. |
| `positions` | `"f64"` (the default) or `"f32"`: the type of the positions in the file, and of the velocities and the forces if they are written. `"f64"` holds the state of either precision mode without loss; `"f32"` holds the state of the mixed mode without loss and rounds that of the double mode to the nearest f32. |
| `velocities`, `forces` | Whether the frames hold them: those of `state()` at the step of the frame. |

- It is the trajectory of the simulation: a simulation takes one
  `TrajectoryReporter` or one `H5MDReporter`, not both (see question 1).
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
the input contract of the frame evaluator ([python-frames.md](python-frames.md)):
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
trajectory_interval  = 1000
```

- The format is chosen as D141 chooses DCD and XTC: by the extension with
  `trajectory_format = "AUTO"`, or by name. Only `.h5md` selects it by
  extension: `.h5` is the extension of checkpoints in the documents, and a
  trajectory named `x.h5` needs `trajectory_format = "H5MD"`.
- `trajectory_precision` is refused with DCD and XTC.
- The control file writes positions and the cell; velocities and forces in
  the frames of `mdir run` are left to a later item (question 2).
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
  H5MD (version 1.0, system `SI`). The strings are those that the reader of
  MDAnalysis knows.
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
  (D210); see question 3.
- **Types.** Little-endian IEEE f64 and f32, i64, i32; strings of fixed
  length in ASCII, as the specification asks.

### Chunks, compression, and flushing

- A chunk of `value` holds whole frames: one frame when a frame is 64 KiB
  or more (2,731 particles in f64), otherwise as many as fill 64 KiB. The
  datasets of one number per frame take chunks of 1,024.
- No compression: the mantissas of positions in f64 do not compress, and
  the HDF5 of the reference build has no deflate filter. The gain of
  gzip with and without the shuffle filter on JAC is measured below from a
  file repacked with h5py; an option is a later item if the numbers ask
  for it (question 4).
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
- The library of the reference build is not thread-safe; frames and
  checkpoints are written under the mutex of the run, one at a time.
  Asynchronous writers (#113) have to keep that.

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

## Validation (to be filled by the implementation)

- Frames read back with `read_h5md` against the `state()` of a
  `CallbackReporter` at the same steps: equal bits of the positions, the
  cell, the velocities and the forces in f64; in f32, the f32 rounding of
  the state. CPU and GPU, double and mixed, orthorhombic and triclinic,
  NVE and NPT. The same frames against the DCD of the same run within the
  rounding of DCD.
- An independent reader: h5py with the paths of the specification by hand,
  and the H5MD reader of MDAnalysis with unit conversion on.
- `--continue` and `Simulation(checkpoint=)`: the file of a run that
  stopped and continued holds the datasets of the run that did not stop.
  A run killed with SIGKILL between frames: the file opens and holds the
  frames flushed.
- Cost on JAC (23,558 atoms), GPU 0: bytes per frame and ms per frame
  against DCD; ms per step without the reporter against `main`.

## Decisions and questions

Decided in this design (say so on the pull request to change one):

1. H5MD is a third trajectory format behind the writer interface of DCD and
   XTC, so that continuation, the frame count of the checkpoint, backups,
   and parts are those of D130 and D149.
2. Only `.h5md` selects it by extension; `.h5` stays the checkpoints'.
3. The file is in nm, ps, kJ/mol: the numbers of the state, labeled.
4. `box/edges` is time-dependent in every ensemble; no `id` element.
5. The first argument after the file is `period`, the name the other
   reporters use.
6. One type for positions, velocities, and forces of a file.

Put to the maintainer:

| | Question | Options | Taken until answered |
|---|---|---|---|
| 1 | May a simulation write a DCD or XTC for viewing and an H5MD for reweighting at once? | (a) one trajectory, as `[output]` has one; (b) a second slot with its own period and its own count in the checkpoint (a new entry of the checkpoint format) | (a) |
| 2 | Velocities and forces in the frames of `mdir run` | (a) Python only, at part boundaries; (b) control-file keys and a frame call of the compiled program that carries them, which changes the program of a run that asks for them | (a) |
| 3 | The potential that the forces sample (D210) per frame | (a) the reported potential energy, free; (b) also D210's shifted energy, which is formed today only in the evaluation of the derivative in the tunables (a part boundary and an evaluation per frame, and a program compiled with `tunable_gradient`), opt-in | (a): the frame evaluator computes $U_{\hat\theta}(S_n)$ itself, and the stored value is a check |
| 4 | Compression | (a) none; (b) an option for gzip with shuffle, where the library has it | (a), with the measured gain |
