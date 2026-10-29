# Driver and Control File for Milestone M0

Status: decided (2026-09-29) and implemented; kept current (last checked
2026-10-21). The reference of the keys is
[Appendix A of the white paper](paper/A-control-file.md).
Section 4 lists what is left.

This document describes the driver of MDIR and its input, the control file.

## 1. The control file

The control file is a TOML file (D24). It has one table for each concern of
a run.

| Table | Holds | In MDIR |
|---|---|---|
| `[input]` | The files of the system: positions, the state of an earlier run | |
| `[output]` | The files that the run writes | |
| `[energy]` | The terms of the potential energy and their cutoffs | `md` |
| `[dynamics]` | The integrator, the time step, the number of steps, the periods of output | `dyn` |
| `[ensemble]` | The ensemble, temperature, and pressure | `ensemble` |
| `[boundary]` | Periodicity and the size of the box | The cell |
| `[execution]` | The target, the threads, the precision | Structural plan parameters |

Later milestones added `[constraints]`, `[[restraints]]`, `[minimize]`
(in place of `[dynamics]`), `[thermostat]`, `[barostat]`, `[pme]`,
`[lj_pme]`, and `[free_energy]`; these are all the tables that the reader
takes ([Section A.1](paper/A-control-file.md#a1-keywords) of the white
paper).

```toml
[input]
coordinates = "argon.pdb"       # positions
# checkpoint = "equilibrated.h5" # the state of an earlier run

[output]
trajectory          = "run.dcd"
checkpoint          = "run.h5"
energy_interval     = 100
trajectory_interval = 100
checkpoint_interval = 2000

[energy]
switch_distance   = 7.5         # Å
cutoff            = 8.5
pairlist_distance = 9.5

[[energy.pair]]
name       = "lj"
expression = "4*epsilon*((sigma/r)^12 - (sigma/r)^6)"
mixing     = "lorentz-berthelot"

[[energy.type]]
name    = "AR"
mass    = 39.95
epsilon = 0.2385                 # kcal/mol
sigma   = 3.4

[dynamics]
integrator = "VELOCITY_VERLET"
time_step  = 0.005              # ps
steps      = 2000
seed       = 314159

[ensemble]
ensemble    = "NVE"
temperature = 94.4               # for the initial velocities

[boundary]
type = "PERIODIC"
box  = [34.7786, 34.7786, 34.7786]  # Å

[execution]
target    = "GPU"               # or "CPU"
threads   = 16                  # for the target CPU
precision = "MIXED"             # SINGLE, MIXED, or DOUBLE
```

Keywords are in lower case and spelled out. Values that name a choice,
such as `VELOCITY_VERLET` and `PERIODIC`, are strings and are read without
regard to case.

For `backend = "STAGES"`, `precision = "MIXED"` keeps neural stage features
and their reverse-sweep adjoints in f32 by default. The model key
`stage_field_precision = "DOUBLE"` opts them into f64 storage; each
exported stage body still computes in its declared type
(D329). The key is available only with `backend = "STAGES"`
and mixed or double execution precision. All staged models in one run
must use the same choice. It is included in the checkpoint fingerprint.

### 1.1 Keywords

The keywords of every table, their values, their defaults, and the
decisions behind them are listed in one place, Appendix A of the white
paper, which this document does not copy:
[Section A.1](paper/A-control-file.md#a1-keywords) is the table of the
keywords, [A.2](paper/A-control-file.md#a2-a-run-from-an-amber-topology)
a commented control file of a run from an Amber topology,
[A.3](paper/A-control-file.md#a3-runs-over-more-than-one-job) the runs
over more than one job,
[A.4](paper/A-control-file.md#a4-templates-of-the-standard-pipeline) the
templates of the standard pipeline,
[A.5](paper/A-control-file.md#a5-preflight) the preflight, and
[A.6](paper/A-control-file.md#a6-the-checkpoint-file) the checkpoint file.
`mdir template md` and `mdir template amber` print control files with
every keyword; the keywords follow the code, not earlier versions, until
a milestone is released.

A keyword of a pair term or of a type that Section A.1 does not list
names a number: a parameter of the type, or a constant of the term, that the
expression uses.

A parameter that has the same value for all types is a constant of the
kernel. A parameter that differs is a field, and `mixing` says how the
parameter of a pair follows from those of its two particles:
`lorentz-berthelot` [[Lorentz1881]](references.md#lorentz1881), [[Berthelot1898]](references.md#berthelot1898), `geometric`, or a table with `arithmetic` or
`geometric` for each parameter.

### 1.2 Units

| | The control file | Inside MDIR |
|---|---|---|
| Length | Å | nm |
| Energy | kcal/mol | kJ/mol |
| Time | ps | ps |
| Mass | amu | amu |
| Temperature, pressure | K, atm | K, bar |

Å, kcal/mol, and ps are what force fields and most users of all-atom
simulations write. With amu they are not consistent with one another: a
kinetic energy in these units needs a factor. nm, kJ/mol, amu, and ps are
consistent, which is what S4 asks of the unit system inside MDIR. The
driver converts when it reads and when it writes.

An expression is evaluated in the units of the control file: the kernel
scales the distance before the expression and the energy after it. The
driver need not know the dimension of a parameter.

The syntax is that of the custom forces of OpenMM (D22, D136): numbers,
names, `+ - * / ^`, parentheses, the functions `sqrt`, `exp`, `log`,
`sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`,
`abs`, `floor`, `ceil`, `erf`, `erfc`, `step` (1 for x ≥ 0), `delta` (1 for
x = 0), `square`, `cube`, `recip`, `sec`, `csc`, `cot`, `min`, `max`,
`atan2(y, x)`, and `select(x, a, b)` (a if x ≠ 0, else b), the tabulated
functions of `[[energy.function]]` (D138), and definitions of names after
semicolons: `k*d^2; d = r - r0`. The terms over tuples, over centers,
over the pairs of a topology, and of the positions may use the time `t`
in ps, that of the end of the step whose forces they give (D145, D148).

A larger tabulated grid may be kept in a file (D178):

```toml
[[energy.function]]
name = "pair_grid"
values_file = "potentials/pair.dat"
shape = [4]
min = 1.0
max = 4.0
```

The file `potentials/pair.dat`, relative to the control file, contains:

```text
# Four energy values, at 1, 2, 3, and 4 Angstrom.
0.0 1.0
4.0 9.0
```

Each value is a finite decimal number, with an optional sign, decimal
point, or `e`/`E` exponent. Whitespace separates values; `#` begins a
comment through the end of the line. Blank lines and line breaks do not
set dimensions. `shape` gives one to three positive integer sizes, each
fitting in an unsigned 32-bit integer, and the file must contain exactly
their product. The last argument varies fastest, matching inline nested
lists: for `[nx, ny]` the first `ny` values span the second argument at the
first point of the first. For `shape = [2, 3]`, the file
`11 12 13 21 22 23` is equivalent to
`values = [[11, 12, 13], [21, 22, 23]]`. The reader converts that order to
the internal table layout before fitting the splines. `values` and `values_file` are mutually
exclusive, and `shape` is only allowed with a file. The existing range,
periodic endpoint, and discrete-function rules apply to the loaded grid.
Values keep the units of the expression that calls the function.

The file is read before compilation, is protected from collisions with
outputs, including resolved part files and the previous checkpoint, and
appears as `tabulated_function` in the manifest's input
hashes and in the bug report. Continuation compares the loaded grid as
inline values: changing its numbers is a change of physics, while moving
the file, changing whitespace or comments, or using equivalent inline
values preserves it. A fresh run uses the same output backups as other
control files; the input grid is never written. This format is a grid of
values, not a GROMACS seven-column potential file; such a file needs
conversion to the chosen expression and grid.

### 1.3 Standard pipeline templates

`mdir template minimize`, `nvt`, `npt`, and `production` print the four
stages of `examples/ala3`, with `system.prmtop` and `system.inpcrd` as
placeholder input paths (D147, roadmap U9). Each command prints one
control file to standard output; it creates no files. The stages are
embedded at build time, so the binary needs no copy of the examples at
run time. `md` and `amber` remain the reference templates.

| Kind | Run | Restraints on the solute's heavy atoms | Input checkpoint | Output |
|---|---|---|---|---|
| `minimize` | 2000 steps of steepest descent | 10 kcal/mol/Å² | None | `min.h5` |
| `nvt` | 50 ps at 300 K, constant volume | 10 kcal/mol/Å² | `min.h5` | `nvt.h5` |
| `npt` | 100 ps at 300 K and 1 atm | 1 kcal/mol/Å² | `nvt.h5` | `npt.h5` |
| `production` | 1 ns at 300 K and 1 atm | None | `npt.h5` | `md.h5`, `md.dcd` every 1 ps |

The dynamics stages use velocity Verlet at 2 fs, stochastic velocity
rescaling, and, at constant pressure, stochastic cell rescaling. All
stages use PME, constraints on hydrogen bonds, and rigid water. The
templates select a GPU in mixed precision and require HDF5 for the
checkpoints; `target = "CPU"` selects the host.

Edit the input paths, lengths of the runs, and execution settings for the
system. The restraint selection `!:WAT & !@H*` assumes water named `WAT`
and hydrogen names beginning with `H`; adapt it and `water_residues` for
other residue names. These stages describe a solute in water with an
isotropic barostat; the bilayer example has its own protocol.

Save the stages in one directory, for example as `1-min.toml`,
`2-nvt.toml`, `3-npt.toml`, and `4-md.toml`, and run them in that order
with `mdir run --continue FILE`. All paths are relative to the control
file. Each stage begins at the checkpoint of the preceding stage, while
`--continue` resumes its own output checkpoint or skips it when complete
(Section 2.7). The coordinate file remains the restraint reference.

### 1.4 Preflight

`mdir check FILE` reads and validates the control file, topology, and
coordinates, and reports the system and the planned run (D151, U6).
Both topology inputs and PDB inputs get the ensemble, integrator and
number of steps, time step in ps and duration in ns, electrostatics and
PME settings, constraint count, target, and precision. A minimization
reports its method and iterations, with no time step or physical duration.
A nonperiodic system has no physical cell or density in the report.

The output list has the files of Section 2.8 in the order of its table:
the log, on standard output and in the file of `log`, the energies and
the terms over centers as columns, the trajectory in DCD, XTC, or H5MD, and the
H5MD checkpoint. Each entry says whether it is enabled, its path and
format, its interval in steps, how many rows, frames, or checkpoints the
whole run writes, and whether the file exists. A minimization checkpoint
is written at the end.
A named trajectory with interval 0 is disabled even if its path exists.
Paths follow the same rule as a run: relative to the control file.

Warnings and notes go to standard error and leave the exit status at 0.
Notes give, for each enabled output file that exists, the name under
which `mdir run` keeps it before it writes its own (D149). Warnings flag
a file that has 99 backups already, a fixed interval between neighbor rebuilds
(D88), dynamics with no checkpoints or energy reports, a missing input
checkpoint, and a requested feature absent from the build (HDF5 or CUDA).
Each warning says what to change or run. Invalid input gives exit status 1.
The check never opens an output for writing, loads an input checkpoint,
or compiles a program. Its duration is the configured length of the stage,
not the remaining time in a continuation. Missing stage checkpoints are
warnings so that a whole pipeline can be inspected before it runs.
Device availability, compilation, and checkpoint compatibility are checked
when the run begins. Automatic PME grid and beta choices are labeled
automatic, with their spacing, tolerance, and interpolation order.

`mdir check FILE --json` writes one JSON object to standard output, also
on input errors. Warnings are included there instead of on standard error.
Its schema begins at version 1:

| Field | Meaning |
|---|---|
| `schema_version`, `ok` | Version 1; `ok` is true when reading and validation succeeded, even with warnings |
| `system` | Particle and type counts, mass in amu, degrees of freedom, supplied velocities, restrained particles, periodicity, and topology counts when present |
| `system.cell_angstrom` | The reduced cell's `diagonal` and `tilt` ($b_x$, $c_x$, $c_y$) in Å; `null` without periodicity |
| `run` | Kind, ensemble, method, steps, `time_step_ps`, `duration_ns`, `temperature_kelvin`, `pressure_atm`, thermostat and barostat coupling, cutoff, PME, constraints, execution, the neighbor structure, and input checkpoint path |
| `run.neighbors` | `requested` (`auto`, `matrix`, `groups`), `structure` (`matrix` or `groups`, `AUTO` resolved for the system), `reason` (`given`, or why `AUTO` took it), `pairlist_distance_angstrom`, and `pruned_distance_angstrom` (`null` without a dual list); the text output has the line `neighbors:` (D261) |
| `run.skins` | `pairlist_distance_source` and `pruned_distance_source` (`given`, `default`, or `cached`: the check measures nothing) and `reason`; the text output has the line `skins:` (D281, [neighbors.md](neighbors.md), Section 6.4) |
| `run.pme` | `null` when disabled; otherwise `grid_points` (three nulls when automatic), `beta_inverse_angstrom` (`null` when automatic), `order`, `max_spacing_angstrom`, and `tolerance` |
| `outputs` | Entries with `kind` (`log`, `energy`, `pull`, `free_energy`, `observables`, `trajectory`, `checkpoint`, `manifest`), `path`, `format`, `interval_steps`, `enabled`, `at_end`, `count` and `count_of` (`rows`, `frames`, `checkpoints`; `count` is null where the run decides, as in a minimization), `exists`, and `backup`, the name under which a run keeps the file that exists (D149), or null; `path` is null for an unconfigured file, and for the log when it goes to standard output only |
| `warnings` | Objects with `code` and `message`: `backup_limit`, `constant_expression`, `empty_selection`, `flexible_water`, `long_time_step`, `short_thermostat_period` (D163a), `brownian_step` (D163b), `charged_soft_core` (D161), `pair_tail_left_out` (D209), `unused_parameter`, `fixed_rebuild_interval`, `no_checkpoint`, `no_energies`, `missing_input_checkpoint`, `hdf5_unavailable`, or `gpu_unavailable`. Each is reported once (`short_thermostat_period` was added twice for a run from a topology before #124), and a run whose particles come from a PDB file reports the warnings of its control file (`unused_parameter`, `constant_expression`, D158) as a run from a topology does (#129) |
| `notes` | Objects with `code` and `message` for what a run will do that needs no change: `output_backup`, an output that exists and the name it will be kept under |
| `errors` | Error messages; empty on success. On failure, `system`, `run`, and `outputs` are absent |

In a minimization, `time_step_ps`, `duration_ns`, and temperature are null;
pressure is null without a barostat. Temperature denotes the bath, or the
temperature used if initial velocities must be drawn. An input checkpoint
path is reported but its contents are not inspected. The output intervals
describe the schedule only when `enabled` is true, and `at_end` takes
precedence over the interval for minimization checkpoints. The manifest
has interval 0 and a null count: its events occur at execution start and
end, independently of the simulation schedule.

### 1.5 Installation doctor

`mdir doctor [--target=all|cpu|gpu] [--timeout=<seconds>]` checks the installation (D155). It
prints the build information of `mdir version`, then compiles and runs an
embedded two-atom Lennard-Jones system for two steps on each requested
target. The CPU uses double precision and two OpenMP threads; the GPU uses
mixed precision. The initial pair energy must agree with the analytic
expression to the log's rounding, and both energy rows must be finite.

By default it checks every built target. A CPU-only build prints `SKIP`
for the GPU; an explicit `--target=gpu` on that build fails. A CUDA build
must pass the driver probe and GPU run: missing devices do not silently
skip the check. `--target=cpu` works without a GPU or its driver. The GPU
probe dynamically loads the driver and reports its supported CUDA API
version, visible device names, compute capabilities, and selected device.
`CUDA_VISIBLE_DEVICES` and `MDRT_DEVICE` select devices as in a normal run.

Each run uses the same executable in a child process, with a limit of
120 seconds (`--timeout=<seconds>` changes it, for a machine busy with
other work; 0 waits without a limit), so a failed CPU run does not prevent
checking the GPU; a run stopped at the limit says so (#663). The
control files, coordinates, captured output, and any compiler reproducer
are in a unique temporary directory. Success removes it; failure prints
its path, the log paths and a rerun command when a run was attempted,
and keeps the files. The child inherits the environment except that
`MDIR_REPRODUCER` points inside this directory. Exit status is 0 when all
requested checks pass, 1 otherwise (including failure to remove the files).

This checks loading, compilation, and basic execution. It reports whether
HDF5 was built but does not test checkpoints, PME, constraints, or the
stability of a user's system; use `mdir check FILE` and the test suite for
those concerns.

## 2. The driver

### 2.1 What it does

```text
mdir run control.toml

  1. read the control file and the files of [input]
  2. build a module of md and dyn ops
  3. compile it: the passes of md, md_exec, and the lowering of the target
  4. load the code and run it
        the code calls the runtime to write energies, frames, checkpoints
  5. write the last checkpoint
```

The driver is the subcommand `run` of the C++ program `tools/mdir`
(D59); `--continue`, `--no-append`, and `--max-walltime` carry a run over
more than one job (Section 2.7). It runs the passes in its own process and executes the code with
the execution engine of MLIR. The
Python library (C5) calls the same steps: one model and IR builder for
both front ends (D191), and the compilation and the simulations of
`lib/Compiler` (D192, D196; [python-compile.md](python-compile.md),
[python-segments.md](python-segments.md)).

### 2.2 The schedule is compiled

The periods of output are known before the run, so the driver compiles the
whole run as one function with nested loops:

```text
the particles are put in the order of their positions      (D44)
for each checkpoint interval           checkpoint_interval steps
    the particles are put in order again
    neighbor structures start empty      (R1)
    for each frame interval            trajectory_interval steps
        for each energy interval       energy_interval steps
            for each step
                one step of the integrator
            mdrt.write_energies
        mdrt.write_frame
    mdrt.write_checkpoint
```

| Property | How |
|---|---|
| The state stays on the device between outputs | `mdrt.write_frame` takes a buffer of the host, so the positions are copied only when a frame is written. |
| A neighbor structure outlives a frame and an energy output | The loop over checkpoint intervals carries it. |
| A restarted run rebuilds where the first run did | R1: the structure starts empty in every checkpoint interval. |
| Every front end rebuilds at the same steps | The evaluation at the start is a segment of its own: the steps begin with a structure of their own, built at step 1, with or without checkpoints (`mdrt.fresh`, D218). |
| Energies are computed only when they are written | The step of an energy interval that is written requests the energy; the others request forces only. |
| A loop over pairs reads the neighbors of a particle from few places in memory | The particles are in the order of their positions. The loop over checkpoint intervals carries the masses, the parameters, and the numbers of the particles as well, because they change places with every new order. |
| The files are in the order of the input | A call that writes takes the numbers of the particles with the field, and the driver writes the value of a particle at the place of its number. |

`examples/argon/argon.mlir` has this form already, with two levels of loops.

The periods must divide one another: `trajectory_interval` and `checkpoint_interval`
are multiples of `energy_interval`.

### 2.3 The log

The log goes to the standard output and, with `log` in `[output]`, to that
file as well (D149, Section 2.8); `energy` writes its rows as columns. Its
first line names the build that wrote it, `MDIR 0.1.0, commit
<12 hexadecimal digits>`, followed by `with uncommitted changes` when the
tree had them (D174). The file holds every line from the first: the run
keeps what it prints before the file opens, at the end of its set-up, and
writes it there. After that it keeps no text of its log, and a run without
`log` keeps none from the same point (#304): its memory does not grow with
the rows. Before, every line of a run without a log file stayed in memory
until the run ended, +154 to +170 bytes a row of the energies between rows
512 and 65536 (a row is about 121 bytes of text), about 60 GB of text for
$5\times10^8$ steps with a row at every step.

```text
INFO:      STEP           TIME      TOTAL_ENE  POTENTIAL_ENE    KINETIC_ENE    TEMPERATURE         VIRIAL       PRESSURE
INFO:      2000        10.0000      -834.1467     -1083.2389       249.0922        96.8538       -70.6513       232.3875
```

The line shows the last row of `examples/argon/argon.toml`.

The whole log of `mdir run examples/argon/argon.toml` on the CPU, as of
2026-10-25 (the times and the rates depend on the machine):

```text
MDIR 0.1.0, commit <12 hexadecimal digits>
MDIR: 864 particles, 2000 steps of 0.005 ps
MDIR: neighbor structure: the matrix with the reach 9.5 Å (AUTO: the target is the CPU)
MDIR: compiled in 0.70 s
INFO:      STEP           TIME      TOTAL_ENE  POTENTIAL_ENE    KINETIC_ENE    TEMPERATURE         VIRIAL       PRESSURE
INFO:         0         0.0000      -834.1316     -1304.8874       470.7558       183.0000     -4920.3575     -2161.8410
INFO:       100         0.5000      -834.1466     -1087.9806       253.8339        98.6980      -300.5656       112.6241
...
INFO:      2000        10.0000      -834.1467     -1083.2389       249.0922        96.8538       -70.6513       232.3875
MDIR: ran in 0.24 s, 0.12 ms per step, 3582.5 ns per day
MDIR: from step 1000 to step 2000, 0.119 ms per step, 3631.4 ns per day
MDIR: neighbor structures were built 114 times, every 17.7 steps on average
MDIR: the momentum at the end is 4.246e-12 amu nm/ps
MDIR: the total energy changed by 1.813e-05 of its value
```

| Column | Unit | Meaning |
|---|---|---|
| `TIME` | ps | |
| `POTENTIAL_ENE` | kcal/mol | |
| `KINETIC_ENE` | kcal/mol | `K`, the kinetic energy of the velocities at the time of the row |
| `TOTAL_ENE` | kcal/mol | `POTENTIAL_ENE + KINETIC_ENE` |
| `TEMPERATURE` | K | $2 K_T / (N_f k_B)$, with $N_f = 3N_{m>0} - N_c - 3$ degrees of freedom (the particles with mass, less the constraints and the motion of the center of mass) and $K_T$ as below |
| `VIRIAL` | kcal/mol | The trace of the virial of the pressure, that of the groups that the barostat scales, $\mathsf W_g = \sum_g \mathbf R_g \otimes \mathbf F_g$ over the rigid groups by their centers and every other particle alone (D264); without constraints the MDIR virial $\mathsf W = \sum \mathbf d_{ij} \otimes \mathbf K(i, j)$, which is positive for repulsion (B8). Other packages print other quantities under this name: the GROMACS virial is $-\mathsf W / 2$ [[GromacsManual2025]](references.md#gromacsmanual2025). |
| `PRESSURE` | atm | $(2 K_P + \mathrm{tr}\,\mathsf W_g) / (3V)$, with $K_P$ as below |
| `CONSERVED` | kcal/mol | With a thermostat or a barostat: the total energy plus what the couplings have taken from the system |
| `VOLUME` | Å³ | With a barostat: the volume of the cell |
| `AREA_XY` | Å² | With a barostat: the area of the xy face spanned by the first two cell vectors (D170); not area per lipid |

The potential energy and the virial include the correction for the
dispersion beyond the cutoff and the energy of the background that
neutralizes a net charge under particle mesh Ewald, both at the volume of
the row; the virial includes those of the virtual sites and of the
restraints, and not those of the constraints, which cancel within a
group. Before the first row the log lists the
terms of the potential energy at the start (`MDIR: the terms at the
start, in kcal/mol`).

**Three kinetic energies (D45).** The velocities of a step are those of a
finite difference of the positions, not those of the trajectory, so their
kinetic energy $K$ is off by a term of the order $\Delta t^2$. The kinetic
energy of the velocities half a step before and after is off as well, by
half as much and in the other direction. The mean of the two half steps
is

$$K_\text{half} = K + \frac{\Delta t^2}{8} \sum_i \frac{\lVert\mathbf F_i\rVert^2}{m_i},$$

which holds exactly without constraints and thermostats. With
constraints the forces do not give the kinetic energies of the half steps,
and the steps of the rows measure them: they take the first half of the
next step, a kick, a drift, and the constraints of the positions, and the
mean of the kinetic energies of the velocities of the two drifts after
their constraints (D203); under Langevin dynamics, by g-BAOAB, the two
drifts of half a step around the end of the step, that of the next
without its noise (D265). Under Langevin dynamics the
temperature takes $K_\text{half}$ in place of $K_T$: the friction and the
noise act on the velocities between the drifts, which the step samples at
the temperature of the bath. The row of step 0 has no step
before it and takes $K$ there. The thermostats control $K_T$: the steps
that close a period of coupling measure $K_\text{half}$ as the rows do
(D263). The pressure takes the kinetic energy of the centers
of the groups at the half steps, which their forces give exactly, with or
without constraints, at step 0 as well (D264,
[[Jung2020]](references.md#jung2020)). With rigid waters, a run ends with
`MDIR: the temperatures over N rows, in K: of the solute ..., of the
solvent ...; from the velocities of the steps alone, ... and ...`: the means
of $T$ and of $2K/(N_fk_B)$ over the rows, with errors from ten blocks of
rows, for the solvent, the waters of SETTLE (6 degrees of freedom each),
and the solute, every other particle, ions included; the three degrees of
freedom of the center of mass are shared in proportion
(D203). The run keeps their sums and the sums of the ten blocks,
whose length the number of rows of the run gives before it begins, and
not the rows (#300), in the order of the pass over the rows that they
replace, so the two lines are the same to the digit. Before, each row kept
48 bytes in vectors that double, 24 GB for $5\times10^8$ steps with a row
at every step, and a Python simulation 64 bytes a call of
`run(1, energy=True)`, about 70 MB after 1.5 million calls
(`output-rows.test`, `python-memory.test`).

| Quantity | Kinetic energy | Reason |
|---|---|---|
| Total energy | $K$ | The sum with the potential energy varies least: for argon with a step of 20 fs, by 0.05 kcal/mol, against 0.10 and 0.12 with the other two. |
| Temperature | $K_T = (K + 2 K_\text{half}) / 3$, the mean of the three times | The terms of the order $\Delta t^2$ cancel. |
| Pressure | $K_P = K_{\text{half},g}$, that of the centers of the groups at the half steps | The positions of the steps satisfy the virial theorem with the kinetic energy of the half steps; for the centers of the groups, which the barostat scales, the forces of the constraints drop out. Without constraints $K_P = K_\text{half}$. |

The estimators are those of Jung, Kobayashi, and Sugita
[[Jung2018]](references.md#jung2018), [[Jung2019]](references.md#jung2019). For liquid argon they were compared with runs at a step of 1 fs,
through the potential energy and the heat capacity:

| Step | Error of the temperature from `K` | From `K_half` | From `K_T` | Error of the pressure from `K` | From `K_P` |
|---|---|---|---|---|---|
| 10 fs | −0.04 K | +0.09 K | +0.05 K | −1.4 atm | −1.0 atm |
| 20 fs | −0.35 K | +0.19 K | +0.01 K | −1.5 atm | +0.1 atm |
| 30 fs | −0.89 K | +0.34 K | −0.07 K | −3.5 atm | −0.1 atm |

The averages have an uncertainty of 0.05 K and 1.3 atm.

The initial velocities have the temperature of the control file in `K`.
The first row of the log therefore shows a temperature that is a little
higher, unless the forces are zero.

With leapfrog, the stored velocities are half a step behind the
positions. Where energies are written, the driver computes the forces at
the positions and from them the velocities of their time, so the log is
that of velocity Verlet, row by row.

The virial is computed in the steps whose energies are written, in the
loops that compute the energy and the forces.

### 2.4 Parts

| Part | Work | Depends on |
|---|---|---|
| Parser of the control file | TOML to a description of the run; errors with the line (`lib/Driver/Control.cpp`) | toml++ |
| Readers of systems | PDB: positions and names (`System.cpp`). Amber: the topology (`prmtop`, also of the format before Amber 7, with dihedrals of several terms given by negative periodicities, D133) with CMAP, and coordinates and velocities from `inpcrd` or `rst7` (`Amber.cpp`, `CMap.cpp`). GROMACS: `top` and `itp` with `#include`, `#define`, `#ifdef`, wildcards of dihedral types, and the order of LEaP for the atoms of dihedrals when the force field asks for it, and coordinates from `gro` (`Gromacs.cpp`). CHARMM: a PSF, its CRD, and the files of parameters (`Charmm.cpp`, D122). Triclinic cells reduced (`Cell.cpp`, D123) | |
| Selections | The masks of Amber for the atoms of restraints (`Selection.cpp`) | |
| Builder | The description of the run to a module (`Builder.cpp`) | |
| Energy expressions | The syntax of D22 to a kernel (`Expression.cpp`) | A parser of expressions |
| Compile and run | Passes in the process, the execution engine (`tools/mdir/Run.cpp`) | MLIR libraries, linked into the tool |
| Output | `mdrt.host_call` to functions of the driver that the execution engine registers: `mdrtWriteEnergies`, `mdrtWriteFrame`, `mdrtWriteCheckpoint`, `mdrtWriteMinimization`, and those of the files of columns, `mdrtWritePull`, `mdrtWriteFreeEnergy`, `mdrtWriteObservables` (`Output.cpp`) | |
| Writers | Trajectories in DCD and XTC (`Trajectory.cpp`, D141) and in H5MD (`H5MD.cpp`, D239); checkpoints in H5MD (`Checkpoint.cpp`) | HDF5 for H5MD |
| Initial velocities | From `temperature` and `seed`: normal numbers from xoshiro256** seeded by splitmix64, without the components along the constraints and with the center of mass at rest, scaled to the temperature (`System.cpp`) | |

### 2.5 What the driver builds

`mdir emit control.toml` prints the module that the driver builds, as
MLIR's printer prints it once the builder's text is parsed (#568: the SSA
names, the numbers, and the layout are the printer's, and the text holds
no comments and no type aliases of the builder; the locations that the
builder gives, the terms of custom expressions, stand after their ops as
before),
`mdir emit control.toml --stage=lowered` the module that is executed, and
`--stage=pipeline` the passes between them.

| Part of the module | From |
|---|---|
| `md.potential @energy` | `[energy]`: one `md.sum_relation` for each pair term, with the truncation and the cutoff |
| `dyn.program @step` | `integrator`, with the constraints, the virtual sites, and the restraints. A second program, `@step_energy`, returns the energy and the virial as well, for the last step before an output. With the barostat of Trotter type, `@step_virial`, `@step_trotter`, and `@step_trotter_energy` take the last two steps of a period (dynamics.md, Section 2.4). |
| `func.func @mdir_run` | The schedule: the loops, the calls that write, and the buffers of the state |

The entry function takes the buffers of the positions, the velocities, the
masses, the fields of the parameters, the tables of the parameters of
pairs of types, the members of each tuple set, and the numbers of the
particles, then the edge lengths of the cell, the time step, and the step
to start from. The cell and the time step are values of the run, not of
the program (P1).

With leapfrog, the stored velocities are half a step behind the positions.
The log has the kinetic energy of the time of a row all the same
(Section 2.3).

### 2.6 Checkpoints

A checkpoint is a file in the H5MD format [[deBuyl2014]](references.md#debuyl2014), version 1.1, with all numbers in
64 bits (D26). It holds the state as the next step needs it.

```text
/h5md                         version [1, 1]; author/name; creator/name
                              "MDIR", creator/version: from mdir run
                              "<version> (<commit>)", "<version>
                              (<commit>, with changes)" from a tree with
                              uncommitted changes; from Python "<version>"
/particles/all/box            dimension, boundary ("periodic" or "none"),
                              edges: three, or the 3x3 matrix of a
                              triclinic cell, a row per vector (nm)
/particles/all/position       step, time, value (1, N, 3)   nm
/particles/all/velocity       step, time, value (1, N, 3)   nm ps-1
/particles/all/force          step, time, value (1, N, 3)   kJ mol-1 nm-1
                              only where the state has forces
/particles/all/id             the numbers of the particles, from 0
/particles/all/species        the types of the particles
/particles/all/mass                                         u
/parameters/mdir              attributes:
    format                    1, the format (D173)
    state_sha256              the hash of the state, checked on reading
    integrator                VELOCITY_VERLET, LEAPFROG, BROWNIAN, or
                              MINIMIZATION
    velocity_offset           the time of the velocities after that of
                              the positions, in steps (-0.5 with leapfrog)
    precision                 single, mixed, or double
    timestep, seed
    first_step, part, outputs_part, trajectory, frames, bath
                              the run that wrote it (D129, D130, D149)
    periodic                  1, or 0 for a run without a periodic cell
                              (D142)
    extras_sha256, front_end, model_sha256, plan_sha256
                              the additional entries (D223): their hash,
                              the front end that wrote the file, and the
                              hashes of the model and of the plan
/parameters/mdir/barostat_state    9 numbers, with a barostat that
                              scales every step (D92, D119)
/parameters/mdir/thermostat_state  the positions, then the velocities,
                              of a Nose-Hoover chain (D163a)
/parameters/mdir/fingerprint  physics, coupling, execution: a string each,
                              a line per entry, "<name>\t<value>"
                              (D172)
/parameters/mdir/free_energy_lambda  with [free_energy], the components of
                              the state of the run, with the attributes
                              free_energy (the states in one line) and
                              free_energy_state (D161)
/parameters/mdir/tunables     with tunables (D223): declarations, names,
                              values<k> for the k-th name, and history
                              (pairs of a step and a version); the
                              attribute version
```

This layout is the contract of release 0.1.0, format 1
(D173). A reader checks `format`: a newer one is refused
as written by a newer MDIR, and a file of a development build before the
release, without the fingerprint and the hash, is refused. A later format
comes with a function that converts a file of the format before, so that
every released format stays readable. `state_sha256` is SHA-256, in a
fixed order, of the step and the time, the positions, velocities, forces,
masses, and species, the cell, the attributes of `/parameters/mdir` from
`integrator` to `periodic` above, the states of the barostat and the thermostat,
and the three texts of the fingerprint; it leaves out `/h5md`, the ids,
`free_energy`, `free_energy_state`, and `free_energy_lambda`, and the
entries of D223. `extras_sha256` is SHA-256 of those entries (the front
end, the hashes of the model and the plan, and the tunables), apart from
`state_sha256` so that a reader of release 0.1.0, which does not know them,
still reads the file. A reader checks both: a file whose state or entries
changed after it was written is refused, with a pointer to `.prev`.

The units are those inside MDIR and are written with the data: nm, ps, u,
and kJ/mol.

| Rule | Reason |
|---|---|
| A value of 32 bits is stored in 64. | The conversion is exact in both directions, so a run in single precision continues without loss. |
| With velocity Verlet the checkpoint holds the forces. | A step begins with the forces of the step before. Forces that are computed again from the positions differ in their last bits, because a neighbor structure that is built again has another order. |
| With leapfrog the time of the velocities is half a step before that of the positions. | The file says what it holds. |
| Neighbor structures start empty after every checkpoint (R1). | The run that continues builds its structure at the first step. The run that was not interrupted must build there too. |
| The file appears under its name only when it is complete and on stable storage: it is written as `.partial`, flushed and `fsync`ed, renamed, and the directory is `fsync`ed (D173). | A run that ends while it writes leaves the checkpoint before; a crash on a file system that delays its writes (ext4 without `auto_da_alloc`, Lustre, NFS) does not leave an empty file under the name. |
| `mdir run` writes the file on a thread of its own while the steps go on (D336, #548). The call of the host copies the state as the program holds it, the thread puts it into the order of the input, hashes it, and writes it as above, with the library's mutex around its part in the library (#262). The next checkpoint, a stop on a signal or the wall time, the end of the run, and the exit of the process wait for the write; a write that failed stops the run at the first of them, with its error and the status 1. | The order, the hash, the file, and its syncs took 0.47 s of each checkpoint of STMV between the steps (49, 311, 115, and under 1 ms); the steps no longer wait for them, and the call takes 14 ms. A stop still comes once the file is on stable storage. |
| It records what defined the run: its fingerprint (D172). | A run that takes it compares, as the next table says. |
| The checkpoint before stays as `<checkpoint>.prev`, a second name made before the rename (D132). | A checkpoint that is damaged after it was written leaves one to go back to; the name of the checkpoint holds a complete state at every moment. |
| It records the step that its run began at, its part, the trajectory and the frames written to it, and the energy that the coupling has taken. | `mdir run --continue` continues the run to its `steps`, its trajectory, and its conserved energy (Section 2.7). |
| With `[free_energy]` it records the states and the state of its run (D161), which the fingerprint holds as well (D172). | `mdir run --continue` refuses another state or other states, naming the change; a run that begins from it as `checkpoint` of `[input]` at another state computes its forces anew rather than begin with those of another energy. |
| The particles are in the order of the input, whatever order the run keeps them in. | The file does not depend on the plan of the run. The run that continues puts the particles in order where it begins, and arrives at the order of the run that was not interrupted (D44). |

A run that continues from a checkpoint arrives at the state of the run that
was not interrupted, bit for bit, in every precision mode and for both
integrators; the tests compare the states. On the CPU this holds for any
number of threads: the sums of the parallel loops (the energies, the
virial, the kinetic energy that the coupling reads) are added over a fixed
partition into chunks and in a fixed order (D171), so a
run gives the same bits from run to run and with any number of threads. On
a GPU it holds in the deterministic mode. Without it, the sums that a
device adds with floating-point atomics (the charges of PME, the loops over
groups, D84) can differ in their last bits from run to run, and a run with
a thermostat or a barostat carries the difference into its state. The two
runs must have the same `checkpoint_interval`.

**The fingerprint** (D172) lists what defined the
run, an entry each, in three groups:

| Group | Entries |
|---|---|
| physics | Every key of `[energy]`, `[pme]`, `[lj_pme]`, `[constraints]`, `[restraints]`, `[boundary]`, and `[free_energy]` as the control file writes it, in a canonical form (keys sorted, a floating-point number as the shortest text of its double, so that `9` and `9.0` are one value, and an integer exactly: below $2^{53}$ in magnitude as the text of its double, from $2^{53}$ on with all its digits, D347, and a value that names a choice in its documented spelling, #702); SHA-256 of the contents of the files of the topology (Amber, GROMACS with its included files, CHARMM structure and parameters), of the masses, and of the reference of positional restraints |
| coupling | Every key of `[dynamics]` but `steps`, and of `[ensemble]`, `[thermostat]`, and `[barostat]` |
| execution | Every key of `[execution]`, and `pairlist_distance`, `pruned_distance`, and `rebuild_interval` of `[energy]`, which decide how the forces are found and not what they are |

A value longer than 160 characters is recorded as its SHA-256. The
entries are the keys as written, not the values the driver resolves: a
key added with its default value counts as a change. `mdir checkpoint
--print=fingerprint file.h5` lists them.

A value that names a choice, which the reader reads without regard to
case (Section 1), is recorded in the spelling that the reader documents
for it, whatever its case in the file (#702): upper case
(`"VELOCITY_VERLET"`, `"AUTO"`, `"CPU"`), but for the rules of `mixing` of
`[[energy.pair]]`, which are in lower case (`"lorentz-berthelot"`). So
`integrator = "velocity_verlet"` continues the run of
`integrator = "VELOCITY_VERLET"`, and `pairlist_distance = "auto"` that of
`"AUTO"` without a note. Which values are choices the reader says: it
records the place in the file of each value that it reads as a choice
(`Control::choiceSpellings`), and the fingerprint, which reads the same
file again, takes the spelling of the values at those places, nested ones
(`scaling` of `[[energy.external]]`, a rule of `mixing`) included. Any
other string, such as a name, a path, an expression, or a selection, is
recorded as written, and a change of its case is a change. The fingerprint
of a file that writes its choices in their documented spelling is the text
of before #702, so its checkpoints continue as before; the format of the
checkpoint (format 1, D173) does not change, since it holds the text of the
entries and no version of their form. Before #702 a choice in another
case, such as `"mixed"` of `precision`, was recorded as written. Such a
checkpoint is still taken by the file that wrote it: an entry matches when
the checkpoint holds either its documented spelling or its text as the file
writes it. The checkpoints that the run then writes hold the documented
spelling, which any case of the choice continues.

| A run that takes a checkpoint | Physics or coupling differ | Execution differs |
|---|---|---|
| `mdir run --continue`, its own | Refused; each changed entry is named, with its value in the checkpoint and in the control file | Continues, with a note in the log; its bits follow the new settings |
| `checkpoint` of `[input]`, another run's | Begins with a note that names each difference, and evaluates the forces and the state of the barostat at its first step; a Nose-Hoover chain begins at rest if `[thermostat]` or `[ensemble]` differ | Takes the forces, the barostat state, and the chain |

Where nothing but the execution differs, a run from another run's
checkpoint continues it bit for bit, as the stages of a pipeline that
change no physics do; where the physics differs, as from equilibration
with restraints to production, the first step takes the forces of the new
physics, not those that the checkpoint stored. Forces evaluated again of
the same physics are those of the checkpoint, bit for bit: the neighbor
structures start empty after every checkpoint (R1).

A run cannot continue with another integrator: the velocities of the two
are not of the same time.

`mdir checkpoint file.h5` describes a checkpoint, with its format and the
version of MDIR that wrote it, and `mdir checkpoint first.h5 second.h5`
compares the states of two. `mdir checkpoint
--print=positions|velocities|forces file.h5` writes a line for each
particle in the order of the input: its number, its mass, and the three
numbers of the field (nm, nm/ps, kJ/mol/nm); `--print=fingerprint` writes
the entries of the fingerprint, a line each.

### 2.7 Runs longer than a job

A run on a cluster outlasts the wall time of a job. The same command line
starts it and continues it until it is complete:

```sh
mdir run --continue --max-walltime 23:50 md.toml
```

| Option | What it does |
|---|---|
| `--continue` | Continues the run from the checkpoint of `[output]` until it has taken `steps` steps from the step it began at (D129). Without a checkpoint the run begins; with one that holds the last step it says that the run is complete and exits with 0. It refuses a checkpoint of other physics or coupling, naming each change (Section 2.6), and steps that remain if they are not whole intervals of the outputs and of the coupling. Raising `steps` extends a run. |
| `--no-append` | With `--continue`, writes the outputs that follow, the log, manifest, files of columns, and frames, to `<name>.partNNNN<ext>`, NNNN the part of the run; later continuations append to the files of that part, which the checkpoint records. Without it, the outputs are appended to the files of the run after what was written past the checkpoint is removed (D130, D149). A run without `--continue` keeps the outputs of an earlier run as `#<name>.<n>#` (Section 2.8). |
| `--max-walltime <time>` | Stops at the last checkpoint that leaves time, within `<time>` from the start of `mdir`, for one more interval between checkpoints as long as the longest so far (D131). In hours (`23.5`) or as `H:MM[:SS]`. The stop waits for the write of its checkpoint, which the interval does not hold since the write runs beside the steps (D336): under a second on STMV. |

SIGTERM and SIGINT ask a run of dynamics that writes checkpoints to stop
at its next checkpoint; a second signal of the same kind ends it at once
(D131). A stop is taken after a checkpoint is written, so the run
continues from it exactly; a scheduler's warning signal (for SLURM,
`--signal=TERM@<seconds>`) must leave time for an interval between
checkpoints.

| Exit status | Meaning |
|---|---|
| 0 | The run is complete |
| 75 | The run stopped at a checkpoint before its last step, on a signal or at the wall time; `--continue` goes on |
| 1 | An error, which the standard error describes |

A run that begins from the checkpoint of another run, as the stages of
`examples/` do, is a new run that begins at the step of that checkpoint
(D129): its step and its time continue, so that the random numbers of the
stages differ, and its `steps` count from there. The checkpoint of the
run it began from is not changed. Where its physics or coupling differ
from that run's, it evaluates the forces of its first step
(D172, Section 2.6).

### 2.8 The outputs of a run

A run writes the files that `[output]` names, at the interval given
there, and nothing else (D149). The names are relative to the control
file. The checkpoint is the only file that a run continues or begins from;
there is no restart file besides it.

| Output | Keyword | Written | Form |
|---|---|---|---|
| The log | always the standard output; with `log`, a file as well | its rows every `energy_interval` steps, and the messages of the run (Section 2.3) | text |
| The energies | `energy` | every `energy_interval` steps: the rows of the log | columns |
| The terms over centers | `pull` | every `energy_interval` steps (D145) | columns |
| dH/dλ and the energies of the states | `free_energy` | every `energy_interval` steps (D161) | columns |
| The observed terms and their derivatives | `observables` | every `energy_interval` steps (D189) | columns |
| The trajectory | `trajectory`, `trajectory_format`, `trajectory_precision`, `trajectory_strings` | every `trajectory_interval` steps | DCD or XTC (D141), or H5MD without loss (D239, [python-h5md.md](python-h5md.md)) |
| The manifest | `manifest` | at execution start and end | JSON Lines (D168) |
| The checkpoint | `checkpoint` | every `checkpoint_interval` steps, and at the end of a minimization; the one before as `<checkpoint>.prev` (D132) | H5MD (Section 2.6) |

The manifest has no step interval. The other intervals are those of the compiled schedule (Section 2.2): the files
of columns take the interval of the rows of the log, at whose steps the
energies are computed, and the frames and the checkpoints come at
multiples of it.

**Run manifest (D168).** Optional `manifest = "run.jsonl"` in `[output]`
writes UTF-8 JSON Lines, one object per line, with `schema_version = 1`.
It records executions, not trajectory frames. A `start` event precedes
execution and an `end` event records `completed` or `stopped`; a start
without an end means completion was not recorded (for example, a crash
or a forced kill). Each event has an `invocation` number local to the file.
The [JSON schema](run-manifest.schema.json) defines every event. The start
records the build (version, git commit, dirty flag, LLVM/MLIR and CUDA
toolkit versions), SHA-256 and byte size of each input, target, precision,
threads, output paths, starting and requested ending steps, and the
checkpoint part. Inputs include the selected restart checkpoint, CHARMM
parameter files, and GROMACS includes actually read; inactive includes
are absent. Input paths are absolute, with filesystem aliases resolved.
The manifest path cannot alias an input or another output, even through
a directory symlink when the output file does not yet exist. No hostname
is recorded.

`effective` gives the seed as a decimal string (preserving all 64 bits),
time step in ps, resolved trajectory format (null without frames), the
requested neighbor structure (`auto`, `matrix`, or `groups`), the
structure it resolved to for the system and why, and the reaches of the
lists in Å (D261), the kinds observed immediately before
lowering, the program's state/force/mass/parameter buffer types, and PME's
grid, spline order, and beta in inverse Å (null without PME). The actual
pass pipeline and `system.warnings` are recorded. A custom `MDIR_PIPELINE`
that omits the normal lowering passes may leave the observed neighbor
kinds empty; the buffer types describe the driver's program interface.

`device` is null on CPU. On CUDA it gives the visible device index, name,
UUID in hexadecimal, compute capability, and driver API version. MDIR
uses the driver API; `runtime_api_version` is null unless a CUDA runtime
API is actually loaded. API versions use CUDA's integer representation
(e.g. 13040 for 13.4). UTC timestamps use the wall clock;
`compile_seconds` and `elapsed_seconds` use monotonic clocks, with elapsed
time measured from the start record through completion or a checkpoint
stop, excluding compilation. The terminal `step` is the completed step,
or the last reported minimization step. `reason` is empty on completion,
or the stop reason (`SIGTERM`, `SIGINT`, or `the wall time`). Input hashes
are captured before compilation; input files must remain unchanged while
the driver reads them.

A fresh run uses the same numbered backups as other outputs. `--continue`
appends execution history without cutting it to the checkpoint, including
when no checkpoint exists yet; `--no-append` selects the part filename.
An already complete run leaves the manifest untouched. A malformed or
unsupported existing manifest is refused before appending; use a new
manifest path to retain a damaged file. The manifest is disabled when its
key is omitted; `mdir check` reports its path without creating it, and
`mdir emit` creates no manifest. An execution that fails before the start
record is written has no record. A manifest is provenance, not a restart
compatibility check or a copy of its inputs.

**Files of columns.** The energies and the terms over centers share one
form, which a program reads without knowing the run:

```text
# step time total potential kinetic temperature virial pressure
# - ps kcal/mol kcal/mol kcal/mol K kcal/mol atm
0 0.000000 -834.146700 -1083.238900 249.092200 96.853800 -70.651300 232.387500
```

The first line names the columns, the second gives the unit of each, `-`
for none, and every row after them is one output, its step an integer and
the other values with six decimals, separated by single spaces. The names
are in lower case; a quantity of a term is `<term>.<quantity>`. The file
of the energies has the columns of the log: `step time total potential
kinetic temperature virial`, then `pressure` with a periodic cell,
`conserved` with a coupling, and `volume area_xy` with a barostat; that of a
minimization `step potential rms_force max_force max_atom step_size`.

**Continuation.** `mdir run --continue` continues the outputs of its run
from the step `s` of its checkpoint, so that the files are those of a run
that was not interrupted:

| Output | A continued run |
|---|---|
| Files of columns | Keeps the rows up to step `s` and appends; the first two lines must be those that the run writes, or it stops before it begins |
| Trajectory | Keeps the frames that the checkpoint counts and appends (D130) |
| Log file | Appends to the whole file, which records what happened, the steps past `s` that are run again included, after the line `MDIR: continues the run after step s` |
| Checkpoint | Replaced at the next interval, as always |
| Manifest | Appends execution history, including attempts past the checkpoint; never cuts history to a simulation step |

With `--no-append` the log, files of columns, trajectory, and manifest go to `<name>.partNNNN<ext>`
instead, NNNN the part of the run, and later continuations append to the
files of that part, which the checkpoint records (`outputs_part`, 0 for
the names of the control file).

**Backups.** `mdir run` without `--continue` keeps the outputs of an
earlier run that has the same names: before it writes, it renames each
file that exists (log, manifest, energies, terms over centers, trajectory,
checkpoint, and the checkpoint before the last, `.prev`) to
`#<name>.<n>#` in its directory, n the least number from 1 that no file
takes, and the log says so (`MDIR: backed up 'md.log' as '#md.log.1#'`).
It keeps at most 99 backups of a file; when a file has them all, the run
stops before it writes or moves any file. Under `--continue` the files
belong to the run: they are continued from its checkpoint, or written
anew when the run begins without one, as after a job that stopped before
its first checkpoint, and none is backed up. The manifest preserves the
history of those attempts by appending even without a checkpoint. Two outputs may not have one
name, nor an output the name of an input.

**`mdir check`** (D151, Section 1.4) lists these outputs: each file with
its format, its interval, and the number of rows, frames, or checkpoints
of the whole run, and notes on the standard error the name under which a
run keeps each file that exists.

```text
outputs:
  log: stdout and md.log (text), every 5000 steps, 101 rows
  energy: md.energy (columns), every 5000 steps, 101 rows
  pull: not configured (columns), disabled
  free_energy: not configured (columns), disabled
  trajectory: md.xtc (XTC), every 5000 steps, 100 frames, exists
  checkpoint: md.h5 (H5MD), every 50000 steps, 10 checkpoints
mdir: note: trajectory output 'md.xtc' exists; mdir run keeps it as '#md.xtc.1#' before it writes its own, and mdir run --continue continues the run of its checkpoint instead
```

### 2.9 The command line

One program with subcommands (D59). The programs of M0 become
subcommands; `mdir-opt` stays a program for developers.

| Command | Does |
|---|---|
| `mdir run control.toml` | Reads the input, compiles the run, and runs it (`mdir-run` of M0). `--continue`, `--no-append`, and `--max-walltime` carry a run over more than one job (D129 to D132, Section 2.7) |
| `mdir template md` | Prints a control file with every keyword |
| `mdir check control.toml` | Reads the input and prints what it found: particles, types, the mass and the density, the cell, the degrees of freedom, the integrator and the target; with a topology also the tuples of each kind, the charges and their sum. Compiles nothing, so it gives no energies: a run of zero steps does. |
| `mdir emit control.toml --stage=<module\|lowered>` | Prints the program of the run, as `--emit` does in M0 |
| `mdir emit control.toml --segments` | Prints instead the program of segments that a Python simulation of the same model runs (D215): with the branches of the start and the loop over the parts, built from the control of a simulation (`compiler::getSegmentsControl`); with `--stage`, as above (#568) |
| `mdir checkpoint <file>` | Prints what a checkpoint holds (`mdir-checkpoint` of M0) |
| `mdir version` | The version, the LLVM it was built with, the targets it has |

Since M1 (described in Sections 1.3 to 1.5 and in
[debugging.md](debugging.md)): `mdir template` also prints the four
stages of a pipeline, `minimize`, `nvt`, `npt`, `production` (D147);
`mdir check` reports the planned run and its outputs, with `--json`
(D151); `mdir doctor` checks the installation (D155); and
`mdir bug-report` gathers what a report of a defect needs.

### 2.10 The readers of topologies

| Format | Files | Keywords of `[input]` |
|---|---|---|
| Amber | `prmtop` (topology), `inpcrd` or `rst7` (positions, velocities, and the cell) | `topology`, `coordinates` |
| GROMACS | `.top` with `.itp` (topology), `.gro` (positions, velocities, and the cell) | `topology`, `coordinates` |
| CHARMM (since M1, D122) | `.psf` (topology), `.crd` (positions; the cell from `[boundary]`), files of topology, parameters, and streams ([charmm.md](charmm.md)) | `topology`, `coordinates`, `parameters` |

| Item | Proposal |
|---|---|
| The readers | MDIR has readers of its own. They accept what the engines accept and share no code with them. The engines and their tools are under the GPL and the LGPL; their behavior is learned from their documentation and their code, and written down as a specification, from which the readers are written. |
| What a reader hands on | A description of the system that does not depend on the format: particles, types, tuple sets with their parameters, the table of pairs of types, exclusions, pairs three bonds apart, constraints, the cell. A format is a reader more. |
| Units and forms | A reader converts to the units and the forms of [conventions.md](conventions.md) (D62): for Amber, twice the force constants of bonds and angles, the charges divided by 18.2223, and $\sigma$ and $\varepsilon$ of Lennard-Jones from `ACOEF` and `BCOEF`. The control file stays in Å and kcal/mol (D36). |
| What is an error | What M1 cannot run: 10-12 terms, 12-6-4 terms, polarizability, virtual sites other than those of water of four sites. Each with the milestone that brings it. |
| Formats of other engines | CHARMM's files (D122); no other format yet |

The GROMACS preprocessor and sections are as in the specification of the
format; M1 reads the sections that an Amber force field in the format of
GROMACS needs: `defaults`, `atomtypes`, `nonbond_params`, `pairtypes`,
`bondtypes`, `angletypes`, `dihedraltypes`, `moleculetype`, `atoms`,
`bonds`, `pairs`, `angles`, `dihedrals`, `exclusions`, `settles`,
`system`, `molecules`.

A `.gro` file is read in fixed columns, as GROMACS reads it: the name
fields take 20 columns, and the distance between the decimal points of
the first atom's x and y sets the width of every field after them, of
the three coordinates and of the three velocities, which GROMACS writes
in fields as wide as the coordinates with one more decimal (`%8.3f` and
`%8.4f` by default). A line has either three velocities that read in such
fields or none, and every line alike; anything else, such as velocities
wider than the coordinates, is an error that names the line, where
GROMACS takes a field that does not read as 0 (#365). Before, velocities
written one column wider than the coordinates (`%12.7f`, then `%13.8f`)
kept the first component, cut to the width, and lost the other two: 81
waters written at 300 K started at 89 K.
The common finite decimal coordinate and velocity fields use a direct
binary64 conversion (D323); other numeric forms retain the
general parser, with the same accepted input and diagnostics.

A dihedral without parameters takes those of the first entry of
`[ dihedraltypes ]` of its function with the most types that are not
wildcards, matched to its atoms in the order of the file or the reverse;
functions 1 and 9 share a table, and an entry of function 9 brings the
entries of the same types on the lines that follow it. When the force
field defines `_FF_AMBER_LEAP_ATOM_REORDERING`, as amber14sb.ff and
amber19sb.ff of GROMACS do, the atoms of each dihedral are then put in the
order that LEaP gives them, as grompp of GROMACS 2026.3 does, while the
parameters stay those of the order of the file. With the types of the
matching entry, a blank for a wildcard: a dihedral whose entry has a force
constant is reversed when its first type sorts after its last, or the two
are equal and its second sorts after its third; an improper whose types,
or whose atoms, an earlier improper of the same function had, over all
molecule types in the order of the file, has the three atoms around the
third, the central one, sorted by type and then by index. The first
improper of its types keeps its order. Reversing a dihedral leaves its
angle alone, but which atoms are the outer ones of an improper changes its
angle: in ubiquitin with amber19sb.ff, the order of the file gives
impropers 0.896 kcal/mol above those of grompp, and the order of LEaP
agrees with them to 5 × 10⁻⁵ kcal/mol (`gromacs-leap-order.test`).

The functions read are those of the Amber, CHARMM, OPLS-AA, and GROMOS
force fields: bonds of function 1, and of function 2, the quartic bond of
GROMOS, a term given by an expression (D136); angles of function 1, of
function 2, the angle of GROMOS in its cosine, a term given by an
expression as well, and of function 5, whose Urey–Bradley
term ½ k_UB (r₁₃ − r13)² between the outer atoms becomes a set of pairs of
its own; dihedrals of functions 1, 4, and 9, and of function 2, the
harmonic improper ½ k (ξ − ξ0)² with ξ − ξ0 taken in [−π, π), and of
functions 3 (Ryckaert and Bellemans, $\sum_n C_n \cos^n(\varphi - \pi)$)
and 5 (Fourier), as terms given by expressions (D136); pairs of
function 1; CMAP of function 1 (D121, [charmm.md](charmm.md)). The
Lennard-Jones of a topology takes `lennard_jones_modifier` in its kernel:
`POTENTIAL_SHIFT`, `FORCE_SWITCH` (the polynomial of GROMACS), or
`POWER_FORCE_SWITCH` (VFSWITCH of CHARMM, which also switches the pairs
three bonds apart).

A topology of Amber is read in either format (D133): with `%FLAG` lines,
or in the format before Amber 7, a title, 30 pointers in lines of 12, and
the sections in a fixed order whose lengths the pointers give (a section
of no items is a blank line), then the solvent pointers, the molecules,
and the box when IFBOX is set; that format has no NUMEXTRA, no SCEE and
SCNB (1.2 and 2.0), and no atomic numbers (from the masses). The reader
takes the conventions that pmemd 26 follows, read from its source for
compatibility:

| Item | Rule |
|---|---|
| Negative periodicity | A dihedral whose type has a negative periodicity also has the term of the next type, and so on up to the first type of positive periodicity |
| Phase | A phase within $10^{-3}$ of $\pm\pi$ is $\pm\pi$; a cosine or sine of the phase below $10^{-6}$ in magnitude is 0. Topologies write $\pi$ as 3.141594 or 3.1415927; JAC's dihedrals are 965.2152 kcal/mol so and 965.2154 without |
| 1-4 pairs | From the entries with positive third and fourth indices, with SCEE and SCNB of the last type of the entry's chain; a type of periodicity 0 carries none. Two entries of the same pair, or a factor of 0, are errors, where Amber counts the pair twice or drops its Coulomb term |
| Exclusions | Without extra points those of the file; with them built again from the bonds, the ends of the angles and of every dihedral, an extra point taking those of its owner ([constraints.md](constraints.md), Section 2) |

What still differs from pmemd, by design: the cosine of an angle is not
clamped to $\pm0.999$ (pmemd evaluates an angle within 2.56° of linear at
that bound); a dihedral whose cross products are below $10^{-3}$ Å² is not
dropped; a net charge below 0.01 e is not spread over the atoms to make
the system neutral (Factor IX: $3.1\times10^{-4}$ e, below the printed
digits); water is found by its residue and atomic numbers (O, H, H), not
by the names O, H1, H2 in any order; and on an odd grid the aliasing
factor of `influence = "OPTIMAL"` takes the principal frequency at the
index $(N-1)/2$, where pmemd takes $-(N+1)/2$ ([pme.md](pme.md),
Section 1.1). The Coulomb
constant, $\beta$ from $\mathrm{erfc}(\beta r_c)$ rather than
$\mathrm{erfc}(\beta r_c)/r_c$, and the average of the correction
for the dispersion are conventions of MDIR ([nonbonded.md](nonbonded.md), Sections 1.2 and 2).

### 2.11 The equivalence check of the builder

The builder writes the module as MLIR text (`Builder.cpp`). Since step 2
of #568 the program keeps it as MLIR's printer prints it once the text is
parsed (`printParsedModule`, `lib/Driver/ProgramModule.cpp`, called by
`buildProgram`), with the locations of its ops, and `mdir emit
--stage=module` and Python's `Program.ir` print it for reading
(`printModuleForReading`); before, they printed the builder's text itself. #568 moves the builder to IR built with
`OpBuilder`, in stages, and no stage may change the generated program.
`scripts/builder-equivalence/check.py` shows this for two builds of the
driver, over a fixed corpus. It is the gate of every later stage of #568.

Construction and canonical printing use the same accepted program dialect set
as lowering: `insertProgramDialects` in `lib/Dialect/Program/ProgramDialects.cpp`.
This includes tensor/linalg staged-model functions and lowering-only dialects.
Valid staged input therefore reaches canonical printing and the loop-carried
value rewrite, including segmented programs. D352's fallback for text that
fails to parse remains unchanged.

```sh
scripts/builder-equivalence/check.py compare OLD NEW   # two mdir programs or build trees
scripts/builder-equivalence/check.py record MDIR --output hashes.tsv
scripts/builder-equivalence/check.py verify MDIR --hashes hashes.tsv
```

The exit status is 0 when every case is identical, 1 when one differs
(each one is named, with the first lines of its difference), and 2 for an
error of the script. The report names a case by its place in the corpus,
for example `test/Driver/x.test:run.toml [gpu] module printed`, never by a
path of the machine.

**The corpus.** By default it has all of these; `--only` takes some of
them:

- `examples`: the control files of `examples/`.
- `tests`: the control files of the lit tests. These are the `*.toml`
  files under `test/` and the `*.toml` parts of the tests that split-file
  divides. The other parts of the test are written beside them, and the
  files that its RUN lines copy into `%t` (`cp %S/PATH... %t[/NAME]`) are
  linked there under the names they are copied to. Other RUN commands
  (sed, scripts, environment variables such as `MDIR_PME_SPREAD`) are not
  followed, so a control file that needs one of them stops with an error,
  the same error in both builds.
- `bench`: the control files of the systems of the prepared Amber suite,
  one level under `MDIR_BENCH_DIR` (or `--bench`). Without either, this
  kind is left out.
- `--control FILE` adds a control file of one's own.

**The cases.** Each control file is a case for each target of `--targets`
(`cpu,gpu` by default), emitted from a copy of the file with `target` set
in `[execution]`, and for each kind of program of `--kinds`: `run`, the
program of the run, and `segments`, the program of segments of a Python
simulation (`mdir emit --segments`), which alone has the branches of the
start and the loop over the parts (both by default; `--kinds run` for a
build of before the option). A case runs in a scratch directory that mirrors the
directories above the control file with links, so a relative input
resolves as it would beside the file. No case writes into the corpus.
`mdir emit` uses no device: the module is printed before anything is
compiled, and the GPU target changes only the text. A target that the
control file does not allow (a model or a neighbor structure that only
runs on a GPU, with `target = "CPU"`) is a case that stops with an error.

**What is compared.** The exit status, the standard output, and the
standard error of each case count; the path of the scratch directory is
replaced by `<corpus>`. A case that stops with an error is therefore
identical only if both builds stop with the same message. `--forms`
selects the forms of the module stage, both by default:

- `raw`: the bytes that `mdir emit --stage=module` prints. Until step 2
  of #568 that was the builder's own text, with its SSA names, its type
  aliases, its comments, and its formats of numbers; since, it is the
  printed module, so the two forms agree.
- `printed`: the module after a round trip through `mdir-opt` (by default
  the one beside NEW; `--printer`) with MLIR's default printer. The
  printer numbers the SSA values itself and prints no comments, no
  aliases of the text, and no locations.

**Locations.** The program holds two kinds, and both are part of it:

- The locations that the builder gives. The ops of a custom expression
  carry the name of their term (`loc("control-file term: wall")`,
  `Expression.cpp`), which the diagnostics of the passes name
  (`external-terms.test`). Emit prints them after their ops, as the
  builder's text had them.
- The positions of the ops in the builder's text (`loc("-":2451:5)`). The
  program's text keeps them, and emit leaves them out, since they refer
  to a text that nobody reads. Until D354 the lowering named a
  kernel that it outlines from a loop by the line of the loop
  (`mdir_run_particle_for_l2451_...`), so the positions were part of the
  lowered program; since, a kernel is named after its op and its ordinal,
  and the positions bear on neither the module nor the lowered program
  that the check compares. A stage that builds a loop with `OpBuilder`
  gives it no position of the text.

The `printed` form leaves out every location. Step 1 of #568 said that
locations were not part of the contract; that was wrong for the
builder's own locations, which diagnostics name, and is corrected here.

`--stages module,lowered` adds the program after the passes, as
`mdir emit --stage=lowered` prints it, with the kernels of the GPU target.
That needs the CUDA toolkit but no device. MLIR's printer prints it, so it
has the one form `printed`, compared whatever `--forms` says (before step
2 of #568, `--forms printed` left it out).

**What each stage keeps.** Step 1 of #568, which adds the check, keeps
both forms, and the lowered program as well. A stage that builds ops with
`OpBuilder` instead of text changes the raw text (names, aliases, numbers)
but must keep the printed form and the lowered program identical. It runs
`compare` (or `verify` against hashes recorded with main) with
`--forms printed --stages module,lowered`, and its PR gives the counts. A
file of hashes depends on the corpus: record it and verify against it on
one machine, with one `MDIR_BENCH_DIR`.

**What step 2 has moved from text to IR so far.**
- The emitters write `func.call`; no text replaces `call` by it when a
  chunk moves into a loop.
- The loops of a program of segments that carry values into code written
  outside them: the branches of the start (`Builder::emitStartBranch`) and
  the loop over the parts (`Builder::emitPartLoop`). The builder writes
  the code as it is, inside a loop marked `mdir.carries`, and once the
  module is parsed, the uses of the values that the loop carries become
  its arguments (`carryLoopValues`, `ProgramModule.cpp`); in the loop over
  the parts, a cast marked `mdir.part_start` makes the uses of `%start`
  those of the step that the part begins after. Before, the builder
  renamed them in its text (`%x0` to `%x0_a` and `%x0$part`, the values of
  the start to `_s`, `%start` to `%start$part`) with `renameValue`, which
  is gone.

On main at 98e41261, a separate build of main's sources compared with a
second build on the CPU of the login node, with the Amber suite prepared,
gave these results:

- The corpus had 234 control files: 13 of `examples`, 180 of `tests`, and
  41 of `bench`. That is 468 cases.
- All 468 cases were identical in the module stage (raw and printed) and
  the lowered stage.
- 252 cases emit a program. The other 216 stop with the same error in
  both builds: test inputs made by other RUN commands, examples whose
  inputs are prepared first or that continue a checkpoint, intentional
  errors of the tests, and targets that the file does not allow.
- With 8 cases at a time, the module and lowered stages took 3.7 minutes.

`test/Lit/builder-equivalence.test` checks the checker on a small corpus.
One build compared with itself is identical, including a control file
that stops with an error. Then two wrappers of it perturb the module: a
comment, which only the raw form sees, and a different cutoff, which both
forms see. Recorded hashes are verified with the build and with the
second wrapper.

### 2.12 The files of the builder

`lib/Driver/Builder.cpp` held the whole builder, 12,500 lines, and most
pull requests edited the same few functions of it (#568: `emitEntry`,
`emitPrograms`, `emitLevel`, `emitTopologyPotential`, `emitCoupling`, and
`build`). Step 3 of #568 splits it by responsibility, in two parts.

**Part 1, the files.** The class `Builder` and the helpers that more than
one part takes move to `lib/Driver/BuilderImpl.h`, in the namespace
`mdir::driver::builder`; the definitions move, unchanged, to one file for
each responsibility:

| File | What it builds |
|---|---|
| `Builder.cpp` | `build`, the order of the parts, and the entry points of `mdir/Driver/Builder.h` |
| `BuilderParameters.cpp` | The fields, tables, and constants of the program: the topology, the dispersion and its tails, PME, LJPME, the prolate splitting, the meshes, the tunable gradients |
| `BuilderPotential.cpp` | `md.potential @energy` and the other potentials: the terms of pairs, of tuples, and of the topology, the Born and external terms, the centroids, the pulls, the observables, the staged models, the restraints |
| `BuilderDynamics.cpp` | The programs of a step (`emitPrograms`), the kinetic energies, the noise, the couplings and the barostat, and the minimization |
| `BuilderConstraints.cpp` | SETTLE, SHAKE and RATTLE, the virtual sites, and the tests and stops of the constraints |
| `BuilderSchedule.cpp` | `@mdir_run`: the levels of the loops, the branches of the start, the parts of a program of segments, and the frames |
| `BuilderOutput.cpp` | The outputs of the pulls, the free energy, the observables, the tunable gradients, and the solvent |

The program that the builder writes does not change: the printed module
and the lowered program of the corpus of Section 2.11 are those of
before. Since the names of kernels no longer carry lines of the builder
(D354), moving code changes no name.

**Part 2, registration and all term kinds (D359, D361, D362, #683).**
The central assembly sites iterate registered adapters: parameter arrays,
shared potential contributions, independent energy columns, entry layouts,
and reporting. They no longer branch on the kind of a term. Every kind has
an explicit factory entry in `registerTermBuilders()`; each build owns its
adapter instances. Explicit references retain static-library reachability
under section garbage collection. There is no self-registration or mutable
process-wide term state.

The moved kinds are externals, pulls, restraints, generalized Born with its
surface contribution, custom tuples/centroids/compound terms, custom pairs,
triplets, staged/opaque models, bonds, angles, dihedrals, Urey–Bradley,
harmonic impropers, special 1–4 pairs, CMAP, and nonbonded terms.
`TermBorn.cpp` keeps the computed Born radii private because the surface
energy consumes them. `TermModel.cpp` privately retains imported bodies,
result names, cutoffs, and pair-list mode for one build. The bonded families
have one registry entry and one source file each. `TermNonbonded.cpp` owns
primary, excluded, reciprocal, and within-selection LJ/Coulomb emission,
PME/prolate/LJPME preparation, constant corrections, and their tunable rules.
Ordinary and special 1–4 LJ pairs share the unchanged scalar emitter in
`TermLennardJones.cpp`.

Read-only hooks take `TermInputs`: physical inputs, bounded derived layout
facts, and deterministic names. `TermParameters` adds named parameter
collections and scalar update services; `TermEmission` adds the output stream
and named shared emitters. An adapter never receives `Builder&`, the complete
mutable program, or friendship. Narrow roles hold operations needed by only
one kind. The nonbonded role borrows `TermNonbondedData`, an explicit reference
projection of mesh/correction values and term data collections. It exposes no
entry, scheduling, run-state, or complete-program access. Its private per-build
binding supplies those same values to the shared long-range correction hooks.

`TermPotentialAddition` adds a named contribution or an explicitly named SSA
result. `ObservedSelection` carries the complete potential selection from its
adapter, without a central dispatch on observed-term kind. `TermDefinition`
records supply both independent potential declarations and energy-output
columns, retaining all 14 built-in columns, including empty ones. Stored
restraint energy remains last and is not a new potential evaluation. Shared
field/table/tuple lists supply entry arguments and buffers; no generic entry
hook is needed. Restraint correction still precedes hosted model evaluation
at the same force/energy/virial sites.

Ordering is explicit:

| Site | Order |
|---|---|
| Particle fields (`parameterRank`) | Nonbonded charges/types -4, model atomic numbers -3, Born/surface -2, restraints -1, externals 0, custom-pair fields 1 |
| Tables | Nonbonded flags, LJ tables and gradient seeds, then custom-pair tunable tables |
| Tuple collections (`getTupleRank`) | Rank -1: bonds, angles, Urey–Bradley, dihedrals, impropers; rank 0: custom tuples; common zero/charge-gradient fields; rank 1: 1–4 pairs; common constraints/sites/anchors; rank 2: CMAP; rank 3: exclusions and within-selection pairs |
| Shared potential (`topologyRank`, then `potentialRank`) | Rank 0: custom pairs (-2), Born/surface (-1), externals (0); rank 1: nonbonded; rank 2: bonds, angles, Urey–Bradley, dihedrals, impropers; rank 3: custom tuples; rank 4: 1–4 pairs, CMAP; rank 5: staged models |
| Independent definitions/output (`TermDefinition`) | Built-in ranks 0–13 in their original column order; groups 14–17: custom tuples, custom pairs, Born/surface, externals; original index orders each variable-length group |
| Bare collection/emission | Pair descriptors before triplet descriptors, then bare dispersion validation; contribution ranks -2 for pairs and 1 for triplets |

Equal ranks retain explicit registry order: externals, pulls, restraints,
Born/surface, tuples, pairs, triplets, models, bonds, angles, Urey–Bradley,
dihedrals, impropers, 1–4 pairs, CMAP, nonbonded. This preserves field and
potential order, sum names, inlining, and loop fusion. Model text remains in
the shared potential; no public or opaque call boundary is introduced.

The complete hook inventory below names active implementations. Defaults
are empty; ordering metadata and masks are counted separately from emission.
No generic behavioral hook has only one implementer.

| Generic hook or metadata query | Implementers |
|---|---|
| `getRoles` | Every adapter; role bindings are explicit metadata |
| `getPotentialMask` | Shared static metadata for bonded families, Born/surface, and externals; specialized dependent/alchemical selection in tuples, custom pairs, 1–4 pairs, nonbonded |
| `parameterRank` | Nonbonded, model, Born/surface, restraints, custom pairs; others use stable rank 0 |
| `potentialRank` | Tuples, custom pairs, Born/surface, triplets; others use stable rank 0 |
| `topologyRank` | Nonbonded, all seven bonded adapters, tuples, models; early adapters use rank 0 |
| `getTupleRank` | Explicit metadata in the bonded/nonbonded adapters; custom tuples use rank 0 |
| `getPairExclusion` | Nonbonded relation-name metadata; consumed by the shared neighborhood emitter |
| `collectBareParameters`, `emitBareContributions` | Custom pairs, triplets |
| `collectParameters` | Externals, restraints, Born/surface, custom pairs, models, nonbonded |
| `collectTopologyTables` | Custom pairs, nonbonded |
| `collectTupleSets` | Custom tuples, all seven bonded adapters, nonbonded |
| `refillArrays` | Custom tuples, custom pairs, 1–4 pairs, nonbonded |
| `collectTunableOutcome` | Custom tuples, custom pairs |
| `collectDispersionConstants`, `collectGradientShiftConstants` | Custom pairs, nonbonded; primary terms precede custom terms |
| `rejectsTunables` | Pulls, nonbonded; the existing diagnostic is unchanged |
| `needsPairNeighborhood` | Custom pairs, nonbonded |
| `emitTopologyPotential` | Externals, Born/surface, custom tuples, custom pairs, models, all seven bonded adapters; nonbonded uses its narrow role at rank 1 |
| `appendTermDefinitions` | Externals, Born/surface, custom tuples, custom pairs, all seven bonded adapters, nonbonded |
| `findObservedTerm`, `getObservedValue` | Externals, custom tuples, custom pairs |

| Narrow role and its complete hooks | Implementer |
|---|---|
| `TermPreparation`: `validate`, `prepare` | Models; read-only inputs and a stream |
| `TermHostEvaluation`: `applies`, `emitDeclarations`, `emit` | Opaque models; borrowed evaluation names |
| `TermPostEvaluation`: `hasForceCorrection`, `emitForceCorrection`, `getStoredTermCount`, `emitStoredTerms` | Restraints; borrowed correction names and the stored-column destination |
| `TermCoordinateOutput`: `hasCoordinateOutput`, `emitAuxiliaryPotentials`, `emitOutput` | Pulls; borrowed output-site names and shared centroid geometry |
| `TermScalarParameters`: `collectBareConstants`, `collectObservedConstants`, `appendInitialConstants` | Custom pairs; named scalar collections and a bounded start-value appender |
| `TermNonbonded`: `collect`, `collectGradientFields`, `collectGradients`, `emit` | Nonbonded; projected mesh/correction data and the shared parameter/emission capabilities |

The audit folds separate tuple/table refill hooks into `refillArrays` and
read-only/mutable tunable classification into `collectTunableOutcome`.
Duplicate declaration/name hooks become `appendTermDefinitions`. One-kind
restraint, pull, and scalar operations are narrowed into their roles; the
unused generic entry hook is removed. Long-range dispersion and gradient
shift corrections are shared by custom pairs and built-in nonbonded terms.
Adding a new writer at a correction site requires explicitly composing its
SSA outputs, rather than reusing another adapter's names. The restraint adapter owns its recorded correction energy for one program build; force-only hooks retain it, and the stored-column hook diagnoses a missing energy before emitting a store.

No control key, support restriction, precision, numerical method, diagnostic,
or output format changes. Each kind commit compares printed module/lowered
corpora and the combined-kind CPU/GPU mixed/double order matrix with the fixed
archived PR base. The completed PR head additionally requires five bitwise
checkpoints and two D309 suite shards. The next stages of #568 replace term
text templates with `OpBuilder` IR, then retain text only for user expressions
and `ModelTerm` at their parsing boundaries.

## 3. Decided

| # | Question | Decision |
|---|---|---|
| 1 | The tables and keywords of the control file | As in Section 1 (D35) |
| 2 | The units of the control file | Å, kcal/mol, ps (D36) |
| 3 | The trajectory format | DCD first (D37); XTC as well (D141); H5MD for frames without loss (D239) |
| 4 | The checkpoint | H5MD (D26, D40), with HDF5 1.14.6, which `scripts/build-hdf5.sh` installs |
| 5 | The TOML library | toml++, in the repository (D38) |
| 6 | The schedule | Compiled (D39) |

## 4. State

| Part | State |
|---|---|
| Control file, with errors that name the line | Implemented |
| Positions from a PDB file | Implemented |
| Energy expressions, types, mixing | Implemented |
| Compile and run in the process, on the CPU and on a GPU | Implemented |
| Log, with the virial and the pressure | Implemented |
| Trajectory in the DCD format | Implemented |
| Trajectory in the XTC format | Implemented (D141) |
| Initial velocities | Implemented. The sequence of random numbers is fixed by the seed and does not depend on a library. |
| Checkpoints in H5MD, and runs that continue from one | Implemented |
| The number of builds of the neighbor structures, in the log | Implemented |
| The particles in the order of their positions, `spatial_order` | Implemented |
| `nbupdate_period` | Not implemented; the keyword is an error |
| Trajectory in H5MD, without loss | Implemented (D239) |
| Velocities in the trajectory, `dcdvelfile` | Not in the trajectory of `mdir run`; `H5MDReporter(velocities=True)` of a Python simulation writes them (D239) |

### Expression parameter names (D188)

The control reader rejects term parameters that collide with names supplied
by their term: coordinates, time, topology properties, declared particle
parameters and their applicable place suffixes, built-in and tabulated
functions, and declared lambda components. The declaration is checked even
when unused, so a constant cannot silently replace a coordinate and its
force. Compound terms accept declared `lambda_<name>` components through
the existing lambda bindings, with forces, component derivatives, and
state energy differences computed by the same differentiation path.
