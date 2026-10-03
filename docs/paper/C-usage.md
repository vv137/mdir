# Appendix C. Using MDIR

This appendix describes how a run is set up, started, continued, and
diagnosed, beside the reference of the control file (Appendix A). Every
command and control file below was run at the commit of this paper; the
outputs are quoted from those runs, and numbers that depend on the machine
or the commit (times, hashes, paths) will differ elsewhere.

## C.1 An installation and its check

MDIR is built as Appendix B.1 describes. `mdir version` prints what the
build supports:

```text
$ mdir version
MDIR 0.1.0
commit: e7a644af9b97743f534a4c0040f98308a9ad40c5
uncommitted changes: no
LLVM 23.1.2
targets: cpu, gpu (CUDA 13.4.20260911 at /usr/local/cuda-13.4)
checkpoints: yes (HDF5)
```

`targets` lists `gpu` when the build found a CUDA toolkit, and
`checkpoints` says whether HDF5 was found; without it a run writes no
checkpoints and cannot be continued (C.5). A short run on each target
checks the rest. `examples/argon/argon.toml` is liquid argon, 864 atoms
with the terms in the control file, 2000 steps on the CPU in double
precision:

```text
$ mdir run examples/argon/argon.toml
...
MDIR: neighbor structures were built 113 times, every 17.9 steps on average
MDIR: the momentum at the end is 3.904e-12 amu nm/ps
MDIR: the total energy changed by 1.813e-05 of its value
```

A copy with `target = "GPU"` and `precision = "MIXED"` in `[execution]`
runs on a device; over its first 200 steps the total energy of the last
row, −834.1525 kcal/mol, is that of the CPU to the printed digits. The
device is the first that `CUDA_VISIBLE_DEVICES` leaves visible, or the one
of that index that `MDRT_DEVICE` names:

```sh
CUDA_VISIBLE_DEVICES=1 mdir run argon-gpu.toml
```

Every run is compiled before it runs (Section 3.6), and the log reports
the time: on the CPU, `MDIR: compiled in 1.32 s` for the dipeptide of
C.2, and 5.1 s for the 4,905 particles of `examples/ala3` with a
barostat.

## C.2 Inputs from Amber, GROMACS, and CHARMM

The control file names the files of the system in `[input]`; the format
follows from the names of the files, or from `format` (Appendix A.1). The
paths are relative to the control file.

*Table C.1. What each format needs.*

| Format | `[input]` | Also |
|---|---|---|
| Amber | `topology` (`.prmtop`, `.parm7`) and `coordinates` (`.inpcrd`, `.rst7`), which hold the cell | |
| GROMACS | `topology` (`.top`) and `coordinates` (`.gro`); `include_paths`, the directories of `#include`, among them the `top` directory of a GROMACS installation for its force fields; `defines`, the names of `#define` | A topology whose waters have SETTLE needs `rigid_water` set either way |
| CHARMM | `format = "CHARMM"`, `topology` (`.psf`, XPLOR kind), `coordinates` (`.crd`), and `parameters`, the files `.rtf`, `.prm`, and `.str` in the order CHARMM reads them | `[boundary] box`, since a `.crd` has no cell; the force switch of CHARMM force fields |
| PDB | `coordinates` only | The terms and the types in `[energy]`, the cell in `[boundary]` (`examples/argon`) |

A minimal run at constant temperature from an Amber topology:

```toml
[input]
topology    = "dipeptide.prmtop"
coordinates = "dipeptide.inpcrd"

[energy]
cutoff            = 9.0
pairlist_distance = 10.0
electrostatics    = "PME"

[dynamics]
time_step = 0.002
steps     = 20

[ensemble]
ensemble    = "NVT"
temperature = 300.0

[thermostat]
method        = "V-RESCALE"
time_constant = 0.5

[constraints]
hydrogen_bonds = true
rigid_water    = true

[boundary]
type = "PERIODIC"

[execution]
target = "CPU"
```

The integrator is velocity Verlet unless `[dynamics]` names another, and
the log has a row every ten steps unless `[output]` sets
`energy_interval`. A step of 2 fs needs the constraints: an Amber topology
leaves its waters flexible, and the same run without `[constraints]`
diverged within ten steps. With `dipeptide.top` and `dipeptide.gro`, the
same system converted by ParmEd, the control file is unchanged but for the
names. A GROMACS topology that includes a force field of the installation
names its directory:

```toml
[input]
topology      = "water.top"     # #include "amber99sb-ildn.ff/forcefield.itp"
coordinates   = "water.gro"
include_paths = ["/opt/gromacs/share/gromacs/top"]
```

and a system of CHARMM takes its files of parameters and its cell:

```toml
[input]
format      = "CHARMM"
topology    = "toy.psf"
coordinates = "toy.crd"
parameters  = ["toy.rtf", "toy.prm"]

[energy]
cutoff                 = 12.0
switch_distance        = 10.0
lennard_jones_modifier = "POWER_FORCE_SWITCH"
dispersion_correction  = "NONE"
pairlist_distance      = 13.0
electrostatics         = "PME"

[boundary]
type = "PERIODIC"
box  = [32.0, 32.0, 32.0]
```

`mdir check FILE` reads the control file and the system without
compiling, and prints what it found:

```text
$ mdir check dipeptide.toml
topology:           dipeptide.prmtop
particles:          1168
residues:           385
types:              9
bonds:              9, 0 with hydrogen
angles:             36
dihedrals:          67, 4 improper
pairs 1-4:          41
excluded pairs:     1244
total charge:       -0.000000 e
total mass:         7026.29 amu
box:                27.1882 29.7181 25.397 Å
density:            0.568579 g/cm³
velocities:         no
degrees of freedom: 2343
```

Values that name a choice are read without regard to case: `"CPU"` and
`"cpu"` are one target.

## C.3 The standard pipeline

A system from a builder is minimized, heated at constant volume,
equilibrated at constant pressure with its solute restrained, and only
then run for production. `mdir template minimize`, `nvt`, `npt`, and
`production` print these four stages for an Amber solute in water, with
the inputs `system.prmtop` and `system.inpcrd` (Appendix A.4, D147):

```sh
mdir template minimize   > 1-min.toml
mdir template nvt        > 2-nvt.toml
mdir template npt        > 3-npt.toml
mdir template production > 4-md.toml
for stage in 1-min 2-nvt 3-npt 4-md; do
  mdir run --continue $stage.toml
done
```

Each stage begins from the checkpoint of the one before (`min.h5`,
`nvt.h5`, `npt.h5`), at its step, so that the steps and the time of the
stages follow one another and their random numbers differ (D129); the
reference of the restraints stays the coordinates of `[input]`. With the
lengths cut to 100 steps, the stages ran from step 0 to 100, 100 to 200,
and 200 to 300, and the last checkpoint records it:

```text
$ mdir checkpoint md.h5
particles:       4905
step:            300
time:            0.6 ps
...
run began at:    step 200
part:            1
trajectory:      md.dcd, 2 frames
```

A stage that begins from the checkpoint of a run at constant pressure
takes the cell of that checkpoint and says so on the standard error
(`warning: the cell of 'npt.h5' ... differs from that of the input`). Run
again, a complete stage says `the run is complete` and exits with 0, so the
loop above also continues a pipeline that a job left unfinished.
`examples/ala3/run.sh` runs the same stages on tri-alanine in OPC water.
The selection of the restraints, `!:WAT & !@H*`, and the residues of rigid
water assume water named `WAT`; other systems adapt them.

## C.4 Target and precision

`[execution]` chooses where a run executes and in which precision mode.

*Table C.2. The choices of `[execution]`.*

| Keyword | Values | When |
|---|---|---|
| `target` | `CPU` (with `threads`), `GPU` | The CPU runs anywhere, with OpenMP threads; the measurements of Section 10 are on a device |
| `precision` | `MIXED`, `DOUBLE`, `SINGLE` | `MIXED`, the mode of production (Section 7); `DOUBLE` to compare with references and to check a result; `SINGLE` keeps the state in f32 as well (Table 7.1) |
| `neighbor_structure` | `MATRIX`, `GROUPS` | `GROUPS`, with a dual list (`pruned_distance` in `[energy]`), is the configuration of the measurements of Section 10 |
| `deterministic` | `false`, `true` | `true` for the same bits from run to run on one binary and device (D84); not with `GROUPS` on a device |

The production stage of C.3 on a device, with the groups and a dual list
whose inner list reaches 0.6 Å past the cutoff:

```toml
[energy]
cutoff            = 9.0
pairlist_distance = 12.0
pruned_distance   = 9.6

[execution]
target             = "GPU"
precision          = "MIXED"
neighbor_structure = "GROUPS"
```

Section 10 measured the suite with a cutoff of 8 Å, an outer reach of
11 Å, and an inner one of 8.6 Å at 2 fs.

## C.5 Runs on a cluster

A run that outlasts a job is started and continued by the same command
(Appendix A.3, D129 to D132):

```sh
mdir run --continue --max-walltime 23:50 md.toml
case $? in
  0)  echo "the run is complete" ;;
  75) echo "stopped at a checkpoint; submit the job again" ;;
  *)  echo "the run failed" ;;
esac
```

`--max-walltime` stops at the last checkpoint that leaves room for one
more interval between checkpoints, and SIGTERM or SIGINT stops at the next
checkpoint; either way the run exits with 75 and says where it stopped:

```text
MDIR: stopped after step 900 on SIGTERM; 'long.h5' holds the state, and `mdir run --continue` goes on from it
```

The next `--continue` begins after that step (`continues the run after
step 900, from 'long.h5', to step 1700 (part 4)`) and arrives at the state
of a run without a stop (Section 9.6). A second signal ends the run at
once. GNU `timeout` sends its signal to the command and again to its
process group, which the run takes as the second signal: under `timeout -s
TERM` the run above ended with 143 and no stop at a checkpoint, and under
`timeout --foreground -s TERM` it stopped at its next checkpoint with 75.
The warning signal of a scheduler must leave time for one interval between
checkpoints.

The checkpoint before the last stays as `<checkpoint>.prev` (D132):
copied over a checkpoint that was damaged after it was written, it lets
`--continue` go on from the interval before. With `--no-append`, the frames of each part go to a file of
their own, `long.part0005.dcd` for part 5 (D130), and otherwise they are
appended to the trajectory after the frames past the checkpoint are cut.
`mdir checkpoint first.h5 second.h5` compares two states and says how
they differ (`the steps differ: 200 and 300`).

## C.6 What a run writes

<!-- revise after U12 -->

| Output | Where | Reference |
|---|---|---|
| The log | The standard output: the terms at the start, a row every `energy_interval` steps, and a summary | Section 6.5 |
| The trajectory | `[output] trajectory`, DCD (Å) or XTC (nm, to a thousandth) by its extension | D141 |
| The checkpoint | `[output] checkpoint`, H5MD with all numbers in 64 bits; `mdir checkpoint --print=positions FILE` (or `velocities`, `forces`) prints a field in the order of the input, in nm, nm/ps, or kJ/mol/nm | D26, docs/driver-m0.md Section 2.6 |
| The coordinates of pulling | `[output] pull_coordinates` (C.7) | D145 |

The control file is in Å, kcal/mol, ps, K, and atm; the checkpoint keeps
the units inside MDIR, nm, ps, and kJ/mol, and writes them with the data.

## C.7 Terms given by expressions

Terms beyond the force field are written in the control file as
expressions (D136 to D139, D145): over tuples of particles, over the
centers of groups, and over the pairs of a topology, in Å, kcal/mol, and
radians, with functions given by tables (D138). A flat-bottomed restraint
between two atoms, two groups pulled apart at 1 Å/ps by a spring on their
centers of mass, and a dihedral given by a periodic spline, on the
dipeptide of C.2:

```toml
[output]
energy_interval  = 10
pull_coordinates = "pull.dat"

[[energy.bond]]
name       = "flat"
expression = "k*max(0, r - r0)^2"
particles  = [[2, 19]]
k  = 10.0
r0 = 4.0

[[energy.bond]]
name       = "pull"
expression = "k*(r - (r0 + v*t))^2"
groups     = [":1-3", ":4"]
k  = 2.0
r0 = 17.0
v  = 1.0

[[energy.dihedral]]
name       = "table"
expression = "torsion(theta)"
particles  = [[5, 7, 9, 15]]

[[energy.function]]
name     = "torsion"
min      = -3.141592653589793
max      = 3.141592653589793
periodic = true
values   = [1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0]
```

The particles are numbered from 1, as in the topology; the groups are
masks of Amber, weighted by mass unless `weighting = "NONE"`; `t` is the
time in ps. The log lists each term among the terms at the start
(`MDIR:   flat   108.296187`), and `pull_coordinates` writes at every row
of the log the coordinates of the term over centers, its energy, and its
force, the force on the second center and its component along the
distance:

```text
# step time pull.r pull.dx pull.dy pull.dz pull.energy pull.f_r pull.fx pull.fy pull.fz
# ps; Å, radians; kcal/mol; kcal/mol/Å, kcal/mol/rad
0 0.000000 17.051243 9.907016 10.411861 9.175460 0.005252 -0.204971 -0.119091 -0.125160 -0.110297
10 0.020000 17.011361 9.963863 10.464706 8.977625 0.000149 0.034557 0.020241 0.021258 0.018237
```

A constant force along the distance is the expression `-f*r`.
Restraints to the positions of the input are a table of their own,
`[[restraints]]` with a mask and a force constant (Appendix A.1), as the
templates of C.3 use them.

## C.8 Implicit solvent, and runs without a cell

Generalized Born takes the radii and the screening of an Amber topology
(D144) and the Coulomb of a cutoff, usually without a periodic cell
(D142). A structure from tleap is minimized first; the peptide of
`test/Driver/Inputs/gb` went from 14,266.7 to −148.2 kcal/mol in 500
steps of steepest descent:

```toml
[input]
topology    = "peptide.prmtop"
coordinates = "peptide.inpcrd"

[output]
checkpoint      = "gb-min.h5"
energy_interval = 100

[energy]
cutoff              = 30.0
pairlist_distance   = 32.0
electrostatics      = "CUTOFF"
implicit_solvent    = "OBC2"
surface_area_energy = 0.0054    # kcal/mol/Å²; 0 for no nonpolar term

[minimize]
method       = "STEEPEST_DESCENT"
steps        = 500
initial_step = 0.01

[boundary]
type = "NONE"

[execution]
target = "CPU"
```

Dynamics then begins from `checkpoint = "gb-min.h5"` in `[input]`, with
`[dynamics]` in place of `[minimize]`, and, in a continuum, Langevin
dynamics as its thermostat (`method = "LANGEVIN"`, `friction = 1.0`).
Without a cell, the run places one around the particles whose images stay
beyond the reach of the neighbor structures, prints it, and stops with a
message if the particles spread past it; a larger `pairlist_distance`
places a larger cell. The log of a run without a cell has no pressure,
and the trajectory no cell.

## C.9 When something fails

`mdir check` is the first step: it reads everything that `mdir run`
reads, and its errors name the file, the line, and the table. A keyword
that is not known lists the valid ones and suggests the nearest (D146):

```text
$ mdir check typo.toml
mdir: typo.toml:6: unknown keyword 'cutof' in [energy]; did you mean 'cutoff'?
valid keywords: cutoff, switch_distance, pairlist_distance, ...
try 'mdir template md' or 'mdir template amber' for a control file with the supported keywords
```

`mdir emit FILE` prints the program that the driver builds,
`--stage=lowered` the program that runs, and `--stage=pipeline` the passes
between. A run that fails in the compiler or on a device is reported with
what `mdir bug-report` collects, the build, the machine, the inputs with
their hashes, and each stage of the compilation, each in a process of its
own:

```sh
mdir bug-report run.toml -o report          # --run adds a run, each kernel waited for
tar czf report.tar.gz report
```

`docs/debugging.md` describes the failures of the passes and of kernels
and the tools for each. The exit status of `mdir run` is 0 for a complete
run, 75 for a stop at a checkpoint, and 1 for an error, which the
standard error describes.
