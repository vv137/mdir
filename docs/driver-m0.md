# Driver and Control File for Milestone M0

Status: decided (2026-09-29), being implemented.

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

Later milestones add `[constraints]`, `[[selection]]`, and
`[[restraints]]`, and one table for each method that needs parameters of
its own.

```toml
[input]
pdbfile = "argon.pdb"            # positions
# rstfile = "equilibrated.h5"   # positions and velocities of an earlier run

[output]
dcdfile = "run.dcd"
rstfile = "run.h5"

[energy]
switchdist   = 7.5               # Å
cutoffdist   = 8.5
pairlistdist = 9.5

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
integrator    = "VVER"
timestep      = 0.005            # ps
nsteps        = 2000
eneout_period = 100
crdout_period = 100
rstout_period = 2000
iseed         = 314159

[ensemble]
ensemble    = "NVE"
temperature = 94.4               # for the initial velocities

[boundary]
type       = "PBC"
box_size_x = 34.7786
box_size_y = 34.7786
box_size_z = 34.7786

[execution]
target    = "gpu"                # or "cpu"
threads   = 16                   # for the target cpu
precision = "mixed"              # single, mixed, or double
```

Keywords are in lower case. Values that name a choice, such as `VVER` and
`PBC`, are strings and are read without regard to case.

### 1.1 Keywords of M0

| Table | Keyword | Meaning in MDIR |
|---|---|---|
| `input` | `pdbfile` | Positions. The name of an atom selects its type. |
| | `rstfile` | The state of an earlier run (D26) |
| `output` | `dcdfile`, `xtcfile` | Trajectory of positions |
| | `rstfile` | Checkpoint (D26) |
| `energy` | `cutoffdist` | The cutoff of `md.neighborhood` |
| | `switchdist` | `truncation(switch, from = ...)`. Equal to `cutoffdist`: no switching. |
| | `vdw_force_switch` | `truncation(force_switch, from = ...)` |
| | `vdw_shift` | `truncation(shift)` |
| | `pairlistdist` | The skin is `pairlistdist − cutoffdist`. |
| | `[[energy.pair]]` | A pair term, given by an expression (D16, D22) |
| | `[[energy.type]]` | A type of particle: its mass and its parameters |
| `dynamics` | `integrator` | `VVER` or `LEAP`: the `dyn.program` |
| | `timestep`, `nsteps` | |
| | `eneout_period`, `crdout_period`, `rstout_period` | The schedule of Section 2.2 |
| | `nbupdate_period` | The rebuild policy `interval`, with the check of A11. Absent: the policy `check` (B1). |
| | `iseed` | The seed of the initial velocities |
| `ensemble` | `ensemble` | `NVE` in M0 |
| | `temperature` | The temperature of the initial velocities |
| `boundary` | `type` | `PBC` in M0 |
| | `box_size_x`, `box_size_y`, `box_size_z` | `md.orthorhombic_cell` |
| `execution` | `target`, `threads`, `precision` | Structural plan parameters |

An unknown keyword is an error. A keyword that is planned but not supported
yet is an error that names the milestone that brings it.

`mdir-run --template md` prints a control file with every keyword and its
default.

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

## 2. The driver

### 2.1 What it does

```text
mdir-run control.toml

  1. read the control file and the files of [input]
  2. build a module of md and dyn ops
  3. compile it: the passes of md, md_exec, and the lowering of the target
  4. load the code and run it
        the code calls the runtime to write energies, frames, checkpoints
  5. write the last checkpoint
```

The driver is a C++ program, `tools/mdir-run`. It runs the passes in its
own process and executes the code with the execution engine of MLIR. The
Python library (C5) comes later and calls the same steps.

### 2.2 The schedule is compiled

The periods of output are known before the run, so the driver compiles the
whole run as one function with nested loops:

```text
for each checkpoint interval           rstout_period steps
    neighbor structures start empty      (R1)
    for each frame interval            crdout_period steps
        for each energy interval       eneout_period steps
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
| Energies are computed only when they are written | The step of an energy interval that is written requests the energy; the others request forces only. |

`examples/argon.mlir` has this form already, with two levels of loops.

The periods must divide one another: `crdout_period` and `rstout_period`
are multiples of `eneout_period`.

### 2.3 The log

```text
INFO:      STEP           TIME      TOTAL_ENE  POTENTIAL_ENE     KINETIC_ENE    TEMPERATURE
INFO:      2000        10.0000      -837.0889     -1087.8131       250.7242        97.4655
```

The line shows the last row of `examples/argon.mlir`, in kcal/mol. Columns
for bonded terms, the virial, and the pressure follow when MDIR computes
them.

### 2.4 Parts

| Part | Work | Depends on |
|---|---|---|
| Parser of the control file | TOML to a description of the run; errors with the line | A TOML library |
| Reader of PDB | Positions and names | |
| Builder | The description of the run to a module | |
| Energy expressions | The syntax of D22 to a kernel | A parser of expressions |
| Compile and run | Passes in the process, the execution engine | MLIR libraries, linked into the tool |
| `mdrt` ops for output | `mdrt.write_energies`, `mdrt.write_frame`, `mdrt.write_checkpoint` | |
| Writers | DCD or XTC, H5MD | HDF5 for H5MD |
| Initial velocities | From `temperature` and `iseed`, with the center of mass at rest | A random number generator |

## 3. Decided

| # | Question | Decision |
|---|---|---|
| 1 | The tables and keywords of the control file | As in Section 1 (D35) |
| 2 | The units of the control file | Å, kcal/mol, ps (D36) |
| 3 | The trajectory format | DCD first; XTC follows (D37) |
| 4 | The checkpoint | H5MD (D26). HDF5 is installed in the home directory. |
| 5 | The TOML library | toml++, in the repository (D38) |
| 6 | The schedule | Compiled (D39) |
