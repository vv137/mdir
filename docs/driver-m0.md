# Driver and Control File for Milestone M0

Status: decided (2026-09-29) and implemented; see Section 4 for what is
left.

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

### 1.1 Keywords

| Table | Keyword | Meaning |
|---|---|---|
| `[input]` | `topology` | The topology: of Amber (`.prmtop`, `.parm7`) or GROMACS (`.top`). Absent for a run from a PDB file, whose terms are in `[energy]`. |
| | `coordinates` | The positions and the cell: of Amber (`.inpcrd`, `.rst7`), GROMACS (`.gro`), or a PDB file, in which the name of an atom selects its type. The reference of restraints. |
| | `format` | `AUTO` (the default: from the names of the files), `AMBER`, `GROMACS`, or `PDB`; `CHARMM` is not supported yet. |
| | `include_paths`, `defines` | With a GROMACS topology: the directories of `#include` and the names that `#define` gives. |
| | `checkpoint` | The checkpoint of an earlier run, which the run continues, taking its cell (and warning on the standard error if the input has another); the input's cell still sets the grid of PME and the reference of restraints. One of a minimization gives the positions only. |
| `[output]` | `trajectory` | Positions, in DCD (`.dcd`). |
| | `checkpoint` | The checkpoint (D26), written every `checkpoint_interval` steps in place of the one before, and at the end of a minimization. |
| | `energy_interval`, `trajectory_interval`, `checkpoint_interval` | Steps between the rows of the log, the frames, and the checkpoints (Section 2.2). The intervals nest, either way for energies and frames. |
| `[energy]` | `cutoff` | The cutoff of `md.neighborhood` (Å). |
| | `pairlist_distance` | The reach of the neighbor structures; the skin is `pairlist_distance − cutoff`. |
| | `rebuild_interval` | **Opt-in, not a default.** 0 (the default): a structure is rebuilt when a particle has moved half the skin, tested at every step. `n`: rebuilt every `n` steps and not tested in between, so it may miss pairs within the cutoff (D88). The run warns at the start, on the standard error and in the log, and reports at the end how many rebuilds found a structure no longer valid, with a warning if any did. |
| | `switch_distance` | `truncation(switch, from = ...)`; equal to `cutoff`: no switching. For terms in the control file. |
| | `lennard_jones_modifier` | `NONE`, `POTENTIAL_SHIFT` (`truncation(shift)`), or `FORCE_SWITCH` (`truncation(force_switch, from = switch_distance)`). For terms in the control file. |
| | `electrostatics` | With a topology: `CUTOFF` or `PME`. |
| | `coulomb_modifier` | With PME: `NONE`, or `POTENTIAL_SHIFT`, the direct sum shifted to zero at the cutoff. |
| | `dispersion_correction` | `NONE` or `ENERGY_PRESSURE`; also in `[[energy.pair]]`. |
| | `[[energy.pair]]` | A pair term, given by an expression (D16, D22). |
| | `[[energy.type]]` | A type of particle: its mass and its parameters. |
| | `[[energy.pair_override]]` | Parameters of a term for one pair of types. |
| `[pme]` | `tolerance`, `beta`, `max_spacing`, `grid`, `order`, `influence` | Particle mesh Ewald (D71): β from `erfc(β r_c) = tolerance` or given; the grid from the largest spacing or given as three numbers of points; the order of the B-splines, 4, 6, or 8; the influence function, `SPME` or `OPTIMAL`. |
| `[dynamics]` | `integrator` | `VELOCITY_VERLET` or `LEAPFROG`: the `dyn.program` (D76). |
| | `time_step`, `steps` | In ps, and the number of steps. |
| | `seed` | Of the initial velocities and of the coupling. |
| | `center_of_mass_interval` | Steps between removals of the motion of the center of mass; with a thermostat, when it acts. |
| `[minimize]` | `method`, `steps`, `initial_step` | `STEEPEST_DESCENT`, the number of steps, and the first step (Å) (D73). Instead of `[dynamics]`. |
| `[ensemble]` | `ensemble` | `NVE`, `NVT` (with `[thermostat]`), or `NPT` (with `[thermostat]` and `[barostat]`). |
| | `temperature`, `pressure` | K, of the initial velocities and the bath; atm, with `NPT`. |
| `[thermostat]` | `method`, `time_constant`, `interval` | `V-RESCALE`, stochastic velocity rescaling; ps; steps between its actions (10 by default). |
| `[barostat]` | `method`, `time_constant`, `compressibility`, `coupling`, `work`, `interval` | `C-RESCALE`, stochastic cell rescaling (D72, D77); ps; 1/atm; `ISOTROPIC`; `EXACT` or `FIRST_ORDER`; the steps of the thermostat. |
| `[constraints]` | `hydrogen_bonds`, `rigid_water`, `water_residues` | SHAKE and RATTLE on the bonds of hydrogen; SETTLE on the waters; the names of the residues of water. |
| `[[restraints]]` | `selection`, `force_constant` | A mask of Amber, and kcal/mol/Å² (D74). |
| `[boundary]` | `type`, `box` | `PERIODIC`; the edges of the cell (Å), without a topology. |
| `[execution]` | `target`, `threads`, `precision` | `CPU` or `GPU`; the threads of the CPU; `SINGLE`, `MIXED`, or `DOUBLE`. |
| | `neighbor_capacity` | Neighbors that a neighbor structure holds per particle. Absent: estimated from the configuration. |
| | `fast_math` | Whether kernels are rewritten in ways that change rounding: the distance in powers of its square, fused multiply-adds, and in f32 approximate divisions and an approximation of erfc within 3.3e-7 (D90). The default is `true`. |
| | `spatial_order` | Whether the run keeps the particles in the order of their positions (D44). The default is `true`. The files of the run are in the order of the input either way. |
| | `neighbor_structure` | `MATRIX` (the default): a row of neighbors for each particle, each pair computed for both. `GROUPS`: groups of 16 particles that share a list, each pair computed once, for the loops whose values all have an exchange contract (D89); on a device, not in the deterministic mode. |
| | `deterministic` | Whether every sum is added in an order that the threads do not decide, so that a run gives the same bits from run to run on the same binary and hardware (D84). The default is `false`: a device then adds the charges of PME with floating-point atomics. |

A keyword of a pair term or of a type that is not listed here names a
number: a parameter of the type, or a constant of the term, that the
expression uses.

A parameter that has the same value for all types is a constant of the
kernel. A parameter that differs is a field, and `mixing` says how the
parameter of a pair follows from those of its two particles:
`lorentz-berthelot` [[Lorentz1881]](references.md#lorentz1881), [[Berthelot1898]](references.md#berthelot1898), `geometric`, or a table with `arithmetic` or
`geometric` for each parameter.

An unknown keyword is an error. `mdir template md` and `mdir template
amber` print control files with every keyword; the keywords follow the
code, not earlier versions, until a milestone is released.

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
(D59). It runs the passes in its own process and executes the code with
the execution engine of MLIR. The
Python library (C5) comes later and calls the same steps.

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
| Energies are computed only when they are written | The step of an energy interval that is written requests the energy; the others request forces only. |
| A loop over pairs reads the neighbors of a particle from few places in memory | The particles are in the order of their positions. The loop over checkpoint intervals carries the masses, the parameters, and the numbers of the particles as well, because they change places with every new order. |
| The files are in the order of the input | A call that writes takes the numbers of the particles with the field, and the driver writes the value of a particle at the place of its number. |

`examples/argon/argon.mlir` has this form already, with two levels of loops.

The periods must divide one another: `trajectory_interval` and `checkpoint_interval`
are multiples of `energy_interval`.

### 2.3 The log

```text
INFO:      STEP           TIME      TOTAL_ENE  POTENTIAL_ENE    KINETIC_ENE    TEMPERATURE         VIRIAL       PRESSURE
INFO:      2000        10.0000      -834.1467     -1083.2389       249.0922        96.8538       -70.6513       232.3875
```

The line shows the last row of `examples/argon/argon.toml`.

| Column | Unit | Meaning |
|---|---|---|
| `TIME` | ps | |
| `POTENTIAL_ENE` | kcal/mol | |
| `KINETIC_ENE` | kcal/mol | `K`, the kinetic energy of the velocities at the time of the row |
| `TOTAL_ENE` | kcal/mol | `POTENTIAL_ENE + KINETIC_ENE` |
| `TEMPERATURE` | K | `2 K_T / (f k_B)`, with `f = 3 N − 3` degrees of freedom and `K_T` as below |
| `VIRIAL` | kcal/mol | The trace of the MDIR virial `W = Σ d_ij ⊗ K(i, j)`, which is positive for repulsion (B8). Other packages print other quantities under this name: the GROMACS virial is `−W / 2` [[GromacsManual2025]](references.md#gromacsmanual2025). |
| `PRESSURE` | atm | `(2 K_P + tr W) / (3 V)`, with `K_P` as below |

The pressure has no correction for the dispersion beyond the cutoff; that
correction comes with M1.

**Three kinetic energies (D45).** The velocities of a step are those of a
finite difference of the positions, not those of the trajectory, so their
kinetic energy `K` is off by a term of the order `Δt²`. The kinetic
energy of the velocities half a step before and after is off as well, by
half as much and in the other direction. The mean of the two half steps
is

```text
K_half = K + (Δt² / 8) Σ_i F_i² / m_i
```

which holds exactly without constraints and thermostats.

| Quantity | Kinetic energy | Reason |
|---|---|---|
| Total energy | `K` | The sum with the potential energy varies least: for argon with a step of 20 fs, by 0.05 kcal/mol, against 0.10 and 0.12 with the other two. |
| Temperature | `K_T = (K + 2 K_half) / 3`, the mean of the three times | The terms of the order `Δt²` cancel. |
| Pressure | `K_P = K_half` | The positions of the steps satisfy the virial theorem with this kinetic energy. |

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
loop over pairs that computes the energy and the forces. Columns for
bonded terms follow when MDIR computes them.

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
| Initial velocities | From `temperature` and `seed`, with the center of mass at rest | A random number generator |

### 2.5 What the driver builds

`mdir emit control.toml` prints the module that the driver builds, and
`mdir emit control.toml --stage=lowered` the module that is executed.

| Part of the module | From |
|---|---|
| `md.potential @energy` | `[energy]`: one `md.sum_relation` for each pair term, with the truncation and the cutoff |
| `dyn.program @step` | `integrator`. With velocity Verlet there is a second program that returns the energy as well, for the last step before an output. |
| `func.func @mdir_run` | The schedule: the loops, the calls that write, and the buffers of the state |

The entry function takes the buffers of the positions, the velocities, the
masses, the fields of the parameters, and the numbers of the particles,
then the edge lengths of the cell and the time step. The cell and the time step are values of the run, not of
the program (P1).

With leapfrog, the stored velocities are half a step behind the positions.
The log has the kinetic energy of the time of a row all the same
(Section 2.3).

### 2.6 Checkpoints

A checkpoint is a file in the H5MD format [[deBuyl2014]](references.md#debuyl2014), version 1.1, with all numbers in
64 bits (D26). It holds the state as the next step needs it.

```text
/h5md                         version, author, creator
/particles/all/box            dimension, boundary, edges
/particles/all/position       step, time, value
/particles/all/velocity       step, time, value
/particles/all/force          step, time, value     with velocity Verlet
/particles/all/id             the numbers of the particles
/particles/all/species        the types of the particles
/particles/all/mass
/parameters/mdir              format, integrator, velocity_offset,
                              precision, timestep, seed
```

The units are those inside MDIR and are written with the data: nm, ps, u,
and kJ/mol.

| Rule | Reason |
|---|---|
| A value of 32 bits is stored in 64. | The conversion is exact in both directions, so a run in single precision continues without loss. |
| With velocity Verlet the checkpoint holds the forces. | A step begins with the forces of the step before. Forces that are computed again from the positions differ in their last bits, because a neighbor structure that is built again has another order. |
| With leapfrog the time of the velocities is half a step before that of the positions. | The file says what it holds. |
| Neighbor structures start empty after every checkpoint (R1). | The run that continues builds its structure at the first step. The run that was not interrupted must build there too. |
| The file appears under its name only when it is complete. | A run that ends while it writes leaves the checkpoint before. |
| The particles are in the order of the input, whatever order the run keeps them in. | The file does not depend on the plan of the run. The run that continues puts the particles in order where it begins, and arrives at the order of the run that was not interrupted (D44). |

A run that continues from a checkpoint arrives at the state of the run that
was not interrupted, bit for bit. This holds on the CPU and on a GPU, in
every precision mode, and for both integrators; the tests compare the
states. The two runs must have the same `checkpoint_interval`.

A run cannot continue with another integrator: the velocities of the two
are not of the same time. It cannot continue in another box.

`mdir checkpoint file.h5` describes a checkpoint, and `mdir checkpoint
first.h5 second.h5` compares the states of two.

## 3. Decided

| # | Question | Decision |
|---|---|---|
| 1 | The tables and keywords of the control file | As in Section 1 (D35) |
| 2 | The units of the control file | Å, kcal/mol, ps (D36) |
| 3 | The trajectory format | DCD first; XTC follows (D37) |
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
| Initial velocities | Implemented. The sequence of random numbers is fixed by the seed and does not depend on a library. |
| Checkpoints in H5MD, and runs that continue from one | Implemented |
| The number of builds of the neighbor structures, in the log | Implemented |
| The particles in the order of their positions, `spatial_order` | Implemented |
| `nbupdate_period` | Not implemented; the keyword is an error |
| Trajectory in the XTC format | Not implemented |
| Velocities in the trajectory, `dcdvelfile` | Not implemented |
