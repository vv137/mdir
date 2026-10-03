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
| `[input]` | `topology` | The topology: of Amber (`.prmtop`, `.parm7`), GROMACS (`.top`), or CHARMM (`.psf`, of the XPLOR kind, with the names of the types). Absent for a run from a PDB file, whose terms are in `[energy]`. |
| | `coordinates` | The positions and the cell: of Amber (`.inpcrd`, `.rst7`), GROMACS (`.gro`), CHARMM (`.crd`, standard or extended; it has no cell), or a PDB file, in which the name of an atom selects its type. The reference of restraints. |
| | `format` | `AUTO` (the default: from the names of the files), `AMBER`, `GROMACS`, `CHARMM`, or `PDB`. A CHARMM force field runs from its own files (D122, [charmm-m1.md](charmm-m1.md)) or from a topology of GROMACS (D121). |
| | `parameters` | With a PSF: the files of topology (`.rtf`), parameters (`.prm`), and streams (`.str`), in the order that CHARMM reads them; a later file replaces what an earlier one defines. |
| | `include_paths`, `defines` | With a GROMACS topology: the directories of `#include` and the names that `#define` gives. |
| | `checkpoint` | The checkpoint of an earlier run, whose state the run begins from, at its step and time (D129), taking its cell (and warning on the standard error if the input has another); the input's cell still sets the grid of PME and the reference of restraints. One of a minimization gives the positions only, and the run begins at step 0. |
| `[output]` | `trajectory` | Positions, in DCD (`.dcd`, Å) or in the compressed XTC of GROMACS (`.xtc`, nm to a thousandth), by the extension of the name (D141). |
| | `trajectory_format` | `AUTO` (the default, from the extension), `DCD`, or `XTC`. |
| | `pull_coordinates` | A file of the terms over the centers of groups at every energy of the log: the step, the time, and for each term its coordinates (`r`, `dx`, `dy`, `dz` in Å, or `theta`), its energy (kcal/mol), and its force, along the distance and on the second center, or $-\partial E/\partial\theta$; continued with the run (D145). |
| | `checkpoint` | The checkpoint (D26), written every `checkpoint_interval` steps in place of the one before, which stays as `<checkpoint>.prev` (D132), and at the end of a minimization. `mdir run --continue` continues the run from it (Section 2.7). |
| | `energy_interval`, `trajectory_interval`, `checkpoint_interval` | Steps between the rows of the log, the frames, and the checkpoints (Section 2.2). The intervals nest, either way for energies and frames. |
| `[energy]` | `cutoff` | The cutoff of `md.neighborhood` (Å). |
| | `pairlist_distance` | The reach of the neighbor structures; the skin is `pairlist_distance − cutoff`. |
| | `pruned_distance` | A dual list (D114): the loops over pairs take an inner list, pruned from the structure of `pairlist_distance` with this reach, between `cutoff` and `pairlist_distance`, whenever a particle has moved too far for it. Needs `neighbor_structure = "GROUPS"`; not with `rebuild_interval`. The default, none, keeps one list. |
| | `rebuild_interval` | **Opt-in, not a default.** 0 (the default): a structure is rebuilt when a particle has moved half the skin, tested at every step. `n`: rebuilt every `n` steps and not tested in between, so it may miss pairs within the cutoff (D88). The run warns at the start, on the standard error and in the log, and reports at the end how many rebuilds found a structure no longer valid, with a warning if any did. |
| | `switch_distance` | For terms in the control file, `truncation(switch, from = ...)`; equal to `cutoff`: no switching. With a topology, where a force switch begins. |
| | `lennard_jones_modifier` | `NONE`, `POTENTIAL_SHIFT` (`truncation(shift)`), `FORCE_SWITCH` (`truncation(force_switch, from = switch_distance)`), or, with a topology only, `POWER_FORCE_SWITCH`, the force switch of Steinbach and Brooks that CHARMM force fields take. With a topology the modifier applies to the Lennard-Jones only, and needs `dispersion_correction = "NONE"` (D121). |
| | `implicit_solvent` | `NONE`, `OBC1`, or `OBC2`: generalized Born from the radii and the screening of a topology of Amber, with `solvent_dielectric` (78.5), `solute_dielectric` (1), and `surface_area_energy` (kcal/mol/Å², 0 for no nonpolar term); with `electrostatics = "CUTOFF"` (D144). |
| | `electrostatics` | With a topology: `CUTOFF`, `PME`, or `REACTION_FIELD`, the field of a dielectric beyond the cutoff acting on every pair of charges within it and on the excluded pairs as well, with `reaction_field_dielectric`, its relative permittivity, or 0 for a conductor (D140). |
| | `coulomb_modifier` | With PME: `NONE`, or `POTENTIAL_SHIFT`, the direct sum shifted to zero at the cutoff. |
| | `dispersion_correction` | `NONE` or `ENERGY_PRESSURE`; also in `[[energy.pair]]`. |
| | `[[energy.pair]]` | A pair term, given by an expression (D16, D22). With a topology, over its pairs that are not excluded, in `r` (Å), `q1`, `q2`, `sigma`, `epsilon` of the pair, `sigma1`, `sigma2`, `epsilon1`, `epsilon2` of each particle (Å, kcal/mol), `coulomb`, the time `t` (ps, D145), and constants, truncated as the Lennard-Jones; `groups = [mask, mask]` keeps the pairs between two masks of Amber (D137). |
| | `[[energy.bond]]`, `[[energy.angle]]`, `[[energy.dihedral]]` | With a topology: a term over tuples of 2, 3, or 4 of its particles, given by an expression in `r` (Å) or `theta` (radians), with `name`, `expression`, `particles` (lists of particle numbers, from 1), and parameters, a number for all tuples or a list of one for each (D136). With `groups` in place of `particles`, 2, 3, or 4 masks of Amber, a term over the centers of the groups, weighted by mass or, with `weighting = "NONE"`, alike; a bond takes `dx`, `dy`, `dz` as well (D139). The time `t` in ps may enter the expression: a reference that moves at a rate (D145). Restraints of distances, angles, and dihedrals, flat-bottomed with `max`, are such terms. |
| | `[[energy.function]]` | A function of one argument that every expression may call by its `name`: `values` at evenly spaced points from `min` to `max`, a natural cubic spline between them and zero outside, or with `periodic = true` a periodic spline, the first and last value equal and the argument taken modulo `max - min` (D138). |
| | `[[energy.type]]` | A type of particle: its mass and its parameters. |
| | `[[energy.pair_override]]` | Parameters of a term for one pair of types. |
| `[pme]` | `tolerance`, `beta`, `max_spacing`, `grid`, `order`, `influence` | Particle mesh Ewald (D71): $\beta$ from $\operatorname{erfc}(\beta r_c) = \texttt{tolerance}$ or given; the grid from the largest spacing or given as three numbers of points; the order of the B-splines, 4, 6, or 8; the influence function, `SPME` or `OPTIMAL`. |
| `[dynamics]` | `integrator` | `VELOCITY_VERLET` or `LEAPFROG`: the `dyn.program` (D76). |
| | `time_step`, `steps` | In ps, and the number of steps of the run, counted from the step it begins at: 0, or the step of the checkpoint of `[input]`. `mdir run --continue` continues the run until it has taken them (D129). |
| | `seed` | Of the initial velocities and of the coupling. |
| | `center_of_mass_interval` | Steps between removals of the motion of the center of mass; with a thermostat, when it acts. |
| `[minimize]` | `method`, `steps`, `initial_step` | `STEEPEST_DESCENT`, the number of steps, and the first step (Å) (D73). Instead of `[dynamics]`. |
| `[ensemble]` | `ensemble` | `NVE`, `NVT` (with `[thermostat]`), or `NPT` (with `[thermostat]` and `[barostat]`). |
| | `temperature`, `pressure` | K, of the initial velocities and the bath; atm, with `NPT`. |
| `[thermostat]` | `method`, `time_constant`, `friction`, `interval` | `V-RESCALE`, stochastic velocity rescaling, with `time_constant` in ps; or `LANGEVIN`, Langevin dynamics by the middle scheme in every step, with `friction` in 1/ps (D135); steps between its actions (10 by default), under `LANGEVIN` those of the removal of the motion of the center of mass and of the barostat (10 by default with a barostat, otherwise none). Langevin dynamics keeps no momentum: its degrees of freedom have no three for the center of mass unless `center_of_mass_interval` removes its motion, and its log no conserved energy. |
| `[barostat]` | `method`, `time_constant`, `compressibility`, `coupling`, `work`, `interval` | `C-RESCALE`, stochastic cell rescaling (D72, D77); ps; 1/atm; `ISOTROPIC`; `TROTTER` (the default; D92), `TROTTER_FIRST_ORDER` (its energy from the virial before the scaling only, a virial less a period), `EXACT`, or `FIRST_ORDER`, which count the work of the barostat in the conserved energy and leave the trajectory alone; under a fast change of the volume the Trotter count drifts by a term of first order in the time step, so such a run is checked with `EXACT`; the steps of the thermostat. |
| `[constraints]` | `hydrogen_bonds`, `rigid_water`, `water_residues`, `analytic_bonds` | SHAKE and RATTLE on the bonds of hydrogen; SETTLE on the waters in double precision, M-SHAKE on their three bonds below it (D112); the names of the residues of water (by default WAT for Amber, TIP3 for CHARMM). `analytic_bonds = true` opts into the checked quadratic projection for one-bond groups (D128); the default is `false`. |
| `[[restraints]]` | `selection`, `force_constant`, `reference_scaling` | A mask of Amber, and kcal/mol/Å² (D74); under a barostat, `"CENTER"` (the default) scales the center of the references with the cell and keeps their shape, `"ALL"` scales each reference with the cell (D124). |
| `[boundary]` | `type`, `box` | `PERIODIC`, or `NONE`, a run without a periodic cell, which places a cell around the particles that keeps every image beyond the reach of the neighbor structures and stops if they spread too far, and refuses PME, a barostat, the correction for the dispersion, and `box` (D142); the edges of the cell (Å), without a topology or with one of CHARMM, whose coordinates have no cell; for CHARMM also its angles α, β, γ (degrees), the cell taken from CHARMM's symmetric frame into MDIR's lower-triangular one (docs/triclinic-m2.md). |
| `[execution]` | `target`, `threads`, `precision` | `CPU` or `GPU`; the threads of the CPU; `SINGLE`, `MIXED`, or `DOUBLE`. |
| | `neighbor_capacity` | Neighbors that a neighbor structure holds per particle at first; a build that finds more makes room. Absent: estimated from the configuration. |
| | `fast_math` | Whether kernels are rewritten in ways that change rounding: the distance in powers of its square, fused multiply-adds, and in f32 approximate divisions and an approximation of erfc within 3.3e-7 (D90). The default is `true`. |
| | `spatial_order` | Whether the run keeps the particles in the order of their positions (D44). The default is `true`. The files of the run are in the order of the input either way. A particle takes the place of its anchor, the oxygen of its water or the center of its group of SHAKE, so that the members of a group of the constraints stay next to each other (D110). |
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

The syntax is that of the custom forces of OpenMM (D22, D136): numbers,
names, `+ - * / ^`, parentheses, the functions `sqrt`, `exp`, `log`,
`sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`,
`abs`, `floor`, `ceil`, `erf`, `erfc`, `step` (1 for x ≥ 0), `delta` (1 for
x = 0), `square`, `cube`, `recip`, `sec`, `csc`, `cot`, `min`, `max`,
`atan2(y, x)`, and `select(x, a, b)` (a if x ≠ 0, else b), the tabulated
functions of `[[energy.function]]` (D138), and definitions of names after
semicolons: `k*d^2; d = r - r0`. The terms over tuples, over centers, and
over the pairs of a topology may use the time `t` in ps, that of the end
of the step whose forces they give (D145).

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
| `TEMPERATURE` | K | $2 K_T / (N_f k_B)$, with $N_f = 3N_{m>0} - N_c - 3$ degrees of freedom (the particles with mass, less the constraints and the motion of the center of mass) and $K_T$ as below |
| `VIRIAL` | kcal/mol | The trace of the MDIR virial $\mathsf W = \sum \mathbf d_{ij} \otimes \mathbf K(i, j)$, which is positive for repulsion (B8). Other packages print other quantities under this name: the GROMACS virial is $-\mathsf W / 2$ [[GromacsManual2025]](references.md#gromacsmanual2025). |
| `PRESSURE` | atm | $(2 K_P + \operatorname{tr}\mathsf W) / (3V)$, with $K_P$ as below |
| `CONSERVED` | kcal/mol | With a thermostat or a barostat: the total energy plus what the couplings have taken from the system |
| `VOLUME` | Å³ | With a barostat: the volume of the cell |

The potential energy and the virial include the correction for the
dispersion beyond the cutoff and the energy of the background that
neutralizes a net charge under particle mesh Ewald, both at the volume of
the row; the virial includes those of the constraints, of the virtual
sites, and of the restraints. Before the first row the log lists the
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
and the log takes $K$ for the temperature and the pressure as well
(D45, amended).

| Quantity | Kinetic energy | Reason |
|---|---|---|
| Total energy | $K$ | The sum with the potential energy varies least: for argon with a step of 20 fs, by 0.05 kcal/mol, against 0.10 and 0.12 with the other two. |
| Temperature | $K_T = (K + 2 K_\text{half}) / 3$, the mean of the three times | The terms of the order $\Delta t^2$ cancel. |
| Pressure | $K_P = K_\text{half}$ | The positions of the steps satisfy the virial theorem with this kinetic energy. |

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
| Readers of systems | PDB: positions and names (`System.cpp`). Amber: the topology (`prmtop`, also of the format before Amber 7, with dihedrals of several terms given by negative periodicities, D133) with CMAP, and coordinates and velocities from `inpcrd` or `rst7` (`Amber.cpp`, `CMap.cpp`). GROMACS: `top` and `itp` with `#include`, `#define`, `#ifdef`, wildcards of dihedral types, and the order of LEaP for the atoms of dihedrals when the force field asks for it, and coordinates from `gro` (`Gromacs.cpp`) | |
| Selections | The masks of Amber for the atoms of restraints (`Selection.cpp`) | |
| Builder | The description of the run to a module (`Builder.cpp`) | |
| Energy expressions | The syntax of D22 to a kernel (`Expression.cpp`) | A parser of expressions |
| Compile and run | Passes in the process, the execution engine (`tools/mdir/Run.cpp`) | MLIR libraries, linked into the tool |
| Output | `mdrt.host_call` to functions of the driver that the execution engine registers: `mdrtWriteEnergies`, `mdrtWriteFrame`, `mdrtWriteCheckpoint`, `mdrtWriteMinimization` (`Output.cpp`) | |
| Writers | Trajectories in DCD and XTC (`Trajectory.cpp`, D141); checkpoints in H5MD (`Checkpoint.cpp`) | HDF5 for H5MD |
| Initial velocities | From `temperature` and `seed`: normal numbers from xoshiro256** seeded by splitmix64, without the components along the constraints and with the center of mass at rest, scaled to the temperature (`System.cpp`) | |

### 2.5 What the driver builds

`mdir emit control.toml` prints the module that the driver builds,
`mdir emit control.toml --stage=lowered` the module that is executed, and
`--stage=pipeline` the passes between them.

| Part of the module | From |
|---|---|
| `md.potential @energy` | `[energy]`: one `md.sum_relation` for each pair term, with the truncation and the cutoff |
| `dyn.program @step` | `integrator`, with the constraints, the virtual sites, and the restraints. A second program, `@step_energy`, returns the energy and the virial as well, for the last step before an output. With the barostat of Trotter type, `@step_virial`, `@step_trotter`, and `@step_trotter_energy` take the last two steps of a period (design-m1.md, Section 11.4). |
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
/h5md                         version, author, creator
/particles/all/box            dimension, boundary, edges
/particles/all/position       step, time, value
/particles/all/velocity       step, time, value
/particles/all/force          step, time, value     with velocity Verlet
/particles/all/id             the numbers of the particles
/particles/all/species        the types of the particles
/particles/all/mass
/parameters/mdir              format, integrator, velocity_offset,
                              precision, timestep, seed, barostat_state;
                              the run that wrote it: first_step, part,
                              trajectory, frames, bath (D129, D130)
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
| The checkpoint before stays as `<checkpoint>.prev`, a second name made before the rename (D132). | A checkpoint that is damaged after it was written leaves one to go back to; the name of the checkpoint holds a complete state at every moment. |
| It records the step that its run began at, its part, the trajectory and the frames written to it, and the energy that the coupling has taken. | `mdir run --continue` continues the run to its `steps`, its trajectory, and its conserved energy (Section 2.7). |
| The particles are in the order of the input, whatever order the run keeps them in. | The file does not depend on the plan of the run. The run that continues puts the particles in order where it begins, and arrives at the order of the run that was not interrupted (D44). |

A run that continues from a checkpoint arrives at the state of the run that
was not interrupted, bit for bit. This holds on the CPU and on a GPU, in
every precision mode, and for both integrators; the tests compare the
states. The two runs must have the same `checkpoint_interval`.

A run cannot continue with another integrator: the velocities of the two
are not of the same time. It cannot continue in another box.

`mdir checkpoint file.h5` describes a checkpoint, and `mdir checkpoint
first.h5 second.h5` compares the states of two. `mdir checkpoint
--print=positions|velocities|forces file.h5` writes a line for each
particle in the order of the input: its number, its mass, and the three
numbers of the field (nm, nm/ps, kJ/mol/nm).

### 2.7 Runs longer than a job

A run on a cluster outlasts the wall time of a job. The same command line
starts it and continues it until it is complete:

```sh
mdir run --continue --max-walltime 23:50 md.toml
```

| Option | What it does |
|---|---|
| `--continue` | Continues the run from the checkpoint of `[output]` until it has taken `steps` steps from the step it began at (D129). Without a checkpoint the run begins; with one that holds the last step it says that the run is complete and exits with 0. It refuses a checkpoint of another time step or seed, and steps that remain if they are not whole intervals of the outputs and of the coupling. Raising `steps` extends a run. |
| `--no-append` | With `--continue`, writes the frames that follow to `<trajectory>.partNNNN.dcd` (or `.xtc`), NNNN the part of the run; later continuations append to that part. Without it, the frames are appended to the trajectory that the checkpoint counts them in, after the frames past the checkpoint are removed (D130). |
| `--max-walltime <time>` | Stops at the last checkpoint that leaves time, within `<time>` from the start of `mdir`, for one more interval between checkpoints as long as the longest so far (D131). In hours (`23.5`) or as `H:MM[:SS]`. |

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
run it began from is not changed.

## 3. Decided

| # | Question | Decision |
|---|---|---|
| 1 | The tables and keywords of the control file | As in Section 1 (D35) |
| 2 | The units of the control file | Å, kcal/mol, ps (D36) |
| 3 | The trajectory format | DCD first (D37); XTC as well (D141) |
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
| Trajectory in the XTC format | Not implemented |
| Velocities in the trajectory, `dcdvelfile` | Not implemented |
