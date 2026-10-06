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
| `[output]` | `log` | The log in a file as well as on the standard output: every line that the run prints there, from its start; a continued run appends to it (D149). |
| | `manifest` | Optional execution history in version-1 JSON Lines (D168): build and input hashes, resolved settings, device, warnings, and start/end events. Disabled if omitted; fresh runs back up an existing file, continuations append history, and `--no-append` uses the part filename. |
| | `energy` | The rows of the log as a file of columns: a line of names, a line of units, and a row for each output (D149). With a barostat, `volume` (Å³) and `area_xy` (Å²), the xy face spanned by the first two cell vectors, follow the energy columns (D170). |
| | `trajectory` | Positions, in DCD (`.dcd`, Å) or in the compressed XTC of GROMACS (`.xtc`, nm to a thousandth), by the extension of the name (D141). |
| | `trajectory_format` | `AUTO` (the default, from the extension), `DCD`, or `XTC`. |
| | `pull` | A file of columns of the terms over the centers of groups at every energy of the log: the step, the time, and for each term its coordinates (`r`, `dx`, `dy`, `dz` in Å, or `theta`), its energy (kcal/mol), and its force, along the distance and on the second center, or $-\partial E/\partial\theta$; continued with the run (D145, D149). |
| | `free_energy` | A file of columns of `[free_energy]` at every energy of the log: the step, the time, $\partial U/\partial\lambda_m$ of each component (`dHdl.<name>`), and $U(\boldsymbol\lambda^{(k)}) - U(\boldsymbol\lambda)$ for every state $k$ (`dU.<k>`), left out with one state, where they are identically 0 (D190), in kcal/mol; continued with the run (D161). |
| | `observables` | A file of columns at every energy of a run of dynamics: the step, the time, and for each term that gives `observe`, in the order of the file, its energy (`<term>.energy`, kcal/mol) and its derivative in each constant it lists (`<term>.d_<constant>`, kcal/mol per unit of the constant); needs a term that observes, and a term that observes needs it (D189). |
| | `checkpoint` | The checkpoint (D26), written every `checkpoint_interval` steps in place of the one before, which stays as `<checkpoint>.prev` (D132), and at the end of a minimization. `mdir run --continue` continues the run from it (Section 2.7). |
| | `energy_interval`, `trajectory_interval`, `checkpoint_interval` | Steps between the rows of the log, the frames, and the checkpoints (Section 2.2). The intervals nest, either way for energies and frames. |
| `[energy]` | `cutoff` | The cutoff of `md.neighborhood` (Å). |
| | `pairlist_distance` | The reach of the neighbor structures; the skin is `pairlist_distance − cutoff`. |
| | `pruned_distance` | A dual list (D114): the loops over pairs take an inner list, pruned from the structure of `pairlist_distance` with this reach, between `cutoff` and `pairlist_distance`, whenever a particle has moved too far for it. Needs `neighbor_structure = "GROUPS"`; not with `rebuild_interval`. The default, none, keeps one list. |
| | `rebuild_interval` | **Opt-in, not a default.** 0 (the default): a structure is rebuilt when a particle has moved half the skin, tested at every step. `n`: rebuilt every `n` steps and not tested in between, so it may miss pairs within the cutoff (D88). The run warns at the start, on the standard error and in the log, and reports at the end how many rebuilds found a structure no longer valid, with a warning if any did. |
| | `switch_distance` | For terms in the control file, `truncation(switch, from = ...)`; equal to `cutoff`: no switching. With a topology, where the selected force or potential switch begins. |
| | `lennard_jones_modifier` | `NONE`, `POTENTIAL_SHIFT` (`truncation(shift)`), `FORCE_SWITCH` (`truncation(force_switch, from = switch_distance)`), or, with a topology only, `POWER_FORCE_SWITCH`, the force switch of Steinbach and Brooks that CHARMM force fields take. With a topology the modifier applies to the Lennard-Jones only, and needs `dispersion_correction = "NONE"` (D121). `SQUARED_DISTANCE_SWITCH` (topology only, D175) selects the cubic potential switch in squared distance of CHARMM VSWITCH [[Brooks1983]](references.md#brooks1983), including 1-4 pairs; it requires `0 < switch_distance < cutoff` and refuses custom pair additions and LJPME. The plain-cutoff dispersion correction is refused because it omits the contribution removed inside the switching interval; use `dispersion_correction = "NONE"`. |
| | `implicit_solvent` | `NONE`, `HCT`, `OBC1`, or `OBC2`: generalized Born from the radii and the screening of a topology of Amber, or with `born_radii = "MBONDI2"` those of mbondi2 by element for any topology, with `solvent_dielectric` (78.5), `solute_dielectric` (1), `surface_area_energy` (kcal/mol/Å², 0 for no nonpolar term), `salt_concentration` (mol/L of a 1:1 salt at the temperature of [ensemble], 0 for none), and `born_radius_cutoff` (Å, the integral of the descreening cut there; 0, the default, takes every pair within the cutoff); with `electrostatics = "CUTOFF"` (D144, D152). |
| | `electrostatics` | With a topology: `CUTOFF`, `PME`, or `REACTION_FIELD`, the field of a dielectric beyond the cutoff acting on every pair of charges within it and on the excluded pairs as well, with `reaction_field_dielectric`, its relative permittivity, or 0 for a conductor (D140). |
| | `coulomb_modifier` | With PME: `NONE`, or `POTENTIAL_SHIFT`, the direct sum shifted to zero at the cutoff. |
| | `dispersion_correction` | `NONE` or `ENERGY_PRESSURE`; also in `[[energy.pair]]`. |
| | `[[energy.pair]]` | A pair term, given by an expression (D16, D22). With a topology, over its pairs that are not excluded, in `r` (Å), `q1`, `q2`, `sigma`, `epsilon` of the pair, `sigma1`, `sigma2`, `epsilon1`, `epsilon2` of each particle (Å, kcal/mol), `coulomb`, the time `t` (ps, D145), constants, and the parameters of each particle as `w1` and `w2` (D165), truncated as the Lennard-Jones; `groups = [mask, mask]` keeps the pairs between two masks of Amber (D137). `observe`, a list of constants of the term given as one number (`[]` for none), writes its energy and its derivatives in them to `[output] observables` (D189). |
| | `[[energy.triplet]]` | Without a topology: a term over the triplets centered on each particle (D160), each center with each unordered pair of its neighbors within `cutoff` (Å, required, at most that of [energy]), given by an expression in `r12`, `r13` (the legs from the center, particle 1, to the ends 2 and 3, Å), `r23` (the far leg, not cut), `theta` (the angle at the center), the time `t` (ps), numbers of the term, and parameters of every `[[energy.type]]` by the suffix of the place (`sigma1` of the center); the energy must not change when 2 and 3 are exchanged. The term must vanish at its cutoff, as the three-body term of Stillinger and Weber does. CPU only. |
| | `[[energy.bond]]`, `[[energy.angle]]`, `[[energy.dihedral]]` | With a topology: a term over tuples of 2, 3, or 4 of its particles, given by an expression in `r` (Å) or `theta` (radians), with `name`, `expression`, `particles` (lists of particle numbers, from 1), and parameters, a number for all tuples or a list of one for each (D136), and those of each particle by its place, `w1` to `wN` (D165). With `groups` in place of `particles`, 2, 3, or 4 masks of Amber, a term over the centers of the groups, weighted by mass or, with `weighting = "NONE"`, alike; a bond takes `dx`, `dy`, `dz` as well (D139). The time `t` in ps may enter the expression: a reference that moves at a rate (D145). Restraints of distances, angles, and dihedrals, flat-bottomed with `max`, are such terms. `observe`, a list of constants of the term given as one number (`[]` for none), writes its energy and its derivatives in them to `[output] observables` (D189). |
| | `[[energy.compound]]` | With a topology: a term over tuples of 2 to 9 of its particles (D165), as the custom compound bond force of OpenMM, given by an expression in `distance(pa, pb)` (Å), `angle(pa, pb, pc)`, and `dihedral(pa, pb, pc, pd)` (radians) of any of the places `p1` to `pN` of a tuple, the time `t` (ps), parameters, a number for all tuples or a list of one for each, and the parameters of each particle by place, `w1` to `wN`; with `name`, `expression`, and `particles`, lists of N particle numbers from 1, all of one length. A name with two underscores is reserved. `observe`, a list of constants of the term given as one number (`[]` for none), writes its energy and its derivatives in them to `[output] observables` (D189). |
| | `[[energy.external]]` | With a topology: a term of the absolute positions of single particles, given by an expression in `x`, `y`, `z` (Å), the charge `q` (e), the time `t` (ps), the parameters of each particle by their names (D165), and parameters, a number for every particle or a list of one for each, over the particles of `selection`, a mask of Amber, or of `particles`, their numbers from 1 (D148): walls, fields, restraints of any shape. `scaling = "NONE"` keeps it fixed in space, its virial $\sum_i \mathbf x_i \otimes \mathbf F_i$; `"CELL"` takes its positions in the frame of the cell, scaled to the cell of the input, without a virial; a run at constant pressure must give it (D154). `observe`, a list of constants of the term given as one number (`[]` for none), writes its energy and its derivatives in them to `[output] observables` (D189). |
| | `[[energy.function]]` | A function that every expression may call by its `name`: of one argument, `values` at evenly spaced points from `min` to `max`, a natural cubic spline between them and zero outside, or with `periodic = true` a periodic spline, the first and last value equal and the argument taken modulo `max - min` (D138). Of two or three arguments (D165), `values` as nested lists, `values[i][j]` the value at the i-th point of the first argument and the j-th of the second, and `min` and `max` as lists of one number for each argument: in each cell the bicubic or tricubic patch that matches the values and the derivatives at its corners, the derivatives taken from splines along the axes as OpenMM takes them, and zero outside; with `periodic = true` periodic along every argument, at least four points along each and the last equal to the first. With `discrete = true` and no `min`, `max`, or `periodic`: the value at the nearest point, the arguments rounded to whole numbers from 0 and clamped to the table, whose derivative is 0, as the discrete functions of OpenMM. `values_file` with `shape = [nx]`, `[nx, ny]`, or `[nx, ny, nz]` reads a whitespace-separated grid in place of `values`, the last argument varying fastest, with `#` comments and exactly the product of the sizes in finite decimal numbers; paths are relative to the control file (D178). |
| | `[[energy.parameter]]` | With a topology: a parameter of each particle (D165), as the per-particle parameters of the custom forces of OpenMM, with `name` (letters, digits, and `_`, not ending in a digit, and none of the names that the terms give) and `value` for the particles of `selection`, a mask of Amber, or of `particles`, their numbers from 1, or for every particle without either; or `values`, one for each particle of the topology. The entries of one name apply in the order of the file, a later one over an earlier one on its particles, and every particle must have a value. A pair term takes the parameter `w` as `w1` and `w2`, a term over tuples as `w1` to `wN` by the place of the particle, and a term of the positions as `w`. Without a topology, the parameters of `[[energy.type]]` give values by type. |
| | `[[energy.type]]` | A type of particle: its mass and its parameters. |
| | `[[energy.pair_override]]` | Parameters of a term for one pair of types. |
| `[pme]` | `tolerance`, `beta`, `max_spacing`, `grid`, `order`, `influence` | Particle mesh Ewald (D71): $\beta$ from $\operatorname{erfc}(\beta r_c) = \texttt{tolerance}$ or given; the grid from the largest spacing or given as three numbers of points; the order of the B-splines, 4, 6, or 8; the influence function, `SPME` or `OPTIMAL`. |
| `[dynamics]` | `integrator` | `VELOCITY_VERLET`, `LEAPFROG`, or `BROWNIAN`: the `dyn.program` (D76). `BROWNIAN` is overdamped Langevin dynamics, the positions only, with `friction` and the temperature of `ensemble = "NVT"`, without a `[thermostat]` (D163b). Its log and file of energies have no total, kinetic energy, temperature, or conserved energy (the velocities are displacements over a step), and its pressure takes the kinetic energy of the bath, $\tfrac12N_fk_BT$. **With constraints, the step must be short:** the mobility $1/(m\gamma)$ moves hydrogens furthest, and a step that moves the lightest atom more than 0.005 nm at random, $\sqrt{2k_BT\Delta t/(m\gamma)}$, is warned of (`brownian_step`); a dipeptide in water at $\gamma$ = 50/ps needs about 0.1 fs, where 2 fs breaks the constraints. |
| | `friction` | With `BROWNIAN` only: the friction $\gamma$ in 1/ps; the mobility of a particle is $1/(m\gamma)$. |
| | `time_step`, `steps` | In ps, and the number of steps of the run, counted from the step it begins at: 0, or the step of the checkpoint of `[input]`. `mdir run --continue` continues the run until it has taken them (D129). |
| | `seed` | Of the initial velocities and of the coupling. |
| | `center_of_mass_interval` | Steps between removals of the motion of the center of mass; with a thermostat, when it acts. |
| `[minimize]` | `method`, `steps`, `initial_step` | `STEEPEST_DESCENT`, the number of steps, and the first step (Å) (D73). Instead of `[dynamics]`. |
| `[ensemble]` | `ensemble` | `NVE`, `NVT` (with `[thermostat]`), or `NPT` (with `[thermostat]` and `[barostat]`). |
| | `temperature`, `pressure` | K, of the initial velocities and the bath; atm, with `NPT`. |
| `[thermostat]` | `method`, `time_constant`, `friction`, `chain_length`, `interval` | `V-RESCALE`, stochastic velocity rescaling, with `time_constant` in ps; `NOSE-HOOVER`, a Nosé–Hoover chain of `chain_length` thermostats (3 by default) with the period `time_constant` in ps, whose energy the conserved energy holds and the checkpoints keep (D163a); a chain driven so far from equilibrium that the factorization of its action no longer follows it stops the run with an error (D[nhc-nan]); a `time_constant` below 20 periods of `interval` is warned of (`short_thermostat_period`); or `LANGEVIN`, Langevin dynamics by the middle scheme in every step, with `friction` in 1/ps (D135); steps between its actions (10 by default), under `LANGEVIN` those of the removal of the motion of the center of mass and of the barostat (10 by default with a barostat, otherwise none). Langevin dynamics keeps no momentum: its degrees of freedom have no three for the center of mass unless `center_of_mass_interval` removes its motion, and its log no conserved energy. |
| `[barostat]` | `method`, `time_constant`, `compressibility`, `coupling`, `work`, `interval` | `C-RESCALE`, stochastic cell rescaling (D72, D77); ps; 1/atm, or three with `ANISOTROPIC`; `ISOTROPIC`, `SEMI_ISOTROPIC` (D119), or `ANISOTROPIC`, each axis on its own (D163c); `TROTTER` (the default; D92), `TROTTER_FIRST_ORDER` (its energy from the virial before the scaling only, a virial less a period), `EXACT`, or `FIRST_ORDER`, which count the work of the barostat in the conserved energy and leave the trajectory alone; under a fast change of the volume the Trotter count drifts by a term of first order in the time step, so such a run is checked with `EXACT`; the steps of the thermostat. |
| `[constraints]` | `hydrogen_bonds`, `rigid_water`, `water_residues`, `analytic_bonds` | SHAKE and RATTLE on the bonds of hydrogen; SETTLE on the waters in double precision, M-SHAKE on their three bonds below it (D112); the names of the residues of water (by default WAT for Amber, TIP3 for CHARMM); without `rigid_water` the waters of an Amber or CHARMM topology run flexible, with a warning (D153). `analytic_bonds = true` opts into the checked quadratic projection for one-bond groups (D128); the default is `false`. |
| `[[restraints]]` | `selection`, `force_constant`, `reference_scaling` | A mask of Amber, and kcal/mol/Å² (D74); under a barostat, `"CENTER"` (the default) scales the center of the references with the cell and keeps their shape, `"ALL"` scales each reference with the cell (D124). |
| `[boundary]` | `type`, `box` | `PERIODIC`, or `NONE`, a run without a periodic cell, which places a cell around the particles that keeps every image beyond the reach of the neighbor structures and stops if they spread too far, and refuses PME, a barostat, the correction for the dispersion, and `box` (D142); the edges of the cell (Å), without a topology or with one of CHARMM, whose coordinates have no cell; for CHARMM also its angles α, β, γ (degrees), the cell taken from CHARMM's symmetric frame into MDIR's lower-triangular one (docs/triclinic-m2.md). |
| `[execution]` | `target`, `threads`, `precision` | `CPU` or `GPU`; the threads of the CPU; `SINGLE`, `MIXED`, or `DOUBLE`. |
| | `neighbor_capacity` | Neighbors that a neighbor structure holds per particle at first; a build that finds more makes room. Absent: estimated from the configuration. |
| | `fast_math` | Whether kernels are rewritten in ways that change rounding: the distance in powers of its square, fused multiply-adds, and in f32 approximate divisions and an approximation of erfc within 3.3e-7 (D90). The default is `true`. |
| | `spatial_order` | Whether the run keeps the particles in the order of their positions (D44). The default is `true`. The files of the run are in the order of the input either way. A particle takes the place of its anchor, the oxygen of its water or the center of its group of SHAKE, so that the members of a group of the constraints stay next to each other (D110). |
| | `neighbor_structure` | `MATRIX` (the default): a row of neighbors for each particle, each pair computed for both. `GROUPS`: groups of 16 particles that share a list, each pair computed once, for the loops whose values all have an exchange contract (D89); on a device, not in the deterministic mode. |
| | `deterministic` | Whether every sum is added in an order that the threads do not decide, so that a run gives the same bits from run to run on the same binary and hardware (D84). A step that writes energies then moves the particles as one that does not, and in mixed precision the forces that the steps carry, and those of the checkpoints, are stored in f64; on a device a sum with a product as an operand is a fused multiply-add by the formula alone (D201, D204). The default is `false`: a device then adds the charges of PME with floating-point atomics. |

A keyword of a pair term or of a type that is not listed here names a
number: a parameter of the type, or a constant of the term, that the
expression uses.

A parameter that has the same value for all types is a constant of the
kernel. A parameter that differs is a field, and `mixing` says how the
parameter of a pair follows from those of its two particles:
`lorentz-berthelot` [[Lorentz1881]](references.md#lorentz1881), [[Berthelot1898]](references.md#berthelot1898), `geometric`, or a table with `arithmetic` or
`geometric` for each parameter.

An unknown keyword is an error that names the file, line, and table,
lists the table's valid keywords, and points to `mdir template md` and
`mdir template amber`. It suggests the nearest valid keyword when that
match is unique and within two character insertions, deletions, or
replacements; it does not change the input. The templates print control
files with every keyword; the keywords follow the code, not earlier
versions, until a milestone is released.

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
the terms over centers as columns, the trajectory in DCD or XTC, and the
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
| `run` | Kind, ensemble, method, steps, `time_step_ps`, `duration_ns`, `temperature_kelvin`, `pressure_atm`, thermostat and barostat coupling, cutoff, PME, constraints, execution, and input checkpoint path |
| `run.pme` | `null` when disabled; otherwise `grid_points` (three nulls when automatic), `beta_inverse_angstrom` (`null` when automatic), `order`, `max_spacing_angstrom`, and `tolerance` |
| `outputs` | Entries with `kind` (`log`, `energy`, `pull`, `free_energy`, `trajectory`, `checkpoint`, `manifest`), `path`, `format`, `interval_steps`, `enabled`, `at_end`, `count` and `count_of` (`rows`, `frames`, `checkpoints`; `count` is null where the run decides, as in a minimization), `exists`, and `backup`, the name under which a run keeps the file that exists (D149), or null; `path` is null for an unconfigured file, and for the log when it goes to standard output only |
| `warnings` | Objects with `code` and `message`: `backup_limit`, `constant_expression`, `empty_selection`, `flexible_water`, `long_time_step`, `short_thermostat_period` (D163a), `brownian_step` (D163b), `charged_soft_core` (D161), `unused_parameter`, `fixed_rebuild_interval`, `no_checkpoint`, `no_energies`, `missing_input_checkpoint`, `hdf5_unavailable`, or `gpu_unavailable` |
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

`mdir doctor [--target=all|cpu|gpu]` checks the installation (D155). It
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

Each run uses the same executable in a child process, with a timeout of
120 seconds, so a failed CPU run does not prevent checking the GPU. The
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

The log goes to the standard output and, with `log` in `[output]`, to that
file as well (D149, Section 2.8); `energy` writes its rows as columns. Its
first line names the build that wrote it, `MDIR 0.1.0, commit
<12 hexadecimal digits>`, followed by `with uncommitted changes` when the
tree had them (D174).

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
| `AREA_XY` | Å² | With a barostat: the area of the xy face spanned by the first two cell vectors (D170); not area per lipid |

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
and the steps of the rows measure them: they take the first half of the
next step, a kick, a drift, and the constraints of the positions, and the
mean of the kinetic energies of the velocities of the two drifts after
their constraints (D203). The row of step 0 has no step
before it and takes $K$ there. The thermostats and the barostats take the
kinetic energy of the velocities they scale. With rigid waters, a run ends with
`MDIR: the temperatures over N rows, in K: of the solute ..., of the
solvent ...; from the velocities of the steps alone, ... and ...`: the means
of $T$ and of $2K/(N_fk_B)$ over the rows, with errors from ten blocks of
rows, for the solvent, the waters of SETTLE (6 degrees of freedom each),
and the solute, every other particle, ions included; the three degrees of
freedom of the center of mass are shared in proportion
(D203).

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
/h5md                         version [1, 1]; author/name; creator/name
                              "MDIR", creator/version "0.1.0 (<commit>)"
/particles/all/box            dimension, boundary ("periodic" or "none"),
                              edges: three, or the 3x3 matrix of a
                              triclinic cell, a row per vector (nm)
/particles/all/position       step, time, value (1, N, 3)   nm
/particles/all/velocity       step, time, value (1, N, 3)   nm ps-1
/particles/all/force          step, time, value (1, N, 3)   kJ mol-1 nm-1
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
```

This layout is the contract of release 0.1.0, format 1
(D173). A reader checks `format`: a newer one is refused
as written by a newer MDIR, and a file of a development build before the
release, without the fingerprint and the hash, is refused. A later format
comes with a function that converts a file of the format before, so that
every released format stays readable. `state_sha256` is SHA-256 of
everything above but `/h5md` and the ids, in a fixed order; a file whose
bytes changed after it was written is refused, with a pointer to `.prev`.

The units are those inside MDIR and are written with the data: nm, ps, u,
and kJ/mol.

| Rule | Reason |
|---|---|
| A value of 32 bits is stored in 64. | The conversion is exact in both directions, so a run in single precision continues without loss. |
| With velocity Verlet the checkpoint holds the forces. | A step begins with the forces of the step before. Forces that are computed again from the positions differ in their last bits, because a neighbor structure that is built again has another order. |
| With leapfrog the time of the velocities is half a step before that of the positions. | The file says what it holds. |
| Neighbor structures start empty after every checkpoint (R1). | The run that continues builds its structure at the first step. The run that was not interrupted must build there too. |
| The file appears under its name only when it is complete and on stable storage: it is written as `.partial`, flushed and `fsync`ed, renamed, and the directory is `fsync`ed (D173). | A run that ends while it writes leaves the checkpoint before; a crash on a file system that delays its writes (ext4 without `auto_da_alloc`, Lustre, NFS) does not leave an empty file under the name. |
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
| physics | Every key of `[energy]`, `[pme]`, `[lj_pme]`, `[constraints]`, `[restraints]`, `[boundary]`, and `[free_energy]` as the control file writes it, in a canonical form (keys sorted, numbers as the shortest text of their double, so that `9` and `9.0` are one value); SHA-256 of the contents of the files of the topology (Amber, GROMACS with its included files, CHARMM structure and parameters), of the masses, and of the reference of positional restraints |
| coupling | Every key of `[dynamics]` but `steps`, and of `[ensemble]`, `[thermostat]`, and `[barostat]` |
| execution | Every key of `[execution]`, and `pairlist_distance`, `pruned_distance`, and `rebuild_interval` of `[energy]`, which decide how the forces are found and not what they are |

A value longer than 160 characters is recorded as its SHA-256. The
entries are the keys as written, not the values the driver resolves: a
key added with its default value counts as a change. `mdir checkpoint
--print=fingerprint file.h5` lists them.

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
| The trajectory | `trajectory`, `trajectory_format` | every `trajectory_interval` steps | DCD or XTC (D141) |
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
requested neighbor kind and the kinds observed immediately before
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

### Expression parameter names (D188)

The control reader rejects term parameters that collide with names supplied
by their term: coordinates, time, topology properties, declared particle
parameters and their applicable place suffixes, built-in and tabulated
functions, and declared lambda components. The declaration is checked even
when unused, so a constant cannot silently replace a coordinate and its
force. Compound terms accept declared `lambda_<name>` components through
the existing lambda bindings, with forces, component derivatives, and
state energy differences computed by the same differentiation path.
