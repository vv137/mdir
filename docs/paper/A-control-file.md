# Appendix A. The control file

M2 Python API preparation (D187) is described in
[python-m2.md](../python-m2.md). It introduces no control-file keys or
changes to the checkpoint and output contracts listed here.

A run is described by a control file in TOML. `mdir check FILE` reads it
and reports the system, planned run, outputs, and warnings (A.5);
`mdir run FILE` compiles and runs it, and
`mdir run --continue FILE` continues it from its checkpoint, over as many
jobs as it takes (A.3);
`mdir emit FILE` prints the program, as built or as lowered;
`mdir template amber` and `mdir template md` print a commented file to
start from. The table below lists every keyword; the example after it is
the output of `mdir template amber` at the commit of this paper.

## A.1 Keywords

An unknown keyword reports the file, line, and table, lists the table's
valid keywords, and points to the templates. A unique nearest spelling
within two character insertions, deletions, or replacements is suggested
as a correction; the input is never corrected automatically. These checks
also apply to table names. User-defined numeric parameters in tables of
terms and types remain valid. A parameter of a term cannot use a name
supplied by that term, a declared per-particle parameter (including its
place suffix where applicable), a built-in or tabulated function, or a
declared `lambda_<name>` component, even when the expression does not use
that name (D188). Rename a colliding parameter and its
uses in the expression. This check does not change keys or file formats.

| Table | Keyword | Meaning |
|---|---|---|
| `[input]` | `topology` | The topology: of Amber (`.prmtop`, `.parm7`), GROMACS (`.top`), or CHARMM (`.psf`, of the XPLOR kind, with the names of the types). Absent for a run from a PDB file, whose terms are in `[energy]`. |
| | `coordinates` | The positions and the cell: of Amber (`.inpcrd`, `.rst7`), GROMACS (`.gro`), CHARMM (`.crd`, standard or extended; it has no cell), or a PDB file, in which the name of an atom selects its type. The reference of restraints. |
| | `format` | `AUTO` (the default: from the names of the files), `AMBER`, `GROMACS`, `CHARMM`, or `PDB`. A CHARMM force field runs from its own files (D122) or from a topology of GROMACS (D121). |
| | `parameters` | With a PSF: the files of topology (`.rtf`), parameters (`.prm`), and streams (`.str`), in the order that CHARMM reads them; a later file replaces what an earlier one defines. |
| | `include_paths`, `defines` | With a GROMACS topology: the directories of `#include` and the names that `#define` gives. |
| | `checkpoint` | The checkpoint of an earlier run, whose state the run begins from, at its step and time (D129), taking its cell (and warning on the standard error if the input has another); the input's cell still sets the grid of PME and the reference of restraints. It takes the forces, the barostat state, and the thermostat chain of the checkpoint only where the physics and the coupling of the two runs are the same; otherwise it notes each difference and evaluates them at its first step (D172). One of a minimization gives the positions only, and the run begins at step 0. |
| `[output]` | `log` | The log in a file as well as on the standard output: every line that the run prints there, from its start; a continued run appends to it (D149). |
| | `manifest` | Optional execution history in version-1 JSON Lines (D168): build and input hashes, resolved settings, device, warnings, and start/end events. Disabled if omitted; fresh runs back up an existing file, continuations append history, and `--no-append` uses the part filename. |
| | `energy` | The rows of the log as a file of columns: a line of names, a line of units, and a row for each output (D149). With a barostat, `volume` (Å³) and `area_xy` (Å²), the xy face spanned by the first two cell vectors, follow the energy columns (D170); this is membrane area only when the membrane normal is along z, and is not area per lipid. |
| | `trajectory` | Positions, in DCD (`.dcd`, Å) or in the compressed XTC of GROMACS (`.xtc`, nm to a thousandth), by the extension of the name (D141). |
| | `trajectory_format` | `AUTO` (the default, from the extension), `DCD`, or `XTC`. |
| | `pull` | A file of columns of the terms over the centers of groups at every energy of the log: the step, the time, and for each term its coordinates (`r`, `dx`, `dy`, `dz` in Å, or `theta`), its energy (kcal/mol), and its force, along the distance and on the second center, or $-\partial E/\partial\theta$; continued with the run (D145, D149). |
| | `free_energy` | A file of columns of `[free_energy]` at every energy of the log: the step, the time, $\partial U/\partial\lambda_m$ of each component (`dHdl.<name>`), and $U(\boldsymbol\lambda^{(k)}) - U(\boldsymbol\lambda)$ for every state $k$ (`dU.<k>`), in kcal/mol, the input of thermodynamic integration and of MBAR (`scripts/free-energy.py`); continued with the run (D161). |
| | `observables` | A file of columns at every energy of a run of dynamics: the step, the time, and for each term that gives `observe`, in the order of the file, its energy (`<term>.energy`, kcal/mol) and its derivative in each constant it lists (`<term>.d_<constant>`, kcal/mol per unit of the constant); needs a term that observes, and a term that observes needs it (D189). |
| | `checkpoint` | The checkpoint (D26), written every `checkpoint_interval` steps in place of the one before, which stays as `<checkpoint>.prev` (D132), and at the end of a minimization. `mdir run --continue` continues the run from it (A.3). |
| | `energy_interval`, `trajectory_interval`, `checkpoint_interval` | Steps between the rows of the log, the frames, and the checkpoints. The intervals nest, either way for energies and frames. |
| `[energy]` | `cutoff` | The cutoff of `md.neighborhood` (Å). |
| | `pairlist_distance` | The reach of the neighbor structures; the skin is `pairlist_distance − cutoff`. |
| | `pruned_distance` | A dual list (D114): the loops over pairs take an inner list, pruned from the structure of `pairlist_distance` with this reach, between `cutoff` and `pairlist_distance`, whenever a particle has moved too far for it. Needs `neighbor_structure = "GROUPS"`; not with `rebuild_interval`. The default, none, keeps one list. |
| | `rebuild_interval` | **Opt-in, not a default.** 0 (the default): a structure is rebuilt when a particle has moved half the skin, tested at every step. `n`: rebuilt every `n` steps and not tested in between, so it may miss pairs within the cutoff (D88). The run warns at the start, on the standard error and in the log, and reports at the end how many rebuilds found a structure no longer valid, with a warning if any did. |
| | `switch_distance` | For terms in the control file, `truncation(switch, from = ...)`; equal to `cutoff`: no switching. With a topology, where the selected force or potential switch begins. |
| | `lennard_jones_modifier` | `NONE`, `POTENTIAL_SHIFT` (`truncation(shift)`), `FORCE_SWITCH` (`truncation(force_switch, from = switch_distance)`), or, with a topology only, `POWER_FORCE_SWITCH`, the force switch of Steinbach and Brooks that CHARMM force fields take. With a topology the modifier applies to the Lennard-Jones only, and needs `dispersion_correction = "NONE"` (D121). `SQUARED_DISTANCE_SWITCH` (topology only, D175) selects the cubic potential switch in squared distance of CHARMM VSWITCH [[Brooks1983]](references.md#brooks1983), including 1-4 pairs; it requires `0 < switch_distance < cutoff` and refuses custom pair additions and LJPME. The plain-cutoff dispersion correction is refused because it omits the contribution removed inside the switching interval; use `dispersion_correction = "NONE"`. |
| | `implicit_solvent` | `NONE`, `HCT`, `OBC1`, or `OBC2`: generalized Born from the radii and the screening of a topology of Amber, or with `born_radii = "MBONDI2"` those of mbondi2 by element for any topology, with `solvent_dielectric` (78.5), `solute_dielectric` (1), `surface_area_energy` (kcal/mol/Å², 0 for no nonpolar term), `salt_concentration` (mol/L of a 1:1 salt at the temperature of [ensemble], 0 for none), and `born_radius_cutoff` (Å, the integral of the descreening cut there; 0, the default, takes every pair within the cutoff); with `electrostatics = "CUTOFF"` (D144, D152). |
| | `electrostatics` | With a topology: `CUTOFF`, `PME`, or `REACTION_FIELD`, the field of a dielectric beyond the cutoff acting on every pair of charges within it and on the excluded pairs as well, with `reaction_field_dielectric`, its relative permittivity, or 0 for a conductor (D140). |
| | `coulomb_modifier` | With PME: `NONE`, or `POTENTIAL_SHIFT`, the direct sum shifted to zero at the cutoff. |
| | `lennard_jones` | With a topology and a periodic cell: `CUTOFF` (default), or `PME`, the dispersion $-c_ic_j/r^6$ of every pair and image summed on a grid of `[lj_pme]`, with $c_i = 2\sqrt{\varepsilon_i}\sigma_i^3$ of the type, and each pair within the cutoff given its own Lennard-Jones (D162). It takes `lennard_jones_modifier` `NONE` or `POTENTIAL_SHIFT`, which shifts its whole direct term, and no correction for the dispersion. |
| | `dispersion_correction` | `NONE` or `ENERGY_PRESSURE`; also in `[[energy.pair]]`. Off with `lennard_jones = "PME"`, which refuses `ENERGY_PRESSURE`. |
| | `[[energy.pair]]` | A pair term, given by an expression (D16, D22). With a topology, over its pairs that are not excluded, in `r` (Å), `q1`, `q2`, `sigma`, `epsilon` of the pair, `sigma1`, `sigma2`, `epsilon1`, `epsilon2` of each particle (Å, kcal/mol), `coulomb`, the time `t` (ps, D145), the components of `[free_energy]` as `lambda_<name>` (D161), constants, and the parameters of each particle as `w1` and `w2` (D165), truncated as the Lennard-Jones; `groups = [mask, mask]` keeps the pairs between two masks of Amber (D137). `observe`, a list of constants of the term given as one number (`[]` for none), writes its energy and its derivatives in them to `[output] observables` (D189). |
| | `[[energy.triplet]]` | Without a topology: a term over the triplets centered on each particle (D160), each center with each unordered pair of its neighbors within `cutoff` (Å, required, at most that of [energy]), given by an expression in `r12`, `r13` (the legs from the center, particle 1, to the ends 2 and 3, Å), `r23` (the far leg, not cut), `theta` (the angle at the center), the time `t` (ps), numbers of the term, and parameters of every `[[energy.type]]` by the suffix of the place (`sigma1` of the center); the energy must not change when 2 and 3 are exchanged. The term must vanish at its cutoff, as the three-body term of Stillinger and Weber does. CPU only. |
| | `[[energy.bond]]`, `[[energy.angle]]`, `[[energy.dihedral]]` | With a topology: a term over tuples of 2, 3, or 4 of its particles, given by an expression in `r` (Å) or `theta` (radians), with `name`, `expression`, `particles` (lists of particle numbers, from 1), and parameters, a number for all tuples or a list of one for each (D136), and those of each particle by its place, `w1` to `wN` (D165). With `groups` in place of `particles`, 2, 3, or 4 masks of Amber, a term over the centers of the groups, weighted by mass or, with `weighting = "NONE"`, alike; a bond takes `dx`, `dy`, `dz` as well (D139). The time `t` in ps may enter the expression: a reference that moves at a rate (D145); so may the components of `[free_energy]`, `lambda_<name>`, whose derivatives `[output] free_energy` writes (D161). `observe`, a list of constants of the term given as one number (`[]` for none), writes its energy and its derivatives in them to `[output] observables` (D189). |
| | `[[energy.compound]]` | With a topology: a term over tuples of 2 to 9 of its particles (D165), as the custom compound bond force of OpenMM, given by an expression in `distance(pa, pb)` (Å), `angle(pa, pb, pc)`, and `dihedral(pa, pb, pc, pd)` (radians) of any of the places `p1` to `pN` of a tuple, the time `t` (ps), declared components `lambda_<name>` of `[free_energy]` (D188), parameters, a number for all tuples or a list of one for each, and the parameters of each particle by place, `w1` to `wN`; with `name`, `expression`, and `particles`, lists of N particle numbers from 1, all of one length. A name with two underscores is reserved. `observe`, a list of constants of the term given as one number (`[]` for none), writes its energy and its derivatives in them to `[output] observables` (D189). |
| | `[[energy.external]]` | With a topology: a term of the absolute positions of single particles, given by an expression in `x`, `y`, `z` (Å), the charge `q` (e), the time `t` (ps), the components `lambda_<name>` of `[free_energy]` (D161), the parameters of each particle by their names (D165), and parameters, a number for every particle or a list of one for each, over the particles of `selection`, a mask of Amber, or of `particles`, their numbers from 1 (D148): walls, fields, restraints of any shape. `scaling = "NONE"` keeps it fixed in space, its virial $\sum_i \mathbf x_i \otimes \mathbf F_i$; `"CELL"` takes its positions in the frame of the cell, scaled to the cell of the input, without a virial; a run at constant pressure must give it (D154). `observe`, a list of constants of the term given as one number (`[]` for none), writes its energy and its derivatives in them to `[output] observables` (D189). |
| | `[[energy.function]]` | A function that every expression may call by its `name`: of one argument, `values` at evenly spaced points from `min` to `max`, a natural cubic spline between them and zero outside, or with `periodic = true` a periodic spline, the first and last value equal and the argument taken modulo `max - min` (D138). Of two or three arguments (D165), `values` as nested lists, `values[i][j]` the value at the i-th point of the first argument and the j-th of the second, and `min` and `max` as lists of one number for each argument: in each cell the bicubic or tricubic patch that matches the values and the derivatives at its corners, the derivatives taken from splines along the axes as OpenMM takes them, and zero outside; with `periodic = true` periodic along every argument, at least four points along each and the last equal to the first. With `discrete = true` and no `min`, `max`, or `periodic`: the value at the nearest point, the arguments rounded to whole numbers from 0 and clamped to the table, whose derivative is 0, as the discrete functions of OpenMM. `values_file` with `shape = [nx]`, `[nx, ny]`, or `[nx, ny, nz]` reads a whitespace-separated grid in place of `values`, the last argument varying fastest, with `#` comments and exactly the product of the sizes in finite decimal numbers; paths are relative to the control file (D178). |
| | `[[energy.parameter]]` | With a topology: a parameter of each particle (D165), as the per-particle parameters of the custom forces of OpenMM, with `name` (letters, digits, and `_`, not ending in a digit, and none of the names that the terms give) and `value` for the particles of `selection`, a mask of Amber, or of `particles`, their numbers from 1, or for every particle without either; or `values`, one for each particle of the topology. The entries of one name apply in the order of the file, a later one over an earlier one on its particles, and every particle must have a value. A pair term takes the parameter `w` as `w1` and `w2`, a term over tuples as `w1` to `wN` by the place of the particle, and a term of the positions as `w`. Without a topology, the parameters of `[[energy.type]]` give values by type. |
| | `[[energy.type]]` | A type of particle: its mass and its parameters. |
| | `[[energy.pair_override]]` | Parameters of a term for one pair of types. |
| `[free_energy]` | `couple` | A mask of Amber of whole molecules that the components `coulomb` and `vdw` decouple from the rest, the interactions within the selection kept at full strength (D161). Not with implicit solvent or `lennard_jones = "PME"`; the Lennard-Jones has a plain cutoff or `lennard_jones_modifier = "POTENTIAL_SHIFT"`, not a switch. |
| | `state` | The state that the run samples, from 0. |
| | `soft_core_alpha`, `soft_core_power` | $\alpha_\text{sc}$ (0.5) and $p$ (1 or 2) of the soft-core of the Lennard-Jones of the decoupled pairs, $r_A^6 = \alpha_\text{sc}\,\sigma^6\lambda_\text{V}^p + r^6$ (Beutler et al.); `soft_core_alpha` must be positive. |
| | `[free_energy.lambdas]` | The components of $\boldsymbol\lambda$, each a list of one value for every state: `coulomb` and `vdw` from 0, the system of the topology, to 1, the selection decoupled (missing ones are 0); any other `name` is the parameter `lambda_name` of the expressions of `[energy]`, as `lambda_coulomb` and `lambda_vdw` are. A state with `vdw > 0` and `coulomb < 1` warns (`charged_soft_core`). |
| `[pme]` | `tolerance`, `beta`, `max_spacing`, `grid`, `order`, `influence` | Particle mesh Ewald (D71): β from `erfc(β r_c) = tolerance` or given; the grid from the largest spacing or given as three numbers of points; the order of the B-splines, 4, 6, or 8; the influence function, `SPME` or `OPTIMAL`. |
| `[lj_pme]` | `tolerance`, `beta`, `max_spacing`, `grid`, `order` | Particle mesh Ewald for the dispersion, with `lennard_jones = "PME"` (D162): β from $g(\beta r_c)$ = `tolerance` (default $10^{-3}$), $g(x) = e^{-x^2}(1 + x^2 + x^4/2)$, or given; the grid from the largest spacing (1.2 Å) or given; the order, 4, 6, or 8. |
| `[dynamics]` | `integrator` | `VELOCITY_VERLET`, `LEAPFROG`, or `BROWNIAN`: the `dyn.program` (D76). `BROWNIAN` is overdamped Langevin dynamics, the positions only, with `friction` and the temperature of `ensemble = "NVT"`, without a `[thermostat]` (D163b). Its log and file of energies have no total, kinetic energy, temperature, or conserved energy (the velocities are displacements over a step), and its pressure takes the kinetic energy of the bath, $\tfrac12N_fk_BT$. **With constraints, the step must be short:** the mobility $1/(m\gamma)$ moves hydrogens furthest, and a step that moves the lightest atom more than 0.005 nm at random, $\sqrt{2k_BT\Delta t/(m\gamma)}$, is warned of (`brownian_step`); a dipeptide in water at $\gamma$ = 50/ps needs about 0.1 fs, where 2 fs breaks the constraints. |
| | `friction` | With `BROWNIAN` only: the friction $\gamma$ in 1/ps; the mobility of a particle is $1/(m\gamma)$. |
| | `time_step`, `steps` | In ps, and the number of steps of the run, counted from the step it begins at: 0, or the step of the checkpoint of `[input]`. `mdir run --continue` continues the run until it has taken them (D129). |
| | `seed` | Of the initial velocities and of the coupling. |
| | `center_of_mass_interval` | Steps between removals of the motion of the center of mass; with a thermostat, when it acts. |
| `[minimize]` | `method`, `steps`, `initial_step` | `STEEPEST_DESCENT`, the number of steps, and the first step (Å) (D73). Instead of `[dynamics]`. |
| `[ensemble]` | `ensemble` | `NVE`, `NVT` (with `[thermostat]`), or `NPT` (with `[thermostat]` and `[barostat]`). |
| | `temperature`, `pressure` | K, of the initial velocities and the bath; atm, with `NPT`. |
| `[thermostat]` | `method`, `time_constant`, `friction`, `chain_length`, `interval` | `V-RESCALE`, stochastic velocity rescaling, with `time_constant` in ps; `NOSE-HOOVER`, a Nosé–Hoover chain of `chain_length` thermostats (3 by default) with the period `time_constant` in ps, whose energy the conserved energy holds and the checkpoints keep (D163a); a `time_constant` below 20 periods of `interval` is warned of (`short_thermostat_period`); or `LANGEVIN`, Langevin dynamics by the middle scheme in every step, with `friction` in 1/ps (D135), either key in the other an error; steps between its actions (10 by default), under `LANGEVIN` those of the removal of the motion of the center of mass and of the barostat (10 by default with a barostat, otherwise none). Langevin dynamics keeps no momentum, so its degrees of freedom have no three for the center of mass unless `center_of_mass_interval` removes it, and its log has no conserved energy. |
| `[barostat]` | `method`, `time_constant`, `compressibility`, `coupling`, `work`, `interval`, `compressibility_z`, `surface_tension`, `surfaces` | `C-RESCALE`, stochastic cell rescaling (D72, D77); ps; 1/atm; `ISOTROPIC`, `SEMI_ISOTROPIC` (x and y scale together from the mean of their pressures, z on its own, D119), or `ANISOTROPIC` (each axis from its own pressure and noise, with `compressibility` one number or three, those of x, y, and z, 0 keeping an edge; the pressures of the axes at the end of the log; a liquid has no stiffness of shape and its edges wander, so this is for solids and cells under unequal stresses, D163c); `TROTTER` (the default; D92), `TROTTER_FIRST_ORDER` (its energy from the virial before the scaling only, a virial less a period), `EXACT`, or `FIRST_ORDER` (not with `SEMI_ISOTROPIC` or `ANISOTROPIC`), which count the work of the barostat in the conserved energy and leave the trajectory alone (a run that changes its volume fast is checked with `EXACT`: the Trotter count then drifts by a term of first order in the time step); the steps of the thermostat; with `SEMI_ISOTROPIC`, the compressibility of z in 1/atm (0 keeps the height; that of x and y by default), the tension of each surface normal to z in dyn/cm (0 by default), and their number (2). |
| `[constraints]` | `hydrogen_bonds`, `rigid_water`, `water_residues`, `analytic_bonds` | SHAKE and RATTLE on the bonds of hydrogen; SETTLE on the waters in double precision, M-SHAKE on their three bonds below it (D112); the names of the residues of water (by default WAT for Amber, TIP3 for CHARMM); without `rigid_water` the waters of an Amber or CHARMM topology run flexible, with a warning (D153). `analytic_bonds = true` enables the checked quadratic one-bond projection (D128); the default is `false`. |
| `[[restraints]]` | `selection`, `force_constant`, `reference_scaling` | A mask of Amber, and kcal/mol/Å² (D74); under a barostat, `"CENTER"` (the default) scales the center of the references with the cell and keeps their shape, `"ALL"` scales each reference with the cell (D124). |
| `[boundary]` | `type`, `box` | `PERIODIC`, or `NONE`, a run without a periodic cell, which places a cell around the particles that keeps every image beyond the reach of the neighbor structures and stops if they spread too far, and refuses PME, a barostat, the correction for the dispersion, and `box` (D142); the edges of the cell (Å), without a topology or with one of CHARMM, whose coordinates have no cell; for CHARMM also its angles α, β, γ (degrees), the cell taken from CHARMM's symmetric frame into MDIR's lower-triangular one (docs/triclinic-m2.md). |
| `[execution]` | `target`, `threads`, `precision` | `CPU` or `GPU`; the threads of the CPU; `SINGLE`, `MIXED`, or `DOUBLE`. |
| | `neighbor_capacity` | Neighbors that a neighbor structure holds per particle at first; a build that finds more makes room. Absent: estimated from the configuration. |
| | `fast_math` | Whether kernels are rewritten in ways that change rounding: the distance in powers of its square, fused multiply-adds, and in f32 approximate divisions and an approximation of erfc within 3.3e-7 (D90). The default is `true`. |
| | `spatial_order` | Whether the run keeps the particles in the order of their positions (D44). The default is `true`. The files of the run are in the order of the input either way. A particle takes the place of its anchor, the oxygen of its water or the center of its group of SHAKE, so that the members of a group of the constraints stay next to each other (D110). |
| | `neighbor_structure` | `MATRIX` (the default): a row of neighbors for each particle, each pair computed for both. `GROUPS`: groups of 16 particles that share a list, each pair computed once, for the loops whose values all have an exchange contract (D89); on a device, not in the deterministic mode. |
| | `deterministic` | Whether every sum is added in an order that the threads do not decide, so that a run gives the same bits from run to run on the same binary and hardware (D84). The default is `false`: a device then adds the charges of PME with floating-point atomics. On the CPU every sum has a fixed order in either mode, independent of `threads` (D171). |

**File-backed grids (D178).** The last argument varies
fastest, exactly as in inline nested lists. For a 2×3 grid,
`values_file = "surface.dat"` and `shape = [2, 3]` read this file:

```text
# Values at the three points of the second argument, for each first point.
11 12 13
21 22 23
```

This is equivalent to `values = [[11, 12, 13], [21, 22, 23]]`:
`f(0, 0) = 11`, `f(0, 2) = 13`, and `f(1, 2) = 23` for a discrete
function; a continuous function gives the same values at the corresponding
grid points of `min` and `max`. Line breaks do not set the dimensions.
The reader converts to the internal table order before interpolation.
The same rule extends to three arguments. Relative paths resolve against
the control file. Only finite decimal numbers are accepted, with whitespace
separators and `#` comments; count, shape, periodic endpoints, and output
collisions are checked before compilation. Resolved part output names and
the previous-checkpoint path are protected as well. The manifest hashes the file,
and continuation compares the loaded grid rather than its filename.

## A.2 A run from an Amber topology

```toml
[input]
topology    = "system.prmtop"   # of Amber (tleap, ParmEd)
coordinates = "system.inpcrd"   # and the box; the reference of restraints
# format    = "AUTO"            # AUTO (from the names), AMBER, GROMACS, PDB
# checkpoint = "earlier.h5"     # the state that the run begins from, at its
#                               # step; one of a minimization gives only
#                               # positions

[output]
# manifest = "run.jsonl"        # execution provenance and continuation history
log                 = "run.log" # the log as well as on the standard output
energy              = "run.energy"  # the rows of the log as columns
trajectory          = "run.dcd" # positions, in DCD or XTC (.xtc)
checkpoint          = "run.h5"  # the state; mdir run --continue goes on
#                               # from it, and the one before is run.h5.prev
# pull              = "run.pull"    # terms over the centers of groups
# free_energy       = "run.dhdl"    # dH/dλ and the energies of the states
#                                   # of [free_energy] at every energy
energy_interval     = 5000      # steps between energies in the log
trajectory_interval = 5000      # steps between frames
checkpoint_interval = 50000     # steps between checkpoints

[energy]
cutoff            = 9.0         # of the direct terms (Å)
pairlist_distance = 10.0        # reach of the neighbor structures (Å)
# pruned_distance = 0           # reach of the inner list of a dual list,
                                # pruned from the structure; needs GROUPS (Å)
# rebuild_interval = 0          # 0: rebuild when a particle has moved half
                                # the skin (default); N: every N steps, not
                                # tested between; may miss pairs (opt-in)
electrostatics    = "PME"       # PME, CUTOFF
coulomb_modifier  = "POTENTIAL_SHIFT"  # NONE, POTENTIAL_SHIFT: the direct
                                       # sum shifted to zero at the cutoff
# dispersion_correction = "ENERGY_PRESSURE"  # NONE, ENERGY_PRESSURE
# lennard_jones   = "CUTOFF"    # CUTOFF, PME: the dispersion beyond the
                                # cutoff on a grid, with no correction

# Particle mesh Ewald; every entry has a default.
# [pme]
# tolerance   = 1.0e-5          # erfc(β r_c), which gives β
# beta        = 0.35            # β (1/Å), instead
# max_spacing = 1.2             # largest spacing of the grid (Å)
# grid        = [48, 48, 48]    # the grid, instead
# order       = 4               # of the B-splines: 4, 6, 8
# influence   = "SPME"          # SPME, OPTIMAL (as sander)

# Particle mesh Ewald of the dispersion, with lennard_jones = "PME";
# every entry has a default.
# [lj_pme]
# tolerance   = 1.0e-3          # g(β r_c) = exp(−x²)(1 + x² + x⁴/2), x = β r_c
# beta        = 0.33            # β (1/Å), instead
# max_spacing = 1.2             # largest spacing of the grid (Å)
# grid        = [48, 48, 48]    # the grid, instead
# order       = 4               # of the B-splines: 4, 6, 8

# Alchemical states: a selection decoupled from the rest (D161).
# [free_energy]
# couple = ":LIG"               # a mask of Amber of whole molecules
# state  = 0                    # the state of this run, from 0
# soft_core_alpha = 0.5         # of the Lennard-Jones of decoupled pairs
# [free_energy.lambdas]         # a value for each state; 0 is the topology
# coulomb = [0.0, 0.5, 1.0, 1.0, 1.0]
# vdw     = [0.0, 0.0, 0.0, 0.5, 1.0]

[dynamics]
integrator = "VELOCITY_VERLET"  # VELOCITY_VERLET, LEAPFROG (velocities
                                # half a step behind), BROWNIAN (with
                                # friction, 1/ps, and no [thermostat])
time_step  = 0.002              # ps
steps      = 500000             # of the run; --continue runs to them
seed       = 314159             # of the velocities and the coupling
# center_of_mass_interval = 10  # steps between removals of the motion of
#                               # the center of mass: with a thermostat,
#                               # when it acts

[ensemble]
ensemble    = "NPT"             # NVE, NVT (with [thermostat]), NPT (with
                                # [thermostat] and [barostat])
temperature = 300.0             # of the velocities and the bath (K)
pressure    = 1.0               # atm, with NPT

[thermostat]
method        = "V-RESCALE"     # stochastic velocity rescaling,
                                # "LANGEVIN", Langevin dynamics, or
                                # "NOSE-HOOVER", a Nose-Hoover chain
time_constant = 0.5             # ps, with V-RESCALE or NOSE-HOOVER
# friction    = 1.0             # 1/ps, with LANGEVIN
# chain_length = 3              # thermostats, with NOSE-HOOVER
interval      = 10              # steps between its actions

[barostat]
method        = "C-RESCALE"     # stochastic cell rescaling
time_constant = 2.0             # ps
# compressibility = 4.56e-5     # 1/atm (4.5e-5 /bar)
# coupling = "ISOTROPIC"        # ISOTROPIC; SEMI_ISOTROPIC: x and y
#                               # together, z on its own; ANISOTROPIC:
#                               # each axis on its own, with a number or
#                               # three for compressibility
# compressibility_z = 4.56e-5   # 1/atm, of z with SEMI_ISOTROPIC (0 keeps
#                               # the height); compressibility by default
# surface_tension = 0.0         # dyn/cm, of each surface normal to z,
# surfaces        = 2           # with SEMI_ISOTROPIC
# work     = "TROTTER"          # TROTTER: the scaling within the drift of
#                               # a step, its energy from the virials before
#                               # and after; TROTTER_FIRST_ORDER: from the
#                               # one before, a virial less; EXACT: the energy of each
#                               # scaling from the scaled positions, whose
#                               # forces the next step takes; FIRST_ORDER:
#                               # from the virial, as GROMACS does
# interval = 10                 # steps between its actions: those of the
#                               # thermostat

[constraints]
hydrogen_bonds = true           # SHAKE and RATTLE on the bonds of hydrogen
analytic_bonds = false          # checked quadratic solve for one-bond groups
rigid_water    = true           # rigid waters: SETTLE in DOUBLE,
                                # M-SHAKE on their three distances below
# water_residues = ["WAT"]      # names of the residues of rigid water

[boundary]
type = "PERIODIC"               # the box is that of the coordinates;
                                # NONE: no periodic cell

[execution]
target    = "GPU"               # CPU, GPU
precision = "MIXED"             # SINGLE, MIXED, DOUBLE
# threads = 1                   # for the target CPU
# neighbor_structure = "MATRIX"  # MATRIX, GROUPS: groups of 16 that share
                                # a list, each pair once

# A minimization instead of dynamics: steepest descent.
# [minimize]
# method       = "STEEPEST_DESCENT"
# steps        = 2000
# initial_step = 0.1            # Å

# Restraints to the positions of 'coordinates', any number of them.
# [[restraints]]
# selection         = "!:WAT & !@H*"  # a mask of Amber
# force_constant    = 10.0            # kcal/mol/Å²
# reference_scaling = "CENTER"        # or "ALL", under a barostat
```

## A.3 Runs over more than one job

`mdir run` takes three options for a run that outlasts a job on a
cluster (docs/driver-m0.md, Section 2.7). Without `--continue`, a run
keeps each output of an earlier run that has its name as `#<name>.<n>#`
before it writes, at most 99 of each (D149):

| Option | Meaning |
|---|---|
| `--continue` | Continues the run from the checkpoint of `[output]` until it has taken its `steps`, counted from the step it began at; without a checkpoint the run begins, and a complete run exits with 0 (D129). It refuses a checkpoint of other physics or coupling, naming each change (D172). What may change: `[execution]` and `pairlist_distance`, `pruned_distance`, and `rebuild_interval` of `[energy]`, with a note in the log; `[output]`, within the rules of its intervals; and a larger `steps`, which extends the run. A change of anything else is a new run, which begins from the checkpoint as `checkpoint` of `[input]`. |
| `--no-append` | With `--continue`, writes the outputs that follow (the log, manifest, files of columns, and frames) to `<name>.partNNNN<ext>` rather than appending them to the files of the run, which are first cut to the checkpoint (D130, D149). |
| `--max-walltime <time>` | Stops at the last checkpoint that leaves time for one more interval between checkpoints, in hours or as `H:MM[:SS]` (D131). |

SIGTERM and SIGINT stop a run at its next checkpoint, and a second signal
stops it at once. A run that stops at a checkpoint exits with 75
(`EX_TEMPFAIL`), from which `--continue` goes on exactly; 0 is a run that
is complete, and 1 an error.

## A.4 Templates of the standard pipeline

`mdir template minimize`, `nvt`, `npt`, and `production` print the stages
of `examples/ala3` with placeholder paths `system.prmtop` and
`system.inpcrd` (D147). They supply restrained minimization (2000 steps),
50 ps at constant volume and 300 K, 100 ps at 1 atm with weaker
restraints, and 1 ns at 1 atm without restraints. The checkpoint chain is
`min.h5`, `nvt.h5`, `npt.h5`, and `md.h5`; production writes `md.dcd`
every 1 ps. Dynamics uses a step of 2 fs, and all stages use PME and
constraints on hydrogen bonds and water. The settings are embedded from
the examples when MDIR is built.

Save each printed control file, edit the input paths, restraint selection,
run length, and execution settings, then execute the files in order with
`mdir run --continue FILE` (A.3). The defaults select a GPU in mixed
precision and need HDF5 checkpoints; `target = "CPU"` uses the host.
The selection `!:WAT & !@H*` and water constraints assume a solute in
water named `WAT`; other systems need their selections and residue names
adapted. The files retain the input coordinates as the restraint reference.
`mdir template md` and `amber` continue to provide the reference files.

## A.5 Preflight

`mdir check FILE` reports the ensemble, configured length in steps and ns,
time step in ps, PME, constraints, target, and precision for every input
format, beside the topology or particle summary (D151). It lists energies,
trajectory, checkpoint, and pulling-coordinate outputs with their paths,
formats, intervals, and whether they are enabled and already exist.
Minimization has an iteration count and a checkpoint at the end, with no
physical duration; a nonperiodic system has no physical cell or density.

Warnings flag existing enabled output files, fixed neighbor-rebuild
intervals, absent energy reports or checkpoints, a missing input checkpoint,
and missing HDF5 or CUDA build support. They include a way to address the
condition and keep exit status 0; invalid input returns 1. The command
does not write files, compile, probe a device, or load a checkpoint. It
describes the stage's configured length, not the work remaining after a
checkpoint, and keeps automatic PME grid and beta settings explicit.

`mdir check FILE --json` writes one JSON object, including on input errors.
`schema_version` is 1; `ok` distinguishes successful input validation from
errors. The report has `system`, `run`, `outputs`, `warnings`, and `errors`;
the first three are absent on failure. Warning entries have `code` and
`message`, while errors are strings. Numeric fields name their units,
such as `duration_ns` and `pressure_atm`; unavailable values are null.
Warnings in this mode are in the report, not on standard error. The full
schema is in [driver-m0.md](../driver-m0.md), Section 1.4.

## A.6 The checkpoint file

A checkpoint is an H5MD 1.1 file [[deBuyl2014]](references.md#debuyl2014)
in the units inside MDIR (nm, ps, u, kJ/mol), every number in 64 bits.
Format 1, below, is the contract of release 0.1.0 (D173):
a reader refuses a newer format and a file of a development build before
the release, and a later format comes with a conversion from the one
before it.

| Path | Holds |
|---|---|
| `/h5md` | `version` [1, 1]; `author/name`; `creator/name` "MDIR" and `creator/version`, MDIR's version and commit |
| `/particles/all/box` | `dimension` 3, `boundary` "periodic" or "none", `edges`: three, or the 3×3 matrix of a triclinic cell, a row per cell vector |
| `/particles/all/{position,velocity,force}` | `step`, `time`, and `value` of shape (1, N, 3); the forces are those the next step begins with |
| `/particles/all/{id,species,mass}` | The numbers of the particles from 0, their types, their masses |
| `/parameters/mdir` | Attributes `format` (1), `state_sha256`, `integrator`, `velocity_offset` (−0.5 with leapfrog), `precision`, `timestep`, `seed`, `first_step`, `part`, `outputs_part`, `trajectory`, `frames`, `bath`, `periodic` |
| `/parameters/mdir/barostat_state` | With a barostat that scales every step, nine numbers (D92, D119) |
| `/parameters/mdir/thermostat_state` | With a Nose–Hoover chain, its positions and then its velocities (D163a) |
| `/parameters/mdir/fingerprint/{physics,coupling,execution}` | What defined the run, a line per entry, its name and value separated by a tab (D172) |

`state_sha256` covers the state and the fingerprint, and is checked on
every read; a file changed after it was written is refused, with a pointer
to `<checkpoint>.prev`. The file is flushed to stable storage before it
takes its name, and its directory after. `mdir checkpoint
--print=fingerprint` lists the fingerprint, and docs/driver-m0.md, Section
2.6, says what a run that takes a checkpoint compares.
