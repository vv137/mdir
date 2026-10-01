# Appendix A. The control file

A run is described by a control file in TOML. `mdir check FILE` reads it
and prints what it found (particles, types, the degrees of freedom, the
integrator, the target); `mdir run FILE` compiles and runs it;
`mdir emit FILE` prints the program, as built or as lowered;
`mdir template amber` and `mdir template md` print a commented file to
start from. The table below lists every keyword; the example after it is
the output of `mdir template amber` at the commit of this paper.

## A.1 Keywords

| Table | Keyword | Meaning |
|---|---|---|
| `[input]` | `topology` | The topology: of Amber (`.prmtop`, `.parm7`) or GROMACS (`.top`). Absent for a run from a PDB file, whose terms are in `[energy]`. |
| | `coordinates` | The positions and the cell: of Amber (`.inpcrd`, `.rst7`), GROMACS (`.gro`), or a PDB file, in which the name of an atom selects its type. The reference of restraints. |
| | `format` | `AUTO` (the default: from the names of the files), `AMBER`, `GROMACS`, or `PDB`; `CHARMM` is not supported yet. |
| | `include_paths`, `defines` | With a GROMACS topology: the directories of `#include` and the names that `#define` gives. |
| | `checkpoint` | The checkpoint of an earlier run, which the run continues, taking its cell (and warning on the standard error if the input has another); the input's cell still sets the grid of PME and the reference of restraints. One of a minimization gives the positions only. |
| `[output]` | `trajectory` | Positions, in DCD (`.dcd`). |
| | `checkpoint` | The checkpoint (D26), written every `checkpoint_interval` steps in place of the one before, and at the end of a minimization. |
| | `energy_interval`, `trajectory_interval`, `checkpoint_interval` | Steps between the rows of the log, the frames, and the checkpoints. The intervals nest, either way for energies and frames. |
| `[energy]` | `cutoff` | The cutoff of `md.neighborhood` (Å). |
| | `pairlist_distance` | The reach of the neighbor structures; the skin is `pairlist_distance − cutoff`. |
| | `pruned_distance` | A dual list (D114): the loops over pairs take an inner list, pruned from the structure of `pairlist_distance` with this reach, between `cutoff` and `pairlist_distance`, whenever a particle has moved too far for it. Needs `neighbor_structure = "GROUPS"`; not with `rebuild_interval`. The default, none, keeps one list. |
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
| `[barostat]` | `method`, `time_constant`, `compressibility`, `coupling`, `work`, `interval` | `C-RESCALE`, stochastic cell rescaling (D72, D77); ps; 1/atm; `ISOTROPIC`; `TROTTER` (the default; D92), `TROTTER_FIRST_ORDER` (its energy from the virial before the scaling only, a virial less a period), `EXACT`, or `FIRST_ORDER`; the steps of the thermostat. |
| `[constraints]` | `hydrogen_bonds`, `rigid_water`, `water_residues` | SHAKE and RATTLE on the bonds of hydrogen; SETTLE on the waters in double precision, M-SHAKE on their three bonds below it (D112); the names of the residues of water. |
| `[[restraints]]` | `selection`, `force_constant` | A mask of Amber, and kcal/mol/Å² (D74). |
| `[boundary]` | `type`, `box` | `PERIODIC`; the edges of the cell (Å), without a topology. |
| `[execution]` | `target`, `threads`, `precision` | `CPU` or `GPU`; the threads of the CPU; `SINGLE`, `MIXED`, or `DOUBLE`. |
| | `neighbor_capacity` | Neighbors that a neighbor structure holds per particle at first; a build that finds more makes room. Absent: estimated from the configuration. |
| | `fast_math` | Whether kernels are rewritten in ways that change rounding: the distance in powers of its square, fused multiply-adds, and in f32 approximate divisions and an approximation of erfc within 3.3e-7 (D90). The default is `true`. |
| | `spatial_order` | Whether the run keeps the particles in the order of their positions (D44). The default is `true`. The files of the run are in the order of the input either way. A particle takes the place of its anchor, the oxygen of its water or the center of its group of SHAKE, so that the members of a group of the constraints stay next to each other (D110). |
| | `neighbor_structure` | `MATRIX` (the default): a row of neighbors for each particle, each pair computed for both. `GROUPS`: groups of 16 particles that share a list, each pair computed once, for the loops whose values all have an exchange contract (D89); on a device, not in the deterministic mode. |
| | `deterministic` | Whether every sum is added in an order that the threads do not decide, so that a run gives the same bits from run to run on the same binary and hardware (D84). The default is `false`: a device then adds the charges of PME with floating-point atomics. |

## A.2 A run from an Amber topology

```toml
[input]
topology    = "system.prmtop"   # of Amber (tleap, ParmEd)
coordinates = "system.inpcrd"   # and the box; the reference of restraints
# format    = "AUTO"            # AUTO (from the names), AMBER, GROMACS, PDB
# checkpoint = "earlier.h5"     # the state that the run continues from; one
#                               # of a minimization gives only positions

[output]
trajectory          = "run.dcd" # positions, in DCD
checkpoint          = "run.h5"  # the state
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

# Particle mesh Ewald; every entry has a default.
# [pme]
# tolerance   = 1.0e-5          # erfc(β r_c), which gives β
# beta        = 0.35            # β (1/Å), instead
# max_spacing = 1.2             # largest spacing of the grid (Å)
# grid        = [48, 48, 48]    # the grid, instead
# order       = 4               # of the B-splines: 4, 6, 8
# influence   = "SPME"          # SPME, OPTIMAL (as sander)

[dynamics]
integrator = "VELOCITY_VERLET"  # VELOCITY_VERLET, LEAPFROG (velocities
                                # half a step behind)
time_step  = 0.002              # ps
steps      = 500000
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
method        = "V-RESCALE"     # stochastic velocity rescaling
time_constant = 0.5             # ps
interval      = 10              # steps between its actions

[barostat]
method        = "C-RESCALE"     # stochastic cell rescaling
time_constant = 2.0             # ps
# compressibility = 4.56e-5     # 1/atm (4.5e-5 /bar)
# coupling = "ISOTROPIC"        # ISOTROPIC
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
rigid_water    = true           # rigid waters: SETTLE in DOUBLE,
                                # M-SHAKE on their three distances below
# water_residues = ["WAT"]      # names of the residues of rigid water

[boundary]
type = "PERIODIC"               # the box is that of the coordinates

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
# selection      = "!:WAT & !@H*"  # a mask of Amber
# force_constant = 10.0            # kcal/mol/Å²
```
