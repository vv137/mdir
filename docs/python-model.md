# Shared model for M2 (D191)

Native implementation of the first item of `python-m2.md` (#74, PR #75).
This PR provides the native C++ model layer for subsequent Python bindings.
It does not provide an importable Python package. The optional bindings and
lowering contribution is [python-compile.md](python-compile.md)
(D192).

## Contract

Owned `LoadedData` captures Amber, GROMACS, or CHARMM topology, coordinates, optional
velocities, cell, and source contents. Loading neither draws velocities nor
initializes CUDA. Physics (`model::System`) and `InitialState` are separate
values, copied from loaded data. Public quantities use nm, ps, kJ/mol, bar,
amu, K, e, and radians. Cells use the driver's reduced lower-triangular convention.
No mutable arrays are shared between loaded data, physics, or prepared runs.

Typed `Integrator`, `Ensemble`, and `Execution` select a fixed CLI-compatible
schedule for parity testing. Persistent arbitrary segments follow in the
third implementation PR. Execution retains a nonnegative logical device under `CUDA_VISIBLE_DEVICES`.
Device resolution follows in the compilation PR; this layer never changes
the process's CUDA device.

The initial subset is imported classical topology terms (Lennard-Jones,
cutoff Coulomb and PME, harmonic bonds/angles, periodic dihedrals, harmonic
impropers, Urey-Bradley, CMAP, special 1-4 pairs, exclusions, SHAKE/SETTLE,
and the two supported three-parent virtual-site kinds). Velocity Verlet,
leapfrog, minimization, NVE, stochastic velocity rescaling NVT, and isotropic
stochastic cell rescaling NPT are included, on CPU/GPU, mixed/double (the
semi-isotropic and the anisotropic coupling and the works of the barostat
since D[python-barostat], [python-segments.md](python-segments.md)).
Basic custom pair and tuple expressions are included with MD-unit coordinate
and energy adaptation at the boundary. Other expression families, implicit
solvent, free energy, LJPME, alternate baths/barostats, and single precision
are deferred and refused. Existing CLI support stays available.

Shared validation checks loaded array shapes, finite values, type and
particle identities before any preparation indexes them. Prepared object
models use the same constraint resolution, cell checks, and IR builder as
file models. Input and unsupported errors carry distinct native error kinds;
the binding PR maps them to Python exceptions.

No control-file keys, checkpoint formats, output formats, or overwrite
behavior change. Model preparation and IR construction create no reports.

## Future MLIP differentiation

The maintainer requested future PyTorch/JAX automatic differentiation on
2026-10-04. Sampling and differentiable energy/observable evaluation have
separate contracts; this is an extension requirement, not an implementation.
Physics, parameters and initial state remain explicit values apart from
execution ownership. Backend adapters must retain parameter identity and
provide energy/force/virial evaluation with explicit derivative contracts
(VJP/JVP and any required higher derivatives). Do not silently coerce framework
tensors to host arrays in those adapters. Current native vectors are owned
classical input values; future framework tensor and parameter-tree adapters
belong at a separate evaluation boundary.

DLPack carries storage and stream synchronization, not an autograd graph.
Framework integrations require explicit PyTorch autograd or JAX derivative
rules, differentiated arguments, units, dtype/device and lifetime contracts.
Gradient evaluation on a fixed state, parameter differentiation (including
force-matching derivatives), ensemble reweighting, and differentiation through
a trajectory are distinct capabilities to validate separately. The initial
M2 model provides no framework import, gradients or trajectory differentiation;
no PyTorch/JAX dependency is added to classical builds.

CHARMM loading accepts XPLOR PSF, CRD, and ordered RTF/PRM/stream parameter
files. Since CRD contains no cell, periodic loading takes an explicit reduced
cell in nm and rotates symmetric-frame positions through the same helper as
the CLI. A zero cell permits nonperiodic loading. Water recognition defaults
to TIP3 for CHARMM and WAT for Amber; GROMACS preserves its SETTLE records.

## Native signatures and limits

The declarations are in `include/mdir/Driver/Model.h`. `loadAmber`,
`loadGromacs` (with includes/defines), and `loadCharmm` return `LoadedData`.
`makeSystem` strips coordinates/cell from copied topology; `makeState`
returns positions, optional velocities, and the reduced cell. Positions and
velocities are flattened `(N, 3)` f64 host arrays in input order. Topology
contains zero-based particle/type identities. The native values are mutable;
`prepare` deep-copies them. Structural version/stale tracking is required
before the compilation service exposes reusable programs to Python.

Custom pairs initially accept constants and optional two selection masks,
without type mixing; their tails enter the correction for the dispersion as
in the control file (D209), and a term can leave it
([below](#the-correction-for-the-dispersion)). Tuple expressions have
arity two, three or four and one value of each parameter per tuple. Pair and
bond `r` is nm, angle/dihedral `theta` is radians, energy is kJ/mol. At the
legacy builder boundary, coordinate variables and the energy expression are
converted; numerical parameter values retain their stated MD meaning.
PME uses spline order four (six and eight since D[python-pme-fields]) and
automatic or explicit grids (at least eight points on each axis). The object model's defaults are stated in the header;
file defaults remain unchanged. Bath pressures use bar and compressibility
uses inverse bar. The temporary fixed schedule obeys the CLI's coupling
multiples; arbitrary counts belong to the segment PR.

Validation: native CPU/GPU-target mixed/double tests compare semantic IR,
fields, tables, tuple members/parameters, constraints and initial state
against file input. Ownership and malformed input are checked on both paths.
A custom spring at 0.3 nm with rest length 0.2 nm and coefficient
100 kJ/mol/nm² gives 1 kJ/mol (tolerance 1e-13), and its central-difference
derivative gives 20 kJ/mol/nm (tolerance 1e-7). These are analytic checks of
the unit boundary, not a claim of Python runtime execution. The unchanged
numerical builder retains the existing independent term-oracle suite.


Drawn velocities and typed positional restraints extend this model
(D198,
[python-velocities-restraints.md](python-velocities-restraints.md)):
`drawVelocities` prepares the system as `prepare` does and calls the CLI's
`assignVelocities`, and `System::restraints` with `restraintReference`
become the control's `[[restraints]]`.

## Terms of the positions and `observe`

`System.external_terms` holds `mdir.ExternalTerm` values, the control
file's `[[energy.external]]` (D148, D233): an expression in
`x`, `y`, `z` (nm) and the charge `q`, in kJ/mol, over the particles of a
mask or of an array of indices, with constants, and `scaling` under a
barostat (D154). Pair, tuple, and external terms take `observe`, the control
file's key (D189, D232): `None`, or a list of constants whose
derivatives are observed with the term's energy. See
[python-observe.md](python-observe.md).

## The correction for the dispersion

D222 (#161) gives the Python model the two parts of the
control file's `dispersion_correction` that it lacked (D209, D210).

| Python | Control file | Meaning |
|---|---|---|
| `System.dispersion` not set (`dispersion_given` is `False`) | no `dispersion_correction` in `[energy]` | `EnergyPressure`; a pair term whose tail diverges or that reads `t` is left out with the warning `pair_tail_left_out`; off without a periodic cell; off with a switch, with the warning `dispersion_switched` |
| `System.dispersion = DispersionCorrection.EnergyPressure` | `dispersion_correction = "ENERGY_PRESSURE"` | on; such a term is an `InputError` naming it; refused without a periodic cell and with a switch |
| `System.dispersion = DispersionCorrection.None_` | `"NONE"` | off |
| `PairTerm.dispersion = None` (default) | no key in `[[energy.pair]]` | the term follows the system |
| `PairTerm.dispersion = DispersionCorrection.None_` | `dispersion_correction = "NONE"` in the term | the term's tail is left out, silently |
| `PairTerm.dispersion = DispersionCorrection.EnergyPressure` | `"ENERGY_PRESSURE"` in the term | the term asks for its tail: a divergent tail is an error, and so is the system's correction off |

Python's `None` and `DispersionCorrection.None_` differ as an absent key
and `"NONE"` do. Setting `System.dispersion` to `None` restores the default;
reading it gives the value, `EnergyPressure` by default. Leaving a term out
omits only its long-range estimate; its energy, forces, and virial within
the cutoff are unchanged. `mdir.compile` raises `pair_tail_left_out` and
`dispersion_switched` as `UserWarning`s, whose messages, like those of a
refused tail, name `PairTerm.dispersion` rather than the control-file key;
a value other than a `DispersionCorrection` or `None` is a `TypeError`; and `Program.plan["dispersion"]` is
`{"correction", "given", "pair_terms"}`, the last a dict from each pair
term's name to whether its tail is in the correction. A tunable
(D213) of a term left out may take any value; for a term in the
correction, an update that would make its tail diverge is refused, as
before.

The numbers are the builder's: the model hands it the control structure of
the control file. With a switch (`Truncation.Switch`, `ForceSwitch`, ...),
which takes part of the potential below the cutoff that the correction
would leave out, the control file refuses the correction even by default.
The Python model refuses it when it is set and, by default, turns it off
with the warning `dispersion_switched`, so that the default `System`, whose
truncation was `Switch` then, still compiled (the maintainer's choice on PR
#187; since D[python-defaults] the default is a plain cutoff, with the
correction on). Before, it kept the correction under a switch.

Validation (`python-dispersion.test`, `-gpu.test`, `Inputs/python_dispersion.py`),
on the 60 A + 60 B mixture of `pair-dispersion.test` with the term
$-c_8/r^8 - a e^{-r/l}/r^4$ over A-A and A-B, $r_c$ = 12 Å, under a plain
cutoff and `Truncation.Shift`, CPU and GPU, double and mixed:

| Check | Reference | Result | Tolerance |
|---|---|---|---|
| Potential, trace of the virial, and pressure at step 0, with the default, an explicit correction, the term's request, its opt-out, and the correction off | `mdir run` on the same input, its energy file (6 decimals of kcal/mol and atm) | equal to the printed digits | 1.5e-6 relative |
| Explicit correction and term's request against the default | the default | equal to the bit | 0 |
| Default less opt-out: the term's tail, $\nu(4\pi/V)N_\text{pairs}\int_{r_c}^\infty r^2u\,dr$ (and under the shift $+\nu(4\pi/3V)N_\text{pairs}f r_c^3u(r_c)$) | Simpson's rule in $\ln r$ of `check_pair_tail.py` | energy −0.033825540 kcal/mol (cutoff), −0.088177850 (shift), within 3.4e-13 relative | 1e-9 (double), 1e-5 (mixed) |
| Its trace of the virial, $-\nu(4\pi/V)N_\text{pairs}\int r^3u'\,dr$, the same under the shift | the same | −0.270926137 kcal/mol, within 3.5e-13 | the same |
| Its pressure, $\operatorname{tr}\mathsf W/3V$ | the same | −0.191478708 bar, within 2.0e-12 | the same |
| $-c_8/(l^5r^3)$, a divergent tail | `mdir check` refuses it under `ENERGY_PRESSURE` in `[energy]` or in the term | `InputError` naming the term in both; by default one warning, energies equal to the opt-out and to `mdir run` | |
| No cell; a switch | the control file | default off without a cell, explicit refused; explicit with a switch refused; by default with a switch off, one warning, energies equal to those with the correction set off | |
| Messages and types | | a refused tail and the warning name `PairTerm.dispersion`; a string or an integer is a `TypeError` | |
| Tunable exponent $p$ of $-c_8/r^p$ updated from 8 to 3 | | taken by a term left out; refused by a term in the correction | |

The differences of the default and the opt-out agree in mixed precision as
in double: the tails are host constants in double, and the two runs
evaluate the same pairs within the cutoff.

## The groups and the dual list

D245 (#270) gives the Python model the two keys of the control
file that select the neighbor structure of the loops over pairs: the
groups of 16 particles that share a list (D89,
[groups-m1.md](groups-m1.md)) and the dual list (D114). Until then a
Python simulation had the neighbor matrix alone, and the rates of
`mdir run` with the groups could not be had from Python.

| Python | Control file | Values |
|---|---|---|
| `Execution.neighbor_structure` | `[execution] neighbor_structure` | `mdir.NeighborStructure.Matrix` (the default), `mdir.NeighborStructure.Groups` |
| `System.pruned_distance` | `[energy] pruned_distance` | a length in nm or a unit quantity (D200), between `System.cutoff` and `System.pairlist_distance`; `0`, the default, keeps one list |

Each is where the control file has it: `neighbor_structure` is a key of
`[execution]`, and `pruned_distance` a key of `[energy]` beside
`pairlist_distance`, with which it is checked and which is
`System.pairlist_distance`. Both are entries of the group `execution` of a
checkpoint's fingerprint, as `mdir run` files them (D172): they decide how
the forces are found, not what they are. `InitialState.draw_velocities`,
which prepares the system for the default execution, leaves the dual list
out: the draw does not read the lists.

`mdir.compile` refuses what `mdir check` refuses, with its words and
`model` for the path of the file, as `InputError`:

| Case | Message |
|---|---|
| `Groups` with `Target.CPU`, or with `Execution.deterministic` | `model: 'neighbor_structure = "GROUPS"' needs 'target = "GPU"' and not 'deterministic'` |
| `pruned_distance` with the matrix | `model: 'pruned_distance' keeps a dual list, which needs 'neighbor_structure = "GROUPS"'` |
| `pruned_distance` not between the cutoff and the pairlist distance | `model: 'pruned_distance' is not between 'cutoff' and 'pairlist_distance'` |
| a negative or non-finite `pruned_distance` | `System.pruned_distance must be positive, or 0 for one list` |
| a pairlist distance above 0.999 of the least edge of the cell, the least of $a_x, b_y, c_z$ of a triclinic one (D242) | the builder's, with the distances in Å: `'pairlist_distance', 26 Å, exceeds 0.999 of the least edge of the cell, 25.3716 Å, the most that the groups take` |

Which loops take the groups is the choice of `md-exec-choose-neighbors`,
as under `mdir run`: those whose values all have an exchange contract
(D89). The others keep the matrix, and the program then builds both; the
loops of the derivative in the tunables (`System.tunable_gradient`, D230)
are such loops. A structure that stays a matrix keeps one list under a
dual list: only the groups have an inner list to prune. Before, the
refresh of such a matrix kept the test of an inner list that it does not
have, and the lowering refused the program (`'md_exec.reference_positions'
op takes the configuration of the pruning of a dual list, which only a
structure of groups keeps`); `md-exec-choose-neighbors` now takes the
reach and the test of the inner list from the refreshes of the structures
that it leaves as matrices
(`test/Dialect/MDExec/Transforms/choose-neighbors.mlir`).

`Program.plan["neighbor_structure"]` is `"matrix"` or `"groups"`, what was
asked (the `neighbor_structure_requested` of a manifest of `mdir run`),
and `Program.plan["pruned_distance"]` the reach of the inner list in nm,
`0.0` without one. `Program.pipeline` has
`md-exec-choose-neighbors{kind=groups}` and the `prune-skin` of
`md-exec-reuse-neighbors`, so the key of the code that a `Program` keeps
(D236), which holds the pipeline and the module, tells the structures
apart.

**What follows the structure.**

- *A writable borrow.* A commit of a cell asks, for a program with the
  groups, that the pairlist distance be at most 0.999 of the least edge of
  the committed cell (D242, [python-dlpack.md](python-dlpack.md)). The
  check stood after a return for orthorhombic cells and so held for
  triclinic cells alone; no Python program had the groups, so nothing
  reached it. It now holds for both kinds of cell, as D242 states.
- *A barostat.* The runtime fails the part in which the barostat takes an
  edge below the pairlist distance over 0.999, with the message of
  `mdir run` (`SimulationError`; the cell of before the part is restored,
  D225).
- *The frame evaluator* (D240). A frame begins an activation, whose
  structures are built for its positions and its cell: frames far from
  each other and in other cells are evaluated as with the matrix. The
  energy of a frame is that of the potential `@tunable`, whose loops
  keep the matrix (above), so it is the number of the program with the
  matrix; the virial is of the loops of the forces, over the groups.
- *Tunables.* An update and `gradient()` are those of the matrix; the
  loops of the derivative keep the matrix (above).
- *Checkpoints.* The fingerprint has `[execution] neighbor_structure` when
  the field was set and `[energy] pruned_distance` when there is a dual
  list, so `mdir run --continue` takes a checkpoint of a Python simulation
  with the same keys and a Python simulation one of `mdir run` without a
  note; a continuation with another structure or reach is the same run
  with other execution, with a note that names the keys (D223).
- *The deterministic mode* refuses the groups (D89): their atomic
  additions have no fixed order. Nothing with the groups is equal to the
  bit, in either front end.

**The order of the particles.** A Python simulation puts its particles
in the order of their positions where an activation begins
(`Execution.reorder`, [python-segments.md](python-segments.md)) and not
again; `mdir run` does so at its start and at each of its checkpoints
(#125). The measurement of #270 finds no rate in the order: `mdir run`
with `spatial_order = false`, which keeps the order of the input
throughout, is as fast as with it, within 0.004 ms a step, with the
matrix, the groups, and the dual list on the three systems below, because
every build sorts the particles into its own order of places (D86), which
is what the loops over pairs read. Sorting again in the course of a
Python simulation, outside the deterministic mode, is therefore not
added; #125 stays a question of the deterministic mode.

**Validation** (`python-groups*.test`, `Inputs/python_groups.py`; the
dipeptide in 382 waters, 1,168 particles, PME on $32^3$, SHAKE and SETTLE,
velocity Verlet at 1 fs; GPU; the largest difference seen and the
tolerance of the test, relative to the larger magnitude of each number
unless a unit is given):

| Check | Reference | Double | Tolerance | Mixed | Tolerance |
|---|---|---|---|---|---|
| Energies at the start: the groups, the groups with `neighbor_capacity = 1`, the dual list (11 Å, pruned at 9.3 Å) | a Python simulation with the matrix | 0 | 1e-10 | 1.3e-7 | 2e-5 |
| Forces at the start, kJ/mol/nm (largest force 1737) | the same | 1.1e-12 | 1e-7 | 5.8e-4 | 0.5 |
| Energies at 5 rows over 100 steps | the same | 4.5e-12 | 1e-7 | 3.9e-5 | 5e-3 |
| Rows of the energy file over 100 steps, Python with the groups and with the dual list | `mdir run` of the same control | 0 (equal to the 6 decimals of the file) | 1e-7 | 1.1e-4 | 5e-3 |
| The same, `mdir run` against a second run of itself | | 0 | 1e-7 | 7.7e-5 | 5e-3 |
| Positions after 100 steps, nm | `mdir run`, its checkpoint | 2.2e-13 | 1e-8 | 6.2e-7 | 1e-3 |
| Two simulations of one `Program`, the second with the code of the first (`program_reused`), energies after 20 steps | each other | 1.9e-13 | 1e-7 | 2.3e-6 | 5e-3 |
| A commit of the cell and the positions scaled by 1.01 through a borrow, energies; forces in kJ/mol/nm | a simulation compiled at that state | 0; 4.6e-13 | 1e-10; 1e-7 | 7.3e-9; 2.4e-4 | 2e-5; 0.5 |
| A commit of a cell of 0.75 of the first (least edge 1.905 nm, reach 2 nm) | | `InputError`, the state unchanged; the matrix takes it | | the same | |
| After an update of the tunables (22 charges times 1.05, $\epsilon$ of the pair OW-OW times 0.9): energies; `gradient()`, relative to its largest entry | a Python simulation with the matrix | 0; 2.8e-16 | 1e-10; 1e-9 | 2.9e-7; 2.2e-7 | 2e-5; 1e-3 |
| The frame evaluator over 5 frames (the start, the state 200 steps later, the start, the later state in a cell 2% wider, the start): energy; virial; `vjp` | the evaluator of the program with the matrix | 0; 3.5e-16; 2.9e-16 | 1e-10; 1e-8; 1e-9 | 6.2e-9; 1.4e-7; 1.2e-7 | 2e-5; 1e-3; 1e-3 |
| The same frame again, after frames far from it | its first evaluation | 0 | 1e-10 | 0 | 2e-5 |
| NPT with the dual list, a checkpoint at step 20 of 40: Python to `mdir run --continue`; `mdir run` to Python, without a note; Python with the dual list to Python with one list and with the matrix, with a note that names the keys. Positions at step 40, nm | the Python run that did not stop | 3.1e-14 | 1e-8 | 8.8e-8 | 1e-3 |
| The potential of every frame of 216 argon atoms under a barostat that shrinks the cell, reach 1.7 nm (0.77 to 0.89 of the edge), cube and tilted, one list and dual (`python-groups-reach-gpu.test`) | the sum in NumPy over every image of every pair within the cutoff | 0 of 79 frames differ by more than 1e-5 kcal/mol | | | |
| The same with a reach of 2.0 nm; of 2.3 nm in the tilted cell | | the part fails with the runtime's message; `compile` refuses with the builder's | | | |

In mixed precision the forces are f32 and two runs part as two runs of
`mdir run` do: a Python simulation follows `mdir run` as closely as
`mdir run` follows itself. Not to the bit: see above.

**Rates** (#270; one RTX 3090, GPU 0 alone, mixed precision, the settings
of the Amber suite's script that the Python model can state, the second
half of one run of 200,000, 60,000, and 20,000 steps; ms per step):

| System | Ensemble | `mdir run`, matrix | Python, matrix | `mdir run`, groups | Python, groups | `mdir run`, dual list | Python, dual list |
|---|---|---|---|---|---|---|---|
| Dipeptide (1,168) | NVE | 0.075 | 0.074 | 0.112 | 0.111 | 0.100 | 0.100 |
| | NPT | 0.087 | 0.087 | 0.112 | 0.114 | 0.106 | 0.107 |
| JAC (23,558) | NVE | 0.258 | 0.259 | 0.218 | 0.218 | 0.205 | 0.206 |
| | NPT | 0.276 | 0.276 | 0.234 | 0.234 | 0.223 | 0.224 |
| Factor IX (90,906) | NVE | 0.853 | 0.855 | 0.650 | 0.650 | 0.598 | 0.600 |
| | NPT | 0.890 | 0.888 | 0.682 | 0.681 | 0.631 | 0.631 |

The matrix and the groups have a pairlist distance of 10 Å; the dual list
11 Å with the inner list at 8.6 Å, as the suite's script sets it. With one
structure the two front ends run at one rate, within 1%; what a Python
simulation lacked was the structure. Parts of 100 steps cost nothing that
the timing resolves, and a step of energy every 100 steps 0.002 to 0.005
ms a step in either front end. The default stays the matrix: on the
dipeptide the groups are the slower structure, and the choice by the size
of the system (G3 of the roadmap) is not made here.
## The influence function and the order of PME, and the analytic bonds

D[python-pme-fields] (#279, from the inventory of #270). Two settings of
the Amber suite's script had no field in the Python model, so a Python run
could not compute the suite's model exactly, and `System.pme_order` took 4
alone.

| Python | Control file | Values |
|---|---|---|
| `System.pme_influence` | `[pme] influence` | `mdir.PMEInfluence.SPME` (the default), `mdir.PMEInfluence.Optimal` (D134) |
| `System.analytic_bonds` | `[constraints] analytic_bonds` | `False` (the default), `True`: a group of SHAKE of one bond is projected in closed form |
| `System.pme_order` | `[pme] order` | 4 (the default), 6, or 8; before, 4 alone |

Another order is an `InputError` with the words of the control file,
`model: expected 4, 6, or 8 for 'order'`; a value of `pme_influence` that
is not of the enumeration is a `TypeError`. The fingerprint of a checkpoint
has `[pme] influence` and `[constraints] analytic_bonds` when the field
was set, as it has `[pme] order`. The model hands the builder the control
structure of the control file: no program or kernel is new.

**Validation** (`python-pme-fields.test`, `-gpu.test`,
`Inputs/python_pme_fields.py`): the dipeptide in 382 waters, PME on
$32^3$, SHAKE and SETTLE, velocity Verlet at 2 fs, NVE from drawn
velocities, in the deterministic mode, on the CPU and a GPU, in double and
in mixed precision; five cases: the optimal influence function, the
analytic bonds, the orders 6 and 8, and the suite's two settings together.

| Check | Reference | Result | Tolerance |
|---|---|---|---|
| The potential at step 0 | the row of the energy file of `mdir run` with the keys (6 decimals of kcal/mol) | within 4.7e-7 kcal/mol | 1e-6 |
| Positions, velocities, and forces after 20 steps | the checkpoint of `mdir run` | equal to the bit, each case | 0 |
| The fingerprint of the checkpoint | that of `mdir run` | equal | equal |
| The potential at step 0 or the positions after 20 steps | the run without the setting | differ, each case: the field acts | more than 0 |
## The defaults are those of the control file

D[python-defaults] (#281; the maintainer's decision on the inventory of
#270). Where the Python model and the control file had different defaults
for one setting, the Python model takes the control file's: a system given
to both front ends with nothing beyond its inputs is one model. Before, it
was two.

| Setting | Before | Now, as the control file without the key | To get the old model |
|---|---|---|---|
| `System.truncation` | `Truncation.Switch` | `Truncation.None_`, a plain cutoff | `system.truncation = mdir.Truncation.Switch` |
| `System.switch_distance` | 1.0 nm | the cutoff (no switch); it follows `System.cutoff` until it is set | `system.switch_distance = 1.0` |
| The correction for the dispersion, in effect | off, with the warning `dispersion_switched` (the default switch turned it off) | on (`EnergyPressure`), without a warning | follows from the switch; or `system.dispersion = mdir.DispersionCorrection.None_` |
| `System.pairlist_distance` | 1.35 nm | 0.15 nm beyond the cutoff; it follows `System.cutoff` until it is set (equal at the default cutoff) | `system.pairlist_distance = 1.35` |
| `Ensemble.com_period` | 0: the motion of the center of mass is never removed | `None`: with a thermostat every `coupling_period`, without one never | `ensemble.com_period = 0` |
| A state without velocities | begins at rest | takes the velocities that `mdir run` draws: at `Ensemble.temperature`, with `Ensemble.seed` | `state.velocities = numpy.zeros((N, 3))` |

- *The truncation.* The control file without `lennard_jones_modifier` and
  `switch_distance` has a plain cutoff. The old default, the generic
  switch from 1.0 nm, was a model that `mdir run` cannot state with a
  topology (it refuses the generic switch there). `Truncation.Switch`
  stays available when set, a model of Python alone.
- *`switch_distance` and `pairlist_distance`.* Not set, they follow
  `System.cutoff` as absent keys do; reading gives the value in effect and
  `None` restores the default. The pairlist distance is computed as the
  control file computes it, in Å, so that both front ends hold the same
  double. With `ForceSwitch` and the other switches `switch_distance` must
  be set below the cutoff, as in the control file.
- *The correction for the dispersion* was on by default and turned off by
  the default switch ([above](#the-correction-for-the-dispersion)); with a
  plain cutoff it is on, as under `mdir run`, and the default model
  compiles without a warning.
- *`com_period`.* `None` hands the control the absent key, which it
  resolves: under a thermostat the motion is removed when the thermostat
  acts, without one never. `0` says never; a number under a thermostat
  must equal `coupling_period`, as `center_of_mass_interval` must equal
  `interval`.
- *Velocities.* `mdir run` draws velocities when its coordinates give none
  (not for a minimization, which begins at rest): from the
  Maxwell-Boltzmann distribution at `[ensemble] temperature` with
  `[dynamics] seed`, which is the seed of the coupling too. `mdir.compile`
  does the same for an `InitialState` without velocities, with
  `Ensemble.temperature` and `Ensemble.seed`: the numbers of `mdir run`,
  and of `state.draw_velocities(system, ensemble.temperature,
  ensemble.seed)`. Velocities that are given are kept, zeros among them:
  `state.velocities = numpy.zeros((N, 3))` begins at rest.
  `InitialState.from_state(state, velocities=False)` gives a state
  without velocities, which is then drawn.

The fingerprint of a checkpoint writes an entry for a setting that was
given, as the control file writes a key
([python-checkpoints.md](python-checkpoints.md)). The fingerprint took the
default cutoff, 1.2 nm, for different from the 12 Å of an absent key (two
doubles compared in nm) and recorded `[energy] cutoff` for every default
model; it compares in Å now.

**Validation** (`python-defaults*.test`, `Inputs/python_defaults.py`): the
dipeptide in 382 waters given to both front ends with nothing beyond the
files, the ensemble, the periodic boundary, and the deterministic mode: no
cutoff, no reach, no truncation, no velocities, no periods.

| Check | Reference | Result | Tolerance |
|---|---|---|---|
| Potential, kinetic, and total energy at step 0, NVE, NVT, and NPT | the row of the energy file of `mdir run` (6 decimals of kcal/mol) | within 4.7e-7 kcal/mol | 1e-6 |
| Positions, velocities, forces, and the cell after 20 steps, the three ensembles, CPU and GPU, double and mixed | the checkpoint of `mdir run` | equal to the bit | 0 |
| The fingerprint of the checkpoint | that of `mdir run` | equal | equal |
| The fields: the defaults, that the two distances follow the cutoff until set, `None` | | as stated above | |
| The model of before, stated with the last column of the table | | compiles with the one warning `dispersion_switched`, begins at rest, and its potential differs from the default model's by 61.6 kJ/mol | |

## Python host array boundary

D193 (#78) exposes the native vectors as independent read-only
NumPy arrays: positions/velocities $(N, 3)$ float64 (absent velocities
$(0, 3)$), reduced cell vectors $(3, 3)$ float64, tuple particles
$(n, \mathrm{arity})$ int64 and parameters as 1-D float64 arrays. Native
C++ storage remains unchanged. Strict buffer/CPU DLPack assignment validates
shape, dtype, contiguity, finite values and particle count, copies values and
advances the binding version. There is no list path. NumPy >=1.23 is required
only for the enabled Python interface. See [python-arrays.md](python-arrays.md).

Setters whose value has a unit also take OpenMM unit quantities, converted
at the boundary (D200, [python-units.md](python-units.md)).

`System.coulomb_modifier` (D205, #127) is the
control file's `[energy] coulomb_modifier`: `CoulombModifier.None_` (the
default, as in the control file) or `CoulombModifier.PotentialShift`, the
real-space Coulomb term of PME shifted to zero at the cutoff
(`Control::pmeShift`). As in the control file it is for PME only; a model
with cutoff electrostatics and the shift raises `InputError` at
`compile`. `System.truncation` remains the Lennard-Jones modifier.
