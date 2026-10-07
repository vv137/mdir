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
stochastic cell rescaling NPT are included, on CPU/GPU, mixed/double.
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
PME uses spline order four and automatic or explicit grids (at least eight
points on each axis). The object model's defaults are stated in the header;
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

## The correction for the dispersion

D[python-dispersion] (#161) gives the Python model the two parts of the
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
truncation is `Switch`, still compiles (the maintainer's choice on PR
#187). Before, it kept the correction under a switch.

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
