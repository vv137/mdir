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
without type mixing or custom dispersion correction. Tuple expressions have
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
