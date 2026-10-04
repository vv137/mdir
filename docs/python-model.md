# Shared model for M2 (D[python-model])

Implementation in progress; the first item of `python-m2.md`.
This PR provides the native C++ model layer for subsequent Python bindings.
It does not provide an importable Python package.

## Contract

Owned `LoadedData` captures Amber or GROMACS topology, coordinates, optional
velocities, cell, and source contents. Loading neither draws velocities nor
initializes CUDA. Physics (`model::System`) and `InitialState` are separate
values, copied from loaded data. Public quantities use nm, ps, kJ/mol, bar,
amu, K, e, and radians. Cells use the driver's upper-triangular convention.
No mutable arrays are shared between loaded data, physics, or prepared runs.

Typed `Integrator`, `Ensemble`, and `Execution` select a fixed CLI-compatible
schedule for parity testing. Persistent arbitrary segments follow in the
third implementation PR. Device selection follows in the compilation PR;
this layer never changes the process's CUDA device.

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
