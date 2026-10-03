# Temporal Cartesian CPU execution

D[cpu-temporal]. Implementation in progress, extending #36; tracks #38.

The experimental `mdir-cpu-lj` executable gains `--steps=N` (default 0),
`--dt=T` (required positive finite value for dynamics), `--state=FILE`
(required for dynamics), and `--skin=S` (default 0, finite nonnegative).
The original coordinate snapshot stays unchanged. State rows are
`id mass vx vy vz`, exactly one per snapshot atom, in any order; masses
must be positive and all values finite. Units are the existing reduced LJ units.
No production TOML keys or output files are introduced.

Positive steps select fixed-cell velocity Verlet. Stdout retains final energy,
virial, and forces, followed by `state id mass x y z vx vy vz` rows for dynamics.
Stderr reports per-step potential, kinetic, total energy, layout epoch, and
rebuild/migration counters. `--repeat` remains fixed-snapshot replay and may
not be combined with dynamics except at its default value 1. Emit modes are
snapshot-only. This is not the production Amber driver.

Each force evaluation refreshes coordinate payloads. Neighbor/map reuse needs
both unchanged ownership and displacement within half the skin. A team-wide
OR controls rebuild/migration. Exchanges finish before layouts or storage
change. Migration transfers every live field of this driver (ID, position,
velocity, mass); old force is consumed by the half-kick before migration and
then recomputed. Async schedules retain the existing restricted reference IR.

Validation will compare moving states and forces with independent Python
velocity Verlet, including internal/periodic faces, edges, corners, empty ranks,
reused/rebuilt maps, mixed/double, and sync/async. Drift tests avoid crossing the
unshifted LJ cutoff discontinuity. Bonded, PME, constraints, thermostats,
barostats, GPU transport, and production-driver integration remain unsupported.
