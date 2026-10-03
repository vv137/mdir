# Temporal Cartesian CPU execution

D[cpu-temporal]. Experimental CPU implementation, extending #36; tracks #38.

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

## Implemented execution and state transition

The implementation is in `tools/mdir-cpu-lj/main.cpp`. The generated force
kernel and its physical cutoff are unchanged. The host performs velocity
Verlet in double precision; `--precision=mixed` selects f32 pair arithmetic
with f64 accumulation, positions, velocities, masses, and integration. The
runtime compiles the kernel once and rebinds dynamic memref descriptors after
any vector resizing. No rank-local count is baked into the compiled kernel.

1. Rank zero reads coordinates and, for dynamics, the complete state table.
   Initial scatter assigns each particle to its Cartesian position owner.
2. Build the map and evaluate the initial complete force at step zero.
3. Apply the first half-kick using the previous complete force, then drift and
   wrap coordinates into the fixed orthorhombic box.
4. All participants reduce two flags: any changed position owner, and any
   displacement since the map epoch exceeding half the skin. Empty ranks
   participate. A changed owner always requires migration and rebuild.
5. If necessary, redistribute state and build a new map; otherwise retain
   local indices, map, interior classification, and candidate rows.
6. Execute the existing verified reference plan for the new coordinate value,
   finish the halo, and combine interior and boundary force/virial/energy.
7. Apply the second half-kick with the new force. Report step diagnostics.

The old force is dead before migration: it was consumed in the first half-kick.
ID, position, mass, and half-step velocity are the complete live state of this
experimental driver. Production integrators have additional live fields and
cannot reuse this packet without extending its contract.

`--steps=0` retains snapshot execution. `--repeat` replays that snapshot;
its map and rows can now be reused. `--state` is read only for positive steps.
Paths are read on rank zero. Precision, halo mode, grid, emit mode, step count,
dt, skin, repeat count, thread count, and SIMD width must agree on every rank;
an exact serialized comparison precedes option-dependent collectives.

## Coverage, freshness, and epoch lifetime

Map construction expands each destination box by cutoff plus skin, with the
existing f32 roundoff margin. It sends each global ID at most once to that
rank, including periodic dimensions of size one or two. Sparse periodic cell
bins use the same conservative support. Candidate rows include neighboring
bins and therefore may include pairs outside that support; the force kernel
always filters with the original physical cutoff. Skin never changes the
potential.

The displacement guard sums actual, unwrapped drift increments since rebuild,
then measures their Euclidean norm. It does not compare wrapped coordinates:
an atom traveling through a whole periodic box cannot hide that displacement.
If each particle moved at most half the skin, pair separation changed by at
most the skin. Rebuild occurs before using a row whose support may have expired.
The guard runs on every step, including when the default skin is zero.

Interior membership is frozen within an epoch. An interior center starts more
than cutoff plus skin plus margin from every split face; unsplit dimensions
impose no face restriction. With unchanged ownership and a valid displacement
guard, it cannot acquire a remote neighbor inside the physical cutoff. Every
other owned center belongs to the boundary stage. The two stages partition
centers; their force arrays can be added because each stage writes zero to the
other stage's centers. They are not independent physical force terms.

The host maintains these separate facts:

| State | Meaning and transition |
|---|---|
| `layoutEpoch` | Incremented whenever the map is rebuilt; this conservative implementation also changes it for a support-only rebuild. |
| cached row epoch | Must equal `layoutEpoch` to reuse local-index rows. Interior and boundary have separate caches. |
| `fieldVersion` | Incremented after each drift; does not change merely because a halo is copied. |
| `ghostVersion` | Set to `fieldVersion` only after receive completion and unpack. |
| `pending` / `ready` | Enforce one start/wait pair per evaluation and prevent a rebuild or drift during communication. |

Each evaluation repacks current coordinates using the retained send-index map,
poisons ghost slots, and refreshes their payload. Boundary dispatch requires
both `ready` and matching versions even if it reuses candidate rows. No cached
coordinate payload is treated as fresh merely because its map is still valid.

These are runtime checks in the reference executor. This change does **not**
implement a general MLIR `FieldStateAnalysis`, semantic contribution ledger,
or proof that an arbitrary user kernel implements LJ. `md_dist.reference_plan`
still verifies the restricted start/interior/wait/boundary schedule and binds
layout/map handles for one invocation. The host repeats that plan on successive
runtime snapshots. A request wait proves transport completion; force readiness
additionally requires both generated stages and their local combination.

## Migration and asynchronous memory lifetime

Migration uses direct all-to-all counts followed by two `MPI_Alltoallv` calls:
one `MPI_LONG_LONG` ID per atom and seven `MPI_DOUBLE` values in the order
`x, y, z, mass, vx, vy, vz`. Explicit packing avoids dependence on C++ struct
padding. Both exchanges complete before new arrays replace old ones. Received
owned particles are sorted by global ID. Ghosts are reconstructed afterward;
all old candidate caches fail the new epoch check. Final output gathers the
current IDs and counts, not the initial decomposition order.

The number of particles is capped at `INT_MAX / 7`; send and receive halo
capacities retain their separate MPI count checks. Capacity overflow, invalid
input, and nonfinite arithmetic terminate through `MPI_Abort`. There is no
recoverable team-wide allocation retry or fault-tolerance protocol yet, and
allocation failure is not intercepted as a coordinated retry.

In async mode, independent packed send and receive arrays remain unchanged
until `MPI_Waitall` finishes all requests. Only then are receives copied into
ghost slots. No migration, resize, or repack overlaps an outstanding request.
Interior force computation reads owned coordinates and writes separate force
storage. This is a specific lifetime discipline, not a general alias-aware
async verifier. MPI progress during the interior kernel remains implementation
dependent; nonblocking calls alone are not a performance guarantee.

## Validation and reproducibility

`scripts/validation/cpu-temporal-lj.py` uses an independent Python velocity
Verlet integrator and the unordered-pair, explicit-periodic-image oracle from
`cpu-hybrid-lj.py`. It compares final IDs, masses, positions, velocities,
forces, virial, and potential energy. Cases cover faces, edges, corners,
periodic crossings, newly entering cutoff pairs, stable/rebuilt epochs,
multi-box travel, and empty systems on 1, 2, and 8 ranks, sync/async and
mixed/double. Tolerances are `3e-10 * (1 + abs(reference))` in double and
`5e-5 * (1 + abs(reference))` in mixed. These are numerical tolerances, not a
bitwise guarantee across decompositions.

`scripts/validation/cpu-temporal-invalid.py` checks missing/duplicate/unknown
IDs, invalid mass/velocity, extra data, invalid options, and inconsistent
participant options with bounded subprocess timeouts. Each must fail with a
driver diagnostic rather than hang.

All 108 trajectory configurations passed. Maximum absolute discrepancies over
energy, virial, forces, positions, velocities, and masses were `8.88e-16` in
double and `5.33e-7` in mixed. The reuse fixture's final potential reference was
`-0.89491878326570395`; its largest compared-component discrepancy was
`4.44e-16` in double and `5.33e-7` in mixed. The map was built once in that case,
versus 24 rebuilds with the small-skin fixture. The cutoff-entry reference was
`-0.023026797688381801`; mixed maximum component error was `2.29e-9`.

The 12 rejection checks passed. The complete CPU lit run passed 191 tests with
69 unsupported (GPU and other opt-in tests), including the new trajectory and
invalid-input smoke test. The first clean configure exposed a missing MPI
launcher substitution (issue #40); explicitly setting
`MPIEXEC_EXECUTABLE=/usr/bin/mpiexec` resolved that configuration failure.
It was not a numerical or intermittent test failure.

No distributed GPU transport or performance claim is made.

The short NVE drift checks avoid cutoff crossings because this experiment uses
an unshifted truncated LJ potential. Cutoff-entry tests check agreement with
the specified discontinuous potential, not energy conservation. This validation
is not a production equilibrium protocol, long-time stability study, or restart
validation.

Under the GPU 1 lock, the existing `pair-terms-gpu` and `lj-pme-gpu` regression
tests both passed (103 seconds). These exercise production CPU/GPU comparison
paths, including mixed/double PME checks; they do not execute the new MPI
trajectory on a GPU. The entire GPU suite was not rerun for this CPU-only
extension.
