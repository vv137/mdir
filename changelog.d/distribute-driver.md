### Added

- D[distribute-driver] adds the CPU `[distribute]` table for fixed-cell velocity Verlet with finite-range pair potentials, migration, and collection at output boundaries.
- Distributed logs and manifests record decomposition choices and imports. Energy intervals report traffic, migrations and phase times. Optional per-step traces and restart tests cover changed grids and particle-indexed seeded draws.
- In deterministic CPU runs, checked fixed-point sums reproduce distributed forces and observables across grid changes and checkpoint continuations.
