### Added

- Optional NCCL peer transport and node-aware placement for distributed
  finite-range GPU MATRIX evaluation (D[distribute-gpu-node]).
- Device-resident Verlet updates, pair-only position halos, cached scalar
  halo fields, and device row reordering on migration.
- Measured selection of compact interior and boundary traversals with the
  distributed `pair_overlap` control.
