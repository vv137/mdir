### Added

- Optional NCCL peer transport and node-aware placement for distributed
  finite-range GPU MATRIX evaluation (D[distribute-gpu-node]).
- Device-resident Verlet updates, pair-only position halos, cached scalar
  halo fields, and device row reordering on migration.
- Measured selection of compact interior and boundary traversals with the
  distributed `pair_overlap` control.
- Incremental face records, rank-local halo route maps, and device-to-device
  exchange of migrating field rows on eligible GPU runs.

- Measured halo skins for directed NCCL runs with an automatic pair-list reach.
