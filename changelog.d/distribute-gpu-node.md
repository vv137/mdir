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
- Joint submission of intra-node and inter-node halo messages, sparse
  updates to retired host rebuild snapshots, and a single collective for
  eligible fused particle observables.
- Constrained multi-device halos enclose periodic groups separately, and
  independent floating-point tuple observables share reductions.
- Keep pointwise updates of owned constraint vectors on the rank's GPU.
