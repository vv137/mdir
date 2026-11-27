### Changed

- Distributed CPU runs evaluate topology terms and disjoint constraint
  groups in parallel (D[distribute-cpu-threads]).
- MPI particle migration and bonded tuple scheduling use rank-local records.

MPI integration carries compact identity intervals between spatial force
evaluations. Neighbor-list exclusion caches use the halo's local endpoints.
