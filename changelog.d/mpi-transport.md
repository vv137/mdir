### Added

- Run CPU domain trajectories across MPI ranks with the optional MPI build
  and `[distribute] transport = "mpi"` (D[mpi-transport]).
- Rank zero reports merged interval and final statistics. Per-rank traces record collective wait. CPU MPI checkpoints support continuation with a different rank count and domain grid.
