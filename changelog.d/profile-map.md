### Added

- **A profile map: the source of each parallel region of the host, as JSON** (D[profile-map], #840). `mdir emit --profile-map=FILE` and `profile_map` in `[output]` write, for each outlined region (`mdir_run..omp_par.N`), the op it comes from, its set, its potentials and terms, and its precision; `scripts/benchmarks/cpu-breakdown/prof.py` takes the parts of a profile from it instead of guessing from the code ([profile-map.md](docs/profile-map.md)).
