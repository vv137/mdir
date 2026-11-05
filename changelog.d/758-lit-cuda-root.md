### Fixed

- **The test suite takes the CUDA toolkit from `CUDA_ROOT`, `CUDA_HOME`, or `CUDA_PATH` when one is set**, as the runtime does, and from the toolkit of the build otherwise (#758). A tree configured with the toolkit at one path now runs its GPU tests on a machine where the toolkit is at another path, without an edit of the generated `lit.site.cfg.py` ([docs/environment.md](docs/environment.md)).
