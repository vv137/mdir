### Fixed

- **A GPU run on a driver older than the CUDA of the build stops with both versions** instead of crashing inside the driver when it loads the kernels, and `mdir doctor --target=gpu` reports it; the driver must support the major version of CUDA of the build, 13.0 for a release (D177, #657).
