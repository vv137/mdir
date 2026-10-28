### Fixed

- **A wheel built from source without auditwheel imports without `LD_LIBRARY_PATH`**: its extension finds HDF5 where the build found it, through its RPATH; a repaired release wheel carries no such path, which `check-wheel.sh` now checks (#634).
