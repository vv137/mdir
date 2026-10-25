### Added

- **`MDIR_BUILD_TESTS`** (CMake, on by default) builds the tests and
  `mdir-opt`; the wheel turns it off, so the wheel builds from its sdist,
  which leaves `test/` out ([docs/building.md](docs/building.md), #624).
