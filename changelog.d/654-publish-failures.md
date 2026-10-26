### Changed

- **The configure step refuses a C++ standard library without the
  floating-point `std::from_chars`** (`std::chars_format`), such as that
  of GCC 9, with a message that names the compiler and the requirement
  (GCC 11 or later), instead of failing deep in the build;
  [docs/building.md](docs/building.md) states it (#654).

### Fixed

- **`scripts/release/publish.sh` builds in a fresh tree and reports a
  failed step**: it removes its build tree before it configures, so that
  `CC` and `CXX` choose the compiler, and a step that fails names itself,
  its log, and the log's first errors instead of ending in silence (#654).
