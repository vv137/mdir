### Added

- **A release builds, checks, and installs its Python wheels**:
  `scripts/release/publish.sh` builds the wheels with the tarball, checks
  them, installs each by pip into a fresh environment of its CPython and
  runs it on the CPU, and one on the GPU
  (`scripts/release/check-wheel-install.sh`), and attaches them to the
  release (#641).
