### Added

- **A workflow publishes the Python package on PyPI**, TestPyPI first,
  which the maintainer starts for a release tag: it checks the release's
  wheels against SHA256SUMS, builds the sdist from the tag, checks their
  sizes, and uploads by trusted publishing; `check-wheel-install.sh
  --index` tests the package as pip takes it from the index (#650).
