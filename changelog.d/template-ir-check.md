### Added

- **The equivalence check compares the instances of the templates of the lowering** (D[template-ir-check], #739): `scripts/builder-equivalence/check.py` takes `--precisions`, `--env`, and `--instances`, and the corpus kind `cases` holds cases for the neighbor matrix and PME in single precision, the tiles of edge 4 and 16, and the potential of PME at the particles (`docs/driver.md`, Section 2.11).
