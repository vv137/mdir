### Changed

- **The `md_alg` ops survive as IR between passes** (D408, #866, part 1). `convert-md-exec-to-gpu` leaves the collection ops of the neighbor matrix's template and of the groups build in the module; an empty pass, `md-alg-transform-collections`, holds the place of the collection-level transformations to come (fusion, the elision of a sort no consumer needs, the removal of intermediate buffers); `convert-md-alg-to-gpu` lowers the ops after it. The accessors of the templates of PME and the counting sort of the device's ragged rows are still lowered where they are created. Each op with an ordering or a floating-point aspect takes a determinism requirement, `unordered`, `reproducible`, or `bitwise`, separate from its ordering contract; the verifier refuses one that the contract rules out, the SIMT lowering one that its code does not meet, and the defaults keep every program's meaning. The PTX of the 60 programs of D406 is unchanged.

### Added

- **The policy of the `md_alg` ops in the log and the manifest** (D408). A run on a device prints `MDIR: md_alg policy:`, each distinct requirement, order, strategy, and launch once with the count of its ops, and the manifest holds one entry for each op in `effective.md_alg_policy` (`docs/run-manifest.schema.json`), so that a bitwise comparison of two runs compares like with like.
