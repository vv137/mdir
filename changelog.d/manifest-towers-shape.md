### Fixed

- **The run manifest schema accepts TOWERS runs** (#972). `docs/run-manifest.schema.json` now lists `towers_shape` and `towers_cell_height` under `effective` and `towers` among the neighbor structures, which `mdir run` already writes. `test/Driver/towers-one-step-gpu.test` validates a TOWERS manifest against the schema.
