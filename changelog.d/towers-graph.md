### Added

- Added `[execution] step_graph` and Python `Execution.step_graph`.
  Constrained fixed-cell TOWERS runs can replay ordinary steps from a CUDA
  graph or time both submission paths at setup. The default is `"LAUNCH"`.
  AUTO skips timing from 408,609 particles and records the deciding count.
  D[towers-graph].
