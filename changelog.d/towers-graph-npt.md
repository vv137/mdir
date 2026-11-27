### Added

- `step_graph = "GRAPH"` replays ordinary TOWERS steps under stochastic cell rescaling (`[barostat] method = "C-RESCALE"`, every coupling and `work`, rectangular or triclinic cells); `AUTO` runs its trial there too. The kernels of a chunk read the launch arguments that the cell sets from a device mirror, which the host fills before each chunk. `MDRT_STEP_MIRROR_FAULT=1` keeps the first cell, for the tests (D[towers-graph-npt], #1017).
