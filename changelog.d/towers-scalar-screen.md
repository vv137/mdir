### Added

- TOWERS runs on Volta with a scalar distance screen. The optional `[execution]` field `towers_shape.screen` selects `AUTO`, `MMA`, or `SCALAR`; AUTO uses the matrix instruction at compute capability 7.5 and above and the scalar screen below, and preflight refuses an explicit MMA request on an older device ([D[towers-scalar-screen]](docs/decisions/towers-scalar-screen.md)).

### Fixed

- TOWERS certification votes compile on Turing: below sm_80 the subgroup sum uses five XOR shuffles instead of `redux.sync` (#978).
