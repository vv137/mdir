### Fixed

- Reading `sim.tunables` from another thread during an update now gives the
  values, version, and history of before, whole, and
  `sim.tunables.snapshot()` returns all three at once. A buffer exported to
  a consumer stream keeps the simulation from running even if its view is
  released during the hand-off (D[python-access-races], #621).
