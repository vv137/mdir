### Added

- **The build of the lists of units on a device** (D[md-alg-segments-build], slice S2 of D417, #894). The groups build of D404 builds a list of units over the copies of rigid kinds and the rest, at their reference points. Each segment $s(f, g)$ is searched with its covering reach $R + e_f + e_g$, into records of its own.
  - `convert-md-exec-to-gpu` lowers the list's empty list and its refresh: the unit tables, the test of the reference points, the stop of the reach against the cell, and the pruning of a dual list.
  - `md-exec-reuse-neighbors` moves its build out of the loop over steps.
  - The coverage query of D357 checks each segment that a loop visits.
  - The serial definition runs over the device's lists with `traverse-oracle=serial`. The SIMT schedule and the route are later slices.
  - On the RTX 5090, replaying trajectories of #795's six systems, the list's build and pruning take 3.1–5.0 µs a step, against 5.3–11.8 µs for #795's list maintenance. They rebuild as often as #795's.
