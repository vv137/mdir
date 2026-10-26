### Changed

- A `dyn.program` states the effects of its body in `effects`, which its
  verifier checks and `dyn.step` takes, and the runtime's functions that a
  program calls declare theirs in `dyn.effects`, so that a step of a run
  with constraints writes the stops of the run rather than every resource
  (D[program-effects], #673). No program changes.
