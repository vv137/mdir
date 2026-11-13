### Changed

- The device primitives that the kernels' templates and the groups build called by name are now typed ops of a new dialect, `md_alg`, each with a verifier and a serial definition that a test compares with its GPU lowering; the programs are unchanged (D[md-alg], #861).
