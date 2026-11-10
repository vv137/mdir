### Added

- On the CPU, a row of a loop over pairs can run several candidates at a time in vector registers, the strategy `host(lanes(W))` chosen by `md-exec-choose-host-pairs` (D[host-pair-vector], #554). `MDIR_HOST_PAIRS=scalar|lanes|lanes:W` overrides the host's policy, and `MDIR_HOST_CPU` compiles the host code for another CPU.

### Changed

- On the CPU the deterministic mode sums the rows of the loops over pairs in 8 lanes in f32 and 4 in f64 on every host, so its bits differ from those of before and are the same on every host; `MDIR_HOST_PAIRS=scalar` gives the scalar rows of before (D[host-pair-vector]).
