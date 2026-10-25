# D[stage-rows-attr]

Recorded at its merge.

**A batched stage body retains a verified static row declaration through bufferization.** #642, part of #607. The `md.stage_rows` i64 attribute stays on the batched function. A named MD helper writes and reads it; the MD dialect rejects wrong type or placement and a value that differs from a statically known leading dimension of the first input. The stage verifier uses the same declaration to compare forward and vector-Jacobian product bodies. Valid stage IR, emitted code, numerical results, control-file keys, formats, defaults, overwrite behavior, and device strategies remain unchanged. `md.stage_body`, `md.stage_disjoint_results`, and the other semantic attribute families remain separate #607 slices. Validation is recorded in PR #643.
