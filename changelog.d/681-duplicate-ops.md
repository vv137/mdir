### Added

- **`scripts/builder-equivalence/duplicate_ops.py` counts the costly ops that fused pair and tuple kernels still compute twice** after fusion and `cse` (the same op, `sqrt` beside `rsqrt`, related `exp`s, shared divisors, powers of one base, radial tables at one index), and the loops left unfused with their reasons, over the builder-equivalence corpus on both targets and in both precisions (#681).
