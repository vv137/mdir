### Fixed

- Parse staged tensor/linalg model functions with the same dialect set as lowering, so canonical printing resolves segmented loop-carried values instead of silently returning raw assembly. D352's raw-text fallback contract remains unchanged (#714).
