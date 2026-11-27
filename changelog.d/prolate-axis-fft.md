### Added

- Prolate PME supports the five-pass device FFT with either assignment
  window in mixed and double precision. Observed steps accumulate
  reciprocal energy and virial in the fused X pass. The FFT plan gives
  the selected backend and reason (D445).
- AUTO takes the axis plan only for mixed TOWERS on `sm_89` with
  `continuity = 2`, `basis_terms = 8` and the B-spline window, on 30³,
  48³ and 105 × 60 × 60; `MDIR_PME_AXIS_FFT=on` selects it elsewhere,
  including the prolate window and tilted cells (D445).
