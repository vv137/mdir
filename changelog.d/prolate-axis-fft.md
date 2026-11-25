### Added

- Prolate PME supports the five-pass device FFT with either assignment
  window in mixed and double precision. Observed steps accumulate
  reciprocal energy and virial in the fused X pass. The FFT plan gives
  the selected backend and reason (D[prolate-axis-fft]).
