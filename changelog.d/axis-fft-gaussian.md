### Changed

- Select the CUDA axis FFT automatically across the measured Gaussian PME
  mesh range in mixed precision on `sm_89` and report the backend and
  reason at setup (D[axis-fft-gaussian], #1029).
- Qualify the D435 triclinic folding meshes for the D445 prolate AUTO rule.
