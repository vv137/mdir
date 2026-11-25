### Changed

- Mixed-precision Gaussian PME on the GPU applies the influence function
  during the inverse FFT load on eligible force-only evaluations. This
  removes a pass over the complex grid. The setup log and run manifest
  report the schedule (D[pme-fused-convolution], #959).
