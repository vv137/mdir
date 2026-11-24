### Changed

- Cubic Gaussian PME on the GPU uses a frame-certified integer spread and writes gathered Cartesian forces directly in atom order. At setup it times available cuFFT load callbacks and explicit sweeps for force-only steps.
