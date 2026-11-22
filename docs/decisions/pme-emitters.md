# D[pme-emitters]

Recorded 2026-11-22.

Build the particle mesh Ewald stages with typed MLIR builders. The window
provides support, weights, slopes, and the transform used in deconvolution.
The splitting supplies the separable or radial influence. The cell maps a
particle to grid coordinates. Select these policies when emitting a stage
so that adding a window or a splitting does not duplicate the FFT,
spreading, convolution, and gathering schedules (#908).

Keep the host row and column transform ABI, the device launch and reduction
order, the precision of each intermediate, and the existing control-file
keys. Compare the complete lowered module and its embedded PTX against the
same base with D406's settings on sm_89 and sm_120. Include the listed,
tiled, deterministic, prolate, and triclinic paths in mixed and double
precision. Record and time any difference before retiring the templates.
