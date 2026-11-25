# D[pme-fused-convolution]

Recorded at its merge.

For mixed-precision Gaussian Coulomb PME in an orthorhombic cell, apply
the influence function in the inverse cuFFT load on eligible force-only
evaluations. Energy, virial, and charge derivatives keep the separate
multiply. The callback uses the same grid, per-axis tables, and zero
mode. Forces can differ by rounding. The setup log and run manifest
record the rule. It introduces no control key or environment variable.
