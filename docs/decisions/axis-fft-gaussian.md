# D[axis-fft-gaussian]

Recorded at its merge.

**Use the five-pass Gaussian PME mesh transform (D435) across the
measured mesh range on mixed-precision `sm_89` devices.**

AUTO uses the axis transform for order-four Gaussian PME in
non-deterministic mixed precision on `sm_89`, including orthorhombic
and triclinic cells. Each axis must be 18–224 points long and the mesh
must contain 7,200–2,612,736 points, inclusive. The transform also
requires even Z and supported factorizations of X, Y, and Z/2. The
range covers the D435 folding meshes, starting at 18 × 20 × 20,
and extends through the paired benchmark mesh of 224 × 108 × 108.
The rule applies across neighbor traversals.

For the D445 prolate split, AUTO selects the triclinic axis plan on
the four D435 folding meshes (18 × 20 × 20, 22 × 24 × 24, 28 × 28 × 30,
and 32³). The existing target, precision, traversal, and split checks
apply to those plans.

The setup log and the manifest's `effective.pme.fft_plans` give the
mesh, selected backend, and reason for each evaluation. Gaussian AUTO
reports the measured range when selecting axis or falling back to cuFFT
for a supported mesh outside it. Unsupported transforms and fixed-order
evaluation use cuFFT. The existing PME keys select the split, grid, and
assignment order. `MDIR_PME_AXIS_FFT` can explicitly select cuFFT or
request a supported axis transform beyond the AUTO range and in double
precision.

Validation compares reciprocal forces with cuFFT in mixed and double
precision and checks fixed-order repeatability. Whole-step and mesh
timings use paired 2 fs runs in opposite backend orders on RTX 4090.
