# D[axis-fft-gaussian]

Recorded at its merge.

Extend the five-pass Gaussian PME mesh transform (D435) to the grids
selected for JAC, Factor IX, TIP3P 10k, Cellulose, and Trp-cage. Keep the
Gaussian split, grid dimensions, and assignment order selected by the
existing PME keys.

Automatic selection qualifies the backend on the measured target, mesh,
and cell class. The setup log and the manifest's
`effective.pme.fft_plans` give the mesh, the selected backend, and the
reason for each evaluation. An unsupported or unqualified plan uses
cuFFT. The existing `MDIR_PME_AXIS_FFT` override selects the backend
for paired measurements, and fixed-order evaluation retains cuFFT.

Validation compares reciprocal forces with cuFFT in mixed and double
precision and checks fixed-order repeatability. Whole-step and mesh
timings use paired 2 fs runs in opposite backend orders on RTX 4090.
