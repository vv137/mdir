# D[prolate-axis-fft]

Recorded at its merge.

Apply the prolate reciprocal multiplier between the forward and inverse X
transforms of D435. The axis product uses the same radial spectrum and
assignment-window tables as the separate cuFFT product. The force-only
evaluation multiplies each complex charge coefficient once. An observed
evaluation also uses its original squared magnitude and the derivative of
the radial spectrum for energy and the six independent virial components.
The window deconvolution depends only on the integer indices and has no
strain derivative (Section 5.6 of the white paper gives the tensor).

Both the prolate assignment and the four-point B-spline assignment can use
the axis product. The latter divides by the squared spline transform on
each axis. The transform follows the mesh precision, including binary64
cross phases for a double mesh. A tilted cell uses the inverse cell to
construct each wave vector and the real inverse projects the end planes
as in D435.

The existing `MDIR_PME_AXIS_FFT` override selects the backend for
validation. AUTO uses the measured mesh rule at setup. Each plan records
the backend and reason. An unsupported transform length, including 188
with prime factor 47, keeps the separate cuFFT product.

AUTO qualifies one target and one split: mixed TOWERS on `sm_89`, with
`continuity = 2`, `basis_terms = 8` and `window = "BSPLINE"`. On that
target and split it takes the axis plan on the orthorhombic meshes 30³,
48³ and 105 × 60 × 60, where the axis plan was faster in two whole-step
timings in opposite orders. It keeps cuFFT on 189 × 90 × 90, where the
axis plan was slower in both. Every other mesh keeps cuFFT with the reason
`no qualified prolate axis timing for this mesh`. Any other target or
split keeps cuFFT with the reason `no qualified timing for this prolate
target or split`, and a tilted cell keeps it with the reason `unqualified
triclinic mesh`. The prolate window was not timed. It is left out of AUTO
rather than timed now, and the override still runs it. Adding the
automatic grids of the B-spline window (D[prolate-bspline-grid]) to the
list is #1003.

The tests compare the axis plan with a binary64 mesh oracle on a cube
(48³) and on an orthorhombic cell whose sides and mesh lengths differ on
every axis (48 × 40 × 36). They also compare it with cuFFT on those cells
and on a tilted cell, in mixed and double precision. For an even mesh
length the stored half mesh gives the Nyquist index the value $+K_a/2$.
The oracle uses the same convention and so does not check it. The
convention matters only for the sign of the off-diagonal virial terms of
modes with one Nyquist index in an orthorhombic cell. Energy, forces and
the diagonal do not depend on it.
