# D[prolate-axis-fft]

Recorded at its merge.

Apply the prolate reciprocal multiplier between the forward and inverse X
transforms of D435. The axis product uses the same radial spectrum and
assignment-window tables as the separate cuFFT product. The force-only
evaluation multiplies each complex charge coefficient once. An observed
evaluation also uses its original squared magnitude and the derivative of
the radial spectrum for energy and the six independent virial components.

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
