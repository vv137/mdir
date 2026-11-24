# D[pme-axis-fft]

Recorded at its merge.

For orthorhombic Gaussian PME on a device, generate a five-pass convolution
from MLIR (issue #961, D301). The emitter takes a static mesh, its physical
half-spectrum pitch, device buffers, and a pointwise operation on a complex
coefficient and its three frequency indices. Forces use the existing spread,
axis tables, and gather. Double precision, charge derivatives, and unsupported
meshes use cuFFT. Qualification is per path.

For dimensions $X,Y,Z$ with even $Z$, factor each of $X,Y,Z/2$ into two
register transforms of at most 32 points each, built from radices 2, 3, 5,
and 7. Each block owns complete lines. A shared-memory exchange with an
inter-factor twiddle takes input digit order $b+Ba$ to frequency $p+Aq$
for length $AB$. Strided passes tile neighboring last-axis slots. The first
pass packs adjacent real values and transforms along Z. It writes the
$H=Z/2+1$ Hermitian slots at complex offsets $(xY+y)P+z$, where $P$ is the
physical pitch. Y forward operates in place. X forward, the pointwise
operation, and X inverse share one launch and keep the X spectrum in
registers. Y inverse and Z complex-to-real complete the convolution.
Both transform directions are unnormalized.

The X operation combines the prepared Gaussian axis factors and sets the
DC multiplier to zero. Observed steps consume the coefficient before the
multiply, weighting the two Hermitian end planes once and interior slots
twice. Reduce the energy and six symmetric virial entries within the block
and finish the partial sums in f64. Force-only kernels omit those sums.

Selection follows the rule category of #952. Record the grid, factor pairs,
line tile sizes, physical pitch, phase evaluation, and influence source in
the generated IR, the setup log, and the manifest. Choose balanced supported
factor pairs and limit the tile
by shared memory and thread count. Use f64-computed constants for register
butterflies, f32 trigonometry for inter-factor phases, and prepared axis
tables for the influence. Pad rows to a multiple of four complex values for
aligned tiles. These rules require no setup timing. Qualification compares
the complete step in both orders against cuFFT on the same device.

The initial automatic rule on `sm_89` covers order-four mixed TOWERS
Gaussian meshes of 40³, 64³, and 144 × 84 × 80. The RTX 4090 runs
qualify those shapes against an independent double reference and paired
whole-step measurements. Other geometries default to cuFFT. A compilation
override can pin cuFFT or request an experimental supported plan.
