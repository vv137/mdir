# D[pme-axis-fft]

Recorded at its merge.

For Gaussian PME on a device, generate a five-pass convolution
from MLIR (issue #961, D301). The emitter takes a static mesh, its physical
half-spectrum pitch, device buffers, and a pointwise operation on a complex
coefficient and its three frequency indices. Forces use the existing spread,
axis tables, and gather. Double precision, charge derivatives, and unsupported
meshes use cuFFT. Qualification is per path.

For dimensions $X,Y,Z$ with even $Z$, factor each of $X,Y,Z/2$ into two
register transforms of at most 32 points each, built from radices 2, 3, 5,
7, and 11. Each block owns complete lines. A shared-memory exchange with an
inter-factor twiddle takes input digit order $b+Ba$ to frequency $p+Aq$
for length $AB$. Strided passes tile neighboring last-axis slots. The first
pass packs adjacent real values and transforms along Z. It writes the
$H=Z/2+1$ Hermitian slots at complex offsets $(xY+y)P+z$, where $P$ is the
physical pitch. Y forward operates in place. X forward, the pointwise
operation, and X inverse share one launch and keep the X spectrum in
registers. Y inverse and Z complex-to-real complete the convolution.
Both transform directions are unnormalized.

The PME lowering supplies the X operation through `FFTPointwise`. It can
read arbitrary captured tables at the three mode indices. For Gaussian PME
it calls the same influence builder as the cuFFT path. In a tilted cell
it forms the Cartesian wave vector from the signed indices and inverse cell
before evaluating the Gaussian. Both geometries set the DC multiplier to zero. Observed steps consume the coefficient before the
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
orthorhombic Gaussian meshes of 40³, 64³, and 144 × 84 × 80. The RTX 4090 runs
qualify those shapes against an independent double reference and paired
whole-step measurements. Other geometries default to cuFFT. A compilation
override can pin cuFFT or request an experimental supported plan.

### Trp-cage and the prolate split

For continuity 2 and eight even Legendre terms, #951 prepares an order-four
B-spline window and the radial spectrum of the constrained density. After
#951 merges, the prolate lowering needs to allocate the pitched spectrum and
pass its own pointwise operation to `emitAxisConvolution` for MATRIX. Reuse
`DevicePME::radialInfluence` with the prepared prolate table buffer. Its
first three axis blocks provide the signed modes and the inverse squared
window transforms. For the cubic window, each axis correction is $[\sin(x)/x]^{-8}$
with $x=\pi n/N$ and unit value at zero. Their product $T$ includes the Coulomb/volume prefactor
on X. The radial data follow those blocks, with metadata and two arrays
of transform values and scaled slopes. The finite-basis evaluator uses
cubic Hermite interpolation at low arguments and endpoint coefficients
from the tails of the two arrays at high arguments.

Let $s=|\mathbf m|^2$ for the physical reciprocal vector and
$k=2\pi r_c\sqrt{s}$. The radial evaluator gives $C=\hat\chi(k)$ and
$L=k\hat\chi'(k)$. For $s>0$, scale the coefficient by $D=TC/s$.
For observed steps let $u$ be its squared magnitude before scaling, times
one half on the stored end planes or one elsewhere. Accumulate $uD$ for
energy and $u[D\delta_{ij}-T(2C-L)m_i m_j/s^2]$ for the six virial
entries. Set both weights to zero at DC. Only the window factors separate
by axis. Both radial weights need the complete wave vector, including the
inverse-cell transform in a tilted box.

The two folding meshes have register factor pairs for $(X,Y,Z/2)$ of
$(3\times6,4\times5,2\times5)$ and $(2\times11,4\times6,3\times4)$.
Radix 11 was added for the latter. Before selecting prolate for folding,
check that force-only and observed MATRIX plans actually select the
emitter on both tilted inputs. Compare reciprocal forces, energy and all
six virials with prolate cuFFT and an independent double mesh, check the
virial against cell-strain energy differences, and time complete steps in
both backend orders. Gaussian runs on these grid sizes validate the FFT
and layout. The radial evaluator and its strain derivative require their
own qualification.

### Gaussian validation on the folding inputs

Both inputs use their original tilted cells, order-four Gaussian PME, and MATRIX traversal. CHARMM36m has 6,545 atoms and ff19SB+OPC has 8,602. The two smaller meshes use the Gaussian multiplier at the grid sizes planned for prolate. Each error triple gives relative force L2, absolute relative reciprocal energy, and relative L2 of the six independent virials. The CPU double reference spreads the input charges, transforms the mesh, and gathers the force independently. Each RTX 4090 timing covers the last 10,000 of 20,000 steps at 2 fs, with energy every 1,000 steps and a 14.5 Å pair list. A runs cuFFT first on each grid. B reverses both grid and backend order.

| System | Mesh | Axis vs mixed cuFFT | Axis vs independent double | A cuFFT / axis (µs) | B cuFFT / axis (µs) |
| --- | --- | --- | --- | ---: | ---: |
| CHARMM36m | 28 × 28 × 30 | 1.62e-06 / 1.91e-07 / 2.09e-07 | 1.18e-05 / 1.62e-06 / 8.92e-07 | 116 / 111 | 117 / 110 |
| CHARMM36m | 18 × 20 × 20 | 1.72e-06 / 4.02e-07 / 5.44e-07 | 1.14e-05 / 9.71e-07 / 7.37e-07 | 118 / 112 | 117 / 114 |
| ff19SB+OPC | 32 × 32 × 32 | 2.07e-06 / 2.73e-07 / 1.79e-07 | 1.27e-05 / 1.18e-06 / 3.28e-06 | 144 / 137 | 145 / 138 |
| ff19SB+OPC | 22 × 24 × 24 | 1.39e-06 / 2.89e-07 / 1.76e-07 | 1.28e-05 / 1.09e-06 / 2.31e-06 | 143 / 137 | 144 / 138 |
