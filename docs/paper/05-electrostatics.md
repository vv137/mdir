# 5. Electrostatics

MDIR computes the electrostatic energy of a periodic system by particle
mesh Ewald (PME) in the smooth form of [[Essmann1995]](references.md#essmann1995), after
[[Darden1993]](references.md#darden1993). This section derives the terms as they are implemented,
the forces and the virial of the reciprocal sum, and how the sum runs on a
device; Sections 5.4 and 5.5 treat the dispersion beyond the cutoff, by a
correction and by the same mesh; Section 5.6 a second kernel of the splitting and the spreading. Internally the energies are in kJ/mol and the Coulomb constant is
$f = 138.935457644$ kJ nm/(mol e²), that of CODATA 2018
[[Tiesinga2021]](references.md#tiesinga2021), whatever the format of the
input: Amber's constant is smaller by a factor 1.0000346 and CHARMM's
larger by 1.0000238, so the electrostatic terms of the same charges differ
from those programs' by these factors (Section 9.1).

## 5.1 The splitting

The Coulomb energy of the charges of the cell and their periodic images,
less the pairs that the force field excludes (bonded neighbors, 1–2, 1–3,
and 1–4, whose 1–4 interaction is a scaled term of its own), is split with
$1/r = \mathrm{erfc}(\beta r)/r + \mathrm{erf}(\beta r)/r$
[[Ewald1921]](references.md#ewald1921):

$$
E = E_\text{dir} + E_\text{rec} + E_\text{self} + E_\text{excl} + E_Q .
$$

*Table 5.1. The terms of the splitting.*

| Term | As implemented | Where it runs |
|---|---|---|
| $E_\text{dir}$ | $f\sum_{i<j,\ r_{ij}<r_c}' q_iq_j\big[\mathrm{erfc}(\beta r_{ij})/r_{ij} - s\big]$, $s = \mathrm{erfc}(\beta r_c)/r_c$ under `coulomb_modifier = "POTENTIAL_SHIFT"`, otherwise 0; the prime omits excluded pairs | The loop over pairs, with Lennard–Jones |
| $E_\text{excl}$ | $-f\sum_{(i,j)\ \text{excluded}} q_iq_j\mathrm{erf}(\beta r_{ij})/r_{ij}$, over every excluded pair, without a cutoff | A loop over the tuples of the excluded pairs |
| $E_\text{self}$ | $-f\frac{\beta}{\sqrt\pi}\sum_i q_i^2$ | A constant of the host |
| $E_Q$ | $-f\frac{\pi Q^2}{2V\beta^2}$, $Q = \sum_i q_i$ [[Hub2014]](references.md#hub2014) | A constant of the host, scaled with $1/V$ |
| $E_\text{rec}$ | Section 5.2 | `md.reciprocal` |

The force of the direct sum follows from Section 3.3 with

$$
-\frac{u'(r)}{r} = f q_iq_j\Big[\frac{\mathrm{erfc}(\beta r)}{r^3}
+ \frac{2\beta}{\sqrt\pi}\frac{e^{-\beta^2r^2}}{r^2}\Big],
$$

and that of an excluded pair with

$$
-\frac{u'(r)}{r} = f q_iq_j\Big[\frac{2\beta}{\sqrt\pi}\frac{e^{-\beta^2r^2}}{r^2}
- \frac{\mathrm{erf}(\beta r)}{r^3}\Big] .
$$

The term $E_Q$ removes the
energy of the uniform background that neutralizes a net charge; its
virial is $3E_Q$, which makes its pressure $E_Q/V$, and its value at the
volume of the step is $E_Q(V_0)\,V_0/V$.

**Parameters** (D71). $\beta$ is given, or found by bisection such that
$\mathrm{erfc}(\beta r_c)$ equals a tolerance, $10^{-5}$ by default:
100 halvings of $[0, h]$, $h$ doubled from 1 nm⁻¹ until
$\mathrm{erfc}(h r_c)$ is below the tolerance. The grid has $K_a$
points along axis $a$, the smallest even number of at least
$L_a/h_\text{max}$ whose prime factors are 2, 3, 5, and 7, with
$h_\text{max} = 1.2$ Å; or it is given. The order $n$ of the B-splines is
4, 6, or 8 (4 by default), and the grid needs at least $2n$ points along
each axis. A charge of 100 e or more is rejected, since the deterministic
mode adds the grid in fixed point at the scale $2^{40}$.

## 5.2 The reciprocal sum

With the structure factor $S(\mathbf m) = \sum_j q_j e^{2\pi
i\,\mathbf m\cdot\mathbf x_j}$ over the reciprocal vectors $\mathbf m$ of
the cell, which for an orthorhombic cell are $m_a = k_a/L_a$ with integer
$k_a$ and for a triclinic one, whose vectors are the rows of the
lower-triangular $H$, $\mathbf m = H^{-1}\mathbf k$ (D123, D125; on a device the
Gaussian is computed at each point, as it is not a product of a factor for
each axis),

$$
E_\text{rec} = \frac{f}{2\pi V}\sum_{\mathbf m\neq 0}
\frac{e^{-\pi^2 m^2/\beta^2}}{m^2}\,\lvert S(\mathbf m)\rvert^2 .
$$

**Interpolation.** With the scaled fractional coordinate
$u_{aj} = K_a x_{aj}/L_a$, the smooth PME interpolates each exponential
by cardinal B-splines $M_n$ of order $n$:

$$
e^{2\pi i k_a u_a/K_a} \approx b_a(k_a)\sum_{l\in\mathbb Z} M_n(u_a - l)\,e^{2\pi i k_a l/K_a},
\qquad
b_a(k) = \frac{e^{2\pi i (n-1)k/K_a}}{\sum_{j=0}^{n-2} M_n(j+1)\,e^{2\pi i kj/K_a}} .
$$

The charges are spread on the grid,

$$
Q(\mathbf l) = \sum_j q_j \prod_{a} M_n(u_{aj} - l_a - p_aK_a)
$$

summed over the periodic shifts $p_a$, so that $S(\mathbf m)\approx
b_1(k_1)b_2(k_2)b_3(k_3)\,\hat Q(\mathbf k)$ with $\hat Q$ the discrete
Fourier transform of $Q$. Then

$$
E_\text{rec} \approx \tfrac12\sum_{\mathbf k\neq 0} B(\mathbf k)\,C(\mathbf k)\,\lvert\hat Q(\mathbf k)\rvert^2,
\qquad
B = \prod_a\lvert b_a(k_a)\rvert^2,\quad
C = \frac{f}{\pi V}\,\frac{e^{-\pi^2 m^2/\beta^2}}{m^2},\quad C(0) = 0 .
$$

The energy is a quadratic form in the spread grid: with
$\hat Q = \mathcal F Q$ and the real diagonal influence map $D$ of
$B C$, it is $E_\text{rec} = \tfrac12\langle\mathcal F Q,
D\mathcal F Q\rangle_\mathrm H$ when the last axis is stored as a
half-spectrum. The pairing weights each interior last-axis frequency
twice, its zero frequency once, and its Nyquist frequency once when
$K_3$ is even. The variation is
$\delta E_\text{rec} = \langle D\mathcal F Q,
\mathcal F\delta Q\rangle_\mathrm H
= \langle\mathcal F^\dagger D\mathcal F Q,
\delta Q\rangle$, so the inverse transform is the transpose needed to
return the grid potential to the particles. Both directions are
unnormalized, as the quadratic form above requires.
The adjoint identity is on the half-spectra of real grids: the zero plane
and, for even $K_3$, the Nyquist plane obey Hermitian symmetry across the
first two axes. The inverse of an arbitrary packed complex array is outside
this contract.
This derivation underlies the declared `md_exec.linear_transform`
adjoint (D306, #480). Its current execution-level form does
not yet put this chain into the semantic `md` potential; that is #452.

The driver computes the moduli $\lvert b_a\rvert^2$ in f64 from the
recursion of $M_n$ and passes them as a table. Under `influence =
"OPTIMAL"` it multiplies them by $\lambda^2$ with $\lambda = S_n(x)/S_{2n}(x)$,
$S_p(x) = \sum_{j=-50}^{50}\big(x/(x+\pi j)\big)^p$, and $x = \pi k/K$, a
factor of the aliasing of the B-splines that brings the energy of the grid
closer to the Ewald sum [[Ballenegger2012]](references.md#ballenegger2012); the default is the influence
function of [[Essmann1995]](references.md#essmann1995). With $\hat U(t) =
(\sin\pi t/\pi t)^n$, the transform of the B-spline over a spacing, Poisson
summation gives $\lvert b_a(k)\rvert^2 = 1/(\hat U^2 S_n^2)$ at every order,
so that $\lvert b_a\rvert^2\lambda^2 = \hat U^2(k/K)/\big(\sum_j\hat U^2(k/K + j)\big)^2$:
`"OPTIMAL"` is eq. 4.2 of [[Ballenegger2012]](references.md#ballenegger2012),
to the truncation of its sums at $\lvert j\rvert \le 50$ ($3.2\times10^{-7}$ at order
4, $4\times10^{-15}$ at order 8).

**The influence function optimal for the forces** (D304).
Analytic differentiation gives the exact gradient of the energy of the grid
for any influence function $G$, so $G$ may be chosen for the accuracy of
the forces alone. For two unit charges, the mesh force is a sum over the
points $\mathbf k$ of the grid and the aliases $\boldsymbol\nu = \mathbf k +
\mathbf J\odot\mathbf K$, $\mathbf J\in\mathbb Z^3$, of the wave vectors
$\tilde{\mathbf m}_\mathbf J$. Its mean-square deviation from the force of
the Ewald sum, averaged over the place of the pair relative to the grid and
integrated over their separation, is quadratic in each $G(\mathbf k)$; its
minimum is eq. 2.28 of [[Ballenegger2012]](references.md#ballenegger2012),

$$
G(\mathbf k) = \frac{f}{\pi V}\,
\frac{\sum_\mathbf J\hat U^2(\boldsymbol\nu)\,e^{-\pi^2\tilde m_\mathbf J^2/\beta^2}}
{\big(\sum_\mathbf J\hat U^2(\boldsymbol\nu)\big)\big(\sum_\mathbf J\hat U^2(\boldsymbol\nu)\,\tilde m_\mathbf J^2\big)},
$$

with $\hat U^2(\boldsymbol\nu) = \prod_a\hat U^2(\nu_a/K_a)$, which `influence =
"P3M"` takes with $\max_a\lvert J_a\rvert \le 2$. The average is that of
uncorrelated charges: it leaves out the force of a charge on itself and
the correlations of a liquid. In an orthorhombic cell the three sums
factor over the axes, $G = (f/\pi V)\prod_aF_a/\sum_a\langle\tilde m_a^2\rangle$
with $F_a = \sum_ju_{a,j}e^{-\pi^2\tilde m_{a,j}^2/\beta^2}/S_a$ and
$\langle\tilde m_a^2\rangle = \sum_ju_{a,j}\tilde m_{a,j}^2$, the weights
$u_{a,j} = \hat U^2(\nu_{a,j}/K_a)/S_a$ normalized by $S_a = \sum_j\hat
U^2(\nu_{a,j}/K_a)$: the arithmetic of the separable factors below, with
$\langle\tilde m_a^2\rangle$ in the place of $m_a^2$. The virial follows by
the same argument as below, each alias moving under the strain:
$W_{ab} = \sum E_\mathbf k\big(\delta_{ab} - (2\pi^2/\beta^2)A_{ab} -
2B_{ab}/\mathrm{tr}\,B\big)$, $A$ and $B$ the means of $\tilde m_a\tilde m_b$
under the weights $u\,e^{-\pi^2\tilde m^2/\beta^2}$ and $u$. On JAC and
Cellulose at $\beta$ = 0.39467 Å⁻¹ it lowers the relative force error
$\Delta$ (Section 5.6) of `"OPTIMAL"` by 0.05% at about 1 Å and by up to 2%
on grids of 2 Å, and that of `"SPME"` by 2 to 3% at about 1 Å and by 2 to
6% at order 6, but raises it by 3 to 4% on grids of order 4 of 1.5 to 2 Å
(`docs/pme.md`, Section 1.1). A triclinic cell is refused (#475). Both
front ends select the influence function,
and the order of the splines: the control file with `influence` and
`order` of `[pme]`, the Python model with `System.pme_influence` and
`System.pme_order` (D246).

**Real transforms.** $Q$ is real, so $\hat Q(-\mathbf k) =
\overline{\hat Q(\mathbf k)}$ and a real-to-complex transform stores
$k_3 \in [0, K_3/2]$. Each stored point is weighted by $h = \tfrac12$ at
$k_3 = 0$ and, when $K_3$ is even, at $k_3 = K_3/2$, and by $h = 1$
elsewhere, so that $E_\text{rec} = \sum_\text{stored} h\,BC\lvert\hat Q\rvert^2$.

**Forces.** Write $E_\text{rec} = \tfrac12\sum_\mathbf l Q(\mathbf l)\,
(\theta\star Q)(\mathbf l)$, where $\theta$ is the inverse transform of
$BC$ and $\star$ the circular convolution. Since $\theta$ is symmetric,

$$
F_{ja} = -\frac{\partial E_\text{rec}}{\partial x_{ja}}
= -q_j\,\frac{K_a}{L_a}\sum_{\mathbf l}\phi(\mathbf l)\,
M_n'(u_{aj} - l_a)\prod_{b\neq a}M_n(u_{bj} - l_b),
$$

with $\phi = \theta\star Q$, the unnormalized backward transform of
$BC\hat Q$, and $M_n'(u) = M_{n-1}(u) - M_{n-1}(u-1)$.

**Virial.** Under a homogeneous strain $\varepsilon$ of the cell a
reciprocal vector changes as $m_a\to m_a - \varepsilon_{ab}m_b$, so
$\partial m^2/\partial\varepsilon_{ab} = -2m_am_b$, and
$\partial V/\partial\varepsilon_{ab} = V\delta_{ab}$. The term of
$\mathbf m$, $E_\mathbf m = \tfrac12 BC\lvert\hat Q\rvert^2$, depends on the
strain through $1/V$ and through $e^{-\pi^2m^2/\beta^2}/m^2$, while
$\hat Q$ does not, the fractional coordinates being fixed. With
$\mathsf W = -\partial E/\partial\varepsilon$,

$$
W_{ab} = \sum_{\mathbf m\neq 0} E_\mathbf m\Big[\delta_{ab} - 2\Big(\frac{1}{m^2} + \frac{\pi^2}{\beta^2}\Big)m_am_b\Big],
\qquad
\mathrm{tr}\,\mathsf W = \sum_{\mathbf m\neq0}E_\mathbf m\Big(1 - \frac{2\pi^2m^2}{\beta^2}\Big).
$$

The isotropic barostat and the log use the trace, the semi-isotropic
barostat the diagonal (Section 6.4).

**Separable factors** (D104). For an orthorhombic cell the Gaussian
factorizes, $e^{-\pi^2m^2/\beta^2} = \prod_a e^{-\pi^2m_a^2/\beta^2}$, and
so does $B$. A kernel therefore computes, at each evaluation, for each
point of each edge, $m_a$, $m_a^2$, and $t_a = e^{-\pi^2m_a^2/\beta^2}
\lvert b_a\rvert^2$ (the first edge also times $f/(\pi V)$), in f64,
each rounded once to the type of the grid; the product with the
transform then needs $BC = t_1t_2t_3/m^2$ and no exponential per point.
The tables follow the cell under a barostat at the cost of one small
kernel, 3 µs a step on Cellulose.

## 5.3 On a device

*Table 5.2. The kernels of the reciprocal sum on a device, in order. The
type of the grid is f32 in the mixed and single modes and f64 in double
precision (D78).*

| Step | Kernel | Method |
|---|---|---|
| Places | `weights` | A thread a particle writes the first point and the fraction $w$ along each axis, six values (D109). The fractional coordinate $x\cdot(1/L) - \lfloor x\cdot(1/L)\rfloor$ is computed in f64, its product with $K$ and the fraction within a point in the type of the grid; the first point is $(\lfloor u\rfloor - n + 1)\bmod K$ |
| Spreading | `spread_bricks` | Order 4 in f32: a warp a particle; lane $16c+4b+a$ adds the points $(a,b,c)$ and $(a,b,c+2)$ of its $4^3$ points into *bricks* of $4\times4$ points in $x$ and $y$ with $z$ inside, by `red.global.add.f32`. A lane computes its weight from $w$ in closed form: $\tfrac{(1-w)^3}{6}$, $\tfrac{3w^3-6w^2+4}{6}$, $\tfrac{-3w^3+3w^2+3w+1}{6}$, $\tfrac{w^3}{6}$ for the points 0 to 3, what the recursion gives to $3\times10^{-16}$ |
| | Copy | The bricks are copied into the grid of the transform through block memory, a tile with rows of 17 values to avoid bank conflicts, and set to zero after the barrier, so they need no clearing kernel (D108) |
| | Tiles | `bin`, `spread_tiles`: the particles binned by their first point into bins of $g^3$ points, a block a bin adds into a tile of $(g + n - 1)^3$ points in block memory and the tile into the grid, by atomics; the default outside the deterministic mode for a grid in f32 of at least 500,000 points, B-splines or the prolate window (Section 5.6) (D282, below) [[Shih2021]](references.md#shih2021) |
| | Otherwise | Orders 6 and 8, or f64, without tiles: a thread for each particle and point along $z$, with floating-point atomics; in the deterministic mode the products $q\prod M$ are scaled by $2^{40}$, rounded to 64-bit integers, added with integer atomics, and scaled back, so the grid does not depend on the order of the threads [[LeGrand2013]](references.md#legrand2013) (D70, D84) |
| Transform | cuFFT | Real-to-complex and back, in the type of the grid; plans are cached and bound to the stream of the kernels |
| Tables | `tables` | The separable factors of Section 5.2 |
| Product | `scale` or `convolve` | $\hat Q\leftarrow BC\,\hat Q$. When the energy or the virial is requested, `convolve` also sums $E_\mathbf m$ and the six components of $\mathsf W$: a block of 128 threads reduces in the type of the grid, stores its sums in f64, and one block adds the blocks in f64, in an order that depends only on the size of the grid; the host copies seven numbers |
| Gathering | `gather_weights` | Order 4 in f32: four threads a particle; each recomputes the splines and their slopes from the stored fractions by the unrolled recursion, in the type of the grid, and the four sum by shuffles (D102). Otherwise four lanes a particle for order 4 and eight for 6 and 8, which recompute the places |

The fraction along the cell stays f64 because a position of a large cell
in f32 would lose the digits that place it within a point: the product
with $K$ in f32 resolves a point to $K\,2^{-24}$, $1.6\times10^{-5}$ of a
point for $K = 270$ (D78, amended).

On the host every step is a parallel loop under OpenMP, in the type of
the forces as on a device (f32 in the mixed mode), and its result does not
depend on the number of threads (D271). The charges are
added in fixed point by integer atomics from a loop over the particles,
whose sum does not depend on their order. The FFT is pocketfft
[[Reinecke2019]](references.md#reinecke2019) (real-to-complex along $z$
and complex along $y$ and $x$, with cached plans), called for blocks of
lines from a loop over the lines of each pass; each line is transformed by
the same code whatever thread takes it, in f64. The product is a loop over
the rows of the grid, the Gaussian influence of a rectangular cell from the
factors of its edges as on a device, its energy and virial added over the
rows in the fixed order of the CPU's reductions (D171), and only on the
steps that write them. The gathering is a loop over the particles.

**Cost.** On Cellulose (408,609 atoms, a grid of 270 × 126 × 126) the
places take 27 µs a step, the additions 214, the copy and the clearing 63
(304 µs in all), the two transforms 292, the product with the virial 30,
and the gathering 90 (RTX 3090; medians of a launch, from D102, D104,
D108, and D109).

**Tiles** (D282). The sum $Q(\mathbf l) = \sum_j q_j\prod_a
M_n(u_{aj} - l_a - p_aK_a)$ may be taken in any partition of the
particles. Bin them by their first point $s_{aj} = (\lfloor u_{aj}\rfloor
- n + 1)\bmod K_a$, the bin $\lfloor s_{aj}/g\rfloor$ along each axis; since
$M_n$ vanishes outside $(0, n)$, a particle of the bin with first point
$g\,\mathbf b$ adds only to the points $g\,\mathbf b + \mathbf t$, $0 \le
t_a < g + n - 1$, wrapped once around the grid at most where $K_a \ge g +
n - 1$. A block accumulates these $(g + n - 1)^3$ partial sums in its own
memory, and the grid is the sum of the tiles, each point of it in at most
$2^3$ of them. The terms and their products are those of the bricks; only
the order of the floating-point additions changes, so the grid agrees
with them to the rounding of the sums, as two runs of the bricks do. This
is the binned, blocked spreading of cuFINUFFT
[[Shih2021]](references.md#shih2021). The particles are binned at each
evaluation into runs of a fixed length per bin, twice the mean
occupancy and 8 more, with an overflow list that one more block adds
directly, so that no scan or sort of the bins is needed; the lanes of a
warp in the same bin share one atomic for their ranks. A block adds to its
tile by atomics of block memory, which on sm_89 are a compare-and-swap
loop: more warps on a tile wait on each other, so a bin of $8^3$ takes 128
threads. The device atomics fall from $n^3$ a particle to about $(1 + (n
- 1)/g)^3$ a grid point, 24 to at most 11 million a step on Cellulose; the grid
must be zeroed first, which the bricks do not need. The deterministic
mode keeps its fixed-point kernel. On the Amber suite in mixed precision
(RTX 4090) the spreading of order 4 takes 95 µs a step on Cellulose
against 165 with the bricks, and 279 against 539 on STMV, the step 0.95
and 0.94 times as long; at orders 6 and 8 the step takes 0.79 to 0.85
times as long on Factor IX, Cellulose, and STMV. Below 500,000 points the
bins are too few to occupy a device (JAC, 512 bins: 1.00 to 1.01 times the
step at order 4 on an RTX 4090, 1.05 on an RTX PRO 6000), and in f64 the
tiles are slower (1.01 to 1.02); neither is tiled by default (the tables
are in `docs/pme.md`, Section 3.3; the conditions of the tiles are
derived in Section 8.5). The gathering is not tiled: in the
prototype of #334 tiles did not pay there at order 4 (#411).

**Particles without a coefficient.** A particle with $q_j = 0$ adds
nothing to $Q(\mathbf l)$, and its force,

$$
\mathbf F_j = -\frac{\partial E_\text{rec}}{\partial\mathbf x_j}
= -q_j\sum_{\mathbf l}\phi(\mathbf l)\,\nabla_j\prod_a M_n(u_{aj} - l_a),
$$

with $\phi$ the grid after the product with $BC$ and the backward
transform, is zero; the energy and the virial are functions of $\hat Q$
alone. The sums of the mesh therefore run over $\mathcal J_q$ and
$\mathcal J_c$ (D262): the oxygen of a
four-site water (OPC, TIP4P-Ew) has no charge, and the hydrogens and the
virtual site of a water no coefficient of the dispersion. Under
`[free_energy]` the charges are $q_j(1 - a_j\lambda_\text{C})$ (Section
6.8), zero at every $\lambda_\text{C}$ where $q_j = 0$, and the share of
particle $j$ in $\partial U/\partial\lambda_\text{C} = \sum_j
(\partial U/\partial q_j)(-a_jq_j)$ is zero too; the sets are taken from
the charges of the topology, not from the scaled ones, so that a particle
that $\lambda_\text{C} = 1$ decouples stays in them. The restriction is
exact: in the deterministic mode the grid is the same sum of integers and
every force the same product, and the run is the same to the bit. The
potential at the particles, $\partial E_\text{rec}/\partial q_j$, is not
zero where $q_j$ is; where the derivative in the charges asks for it, it
is gathered at every particle.

A set is a field of the particles, its first entries the places of its
members in ascending order and the rest $-1-j$ for each other particle
$j$; it is made from the
topology and made again, by counting in chunks of 256 places and a prefix
sum, wherever the particles take a new order (at the start and at each
segment), so that it follows the order of the particles. A kernel takes
an entry of the set where it took a particle. In the gathering a warp
whose first entry is past the last member skips the sum and stores the
forces of zero of the particles it holds: the gathering writes every
force. Where a set holds more than half the particles, a device gathers
every particle instead, as without a set: the oxygen of a four-site water
sits beside its virtual site in the order of the particles, and a warp
reads the points of the grid of both at once, so that the oxygen costs
almost nothing there, while a warp of the set reads those of more
waters. On 103,292 OPC waters (RTX 4090, mixed precision) the
reciprocal sum takes 241 µs a step where it took 274, against 225 for as
many TIP3P waters; with the mesh of the dispersion as well, on 31,808 OPC
waters the two sums take 163 µs where they took 203. A model with no such
particle, or with tunable parameters that could move a coefficient across
zero, has no set, and its kernels are those of before.

**Accuracy.** On Cellulose the reciprocal energy in mixed precision is
30206.42227 kcal/mol against 30206.42041 in double precision, a relative
difference of $6\times10^{-8}$ (D104). Section 9 compares the terms with
sander and with a direct Ewald sum.

## 5.4 The long-range correction of dispersion

Beyond the cutoff the attraction of Lennard–Jones is not zero. With the
particles uniform beyond $r_c$ at the density $N/V$, the missing energy
of a pair potential $-C_6/r^6$ is [[AllenTildesley2017]](references.md#allentildesley2017) [[Shirts2007]](references.md#shirts2007)

$$
E_\text{disp} = \frac{N}{2}\,\frac{N}{V}\int_{r_c}^\infty 4\pi r^2\Big(-\frac{\langle C_6\rangle}{r^6}\Big)\,dr
= -\frac{2\pi}{3}\,\frac{N^2\langle C_6\rangle}{V r_c^3},
$$

and the missing virial, from $\mathbf d\otimes\mathbf K$ over the same
pairs, is $\mathrm{tr}\,\mathsf W_\text{disp} = 6E_\text{disp}$, a
pressure of $2E_\text{disp}/V$. The repulsion is left out.

**Why the virial is $6E_\text{disp}$, not $3E_\text{disp}$.** As
$E_\text{disp}\propto 1/V$, its own derivative gives only
$-dE_\text{disp}/d\ln\mu = 3E_\text{disp}$ under a scaling of the cell by
$\mu$. The rest is the boundary of the cutoff. Under a plain cutoff a pair
that a scaling carries across $r_c$ changes the truncated energy by
$u(r_c) = -C_6/r_c^6$, which the virial of the truncated sum does not see. At a uniform density, the pairs in the shell
that a scaling by $\mu$ moves across are
$\tfrac{N^2}{2V}\,4\pi r_c^2\cdot r_c\,d\ln\mu$, so the truncated energy
changes by $\tfrac{2\pi N^2}{V}\tfrac{\langle C_6\rangle}{r_c^3}\,d\ln\mu =
-3E_\text{disp}\,d\ln\mu$ beyond its virial. With the tail's own change,
the energy outside the virial of the truncated sum changes by
$-6E_\text{disp}\,d\ln\mu$, which is what
$\mathrm{tr}\,\mathsf W_\text{disp} = 6E_\text{disp}$ counts. The
counts of the work of the barostat that take the virial (Section 6.4) are
therefore consistent with the energy of the log, while the exact count,
which evaluates the truncated energy at the scaled positions and so sees
the pairs that cross, adds the tail's change alone,
$E_\text{disp}(V') - E_\text{disp}(V)$.

For a run from
a topology, $C_6 = 4\varepsilon\sigma^6$ and the average counts the pairs
of types and subtracts the excluded pairs, as GROMACS does
[[GromacsManual2025]](references.md#gromacsmanual2025):

$$
\langle C_6\rangle = \frac{\sum_{a,b} N_a(N_b - \delta_{ab})\,C_{6,ab} - 2\sum_\text{excluded} C_{6,ij}}{N(N-1) - 2N_\text{excluded}} .
$$

For a potential given as an expression, $C_{6,ab}$ is $-r^6u_{ab}(r)$
evaluated at $10^4r_c$ and checked against its value at $10^3r_c$ to
$10^{-9}$. Both terms are constants of the host at the volume of the
start, scaled by $V_0/V$ as the cell changes.

**Pair terms over a topology** (D209). A term
`[[energy.pair]]` over the pairs of a topology, such as an NBFIX written
in the control file as $u(\sigma',\varepsilon') - u(\sigma,\varepsilon)$
over two groups, has a tail of its own, of any form. With the same uniform
density, the pairs $\{i,j\}$ that the term counts (its groups select them
and the topology does not exclude them) add

$$
E_\text{tail} = \nu\,\frac{4\pi}{V}\sum_{\{i,j\}} I_{ij},
\qquad I_{ij} = \int_{r_c}^\infty r^2u_{ij}(r)\,dr,
\qquad \nu = \frac{N^2}{N(N-1) - 2N_\text{excluded}},
$$

where $\nu$ is the factor that $N^2\langle C_6\rangle$ above gives each
pair, so that both corrections follow one convention; for
$u = -C_6/r^6$, $E_\text{tail}$ is $E_\text{disp}$. The virial is the
work that the pairs beyond $r_c$ would do under a scaling of the cell,
$-\nu(4\pi/V)\sum\int_{r_c}^\infty r^3u'_{ij}\,dr$, which by parts,
as $r^3u_{ij}\to0$, is

$$
\mathrm{tr}\,\mathsf W_\text{tail} = \nu\,\frac{4\pi}{V}\sum_{\{i,j\}}\big(3I_{ij} + r_c^3u_{ij}(r_c)\big).
$$

The term $3I_{ij}$ is the tail's own change, $-dE_\text{tail}/d\ln\mu$, and
$r_c^3u_{ij}(r_c)$ the shell of pairs that a scaling carries across the
cutoff, as derived above; for $-C_6/r^6$, $r_c^3u(r_c) = 3I$ and
$\mathrm{tr}\,\mathsf W_\text{tail} = 6E_\text{tail}$. The tail is
part of the potential energy as a function of the constants of the term,
$\xi$ and $\boldsymbol\lambda$, and of $V$: $U = U_{r<r_c}(\mathbf x;\xi) +
E_\text{tail}(V;\xi)$. So $\partial F/\partial\xi =
\langle\partial U/\partial\xi\rangle$ (Section 6.8) includes
$\partial E_\text{tail}/\partial\xi$, and so do $dH/d\lambda$, the
energies of the states of `[free_energy]`, and the derivatives of
`observe`. At constant volume it shifts the mean force by a constant and
leaves its covariance with any observable unchanged; at constant pressure
it varies as $1/V$. Without it, a correction term would leave out the
long-range part of what it changes: 0.80 kcal/mol at $\sigma'$ = 3.7 Å on
150 + 150 particles, and the pressure and the density of NPT with it:
1.1% of the volume of 250 + 250 such particles at 100 K, which the tail
restores to that of `[ nonbond_params ]` within 0.2 standard errors.

Pairs whose particles agree in all that the expression reads (the type,
the charge if it reads $q$, the parameters of each particle, the flags of
the groups) have the same $I_{ij}$, so the sum runs over pairs of such
classes, their counts products less the excluded pairs. In
$s = r_c/r$,

$$
I = r_c^3\int_0^1 u(r_c/s)\,s^{-4}\,ds,
$$

whose integrand for $u = \sum_k a_kr^{-k}$ is
$\sum_k a_kr_c^{3-k}s^{k-4}$, a polynomial in $s$ for integers
$k\ge4$. The adaptive Gauss–Kronrod rule of 7 and 15 points
[[Piessens1983]](references.md#piessens1983) integrates a polynomial of
degree up to 13 exactly on its first interval, with an estimate of the
error at round-off, so the tail of a Lennard-Jones ($s^2$ and $s^8$) or of
$r^{-8}$ is its closed form to round-off, and any other expression is
integrated adaptively to $10^{-13}$ relative. The error is relative to the
integral or to the size of the terms that the expression sums, whichever
is larger: the size takes every sum and difference over the absolute values
of its operands. A correction that is the difference of two potentials,
such as an NBFIX at the force field's own $\sigma$ and $\varepsilon$ where a
fit begins, has a tail of 0 and values at the level of the rounding of the
two terms, which no error relative to the value could reach. The integral
converges only if $r^3u(r)\to0$; the driver requires $r^3u$ to fall by at
least a decade per decade of $r$ from $10^3r_c$ to $10^6r_c$, counting a
value within $10^{-12}$ of the size of its terms as 0, and refuses the term
otherwise, as it refuses a term of the time, which has no constant tail.
Such a term is left out with a warning under the default correction, and
refused when the control file asks for the correction; `dispersion_correction
= "NONE"` in the term leaves it out explicitly. Leaving a term out
removes $E_\text{tail}(V;\xi)$ of that term from $U$: the sampled
potential is then its energy within $r_c$ alone, and the pressure,
$-\partial F/\partial V$, loses that term's
$\mathrm{tr}\,\mathsf W_\text{tail}/3V$, while the energy, forces,
and virial of its pairs within $r_c$ are unchanged. A term that
decays as $r^{-3}$ or more slowly, such as $1/r$, has no such limit at a
uniform density and needs a lattice sum instead. The Python model takes
the same choices (D222): a correction that is set on
the system, or a term that asks for its tail, is the control file's key
given, and a term's own `None_` its opt-out, except that under a
switch its default correction is off with a warning rather than an error,
so that its default system, which switches, compiles; the difference of the
default and the opt-out of a term $-c_8/r^8 - ae^{-r/l}/r^4$ is its
$E_\text{tail}$ and $\mathrm{tr}\,\mathsf W_\text{tail}$ to
$4\times10^{-13}$ relative, against a quadrature in $\ln r$ of its own.
The derivatives in $\xi$ and $\lambda$ are central differences of the
integral, extrapolated to a zero step (Richardson), on the host.

The topology's correction leaves its repulsion out, as Amber and GROMACS
do, while a pair term's tail is its whole expression: an NBFIX written as
a correction term differs from the same pair in `[ nonbond_params ]` by
the tail of $\Delta C_{12}$, $\nu(4\pi/V)N_AN_B\,\Delta C_{12}/(9r_c^9)$,
$3.3\times10^{-4}$ kcal/mol in `pair-dispersion.test`, to which they
agree to $2.4\times10^{-8}$ kcal/mol in the energy and $10^{-7}$ in the
virial.

**The shift and its estimate** (D210). Under
`POTENTIAL_SHIFT` each pair within $r_c$ has $u_\text{shift} = u - u(r_c)$,
so the energy lacks $\sum_{r_{ij}<r_c}u_{ij}(r_c) = N_\text{in}\,u(r_c)$ of
the unshifted one, and so do the quantities of `[free_energy]` and `observe`
under a plain cutoff (Section 6.8). The correction adds its mean at the
density of the tail. Around a particle the number of others within $r_c$ is
$\rho\int_{r<r_c}g\,dV = \rho\tfrac{4\pi}{3}r_c^3 + \rho\int_{r<r_c}(g-1)\,dV$,
and when $g\approx1$ beyond $r_c$ the last integral is the whole of
$\rho\int(g-1)\,dV = \rho k_BT\kappa_T - 1$, the compressibility sum rule
[[HansenMcDonald2013]](references.md#hansenmcdonald2013), which is close to
$-1$ for a liquid of low compressibility $\kappa_T$: the particle's own
excluded volume. In the convention of the tail ($\nu$ per pair), the
estimate is therefore

$$
E_\text{sh} = \nu\Big(\frac{4\pi r_c^3}{3V} - \frac1N\Big)\sum_{\{i,j\}}u_{ij}(r_c),
$$

of the $r^{-6}$ part alone for the topology's Lennard-Jones, as its tail,
where it is $E_\text{disp}\,(1 - V/(N\tfrac{4\pi}{3}r_c^3))$, and of the
whole $u(r_c)$ for a pair term. It changes no force, and so no virial: the
shell of pairs that a scaling carries across $r_c$, which under a plain
cutoff changes the truncated energy by $-3E_\text{disp}\,d\ln\mu$ above,
changes the shifted energy by nothing, and $E_\text{sh}$ changes instead.
It has two parts, $E_\text{sh} = c_1/V - c_2$: the first, $\nu\tfrac{4\pi
r_c^3}{3V}\sum u_{ij}(r_c)$, is $E_\text{disp}$ for the topology's
Lennard-Jones and follows the volume as the tail does, changing by
$-3E_\text{disp}\,d\ln\mu$, the same as the shell; the second,
$\tfrac{\nu}{N}\sum u_{ij}(r_c)$, the particle itself taken out of its
neighbors, does not depend on the volume. Under a barostat the first is
scaled by $V_0/V$ from the volume $V_0$ at which the constants were
computed, with the tail, and the second is added as it is (#224: both were
scaled, which left $c_2(1 - V_0/V)$ in the energy, the columns of
`observe`, the free-energy file, and the derivative of Section 3.6 once
the cell had changed). GROMACS counts the neighbors the same
way, with the particle left out, and gives the estimate no virial
(`DispCorr = EnerPres` with `vdw-modifier = Potential-shift`): on 60 + 60
Lennard-Jones particles with an NBFIX pair and on the 224 particles of
propane in water of `test/Driver/Inputs/gromacs`, the correction's
energy, $-3.472644$ and $-3.788929$ kcal/mol, its pressure, and its virial
equal those of GROMACS 2026.3 to their printed digits. Counting all
$\rho\tfrac{4\pi}{3}r_c^3$ instead would put $\tfrac N2\langle C_6\rangle/r_c^6$
more in the energy, 0.067 kcal/mol on the 120 particles. On the fluid of
Section 6.8 the mean number within $r_c$ of a particle, $132.6\pm0.3$,
agrees with $\rho\tfrac{4\pi}{3}r_c^3 - 1 = 133.0$ and not with $134.0$.

## 5.5 Particle mesh Ewald for the dispersion

The correction of Section 5.4 assumes that the density beyond the cutoff
is uniform, a pair distribution $g(r) = 1$. That fails where it matters
most for the dispersion: at an interface or across a membrane the density
along the normal is not uniform, and a correction from the mean density
misplaces the attraction between the layers, which changes surface
tensions and the area per lipid of bilayers
[[Wennberg2013]](references.md#wennberg2013). `lennard_jones = "PME"`
(D162) sums the dispersion over every pair and image instead, by the
Ewald sum for $1/r^6$ of [[Essmann1995]](references.md#essmann1995), so
that the pairs beyond the cutoff count at their actual distances.

**The splitting.** With $C_{6,ij} = c_ic_j$, the dispersion of the cell
and its images is split with

$$
\frac{1}{r^6} = \frac{\gamma(\beta r)}{r^6} + \frac{1 - \gamma(\beta r)}{r^6},
\qquad
\gamma(x) = e^{-x^2}\Big(1 + x^2 + \frac{x^4}{2}\Big),
$$

the first part short-ranged, the second smooth and finite at $r = 0$,
where it is $\beta^6/6$; $1 - \gamma(x) = P(3, x^2)$, the regularized
lower incomplete gamma function. (The paper writes $\gamma$ for this
share, since $g$ is the pair distribution.) The Fourier transform of the
smooth part, $\int e^{-2\pi i\mathbf m\cdot\mathbf r}\,(1 - \gamma(\beta
r))\,r^{-6}\,d^3r = \tfrac{\pi^{3/2}\beta^3}{3}F(b)$ with $b =
\pi\lvert\mathbf m\rvert/\beta$ and

$$
F(b) = (1 - 2b^2)\,e^{-b^2} + 2\sqrt\pi\,b^3\mathrm{erfc}(b),
\qquad F(0) = 1,
$$

gives the reciprocal sum

$$
E^\text{d}_\text{rec} = -\frac{\pi^{3/2}\beta^3}{6V}\sum_{\mathbf m} F\Big(\frac{\pi\lvert\mathbf m\rvert}{\beta}\Big)\,\lvert S_c(\mathbf m)\rvert^2,
\qquad S_c(\mathbf m) = \sum_j c_j e^{2\pi i\,\mathbf m\cdot\mathbf x_j},
$$

which, unlike the Coulomb sum, has a term at $\mathbf m = 0$,
$-\pi^{3/2}\beta^3(\sum_jc_j)^2/(6V)$: the mean-field attraction of a
uniform density, which Section 5.4 estimates from the cutoff outward.
The grid of Section 5.2 computes the sum with the coefficients $c_j$ in
place of the charges and the influence function $C^\text{d}(\mathbf m) =
-\pi^{3/2}\beta^3 F(b)/(3V)$; the spreading, the transforms, and the
gathering are those of the charges.

*Table 5.3. The terms of the dispersion under `lennard_jones = "PME"`.*

| Term | As implemented | Where it runs |
|---|---|---|
| $E^\text{d}_\text{dir}$ | $\sum_{i<j,\ r_{ij}<r_c}'\big[u_{ij}(r_{ij}) + c_ic_j(1 - \gamma(\beta r_{ij}))/r_{ij}^6 - s_{ij}\big]$, with $u_{ij}$ the pair's own Lennard-Jones and $s_{ij}$ the bracket at $r_c$ under `lennard_jones_modifier = "POTENTIAL_SHIFT"`, otherwise 0; the prime omits excluded pairs | The loop over pairs, with the Coulomb terms |
| $E^\text{d}_\text{excl}$ | $\sum_{(i,j)\ \text{excluded}} c_ic_j(1 - \gamma(\beta r_{ij}))/r_{ij}^6$, without a cutoff | A loop over the tuples of the excluded pairs |
| $E^\text{d}_\text{self}$ | $\frac{\beta^6}{12}\sum_i c_i^2$ | A constant of the host, independent of the volume |
| $E^\text{d}_\text{rec}$ | Above | `md.reciprocal` with `dispersion` |

**The coefficients.** The grid needs $C_{6,ij}$ to be a product: $c_i =
2\sqrt{\varepsilon_{aa}}\,\sigma_{aa}^3$ of the type $a$ of particle $i$
with itself, so that $c_ic_j = 4\sqrt{\varepsilon_{aa}\varepsilon_{bb}}\,
(\sigma_{aa}\sigma_{bb})^3$, the geometric rule for both parameters.
Lorentz–Berthelot force fields and pairs set apart (NBFIX) do not follow
it. The direct term of a pair within the cutoff therefore takes the
pair's own Lennard-Jones, $u_{ij}$ with its own $\sigma_{ij}$ and
$\varepsilon_{ij}$, and adds back what the grid takes there,
$c_ic_j(1 - \gamma)/r^6$: within the cutoff every pair has its own
potential, and beyond it the geometric $-c_ic_j/r^6$ less the share
$\gamma(\beta r) \le \gamma(\beta r_c)$ that no term holds, as Wennberg
et al. propose [[Wennberg2013]](references.md#wennberg2013) and OpenMM and
GROMACS do. The difference between the rules beyond the cutoff is not
corrected. The 1–4 pairs are among the excluded pairs, and their scaled
Lennard-Jones remains a term of its own.

**Parameters.** $\beta$ is given, or found by bisection, as in
Section 5.1, such that $\gamma(\beta r_c)$ equals a tolerance, $10^{-3}$
by default: the share of the dispersion at the cutoff that is left out
(GROMACS's `ewald-rtol-lj` is the same quantity); for $r_c = 8$ Å that is
$\beta = 4.189$ nm⁻¹. The grid has its own points and order, by default
from the same largest spacing, 1.2 Å. The correction of Section 5.4 is
off, and asking for it is an error; so is a switch of the
Lennard-Jones, which would take away within the cutoff what the grid
does not give back. A potential shift applies to the whole bracket of
$E^\text{d}_\text{dir}$.

**The excluded pairs.** At small $y = \beta^2r^2$, $1 - \gamma$ is a
difference of numbers near 1 that leaves $y^3/6$; for the pairs of a
molecule, 1 to 3 Å apart, f32 would keep three digits of it. Below
$y = 1$ the kernel takes the series

$$
1 - \gamma(\beta r) = e^{-y}y^3\sum_{k\ge0}\frac{y^k}{(k+3)!}
$$

to $k = 10$, whose terms are positive and whose remainder is below
$10^{-10}$ of the sum, and the closed form above $y = 1$.

**The virial.** As in Section 5.2, $\hat Q$ does not depend on the
strain, and $E^\text{d}_\mathbf m = h\,B\,C^\text{d}\lvert\hat Q\rvert^2$
depends on it through $1/V$ and through $b$, with $\partial
b/\partial\varepsilon_{ab} = -b\,m_am_b/m^2$. With $F'(b) = 6b\,(\sqrt\pi
\,b\mathrm{erfc}(b) - e^{-b^2})$ and $b^2/m^2 = \pi^2/\beta^2$,

$$
W^\text{d}_{ab} = \sum_{\mathbf m}\Big[E^\text{d}_\mathbf m\,\delta_{ab}
+ h\,B\,\lvert\hat Q\rvert^2\,\frac{\pi^{3/2}\beta^3}{3V}\,
\frac{6\pi^2}{\beta^2}\big(e^{-b^2} - \sqrt\pi\,b\mathrm{erfc}(b)\big)\,m_am_b\Big].
$$

The term $\mathbf m = 0$ contributes $E^\text{d}_0\,\delta_{ab}$, the
virial of a constant over the volume, and the self term none. The direct
and excluded terms are differentiated as every pair term is
(Section 3.3). $\mathrm{tr}\,\mathsf W$ of three OPC waters agrees
with $-dU/d\lambda$ to $2\times10^{-7}$.

**On a device.** The kernels of Table 5.2 run twice, once for each sum.
The influence function of the dispersion is not a product of a factor for
each axis, so its tables hold $m_a$, $m_a^2$, and $\lvert b_a\rvert^2$
alone (the first axis times the prefactor), and `scale` and `convolve`
compute $F(b)$ at each point, with one exponential and one complementary
error function, in the type of the grid. On JAC (23,558 particles,
$r_c$ = 8 Å, mixed precision, RTX 3090) a step takes 0.270 ms against
0.210 with the cutoff and the correction: the second grid of $54^3$
points, and the exponential in the direct term (the loop over pairs
takes 70 µs against 64). OpenMM 8.6.1 goes from 0.210 to 0.285 ms on the
same system and grids. Both sums may run on the second stream of D87, to
the same state in the deterministic mode.

**Accuracy.** The reciprocal sum of the dipeptide in water (1168
particles) in mixed precision is $-138.130538$ kcal/mol against
$-138.130465$ in double precision, $5\times10^{-7}$ of it. Section 9
compares the terms with OpenMM, GROMACS, and a direct Ewald sum.

## 5.6 Ewald summation with prolates

The Gaussian of Section 5.1 is one choice of a splitting. Any even kernel
$\chi$ on the line with $\int_0^\infty\chi = 1$ splits the Coulomb energy
in the same way, and `[pme] kernel = "PROLATE"` takes the one that
concentrates the most of itself in a given band limit, the prolate
spheroidal wave function of order zero [[Slepian1961]](references.md#slepian1961),
as both the splitting and the window of the spreading: Ewald summation with
prolates (ESP) [[Liang2026]](references.md#liang2026) (D266).

**The splitting.** With $\Phi(y) = \int_0^y\chi$ and $x = r/r_c$,

$$
\frac1r = \frac{1 - \Phi(r/r_c)}{r} + \frac{\Phi(r/r_c)}{r},
$$

and $\chi = \psi_0^c/C_0$ on $[-1, 1]$, 0 beyond, so that $\Phi = 1$ from
$x = 1$ on: the first part vanishes at the cutoff, up to the size of
$\psi_0^c(1) = \eta$, which sets $c$; the second is smooth. The Gaussian is
$\chi(x) = 2\sqrt{\alpha/\pi}\,e^{-\alpha x^2}$ with $\beta = \sqrt\alpha/r_c$,
for which $\Phi(y) = \mathrm{erf}(\beta r_c y)$: the terms of Table 5.1 follow
from $\chi$ as they follow from the Gaussian.

- **The direct and excluded pairs.** $f q_iq_j(1 - \Phi(r/r_c))/r$ within
  the cutoff and $-f q_iq_j\Phi(r/r_c)/r$ for an excluded pair; the force
  of the first is $f q_iq_j\big[(1-\Phi)/r^3 + \chi(r/r_c)/(r_cr^2)\big]$
  times $\mathbf d$, the derivative of $\Phi$ being $\chi$.
- **The self term.** The smooth part at $r \to 0$ is
  $\Phi(r/r_c)/r \to \chi(0)/r_c$, and each charge meets itself in the
  reciprocal sum once with a half: $E_\text{self} = -f\frac{\chi(0)}{2r_c}\sum_i q_i^2$;
  for the Gaussian, $\chi(0)/(2r_c) = \beta/\sqrt\pi$.
- **The reciprocal sum.** The transform of the smooth part,
  $\int e^{-2\pi i\mathbf m\cdot\mathbf r}\Phi(r/r_c)/r\,d^3r =
  \hat\chi(2\pi r_c\lvert\mathbf m\rvert)/(\pi m^2)$, with $\hat\chi$ the
  transform of $\chi$ as an even function on the line, halved so that
  $\hat\chi(0) = 1$ (the Coulomb kernel $1/(\pi m^2)$ at small $m$): this
  follows from $\Phi(r/r_c)/r = \int_0^{r/r_c}\chi(t)\,dt/r$ written as a
  superposition of shells. So $C(\mathbf m) = \frac{f}{\pi V}\hat\chi(2\pi
  r_c\lvert\mathbf m\rvert)/m^2$ in place of the Gaussian factor; for the
  Gaussian $\hat\chi(2\pi r_c m) = e^{-\pi^2m^2/\beta^2}$. For $\psi_0^c$,
  $\int_{-1}^1\psi_0^c(t)e^{icxt}dt = \lambda_0\psi_0^c(x)$, so
  $\hat\chi(k) = \lambda_0\psi_0^c(k/c)/(2C_0)$ for $k \le c$, and its
  value beyond, of the size $\eta$, is taken as 0, as the paper's code
  takes it.
- **The net charge.** The background removes the $m \to 0$ limit of the
  sum of a uniform charge, $\frac{fQ^2}{2\pi V}\lim(1 - \hat\chi(2\pi r_cm))/m^2$;
  with $\hat\chi(k) = 1 - k^2\langle x^2\rangle/2 + O(k^4)$,
  $\langle x^2\rangle = \int x^2\chi/\int\chi$ over $[-1, 1]$,
  $E_Q = -f\pi Q^2r_c^2\langle x^2\rangle/V$; for the Gaussian
  $\langle x^2\rangle = 1/(2\alpha)$ gives $-f\pi Q^2/(2V\beta^2)$.
- **The virial.** As in Section 5.2, only $1/V$ and the factor of
  $\mathbf m$ depend on the strain. With $\xi = 2\pi\lvert\mathbf m\rvert$
  and $g = d\ln\hat\chi/d\ln\xi$,
  $$
  W_{ab} = \sum_{\mathbf m\ne0}E_\mathbf m\Big[\delta_{ab} - (2 - g)\frac{m_am_b}{m^2}\Big],
  $$
  which for the Gaussian, $g = -2\pi^2m^2/\beta^2$, is the virial of
  Section 5.2. $g\hat\chi = (k/c)\lambda_0\psi_0^{c\,\prime}(k/c)/(2C_0)$
  at $k = r_c\xi$.

**The window.** The spreading takes $\phi_1(d) = \psi_0^{c_1}(2d/P) -
\psi_0^{c_1}(1)$ on $P$ points along each edge, $\lvert d\rvert \le P/2$ in
points of the grid, with its own bandwidth $c_1$, and the product with the
transform of the grid divides by the window's transform,
$B(\mathbf k) = \prod_a\hat\phi_1(2\pi k_a/K_a)^{-2}$ (paper, eq. 9), which
for the B-splines is $\prod_a\lvert b_a\rvert^2$. A particle at the scaled
coordinate $u$ adds to the points from $\lfloor u - h\rfloor - \lfloor
P/2\rfloor + 1$, $h = \tfrac12$ for an odd $P$, so that the points are those
within $P/2$ of $u$. The weights of each point, and their slopes in $u$,
are Chebyshev series in the fraction of the place, within $10^{-14}$ of
$\psi(0)$; the forces take the slopes of the window, so that $\mathbf F_i
= -\partial E_\text{rec}/\partial\mathbf x_i$ of the discrete sum as in
Section 5.2.

*Particles without a charge.* The window, like the B-splines, enters
$Q(\mathbf l)$ multiplied by $q_j$, and the force on particle $j$ is
$-q_j$ times the sum of $\phi$ against the slopes of its window: a
particle with $q_j = 0$ adds nothing to the grid and takes a force of zero.
The sums of the mesh of the prolate kernel therefore run over
$\mathcal J_q$ as those of the Gaussian do (D262), with the
same set, the same rule for the gathering on a device, and the same bits
in the deterministic mode as without it.

*Why the value at the edge is taken off.* The paper's window (its eq. 14)
is $\psi_0^{c_1}(2d/P)$, which is $\psi_0^{c_1}(1) \approx \eta/2$ at the
edge of its support, not 0. When a particle crosses a plane of the grid,
its first point drops out at $d = -P/2$ with that weight, and a new point
enters at $d = +P/2$ with the same weight: the charge on the grid, and with
it $E_\text{rec}$, jumps by a term of order $\eta\,q_i\phi$, while the
force, the derivative between the crossings, has no impulse for it. The
energy of a run then drifts as under an unshifted cutoff: on JAC in NVE,
mixed precision, at $\eta = 10^{-3}$ and 24³, by 75 kcal/mol/ns. B-splines
vanish at their edges, so the Gaussian kernel never had this. With the
constant taken off, $\phi_1$ is continuous on the line, $E_\text{rec}$ is a
continuous function of the positions whose gradient the forces are, and
$\hat\phi_1$, and with it $B$, follows from the same quadrature,
$$
\hat\phi_1(\kappa) = \frac P2\int_{-1}^1\big(\psi_0^{c_1}(t) - \psi_0^{c_1}(1)\big)\cos\frac{\kappa Pt}{2}\,dt
= \hat\phi_1^\text{paper}(\kappa) - 2\psi_0^{c_1}(1)\frac{\sin(\kappa P/2)}{\kappa}.
$$
The last term decays as $1/\kappa$ where the paper's window has the jump's
own $1/\kappa$ tail; the two cancel in leading order, and $\phi_1$, whose
first discontinuity is in its slope, has the faster decay $1/\kappa^2$.
The change costs 2% of $\Delta$ at the default and 12% at
$\eta = 10^{-3}$ (JAC: $1.02$ to $1.04\times10^{-5}$ at 50³ and $6.2$ to
$6.9\times10^{-4}$ at 24³), and over 2 ns the drift falls to 5.1
kcal/mol/ns at $10^{-3}$, the slopes of its halves 2.7 and 6.3 (the
Gaussian's unshifted direct sum at the same tolerance: 281) and to that of the Gaussian default, within the scatter of a
slope, at $10^{-5}$. This shift is MDIR's own: the paper truncates its window
without it, and we know of no source that takes the value at the edge off
in this way. The aliasing estimate below keeps the paper's window,
with which it was calibrated.

**$\psi_0^c$ itself.** The prolate differential operator
$-\frac{d}{dx}(1 - x^2)\frac{d}{dx} + c^2x^2$ commutes with the band-limited
transform and so has the same eigenfunctions; in the normalized Legendre
polynomials it is tridiagonal, coupling degree $k$ with $k \pm 2$, and
$\psi_0^c$ is the eigenvector of its smallest eigenvalue
[[Xiao2001]](references.md#xiao2001) [[Osipov2013]](references.md#osipov2013),
found on the host in f64 by bisection and inverse iteration. Normalized to
$\int\psi^2 = 1$, $\psi_0^c(1) = \eta$ gives the bandwidths of Table 2 of
the paper (9.5392, 12.024, and 14.471 at $10^{-3}$, $10^{-4}$, and
$10^{-5}$); $\psi$, $\lambda_0$, $C_0$, and $\hat\chi$ agree with the
eigenfunction of the integral operator by Gauss–Legendre in 40-digit
arithmetic to $1$–$4\times10^{-16}$.

**Parameters.** $\eta = 10^{-5}$ by default, which gives $c = 14.4712$;
$c_1 = c(\eta/2)$, as the paper's code takes it; $P = 6$; and the grid of
the paper's eq. 15, $n_f = \lceil cL/(\pi r_c)\rceil$ along each edge,
refined at one spacing for the three edges until its estimate of the
aliasing (eq. 10),
$$
E_A = 6\,\frac{\sum_\mathbf k w(\mathbf k)\lvert\hat\phi_1(2\pi(k_a + K_a)/K_a)/\hat\phi_1(2\pi k_a/K_a)\rvert}{\sum_\mathbf k w(\mathbf k)},
\qquad w = \hat\chi(2\pi r_c\lvert\mathbf m\rvert)/m^2
$$
over the band of $\hat\chi$, the largest of the three axes, is at most
$\eta/2$. The factor $\tfrac12$ is measured, not derived. Against a
Gaussian reference at $\mathrm{erfc}(\beta r_c) = 10^{-13}$, 0.33 Å, and
order 8 (which agrees with the prolate kernel at $\eta = 10^{-9}$ to
$1.3\times10^{-7}$), with $\Delta$ over the Coulomb forces of the topology,
its exclusions and 1-4 scaling included (the forces of the system less
those of the same system without charges): $\Delta$ of a fine grid, the
floor that the tails of $\chi$ and $\phi_1$ leave, is $0.6$ to $0.9\,\eta$;
$E_A$ lies 2.5 to 3 times below $\Delta$ near that floor; and with
$E_A \le \eta$ the grids of $P = 5$ and 7 fell at 2.7 to 2.9 times the
floor while $P = 6$ fell at 1.3 to 1.5. With $E_A \le \eta/2$ every
$\eta$ from $10^{-3}$ to $10^{-5}$ and $P$ from 5 to 8 lands within 1.3
times its floor on JAC and Cellulose, so that $\Delta \lesssim 1.2\,\eta$;
at $\eta = 10^{-6}$, 1.4 to 2.0 times ($\Delta \le 1.7\,\eta$). Where
twice the grid of eq. 15 still leaves $E_A$ above $\eta/2$, the window is
too short for the tolerance (with 8 points at $10^{-9}$, $E_A$ falls only
to $6.4\times10^{-10}$ on 500³), and the automatic grid is refused with
what to do. The same tolerance is thus a very different accuracy with the
two kernels: the Gaussian's $\mathrm{erfc}(\beta r_c) = 10^{-5}$ with its
grid of 1.2 Å gives $\Delta = 1.4\times10^{-3}$ on JAC and Cellulose, the
prolate kernel's $\eta = 10^{-5}$ gives $1.0$ to $1.2\times10^{-5}$. The two tolerances mean different things. The
Gaussian's truncates the direct sum alone, as pmemd's `dsum_tol` and
GROMACS's `ewald-rtol` do (both $10^{-5}$ by default), and leaves $\Delta$
to the mesh: B-splines of order 4 at about 1 Å in pmemd and MDIR and
1.2 Å in GROMACS give $\Delta \approx 10^{-3}$ at the standard settings of
all three (MDIR's default $1.4\times10^{-3}$, the Amber suite's
$7.9\times10^{-4}$ on Cellulose). $\eta$ with the grid rule sets $\Delta$
itself, so that the prolate default is about a hundred times the force
accuracy of the standard settings, not the same setting.
(Over the Coulomb forces of all pairs without exclusions, as the paper and
the earlier sweep of #334 measured it, the same errors read about 4.5 times
smaller, the bonded pairs inflating the norm.)

**The pair terms.** $1 - \Phi(x)$ and $\Phi(x)/x$, the latter computed as
$\int_0^1\chi(xt)\,dt$ with no cancellation at small $x$, are tables of
cells of polynomials in the place within a cell (Section 3.1, the tables of
the control file): 64 cells of 10 terms within $10^{-14}$ for kernels in
f64, 128 cells of 4 terms within $4\times10^{-10}$ for kernels in f32. The
excluded pairs take $\Phi(x)/(xr_c)$ within the cutoff and $1/r$ beyond it.

In the loop over pairs of a kernel in f32 (D280) the
direct sum is instead a function of $r^2$ alone, so that it becomes a
table of $r^2$ as the Gaussian's $\mathrm{erfc}(\beta r)/r$ does
(Section 7.2): no square root, no index of a cell, one load and three
fused multiply–adds a pair. Since $\Phi(1) = 1$,
$$
1 - \Phi(x) = \int_x^1 \chi(t)\,dt = (1 - x)\,\bar\chi(x),
\qquad
\bar\chi(x) = \int_0^1 \chi\big(x + (1 - x)\tau\big)\,d\tau ,
$$
the mean of $\chi$ over $[x, 1]$, a polynomial of the degree of the
series of $\psi_0^c$, which Gauss–Legendre quadrature in $\tau$ gives
without cancellation and which a Chebyshev series in $2x - 1$ holds to
$10^{-14}$ of its largest term (20, 23, and 27 terms at
$\eta = 10^{-3}$, $10^{-5}$, $10^{-9}$). The term of a pair is then
$f q_i q_j (1 - x)\bar\chi(x)/r$, a function of $r = \sqrt{s}$ that the
compiler outlines and tabulates in $s$; its derivative, the force, is
outlined with it. The factor $1 - x$ matters near the cutoff, where
$1 - \Phi(x) \approx \chi(1)(1 - x)$ is of the order of
$\eta(1 - x)$: the same function written as $1/r - \Phi(x)/(x r_c)$,
which is a polynomial in $x^2$ since $\chi$ is even, cancels there to
below the rounding of f64 relative to its value, and the energy's table
missed the test of Section 7.2 at every number of bits. Written with the
factor, the relative error of the function is that of $1 - x$ and of
$\bar\chi$.
On JAC ($r_c = 9$ Å) the force's table has $b = 7$ at $\eta = 10^{-3}$
and $10^{-4}$ and $b = 8$ at $10^{-5}$ (20 and 40 KB), the energy's
$b = 8$ and 9, against $b = 7$ for the Gaussian's at the same cutoff; the
cubics in f32 are within $2.3\times10^{-7}$ of the function, relative to
its largest value on an interval. The kernels of f64 keep the cells,
which take less than the series evaluated as itself. The forces move by
$3.4$ to $4.1\times10^{-7}$ of the RMS Coulomb force, and $\Delta$ in mixed
precision not at its third digit. On the CPU target (16 cores, mixed
precision, 9 Å) the loop over pairs then takes 0.98 to 1.02 times the
Gaussian's, from 1.5 to 1.85, and the step of the prolate default 0.96
to 1.03 times that of the fastest Gaussian order at $\Delta \approx
10^{-5}$ and 1.05 to 1.24 times at $10^{-4}$ and $10^{-3}$, where the
window of the spreading is what remains of the difference.

**On a device.** The kernels are generated from those of the dispersion
(Section 5.5) and the bricks of D108. The coefficients of the window are
converted once a sum into the type of the grid, term by term, and read
through the global (read-only) path rather than the constant path, so that
the lanes of a term read adjacent words: in a prototype, the same
coefficients in constant memory, read at addresses that differ across the
lanes of a warp, made the gathering 1.5 to 2.8 times slower for an odd
number of terms, which the code generated for it (twice the loads from the
constant bank for an odd number) and the timings point to; there were no
counters of the hardware to confirm the cause (PR #337, P0). The spreading and the gathering run a warp a particle;
lanes $0$ to $3P - 1$ evaluate each weight once and the others take them by
shuffles. The deterministic mode adds the grid in fixed point (D70).

**Accuracy.** On three OPC waters the Coulomb terms sum to within
$1.6\times10^{-6}$ of the Ewald sum over the wave vectors at the default;
the trace of the virial agrees with $-dU/d\lambda$ to $2.2\times10^{-6}$,
its diagonal with $-dU/d\varepsilon_{aa}$ to $6.6\times10^{-6}$, and all
nine components of $\partial U/\partial\boldsymbol\varepsilon$ in a sheared
rhombic dodecahedron with central differences of the energy to
$1.5\times10^{-8}$ in double precision (Section 3.6); in a rhombic
dodecahedron the default is within $1.0\times10^{-8}$ of the Gaussian
kernel at a tight setting. $\partial H/\partial\lambda$ of a decoupled
ethanol agrees with central differences of the energies of the states
beside it to $2.9\times10^{-5}$ kcal/mol, within the tolerance of the
Gaussian's test, and the derivative in the charges to $7.6\times10^{-11}$
relative in double precision. On JAC, $\Delta = 1.04\times10^{-5}$ at the default (50³) on the
CPU and on a GPU in double; mixed precision adds about $1.0\times10^{-4}$
with either kernel ($\Delta = 1.02\times10^{-4}$ for the prolate default,
the Gaussian of order 6 at $1.1\times10^{-5}$, and order 8 at
$1.1\times10^{-5}$ alike), so that below $10^{-4}$ the accuracy of a run
in mixed precision is that of the precision, not of the kernel. In NVE
over 2 ns the energy of JAC drifts by $-0.06$ kcal/mol/ns at the default
(the slopes of its halves 0.87 and 0.29) against $-0.01$ (0.13, 0.86) for
the Gaussian default; in NPT over 1 ns its volume is $234{,}301 \pm 113$
Å³ against $234{,}470 \pm 101$, 1.1 combined standard errors apart.

**Cost.** ESP was built for many processors, where the smaller grid saves
communication: the transposes of a distributed transform move data in
proportion to the points of the grid, and at the same $\Delta$ the prolate
grid has 6 to 7 times fewer points than the Gaussian of order 6 at
$10^{-5}$ and 10 to 12 times fewer at $10^{-4}$; its spreading reaches
$P - 1$ layers beyond a domain on each side, 5 for $P = 6$, as many as
order 6 and two more than order 4. On one device the smaller grid saves the
transform, a small part of the step, and the spreading of $P^3$ points
costs more. MDIR's whole step in mixed precision (JAC, Cellulose, STMV;
an RTX 4090 and an RTX PRO 6000, each comparison within one job on one
device; `main` at 0a336ff9, with tiled spreading where its rule takes it,
Section 5.3, D282) takes 1.17 to 1.44 times as long with the prolate kernel
at $\Delta \approx 10^{-3}$, the tier of the standard settings of pmemd,
GROMACS, and MDIR, as with the Gaussian default (order 4); 0.94 to 1.15
times as long at $\Delta \approx 10^{-4}$ as order 6; and 0.84 to 1.02
times at $10^{-5}$ as the faster of orders 6 and 8, faster on Cellulose and
STMV (the table, with the grids, is in `docs/pme.md`, Section 7). The
Gaussian kernel stays the default. The tiles serve both kernels: with them
the prolate default took 0.83 to 0.87 times the step without tiles on
Factor IX, Cellulose, and STMV (RTX 4090), its spreading 72, 209, and
540 µs a step against 137, 585, and 1640.
