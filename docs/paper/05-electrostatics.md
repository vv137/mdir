# 5. Electrostatics

MDIR computes the electrostatic energy of a periodic system by particle
mesh Ewald (PME) in the smooth form of [[Essmann1995]](references.md#essmann1995), after
[[Darden1993]](references.md#darden1993). This section derives the terms as they are implemented,
the forces and the virial of the reciprocal sum, and how the sum runs on a
device. Internally the energies are in kJ/mol and the Coulomb constant is
$f = 138.935457644$ kJ nm/(mol e²).

## 5.1 The splitting

The Coulomb energy of the charges of the cell and their periodic images,
less the pairs that the force field excludes (bonded neighbors, 1–2, 1–3,
and 1–4, whose 1–4 interaction is a scaled term of its own), is split with
$1/r = \operatorname{erfc}(\beta r)/r + \operatorname{erf}(\beta r)/r$
[[Ewald1921]](references.md#ewald1921):

$$
E = E_\text{dir} + E_\text{rec} + E_\text{self} + E_\text{excl} + E_Q .
$$

*Table 5.1. The terms of the splitting.*

| Term | As implemented | Where it runs |
|---|---|---|
| $E_\text{dir}$ | $f\sum_{i<j,\ r_{ij}<r_c}' q_iq_j\big[\operatorname{erfc}(\beta r_{ij})/r_{ij} - s\big]$, $s = \operatorname{erfc}(\beta r_c)/r_c$ under `coulomb_modifier = "POTENTIAL_SHIFT"`, otherwise 0; the prime omits excluded pairs | The loop over pairs, with Lennard–Jones |
| $E_\text{excl}$ | $-f\sum_{(i,j)\ \text{excluded}} q_iq_j\operatorname{erf}(\beta r_{ij})/r_{ij}$, over every excluded pair, without a cutoff | A loop over the tuples of the excluded pairs |
| $E_\text{self}$ | $-f\frac{\beta}{\sqrt\pi}\sum_i q_i^2$ | A constant of the host |
| $E_Q$ | $-f\frac{\pi Q^2}{2V\beta^2}$, $Q = \sum_i q_i$ [[Hub2014]](references.md#hub2014) | A constant of the host, scaled with $1/V$ |
| $E_\text{rec}$ | Section 5.2 | `md.reciprocal` |

The force of the direct sum follows from Section 3.3 with

$$
-\frac{u'(r)}{r} = f q_iq_j\Big[\frac{\operatorname{erfc}(\beta r)}{r^3}
+ \frac{2\beta}{\sqrt\pi}\frac{e^{-\beta^2r^2}}{r^2}\Big],
$$

and that of an excluded pair with

$$
-\frac{u'(r)}{r} = f q_iq_j\Big[\frac{2\beta}{\sqrt\pi}\frac{e^{-\beta^2r^2}}{r^2}
- \frac{\operatorname{erf}(\beta r)}{r^3}\Big] .
$$

The term $E_Q$ removes the
energy of the uniform background that neutralizes a net charge; its
virial is $3E_Q$, which makes its pressure $E_Q/V$, and its value at the
volume of the step is $E_Q(V_0)\,V_0/V$.

**Parameters** (D71). $\beta$ is given, or found by bisection such that
$\operatorname{erfc}(\beta r_c)$ equals a tolerance, $10^{-5}$ by default:
100 halvings of $[0, h]$, $h$ doubled from 1 nm⁻¹ until
$\operatorname{erfc}(h r_c)$ is below the tolerance. The grid has $K_a$
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
lower-triangular $H$, $\mathbf m = H^{-1}\mathbf k$ (on the CPU, D123),

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

The driver computes the moduli $\lvert b_a\rvert^2$ in f64 from the
recursion of $M_n$ and passes them as a table. Under `influence =
"OPTIMAL"` it multiplies them by $\lambda^2$ with $\lambda = S_n(x)/S_{2n}(x)$,
$S_p(x) = \sum_{j=-50}^{50}\big(x/(x+\pi j)\big)^p$, and $x = \pi k/K$, a
factor of the aliasing of the B-splines that brings the energy of the grid
closer to the Ewald sum [[Ballenegger2012]](references.md#ballenegger2012); the default is the influence
function of [[Essmann1995]](references.md#essmann1995).

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
\operatorname{tr}\mathsf W = \sum_{\mathbf m\neq0}E_\mathbf m\Big(1 - \frac{2\pi^2m^2}{\beta^2}\Big).
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
| | Otherwise | Orders 6 and 8, or f64: a thread for each particle and point along $z$, with floating-point atomics; in the deterministic mode the products $q\prod M$ are scaled by $2^{40}$, rounded to 64-bit integers, added with integer atomics, and scaled back, so the grid does not depend on the order of the threads [[LeGrand2013]](references.md#legrand2013) (D70, D84) |
| Transform | cuFFT | Real-to-complex and back, in the type of the grid; plans are cached and bound to the stream of the kernels |
| Tables | `tables` | The separable factors of Section 5.2 |
| Product | `scale` or `convolve` | $\hat Q\leftarrow BC\,\hat Q$. When the energy or the virial is requested, `convolve` also sums $E_\mathbf m$ and the six components of $\mathsf W$: a block of 128 threads reduces in the type of the grid, stores its sums in f64, and one block adds the blocks in f64, in an order that depends only on the size of the grid; the host copies seven numbers |
| Gathering | `gather_weights` | Order 4 in f32: four threads a particle; each recomputes the splines and their slopes from the stored fractions by the unrolled recursion, in the type of the grid, and the four sum by shuffles (D102). Otherwise four lanes a particle for order 4 and eight for 6 and 8, which recompute the places |

The fraction along the cell stays f64 because a position of a large cell
in f32 would lose the digits that place it within a point: the product
with $K$ in f32 resolves a point to $K\,2^{-24}$, $1.6\times10^{-5}$ of a
point for $K = 270$ (D78, amended).

On the host every step is f64: the charges are added in fixed point by a
serial loop, the FFT is pocketfft [[Reinecke2019]](references.md#reinecke2019) (serial, real-to-complex
along $z$ and complex along $y$ and $x$, with cached plans), and the
product and gathering are loops over the grid and the particles, the
gathering parallel under OpenMP.

**Cost.** On Cellulose (408,609 atoms, a grid of 270 × 126 × 126) the
places take 27 µs a step, the additions 214, the copy and the clearing 63
(304 µs in all), the two transforms 292, the product with the virial 30,
and the gathering 90 (RTX 3090; medians of a launch, from D102, D104,
D108, and D109).

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
pairs, is $\operatorname{tr}\mathsf W_\text{disp} = 6E_\text{disp}$, a
pressure of $2E_\text{disp}/V$. The repulsion is left out.

**Why the virial is $6E_\text{disp}$, not $3E_\text{disp}$.** As
$E_\text{disp}\propto 1/V$, its own derivative gives only
$-dE_\text{disp}/d\ln\mu = 3E_\text{disp}$ under a scaling of the cell by
$\mu$. The rest is the boundary of the cutoff. A run with a correction
for the dispersion takes a plain cutoff (the driver rejects a switch or a
shift with it), so a pair that a scaling carries across $r_c$ changes the
truncated energy by $u(r_c) = -C_6/r_c^6$, which the virial of the
truncated sum does not see. At a uniform density, the pairs in the shell
that a scaling by $\mu$ moves across are
$\tfrac{N^2}{2V}\,4\pi r_c^2\cdot r_c\,d\ln\mu$, so the truncated energy
changes by $\tfrac{2\pi N^2}{V}\tfrac{\langle C_6\rangle}{r_c^3}\,d\ln\mu =
-3E_\text{disp}\,d\ln\mu$ beyond its virial. With the tail's own change,
the energy outside the virial of the truncated sum changes by
$-6E_\text{disp}\,d\ln\mu$, which is what
$\operatorname{tr}\mathsf W_\text{disp} = 6E_\text{disp}$ counts. The
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
