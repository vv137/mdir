# 5. Electrostatics

MDIR computes the electrostatic energy of a periodic system by particle
mesh Ewald (PME) in the smooth form of [[Essmann1995]](references.md#essmann1995), after
[[Darden1993]](references.md#darden1993). This section derives the terms as they are implemented,
the forces and the virial of the reciprocal sum, and how the sum runs on a
device; Sections 5.4 and 5.5 treat the dispersion beyond the cutoff, by a
correction and by the same mesh. Internally the energies are in kJ/mol and the Coulomb constant is
$f = 138.935457644$ kJ nm/(mol e²), that of CODATA 2018
[[Tiesinga2021]](references.md#tiesinga2021), whatever the format of the
input: Amber's constant is smaller by a factor 1.0000346 and CHARMM's
larger by 1.0000238, so the electrostatic terms of the same charges differ
from those programs' by these factors (Section 9.1).

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
$\mu$. The rest is the boundary of the cutoff. Under a plain cutoff a pair
that a scaling carries across $r_c$ changes the truncated energy by
$u(r_c) = -C_6/r_c^6$, which the virial of the truncated sum does not see. At a uniform density, the pairs in the shell
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
\operatorname{tr}\mathsf W_\text{tail} = \nu\,\frac{4\pi}{V}\sum_{\{i,j\}}\big(3I_{ij} + r_c^3u_{ij}(r_c)\big).
$$

The term $3I_{ij}$ is the tail's own change, $-dE_\text{tail}/d\ln\mu$, and
$r_c^3u_{ij}(r_c)$ the shell of pairs that a scaling carries across the
cutoff, as derived above; for $-C_6/r^6$, $r_c^3u(r_c) = 3I$ and
$\operatorname{tr}\mathsf W_\text{tail} = 6E_\text{tail}$. The tail is
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
$\operatorname{tr}\mathsf W_\text{tail}/3V$, while the energy, forces,
and virial of its pairs within $r_c$ are unchanged. A term that
decays as $r^{-3}$ or more slowly, such as $1/r$, has no such limit at a
uniform density and needs a lattice sum instead. The Python model takes
the same choices (D[python-dispersion]): a correction that is set on
the system, or a term that asks for its tail, is the control file's key
given, and a term's own `None_` its opt-out, except that under a
switch its default correction is off with a warning rather than an error,
so that its default system, which switches, compiles; the difference of the
default and the opt-out of a term $-c_8/r^8 - ae^{-r/l}/r^4$ is its
$E_\text{tail}$ and $\operatorname{tr}\mathsf W_\text{tail}$ to
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
changes the shifted energy by nothing, and $E_\text{sh}\propto1/V$ changes
by $-3E_\text{sh}\,d\ln\mu$ instead, the same as the shell to the factor
$1 - V/(N\tfrac{4\pi}{3}r_c^3)$. GROMACS counts the neighbors the same
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
F(b) = (1 - 2b^2)\,e^{-b^2} + 2\sqrt\pi\,b^3\operatorname{erfc}(b),
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
\,b\operatorname{erfc}(b) - e^{-b^2})$ and $b^2/m^2 = \pi^2/\beta^2$,

$$
W^\text{d}_{ab} = \sum_{\mathbf m}\Big[E^\text{d}_\mathbf m\,\delta_{ab}
+ h\,B\,\lvert\hat Q\rvert^2\,\frac{\pi^{3/2}\beta^3}{3V}\,
\frac{6\pi^2}{\beta^2}\big(e^{-b^2} - \sqrt\pi\,b\operatorname{erfc}(b)\big)\,m_am_b\Big].
$$

The term $\mathbf m = 0$ contributes $E^\text{d}_0\,\delta_{ab}$, the
virial of a constant over the volume, and the self term none. The direct
and excluded terms are differentiated as every pair term is
(Section 3.3). $\operatorname{tr}\mathsf W$ of three OPC waters agrees
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
