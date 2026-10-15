# Particle Mesh Ewald in M1

Status: decided (2026-09-30), D69 to D71. Stage M1h of
[design-m1.md](design-m1.md), whose Section 8.2 lists the parts. The keys
in brackets are those of [references.md](references.md).

## 1. The sum

The Coulomb energy of a periodic system of point charges is split by the
Ewald sum [[Ewald1921]](references.md#ewald1921) with a splitting
parameter $\beta$, in the units of MDIR with $f$ the Coulomb constant
([conventions.md](conventions.md)):

$$
\begin{aligned}
E &= E_\text{dir} + E_\text{rec} + E_\text{self} + E_\text{excl} + E_Q, \\
E_\text{dir} &= f \sum_{i < j,\ \text{not excluded},\ r_{ij} < r_c} q_i q_j \Big(\frac{\mathrm{erfc}(\beta r_{ij})}{r_{ij}} - s\Big), \\
E_\text{rec} &= \frac{f}{2\pi V} \sum_{\mathbf m \ne 0} \frac{\exp(-\pi^2 m^2/\beta^2)}{m^2}\, \lvert S(\mathbf m)\rvert^2, \\
E_\text{self} &= -\frac{f\beta}{\sqrt\pi} \sum_i q_i^2, \\
E_\text{excl} &= -f \sum_{(i,j)\ \text{excluded}} q_i q_j \frac{\mathrm{erf}(\beta r_{ij})}{r_{ij}}, \\
E_Q &= -\frac{f\pi Q^2}{2V\beta^2}.
\end{aligned}
$$

| Symbol | Meaning |
|---|---|
| $\mathbf m$ | A vector of the reciprocal lattice, $(k_x / L_x, k_y / L_y, k_z / L_z)$ with whole $k$, in nm⁻¹; $m = \lVert\mathbf m\rVert$ |
| $S(\mathbf m)$ | The structure factor $\sum_i q_i \exp(2\pi i\, \mathbf m \cdot \mathbf x_i)$ |
| $s$ | The shift of the direct sum: $\mathrm{erfc}(\beta r_c) / r_c$, or $0$ |
| excluded | The pairs of the relation `excluded` of the topology: one, two, and three bonds apart, with the sites of Section 19 of design-m1.md. The pairs three bonds apart are among them; their scaled Coulomb is the term `pairs14`, as before, and $E_\text{excl}$ takes their share of $E_\text{rec}$ out again. |
| $Q$ | The net charge $\sum_i q_i$; $E_Q$ is the energy of the uniform background that neutralizes it [[Hub2014]](references.md#hub2014) |

$E_\text{rec}$ is computed by smooth particle mesh Ewald
[[Essmann1995]](references.md#essmann1995), after
[[Darden1993]](references.md#darden1993): the charges are spread to a
grid of $K_1 \times K_2 \times K_3$ points with cardinal B-splines of order $p$
(4 by default), the grid is transformed by a real-to-complex FFT, each
point is multiplied by the influence function, and the inverse transform
gives the potential on the grid, from which the forces follow by the
derivatives of the same B-splines:

$$
\begin{aligned}
Q(\mathbf k) &= \sum_i q_i\, M_p(K_1 u_{1i} - k_1)\, M_p(K_2 u_{2i} - k_2)\, M_p(K_3 u_{3i} - k_3), \\
E_\text{rec} &= \tfrac12 \sum_{\mathbf k} Q(\mathbf k)\, (\theta \star Q)(\mathbf k), \qquad \theta = \mathcal F^{-1}[BC], \\
B(\mathbf m) &= \lvert b_1(m_1)\rvert^2\, \lvert b_2(m_2)\rvert^2\, \lvert b_3(m_3)\rvert^2 \quad \text{(the Euler exponential splines)}, \\
C(\mathbf m) &= \frac{f}{\pi V}\, \frac{\exp(-\pi^2 m^2/\beta^2)}{m^2}, \qquad C(0) = 0,
\end{aligned}
$$

with $u_{ai}$ the fractional coordinates of particle $i$ along edge $a$.

### 1.1 The influence function

Two influence functions are in use, and `influence` chooses:

| `influence` | Influence function | Engine |
|---|---|---|
| `"SPME"`, the default | $BC$ as above [[Essmann1995]](references.md#essmann1995) | GROMACS |
| `\"OPTIMAL\"` | $BC$ times $\lambda_1(m_1)^2\, \lambda_2(m_2)^2\, \lambda_3(m_3)^2$, with $\lambda(m) = S_n(x) / S_{2n}(x)$, $S_p(x) = \sum_{j=-50}^{50} \big(x / (x + \pi j)\big)^p$, $x = \pi m / K$, and $\lambda(0) = 1$: a factor for each edge from the sums over the aliases of $m$ of the B-splines of order $n$ and $2n$, which brings the energy of the grid toward that of the Ewald sum in the mean (on influence functions of mesh Ewald, [[Ballenegger2012]](references.md#ballenegger2012)) | sander, by default |

Neither is more accurate for every system. On the dipeptide in OPC with
ff19SB and a grid of 30 × 36 × 25 points of order 4 the reciprocal sum of
`"SPME"` is below the Ewald sum by 1.8 × 10⁻³ of it, that of `"OPTIMAL"` by
2 × 10⁻⁴; on the three waters of `test/Driver/pme.test` with 35³ points,
`"SPME"` is off by −7 × 10⁻⁴ and `"OPTIMAL"` by +2 × 10⁻³. Both approach
the Ewald sum as the grid and the order grow. MDIR takes the one of the
literature by default.

For a number of points $K$ that is odd, MDIR takes $m$ from $-(K - 1)/2$ to
$(K - 1)/2$; sander and pmemd take the index $(K - 1)/2$ as
$m = -(K + 1)/2$ in the factor $\lambda$, so that the two differ on odd
grids: on Cellulose with 270 × 125 × 125 points the electrostatic energy
of MDIR differs from pmemd's by $1.6\times10^{-6}$ of it and the forces
by $10^{-3}$ of the rms force, with 270 × 128 × 128 points by
$2\times10^{-9}$ and $1.5\times10^{-7}$ (`scripts/validation/suite`). On
even grids they agree. MDIR chooses even grids itself.

### 1.2 The virial

The virial of the reciprocal sum, in the convention of
[conventions.md](conventions.md) ($\mathsf W = \sum \mathbf d \otimes \mathbf F$, the pressure
$(2K + \mathrm{tr}\,\mathsf W) / (3V)$), is the derivative under a strain of the cell:

$$\mathsf W_{ab} = \sum_{\mathbf m \ne 0} E_\mathbf m \Big(\delta_{ab} - 2\Big(\frac{1}{m^2} + \frac{\pi^2}{\beta^2}\Big) m_a m_b\Big),$$

where $E_\mathbf m$ is the share of $\mathbf m$ in $E_\text{rec}$. $E_Q$ adds $E_Q \delta_{ab}$; $E_\text{self}$ adds
nothing; $E_\text{dir}$ and $E_\text{excl}$ add the virial of their pairs, as every term over
pairs does.

## 2. What is new, and where (D69)

| Part | Where |
|---|---|
| $E_\text{dir}$ | The pair term of the topology, $f q_i q_j \big(\mathrm{erfc}(\beta r) / r - s\big)$: no new op. `math.erfc` has a derivative. |
| $E_\text{excl}$ | An `md.sum_tuples` over the relation `excluded` with `distance(0, 1)` and the charges gathered: no new op |
| $E_\text{self}$, $E_Q$ | Numbers from the driver |
| $E_\text{rec}$ | One new op, `md.reciprocal %x, %q, %cell, %moduli`, with the grid, the order, $\beta$, and the Coulomb constant as attributes, that yields the energy, the forces, and the virial at once, all from the grid. Differentiation takes the forces and the virial from it; it is not differentiated through. |
| The lowering of `md.reciprocal` | Templates in IR, as the build of neighbor structures is (`lib/Runtime/Templates`): spreading, the product with the influence function with the sums of the energy and the virial, and gathering the forces, for the CPU and for a GPU. The FFT is a call of the runtime. A triclinic cell takes functions of its own, `_triclinic` in `PME.mlir` and the kernels of `PMEGPUTriclinic.mlir` on a device (D123, D125). With a fourth result, the op also yields the potential of the grid at the particles, $\partial E/\partial q_i$, gathered with the weights of the splines (`pme_potential`), for the derivative in the charges (D231, [python-gradient.md](python-gradient.md#the-charges-with-particle-mesh-ewald)) |
| FFT | On the host, pocketfft [[Reinecke2019]](references.md#reinecke2019) under the BSD license, in `libmdrt`; on a device, cuFFT of the CUDA toolkit, in `libmdrt_cuda`, with a plan kept for each size of grid (D64). |
| The influence function | Computed where it is used, from $\beta$, the cell, and the factors of the three edges, $\lvert b_a(k)\rvert^2$ (times the factor of the aliasing with `"OPTIMAL"`), which the driver passes as a table of three rows: the factors do not depend on the cell, so a cell that changes (the barostat, M1j) needs nothing new. On a device a kernel multiplies them, at each evaluation, by $\exp(-\pi^2 m^2/\beta^2)$ along each edge in f64, and keeps the wave numbers and their squares beside them, each rounded once to the type of the grid (D104) |

## 3. Spreading that does not depend on the order (D70)

Many particles add to one point of the grid. Additions of floating-point
numbers in the order in which threads arrive give results that differ from
run to run, which breaks the deterministic level (P6). The grid is
therefore accumulated in fixed point [[LeGrand2013]](references.md#legrand2013):
each contribution $q_i M_p M_p M_p$ is rounded to a 64-bit integer at the scale
2⁴⁰ and added with an integer atomic, whose sum does not depend on the
order. The grid is converted to floating point before the FFT.

This is the deterministic mode of a run (`deterministic = true` in
`[execution]`, D84). By default a device adds the contributions in the type
of the grid with floating-point atomics, straight into the grid that the
FFT takes, and the last bits of the grid depend on the order of the
threads. A thread adds the contributions of one particle at one point
along z to the points of the other two directions, so that the threads of
a particle add to neighboring points at once (as GROMACS arranges them);
on JAC (RTX 3090) the spreading took 38 µs a step in fixed point with a
thread per particle and takes 24 in `f32` with a thread per particle and
point along z. The NVPTX backend of LLVM turns an atomic addition of `f32`
into a loop of compare-and-swap, four times slower; the template for `f32`
takes PTX's own reduction, `red.relaxed.gpu.global.add.f32`.

With splines of order 4 and the grid in `f32`, the default mode spreads
in three kernels. The first places each particle, a thread a particle,
into arrays by component (6 values a particle, the first points and the
fractions along x, y, z, D109), a scratch of the op; the additions compute
the weights of the splines from the fractions in closed form, and the
gathering computes the splines and their slopes from them by the
recursion (D102). The second adds the charges with a warp for each particle into a grid
of bricks: 4 × 4 points in x-y with z inside them, so that the points of a
particle at one z are 16 consecutive values of at most 4 bricks, and lane
16 c + 4 b + a of the warp, with c 0 or 1, adds the points (a, b, c) and
(a, b, c + 2); the bricks are
a buffer of their own, zero when it is allocated. The third copies the
bricks into the grid of the transform and leaves them zero (D108): a block
takes a brick and 32 of its slabs along z, reads the 512 values in order
through a tile of the memory of the block, writes zeros back after the
reads (written beside them, the reads waited on the stores, 204 µs), and a
warp writes each row of the brick as 32 values along z in order. On
Cellulose (408,609 atoms, grid 270 × 126 × 126, RTX 3090) they take 27,
214, and 63 µs a step, 304 in all (from 47, 231 with 28 to clear the
bricks, and 61: the copy in the order of the grid read the bricks 64 bytes
apart, and the places held 18 values a particle), against 553 for the spreading of a thread for each particle and
point along z; pmemd.cuda takes about 344 (93 for its weights, 207 for its
additions, 44 for its copy). Standalone
(`scripts/experiments/neighbor-structures/spread.cu`): the additions in the
order of the transform take 351 µs with a warp a particle and 434 with 4
threads, in bricks 257; atomics of `i32` in fixed point are slower than
those of `f32`; and computing the splines in every lane of a warp, as a
first version did, left the kernel bound by its instructions (375 a
particle, 442 µs). The orders 6 and 8, and the deterministic mode, keep the
spreading of a thread for each particle and point along z.

The gathering
takes a thread for each particle and point along z (the least power of 2
not below the order, those beyond it adding zeros), and adds their sums
with shuffles: 127 µs on Cellulose, from 339 with a thread a particle.

The loops over the order in the kernels of the template unroll when the
template is instantiated, and the arrays of the B-splines become values
(`sroa`, `mem2reg`); left as loops, the arrays were in local memory, and
the gathering on Cellulose took 303 µs. The canonicalization that goes
with the unrolling hoists the constants of the kernels to their functions;
they are sunk back into the kernels, or the outlining makes them
arguments and a division by 32 one by an argument (the additions into the
bricks took 350 µs so, against 231). The gathering computes its splines
again rather than read them with their slopes from the weights: 155 to
158 µs so, against 127 to 134; by batches of 32 particles a warp, which
compute the splines once into memory of the block and then read the grid
with 4 threads a particle in rounds, it took 155 µs as well. From the
first points and the fractions of the particles, which the kernel of the
weights stores, with the splines and slopes computed again in f32 and the
points numbered in i32, it takes 90 µs (D102); pmemd.cuda gathers in 79
µs.

| Item | Value |
|---|---|
| Scale | 2⁴⁰: a contribution is resolved to 10⁻¹² e, and a point holds up to 2⁶³ / 2⁴⁰ ≈ 8 × 10⁶ e, far beyond any point of a real system. A charge of 100 e or more is rejected. A position that is not a number converts to an undefined integer; the run has failed by then, but the grid does not say so. |
| Host | The same fixed point, added by one thread in a plain loop (`@mdrt.pme_spread` of `PME.mlir`); the conversion of the grid and the gathering are parallel loops, under OpenMP with more than one thread |
| Mixed precision | The positions, the charges, and the forces as they are stored. On the host the B-splines, the grid, and the FFT are in f64; on a device they are in the type of the forces, f32 in the mixed and single modes, with the fractions of the positions along the cell (the position times one over the edge, less its floor) and the edges of the cell in f64, and the point of the grid and the fraction within it in the type of the forces (D78, amended). The energy and the virial are summed in f64 across blocks (below) |
| The sums of a device | A thread for each point of the transform; a block of 128 threads adds up its energy and virial by a tree (`gpu.all_reduce`) in the type of the grid and stores them in f64, and one block adds up the blocks in f64; the host copies seven numbers (D104). The order of every sum depends on the size of the grid alone, so the energy and the virial do not depend on the order of the threads either |

### 3.1 The particles without a coefficient (D262)

A particle with $q_i = 0$ adds nothing to the grid, and its force,
$-q_i\sum_k\phi(k)\nabla_i\theta_i(k)$, is zero; the energy and the virial
come from the grid alone. The oxygen of a four-site water (OPC,
TIP4P-Ew) carries no charge, and the hydrogens and the virtual site of a
water have $c_i = 0$ on the grid of the dispersion (D162). The reciprocal
sums spread and gather only the particles whose coefficient can be
nonzero, from a list of their places, rather than with a test of each
particle in the kernels: the oxygen is one site in four, so a test would
leave every warp at work.

| Item | Value |
|---|---|
| The list | A field of the particles of `i32`, `pme_listed` for the charges and `ljpme_listed` for the coefficients $c_i$: the places of the listed particles in ascending order, then −1 − the place of each of the others, in ascending order. The builder makes it from the topology (`collectListed`); where the particles take a new order (`emitReorder`: at the start, and at each segment of checkpoints with `spatial_order`), `md_exec.nonzero` makes it again from the permuted field, in place of a permutation. `md.reciprocal ... listed(%l)` takes it |
| `md_exec.nonzero` | On the host two loops, one for the listed particles and one for the others; on a device a thread for each chunk of 256 places that counts its listed particles, one thread that sums the counts of the chunks before each and of all, and a thread for each chunk that writes its entries from there. All in integers, so the list is the same in every mode. It runs where the order changes, not at every step |
| `[free_energy]` | The test is on the charge of the topology, never on $q_i(1 - a_i\lambda)$ (D161): a charge of zero is zero at every $\lambda$, and its share of $\partial U/\partial\lambda$ is zero too. A particle that $\lambda = 1$ decouples stays listed, since $\lambda$ changes during a run |
| Kernels on a device | The listed kernels (`_listed` in `PMEGPU.mlir` and `PMEGPUTriclinic.mlir`, after the others, from `scripts/generate-pme-gpu-template.py`) take an entry of the list where the others take a particle: a thread and a point along z in the spreading, a warp in the bricks, the lanes of a particle in the gathering. The weights of order 4 are those of every particle, at its place, from the kernel without a list. The items past the last listed place spread nothing; in the gathering, a warp whose first entry is past it skips the gathering as a whole, before its shuffles, and the threads past it in the warp that holds the last entry compute for the first particle of their warp and do not store its force |
| The forces of the others | The entries past the last listed place hold −1 − the place of each of the others, and the first thread of each such entry in the gathering stores its force of zero: the gathering writes every force, the listed ones and the zeros, and nothing else does. On the host the gathering stores the zeros first, then the forces of the listed particles |
| `gather_all` | Where more than half the particles are listed, the builder marks the sum `gather_all`, and a device spreads the listed particles and gathers every particle with the kernel without a list, whose force on a particle of charge zero is the product with zero. The gathering reads the grid around each particle; in the order of the particles the oxygen of a four-site water sits beside its virtual site, a warp reads the points of both at once, and the oxygen costs almost nothing, while the gathering over the list reads the points of more waters in a warp: on the 31,808 waters of OPC (RTX 4090, mixed precision) the gathering over the list took 10.9 µs a step against 10.1 over every particle. Where few particles are listed (the grid of the dispersion, the oxygen alone) the list pays: 8.5 µs against 10.1. The host always gathers over the list, where a particle costs what it costs |
| The potential at the particles | $\partial E/\partial q_i$ (D231) is not zero where $q_i$ is: a sum with `potential` takes no list |
| When a program has a list | Where the topology has a coefficient of zero, and no tunable may change it: a tunable charge for the list of the charges, a tunable $\sigma$ or $\epsilon$ (of a type or of a pair) for that of the dispersion. An update of the tunables keeps the text of the program, and a charge moved across zero would change it. A tunable of another kind leaves the list. Without a list the module is that of before, so a system with no such particle (JAC, Factor IX, STMV of the Amber suite) runs the same kernels |
| Determinism | The grid in fixed point is the same sum of integers, and each force the same product: in the deterministic mode, and on the host, the run is the same to the bit as without the list (checked against `main` ae971477 on the dipeptide in OPC, [`pme-skip-zero.test`](../test/Driver/pme-skip-zero.test)) |

## 4. Parameters (D71)

| Keyword | Meaning | Default |
|---|---|---|
| `electrostatics = "PME"` in `[energy]` | Particle mesh Ewald, with the cutoff `cutoff` for the direct sum | |
| `coulomb_modifier` in `[energy]` | `"POTENTIAL_SHIFT"` shifts the direct sum to 0 at the cutoff, as GROMACS does by default; `"NONE"` does not, as sander | `"NONE"` |
| `beta` in `[pme]` | $\beta$, in Å⁻¹ | From `tolerance` |
| `tolerance` in `[pme]` | $\beta$ such that $\mathrm{erfc}(\beta r_c) = \texttt{tolerance}$, by bisection, as both engines find it; less than 1 | 10⁻⁵ |
| `grid` in `[pme]` | The numbers of points of the grid along the three edges, `[nx, ny, nz]`, each 8 or more | From `max_spacing` |
| `max_spacing` in `[pme]` | The largest spacing of the grid, in Å; each number of points is the smallest even product of 2, 3, 5, and 7 that gives no wider spacing in the cell of the file of coordinates. The grid stays as a barostat changes the cell, finer as it shrinks and coarser as it grows, and a run that continues from a checkpoint has the grid it began with | 1.2 |
| `order` in `[pme]` | The order of the B-splines: 4, 6, or 8 | 4 |
| `influence` in `[pme]` | `"SPME"` or `"OPTIMAL"` (Section 1.1) | `"SPME"` |

The table `[pme]` is for `electrostatics = "PME"` only, and a grid needs at
least twice the order of points along each edge.

The tolerance of sander, `dsum_tol`, is $\mathrm{erfc}(\beta r_c) / r_c$ with $r_c$ in Å,
not $\mathrm{erfc}(\beta r_c)$: its default of 10⁻⁵ gives a larger $\beta$ than
`tolerance` of 10⁻⁵ does. A comparison gives $\beta$ itself.

$\beta$, the grid, and the order can each be given, so that a run can take those
of sander (`ew_coeff`, `nfft1` to `nfft3`, `order`) or of GROMACS
(`ewald-rtol`, `fourier-nx` to `-nz`, `pme-order`) for a comparison.

## 5. How the engines differ

| Item | sander (`eedmeth = 1`) | GROMACS (Verlet) | In MDIR |
|---|---|---|---|
| The direct sum at the cutoff | Not shifted | Shifted (`coulomb-modifier = Potential-shift`) | `coulomb_modifier` |
| The terms of the log | `EEL`: all but the pairs three bonds apart | `Coulomb (SR)` and `Coul. recip.`; which of $E_\text{self}$, $E_\text{excl}$, and the shift of excluded pairs goes into which is found by comparison before the terms are compared one by one | The five terms apart, and their sum |
| A net charge | The background, with a warning | The background, with a warning | $E_Q$ |
| The influence function | With the factor $\lambda$ (`opt_infl`) | Without | `influence` |
| The grid | Products of 2, 3, and 5 | Products of 2, 3, 5, and 7 | Products of 2, 3, 5, and 7 when chosen |
| The net force of the reciprocal sum, which is not 0 on a grid | Removed at every step (`netfrc`) | Left; the motion of the center of mass is removed | Left; `center_of_mass_interval` removes the motion |

## 6. Validation

| Test | Compared with | Tolerance |
|---|---|---|
| The three waters of Section 19 of design-m1.md, energy and forces | An Ewald sum in a script: the direct sum over periodic images, many wave vectors, $E_\text{self}$, $E_\text{excl}$ | 10⁻⁶ with a fine grid |
| The virial | $-dU/d\lambda$ under a uniform scaling of the positions and the cell at fixed numbers of points (`test/Driver/Inputs/check_virial.py`) | 10⁻⁵ |
| Dipeptide in OPC | sander with `ew_coeff`, `nfft`, and `order` of MDIR | The ratio of the Coulomb constants |
| A peptide in water | GROMACS with the same $\beta$, grid, and order, with the shift | 10⁻⁶ (mixed precision) |
| Rigid OPC with SETTLE at 2 fs, at constant energy | The change of the total energy against the square of the time step | |
| CPU, OpenMP, GPU, mixed precision | Each other; a GPU in double equals the CPU where spreading is deterministic | |

Results (2026-09-30):

| Test | Result |
|---|---|
| Three waters, 64³ points of order 8 (`test/Driver/pme.test`) | The direct sum, the excluded pairs, and the self term equal those of the Ewald sum to the digits of the log; the reciprocal sum to 6 digits |
| The ff19SB system of D65, against an Ewald sum with numpy | The direct sum, the excluded pairs, and the self term to 10⁻⁹; the reciprocal sum reaches the Ewald sum as the grid grows, as sander's does |
| The virial of each term | Against its analytic value, and the reciprocal sum against $\sum_{\mathbf m} E_\mathbf m (1 - 2\pi^2 m^2/\beta^2)$ of a script: to 10⁻⁵ |
| Dipeptide in OPC against sander, `"OPTIMAL"`, 30 × 30 × 24 points (`test/Driver/amber-pme.test`) | The electrostatic energy agrees to 2 × 10⁻⁵ kcal/mol, the precision of sander's log |
| The same with rigid water and the shift, 0.5 ps at constant energy (`test/Driver/pme-settle.test`, `pme-gpu.test`) | The total energy stays within 0.3 kcal/mol; its largest change is 5.9 × 10⁻⁵ of it at 1 fs and 7.8 × 10⁻⁶ at 0.5 fs. A GPU in double gives the energies of the CPU to the digits of the log, and the same from run to run. |
