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
| FFT | On the host, pocketfft [[Reinecke2019]](references.md#reinecke2019) under the BSD license, in `libmdrt`; on a device, cuFFT of the CUDA toolkit, in `libmdrt_cuda`, with a plan kept for each size of grid, direction, and precision until the process ends, in a table that grows; a plan that cannot be made stops the run, as a failure of the simulation in Python (#323) (D64). |
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
particle, 442 µs). Without tiles, the orders 6 and 8 keep the spreading of
a thread for each particle and point along z; the deterministic mode always
does. A grid in f32 of at least 500,000 points spreads through tiles of
the memory of a block instead, at every order and with the prolate kernel
(Section 3.3).

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
| Host | The same fixed point, added with integer atomics from a parallel loop over the particles (`@mdrt.pme_spread` of `PME.mlir`, D271); every other part is a parallel loop too, under OpenMP with more than one thread (Section 3.2) |
| Mixed precision | The positions, the charges, and the forces as they are stored. On the host and on a device the B-splines, the grid, and its transform are in the type of the forces, f32 in the mixed and single modes (on the host since D271, in f64 before), the arithmetic of pocketfft in f64 on each line, with the fractions of the positions along the cell (the position times one over the edge, less its floor) and the edges of the cell in f64, and on a device the point of the grid and the fraction within it in the type of the forces (D78, amended). The host keeps the point and the fraction in f64 and rounds the fraction once to the type of the forces for the B-splines, so that the mixed modes of the host and a device differ in this step (the maintainer's ruling on PR #366: a place is three operations a particle, against 64 points, so f32 would save nothing there). The energy and the virial are summed in f64 across blocks (below) |
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
| The prolate kernel | `md.reciprocal ... prolate` (Section 7, D266) takes the same list and `gather_all` by the same rule. On the host the listed functions of `PME.mlir` take the table of the op, as the others do, and place and weigh by the window. On a device `PMEGPUProlate.mlir` has `_listed` kernels after the others (from `scripts/generate-pme-prolate-gpu-template.py`): the spreading into bricks and in fixed point and the gathering (orthorhombic and triclinic), a warp an entry of the list where the others take a warp a particle; a warp whose entry is past the last listed place spreads nothing and, in the gathering, stores the force of zero of its particle from its first lane. The places are those of every particle. Checked on examples/ala3 in OPC with the prolate kernel ([`prolate-skip-zero.test`](../test/Driver/prolate-skip-zero.test)) |
| When a program has a list | Where the topology has a coefficient of zero, and no tunable may change it: a tunable charge for the list of the charges, a tunable $\sigma$ or $\epsilon$ (of a type or of a pair) for that of the dispersion. An update of the tunables keeps the text of the program, and a charge moved across zero would change it. A tunable of another kind leaves the list. Without a list the module is that of before, so a system with no such particle (JAC, Factor IX, STMV of the Amber suite) runs the same kernels |
| Determinism | The grid in fixed point is the same sum of integers, and each force the same product: in the deterministic mode, and on the host, the run is the same to the bit as without the list (checked against `main` ae971477 on the dipeptide in OPC, [`pme-skip-zero.test`](../test/Driver/pme-skip-zero.test)) |

### 3.2 The host in parallel (D271)

Every part of the sum on the host is a parallel loop of the compiled code
(`scf.parallel`, lowered by `convert-scf-to-openmp` with the other loops of
the CPU, D171), and the result is the same to the bit for any number of
threads, in every mode:

| Part | On the host |
|---|---|
| Spreading | A particle an iteration (`pme_spread_one`, a function of its own so that its weights are on its own stack); its 64 values (order 4) are added to the i64 grid by `memref.atomic_rmw addi`, a `lock add` on x86. An integer sum does not depend on the order of the threads (D70). The grid is zeroed and converted to the floating-point grid by parallel loops. The first point along each edge is taken within the grid once, so a point wraps around by a subtraction, not a division |
| FFT | pocketfft [[Reinecke2019]](references.md#reinecke2019) in `runtime/mdrt_fft.c`, called for blocks of lines from a parallel loop over each of the three passes: the real lines along $z$ a plane of $x$ a block; the complex lines along $y$ and along $x$ in tiles of 8 neighboring columns of $z$, which are copied to contiguous lines, transformed, and copied back, so that the strided lines are read by cache lines. The plans are made before the loops (`mdrtFFTPrepare`); pocketfft reads a plan and allocates its own scratch, so the threads share them. Each line is transformed by the same code whatever thread takes it, and nothing is summed across lines. The arithmetic of a line is in f64; a grid in f32 is converted where the line is copied |
| Product | A row of the grid ($k_1 k_2$ rows of $k_3/2 + 1$ points) an iteration. The Gaussian influence function of a rectangular cell from the factors of its edges (D104): $m_a$, $m_a^2$, and $\exp(-\pi^2 m_a^2/\beta^2)$ times the factor of the B-splines (along $x$ also $f/(\pi V)$), computed in f64 once an evaluation, $k_1 + k_2 + k_3$ exponentials, and rounded once to the type of the grid. A triclinic cell, the dispersion (D162), and the prolate kernel (D266) compute the influence of each point in f64 (`pme_influence`) and round it once. The energy and the virial of a row are summed in the type of the grid and returned in f64, and the rows in the fixed partition of D171. A step that writes neither calls `pme_scale`, which only multiplies, as on a device |
| Places | In f64 on the host: the fraction along the cell, the point of the grid, and the fraction within it, which is rounded once to the type of the grid for the B-splines. A device takes the point and the fraction within it in the type of the forces (D78), so the mixed modes of the two differ in this step |
| Gathering | A particle an iteration, its sums in the type of the grid, its force from them in f64 |

### 3.3 Spreading through tiles

Outside the deterministic mode a device may spread the charges through
tiles of the memory of a block (D282, issue #336), as cuFINUFFT
spreads [[Shih2021]](references.md#shih2021): the particles are binned by
their first point into bins of $E^3$ points of the grid, and a block adds
the charges of a bin into a tile of $(E + P - 1)^3$ points, the bin and
$P - 1$ more on the high side of each axis, with atomics of the memory of
the block; then it adds the tile to the grid with atomics of the device.
A point of the grid is in at most 8 tiles, so the atomics of the device
fall from $P^3$ a particle to about $(1 + (P - 1)/E)^3$ a point of the
grid: on Cellulose, from 24 million a step into the bricks (its 378,000
charged sites) to at most 11 million (8,192 tiles of 1,331 points)
with bins of 8 points.

| Item | Value |
|---|---|
| The choice | `scatter(tile(E))` on `md_exec.reciprocal`, the strategy `tile` of the privatized scatter-accumulate, which `md-exec-choose-scatter` sets on a program for a device: by default bins of 8 points for every sum in f32 whose grid has at least 500,000 points. Its preconditions (the mode, the edges of the template, a grid at least $E + P - 1$ points along each axis, 48 KiB of block memory), where they are checked, the measurements behind the threshold, and the switch `MDIR_PME_SPREAD` are in [device-primitives.md](device-primitives.md), Section 2 (D[privatize-scatter]). Bins of 16 points do not fit in f64 or at $P \ge 6$ |
| Places | The weights kernel of Section 3, now of every order: the first points and the fractions of every particle, 6 values. The tiles do not depend on the cell, so a triclinic cell takes them too (its own places); so does the sum of the dispersion (D162) |
| Binning | One kernel zeroes the grid of the transform, to which the tiles are added, and the counts of the bins. In a second, a thread an entry (a particle, or an entry of the list of D262) finds its bin from its first points; the lanes of a warp with the same bin (`match.any.sync`) take their ranks from one atomic of the first of them, so that a warp of particles in the order of their positions adds to a count once or a few times; the particle takes the place of its rank in a run of $C$ places of its bin, $C = 2\lceil n/n_\text{bins}\rceil + 8$, or, past $C$, a place in a list of the particles that overflow. No scan of the counts, and no sort. The bins are made at every evaluation: the cells of the neighbor structures are made at each build, with a geometry of their own, and the first point of a particle changes between builds. What the binning takes from the order of the particles (D44, D86) is the locality of the warps, which keeps the atomics of the counts few |
| The tile kernel | A block of 128 threads a bin of 4 or 8 points, 512 of 16, and one more block for the overflow. In rounds of a particle a thread, the block reads the charges, the first points, and the fractions of the bin's particles into the memory of the block; then a warp takes a particle, its lanes $0$ to $3P - 1$ compute the weight of their point and axis (order 4 in closed form, D109; 6 and 8 by the recursion; the prolate window by its series, Section 7), the other lanes take them by shuffles, and each lane adds the product of its point, $z$ fastest, to the tile. The atomics of the memory of the block are a loop of compare-and-swap on the devices measured (`ATOMS.CAST.SPIN`, sm_89), and more warps on one tile wait for each other: on Cellulose the tile kernel took 67.6, 74.5, and 106.6 µs with bins of 8 and 128, 256, and 512 threads. The nonzero points of the tile are then added to the grid; the block of the overflow adds its particles to the grid directly |
| Mixed and double | The tile and the grid in the type of the forces. Equal to the spreading of Section 3 to the rounding of the atomics ([`pme-tiles-gpu.test`](../test/Driver/pme-tiles-gpu.test): one step of $10^{-6}$ ps of the OPC box, 1,570 sites, the forces of the checkpoint and the rows of energies against `global`). The step is that short so that both runs evaluate the forces at the same positions in f32: after it the positions of two runs differ by $4\times10^{-16}$ nm at most. After a step of 0.001 ps they differed by about $10^{-12}$ nm, from the rounding of the forces of step 0, and one coordinate $2\times10^{-12}$ nm from a midpoint between two numbers of f32 rounded to the other one in about 1 run of 400; the bonded forces of its atom then changed by 0.1 kJ/mol/nm, a relative rms of $2.46\times10^{-6}$ (#427). The relative rms of the difference of the forces in mixed precision, over five repeats: order 4 with bins of 4, 8, and 16 points 0.9 to 1.2 × 10⁻⁷, order 6 1.0 to 1.1 × 10⁻⁷, order 8 1.0 to 1.1 × 10⁻⁷, the prolate kernel 0.7 × 10⁻⁷, a rhombic dodecahedron 1.1 to 2.5 × 10⁻⁸, the box made 200 Å long, where many particles overflow the runs of their bins, 0.8 to 1.0 × 10⁻⁷, LJ-PME (D162) 1.0 to 1.1 × 10⁻⁷, where two runs of `global` differ by 0.8 to 1.1 × 10⁻⁷ at order 4 and 0.6 to 0.7 × 10⁻⁷ at order 8; the energies within 1.8 × 10⁻⁶ of the largest of their columns, as two runs of `global` within 9.2 × 10⁻⁷. In double precision 4.9 to 5.2 × 10⁻¹⁶ at order 4 (two runs of `global`: 1.4 to 1.6 × 10⁻¹⁶), and 4.9 × 10⁻¹³ for the prolate kernel, whose spreading without tiles is in fixed point in double (Section 7). compute-sanitizer finds no error or hazard in the binning and the tiles, nor in the spreading of order 8 with and without tiles ([`test/Sanitizer/pme-tiles-gpu.test`](../test/Sanitizer/pme-tiles-gpu.test)) |
| The deterministic mode | Not tiled: its grid is in fixed point (D70, D84), added by a thread for each particle and point along $z$, the same to the bit as before. Integer atomics in a tile would give the same sums, but the products $q M M M$ would have to be rounded to fixed point as that kernel rounds them, which the tiles do not reproduce; the lowering refuses `tile` in that mode |
| The gathering | Not tiled. In the prototype of #334 (P0, Cellulose, RTX 4090) a gathering of order 4 from tiles took 51 µs against 39 from the grid with 8 lanes a particle; at order 8 a tiled gathering with 32 lanes took 193 µs against 358 for 8 lanes from the grid, which is MDIR's present form of orders 6 and 8: the lanes, more than the tiles, are what pays there (issue #411) |

The step of the Amber suite in mixed precision, without tiles and with
bins of 8 points (`MDIR_PME_SPREAD=tiles8`, whatever the default would
choose), alternated in rounds within one job on one node, the first round
dropped (`scripts/benchmarks/pme-tiles/bench.py`), and the spreading,
binning and zeroing included, from Nsight Systems
(`scripts/benchmarks/pme-tiles/trace.py` splits it by kernel). Without
tiles is `main` at order 4 and this branch's `global` otherwise; at
order 4 `global` and `main` took the same time, within 0.6% on the RTX
4090 and within the scatter of the rounds on the RTX PRO 6000.
In each table the rows from the first of JAC on are those of grids below
500,000 points or in f64, which the default does not tile.

*RTX 4090 (ada-002 at order 4, ada-001 otherwise), µs:*

| System | Spreading | Step without | Step with tiles | Ratio | Spreading without | Spreading with tiles |
|---|---|---:|---:|---:|---:|---:|
| Factor IX NVE | order 4 | 341.9 | 328.4 | 0.960 | 41.5 | 27.7 |
| Factor IX NPT | order 4 | 365.6 | 350.7 | 0.959 | 41.5 | 27.7 |
| Cellulose NVE | order 4 | 1523.8 | 1443.2 | 0.947 | 165.0 | 95.1 |
| Cellulose NPT | order 4 | 1575.7 | 1500.5 | 0.952 | 167.8 | 95.0 |
| STMV NPT 4 fs | order 4 | 4967.9 | 4687.2 | 0.943 | 538.5 | 279.3 |
| Factor IX NVE | order 6 | 443.3 | 378.4 | 0.853 | 129.5 | 56.0 |
| Factor IX NVE | order 8 | 555.9 | 468.6 | 0.843 | 230.4 | 134.7 |
| Factor IX NVE | prolate, ε = 10⁻⁵ | 512.7 | 444.0 | 0.866 | 136.7 | 71.7 |
| Cellulose NVE | order 6 | 1928.6 | 1622.6 | 0.841 | 519.4 | 191.6 |
| Cellulose NVE | order 8 | 2432.1 | 1996.5 | 0.821 | 939.8 | 495.8 |
| Cellulose NVE | prolate, ε = 10⁻⁵ | 2263.3 | 1904.1 | 0.841 | 585.4 | 209.2 |
| Cellulose NVE | prolate, ε = 10⁻⁴ | 2184.6 | 1872.2 | 0.857 | 595.8 | 268.7 |
| STMV NPT 4 fs | order 6 | 6298.1 | 5189.9 | 0.824 | 1499.0 | 525.8 |
| STMV NPT 4 fs | order 8 | 7809.2 | 6160.4 | 0.789 | 2713.1 | 1367.4 |
| STMV NPT 4 fs | prolate, ε = 10⁻⁵ | 6485.5 | 5379.9 | 0.830 | 1640.1 | 539.8 |
| STMV NPT 4 fs | prolate, ε = 10⁻⁴ | 6603.5 | 5370.9 | 0.813 | 1773.7 | 506.7 |
| STMV NPT 4 fs | prolate, ε = 10⁻³ | 6684.8 | 5123.9 | 0.767 | 2165.7 | 579.5 |
| JAC NVE | order 4 | 140.0 | 141.0 | 1.007 | 12.0 | 12.2 |
| JAC NPT | order 4 | 153.7 | 154.9 | 1.008 | 12.1 | 12.2 |
| JAC NVE 4 fs | order 4 | 150.6 | 150.9 | 1.002 | 11.9 | 12.6 |
| JAC NPT 4 fs | order 4 | 159.6 | 160.2 | 1.004 | 12.2 | 12.4 |
| JAC NVE | order 6 | 168.2 | 156.1 | 0.928 | 37.4 | 21.6 |
| JAC NVE | order 8 | 202.0 | 184.4 | 0.913 | 63.0 | 46.3 |
| JAC NVE | prolate, ε = 10⁻⁵ | 183.8 | 179.6 | 0.978 | 42.1 | 38.0 |
| JAC NVE | prolate, ε = 10⁻⁴ | 190.0 | 252.7 | 1.330 | 42.9 | 107.7 |
| JAC NVE | prolate, ε = 10⁻³ | 186.4 | 369.7 | 1.984 | 51.1 | 235.3 |
| Factor IX NVE | prolate, ε = 10⁻⁴ | 478.3 | 443.9 | 0.928 | 155.8 | 121.6 |
| Factor IX NVE | prolate, ε = 10⁻³ | 480.1 | 563.6 | 1.174 | 162.2 | 245.3 |
| Cellulose NVE | prolate, ε = 10⁻³ | 2195.7 | 1766.9 | 0.805 | 704.6 | 264.8 |
| JAC NVE, double | order 4 | 2221.1 | 2235.4 | 1.006 | 27.6 | 38.4 |
| Factor IX NVE, double | order 4 | 7731.6 | 7836.8 | 1.014 | 81.5 | 128.2 |
| Cellulose NVE, double | order 4 | 35265.3 | 36000.0 | 1.021 | 307.7 | 501.0 |

*RTX PRO 6000 (ada-004; the step there varied by up to 4% from round to
round, with another job of this measurement on the node, so the
spreading is the firmer number), µs:*

| System | Spreading | Step without | Step with tiles | Ratio | Spreading without | Spreading with tiles |
|---|---|---:|---:|---:|---:|---:|
| Factor IX NVE | order 4 | 247.0 | 245.7 | 0.995 | 22.9 | 22.3 |
| Cellulose NVE | order 4 | 1142.1 | 1122.9 | 0.983 | 96.8 | 82.0 |
| STMV NPT 4 fs | order 4 | 3699.9 | 3587.4 | 0.970 | 289.4 | 220.6 |
| Factor IX NVE | order 6 | 304.9 | 295.7 | 0.970 | 62.4 | 44.1 |
| Factor IX NVE | order 8 | 376.2 | 427.8 | 1.137 | 115.0 | 95.1 |
| Factor IX NVE | prolate, ε = 10⁻⁵ | 369.9 | 345.5 | 0.934 | 61.4 | 49.8 |
| Cellulose NVE | order 6 | 1429.5 | 1384.4 | 0.968 | 262.6 | 167.6 |
| Cellulose NVE | order 8 | 1747.5 | 1670.1 | 0.956 | 480.8 | 359.0 |
| Cellulose NVE | prolate, ε = 10⁻⁵ | 1739.9 | 1651.0 | 0.949 | 313.9 | 185.3 |
| STMV NPT 4 fs | order 6 | 4497.1 | 4369.3 | 0.972 | 753.3 | 465.0 |
| STMV NPT 4 fs | order 8 | 5349.9 | 5075.4 | 0.949 | 1384.6 | 999.5 |
| STMV NPT 4 fs | prolate, ε = 10⁻⁵ | 5035.6 | 4559.8 | 0.905 | 929.0 | 494.4 |
| JAC NVE | order 4 | 122.1 | 128.1 | 1.050 | 6.7 | 10.0 |
| JAC NVE | order 6 | 135.3 | 137.2 | 1.014 | 19.6 | 18.7 |
| JAC NVE | order 8 | 154.5 | 158.9 | 1.028 | 34.1 | 36.5 |
| JAC NVE | prolate, ε = 10⁻⁵ | 147.1 | 162.0 | 1.101 | 21.6 | 35.4 |

Factor IX at order 8 on the RTX PRO 6000: the rounds took 368, 358, and
498 µs with tiles against 399, 371, and 381 without; the third is the
disturbed one. On Cellulose (RTX 4090) the spreading of 95 µs is 16 for
zeroing the grid, 8 for the binning, and 71 for the tiles; on STMV
(RTX 4090) 44, 26, and 209 (the zeroing of a grid of 12 million points).

## 4. Parameters (D71)

| Keyword | Meaning | Default |
|---|---|---|
| `electrostatics = "PME"` in `[energy]` | Particle mesh Ewald, with the cutoff `cutoff` for the direct sum | |
| `coulomb_modifier` in `[energy]` | `"POTENTIAL_SHIFT"` shifts the direct sum to 0 at the cutoff, as GROMACS does by default; `"NONE"` does not, as sander. **At a tolerance looser than the default, take `POTENTIAL_SHIFT` for a run at constant energy or one whose conserved energy is watched** (#346): unshifted, each pair that crosses the cutoff changes the energy by $fq_iq_j\,\mathrm{erfc}(\beta r_c)/r_c$, and the total energy of JAC (NVE, mixed precision, 2 fs, 2 ns) drifts by 281 kcal/mol/ns at `tolerance = 1e-3` (the slopes of the two halves 247 and 288) and 2.9 at `1e-4` (1.7 and 6.0); at the default $10^{-5}$ it is −0.01 (0.13 and 0.86), within the scatter of a slope. The shift changes the energy the log reports, not the forces. `NONE` stays the default, as in Amber, so that energies compare with sander's and pmemd's. The prolate kernel (`kernel = "PROLATE"`) needs no shift: its direct term $1 - \Phi(r/r_c)$ is 0 at the cutoff, and the modifier changes nothing there. | `"NONE"` |
| `kernel` in `[pme]` | `"GAUSSIAN"`: the Gaussian splitting with B-splines; `"PROLATE"`: the prolate spheroidal wave function for both, Ewald summation with prolates (Section 7, D266) | `"GAUSSIAN"` |
| `beta` in `[pme]` | $\beta$, in Å⁻¹; refused with `kernel = "PROLATE"` | From `tolerance` |
| `tolerance` in `[pme]` | The size of the tail of the splitting kernel at the cutoff. Gaussian: $\beta$ such that $\mathrm{erfc}(\beta r_c) = \texttt{tolerance}$, by bisection, as both engines find it. Prolate: the bandwidth $c$ such that $\psi_0^c(1) = \texttt{tolerance}$. **The same number gives very different force errors with the two kernels**: the Gaussian default of 10⁻⁵ gives a relative force error $\Delta = 1.4\times10^{-3}$ (JAC and Cellulose, with its default grid; $\Delta$ over the Coulomb forces of the topology, Section 7), the prolate 10⁻⁵ gives 1.0 to 1.2 × 10⁻⁵, in double precision; mixed precision adds about 10⁻⁴ with either kernel. The Gaussian's tolerance truncates the direct sum alone, as pmemd's `dsum_tol` and GROMACS's `ewald-rtol` do (both 10⁻⁵ by default), and the mesh sets $\Delta$: order 4 at about 1 Å (pmemd, MDIR) or 1.2 Å (GROMACS) gives $\Delta \approx 10^{-3}$ at the standard settings of all three. The prolate tolerance, with its grid rule, sets $\Delta$ itself: its default is about a hundred times the force accuracy of the standard settings. Looser than the default, the Gaussian's unshifted direct sum makes the energy of a run drift (JAC: 281 kcal/mol/ns at 10⁻³, 2.9 at 10⁻⁴): take `coulomb_modifier = "POTENTIAL_SHIFT"` for a run at constant energy (#346). Less than 1 | 10⁻⁵ |
| `bandwidth` in `[pme]` | Prolate only: $c$, instead of the tolerance | From `tolerance` |
| `window_bandwidth` in `[pme]` | Prolate only: $c_1$ of the window of the spreading | $c(\texttt{tolerance}/2)$ |
| `grid` in `[pme]` | The numbers of points of the grid along the three edges, `[nx, ny, nz]`, each 8 or more | From `max_spacing` |
| `max_spacing` in `[pme]` | The largest spacing of the grid, in Å; each number of points is the smallest even product of 2, 3, 5, and 7 that gives no wider spacing in the cell of the file of coordinates. The grid stays as a barostat changes the cell, finer as it shrinks and coarser as it grows, and a run that continues from a checkpoint has the grid it began with | 1.2 |
| `order` in `[pme]` | The order of the B-splines: 4, 6, or 8; prolate: the points $P$ of the window, 5 to 8 | 4; prolate 6 |
| `influence` in `[pme]` | `"SPME"` or `"OPTIMAL"` (Section 1.1); refused with `kernel = "PROLATE"` | `"SPME"` |

With `kernel = "PROLATE"` and neither `grid` nor `max_spacing` given, the
grid has the modes of eq. 15 of [[Liang2026]](references.md#liang2026),
$n_f = \lceil c L / (\pi r_c)\rceil$ along each edge, raised until the aliasing
estimate of its eq. 10, relative to the sum it perturbs, is at most the
tolerance; the log names $c$, $c_1$, $P$, the grid, and the estimate. On JAC
the estimate lies above the relative force error by 1.2 to 3 at every grid
and order measured (issue #334).

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

## 7. The prolate kernel (D266)

`kernel = "PROLATE"` replaces both the Gaussian of the splitting and the
B-splines of the spreading by the prolate spheroidal wave function of order
zero, $\psi_0^c$, the eigenfunction of largest eigenvalue of
$\varphi \mapsto \int_{-1}^{1} e^{icxt}\varphi(t)\,dt$ [[Slepian1961]](references.md#slepian1961):
"Ewald summation with prolates" (ESP) [[Liang2026]](references.md#liang2026).
The white paper derives the terms (Section 5.6); this section says where
they are.

| Part | Where |
|---|---|
| $\psi_0^c$, $\lambda_0$, $C_0 = \int_0^1\psi_0^c$, $\langle x^2\rangle$, the transform | `lib/Driver/Prolate.cpp`: a series of Legendre polynomials of even degree from the eigenvector of the tridiagonal matrix of the prolate operator [[Xiao2001]](references.md#xiao2001), normalized to $\int_{-1}^{1}\psi^2 = 1$; $c$ solves $\psi_0^c(1) = \varepsilon$, which gives Table 2 of the paper at 10⁻³, 10⁻⁴, and 10⁻⁵. Against the eigenfunction of the integral operator by Gauss–Legendre in mpmath at 40 digits to 1–4 × 10⁻¹⁶ (`test/Driver/prolate-function.test`) |
| $c$, $c_1$, $P$, the grid, the self term and the background | `Builder::collectProlate`: $c_1 = c(\varepsilon/2)$ as the paper's code takes it; $P = 6$; the grid by eq. 15, refined at one spacing for the three edges until the aliasing estimate of eq. 10 is at most ε/2 (below), and refused, with what to do, where twice that grid leaves it above; self $-f\,\chi(0)/(2r_c)\sum q^2$; background $-f\pi Q^2 r_c^2\langle x^2\rangle/V$ |
| The table of `md.reciprocal` (`prolate`) | Rows 0 to 2 the factors $1/\hat\phi_1(2\pi k/K)^2$ of each edge; row 3 the weights and slopes of the window $\psi_0^{c_1}(2d/P) - \psi_0^{c_1}(1)$ (below), a Chebyshev series in the fraction of the place for each of the $P$ points, to 10⁻¹⁴ of $\psi(0)$; row 4 the series of $\hat\chi$ and of $k\hat\chi'(k)$ in $2x^2 - 1$ |
| The direct, excluded, and alchemical pairs | $1 - \Phi(x)$ and $\Phi(x)/x$, $x = r/r_c$, as tables of cells of polynomials (`pme_far`, `pme_near`): 64 cells of 10 terms within 10⁻¹⁴ for kernels in f64, 128 cells of 4 terms within 4 × 10⁻¹⁰ for kernels in f32. In the loop over pairs of a kernel in f32 under `fast_math`, the direct sum is instead $(1 - x)\bar\chi(x)/r$, $\bar\chi$ the mean of $\chi$ over $[x, 1]$ as a Chebyshev series in $2x - 1$ (`md.chebyshev`, `prolate::getFarSeries`), which a table of $r^2$ holds as it holds the Gaussian's (D94, D280; below); `MDIR_PROLATE_CELLS=1` keeps the cells there |
| The host | `PME.mlir`: the window by Clenshaw's recurrence where the B-splines were, chosen by the five rows of the table; the influence and its virial in the product |
| A device | `PMEGPUProlate.mlir`, generated by `scripts/generate-pme-prolate-gpu-template.py`: the coefficients converted once a sum into the type of the grid, term by term (a buffer, not constant memory: issue #337, P0); the places; the spreading a warp a particle, the lanes $0$ to $3P-1$ evaluating each weight once and the others taking them by shuffles, into the bricks of D108 or, in the deterministic mode and in f64, in fixed point (D70); the influence of the product as the dispersion's tables are (D104, D162) with $\hat\chi$ in place of $F$; the gathering a warp a particle, every value of the grid read once for the three components. Triclinic cells have their places, gathering, and influence |

LJ-PME (D162) stays Gaussian and can be combined with it. The derivative
in the charges (D231) takes `pme_prolate_potential`.

**The window at the edge of its support.** The paper's window (its eq. 14)
is $\psi_0^{c_1}(2d/P)$, whose value at $\lvert d\rvert = P/2$ is
$\psi_0^{c_1}(1) \approx \varepsilon/2$, not 0. When a particle crosses a
plane of the grid, its point at $d = -P/2$ leaves with that weight, and a
point at $d = +P/2$ enters with it: the charge on the grid and the energy
of the mesh jump, and the forces, the derivative between the crossings,
have no impulse for it. On JAC in NVE (mixed precision, 24³, ε = 10⁻³) the
energy drifted by 75 kcal/mol/ns. MDIR takes the value off,
$\phi_1(d) = \psi_0^{c_1}(2d/P) - \psi_0^{c_1}(1)$, continuous on the line,
and $\hat\phi_1$ (with it the factors $B$) follows from the same
quadrature, $\hat\phi_1(\kappa) = \hat\phi_1^\text{paper}(\kappa) -
2\psi_0^{c_1}(1)\sin(\kappa P/2)/\kappa$ (white paper, Section 5.6). This
is MDIR's own: the paper truncates without it, and we know of no source
that takes the value off in this way. It costs 2% of
$\Delta$ at the default and 12% at ε = 10⁻³ (JAC: 1.02 → 1.04 × 10⁻⁵ at
50³, 6.2 → 6.9 × 10⁻⁴ at 24³; Cellulose: 1.20 → 1.21 × 10⁻⁵ and 8.9 →
9.95 × 10⁻⁴). The aliasing estimate below keeps the paper's window, with
which it was calibrated.

**The grid.** The force error $\Delta$ (eq. 16 of the paper) is measured
over the Coulomb forces of the topology, its exclusions and 1-4 scaling
included: the forces of the system less those of the same system with
every charge zero, against the Gaussian kernel in double at
$\mathrm{erfc}(\beta r_c) = 10^{-13}$, 0.33 Å, and order 8, which agrees
with the prolate kernel at ε = 10⁻⁹ and 8 points (an independent
splitting) to 1.3 × 10⁻⁷
(`scripts/benchmarks/prolate-ewald/grid_sweep.py`; one step on the CPU in
double at explicit grids). Over the Coulomb forces of all pairs without
exclusions, as the paper measures it, the same errors read about 4.5 times
smaller: the bonded pairs inflate the norm. On a fine grid $\Delta$ is
0.6 to 0.9 ε, the floor of the tails of $\chi$ and $\phi_1$; the estimate
of eq. 10 lies 2.5 to 3 times *below* $\Delta$ near that floor, and with
the paper's bound, estimate ≤ ε, the grids of $P$ = 5 and 7 fell at 2.7 to
2.9 times the floor (ε = 10⁻⁴, 36³; ε = 10⁻⁵, 36³) while $P = 6$ fell at
1.3 to 1.5. The grid is therefore the smallest at one spacing whose
estimate is at most ε/2 (ruled on PR #337). The automatic grids and their
$\Delta$, CPU in double, with the window of MDIR:

| ε | $P$ | JAC grid | $\Delta$ | Cellulose grid | $\Delta$ |
|---|---|---|---|---|---|
| 10⁻³ | 5 | 24³ | 6.9 × 10⁻⁴ | 100×48×48 | 9.95 × 10⁻⁴ |
| 10⁻³ | 6 | 24³ | 6.4 × 10⁻⁴ | 100×48×48 | 6.7 × 10⁻⁴ |
| 10⁻⁴ | 5 | 42³ | 9.5 × 10⁻⁵ | 180×84×84 | 9.5 × 10⁻⁵ |
| 10⁻⁴ | 6 | 32³ | 7.8 × 10⁻⁵ | 128×64×64 | 8.5 × 10⁻⁵ |
| 10⁻⁴ | 7 | 30³ | 7.2 × 10⁻⁵ | 126×60×60 | 7.6 × 10⁻⁵ |
| **10⁻⁵** | **6** | **50³** | **1.04 × 10⁻⁵** | **210×98×96** | **1.21 × 10⁻⁵** |
| 10⁻⁵ | 7 | 40³ | 7.8 × 10⁻⁶ | 160×80×80 | 1.0 × 10⁻⁵ |
| 10⁻⁵ | 8 | 36³ | 7.8 × 10⁻⁶ | 150×72×72 | 8.2 × 10⁻⁶ |
| 10⁻⁶ | 7 | 56³ | 1.5 × 10⁻⁶ | 240×112×112 | 1.5 × 10⁻⁶ |
| 10⁻⁶ | 8 | 48³ | 9.0 × 10⁻⁷ | 192×90×90 | 1.7 × 10⁻⁶ |

From 10⁻³ to 10⁻⁵ every case lands within 1.3 times the floor of its ε
and $P$ (measured with the paper's window, which differs from MDIR's by
less than 2% of $\Delta$ at 10⁻⁴ and 10⁻⁵), so that
$\Delta \le 1.21\,\varepsilon$; at 10⁻⁶, within 2 times
($\Delta \le 1.7\,\varepsilon$), the case where Cellulose differs from
JAC. The rule of the paper gave 48³ on JAC and
192×98×98 on Cellulose at the default ($\Delta$ = 1.3 and 1.5 × 10⁻⁵).
Where twice the grid of eq. 15 still leaves the estimate above ε/2, the
window is too short for the tolerance, and the automatic grid is refused
with the estimate there and what to do: more points of the window, a
looser `tolerance`, or `grid` (with 8 points at ε = 10⁻⁹ the estimate
falls only to 6.4 × 10⁻¹⁰ on 500³). STMV takes 180³ at the default.

**Validation.**
- Three OPC waters against the Ewald sum over the wave vectors
  (`test/Driver/prolate.test`, `prolate-gpu.test`): the Coulomb terms sum
  to within 1.6 × 10⁻⁶ of it at the default, and to the printed precision
  at ε = 10⁻⁷ with 8 points; the trace of the virial against
  $-dU/d\lambda$ to 2.2 × 10⁻⁶ and each diagonal component against
  $-dU/d\varepsilon_{aa}$ to 6.6 × 10⁻⁶; a GPU in double gives the terms
  of the CPU to the digits of the log, and the deterministic mode the same
  bits in two runs.
- The dodecahedron of `triclinic.test` against the Gaussian kernel at a
  tight setting: 1.0 × 10⁻⁸ at the default, 1.3 × 10⁻⁹ at ε = 10⁻⁹ with
  8 points on 64³ (`prolate-triclinic.test`).
- The off-diagonal virial: the nine components of
  $\partial U/\partial\boldsymbol\varepsilon$ of a frame (D252) in a sheared
  rhombic dodecahedron against central differences of the energy, the
  deformed cell rotated back below the diagonal: 1.5 × 10⁻⁸ relative in
  double, 7.8 × 10⁻⁶ in mixed precision, on the CPU and a GPU
  (`python-frame-gradients-prolate.test`, `-gpu`).
- [free_energy]: $\partial H/\partial\lambda$ of the Coulomb, the
  Lennard-Jones, and a restraint of ethanol decoupled from TIP3P against
  central differences of the energies of the states beside it, 2.9 × 10⁻⁵
  kcal/mol in double and 4.9 × 10⁻⁵ in mixed precision; λ = 0 equal to the
  run without [free_energy] (`free-energy-prolate.test`, `-gpu`).
- The derivative in the charges (D231): 25 entries on the dipeptide with a
  net charge and 11 in a triclinic cell against central differences of the
  energy, 7.6 × 10⁻¹¹ relative in double and 3.7 × 10⁻⁶ in mixed precision
  (`python-tunable-gradient-prolate.test`, `-gpu`).
- JAC, $\Delta$ as above: 1.04 × 10⁻⁵ at the default on the CPU and on a
  GPU in double. Mixed precision adds about 1.0 × 10⁻⁴ with either kernel:
  1.02 × 10⁻⁴ for the prolate default and for the Gaussian of orders 6 and
  8 at 1.1 × 10⁻⁵ alike; at the Gaussian default it is lost in 1.4 × 10⁻³.
  Below 10⁻⁴ the accuracy of a run in mixed precision is that of the
  precision, not of the kernel.
- JAC dynamics, mixed precision, 2 fs, the slope of a straight line fitted
  to the total energy over 2 ns of NVE, with the slopes of its two halves
  (`scripts/benchmarks/prolate-ewald/dynamics.py`):

  | Run | Drift (kcal/mol/ns) | Halves |
  |---|---|---|
  | Gaussian, default (10⁻⁵) | −0.01 | 0.13, 0.86 |
  | Prolate, default (10⁻⁵, 50³) | −0.06 | 0.87, 0.29 |
  | Gaussian, 10⁻⁴ | 2.9 | 1.7, 6.0 |
  | Prolate, 10⁻⁴ (32³) | 2.9 | 0.06, 3.8 |
  | Gaussian, 10⁻³ (a loose direct-sum tolerance, not a standard setting) | 281 | 247, 288 |
  | Prolate, 10⁻³ (24³) | 5.1 | 2.7, 6.3 |
  | Prolate, 10⁻³, the paper's window (1 ns) | 75 | 84, 78 |

  At the default the two are alike, within the scatter of a slope; at
  looser tolerances the Gaussian's unshifted direct sum dominates (#346).
  NPT at 1 bar over 1 ns, the mean volume of the second half with the
  standard error of 10 blocks: 234,301 ± 113 Å³ (prolate default) against
  234,470 ± 101 (Gaussian default), 1.1 combined standard errors apart, and
  234,422 ± 114 at 10⁻³.

**The whole step at the same $\Delta$.** MDIR's step in mixed precision
(JAC NVE 2 fs, Cellulose NVE 2 fs, STMV NPT 4 fs; the settings of the
Amber suite but `[pme]`), ms/step, the mean of the rounds alternated in
one job after the first (5 on JAC and Cellulose; 4 or 5 on STMV, whose
jobs reached their time limit; the spread of the rounds is 0.1 to 0.5%), each column on one device kind. $\Delta$ is that
of the kernel in double at the grid that ran; on STMV it is not measured
(its grids follow by the spacing of JAC). The binary is that of b449357f (5506216c after the rebase on `main`), the head of the branch before
the grid rule and the window were changed, with the grids given by their
spacing (the window's change costs no time). "Points" is the size of the grid, the data that the transposes of a
distributed transform move.

| System | Tier (measured Δ) | Kernel, tolerance | Grid | Points | Δ | RTX 4090 | RTX PRO 6000 |
|---|---|---|---|---|---|---|---|
| JAC | ≈ 10⁻³, standard | Gaussian default, erfc 10⁻⁵ | 54³ | 157,464 | 1.4 × 10⁻³ | 0.132 | 0.114 |
| JAC | ≈ 10⁻³, standard | Gaussian, the suite's (pmemd's) | 64³ | 262,144 | 7.3 × 10⁻⁴ | 0.133 | 0.114 |
| JAC | ≈ 10⁻³, standard | Prolate 10⁻³, P = 5 | 28³ | 21,952 | 7.4 × 10⁻⁴ | 0.160 | 0.144 |
| JAC | ≈ 10⁻⁴ | Gaussian order 4, erfc 10⁻⁵ | 98³ | 941,192 | 1.1 × 10⁻⁴ | 0.175 | 0.143 |
| JAC | ≈ 10⁻⁴ | Gaussian order 6, erfc 10⁻⁶ | 70³ | 343,000 | 6.7 × 10⁻⁵ | 0.166 | 0.133 |
| JAC | ≈ 10⁻⁴ | Prolate 10⁻⁴, P = 6 | 32³ | 32,768 | 7.8 × 10⁻⁵ | 0.189 | 0.161 |
| JAC | ≈ 10⁻⁵ | Gaussian order 6, erfc 10⁻⁶ | 96³ | 884,736 | 8.5 × 10⁻⁶ | 0.182 | 0.147 |
| JAC | ≈ 10⁻⁵ | Gaussian order 8, erfc 10⁻⁶ | 70³ | 343,000 | 1.1 × 10⁻⁵ | 0.197 | 0.152 |
| JAC | ≈ 10⁻⁵ | **Prolate default, P = 6** | 50³ | 125,000 | 1.04 × 10⁻⁵ | 0.177 | 0.152 |
| JAC | ≈ 10⁻⁵ | Prolate 10⁻⁵, P = 7 | 42³ | 74,088 | 7.2 × 10⁻⁶ | 0.200 | 0.163 |
| JAC | ≈ 10⁻⁵ | Prolate 10⁻⁵, P = 8 | 40³ | 64,000 | 7.1 × 10⁻⁶ | 0.225 | 0.182 |
| Cellulose | ≈ 10⁻³, standard | Gaussian default, erfc 10⁻⁵ | 224×108×108 | 2,612,736 | 1.5 × 10⁻³ | 1.449 | 1.105 |
| Cellulose | ≈ 10⁻³, standard | Gaussian, the suite's (pmemd's) | 256×128×128 | 4,194,304 | 7.9 × 10⁻⁴ | 1.478 | 1.107 |
| Cellulose | ≈ 10⁻³, standard | Prolate 10⁻³, P = 5 | 100×50×48 | 240,000 | 9.7 × 10⁻⁴ | 1.939 | 1.558 |
| Cellulose | ≈ 10⁻⁴ | Gaussian order 4, erfc 10⁻⁵ | 400×196×192 | 15,052,800 | 1.2 × 10⁻⁴ | 2.258 | 1.475 |
| Cellulose | ≈ 10⁻⁴ | Gaussian order 6, erfc 10⁻⁶ | 294×144×140 | 5,927,040 | 6.1 × 10⁻⁵ | 2.072 | 1.495 |
| Cellulose | ≈ 10⁻⁴ | Prolate 10⁻⁴, P = 6 | 126×64×60 | 483,840 | 9.0 × 10⁻⁵ | 2.193 | 1.688 |
| Cellulose | ≈ 10⁻⁵ | Gaussian order 6, erfc 10⁻⁶ | 378×192×180 | 13,063,680 | 1.05 × 10⁻⁵ | 2.282 | 1.651 |
| Cellulose | ≈ 10⁻⁵ | Gaussian order 8, erfc 10⁻⁶ | 294×144×140 | 5,927,040 | 1.0 × 10⁻⁵ | 2.621 | 1.817 |
| Cellulose | ≈ 10⁻⁵ | **Prolate 10⁻⁵, P = 6** | 210×108×100 | 2,268,000 | 1.06 × 10⁻⁵ | 2.155 | 1.765 |
| Cellulose | ≈ 10⁻⁵ | Prolate 10⁻⁵, P = 7 | 168×84×80 | 1,128,960 | 8.0 × 10⁻⁶ | 2.695 | 2.010 |
| Cellulose | ≈ 10⁻⁵ | Prolate 10⁻⁵, P = 8 | 150×80×72 | 864,000 | 8.0 × 10⁻⁶ | 2.914 | 2.141 |
| STMV | ≈ 10⁻³, standard | Gaussian default, erfc 10⁻⁵ | 192³ | 7,077,888 | — | 4.489 | 3.432 |
| STMV | ≈ 10⁻³, standard | Gaussian, the suite's (pmemd's) | 224×224×240 | 12,042,240 | — | 4.836 | 3.656 |
| STMV | ≈ 10⁻³, standard | Prolate 10⁻³, P = 5 | 90³ | 729,000 | — | 5.714 | 4.644 |
| STMV | ≈ 10⁻⁴ | Gaussian order 6, erfc 10⁻⁶ | 250×252×256 | 16,128,000 | — | 6.191 | 4.437 |
| STMV | ≈ 10⁻⁴ | Prolate 10⁻⁴, P = 6 | 120³ | 1,728,000 | — | 6.424 | 5.060 |
| STMV | ≈ 10⁻⁵ | Gaussian order 6, erfc 10⁻⁶ | 320×324×336 | 34,836,480 | — | 8.062 | 5.572 |
| STMV | ≈ 10⁻⁵ | Gaussian order 8, erfc 10⁻⁶ | 250×252×256 | 16,128,000 | — | 7.572 | 5.278 |
| STMV | ≈ 10⁻⁵ | **Prolate default, P = 6** | 180³ | 5,832,000 | — | 6.439 | 5.279 |
| STMV | ≈ 10⁻⁵ | Prolate 10⁻⁵, P = 7 | 144×144×150 | 3,110,400 | — | 7.592 | 5.689 |
| STMV | ≈ 10⁻⁵ | Prolate 10⁻⁵, P = 8 | 128×140×140 | 2,508,800 | — | 8.267 | 5.973 |

**The grid points at the same $\Delta$.** The prolate kernel has 7.1
(JAC), 5.8 (Cellulose; 6.6 at its automatic 210×98×96), and 6.0 (STMV)
times fewer points than the Gaussian of order 6 at $\Delta \approx 10^{-5}$,
and 2.6 to 2.8 times fewer than order 8; at $10^{-4}$, 10.5, 12.2, and 9.3
times fewer than order 6; at the accuracy of the default, 7 to 11 times
fewer than the Gaussian default. In a distributed run the transposes of
the 3D transform move data in proportion to the points, which is where the
kernel was meant to pay [[Liang2026]](references.md#liang2026). The
spreading reaches $P - 1$ layers of the grid beyond a domain on each side:
5 for $P = 6$, as many as the B-splines of order 6, two more than order 4.

**Reading.** The standard production settings of pmemd, GROMACS, and
MDIR all lie in the tier $\Delta \approx 10^{-3}$, where the prolate
kernel's step is 21 to 41% slower on one device than the Gaussian's; it is
1.04 to 1.2 times
slower at $10^{-4}$ (against order 6), and from 15% faster (STMV, RTX 4090)
to 7% slower (Cellulose, RTX PRO 6000) at $10^{-5}$ against the faster of
orders 6 and 8. $P$ = 7 and 8, on their smaller grids, are slower than
$P = 6$ everywhere, which stays the default. The order-4 path is unchanged:
the Gaussian default and the suite's settings take the time of `main`
within 1% on every system and device. The loop over pairs takes 44.2 µs on
JAC (RTX 4090) against 42.4 µs with the Gaussian, since its tables read 4
coefficients a pair in f32. These timings are without tiles. Tiled
spreading (Section 3.3, D282) serves both kernels: on a grid in
f32 of at least 500,000 points the prolate default takes 0.83 to 0.87
times the step without tiles on Factor IX, Cellulose, and STMV (RTX
4090), and the Gaussian of orders 6 and 8 0.79 to 0.85; the comparison of
the kernels is still to be made again with the tiles.

**On the CPU target (issue #371).** With PME on the host in parallel
(D271), the comparison was repeated on the CPU target. The runs are
`mdir run` in mixed precision, inside Slurm jobs of 16 cores (AMD EPYC 7543
or 9354; the CPU of each job is in the third table). The cutoff is 9 Å for
the Coulomb and Lennard-Jones sums of every system, so JAC differs from the
GPU table above and from D271, which took the suite's 8 Å. The systems are
JAC (23,558 sites), boxes of 34,386 and 103,292 TIP3P waters (103,158 and
309,876 sites, the states of D267 equilibrated at 1 bar), and Cellulose
(408,609 sites). $\Delta$ is measured anew at 9 Å as above (one step on the
CPU in double against the Gaussian at erfc $10^{-13}$, 0.33 Å, order 8),
since both kernels' grids follow the cutoff.

- **Tiers.** The prolate kernel at its automatic grid for ε = 10⁻³, 10⁻⁴,
  and 10⁻⁵, with $P$ = 5, 6, and 7 ($P = 5$ is refused at 10⁻⁵). For each
  Gaussian order 4, 6, and 8, the grid of fewest points, over
  $\mathrm{erfc}(\beta r_c)$ from 10⁻⁴ to 10⁻⁷, whose $\Delta$ is at most
  1.1 times that of the prolate default at that ε. Order 4 so matched is
  the paper's kind of baseline, its grid raised to the accuracy; the
  fastest order at 16 threads is the fair one. At 10⁻³ it is order 4 on
  every system (erfc 10⁻⁴), at 10⁻⁴ and 10⁻⁵ order 6 but on JAC at 10⁻⁴,
  where orders 4 and 6 take the same time. At 9 Å the Gaussian default
  (erfc 10⁻⁵, 1.2 Å) has $\Delta$ = 6.7 to 7.5 × 10⁻⁴, and erfc 10⁻⁵ alone
  leaves a floor of 4.4 × 10⁻⁵ on JAC.
- **Mixed precision** adds the same error to every case of a system:
  $\Delta$ in mixed precision is 1.0 × 10⁻⁴ (JAC), 1.6 and 2.1 × 10⁻⁴ (the
  waters), and 2.9 × 10⁻⁴ (Cellulose) at 10⁻⁵, about that at 10⁻⁴, and
  within 12% of the double at 10⁻³. As above, the tiers are labeled by the
  kernel's $\Delta$ in double.
- **Timing.** One job per system and tier, from a frozen copy of the
  program, alternates the runs in 6 rounds and drops the first; each run
  takes about 20 s of steps (at least 50). Every case runs at 16 threads,
  and the prolate default and the three Gaussian orders also at 4 and at
  1 thread. Everything but `[pme]` is the control of
  `scripts/benchmarks/cpu-breakdown/bench.py` (D267). The standard
  deviation of the rounds is mostly under 1% of the mean, and up to 7% in
  the jobs on the login node, which other jobs shared.
- **Parts** come from one further run of each case under the sampler of
  D267: the share of the wall clock of the main thread, times the mean
  step. "Rest" is the builds of the list, the bonded terms, the
  constraints, and the integration. Two sampled runs hung in the sampler
  and were repeated (#384, PR #385).

The scripts are `cpu_cases.py`, `cpu_job.py`, and `cpu_table.py --doc` in
`scripts/benchmarks/prolate-ewald/`. The whole step, ms (the fastest
prolate $P$ where it is not 6, and at 10⁻³ MDIR's default):

| System | ε | Kernel | Grid | Points | Δ | 16 threads | 4 threads | 1 thread |
|---|---|---|---|---|---|---|---|---|
| JAC | 10⁻³ | Prolate, P = 6 | 24×24×24 | 13,824 | 5.6 × 10⁻⁴ | 21.4 | 72.2 | 274.0 |
| JAC | 10⁻³ | Prolate, P = 5 (fastest) | 24×24×24 | 13,824 | 6.0 × 10⁻⁴ | 20.2 | — | — |
| JAC | 10⁻³ | Gaussian, order 4 | 50×50×50 | 125,000 | 6.2 × 10⁻⁴ | 12.5 | 43.1 | 159.2 |
| JAC | 10⁻³ | Gaussian, MDIR's default | 54×54×54 | 157,464 | 7.3 × 10⁻⁴ | 12.7 | — | — |
| JAC | 10⁻⁴ | Prolate, P = 6 | 28×28×28 | 21,952 | 7.1 × 10⁻⁵ | 21.1 | 72.2 | 275.3 |
| JAC | 10⁻⁴ | Prolate, P = 5 (fastest) | 40×40×40 | 64,000 | 6.9 × 10⁻⁵ | 20.0 | — | — |
| JAC | 10⁻⁴ | Gaussian, order 4 | 96×96×96 | 884,736 | 7.2 × 10⁻⁵ | 13.9 | 46.3 | 171.3 |
| JAC | 10⁻⁵ | Prolate, P = 6 | 48×48×48 | 110,592 | 8.0 × 10⁻⁶ | 21.0 | 73.7 | 283.8 |
| JAC | 10⁻⁵ | Gaussian, order 4 | 210×210×210 | 9,261,000 | 8.0 × 10⁻⁶ | 31.8 | 98.2 | 381.1 |
| JAC | 10⁻⁵ | Gaussian, order 6 (fastest) | 90×90×90 | 729,000 | 6.0 × 10⁻⁶ | 15.2 | 49.4 | 180.3 |
| TIP3P 34k | 10⁻³ | Prolate, P = 6 | 36×36×36 | 46,656 | 5.5 × 10⁻⁴ | 75.0 | 294.9 | 1127.1 |
| TIP3P 34k | 10⁻³ | Prolate, P = 5 (fastest) | 36×36×36 | 46,656 | 5.9 × 10⁻⁴ | 71.1 | — | — |
| TIP3P 34k | 10⁻³ | Gaussian, order 4 | 84×84×84 | 592,704 | 5.7 × 10⁻⁴ | 43.8 | 168.8 | 696.4 |
| TIP3P 34k | 10⁻³ | Gaussian, MDIR's default | 90×90×90 | 729,000 | 6.7 × 10⁻⁴ | 44.6 | — | — |
| TIP3P 34k | 10⁻⁴ | Prolate, P = 6 | 48×48×48 | 110,592 | 6.7 × 10⁻⁵ | 88.6 | 332.4 | 1262.8 |
| TIP3P 34k | 10⁻⁴ | Prolate, P = 5 (fastest) | 64×64×64 | 262,144 | 7.2 × 10⁻⁵ | 85.1 | — | — |
| TIP3P 34k | 10⁻⁴ | Gaussian, order 4 | 160×160×160 | 4,096,000 | 6.7 × 10⁻⁵ | 62.8 | 212.0 | 819.1 |
| TIP3P 34k | 10⁻⁴ | Gaussian, order 6 (fastest) | 90×90×90 | 729,000 | 6.3 × 10⁻⁵ | 55.9 | 207.8 | 794.5 |
| TIP3P 34k | 10⁻⁵ | Prolate, P = 6 | 80×80×80 | 512,000 | 8.0 × 10⁻⁶ | 75.9 | 303.8 | 1164.3 |
| TIP3P 34k | 10⁻⁵ | Gaussian, order 4 | 350×350×350 | 42,875,000 | 7.4 × 10⁻⁶ | 133.8 | 421.6 | 1538.7 |
| TIP3P 34k | 10⁻⁵ | Gaussian, order 6 (fastest) | 140×140×140 | 2,744,000 | 6.7 × 10⁻⁶ | 52.8 | 202.6 | 776.8 |
| TIP3P 103k | 10⁻³ | Prolate, P = 6 | 54×54×54 | 157,464 | 5.6 × 10⁻⁴ | 222.5 | 856.8 | 3300.6 |
| TIP3P 103k | 10⁻³ | Prolate, P = 5 (fastest) | 54×54×54 | 157,464 | 6.1 × 10⁻⁴ | 213.2 | — | — |
| TIP3P 103k | 10⁻³ | Gaussian, order 4 | 120×120×120 | 1,728,000 | 5.8 × 10⁻⁴ | 133.3 | 505.9 | 1994.0 |
| TIP3P 103k | 10⁻³ | Gaussian, MDIR's default | 126×126×126 | 2,000,376 | 7.0 × 10⁻⁴ | 136.2 | — | — |
| TIP3P 103k | 10⁻⁴ | Prolate, P = 6 | 70×70×70 | 343,000 | 6.8 × 10⁻⁵ | 224.3 | 872.2 | 3348.9 |
| TIP3P 103k | 10⁻⁴ | Prolate, P = 5 (fastest) | 90×90×90 | 729,000 | 7.7 × 10⁻⁵ | 218.3 | — | — |
| TIP3P 103k | 10⁻⁴ | Gaussian, order 4 | 240×240×240 | 13,824,000 | 6.3 × 10⁻⁵ | 158.3 | 587.0 | 2243.3 |
| TIP3P 103k | 10⁻⁴ | Gaussian, order 6 (fastest) | 126×126×126 | 2,000,376 | 7.1 × 10⁻⁵ | 149.0 | 557.9 | 2137.9 |
| TIP3P 103k | 10⁻⁵ | Prolate, P = 6 | 108×108×108 | 1,259,712 | 8.5 × 10⁻⁶ | 257.3 | 948.4 | 3701.4 |
| TIP3P 103k | 10⁻⁵ | Gaussian, order 4 | 480×480×480 | 110,592,000 | 8.2 × 10⁻⁶ | 387.6 | 1155.6 | 4345.6 |
| TIP3P 103k | 10⁻⁵ | Gaussian, order 6 (fastest) | 192×192×192 | 7,077,888 | 8.0 × 10⁻⁶ | 175.6 | 620.2 | 2328.7 |
| Cellulose | 10⁻³ | Prolate, P = 6 | 90×48×42 | 181,440 | 5.8 × 10⁻⁴ | 394.7 | 1387.1 | 5017.7 |
| Cellulose | 10⁻³ | Prolate, P = 5 (fastest) | 90×48×42 | 181,440 | 6.6 × 10⁻⁴ | 388.7 | — | — |
| Cellulose | 10⁻³ | Gaussian, order 4 | 210×100×100 | 2,100,000 | 6.1 × 10⁻⁴ | 244.4 | 848.0 | 3080.7 |
| Cellulose | 10⁻³ | Gaussian, MDIR's default | 224×108×108 | 2,612,736 | 7.5 × 10⁻⁴ | 250.1 | — | — |
| Cellulose | 10⁻⁴ | Prolate, P = 6 | 120×56×56 | 376,320 | 7.1 × 10⁻⁵ | 397.7 | 1407.7 | 5142.5 |
| Cellulose | 10⁻⁴ | Prolate, P = 5 (fastest) | 150×72×72 | 777,600 | 9.6 × 10⁻⁵ | 389.4 | — | — |
| Cellulose | 10⁻⁴ | Gaussian, order 4 | 384×192×192 | 14,155,776 | 7.7 × 10⁻⁵ | 296.0 | 962.4 | 3409.3 |
| Cellulose | 10⁻⁴ | Gaussian, order 6 (fastest) | 224×108×108 | 2,612,736 | 7.0 × 10⁻⁵ | 260.6 | 923.4 | 3294.2 |
| Cellulose | 10⁻⁵ | Prolate, P = 6 | 180×90×90 | 1,458,000 | 1.0 × 10⁻⁵ | 360.5 | 1404.4 | 5545.2 |
| Cellulose | 10⁻⁵ | Prolate, P = 7 (fastest) | 140×70×70 | 686,000 | 7.8 × 10⁻⁶ | 359.1 | — | — |
| Cellulose | 10⁻⁵ | Gaussian, order 4 | 840×392×392 | 129,077,760 | 9.8 × 10⁻⁶ | 513.5 | 2261.1 | 5719.2 |
| Cellulose | 10⁻⁵ | Gaussian, order 6 (fastest) | 336×160×160 | 8,601,600 | 9.0 × 10⁻⁶ | 304.4 | 921.4 | 3196.7 |

Parts at 16 threads, ms a step:

| System | ε | Kernel | Spreading | Window or B-splines | FFT | Product | Gathering | Pairs | Rest | Step |
|---|---|---|---|---|---|---|---|---|---|---|
| JAC | 10⁻³ | Prolate, P = 6 | 2.54 | 1.26 | 0.05 | 0.03 | 0.40 | 13.53 | 3.56 | 21.37 |
| JAC | 10⁻³ | Prolate, P = 5 | 1.51 | 1.18 | 0.05 | 0.03 | 0.26 | 13.56 | 3.61 | 20.19 |
| JAC | 10⁻³ | Gaussian, order 4 | 0.66 | 0.11 | 0.17 | 0.02 | 0.19 | 7.82 | 3.57 | 12.52 |
| JAC | 10⁻³ | Gaussian, MDIR's default | 0.65 | 0.11 | 0.18 | 0.03 | 0.21 | 7.88 | 3.60 | 12.66 |
| JAC | 10⁻⁴ | Prolate, P = 6 | 2.24 | 1.36 | 0.06 | 0.04 | 0.43 | 13.49 | 3.52 | 21.14 |
| JAC | 10⁻⁴ | Prolate, P = 5 | 1.08 | 1.28 | 0.10 | 0.14 | 0.34 | 13.45 | 3.58 | 19.97 |
| JAC | 10⁻⁴ | Gaussian, order 4 | 1.03 | 0.08 | 0.84 | 0.11 | 0.34 | 7.81 | 3.73 | 13.95 |
| JAC | 10⁻⁵ | Prolate, P = 6 | 1.71 | 1.47 | 0.14 | 0.23 | 0.46 | 13.48 | 3.54 | 21.03 |
| JAC | 10⁻⁵ | Gaussian, order 4 | 6.82 | 0.09 | 11.46 | 1.10 | 0.95 | 7.69 | 3.69 | 31.80 |
| JAC | 10⁻⁵ | Gaussian, order 6 | 2.03 | 0.20 | 0.81 | 0.08 | 0.67 | 7.83 | 3.61 | 15.23 |
| TIP3P 34k | 10⁻³ | Prolate, P = 6 | 5.38 | 5.04 | 0.10 | 0.05 | 1.48 | 51.81 | 11.19 | 75.05 |
| TIP3P 34k | 10⁻³ | Prolate, P = 5 | 2.96 | 4.64 | 0.08 | 0.08 | 0.91 | 51.80 | 10.59 | 71.07 |
| TIP3P 34k | 10⁻³ | Gaussian, order 4 | 1.64 | 0.45 | 0.66 | 0.05 | 0.78 | 29.35 | 10.86 | 43.79 |
| TIP3P 34k | 10⁻³ | Gaussian, MDIR's default | 1.78 | 0.40 | 0.80 | 0.04 | 0.87 | 29.52 | 11.17 | 44.58 |
| TIP3P 34k | 10⁻⁴ | Prolate, P = 6 | 4.17 | 6.23 | 0.15 | 0.25 | 1.88 | 63.99 | 11.89 | 88.56 |
| TIP3P 34k | 10⁻⁴ | Prolate, P = 5 | 3.29 | 5.72 | 0.29 | 0.52 | 1.17 | 62.32 | 11.77 | 85.07 |
| TIP3P 34k | 10⁻⁴ | Gaussian, order 4 | 5.86 | 0.44 | 4.52 | 0.60 | 1.51 | 37.10 | 12.73 | 62.75 |
| TIP3P 34k | 10⁻⁴ | Gaussian, order 6 | 5.06 | 0.94 | 0.87 | 0.12 | 2.12 | 34.65 | 12.15 | 55.92 |
| TIP3P 34k | 10⁻⁵ | Prolate, P = 6 | 3.71 | 6.14 | 0.50 | 0.79 | 1.53 | 52.01 | 11.19 | 75.87 |
| TIP3P 34k | 10⁻⁵ | Gaussian, order 4 | 27.03 | 0.41 | 55.18 | 4.96 | 4.25 | 30.70 | 11.30 | 133.83 |
| TIP3P 34k | 10⁻⁵ | Gaussian, order 6 | 5.69 | 0.79 | 2.95 | 0.14 | 2.19 | 29.85 | 11.16 | 52.77 |
| TIP3P 103k | 10⁻³ | Prolate, P = 6 | 13.68 | 14.75 | 0.23 | 0.24 | 4.76 | 157.39 | 31.50 | 222.55 |
| TIP3P 103k | 10⁻³ | Prolate, P = 5 | 7.41 | 13.13 | 0.26 | 0.26 | 3.19 | 157.39 | 31.60 | 213.24 |
| TIP3P 103k | 10⁻³ | Gaussian, order 4 | 4.54 | 1.14 | 1.78 | 0.16 | 2.37 | 88.45 | 34.85 | 133.30 |
| TIP3P 103k | 10⁻³ | Gaussian, MDIR's default | 5.52 | 1.22 | 2.12 | 0.16 | 2.34 | 89.03 | 35.86 | 136.25 |
| TIP3P 103k | 10⁻⁴ | Prolate, P = 6 | 10.64 | 16.12 | 0.55 | 0.51 | 4.59 | 159.70 | 32.19 | 224.29 |
| TIP3P 103k | 10⁻⁴ | Prolate, P = 5 | 6.29 | 14.75 | 0.76 | 1.04 | 3.30 | 160.05 | 32.09 | 218.29 |
| TIP3P 103k | 10⁻⁴ | Gaussian, order 4 | 12.91 | 1.05 | 13.20 | 1.10 | 3.27 | 90.81 | 35.99 | 158.34 |
| TIP3P 103k | 10⁻⁴ | Gaussian, order 6 | 12.39 | 2.55 | 2.15 | 0.15 | 5.23 | 90.45 | 36.09 | 149.02 |
| TIP3P 103k | 10⁻⁵ | Prolate, P = 6 | 11.52 | 20.70 | 1.40 | 2.49 | 5.15 | 181.83 | 34.20 | 257.28 |
| TIP3P 103k | 10⁻⁵ | Gaussian, order 4 | 78.25 | 1.29 | 151.38 | 19.45 | 12.42 | 91.38 | 33.42 | 387.59 |
| TIP3P 103k | 10⁻⁵ | Gaussian, order 6 | 20.93 | 2.86 | 6.92 | 0.73 | 6.41 | 99.02 | 38.69 | 175.56 |
| Cellulose | 10⁻³ | Prolate, P = 6 | 16.70 | 20.53 | 0.24 | 0.37 | 6.09 | 269.05 | 81.76 | 394.73 |
| Cellulose | 10⁻³ | Prolate, P = 5 | 9.38 | 18.91 | 0.27 | 0.34 | 4.54 | 270.91 | 84.39 | 388.73 |
| Cellulose | 10⁻³ | Gaussian, order 4 | 7.41 | 1.51 | 2.27 | 0.22 | 3.20 | 149.75 | 80.10 | 244.44 |
| Cellulose | 10⁻³ | Gaussian, MDIR's default | 8.40 | 1.33 | 2.81 | 0.26 | 3.41 | 152.61 | 81.28 | 250.10 |
| Cellulose | 10⁻⁴ | Prolate, P = 6 | 15.92 | 22.83 | 0.49 | 0.74 | 6.49 | 268.57 | 82.64 | 397.67 |
| Cellulose | 10⁻⁴ | Prolate, P = 5 | 10.19 | 20.35 | 0.87 | 1.44 | 4.20 | 269.31 | 83.04 | 389.40 |
| Cellulose | 10⁻⁴ | Gaussian, order 4 | 21.74 | 1.92 | 16.32 | 2.05 | 5.32 | 161.37 | 87.25 | 295.97 |
| Cellulose | 10⁻⁴ | Gaussian, order 6 | 16.20 | 3.29 | 2.61 | 0.38 | 7.26 | 151.22 | 79.68 | 260.64 |
| Cellulose | 10⁻⁵ | Prolate, P = 6 | 13.22 | 21.62 | 1.32 | 2.12 | 5.87 | 232.87 | 83.45 | 360.47 |
| Cellulose | 10⁻⁵ | Prolate, P = 7 | 19.74 | 19.98 | 0.69 | 1.02 | 8.08 | 229.03 | 80.56 | 359.10 |
| Cellulose | 10⁻⁵ | Gaussian, order 4 | 85.62 | 1.37 | 177.02 | 15.82 | 15.36 | 135.76 | 82.60 | 513.54 |
| Cellulose | 10⁻⁵ | Gaussian, order 6 | 23.96 | 3.20 | 8.83 | 0.64 | 8.18 | 161.33 | 98.30 | 304.43 |

The prolate default's step over the Gaussian's, and over the fastest
Gaussian with a Coulomb-only cutoff of 7 and 6 Å, estimated from the parts
at 16 threads. The estimate scales the loop over pairs and the builds of
the list with the cube of the reach (cutoff + 1.5 Å), the points of the
grid with the inverse cube of the cutoff for both kernels (the Gaussian's
spacing follows $1/\beta \propto r_c$ at a fixed erfc, the prolate
kernel's follows $r_c$ by eq. 15), the FFT with $M \log M$, and the
product with $M$; the spreading, the window or B-splines, the gathering,
and the rest stay. That is the paper's setting without the Lennard-Jones;
in MDIR the cutoff is shared with it, and a shorter Coulomb cutoff alone
would leave the loop over pairs as it is and only enlarge the grid.

| System | CPU | ε | / order 4, 16 / 4 / 1 threads | / fastest, 16 / 4 / 1 threads | / fastest, 7 Å | / fastest, 6 Å |
|---|---|---|---|---|---|---|
| JAC | AMD EPYC 7543 | 10⁻³ | 1.71 / 1.67 / 1.72 | 1.71 / 1.67 / 1.72 | 1.73 | 1.71 |
| JAC | AMD EPYC 7543 | 10⁻⁴ | 1.52 / 1.56 / 1.61 | 1.52 / 1.56 / 1.63 | 1.44 | 1.37 |
| JAC | AMD EPYC 7543 | 10⁻⁵ | 0.66 / 0.75 / 0.74 | 1.38 / 1.49 / 1.57 | 1.21 | 1.07 |
| TIP3P 34k | AMD EPYC 9354 | 10⁻³ | 1.71 / 1.75 / 1.62 | 1.71 / 1.75 / 1.62 | 1.73 | 1.69 |
| TIP3P 34k | AMD EPYC 7543 | 10⁻⁴ | 1.41 / 1.57 / 1.54 | 1.58 / 1.60 / 1.59 | 1.49 | 1.40 |
| TIP3P 34k | AMD EPYC 9354 | 10⁻⁵ | 0.57 / 0.72 / 0.76 | 1.44 / 1.50 / 1.50 | 1.27 | 1.15 |
| TIP3P 103k | AMD EPYC 9354 | 10⁻³ | 1.67 / 1.69 / 1.66 | 1.67 / 1.69 / 1.66 | 1.68 | 1.64 |
| TIP3P 103k | AMD EPYC 9354 | 10⁻⁴ | 1.42 / 1.49 / 1.49 | 1.51 / 1.56 / 1.57 | 1.44 | 1.37 |
| TIP3P 103k | AMD EPYC 7543 | 10⁻⁵ | 0.66 / 0.82 / 0.85 | 1.47 / 1.53 / 1.59 | 1.32 | 1.20 |
| Cellulose | AMD EPYC 7543 | 10⁻³ | 1.61 / 1.64 / 1.63 | 1.61 / 1.64 / 1.63 | 1.58 | 1.52 |
| Cellulose | AMD EPYC 7543 | 10⁻⁴ | 1.34 / 1.46 / 1.51 | 1.53 / 1.52 / 1.56 | 1.45 | 1.38 |
| Cellulose | AMD EPYC 9354 | 10⁻⁵ | 0.70 / 0.62 / 0.97 | 1.18 / 1.52 / 1.73 | 1.08 | 0.99 |

**Reading (CPU).**
- **Against the fastest Gaussian order**, the prolate default's step is
  longer at every tier, on every system, and at every number of threads
  measured. The ratio is 1.6 to 1.75 at 10⁻³ (where order 4 at erfc 10⁻⁴
  takes the time of MDIR's default within 2.3%, so that the ratio to the
  default is 1.6 to 1.7), 1.5 to 1.6 at 10⁻⁴, and
  1.2 to 1.5 at 10⁻⁵ at 16 threads (1.5 to 1.7 at 1 thread). The fastest
  prolate $P$ changes these by about 5% at most.
- **Against order 4 with its grid raised**, the paper's kind of baseline,
  the ratio is 1.3 to 1.75 at 10⁻³ and 10⁻⁴, and 0.57 to 0.70 at 10⁻⁵
  at 16 threads (0.62 to 0.97 at 4 and 1). At 10⁻⁵ order 4 needs 9.3
  to 129 million points, 84 to 89 times those of the prolate kernel, and
  its FFT is 34 to 41% of its step.
- **The FFT is a small part of the step on one node.** In the fastest
  Gaussian at 16 threads it is 0.9 to 1.6% of the step at 10⁻³ and 10⁻⁴
  (6% for order 4 on JAC at 10⁻⁴, as fast as order 6 there) and 2.9 to
  5.6% at 10⁻⁵, and about the same share at 1 thread. The prolate kernel's 5 to 7 times fewer points
  than order 6 at 10⁻⁵ save at most that share.
- **The loop over pairs is the largest part**, 50 to 70% of the step, and
  the prolate kernel's takes 1.7 to 1.85 times the Gaussian's at the same
  cutoff and list (1.4 to 1.7 on Cellulose at 10⁻⁵, whose sampled runs
  vary more). Its direct term is a
  polynomial of $x = r/r_c$ in a cell (a square root, a floor, and Horner's
  rule for the value and the slope), where the Gaussian's is one table of
  $r^2$ in f32 (D94); #380. The difference, 5.7 ms on JAC to 70 to 120 ms
  on Cellulose, is larger than the whole mesh of the fastest Gaussian.
  *Since D280* the direct sum is such a table too, and the
  loop takes 0.98 to 1.02 times the Gaussian's (below).
- **The mesh alone** (spreading, window or B-splines, FFT, product,
  gathering) of the prolate default takes 0.99 to 1.09 times that of
  order 6 at 10⁻⁵ at 16 threads. Its window, by Clenshaw's recurrence,
  costs 1.3 to 23 ms against 0.1 to 3.3 for the B-splines, the largest
  part of its mesh on the three larger systems.
- **Threads.** From 16 threads to 1 the ratio to the fastest Gaussian
  changes by at most 0.2 (0.55 on Cellulose at 10⁻⁵): on one core the FFT
  is no larger a share of the step at these sizes.
- **A shorter, Coulomb-only cutoff** (estimated) brings the ratio to the
  fastest Gaussian at 10⁻⁵ to 1.08 to 1.32 at 7 Å and 0.99 to 1.20 at 6 Å;
  at 10⁻³ and 10⁻⁴ it stays 1.4 to 1.7.

On the CPU target of one node, therefore, the prolate kernel takes less
time than order 4 with its grid raised at 10⁻⁵ only, and than the fastest
Gaussian order at none of the three accuracies measured; the loop over
pairs (#380) and the window, not the FFT, set the difference.
With the direct sum as a table of $r^2$ (below), the loop over pairs no
longer sets a difference, and the window does.

**The direct sum as a table of $r^2$ (issue #380, D280).**
Before this change the prolate kernel's loop over pairs took 1.7 to 1.85
times the Gaussian's on the CPU (above) and, measured anew on one RTX 4090
in the jobs below, 1.02 to 1.12 times (JAC 53.0 against 50.3 µs, Cellulose
770.6 against 690.7 µs at 10⁻³). The direct sum is now
$f q_i q_j (1 - x)\bar\chi(x)/r$, with $\bar\chi$ the mean of $\chi$ over
$[x, 1]$ as a Chebyshev series in $2x - 1$ (`md.chebyshev`), so that
`md-exec-simplify-distance` takes it and its derivative into functions of
$r^2$ and `md-exec-expand-radial` makes them tables of cubics in f32 with
the fit and the test of D94 (white paper, Section 5.6): the kernel reads
the prolate kernel's direct sum as it reads the Gaussian's, with no square
root, no index of a cell, and one load of 16 bytes. The table brings the
loop to the Gaussian's time, the bound that the other candidates of #380
(cells in $r^2$, a branch-free piecewise form) aimed at, and those keep
the gathered loads of the cells, so they were not built; vectorization
across pairs is step 4 of D267, which a kernel of this shape shares with
the Gaussian's. The cells stay in kernels of f64, the pairs of
[free_energy], and the excluded pairs, and with `MDIR_PROLATE_CELLS=1`.

The tables, which depend on ε and the cutoff alone (here 9 Å), rebuilt as
the pass builds them (`scripts/benchmarks/prolate-ewald/pair_table.py`);
the Gaussian's force table at the same cutoff has 7 bits, 20 KB. A step
reads the force's table, and the energy's only where it writes energies.

| ε | Terms of $\bar\chi$ | Force: bits, intervals, size | Energy: bits, intervals, size | The cubics in f32, of the largest value of an interval (force, energy) |
|---|---|---|---|---|
| 10⁻³ | 20 | 7, 1281, 20 KB | 8, 2561, 40 KB | 1.1, 1.2 × 10⁻⁷ |
| 10⁻⁴ | 21 | 7, 1281, 20 KB | 9, 5121, 80 KB | 1.2, 2.3 × 10⁻⁷ |
| 10⁻⁵ | 23 | 8, 2561, 40 KB | 9, 5121, 80 KB | 1.1, 1.2 × 10⁻⁷ |

**Accuracy.** $\Delta$ as above (one step from the coordinates of the
topology, CPU and GPU, against the Gaussian at erfc $10^{-13}$, 0.33 Å,
order 8 in double, `scripts/benchmarks/prolate-ewald/pair_delta.py`), with
the cells (`MDIR_PROLATE_CELLS=1`) and with the table, and the difference
of their forces relative to the RMS Coulomb force of the reference. In
double precision both are the cells: the program is that of `main` byte
for byte, and the CPU gives the same forces to the bit.

| System | ε | $\Delta$, double | $\Delta$, mixed, CPU: cells, table | $\Delta$, mixed, GPU: cells, table | Table − cells, CPU, GPU |
|---|---|---|---|---|---|
| JAC | 10⁻³ | 5.65 × 10⁻⁴ | 5.742, 5.742 × 10⁻⁴ | 5.742, 5.742 × 10⁻⁴ | 3.8, 3.4 × 10⁻⁷ |
| JAC | 10⁻⁴ | 7.10 × 10⁻⁵ | 1.2426, 1.2426 × 10⁻⁴ | 1.2422, 1.2422 × 10⁻⁴ | 3.7, 3.5 × 10⁻⁷ |
| JAC | 10⁻⁵ | 7.95 × 10⁻⁶ | 1.0199, 1.0199 × 10⁻⁴ | 1.0197, 1.0197 × 10⁻⁴ | 3.9, 3.6 × 10⁻⁷ |
| TIP3P 34k | 10⁻³ | 6.24 × 10⁻⁴ | 6.504, 6.503 × 10⁻⁴ | 6.504, 6.504 × 10⁻⁴ | 3.9, 3.6 × 10⁻⁷ |
| TIP3P 34k | 10⁻⁴ | 6.89 × 10⁻⁵ | 1.963, 1.963 × 10⁻⁴ | 1.963, 1.963 × 10⁻⁴ | 3.9, 3.6 × 10⁻⁷ |
| TIP3P 34k | 10⁻⁵ | 1.124 × 10⁻⁵ | 1.839, 1.839 × 10⁻⁴ | 1.839, 1.839 × 10⁻⁴ | 4.1, 3.8 × 10⁻⁷ |

The table moves the forces by 3.4 to 4.1 × 10⁻⁷ of the RMS Coulomb force,
which leaves $\Delta$ of mixed precision (1.0 to 6.5 × 10⁻⁴, set by the
precision below 10⁻³) the same to three or four digits at every tier, and
is 20 to 1600 times below $\Delta$ of the kernel in double.

**The step on the CPU.** One job per system and tier alternates `main` and
the branch (frozen copies of their programs) for the prolate default and
the fastest Gaussian of the tables above (the same order and grid, the
erfc tolerance 10⁻⁴, 10⁻⁵, 10⁻⁶ at the three tiers), in 5 rounds with the
first dropped, as above; the pairs are the sampler's share of the main
thread's wall clock times the rate of the sampled run (`pair_job.py` and
`pair_summary.py` in `scripts/benchmarks/prolate-ewald/`). One round of
the branch's prolate kernel on Cellulose at 10⁻³ (534.9 ms, twice the
others, from another job on the node) is left out, and one sampled run at
1 thread (the Gaussian of `main` on TIP3P 34k at 10⁻⁵) ended without its
rate and is not used. ms a step:

| System | CPU | ε | Prolate, `main`: 16 / 4 / 1 threads | Prolate, branch: 16 / 4 / 1 | Gaussian: 16 / 4 / 1 | Pairs at 16: prolate `main`, branch, Gaussian |
|---|---|---|---|---|---|---|
| JAC | AMD EPYC 7543 | 10⁻³ | 21.1 / 70.7 / 270.8 | 15.3 / 48.8 / 179.4 | 12.3 / 42.3 / 159.6 | 13.5, 7.6, 7.5 |
| JAC | AMD EPYC 7543 | 10⁻⁴ | 21.4 / 71.6 / 275.3 | 15.6 / 50.2 / 183.8 | 14.2 / 46.6 / 172.4 | 13.4, 7.6, 7.6 |
| JAC | AMD EPYC 7543 | 10⁻⁵ | 21.1 / 73.5 / 282.6 | 15.2 / 51.6 / 191.2 | 15.3 / 49.6 / 178.8 | 13.4, 7.6, 7.6 |
| TIP3P 34k | AMD EPYC 7543 | 10⁻³ | 86.1 / 324.3 / 1234.1 | 59.2 / 218.6 / 830.0 | 49.4 / 179.2 / 739.2 | 61.1, 34.1, 34.1 |
| TIP3P 34k | AMD EPYC 7543 | 10⁻⁴ | 85.2 / 327.0 / 1246.8 | 58.1 / 220.4 / 841.3 | 53.4 / 203.7 / 791.1 | 61.9, 33.9, 33.5 |
| TIP3P 34k | AMD EPYC 9354 | 10⁻⁵ | 82.0 / 306.0 / 1168.2 | 61.5 / 218.3 / 838.6 | 64.4 / 208.4 / 780.1 | 56.8, 36.4, 37.2 |
| TIP3P 103k | AMD EPYC 7543 | 10⁻³ | 253.4 / 931.1 / 3664.0 | 176.4 / 623.9 / 2431.4 | 149.5 / 535.4 / 2120.9 | 182.8, 101.3, 100.3 |
| TIP3P 103k | AMD EPYC 9354 | 10⁻⁴ | 224.8 / 884.4 / 3392.8 | 162.1 / 617.1 / 2359.3 | 153.3 / 578.0 / 2166.9 | 160.1, 91.9, 89.9 |
| TIP3P 103k | AMD EPYC 7543 | 10⁻⁵ | 257.4 / 955.2 / 3784.3 | 180.7 / 652.9 / 2545.3 | 175.8 / 624.8 / 2336.3 | 182.3, 101.1, 100.8 |
| Cellulose | AMD EPYC 7543 | 10⁻³ | 388.2 / 1372.7 / 5052.3 | 270.8 / 953.0 / 3414.3 | 246.0 / 846.2 / 3061.2 | 257.2, 143.9, 143.9 |
| Cellulose | AMD EPYC 7543 | 10⁻⁴ | 386.6 / 1384.7 / 5073.5 | 272.1 / 962.9 / 3462.8 | 258.4 / 912.7 / 3245.3 | 256.5, 145.6, 145.6 |
| Cellulose | AMD EPYC 9354 | 10⁻⁵ | 356.4 / 1336.6 / 4785.5 | 257.1 / 961.1 / 3411.0 | 256.0 / 943.3 / 3275.9 | 224.8, 128.2, 127.4 |

The prolate default's step over the fastest Gaussian's, before and after,
in the form of the table above, and its loop over pairs over the
Gaussian's:

| System | CPU | ε | / fastest, 16 / 4 / 1 threads, `main` | / fastest, 16 / 4 / 1, branch | Pairs / Gaussian's, `main`, branch |
|---|---|---|---|---|---|
| JAC | AMD EPYC 7543 | 10⁻³ | 1.71 / 1.67 / 1.72 | 1.24 / 1.16 / 1.12 | 1.79, 1.01 |
| JAC | AMD EPYC 7543 | 10⁻⁴ | 1.51 / 1.54 / 1.59 | 1.10 / 1.08 / 1.07 | 1.76, 1.00 |
| JAC | AMD EPYC 7543 | 10⁻⁵ | 1.38 / 1.49 / 1.58 | 0.99 / 1.04 / 1.07 | 1.77, 1.01 |
| TIP3P 34k | AMD EPYC 7543 | 10⁻³ | 1.74 / 1.81 / 1.67 | 1.20 / 1.22 / 1.12 | 1.80, 1.00 |
| TIP3P 34k | AMD EPYC 7543 | 10⁻⁴ | 1.59 / 1.60 / 1.60 | 1.09 / 1.08 / 1.06 | 1.83, 1.01 |
| TIP3P 34k | AMD EPYC 9354 | 10⁻⁵ | 1.28 / 1.48 / 1.50 | 0.96 / 1.05 / 1.07 | 1.51, 0.98 |
| TIP3P 103k | AMD EPYC 7543 | 10⁻³ | 1.70 / 1.74 / 1.73 | 1.18 / 1.17 / 1.15 | 1.82, 1.01 |
| TIP3P 103k | AMD EPYC 9354 | 10⁻⁴ | 1.46 / 1.56 / 1.57 | 1.06 / 1.07 / 1.09 | 1.75, 1.02 |
| TIP3P 103k | AMD EPYC 7543 | 10⁻⁵ | 1.47 / 1.54 / 1.62 | 1.03 / 1.05 / 1.09 | 1.80, 1.00 |
| Cellulose | AMD EPYC 7543 | 10⁻³ | 1.61 / 1.62 / 1.65 | 1.10 / 1.13 / 1.12 | 1.79, 1.00 |
| Cellulose | AMD EPYC 7543 | 10⁻⁴ | 1.50 / 1.52 / 1.57 | 1.05 / 1.06 / 1.07 | 1.77, 1.00 |
| Cellulose | AMD EPYC 9354 | 10⁻⁵ | 1.41 / 1.42 / 1.47 | 1.00 / 1.02 / 1.04 | 1.76, 1.01 |

**On one RTX 4090** (`--idle`, the same cases in mixed precision, 5 rounds
alternated in one job a system, the first dropped; the loop over pairs is
the kernel of the step's pairs under `MDRT_PROFILE` and `MDRT_WAIT`, µs a
call), ms a step:

| System | ε | Prolate: `main`, branch | Gaussian: `main`, branch | Pairs, prolate: `main`, branch | Pairs, Gaussian: `main`, branch |
|---|---|---|---|---|---|
| JAC | 10⁻³ | 0.202, 0.200 | 0.147, 0.147 | 53.0, 50.9 | 50.3, 50.4 |
| JAC | 10⁻⁴ | 0.195, 0.194 | 0.171, 0.171 | 52.8, 51.1 | 52.0, 51.5 |
| JAC | 10⁻⁵ | 0.195, 0.193 | 0.194, 0.194 | 53.4, 51.6 | 51.1, 51.5 |
| TIP3P 34k | 10⁻³ | 0.608, 0.592 | 0.392, 0.391 | 192.7, 174.1 | 173.9, 175.7 |
| TIP3P 34k | 10⁻⁴ | 0.591, 0.569 | 0.508, 0.508 | 195.2, 174.1 | 174.7, 173.3 |
| TIP3P 34k | 10⁻⁵ | 0.572, 0.563 | 0.579, 0.579 | 194.2, 176.8 | 175.4, 176.3 |
| TIP3P 103k | 10⁻³ | 1.801, 1.751 | 1.121, 1.123 | 573.9, 524.0 | 523.1, 521.0 |
| TIP3P 103k | 10⁻⁴ | 1.700, 1.651 | 1.517, 1.518 | 572.5, 524.1 | 522.6, 521.8 |
| TIP3P 103k | 10⁻⁵ | 1.684, 1.644 | 1.617, 1.615 | 572.5, 532.2 | 522.5, 523.1 |
| Cellulose | 10⁻³ | 2.480, 2.397 | 1.628, 1.628 | 770.6, 693.7 | 690.7, 692.7 |
| Cellulose | 10⁻⁴ | 2.401, 2.327 | 2.100, 2.099 | 767.7, 694.5 | 695.5, 694.4 |
| Cellulose | 10⁻⁵ | 2.375, 2.314 | 2.253, 2.251 | 767.2, 703.0 | 696.0, 695.1 |

**Reading (the table of $r^2$).**
- **The loop over pairs** of the prolate kernel takes 0.98 to 1.02 times
  the Gaussian's on the CPU (1.51 to 1.83 before) and 0.99 to 1.02 times
  on the GPU (1.02 to 1.12 before).
- **The step** of the prolate default on the CPU at 16 threads is 0.68 to
  0.75 of `main`'s. Against the fastest Gaussian it is 1.10 to
  1.24 times at 10⁻³ (1.61 to 1.74 before), 1.05 to 1.10 at 10⁻⁴ (1.46 to
  1.59), and 0.96 to 1.03 at 10⁻⁵ (1.28 to 1.47); at 1 thread 1.12 to
  1.15, 1.06 to 1.09, and 1.04 to 1.09. On one GPU the step falls by 1 to
  4% and is 0.97 to 1.03 times the Gaussian's at 10⁻⁵.
- **What is left** at 10⁻³ and 10⁻⁴ is the mesh: the window by Clenshaw's
  recurrence (1.3 to 23 ms against 0.1 to 3.3 for the B-splines, above)
  and the spreading of $P^3$ points.
- **The Gaussian is unchanged.** Its lowered program, and those of the
  prolate kernel in double precision and with `MDIR_PROLATE_CELLS=1`, are
  `main`'s byte for byte on the CPU; the Gaussian's step takes 0.994 to
  1.021 times `main`'s on the CPU at 16 threads (1.021 on Cellulose at
  10⁻³, within its spread of 6 ms) and the same within 0.2% on the GPU.
- **The compile time** of the prolate kernel grows by 0.5 s on JAC
  (4.54 to 5.04 s) and on TIP3P 103k (2.45 to 2.91 s), the search for two
  tables, and by less than 0.1 s on the others.
