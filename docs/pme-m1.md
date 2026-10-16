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
| Host | The same fixed point, added with integer atomics from a parallel loop over the particles (`@mdrt.pme_spread` of `PME.mlir`, D[cpu-pme-parallel]); every other part is a parallel loop too, under OpenMP with more than one thread (Section 3.2) |
| Mixed precision | The positions, the charges, and the forces as they are stored. On the host and on a device the B-splines, the grid, and its transform are in the type of the forces, f32 in the mixed and single modes (on the host since D[cpu-pme-parallel], in f64 before), the arithmetic of pocketfft in f64 on each line, with the fractions of the positions along the cell (the position times one over the edge, less its floor) and the edges of the cell in f64, and the point of the grid and the fraction within it in the type of the forces (D78, amended). The energy and the virial are summed in f64 across blocks (below) |
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

### 3.2 The host in parallel (D[cpu-pme-parallel])

Every part of the sum on the host is a parallel loop of the compiled code
(`scf.parallel`, lowered by `convert-scf-to-openmp` with the other loops of
the CPU, D171), and the result is the same to the bit for any number of
threads, in every mode:

| Part | On the host |
|---|---|
| Spreading | A particle an iteration (`pme_spread_one`, a function of its own so that its weights are on its own stack); its 64 values (order 4) are added to the i64 grid by `memref.atomic_rmw addi`, a `lock add` on x86. An integer sum does not depend on the order of the threads (D70). The grid is zeroed and converted to the floating-point grid by parallel loops. The first point along each edge is taken within the grid once, so a point wraps around by a subtraction, not a division |
| FFT | pocketfft [[Reinecke2019]](references.md#reinecke2019) in `runtime/mdrt_fft.c`, called for blocks of lines from a parallel loop over each of the three passes: the real lines along $z$ a plane of $x$ a block; the complex lines along $y$ and along $x$ in tiles of 8 neighboring columns of $z$, which are copied to contiguous lines, transformed, and copied back, so that the strided lines are read by cache lines. The plans are made before the loops (`mdrtFFTPrepare`); pocketfft reads a plan and allocates its own scratch, so the threads share them. Each line is transformed by the same code whatever thread takes it, and nothing is summed across lines. The arithmetic of a line is in f64; a grid in f32 is converted where the line is copied |
| Product | A row of the grid ($k_1 k_2$ rows of $k_3/2 + 1$ points) an iteration. The Gaussian influence function of a rectangular cell from the factors of its edges (D104): $m_a$, $m_a^2$, and $\exp(-\pi^2 m_a^2/\beta^2)$ times the factor of the B-splines (along $x$ also $f/(\pi V)$), computed in f64 once an evaluation, $k_1 + k_2 + k_3$ exponentials, and rounded once to the type of the grid. A triclinic cell, the dispersion (D162), and the prolate kernel (D266) compute the influence of each point in f64 (`pme_influence`) and round it once. The energy and the virial of a row are summed in the type of the grid and returned in f64, and the rows in the fixed partition of D171. A step that writes neither calls `pme_scale`, which only multiplies, as on a device |
| Gathering | A particle an iteration, its sums in the type of the grid, its force from them in f64 |

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
| The direct, excluded, and alchemical pairs | $1 - \Phi(x)$ and $\Phi(x)/x$, $x = r/r_c$, as tables of cells of polynomials (`pme_far`, `pme_near`): 64 cells of 10 terms within 10⁻¹⁴ for kernels in f64, 128 cells of 4 terms within 4 × 10⁻¹⁰ for kernels in f32 |
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
coefficients a pair in f32. Tiled spreading (#336), which would serve both
kernels, is the next step; the kernels are then to be compared again.
