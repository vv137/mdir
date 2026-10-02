# Triclinic cells (M2)

Status: 2026-10-02. P0 in part: the readers take triclinic cells (Amber's
angles and `IFBOX` 2 and 3, the nine numbers of `.gro`, six numbers of
`[boundary] box` for CHARMM with the rotation of its frame), reduce them,
and `mdir check` prints them (`triclinic-cell.test`); the build refuses
them until P1. Cells are orthorhombic in M1. This document
plans general triclinic cells: the truncated octahedra and rhombic
dodecahedra of solutes (Amber's `solvateOct`, GROMACS's `editconf -bt`),
and the hexagonal cells of CHARMM-GUI's membranes. It rests on a survey of
where MDIR assumes the orthorhombic cell, a reading of OpenMM's
implementation, the GROMACS manual and source, and the documentation of
CHARMM 51b1, with the convention of CHARMM checked against the program.

A truncated octahedron holds a sphere in 77% of the volume of the cube
that holds it, a rhombic dodecahedron in 71%: a solvated protein needs
23% to 29% fewer waters, and its runs are that much faster.

## 1. The cell inside MDIR

The cell is the matrix of the GROMACS manual
[[GromacsManual2025]](references.md#gromacsmanual2025), lower triangular,
with the box vectors as its rows:

$$
H = \begin{pmatrix} \mathbf a \\ \mathbf b \\ \mathbf c \end{pmatrix}
  = \begin{pmatrix} a_x & 0 & 0 \\ b_x & b_y & 0 \\ c_x & c_y & c_z \end{pmatrix},
\qquad a_x, b_y, c_z > 0,
$$

in the reduced form

$$
|b_x| \le \tfrac12 a_x, \qquad |c_x| \le \tfrac12 a_x, \qquad |c_y| \le \tfrac12 b_y .
$$

A position is $\mathbf x = \mathbf s H$ with $\mathbf s$ its fractional
coordinates; the volume is $a_x b_y c_z$. OpenMM requires the same form and
GROMACS keeps it (its `correct_box` adds lattice vectors when a bound is
exceeded by more than 0.1%). An orthorhombic cell is the case
$b_x = c_x = c_y = 0$.

**From lengths and angles** (Amber's `inpcrd` and PDB `CRYST1`, CHARMM's
`[boundary] box` of six numbers):

$$
a_x = a, \quad b_x = b\cos\gamma, \quad b_y = b\sin\gamma, \quad
c_x = c\cos\beta, \quad c_y = c\,\frac{\cos\alpha - \cos\beta\cos\gamma}{\sin\gamma},
\quad c_z = \sqrt{c^2 - c_x^2 - c_y^2},
$$

then the one pass of reduction that OpenMM and GROMACS make:
$\mathbf c \mathrel{-}= \mathbf b\,\mathrm{round}(c_y/b_y)$,
$\mathbf c \mathrel{-}= \mathbf a\,\mathrm{round}(c_x/a_x)$,
$\mathbf b \mathrel{-}= \mathbf a\,\mathrm{round}(b_x/a_x)$, with the bounds
checked to a tolerance of $10^{-6}$ relative. Amber's truncated octahedron,
$a = b = c$ and $\alpha = \beta = \gamma = 109.4712°$, lands on a bound:
$c_x = -a/3$ and $c_y = b_y/2$ exactly, and the rounding of $\pm\tfrac12$
must leave it there. GROMACS's octahedron of `editconf`, lengths $d$ and
angles 70.53°, 109.47°, 70.53°, is the same lattice in another basis,
$\mathbf a = (d, 0, 0)$, $\mathbf b = (d/3, 2\sqrt2 d/3, 0)$,
$\mathbf c = (-d/3, \sqrt2 d/3, \sqrt6 d/3)$. The first test of this work
reads both and finds one lattice.

**From GROMACS's `.gro`**: the nine numbers are
$a_x\, b_y\, c_z\, a_y\, a_z\, b_x\, b_z\, c_x\, c_y$; $a_y$, $a_z$, $b_z$
must be 0.

**From CHARMM.** CHARMM keeps a cell as the symmetric matrix
$H_s = G^{1/2}$, the positive square root of the metric
$G = H H^{\mathsf T}$ of lengths and angles, whose rows are its vectors
(its `XTLABC`). It has the same lengths and angles as $H$, so
$H_s = H R$ for a rotation $R = H^{-1} H_s$, and a position of CHARMM is
taken into MDIR's frame as $\mathbf x = \mathbf x_s R^{\mathsf T}$. For a
rectangular cell $R = I$. Checked against CHARMM 51b1: in a hexagonal cell
of $a = b = 20$ Å and $\gamma = 120°$, two particles at the origin and at
(20, 4, 0) Å interact as if 9.2 Å apart, the image along
$\mathbf a_s = 20(\cos 15°, -\sin 15°, 0)$, not 4 Å apart along
$\mathbf a = (20, 0, 0)$. A trajectory of MDIR is in its own frame; a user
who takes it back to CHARMM applies $R^{-1}$.

**Writers.** A DCD frame stores $a$, $\cos\gamma$, $b$, $\cos\beta$,
$\cos\alpha$, $c$, as NAMD and OpenMM write it and as cpptraj, MDTraj, and
VMD read it (CHARMM's own DCDs store $H_s$ instead). A checkpoint stores
$H$ whole; one of M1, three edges, is read as a diagonal $H$. The log
reports the volume $a_x b_y c_z$; `scripts/render/movie.py` and
`scripts/benchmarks/mdbench/structure.py` take the angles of a DCD.

## 2. Invariants

| | Invariant | Where it is kept |
|---|---|---|
| I1 | $H$ is lower triangular and reduced | Reduced on input; the barostat keeps it (I3) |
| I2 | $r_c \le \tfrac12 \min(a_x, b_y, c_z)$ | Checked at the start and at every change of the cell (`mdrtSetBox`), in place of the edge of twice the cutoff |
| I3 | The barostat scales $H' = H\,\mathrm{diag}(\boldsymbol\mu)$ and $\mathbf x' = \mathbf x\,\mathrm{diag}(\boldsymbol\mu)$ only: isotropic and semi-isotropic coupling | The barostat refuses any other coupling |
| I4 | A neighbor structure stores the image of an entry as lattice indices $\mathbf n = (n_a, n_b, n_c)$, applied as $\mathbf n H$ | The builds of the groups and of the dual list (D115) |

**I2 and the minimum image.** Every nonzero vector of a reduced lattice is
at least $\min(a_x, b_y, c_z)$ long, and the brick
$|d_x| \le \tfrac12 a_x$, $|d_y| \le \tfrac12 b_y$, $|d_z| \le \tfrac12 c_z$
tiles space under it. The minimum image of a displacement is therefore one
pass, in this order:

$$
\mathbf d \mathrel{-}= \mathbf c\,\mathrm{round}(d_z/c_z), \qquad
\mathbf d \mathrel{-}= \mathbf b\,\mathrm{round}(d_y/b_y), \qquad
d_x \mathrel{-}= a_x\,\mathrm{round}(d_x/a_x),
$$

exact for every pair whose nearest image is within
$\tfrac12 \min(a_x, b_y, c_z)$, so for every pair within the cutoff under
I2. The bound is on $r_c$, the reach of the kernels that take an image per
pair (tuples, the matrix), not on the reach of a list: a list that reaches
past it keeps an entry for each image within its reach (D115), and I4 makes
that exact. GROMACS's further bound $r_c \le b_y - |c_y|$ comes from its
search, which tries one lattice vector at a time; MDIR's does not.

**I3 and what stays as it is.** $H\,\mathrm{diag}(\boldsymbol\mu)$ scales
the columns of $H$: it stays lower triangular, and each bound of I1
involves one column, so it stays reduced. The lattice shifts of I4 are
invariant, a displacement scales as $\mathbf d\,\mathrm{diag}(\boldsymbol\mu)$,
and so $|\mathbf d'| \ge \min_a \mu_a |\mathbf d|$: the test of validity of
D80 holds unchanged, with $\mu_a$ the ratio of the diagonals of $H$ to those
of the reference cell. The pressure of each axis is the diagonal of the
virial over $a_x b_y c_z$, as now, and the reference positions of restraints
scale by $\mathrm{diag}(H)/\mathrm{diag}(H_\text{file})$, as D74 has it. A
coupling of the shape of the cell (anisotropic, or OpenMM's flexible
barostat) breaks I3 and is not in this work.

## 3. What changes

From the survey of M1 (line counts are of today's tree):

| Area | Today | Triclinic |
|---|---|---|
| IR | `md.orthorhombic_cell %lx, %ly, %lz`; the lowerings carry the cell as `vector<3xf64>` | `md.triclinic_cell` of six numbers; the lowerings carry the diagonal and the tilts $(b_x, c_x, c_y)$. `!md.cell` already speaks of lattice vectors |
| Specialization | — | Whether a run is triclinic is fixed when it is compiled, as OpenMM fixes it when a context is made: an orthorhombic run compiles to the code of today, at no cost |
| Minimum image | $\mathbf d - \mathbf L \odot \mathrm{roundeven}(\mathbf d / \mathbf L)$ in the kernels of pairs and tuples and in the templates | The pass of Section 2; 9 multiply-adds in a chain against 6 independent ones |
| The matrix (CPU and GPU) | Cells along each axis over $[0, L)$ | Particles wrapped into the Cartesian brick $[0,a_x) \times [0,b_y) \times [0,c_z)$ by the same pass; cells of the brick; the search reaches across the faces of $z$ and $y$ with the shifts $\mathbf c$ and $\mathbf b$, whose $x$ and $y$ parts move the window of cells |
| Groups (GPU) | Columns in $x$-$y$, sorted in $z$; an entry carries $e \in [-4, 4]$ per axis, applied as $\mathbf e \odot \mathbf L$ in the gather and the pruning; D115 adds images when $2(h_a + R) \ge L_a$ | Columns of the brick, as GROMACS and OpenMM keep them; $\mathbf e$ read as $\mathbf n$ and applied as $\mathbf n H$: five multiply-adds once per entry, nothing per pair; the image of a candidate chosen by the pass around the center of the group; D115's condition on $a_x$, $b_y$, $c_z$ |
| PME | Fractional coordinates $x/L$; $\mathbf k = \mathbf m / \mathbf L$; the Gaussian $\exp(-\pi^2 k^2/\beta^2)$ as a product of tables of each axis (D104) | $\mathbf s = \mathbf x H^{-1}$, three multiply-adds more, $H^{-1}$ triangular; $\mathbf k = \mathbf m H^{-\mathsf T}$; forces through $H^{-\mathsf T}$; the Gaussian computed directly, since $k^2$ is no longer a sum of one term per axis; the grid from $|\mathbf a|, |\mathbf b|, |\mathbf c|$; the virial keeps its form, $\delta - 2(1/k^2 + \pi^2/\beta^2)\,\mathbf k \otimes \mathbf k$ |
| Barostat | Edges times $\boldsymbol\mu$; volume $L_xL_yL_z$ | $H\,\mathrm{diag}(\boldsymbol\mu)$ (I3); volume $a_xb_yc_z$; the tilts scale with their columns |
| Restraints | References times the edges over those of the file | The same with the diagonal of $H$ (I3) |
| Readers | inpcrd angles parsed and refused; `.gro` off-diagonals refused; `IFBOX = 2` refused | Accepted, converted, and reduced (Section 1) |
| Writers | DCD cosines 0; checkpoint of three edges | Section 1 |

## 4. Cost

An orthorhombic run is unchanged. In a triclinic run the tuples take the
longer minimum image (a few percent of a step), the groups pay five
multiply-adds per entry instead of three and nothing per pair, the pruning
pays the same per entry, and the convolution of PME evaluates an
exponential per point of the grid instead of a product of tables; it is
bound by memory. With 23% to 29% fewer particles for the same solute, a
triclinic run of a protein should be faster than its rectangular
equivalent; the measurement is part of the validation.

## 5. Phases, each with its oracle

| Phase | Work | Oracle |
|---|---|---|
| P0 | The cell: readers (inpcrd, `IFBOX = 2`, `.gro`, `[boundary]` with angles, CHARMM's rotation), reduction, I2, writers (DCD, checkpoint, log) | Amber's `solvateOct` and GROMACS's octahedron give one lattice; a hexagonal cell of CHARMM round-trips through $R$ |
| P1 | CPU: tuples, the matrix, PME | GROMACS by a rerun on TIP3P in a truncated octahedron and in a rhombic dodecahedron (double precision, terms and forces); sander on a protein from `solvateOct` |
| P2 | GPU: the matrix and PME | P1 on the CPU |
| P3 | GPU: groups, the dual list, D115's images | The matrix |
| P4 | The barostat (I3); CHARMM's hexagonal cells | Conservation at constant energy; the density of water against a rectangular cell; CHARMM 51b1 on a hexagonal cell in its frame |
| P5 | The rates of a protein in an octahedron against its cube; docs and the white paper | pmemd.cuda and GROMACS on the same systems |

The defect of the barostat with restraints (roadmap, Section 1) touches the
same code as P4 and is fixed before it.
