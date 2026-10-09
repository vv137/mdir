# Triclinic cells (M2)

Status: 2026-10-02. P0 and P1 done (D123): the readers take triclinic
cells (Amber's angles and `IFBOX` 2 and 3, the nine numbers of `.gro`, six
numbers of `[boundary] box` for CHARMM with the rotation of its frame) and
reduce them; on the CPU, the minimum image in one pass, the neighbor matrix
in fractional coordinates, and PME through $H^{-1}$ run them at constant
volume. A truncated octahedron of Amber (549 TIP3P, Na⁺, Cl⁻) agrees with
sander in every term, the electrostatics to the ratio of the Coulomb
constants; a rhombic dodecahedron of GROMACS (631 TIP3P) agrees with a rerun
of GROMACS to $4\times10^{-7}$ in the bonded terms and the Lennard-Jones and
$4.5\times10^{-6}$ in Coulomb, its single precision; 10 ps at constant
energy in the octahedron conserve it (`triclinic.test`,
`triclinic-cell.test`). The device (P2, P3) and the barostat (P4) refuse
them. Cells are orthorhombic in M1. This document
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
$\cos\alpha$, $c$, as NAMD and OpenMM write it and as MDTraj, MDAnalysis,
and VMD read it. CHARMM's own DCDs store $H_s$ instead, and cpptraj reads a
DCD of the version of CHARMM (MDIR's are) as such unless told otherwise:
`trajin md.dcd ucell` reads MDIR's triclinic frames right (checked on a
truncated octahedron: without `ucell`, 29.608 Å and 91.28° for 29.604 Å
and 109.47°). A rectangular cell reads the same either way. A checkpoint stores
$H$ whole; one of M1, three edges, is read as a diagonal $H$. The log
reports the volume $a_x b_y c_z$. `scripts/render/movie.py` takes the
angles of a DCD (and the PSF of CHARMM as a structure): it makes molecules
whole with the minimum image in one pass, puts their centers in the image
nearest to the center of the view, which near the faces of an octahedron
or a dodecahedron is not always the one-pass image but one a lattice
vector from it, and draws the Wigner–Seitz cell of
the lattice, the box of an orthorhombic cell, the truncated octahedron, the
rhombic dodecahedron, or the hexagonal prism, from the planes halfway to
the 26 nearest points of the lattice; an orthorhombic frame renders as
before, pixel for pixel. Analyses with cpptraj read MDIR's triclinic frames
with `ucell`, as above.

## 2. Invariants

| | Invariant | Where it is kept |
|---|---|---|
| I1 | $H$ is lower triangular and reduced | Reduced on input; the barostat keeps it (I3) |
| I2 | $r_c \le \tfrac12 \min(a_x, b_y, c_z)$ | Checked at the start, at a commit of a writable borrow (D238), and at every change of the cell (`mdrtSetBox`), in place of the edge of twice the cutoff. With the neighbor matrix it is all that is asked of the cell (D241) |
| I3 | The barostat scales $H' = H\,\mathrm{diag}(\boldsymbol\mu)$ and $\mathbf x' = \mathbf x\,\mathrm{diag}(\boldsymbol\mu)$ only: isotropic and semi-isotropic coupling | The barostat refuses any other coupling |
| I4 | A neighbor structure that stores the image of an entry stores it as lattice indices $\mathbf n = (n_a, n_b, n_c)$, applied as $\mathbf n H$ | The builds of the groups and of the dual list (D115). The matrix stores no image: an entry is a particle, and its loops take the image of the pass below |

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
I2. The bound is on $r_c$, the reach of the kernels that take that image
at each evaluation (tuples, the loops over the matrix). It is not a bound
on the reach $R$ of a neighbor structure, but then the *build* of a
structure must not take the image of the pass for the nearest one beyond
$\tfrac12 \min(a_x, b_y, c_z)$: the groups keep an entry for each image
within their reach (D115, I4), which they hold completely for a reach
below $\min(a_x, b_y, c_z)$ ([groups-m1.md](groups-m1.md), Section 5.2,
D242), and the matrix tests every image within its reach, as
follows. GROMACS's further bound $r_c \le b_y - |c_y|$ comes
from its search, which tries one lattice vector at a time; MDIR's does not.

**The images of a pair within a reach (D241, #258).** The
image of the pass lies in the brick, and it is the nearest image only if
the nearest image lies in the brick too. A pair whose nearest image
$\mathbf d$ has $\tfrac12 c_z < \lvert d_z\rvert$ and
$\lVert\mathbf d\rVert < R$ is taken by the pass to
$\mathbf d \mp \mathbf c$, whose $x$ and $y$ move by the tilts: in a
rhombic dodecahedron with $c_z$ = 1.72 nm, a pair 0.91 nm apart along
$z$ comes out 1.67 nm apart. A build that tests that image alone leaves the
pair out of a list of $R$ = 1.0 nm, and the pair is then absent when it
comes within the cutoff. Until D241 the build of the matrix
did so, and the builder refused $R > \tfrac12 \min(a_x, b_y, c_z)$,
which a barostat could still take the cell past.

Because $H$ is lower triangular, the components of an image
$\mathbf r' = \mathbf r - \mathbf n H$ are

$$
z' = r_z - n_c c_z, \qquad y' = r_y - n_c c_y - n_b b_y, \qquad
x' = r_x - n_c c_x - n_b b_x - n_a a_x :
$$

$z'$ depends on $n_c$ alone, $y'$ on $n_c$ and $n_b$, and $x'$ on all
three. An image within $R$ has $\lvert z'\rvert$, $\lvert y'\rvert$,
and $\lvert x'\rvert$ below $R$. So $n_c$ is an integer between
$(r_z - R)/c_z$ and $(r_z + R)/c_z$; for each, with
$y_1 = r_y - n_c c_y$, $n_b$ is an integer between $(y_1 - R)/b_y$ and
$(y_1 + R)/b_y$; and for each, with $x_1 = r_x - n_c c_x - n_b b_x$, $n_a$
is an integer between $(x_1 - R)/a_x$ and $(x_1 + R)/a_x$. These nested
ranges are exactly the images in the cube $[-R, R]^3$. None within $R$ is
outside them, since each condition is necessary; and none of them can be
dropped by a test of one component, since each is in the cube. (The
sphere would trim the corners of the cube, at a square root a level; the
test of the distance does that.) The argument uses the triangular form
alone: it holds for any reach, and for a cell that is not reduced.

Counted from the image of the pass, where $\lvert r_z\rvert \le
\tfrac12 c_z$ and $y_1$ and $x_1$ are within half of $b_y$ and $a_x$
after their rounding, a range along the axis $a$ of length $L_a$ is

| Reach | The integers of the range |
|---|---|
| $R \le \tfrac12 L_a$ | 0 alone: the image of the pass |
| $\tfrac12 L_a < R \le L_a$ | 0, and $\operatorname{sign}(r_a)$ where $L_a - \lvert r_a\rvert < R$: the image on the other side |
| $R > L_a$ | one more a side for each further $L_a$ |

With $R \le \tfrac12 \min(a_x, b_y, c_z)$ the image of the pass is the
only one, which is the statement above. Up to $R = \min(a_x, b_y, c_z)$
there are at most 8, each of which occurs. Beyond it a fixed set would
not do: in the flat cell $(6, 6, 1;\ 0, 1.5, 1.5)$ the pass leaves the
displacement $(3, 3, 0.5)$ as it is, 4.27 long, and its nearest image is
$2\mathbf c$ away, $(0, 0, -1.5)$.

**The build of the matrix** is one of two functions, chosen at each build
by the cell of that build. Where $R \le 0.49 \min(a_x, b_y, c_z)$ (0.49
for the margin of the search in f32) it is the build that tests the image
of the pass alone, `..._triclinic_pass`, written by hand and unchanged.
In a narrower cell it is `..._triclinic_images`: the same function with
the test of a candidate replaced, which
`scripts/generate-matrix-images-template.py` writes from the first into the
same template. It tests a candidate with the image of the pass and, if that
is beyond $R$, runs the nested ranges (`@mdrt.image_within_triclinic`,
`@mdrt_gpu_matrix_image_within`) and takes the candidate if any image is
within $R$. Two functions rather than a branch in the loops of one: a
branch that is never taken in the search of the device, on the flag of the
build or on the candidate, cost 6% of a build of 19,072 particles (0.03 ms
of 0.48 on an RTX 3090, 1% of a step), and a cell wider than twice the
reach, the common case, should build as before. A row holds a particle once, whatever the number of its images
within the reach. That is enough: two images of a pair differ by a nonzero
lattice vector, which is at least $\min(a_x, b_y, c_z) \ge 2 r_c$ long, so
at most one is within $r_c$; it is within $\tfrac12 \min(a_x, b_y, c_z)$,
so it is the image that the pass of the loop over pairs finds. A pair with
two images within $R$ and one within $r_c$ is evaluated once, at that one.
The cells of the search need no change: a pair with an image within $R$ has
its cells of the fractional coordinates within the range of the search
around the torus, whichever image it is. Nor does the test of validity
(D80): a pair that the build left out had every image beyond $R$, and the
argument of D80 holds for each image.

The ranges count at most 64 images a side, so that a cell that a failed
run has blown up gives loops that end (as #168 has it for the cells). They
hold every image while $R \le 63.5 \min(a_x, b_y, c_z)$, which under I2
is so for `pairlist_distance` $\le 127\,\times$ `cutoff`: the builder
refuses a longer one for the matrix of a triclinic cell, a test that does
not depend on the cell. The build, a commit (D238), and the runtime under
a barostat therefore ask the same of a cell with the matrix: I2.

**I3 and what stays as it is.** $H\,\mathrm{diag}(\boldsymbol\mu)$ scales
the columns of $H$: it stays lower triangular, and each bound of I1
involves one column, so it stays reduced. The lattice shifts of I4 are
invariant, a displacement scales as $\mathbf d\,\mathrm{diag}(\boldsymbol\mu)$,
and so $|\mathbf d'| \ge \min_a \mu_a |\mathbf d|$: the test of validity of
D80 holds unchanged, with $\mu_a$ the ratio of the diagonals of $H$ to those
of the reference cell. The pressure of each axis is the diagonal of the
virial over $a_x b_y c_z$, as now, and the center of the reference positions
of restraints scales by $\mathrm{diag}(H)/\mathrm{diag}(H_\text{file})$, as
D124 has it. A
coupling of the shape of the cell (anisotropic, or OpenMM's flexible
barostat) breaks I3 and is not in this work.

## 3. What changes

From the survey of M1 (line counts are of today's tree):

| Area | Today | Triclinic |
|---|---|---|
| IR | `md.orthorhombic_cell %lx, %ly, %lz`; the lowerings carry the cell as `vector<3xf64>` | `md.triclinic_cell` of six numbers; the lowerings carry the diagonal and the tilts $(b_x, c_x, c_y)$. `!md.cell` already speaks of lattice vectors |
| Specialization | — | Whether a run is triclinic is fixed when it is compiled, as OpenMM fixes it when a context is made: an orthorhombic run compiles to the code of today, at no cost |
| Minimum image | $\mathbf d - \mathbf L \odot \mathrm{roundeven}(\mathbf d / \mathbf L)$ in the kernels of pairs and tuples and in the templates | The pass of Section 2; 9 multiply-adds in a chain against 6 independent ones |
| The matrix (CPU and GPU) | Cells along each axis over $[0, L)$ | Particles wrapped into $[0, 1)^3$ of the fractional coordinates; cells of the fractional coordinates, as many as the widths of the cell between its faces hold, searched around the torus (D123, D125); a candidate is taken if any of its images is within the reach (Section 2, D241) |
| Groups (GPU) | Columns in $x$-$y$, sorted in $z$; an entry carries $e \in [-4, 4]$ per axis, applied as $\mathbf e \odot \mathbf L$ in the gather and the pruning; D115 adds images when $2(h_a + R) \ge L_a$ | Columns of the brick, as GROMACS and OpenMM keep them; $\mathbf e$ read as $\mathbf n$, in five bits a vector, and applied as $\mathbf n H$: five multiply-adds once per entry, nothing per pair; the image of a candidate chosen by the pass around the center of the group; D115's condition on $a_x$, $b_y$, $c_z$; complete for a reach below the least of them (D242) |
| PME | Fractional coordinates $x/L$; $\mathbf k = \mathbf m / \mathbf L$; the Gaussian $\exp(-\pi^2 k^2/\beta^2)$ as a product of tables of each axis (D104) | $\mathbf s = \mathbf x H^{-1}$, three multiply-adds more, $H^{-1}$ triangular; $\mathbf k = \mathbf m H^{-\mathsf T}$; forces through $H^{-\mathsf T}$; the Gaussian computed directly, since $k^2$ is no longer a sum of one term per axis; the grid from $|\mathbf a|, |\mathbf b|, |\mathbf c|$; the virial keeps its form, $\delta - 2(1/k^2 + \pi^2/\beta^2)\,\mathbf k \otimes \mathbf k$ |
| Barostat | Edges times $\boldsymbol\mu$; volume $L_xL_yL_z$ | $H\,\mathrm{diag}(\boldsymbol\mu)$ (I3); volume $a_xb_yc_z$; the tilts scale with their columns (D127) |
| Restraints | Centers of the references times the edges over those of the file (D124) | The same with the diagonal of $H$ (I3) |
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
| P0 | Done (D123): the cell, readers (inpcrd, `IFBOX = 2`, `.gro`, `[boundary]` with angles, CHARMM's rotation), reduction, I2, writers (DCD, checkpoint, log) | Amber's `solvateOct` and GROMACS's octahedron give the same shape |
| P1 | Done (D123): the CPU, tuples, the matrix, PME | sander on Amber's octahedron; GROMACS on a rhombic dodecahedron; conservation |
| P2 | Done (D125): the device with the neighbor matrix: the cell as on the CPU, the tuples and pairs with the shared minimum image, a triclinic build of the matrix in fractional coordinates, and the kernels of PME of `PMEGPUTriclinic.mlir` | P1 on the CPU: the terms and 200 steps to the printed digits in double precision |
| P3 | Done (D126). The device: groups, the dual list, D115's images. The search of candidates wraps the indices of cells per axis, which is wrong across the face of z (or y) of a tilted cell, where a neighbor is displaced by $\mathbf c$ (or $\mathbf b$), $x$ and $y$ with it. As GROMACS does, each group searches with shifts $t_z, t_y \in \{-1, 0, 1\}$ and recomputes its window of columns for each; the image of an entry is the lattice shift $\mathbf n$ that the search found (I4), not one chosen per axis; the kernel of D115 and the gather take $\mathbf n H$ | The matrix of the device (P2), in the same precision |
| P4 | Done (D127): the barostat (I3); CHARMM's hexagonal cells | Conservation at constant energy; the density of water against a rectangular cell; CHARMM 51b1 on a hexagonal cell in its frame |
| P5 | The rates of a protein in an octahedron against its cube; docs and the white paper | pmemd.cuda and GROMACS on the same systems |

The defect of the barostat with restraints (roadmap, Section 1), which
touches the same code as P4, was fixed before it (D124).
