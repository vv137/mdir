# Constraints and Virtual Sites

Status: kept current; last checked against the code on 2026-11-12. The
decisions are D56, D63, D68, and D403 and those that amended
them ([decisions.md](decisions.md)).

This document describes the constraints, SETTLE, SHAKE, and RATTLE
(Section 1), the virtual sites (Section 2), and the rigid groups that the
two make (Section 3, D403).

## 1. Constraints

Decided: constraints in M1 (D56). Designed and described in this section.

| Item | Decision |
|---|---|
| Water | SETTLE [[Miyamoto1992]](references.md#miyamoto1992), which solves the three distances of a rigid water in closed form |
| Which molecules | Chosen (D63): from a GROMACS topology, those with `[ settles ]`; from an Amber topology, the residues named in `water_residues`, `["WAT"]` by default as in sander. A named residue that is not one heavy atom and two hydrogens with the three distances given is an error. `rigid_water = false` constrains the same water with SHAKE, and `hydrogen_bonds = false` leaves it flexible. |
| The bonds of hydrogen | SHAKE [[Ryckaert1977]](references.md#ryckaert1977), with RATTLE [[Andersen1983]](references.md#andersen1983) for the velocities under velocity Verlet |
| Execution | The constrained bonds fall into clusters with no particle in common: a heavy atom with its hydrogens, a water. A thread takes a cluster and writes to its particles only, so no two threads write to one particle. A loop over clusters is a new kind of loop. |

### 1.1 SETTLE

| Item | Rule |
|---|---|
| Positions | After the drift, each water is brought back to its shape by the closed form of Miyamoto and Kollman [[Miyamoto1992]](references.md#miyamoto1992): the new triangle, about its center of mass, is the rigid one turned by three angles in the frame of the old plane. It is the solution of SHAKE with the constraint forces along the old bonds (checked against SHAKE converged to 10⁻¹⁵: 2 × 10⁻¹⁴ nm). The velocities take the change over the step, $\Delta\mathbf x / \Delta t$, as the first half of RATTLE [[Andersen1983]](references.md#andersen1983). |
| Positions below double precision | M-SHAKE [[Krautler2001]](references.md#krautler2001) on the three bonds of the water, the two O–H and the H–H, with the iterations of Newton of the groups of SHAKE (D112). SETTLE computes the new positions, of the size of the water, and subtracts the old: in f32 the rounding of that change, about 10⁻⁷ Å, divided by the step, is an error of the velocities whose square adds kinetic energy at every step, the drift that [[Jung2026]](references.md#jung2026) finds for single precision. JAC NVE took 830 kcal/mol in 2 ns with SETTLE in f32 (after the fix of the divisions, 286), and 2 with M-SHAKE in f32, which carries the change itself; SETTLE in f64 also conserved the energy, at 581 against 724 ns/day. Both solve the same equations, SHAKE with the forces along the old bonds. |
| Velocities | After the second kick, the impulses along the three bonds that leave no velocity along them: three linear equations, solved in closed form |
| Old positions | Taken in the periods of the cell of the new ones: the bonds before the drift are moved by the same lattice vectors as the new bonds in the minimum image |
| Virial | The mean of those of the forces of the constraints over the two halves of the step: $\mathbf G_i = 2 m_i \Delta\mathbf x_i / \Delta t^2$ of the positions, with the bonds before the drift for arms, and $\mathbf G_i = 2 m_i \Delta\mathbf v_i / \Delta t$ of the velocities, with the bonds after it; $\sum (\mathbf x_i - \mathbf x_O) \otimes \mathbf G_i$ of each. The mean is that of the pressure (D45). |
| At the start | The drawn velocities have their parts along the bonds removed before they are scaled to the temperature. The positions are not constrained, as sander does not. |
| Degrees of freedom | Three fewer for each water |
| A failure | The kernel of the constraints of the positions of each set (`settles`, then each set of SHAKE), an `md.gather_tuples` with a second result of `i1`, tests every constrained length that it gives (with `MDIR_CONSTRAINT_TEST=separate`, an `md.sum_tuples` of `i1` over each set at the new positions does): $\lvert r^2/d^2 - 1\rvert$ above $10^{-4}$ below double precision and $10^{-8}$ in double, or not a number, stops the run with the step and the set (`mdrtStopConstraintFailure`). On a device the test is in the integration kernel of the step (D110), with a flag for each set and one event (the separate tests are one kernel of their own), which the host reads after the evaluation of the step, so that the read adds no wait; a build of the step that finds positions that are not numbers records its stop until then, and reports it only if every set passed (D278) |
| Temperature of the log | The optimal estimate, with the kinetic energies of the half steps measured from the velocities after the constraints of the positions (D45; D203), which the thermostats control (D263) |
| In the IR | An `md.gather_tuples` over a tuple set `settles` of arity 3, the oxygen first, for each of the two halves, in the program of the step; the virial an `md.sum_tuples` |
| Which waters | Those of `[ settles ]` of GROMACS; from Amber the residues of `water_residues`, `["WAT"]` by default, which are an oxygen and two hydrogens with at most virtual sites after them. Their bonds and angles are dropped. |

### 1.2 SHAKE and RATTLE

| Item | Rule |
|---|---|
| Groups | With `hydrogen_bonds = true`, the bonds of hydrogen (`BONDS_INC_HYDROGEN` of Amber; a bond with an atom of atomic number 1 in GROMACS) outside the waters of SETTLE, grouped by their heavy atom: one to three hydrogens. A hydrogen bonded to two atoms, two hydrogens bonded to each other, and more than three hydrogens on an atom are errors. The bonds are dropped from the bond terms, as sander does with `ntf = 2`. |
| Positions | SHAKE [[Ryckaert1977]]: the atoms move along the bonds before the drift, the hydrogens by $\lambda_j \mathbf s_j / m_j$ and the heavy atom by $-\sum_j \lambda_j \mathbf s_j / m_0$. Each of 6 iterations of Newton solves the constraints linearized at the current bonds exactly, $\sum_j 2 (\mathbf r_k \cdot \mathbf s_j)(\delta_{kj} / m_j + 1 / m_0)\, \lambda_j = d_k^2 - r_k^2$, by the elimination of RATTLE, so that the error squares with each; sweeps bond by bond shrink it only by about $m_H / m_X$, close to $1/2$ with repartitioned masses of hydrogen, which 12 sweeps left at 10⁻⁴ of a bond and a step of 4 fs made unstable. The number is fixed, so the kernel has no test of convergence. |
| Optional one-bond projection | `analytic_bonds = true` replaces the six Newton iterations for isolated X–H groups with the rationalized near root of the same fixed-old-direction quadratic (D128). It checks the original squared-distance residual before accepting; failure or a backward predictor takes the existing Newton path. The default is `false`. Groups with two or three bonds and waters keep their existing solvers. |
| Velocities | RATTLE [[Andersen1983]](references.md#andersen1983): the impulses along the bonds solve $A \boldsymbol\tau = -\mathbf b$ with $A_{ij} = \delta_{ij} / m_j + \mathbf e_i \cdot \mathbf e_j / m_0$ exactly, by Gaussian elimination of at most $3 \times 3$ |
| Virial, degrees of freedom, the start | As for SETTLE: the mean of the virials of the forces of the constraints over the two halves of the step, one degree of freedom less for each bond, and drawn velocities without parts along the bonds. The positions of the file are not constrained before the first step, as sander does not; the first step brings the bonds to their lengths. At the start, where no step has given the forces of the constraints, the log takes their virial as $\sum (\mathbf x_k - \mathbf x_0) \cdot \mathbf G^0_k - 2 K_\text{int}$ over each group: $\mathbf G^0 = m P(\mathbf F/m) - \mathbf F$ keeps the accelerations on the constraints, and $-2 K_\text{int}$, twice the kinetic energy of the motion within the group, is the rest for a group that the constraints keep rigid (a water, a bond); for a group of SHAKE with two or three hydrogens it counts the bending as well, a few bar in a protein. |
| In the IR | An `md.gather_tuples` over a tuple set `shake1`, `shake2`, or `shake3`, the heavy atom first, for each half of the step; SETTLE comes first, then the groups of SHAKE, then the placement of the virtual sites |

Three consequences reach other stages, and are recorded now:

| Consequence | Where |
|---|---|
| The forces of the constraints add to the atomic virial | Not the pressure: the log and the barostat take that of the rigid groups by their centers, to which the forces of the constraints add nothing (D264, [dynamics.md](dynamics.md), Section 2) |
| Every constraint removes a degree of freedom: `f = 3N − N_c − 3` | The temperature, and the thermostat |
| The kinetic energies at the half steps (D45) must be measured from constrained velocities, not computed from the forces | The temperature of the log (D203) and of the thermostats, which control it (D263): the steps of the rows and those that close a period of coupling measure them. The centers of the rigid groups need no measurement: their forces give them (D264) |

## 2. Virtual sites

A virtual site is a point with a charge and no mass whose position follows
from those of a few atoms. OPC [[Izadi2014]](references.md#izadi2014) and
TIP4P-Ew put the negative charge of water on one, an extra point in Amber.
M1 has the site of water of four sites in both formats (D65, D68).

### 2.1 Placement

For a site `s` built from `i`, `j`, and `k`, with `d_ji = x_j − x_i` and
`d_ki = x_k − x_i` in the minimum image:

| Kind | Position | Parameters |
|---|---|---|
| Extra point of Amber | `x_s = x_i + a (û + v̂) / \|û + v̂\|`, with `û = d_ji / \|d_ji\|` and `v̂ = d_ki / \|d_ki\|` | `a`, the length of the bond from the oxygen to the extra point in the `prmtop` |
| `[ virtual_sites3 ]`, function 1, of GROMACS | `x_s = x_i + a d_ji + b d_ki` | `a`, `b` |

The extra point of Amber depends only on the directions of the two bonds
to the hydrogens. It lies where the linear rule puts it only when the two
bonds have the lengths that $a$ of GROMACS was computed for:
$a = b = d / (2 r_\text{OH} \cos(\theta/2))$.

### 2.2 Forces and virial

The force on a site goes to its atoms by the transpose of the derivative of
its position, $\mathbf F_k \mathrel{+}= (\partial\mathbf x_s/\partial\mathbf x_k)^{\mathsf T} \mathbf F_s$, and the site keeps none. For the
linear rule the atoms take $(1 - a - b)\, \mathbf F_s$, $a\, \mathbf F_s$, and $b\, \mathbf F_s$. For the
extra point of Amber, with $\hat{\mathbf b} = (\hat{\mathbf u} + \hat{\mathbf v}) / \lVert\hat{\mathbf u} + \hat{\mathbf v}\rVert$ and $P(\mathbf n) = I - \mathbf n \mathbf n^{\mathsf T}$:

$$
\begin{aligned}
\mathbf F_j &= \frac{a}{\lVert\mathbf d_{ji}\rVert\, \lVert\hat{\mathbf u} + \hat{\mathbf v}\rVert}\, P(\hat{\mathbf u})\, P(\hat{\mathbf b})\, \mathbf F_s, \\
\mathbf F_k &= \frac{a}{\lVert\mathbf d_{ki}\rVert\, \lVert\hat{\mathbf u} + \hat{\mathbf v}\rVert}\, P(\hat{\mathbf v})\, P(\hat{\mathbf b})\, \mathbf F_s, \\
\mathbf F_i &= \mathbf F_s - \mathbf F_j - \mathbf F_k.
\end{aligned}
$$

A hydrogen takes no force along its own bond: stretching the bond does
not move the site.

The virial of the pair terms is summed with the site as a particle. The
sum over the atoms after the move differs by $\sum_k (\mathbf x_k - \mathbf x_s) \otimes \mathbf F_k$ over
the three atoms of each site, which is 0 for the linear rule, since the
weights sum to 1, and is added for the extra point of Amber.

### 2.3 In a step

| Item | Rule |
|---|---|
| Placement | After every drift, and once where the run begins, whatever the file of coordinates says. The positions that a step returns have their sites placed, so frames and checkpoints need nothing more. |
| Forces | Moved after every evaluation of the forces, before the kick |
| Mass and velocity | A site has mass 0. A kick leaves a particle of mass 0 alone, and the velocities drawn at the start give it none; a Brownian step gives it velocity 0, and does not read the velocities (#19). |
| Degrees of freedom | $3N_m - N_c - 3$: three for each particle with a mass ($N_m$; a site counts none), less one for each constraint, less three for the center of mass where its momentum is kept, which Langevin and Brownian dynamics do not keep unless `center_of_mass_interval` removes its motion |
| Exclusions | Amber: a site is excluded from its oxygen and from what its oxygen is excluded from, as sander rebuilds them; the list of the file must not exclude more. GROMACS: those of `[ exclusions ]`; no bond reaches a site. |
| Bonded terms | Amber: the bond from the oxygen to the extra point gives its length and is dropped, with every term that has an extra point. |
| In the IR | The placement and the move of the forces are `md.gather_tuples` over a tuple set `sites_amber` or `sites_linear` of arity 4, the site first, in the programs of the step; the change of the virial is an `md.sum_tuples` |

### 2.4 What sander and GROMACS do otherwise

| Item | sander | GROMACS |
|---|---|---|
| Constraints before the first step | None: the energies at the start are those of the file | SETTLE, with `continuation = no` |
| Velocity of a site in its output | 0 | That of the placement |
| Which atoms are sites | The atoms of type `EP`, with a frame chosen from the neighbors of the owner; MDIR takes the frame of water only | Those that a section of virtual sites places, whatever the particle type of their atom type says (V or D, and A in the OPC of amber19sb.ff, as grompp takes it), with no mass |

Frames of lone pairs and of TIP5P in Amber, and the other kinds of
virtual sites of GROMACS, are rejected.

### 2.5 Validation

| Test | Result |
|---|---|
| Dipeptide in OPC, the terms at the start, against sander (`test/Driver/amber-opc.test`) | Agree but for the conventions of amber.test |
| A peptide in TIP4P-Ew, against GROMACS ([validation.md](validation.md)) | Agree to 4 × 10⁻⁶ |
| Three OPC waters: the change of the total energy with the time step (`test/Driver/virtual-sites.test`) | Shrinks as its square, 6.8 × 10⁻⁴ at 0.1 fs and 1.7 × 10⁻⁴ at 0.05 fs |
| The same: the trace of the virial against $-dU/d\lambda$ under a uniform scaling of the atoms and the cell | Agree to 6 × 10⁻⁶; without the change of Section 2.2, 35.84 instead of 43.57 kcal/mol |
| The order of the positions, against that of the files (`test/Driver/amber-opc-spatial_order.test`) | The same states to 2 × 10⁻¹⁵ |

## 3. Rigid groups

Decided: D403 (#855). `findRigidGroups`
(`lib/Driver/RigidGroups.cpp`) finds the molecules of a prepared topology
that move as rigid bodies and sorts them into kinds, from the constraints
(Section 1) and the definitions of the sites (Section 2) alone. It reads
no atomic number, no name of an atom or a residue, and assumes no order
of elements. `mdir check` reports the result ([driver.md](driver.md),
Section 1.4); no run reads it yet. Its first consumer is the generator of
the pair segments of D405 ([ops.md](ops.md), Section 4.12), which
takes up to two kinds, refuses one whose copies have
`incompleteExclusions` or `outsideExclusions`, and reads where
Lennard-Jones sits from the tables of the topology.

### 3.1 Molecules and frames

A molecule is a connected component of the graph over the particles whose
edges are the bonds, the constrained distances (the three of a group of
SETTLE, and the center and each hydrogen of a group of SHAKE), and the
pairs of a site with each of the three atoms that place it. The **frame**
of a molecule is the set of its particles with a mass; its other
particles are sites.

**R1.** A molecule is a rigid group when a constraint fixes the distance
of every pair of atoms of its frame.

*It is sufficient.* If $\lVert\mathbf x_a - \mathbf x_b\rVert = d_{ab}$
for every pair of the frame, two configurations that satisfy the
constraints have the same matrix of distances, and are therefore
congruent: one is the other moved by an isometry of space. The only
freedom left to the frame is that of a rigid body. For four or more atoms
that are not in a plane the isometry may be a reflection, which the
distances allow and a continuous motion does not reach; no record of
today gives such a frame (below), and a record that will must say which
hand it holds.

*What it leaves out.* A frame of $m$ atoms may be rigid with fewer than
$m(m-1)/2$ distances ($3m - 6$ suffice in general position for
$m \ge 3$), so the rule is not necessary in general. For the records that
`validateTopology` accepts today it is: the groups of SETTLE and SHAKE
share no particle, a group of SETTLE holds the 3 distances of 3 atoms, and
a group of SHAKE holds the $h$ distances of a center to its $h = 1, 2, 3$
hydrogens, a tree, which leaves the angles between them free unless
$h = 1$. A molecule whose frame is joined by constraints is therefore one
group, and the frames that the rule accepts are exactly the rigid ones:

| Frame | Atoms | Constrained pairs | Rigid group |
|---|---|---|---|
| A single atom | 1 | 0 of 0 | Yes: nothing to hold |
| A group of SHAKE with one hydrogen that is the whole molecule | 2 | 1 of 1 | Yes |
| A group of SETTLE that is the whole molecule | 3 | 3 of 3 | Yes |
| A group of SHAKE with two or three hydrogens | 3, 4 | 2 of 3, 3 of 6 | No: the angles at the center are free |
| Any other molecule: a bond that no constraint holds, or more than one group | | fewer than all | No |

A molecule that fails the rule is counted with the reason "a pair of
massive atoms without a constrained distance" (`flexible_frame`). A
constraint record of a later decision that holds a frame with fewer than
all its distances would need the rule widened; until then the rule is the
exact condition and its test is one count per molecule.

A topology whose constraints are not resolved (a flexible run, or one
before `prepareTopologySystem`) has no rigid groups beyond its single
atoms.

### 3.2 Kinds

Two rigid groups are copies of one **kind** when they have the same
number of particles and, taking the particles of each in the order of
their numbers, the same places of the frame atoms, the same type and
charge at each place, the same constrained distance for each pair of the
frame, and the same weights of each site (Section 3.3). The masses are
not compared. The kinds are given with the most populous first, and
kinds of equal count in the order of their first particle.

"The same" is equality of the f64 values, with no tolerance, for three
reasons. The copies of a molecule take their parameters from the same
text of a topology through the same arithmetic, so equal inputs give
equal values, and a difference in the last place is a difference of the
input. A tolerance is not transitive: with one, whether $A$ and $C$ are
one kind would depend on a $B$ between them and on the order of the
molecules. And a consumer that replaces the values of each copy by those
of the kind changes no number of the run only under equality. The cost is
that two descriptions of one site (a bisector and a linear one for the
same point) are one kind only where their weights are the same numbers;
weights that differ in the last place are two kinds, each correct.

A kind's record (`RigidKind`) has: the number of particles of a copy; the
places of the frame; the charge and the type of each place; the distances
of the frame, for the pairs of its places in lexicographic order; for
each site its place and its weights over the frame; and the particles of
every copy. Three facts that a consumer may ask are counted over the
copies and decide neither rigidity nor the kind:

| Count | A copy is counted when |
|---|---|
| `scattered` | its particles are not consecutive numbers |
| `incompleteExclusions` | some pair of its own particles is not excluded |
| `outsideExclusions` | an excluded pair has one particle in it and one outside |

The Lennard-Jones parameters of a place are those of its type, in the
tables of the topology; whether a place has any is for the consumer to
read there. Nothing is refused because of them.

### 3.3 Sites as constant-weight combinations

**R2.** Each site $s$ of a rigid group is a combination of the frame
atoms with weights that are constants of the kind and sum to 1,

$$\mathbf x_s = \sum_a w_a \mathbf x_a, \qquad \sum_a w_a = 1 .$$

The sum to 1 makes the site follow a translation of the frame, and the
linearity a rotation, so the weights are the same in every configuration
of the rigid frame. The weights follow from the two kinds of Section 2.1,
for a site of the atoms $i$, $j$, $k$:

*The linear site*, $\mathbf x_s = \mathbf x_i + a\,\mathbf d_{ji} +
b\,\mathbf d_{ki}$, is the combination $w_i = 1 - a - b$, $w_j = a$,
$w_k = b$ on any frame, rigid or not.

*The extra point of Amber*, $\mathbf x_s = \mathbf x_i + a\,(\hat{\mathbf
u} + \hat{\mathbf v})/\lVert\hat{\mathbf u} + \hat{\mathbf v}\rVert$, is
not linear in the positions. On a rigid frame the three distances
$d_{ij}$, $d_{ik}$, $d_{jk}$ are constants, and with
$\mathbf d_{ji} = d_{ij}\hat{\mathbf u}$, $\mathbf d_{ki} =
d_{ik}\hat{\mathbf v}$ the law of cosines,
$d_{jk}^2 = d_{ij}^2 + d_{ik}^2 - 2 d_{ij} d_{ik}\,\hat{\mathbf
u}\cdot\hat{\mathbf v}$, gives

$$d_{ij} d_{ik} \lVert\hat{\mathbf u} + \hat{\mathbf v}\rVert^2
= d_{ij} d_{ik} (2 + 2\,\hat{\mathbf u}\cdot\hat{\mathbf v})
= (d_{ij} + d_{ik})^2 - d_{jk}^2 \equiv Q , \qquad (3.1)$$

so that $\mathbf x_s - \mathbf x_i = (a/\lVert\hat{\mathbf u} +
\hat{\mathbf v}\rVert)(\mathbf d_{ji}/d_{ij} + \mathbf d_{ki}/d_{ik})$ is

$$w_j = \frac{a}{\sqrt Q}\sqrt{\frac{d_{ik}}{d_{ij}}}, \qquad
w_k = \frac{a}{\sqrt Q}\sqrt{\frac{d_{ij}}{d_{ik}}}, \qquad
w_i = 1 - w_j - w_k . \qquad (3.2)$$

This is exact, and its condition is that the frame holds the three
distances. `getBisectorWeights` computes $Q$ as
$(d_{ij} + d_{ik} - d_{jk})(d_{ij} + d_{ik} + d_{jk})$. For
$d_{ij} = d_{ik} = d$, the water of four sites,
$w_j = w_k = a / (2\sqrt{d^2 - d_{jk}^2/4})$: $a$ over twice the height
of the triangle. The form has no direction where the two bonds are
opposite, $Q \le 0$, and such a site has no weights.

*Off the rigid frame* the two forms differ. If the lengths are held
within a relative $\tau$, $\ell = d(1 + \varepsilon)$ with
$\lvert\varepsilon\rvert \le \tau$, then with $S = \lVert\hat{\mathbf u} +
\hat{\mathbf v}\rVert$ and $S_0$ its value on the rigid frame the
combination minus the bisector is
$a[(1/S_0 - 1/S)(\hat{\mathbf u} + \hat{\mathbf v}) +
(\varepsilon_{ij}\hat{\mathbf u} + \varepsilon_{ik}\hat{\mathbf v})/S_0]$.
For $d_{ij} = d_{ik}$, $\delta(\hat{\mathbf u}\cdot\hat{\mathbf v}) =
-(4 - S^2)(2\varepsilon_{jk} - \varepsilon_{ij} - \varepsilon_{ik})/2$
to first order, and $\delta S/S = \delta(\hat{\mathbf u}\cdot\hat{\mathbf
v})/S^2$, so the difference is at most

$$a\,\tau\left(\frac{2}{S} + \frac{2(4 - S^2)}{S^2}\right) + O(\tau^2) ,
 \qquad (3.3)$$

$4.85\,a\tau$ for the OPC water ($S = 1.237$): $8 \times 10^{-7}$ Å at
$\tau = 10^{-6}$. A consumer that places sites by the weights therefore
places them as the engine does to the tolerance of its constraints, not
bitwise, and says so where it does.

A site of any other form is not assumed to be a combination: its
molecule is counted with the reason "a site that is no constant-weight
combination of the frame" (`site_not_constant_weight`). The readers of
today give only the two kinds above (Section 2.4: the frames of lone
pairs and of TIP5P in Amber, and the other kinds of virtual sites of
GROMACS, are rejected when the topology is read), so from a file this
reason is reached only by a bisector without a direction, which
`validateTopology` refuses as lengths of SETTLE; the reason is there for
the kinds of sites that a later reader adds, such as the two sites out of
the plane of a water of five sites, which are no combination of three
atoms with weights that sum to 1. A site of that kind is a combination of
the frame atoms and the cross product of two bonds, whose weights are
constants on a rigid frame as well; giving it to consumers needs a
record with that third term, which this decision does not add.

### 3.4 Validation

| Test | Result |
|---|---|
| Made topologies with atomic numbers and names that tell nothing: three-atom groups, groups with a bisector site and with a linear site, a single atom, a constrained pair, a three-atom group with Lennard-Jones on every atom (`test/Driver/rigid-groups.test`) | 9 molecules in 5 kinds, in the order of count and first particle |
| What makes a kind: a charge, a length, or a weight one unit in the last place away; another type; the two outer atoms of a site exchanged | Two kinds; two kinds; one kind |
| One case for each reason: free bonds, a group of SHAKE with 2 and 3 hydrogens, a rigid triple bonded to one more atom, a site on a flexible frame; a site of an unknown form, a bisector without a direction | Counted with `flexible_frame`; with `site_not_constant_weight` |
| Eq. (3.2) against the placement of Section 2.1 on 20,000 random frames of three different lengths, each coordinate within $\epsilon\,(4(1 + \sum_a\lvert w_a\rvert)X + 64\lvert a\rvert/S)$, $\epsilon = 2^{-52}$, $X$ the largest coordinate | Within the bound; the largest error is 0.23 of it |
| The weights of the kinds against the placement on 20,000 rotated and shifted copies of a frame, with $40\,\epsilon\,a X/d$ more for Eq. (3.3), and for a linear site with $a \ne b$ | Within the bound; the largest error is 0.08 and 0.19 of it |
| The same tests with $w_j$ and $w_k$ exchanged, the ratio of Eq. (3.2) inverted, a sign of $w_j$, of $w_i$, or of $d_{jk}$ in $Q$ changed, the weights put on exchanged frame atoms, or $a$ and $b$ of the linear site exchanged | Each fails |
| A topology of GROMACS with waters of three and four sites, a rigid molecule with Lennard-Jones on its three atoms, an ion, and flexible waters; the OPC waters of Amber with and without their constraints (`test/Driver/check-rigid-groups.test`) | 4 kinds and 2 molecules left out; 387 of 388, the weights 0.147721 of Eq. (3.2); none without the constraints |
