# Constraints and Virtual Sites

Status: kept current; last checked against the code on 2026-10-21. The
decisions are D56, D63, and D68 and those that amended them
([decisions.md](decisions.md)).

This document describes the constraints, SETTLE, SHAKE, and RATTLE
(Section 1), and the virtual sites (Section 2).

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
