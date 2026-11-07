# D[force-matching-sites]

Recorded at its merge.

**The product of force matching in $\boldsymbol\theta$ (D381) takes virtual sites, linear ones and the extra points of Amber, by the tangent of their placement; its derivative in the positions (D383) takes linear sites, by the spread of the forces of the energy and of the tangent to the atoms. For the extra points of Amber, whose Jacobian depends on the positions, the derivative in the positions stays refused by name.**

Part 2 of 4 of #771 (part of #726). It builds on D298 (the spread of the sites), D381, and D383. The maintainer ruled on PR #773 to include the product in $\boldsymbol\theta$ for the extra points of Amber.

## The quantity

Let $\mathbf X(\mathbf x)$ be the places of all particles, the sites placed from their atoms, and $\mathsf J = \partial\mathbf X/\partial\mathbf x$. The potential is evaluated at $\mathbf X$, and the gradient that the evaluator returns is

$$
\mathbf G = \mathsf J^{\mathsf T}\,\frac{\partial U}{\partial\mathbf X},
$$

whose rows on the sites are zero. For the cotangents $g$ and $\bar{\mathbf G}$,

$$
\Phi = g\,U + \bar{\mathbf G}\cdot\mathbf G = g\,U + (\mathsf J\bar{\mathbf G})\cdot\frac{\partial U}{\partial\mathbf X}.
$$

$\mathsf J$ does not depend on $\boldsymbol\theta$, so the product in $\boldsymbol\theta$ is that of D381 along the tangent $\mathsf J\bar{\mathbf G}$, taken at the frame, for any placement. $\bar{\mathbf G}_s$ meets a zero row of $\mathbf G$ and is not read. The derivative in the positions is

$$
\frac{\partial\Phi}{\partial\mathbf x} = \mathsf J^{\mathsf T}\Bigl(g\,\frac{\partial U}{\partial\mathbf X} + \mathsf H_{\mathbf X}\,\mathsf J\bar{\mathbf G}\Bigr) + \frac{\partial}{\partial\mathbf x}\Bigl[(\mathsf J\bar{\mathbf G})\cdot\frac{\partial U}{\partial\mathbf X}\Bigr]_{\partial U/\partial\mathbf X\ \text{fixed}},
$$

with $\mathsf H_{\mathbf X}$ the Hessian of $U$ in $\mathbf X$. The last term, the curvature of the placement, vanishes where $\mathsf J$ is constant.

- **Linear sites**, $\mathbf X_s = (1-a-b)\,\mathbf x_i + a\,\mathbf x_j + b\,\mathbf x_k$ (`[ virtual_sites3 ]`, function 1, of GROMACS): $\mathsf J$ is constant, and the derivative in the positions is D383's quantity at $\mathbf X$, spread by $\mathsf J^{\mathsf T}$. Its rows on the sites are zero.
- **The extra points of four-site water of Amber**, $\mathbf X_s = \mathbf x_i + d\,\mathbf s/\lvert\mathbf s\rvert$ with $\mathbf s = \hat{\mathbf u}+\hat{\mathbf v}$, $\hat{\mathbf u}$ and $\hat{\mathbf v}$ the unit vectors from $i$ to $j$ and to $k$ (`load_amber` gives them for OPC and TIP4P-Ew): the placement follows the directions of the bonds, and $\mathsf J$ depends on the positions. Along the tangents $\delta\mathbf u = \bar{\mathbf G}_j - \bar{\mathbf G}_i$ and $\delta\mathbf v = \bar{\mathbf G}_k - \bar{\mathbf G}_i$ of the bonds $\mathbf u$ and $\mathbf v$,

  $$
  \delta\hat{\mathbf u} = \frac{\delta\mathbf u - \hat{\mathbf u}\,(\hat{\mathbf u}\cdot\delta\mathbf u)}{\lvert\mathbf u\rvert},\qquad
  \delta\mathbf s = \delta\hat{\mathbf u} + \delta\hat{\mathbf v},\qquad
  (\mathsf J\bar{\mathbf G})_s = \bar{\mathbf G}_i + \frac{d}{\lvert\mathbf s\rvert}\,\bigl(\delta\mathbf s - \hat{\mathbf s}\,(\hat{\mathbf s}\cdot\delta\mathbf s)\bigr).
  $$

  The product in $\boldsymbol\theta$ takes it. The derivative in the positions would need the curvature term, which is not formed: it is refused by name, "virtual sites placed by the directions of their bonds". That term is a later item of #771.

**The two definitions give different forces on the atoms.** On a water at the geometry of the model, the extra point of Amber and the linear site with $a = b = d/(2\,r_{\mathrm{OH}}\cos(\theta_{\mathrm{HOH}}/2))$ are one place, and $U$ is the same. Their $\mathsf J$ differ: the spread of the force on the site (docs/constraints.md, Sections 2.1 and 2.2) moves it to the atoms by different rules, so $\mathbf G$, a loss on it, and its derivative in $\boldsymbol\theta$ differ. Force matching should use the definition of the source of the reference forces: the prmtop of Amber for forces from Amber, the topology of GROMACS for forces from GROMACS.

## The product program

- **The tangent.** After the boundary hands over $\bar{\mathbf G}$, each set of sites replaces the entry of its sites by $(\mathsf J\bar{\mathbf G})_s$ (`buildPlacedSiteTangents`, `lib/Driver/SiteSpreadIR.cpp`). The tangents of the members are read as they are, never through the minimum image, since a tangent is not a position; the extra points of Amber read the displacements of the bonds at the frame. The entry of the site is changed by adding the difference, as the positions are, so a cotangent on a site leaves a rounding in the last place.
- **The spread.** The variant of D383 spreads the forces of the energy and those of the tangent by the spread of the step (D298), written or derived as `MDIR_SITE_SPREAD` selects, before `mdrtWriteFrameProductForces`. It refuses a program with the extra points of Amber.

Programs without sites emit nothing new.

## The interface

No new key and no new argument: `FrameEnergies.vjp` of D381 and D383 accepts programs with virtual sites as above, which it refused.

## Validation

`test/Driver/python-frame-sites.test` (CPU) and `python-frame-sites-gpu.test` (RTX 5090), 2 frames each, Lennard-Jones and Coulomb under a cutoff with the shift of D210. Each definition of the site is checked against a NumPy model of its own; the references are Richardson central differences (steps $h$ and $h/2$, $h = 10^{-4}$) of the NumPy $\Phi$ in $\boldsymbol\theta$ and in $\mathbf x$. The relative error of the largest entry:

| Quantity | CPU double | CPU mixed | GPU double | GPU mixed |
|---|---|---|---|---|
| Linear, two waters: the product in $\boldsymbol\theta$ | 2.3e-12 | 1.4e-6 | 2.3e-12 | 7.4e-7 |
| Linear, two waters: the derivative in the positions | 2.4e-12 | 8.6e-7 | 2.4e-12 | 8.2e-7 |
| Linear, two waters: the same against differences of $\Phi$ from the evaluator | 2.1e-12 | | 2.1e-12 | |
| Linear, eight waters: the product in $\boldsymbol\theta$ | 2.8e-13 | 1.7e-7 | 4.7e-13 | 1.6e-8 |
| Linear, eight waters: the derivative in the positions | 3.4e-12 | 4.4e-7 | 3.3e-12 | 3.8e-7 |
| Linear, eight waters: the same against differences of $\Phi$ from the evaluator | 4.7e-12 | | 5.2e-12 | |
| Amber: the product in $\boldsymbol\theta$ | 5.6e-13 | 7.8e-7 | 5.8e-13 | 7.9e-7 |
| Amber: the same against differences of the rows of D230 along $\bar{\mathbf G}$ ($h = 10^{-6}$) | 2.0e-10 | | 1.9e-10 | |

Tolerances: 1e-8 against NumPy and 1e-7 against the evaluator in double; 2e-5 in mixed.

- **The linear sites** are OPC of amber19sb.ff as GROMACS gives it, in a topology that the test writes: two waters in a cell far larger than the cutoff, and eight in a cell of 2.0 nm through the minimum image. $\boldsymbol\theta$ is $\sigma_{OO}$, $\epsilon_{OO}$, and the charges of O, H, and M, tied by a map. The energy and the gradient agree with NumPy to at most 1.5e-14 and 3.1e-15 in double, 6.4e-6 and 1.3e-6 in mixed (tolerances 1e-10 and 1e-5). With `MDIR_SITE_SPREAD=derived` (D298), the eight waters give the same numbers on the CPU.
- **The extra points of Amber** are OPC from the prmtop of `test/Driver/Inputs/opc`, cut by ParmEd to ACE-ALA-NME and 40 waters (182 particles), with the same $\boldsymbol\theta$. NumPy places the sites by the bisector and takes $\mathsf J\bar{\mathbf G}$ by a complex step. The frames are chosen so that no pair with a particle of a water lies within $2\times10^{-5}$ nm of the cutoff. There the Coulomb force jumps, and in mixed precision a distance within rounding of the cutoff may fall on either side: on the full fixture of 1570 particles, one pair at $2\times10^{-7}$ nm of it gave 9.9e-4 of the largest entry in mixed. For the same reason the differences of the rows of D230 are taken in double only.
- **Mixed against double:** the product in $\boldsymbol\theta$, 1.4e-6, 1.7e-7, and 7.8e-7 (CPU), 7.4e-7, 1.6e-8, and 7.9e-7 (GPU); the derivative in the positions, 8.6e-7 and 4.4e-7 (CPU), 8.2e-7 and 3.8e-7 (GPU); tolerance 1e-5.
- **The product in $\boldsymbol\theta$ of D383's variant** equals that of D381 to the bit, and the rows of the sites in the derivative in the positions are zero to the bit, for the linear sites.
- **Another cotangent on the sites** changes the products by at most 3.8e-16 in double and 1.5e-7 in mixed (tolerances 1e-14 and 1e-6): the rounding of the placement.
- **One place on rigid waters.** On frames of 40 waters at the geometry of the model, each turned and moved, the prmtop and ParmEd's topology of GROMACS, with $a = b = d/(2\,r_{\mathrm{OH}}\cos(\theta_{\mathrm{HOH}}/2)) = 0.14772061$ and the tunables at the values of the prmtop, give the same energy: 5.1e-15 and 6.8e-15 in double (CPU, GPU), 3.7e-7 and 6.1e-10 in mixed (tolerances 1e-12 and 1e-5). Their gradients and products are not compared: the two spread the force on a site differently.
- **The refusals:** with the extra points of Amber, `gradients=("tunables", "positions")` is refused by name, and the product in $\boldsymbol\theta$ is given after it, the same to the bit. D381's test now expects that product where it expected the refusal; D383's test of the refusal in the positions passes unchanged.

**Programs without sites are unchanged**, against `main` 305ec589 built in a tree of its own:

- The equivalence corpus (`scripts/builder-equivalence/check.py`, the module and the lowered program): 1030 cases of 258 control files identical, 442 of them stopping with the same error in both.
- The order matrix (`scripts/builder-equivalence/order.py`): 560 cases identical (480 with the exact pair list, 40 padded, 40 searched), 108 of them stopping with the same error in both.
- Five checkpoints (JAC NVE, Factor IX NPT, and Cellulose NVE of the Amber suite, 200 steps each; the ala3 minimization; argon, 200 steps, on the GPU), in deterministic mode with the compile cache off, both executables alternated in one job on the RTX 5090: every dataset of the final checkpoint (positions, velocities, and box, and the forces in the four runs that write them; the minimization of ala3 writes none) identical to the bit, 5 of 5.
