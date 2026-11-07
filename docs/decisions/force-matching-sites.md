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

(Filled in by the PR.)
