# D[force-matching-sites]

Recorded at its merge.

**The product of force matching in $\boldsymbol\theta$ (D381) and its derivative in the positions (D383) take linear virtual sites: the product program places the tangent as the positions are placed and spreads the forces of the energy and of the tangent to the atoms as a step spreads its forces. Virtual sites that follow the directions of their bonds, the extra points of Amber, stay refused by name.**

Part 2 of 4 of #771 (part of #726). It builds on D298 (the spread of the sites), D381, and D383.

## The quantity

A linear site $s$ is placed at $\mathbf X_s = (1-a-b)\,\mathbf x_i + a\,\mathbf x_j + b\,\mathbf x_k$ (`[ virtual_sites3 ]`, function 1, of GROMACS), and an atom at $\mathbf X_m = \mathbf x_m$. The Jacobian $\mathsf J = \partial\mathbf X/\partial\mathbf x$ is constant. The potential is evaluated at $\mathbf X$, and the gradient that the evaluator returns is

$$
\mathbf G = \mathsf J^{\mathsf T}\,\frac{\partial U}{\partial\mathbf X},
$$

whose rows on the sites are zero. For the cotangents $g$ and $\bar{\mathbf G}$,

$$
\Phi = g\,U + \bar{\mathbf G}\cdot\mathbf G = g\,U + (\mathsf J\bar{\mathbf G})\cdot\frac{\partial U}{\partial\mathbf X},
$$

so the product in $\boldsymbol\theta$ is that of D381 with the tangent $\mathsf J\bar{\mathbf G}$, and

$$
\frac{\partial\Phi}{\partial\mathbf x} = \mathsf J^{\mathsf T}\Bigl(g\,\frac{\partial U}{\partial\mathbf X} + \mathsf H_{\mathbf X}\,\mathsf J\bar{\mathbf G}\Bigr),
$$

with $\mathsf H_{\mathbf X}$ the Hessian of $U$ in $\mathbf X$: D383's quantity at $\mathbf X$, spread by $\mathsf J^{\mathsf T}$. Its rows on the sites are zero. $\bar{\mathbf G}_s$ meets a zero row of $\mathbf G$ and is not read.

## The product program

- **The tangent.** After the boundary hands over $\bar{\mathbf G}$, each set of sites replaces the entry of its sites by $(1-a-b)\,\bar{\mathbf G}_i + a\,\bar{\mathbf G}_j + b\,\bar{\mathbf G}_k$ (`buildPlacedSiteTangents`, `lib/Driver/SiteSpreadIR.cpp`). A tangent is not a position: the members are read as they are, never through the minimum image. The entry of the site is changed by adding the difference, as the positions are, so a cotangent on a site leaves a rounding in the last place.
- **The spread.** The variant of D383 spreads the forces of the energy and those of the tangent by the spread of the step (D298), written or derived as `MDIR_SITE_SPREAD` selects, before `mdrtWriteFrameProductForces`.
- **Refused.** The extra points of four-site water of Amber, $\mathbf x_i + a\,(\hat{\mathbf u}+\hat{\mathbf v})/\lvert\hat{\mathbf u}+\hat{\mathbf v}\rvert$, follow the directions of the bonds. Their $\mathsf J$ depends on the positions, which adds to $\partial\Phi/\partial\mathbf x$ a term of the second derivative of the placement. The product in $\boldsymbol\theta$ would need only $\mathsf J\bar{\mathbf G}$ at the frame, but that tangent of their placement is not written either. Both products refuse a program with them, by name: "virtual sites placed by the directions of their bonds". `load_amber` gives this kind for OPC and TIP4P-Ew; `load_gromacs` gives the linear kind for the waters of its force fields.

Programs without sites emit nothing new.

## The interface

No new key and no new argument: `FrameEnergies.vjp` of D381 and D383 accepts programs with linear sites, which it refused.

## Validation

(Filled in by the PR.)
