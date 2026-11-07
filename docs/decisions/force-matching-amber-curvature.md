# D[force-matching-amber-curvature]

Recorded at its merge.

**The derivative in the positions of the product of force matching (D383) takes the extra points of Amber, whose $\mathsf J$ depends on the positions, by the curvature of their placement: the derivative of the spread $\mathsf J^{\mathsf T}$ along the cotangent, the force of the energy on the site held fixed, added to the spread forces of the tangent.**

The later item of #771 that D384 recorded (part of #726). It builds on D383 (the variant of the product program), D384 (the tangent and the spread of the sites in the products), and D298 (the spread of the sites).

## The quantity

Let $\mathbf X(\mathbf x)$ be the places of all particles, the sites placed from their atoms, $\mathsf J = \partial\mathbf X/\partial\mathbf x$, and $\mathbf f = \partial U/\partial\mathbf X$ at $\mathbf X(\mathbf x)$. The evaluator returns $\mathbf G = \mathsf J^{\mathsf T}\mathbf f$, zero on the sites, and for the cotangents $g$ and $\bar{\mathbf G}$ of a frame

$$
\Phi = g\,U + \bar{\mathbf G}\cdot\mathbf G = g\,U + (\mathsf J\bar{\mathbf G})\cdot\mathbf f.
$$

By the chain rule, $\partial\mathbf f/\partial\mathbf x = \mathsf H_{\mathbf X}\mathsf J$ with $\mathsf H_{\mathbf X}$ the Hessian of $U$ in $\mathbf X$, so

$$
\frac{\partial\Phi}{\partial\mathbf x} = \mathsf J^{\mathsf T}\bigl(g\,\mathbf f + \mathsf H_{\mathbf X}\,\mathsf J\bar{\mathbf G}\bigr) + \mathbf C,\qquad
\mathbf C = \frac{\partial}{\partial\mathbf x}\Bigl[(\mathsf J(\mathbf x)\,\bar{\mathbf G})\cdot\mathbf f\Bigr]_{\mathbf f\ \text{fixed}}.
$$

D384 forms the first term. $\mathbf C$, the curvature of the placement, is the part that this decision adds. With $\varphi(\mathbf x) = \mathbf f\cdot\mathbf X(\mathbf x)$ at a fixed $\mathbf f$, $(\mathsf J\bar{\mathbf G})\cdot\mathbf f = \bar{\mathbf G}\cdot\nabla\varphi$, so $\mathbf C = \nabla^2\varphi\,\bar{\mathbf G}$. The matrix of the second derivatives of $\varphi$ is symmetric, so $\mathbf C$ is also the derivative of the spread along $\bar{\mathbf G}$:

$$
\mathbf C = \frac{\mathrm d}{\mathrm d\epsilon}\Bigl[\mathsf J(\mathbf x + \epsilon\bar{\mathbf G})^{\mathsf T}\,\mathbf f\Bigr]_{\epsilon = 0}.
$$

$\mathbf C$ is linear in $\mathbf f$ and in $\bar{\mathbf G}$ and does not scale with $g$. It is zero where $\mathsf J$ is constant, for the linear sites. Each site enters through its own rows, so $\mathbf C$ is a sum over the sites, and $\bar{\mathbf G}_s$, the cotangent of a site, is not read: $\mathsf J$ depends on the positions of the atoms alone.

**The extra points of Amber.** $\mathbf X_s = \mathbf x_i + d\,\hat{\mathbf s}$, with $\mathbf s = \hat{\mathbf u} + \hat{\mathbf v}$, $\mathbf u = \mathbf x_j - \mathbf x_i$ and $\mathbf v = \mathbf x_k - \mathbf x_i$ in the minimum image, and $\hat{\mathbf n} = \mathbf n/\lvert\mathbf n\rvert$. With $\mathsf P_{\hat{\mathbf n}} = \mathsf I - \hat{\mathbf n}\hat{\mathbf n}^{\mathsf T}$, the derivative of $\hat{\mathbf n}$ is $\mathsf P_{\hat{\mathbf n}}/\lvert\mathbf n\rvert$, and the spread (docs/constraints.md, Section 2.2) has the rows

$$
(\mathsf J^{\mathsf T}\mathbf f)_j = \frac{d}{\lvert\mathbf u\rvert\,\lvert\mathbf s\rvert}\,\mathsf P_{\hat{\mathbf u}}\,\mathsf P_{\hat{\mathbf s}}\,\mathbf f_s,\qquad
(\mathsf J^{\mathsf T}\mathbf f)_k = \frac{d}{\lvert\mathbf v\rvert\,\lvert\mathbf s\rvert}\,\mathsf P_{\hat{\mathbf v}}\,\mathsf P_{\hat{\mathbf s}}\,\mathbf f_s,
$$

and $(\mathsf J^{\mathsf T}\mathbf f)_i = \mathbf f_s - (\mathsf J^{\mathsf T}\mathbf f)_j - (\mathsf J^{\mathsf T}\mathbf f)_k$ from this site, the owner taking the translation. Along $\delta\mathbf u = \bar{\mathbf G}_j - \bar{\mathbf G}_i$ and $\delta\mathbf v = \bar{\mathbf G}_k - \bar{\mathbf G}_i$, the derivatives of the frame are

$$
\delta\hat{\mathbf u} = \frac{\mathsf P_{\hat{\mathbf u}}\,\delta\mathbf u}{\lvert\mathbf u\rvert},\qquad
\delta\lvert\mathbf u\rvert = \hat{\mathbf u}\cdot\delta\mathbf u,\qquad
\delta\mathbf s = \delta\hat{\mathbf u} + \delta\hat{\mathbf v},\qquad
\delta\lvert\mathbf s\rvert = \hat{\mathbf s}\cdot\delta\mathbf s,\qquad
\delta\hat{\mathbf s} = \frac{\mathsf P_{\hat{\mathbf s}}\,\delta\mathbf s}{\lvert\mathbf s\rvert},
$$

and those of $\mathbf v$ alike; $(d/\lvert\mathbf s\rvert)\,\mathsf P_{\hat{\mathbf s}}\,\delta\mathbf s$ is the tangent of the placement of D384. With $\mathbf c = \mathsf P_{\hat{\mathbf s}}\mathbf f_s$ and $\mathbf p_u = \mathsf P_{\hat{\mathbf u}}\mathbf c$, the product rule gives

$$
\delta\mathbf c = -\bigl[\delta\hat{\mathbf s}\,(\hat{\mathbf s}\cdot\mathbf f_s) + \hat{\mathbf s}\,(\delta\hat{\mathbf s}\cdot\mathbf f_s)\bigr],\qquad
\delta\mathbf p_u = \delta\mathbf c - \delta\hat{\mathbf u}\,(\hat{\mathbf u}\cdot\mathbf c) - \hat{\mathbf u}\,(\delta\hat{\mathbf u}\cdot\mathbf c + \hat{\mathbf u}\cdot\delta\mathbf c),
$$

$$
\mathbf C_j = \frac{d}{\lvert\mathbf u\rvert\,\lvert\mathbf s\rvert}\Bigl[\delta\mathbf p_u - \Bigl(\frac{\delta\lvert\mathbf u\rvert}{\lvert\mathbf u\rvert} + \frac{\delta\lvert\mathbf s\rvert}{\lvert\mathbf s\rvert}\Bigr)\mathbf p_u\Bigr],\qquad
\mathbf C_i = -\mathbf C_j - \mathbf C_k,\qquad
\mathbf C_s = \mathbf 0,
$$

and $\mathbf C_k$ as $\mathbf C_j$ with $\mathbf v$ for $\mathbf u$. The terms in $\delta\lvert\mathbf u\rvert$ and $\delta\lvert\mathbf s\rvert$ are the derivative of the factor $d/(\lvert\mathbf u\rvert\lvert\mathbf s\rvert)$. A NumPy model of these rows agrees with a complex step of the spread to 1.2e-15 of the largest entry on random frames.

**The conventions of the program.** The program carries forces, $-\mathbf f$. The variant hands the host the spread forces of the energy, $-\mathsf J^{\mathsf T}\mathbf f$, and those of the tangent, $-\mathsf J^{\mathsf T}\mathsf H_{\mathbf X}\mathsf J\bar{\mathbf G}$, whose signs `mdrtWriteFrameProductForces` reverses, and the host applies $g$ to the first (D383). $\mathbf C$ is linear in $\mathbf f$, so the rows formed from the force on the site, $-\mathbf f_s$, are $-\mathbf C$: they are added to the forces of the tangent, after their spread, and the host's reversal gives $\mathbf C$ in $\partial\Phi/\partial\mathbf x$ with no factor $g$.

## The program

- **`SiteFrame`** (`lib/Driver/SiteSpreadIR.cpp`), the frame of a site of Amber that the placement and the spread already share, gains `motion`, the derivatives $\delta\hat{\mathbf u}$, $\delta\hat{\mathbf v}$, $\delta\mathbf s$, and $\mathsf P_{\hat{\mathbf s}}\delta\mathbf s$ along the tangents of the bonds, and `spreadTangent`, the derivative of its `spread` along them by the rules above. `buildPlacedSiteTangents` takes the tangent of the placement from `motion`, with the operations it had, in the same order.
- **`buildSiteSpreadCurvature`** (`lib/Driver/SiteIR.h`, next to `buildPlacedSiteTangents` and `buildSiteSpread`) is one gather over the sites: the displacements of the bonds at the frame, the forces of the energy on the sites before their spread, and the tangent of the atoms, read as it is, never through the minimum image (a tangent is not a position). Its kernel yields $\mathbf 0$, $-\mathbf C_i$, $-\mathbf C_j$, $-\mathbf C_k$ for the site, the owner, and the two atoms, added to the forces of the tangent.
- **`buildTunableProductOutput`** takes a second callback beside the spread: the curvature, from the forces of the energy before their spread, applied to the forces of the tangent after it. The product entry passes it for programs with the extra points of Amber alone; the refusal of D384 goes.
- With `MDIR_SITE_SPREAD=derived` (D298) the spread of the forces is derived and the curvature is still the written one (see Alternatives).

Nothing changes for programs without the extra points of Amber or for the product in $\boldsymbol\theta$: the energy, its tangent, and their derivatives are the same requests of one `md.evaluate`, and the curvature is a gather after it, on the forces alone.

## Alternatives

- **Derive the curvature from the work potential of the placement** (D298's derived spread, with the request `tangent_forces` of D383 on it): one source for the placement and its derivatives. It is not taken here: whether the tangent rules of D376 take that potential, whose field (the forces on the sites) is an argument, is not established, and the written form is two dozen operations on `SiteFrame`, checked against a complex step. It stays open for a later change to the derived spread.
- **Keep the refusal** and point force matching with OPC or TIP4P-Ew at a topology of GROMACS: wrong for forces from Amber, whose spread is that of the bisector (D384, *The two definitions give different forces on the atoms*).

## The interface

No new key and no new argument. `FrameEnergies.vjp(..., gradients=("tunables", "positions"))`, and through `mdir.torch` `torch.autograd.grad(loss, x)` and a `loss.backward()` that reaches the positions (D385), give the derivative in the positions for programs with the extra points of Amber, which they refused. The reciprocal sum of PME stays refused there by name.

## Validation

`test/Driver/python-frame-sites-amber.test` (CPU) and `python-frame-sites-amber-gpu.test` (RTX 5090), with ParmEd: OPC from the prmtop of `test/Driver/Inputs/opc`, cut to ACE-ALA-NME and 40 waters (182 particles), the fixture of D384, 2 frames whose pairs with a particle of a water all lie farther than $2\times10^{-5}$ nm from the cutoff of 0.8 nm, Lennard-Jones and Coulomb shifted at the cutoff (D210).

- **The part of the extra points.** The rest of $U$ depends on neither the places of the extra points nor their charge, so the derivative less that of the same evaluator with the charge of the extra points set to zero is the derivative of $\Phi_M = g\,U_M + \bar{\mathbf G}\cdot\mathbf G_M$, $U_M$ the Coulomb of the pairs with an extra point. NumPy forms $\Phi_M$ with the bisector placement and $\mathsf J\bar{\mathbf G}$ in closed form (D384's formula), analytic in $\mathbf x$, and differentiates it by a complex step in each coordinate ($h = 10^{-30}$) and by Richardson central differences (steps $h$ and $h/2$, $h = 4\times10^{-6}$ nm, so that no pair crosses the cutoff). On the first frame $\mathbf C$ is 28% of the largest entry of that part.
- **The whole**, against Richardson differences of $\Phi$ from the evaluator (its energy and gradient, every displaced frame in one call, $h = 4\times10^{-6}$ nm) in double, and in mixed precision against double.

The relative error of the largest entry:

| Quantity | CPU double | CPU mixed | GPU double | GPU mixed |
|---|---|---|---|---|
| The part of the extra points, against the complex step of the NumPy $\Phi_M$ | 2.6e-15 | 1.5e-6 | | |
| The same, against Richardson differences of the NumPy $\Phi_M$ | 2.3e-10 | 1.5e-6 | | |
| The whole, against Richardson differences of $\Phi$ from the evaluator | 1.0e-10 | | | |
| The whole, in mixed precision against double | | 2.2e-6 | | |
| The product in $\boldsymbol\theta$, against central differences of the NumPy $\Phi$ (D384's check) | 5.6e-13 | 7.8e-7 | | |

Tolerances: 1e-8 against NumPy and 1e-7 against the evaluator in double; 2e-5 against NumPy and 1e-5 against double in mixed (those of D383 and D384).

- With `MDIR_SITE_SPREAD=derived`, on the CPU in double: 3.1e-15, 2.3e-10, and 1.1e-10.
- **The product in $\boldsymbol\theta$ of the variant** equals that of D381 to the bit, and the rows of the sites in the derivative in the positions are zero to the bit, in all four modes.
- **Another cotangent on the sites** changes the product in $\boldsymbol\theta$ and the derivative in the positions by at most 1.0e-16 in double and 3.3e-8 in mixed (CPU; tolerances 1e-14 and 1e-6): the rounding of the placement of the tangent.
- **Through `mdir.torch`** (`python-frame-hvp-torch-amber.test`, `-gpu.test`): every form of the backward, `grad(L, x)`, `grad(L, [theta, x])`, and `L.backward()` among them, with and without a term of the energy in the loss, against `FrameEnergies.vjp`: $\boldsymbol\theta$ to the bit, $\mathbf x$ 0 in double and 3.5e-14 in mixed on the CPU (tolerances 1e-14 and 1e-9), one pass of each product that the form takes. These forms were refused.
- **D383's test of the refusals** (`python-frame-hvp.test`) expected the refusal on the full OPC fixture (1570 particles); it now expects the derivative, finite and zero in the rows of the sites.
