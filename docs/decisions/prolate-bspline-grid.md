# D[prolate-bspline-grid]

Recorded at its merge.

**With `kernel = "PROLATE"` and `window = "BSPLINE"`, the automatic grid is
refined until the mesh error estimate $E_B$ of the B-spline window is at
most `tolerance` $\eta$ (at most $\eta/2$ for the original density with this
window), from the grid of eq. 15 of [Liang2026] up to eight times finer,
and a bound not met there is refused.** Issue #985. Before, D440's
automatic grid stayed at the spacing $\pi r_c/c$ for this window, and the
relative force error $\Delta$ stayed at 1.7 to 3.6 × 10⁻³ for every
tolerance.

$E_B = \sqrt{Q/R}$ is the expected error of the reciprocal force between
two charges at uncorrelated places, relative to that force. $Q$ is the
error functional of analytic differentiation [HockneyEastwood1988]
[Ballenegger2012] for the B-spline of order $n$ with the deconvolution
$\prod_a\hat U^{-2}(k_a/K_a)$, with $\varrho_\mathbf J =
\prod_a(k_a/(k_a + J_aK_a))^n$ and $\mathcal G(\mathbf m) =
\hat\chi(2\pi r_c\lvert\mathbf m\rvert)/m^2$:
$Q = \sum_\mathbf k[\sum_{\mathbf J\ne\mathbf 0}(\varrho_\mathbf J^2\mathcal
G(\tilde{\mathbf m}_\mathbf 0) - \mathcal G(\tilde{\mathbf m}_\mathbf
J))^2\tilde m_\mathbf J^2 + \mathcal G(\tilde{\mathbf m}_\mathbf
0)^2\sum_{\mathbf J\ne\mathbf J'}\varrho_\mathbf J^2\varrho_{\mathbf
J'}^2\tilde m_\mathbf J^2]$ and $R = \sum_\mathbf k\sum_\mathbf J\mathcal
G(\tilde{\mathbf m}_\mathbf J)^2\tilde m_\mathbf J^2$, over the grid and
its aliases $\lvert J_a\rvert \le 1$. $\hat\chi$ is the transform the
kernels take: that of the constrained density in full, or that of the
prolate density within its band. The derivation is in the white paper,
Section 5.6.

The spacing is found by bisection, one spacing for the three edges, with
sizes even and 7-smooth as for every automatic grid. The estimate is
computed on every grid of this window. The log gives it with its bound
and the source of the grid; the manifest records `aliasing_estimate`,
`aliasing_bound`, and `grid_source` (`grid`, `max_spacing`, `automatic`).
Explicit grids, `max_spacing`, the prolate window, and the Gaussian kernel
are unchanged.

*Checked.* $E_B$ against a NumPy sum over the whole grid with its own
density and quadrature, to $10^{-6}$, for continuity 1 to 3 and 8 or 40
terms (`prolate-function.test`). On 1200 charges at uncorrelated places
the error of the reciprocal force is 1.10 to 1.19 times $E_B$ at $c$ =
9.5392 and 14.471, one to three times the grid of eq. 15.
`prolate-bspline-grid.test` and `prolate-bspline-grid-gpu.test` check
on 400 such charges at $10^{-3}$ and $10^{-4}$, in mixed and double
precision, that the grid of eq. 15 misses the tolerance (8.1 and 7.3 ×
10⁻³) and that the automatic grid meets it (0.77 and 1.12 times it, in
0.7 to 1.3), with an error 1.29 and 1.35 times the $E_B$ of the manifest
(in 1.1 to 1.55; the excess is larger on fewer charges). An $E_B$ doubled
in the driver fails both (0.52 and 0.58 times $\eta$, 0.66 and 0.69 times
the doubled $E_B$). `prolate-bspline-manifest.test` checks the three
fields of the manifest and the log for the automatic grid, a given grid,
and `max_spacing`, and the bound $\eta/2$ for continuity 0 with this
window; `prolate-control.test` that a given grid at $10^{-9}$ runs. On
the Coulomb forces of TIP3P water (9,989 atoms), JAC, Factor IX, and
Cellulose, against the Gaussian kernel in double precision at
$\mathrm{erfc}(\beta r_c) = 10^{-10}$, order 8, and 0.5 Å (which agrees with
OpenMM's PME at an Ewald tolerance of $10^{-6}$ to $2.1\times10^{-6}$ on
three of them; on Factor IX OpenMM's setting is off by $2.8\times10^{-5}$),
$\Delta/\eta$ at the automatic grid is 0.66 to 0.77 at $10^{-3}$, 0.31 to
0.41 at $10^{-4}$, and 0.27 to 0.34 at $10^{-5}$ on the CPU and on a GPU in
double precision, and the same in mixed precision but at $10^{-5}$, up to
0.46 (the table is in the white paper). Before, the grid of eq. 15 gave
1.7 to 3.6 × 10⁻³ on the same systems.

*Limits.* The measurements are for continuity 2 and eight terms with $c$
from $\eta$. `tolerance` sets the bound also with an explicit `bandwidth`,
as for the prolate window. Below $c(\eta)$, $\Delta$ can exceed $\eta$
(JAC, `bandwidth = 9.5392`, $\eta = 10^{-4}$: 108³, $1.69\times10^{-4}$),
and with `bandwidth` alone the default $10^{-5}$ is refused, with a message
that says so. $\Delta/E_B$ on JAC at $c$ = 9.5392 and 120³ is 1.14 for
continuity 3 and 2.4 for 40 terms. The excess of 1.10 to 1.19 on
uncorrelated charges falls with their number (1.07 for 4800 in the same
cube), the trend of the force of a charge on itself, which $Q$ leaves out.
In mixed precision on the CPU, $\Delta$ at $10^{-5}$ rises to up to
$0.91\,\eta$ from the rounding of binary32.

*Not taken.* A bound fitted to $\Delta$ rather than to $E_B$: the ratio
$\Delta/E_B$ of molecular systems runs from 0.2 to 1.7 over the grids and
bandwidths measured, since the charges of neutral molecules are
correlated. Weighting $Q$ by an intramolecular charge structure factor
narrowed it to a factor of two but left a trend with $c$, and would make
the grid depend on the coordinates. The bound therefore over-refines at
tight tolerances: at $10^{-4}$ JAC reaches $\Delta = 7.7\times10^{-5}$ on
80³ against the automatic 108³, and the default $10^{-5}$ gives Cellulose
882 × 420 × 420.
