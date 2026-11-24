# D[pme-c3]

Recorded at its merge.

Select the endpoint-constrained Coulomb splitting with `[pme] kernel = "PROLATE_C3"`.
Use `bandwidth` for the split and `grid` for the mesh. Charge assignment
uses an order-four cardinal B-spline. Reject explicit `beta`, `influence`,
`window_bandwidth`, and any order other than four. Gaussian is the default.

Maximize spectral concentration in the even Legendre basis through degree
78, subject to $\chi(1)=\chi'(1)=\chi''(1)=0$, and normalize
$\int_0^1\chi=1$. Section 5.6 of the paper derives the constrained solve.
For $x=r/r_c$ and $\Phi(x)=\int_0^x\chi$, the direct pair potential
$\phi=(1-\Phi)/r$ joins zero through its third derivative at the cutoff.
MATRIX, GROUPS and TOWERS evaluate $\Phi(x)/x$ as one degree-39 Chebyshev
polynomial in $2x^2-1$ and differentiate that polynomial for the force.
Use the cosine transform of the same density at every nonzero mesh mode,
the cubic window deconvolution, and the corresponding self, net-charge
and exclusion corrections.

On Cellulose at an 8 Å cutoff the earlier C² split had
$\phi'''(r_c^-)=-0.397$ Å⁻⁴ and zero exterior derivative because
$\chi''(1)\ne0$. Its excess NVE drift over a Gaussian at matched
force error was $+39.7\pm6.2$ kcal/mol/ns at 2 fs and $+3.1\pm3.4$
at 1 fs, consistent with a time-step-squared integration error. The
reciprocal force agreed with the reciprocal energy gradient. Order-six
assignment and analytic evaluation of the direct potential each left
the excess in place. C³ reduced it to $+13.4\pm5.7$ kcal/mol/ns at
2 fs (four matched velocity seeds, mean ± standard error). The paired
95% interval spans $-4.9$ to $+31.6$ kcal/mol/ns.

At bandwidth 9.5392, both splits first reached relative RMS Coulomb
force error $10^{-3}$ on the same tested meshes, $40^3$ on JAC and
$160\times80\times80$ on Cellulose. Cellulose reciprocal time on RTX
4090 increased from 126.48 to 130.20 µs per evaluation (+2.9%).
