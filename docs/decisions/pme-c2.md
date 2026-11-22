# D[pme-c2]

Recorded at its merge.

Add `[pme] kernel = "PROLATE_C2"` for the constrained prolate splitting
with a four-point cubic B-spline. The default is `"GAUSSIAN"`. The
prototype targets relative RMS full-force errors of about $10^{-3}$ to
$10^{-4}$. This is an empirical range and depends on mesh and bandwidth.
Use the existing `bandwidth` for the splitting and `grid` for explicit
mesh dimensions. The window fixes `order` at four. Diagnose a conflicting
order, Gaussian `beta`, `window_bandwidth`, or explicit `influence`.
The design introduces no file format or output.

For cutoff $r_c$, construct the even concentration kernel $\chi$ in the
Legendre basis with unit integral on $[0,1]$ and
$\chi(1)=\chi'(1)=0$ (#466). With $\Phi(x)=\int_0^x\chi(y)\,dy$,
the long-range potential is $\Phi(r/r_c)/r$ inside the cutoff and $1/r$
outside. The endpoint conditions match the value and first two derivatives.
Use the cosine transform of $\chi$ on every nonzero reciprocal mode and
the cubic window deconvolution. Include the self energy
$-k_e\chi(0)\sum_i q_i^2/(2r_c)$ and the long-range exclusion correction.

One evaluator supplies the short-range value and its squared-distance
derivative to matrix, groups, and the TOWERS traversal when available.
For $s=r^2>0$, $t=s/r_c^2$ and $L(t)=\Phi(\sqrt t)/\sqrt t$, return
$s^{-1/2}-L(t)/r_c$ and $-1/(2s^{3/2})-L'(t)/r_c^3$ inside the cutoff.
Evaluate both from the same interpolant and multiply by $k_e q_i q_j$.
Both vanish at and beyond the cutoff.

Check against an independent double-precision reciprocal oracle at the
same splitting, finite differences of energy, and tight full-force
references in CPU and GPU mixed and double precision. Compare the NVE
energy slope and both half-run slopes on JAC and Cellulose over at least
2 ns against Gaussian runs at matched measured force error. The prototype
measurements in #466 do not establish trajectory drift.
