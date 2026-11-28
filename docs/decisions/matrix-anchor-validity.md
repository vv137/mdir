# D[matrix-anchor-validity]

Recorded at its merge.

**Constraint-bounded validity for the single MATRIX list.**

Issue #930. Keep directed site rows and the site-pair force cutoff.
The optional control key `[execution].matrix_validity` selects `SITE`
(the default) or `ANCHOR` for checked MATRIX lists. An anchor is the
oxygen of a rigid water, the heavy atom of a constrained hydrogen group,
or a lone site. The covering radius comes from the constraints and their
numerical tolerance.

Write $A(i)$ for the unit containing site $i$, $c_A$ for its anchor, and
$\rho_A$ for an upper bound on the distance of its sites from $c_A$.
In a fixed periodic cell, the minimum-image distance is the metric on the
periodic quotient, so the triangle inequality gives
$d(c_A,c_B) \le \rho_A + d(x_i,x_j) + \rho_B$ for sites $i \in A$ and
$j \in B$. If $d(x_i,x_j) < r_c$ at evaluation, and the largest anchor
displacement since the build is $d_{\max}$, the build-time distance
between the anchors was less than
$r_c + 2d_{\max} + \rho_A + \rho_B$. Keeping every such unit pair within
$r_c + s + \rho_A + \rho_B$ covers all site pairs in the cutoff while
$2d_{\max} \le s$.
