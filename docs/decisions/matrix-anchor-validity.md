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

For a cell scaled by the diagonal map of D80, write $m$ for its smallest
scale and $E_{AB} = \rho_A + \rho_B$. A pair excluded at the build was
farther apart than $r_c+s+E_{AB}$ in every image. Scaling maps the old
images to the new images. Their separation is now greater than
$m(r_c+s+E_{AB})-2d_{\max}$ before subtracting the covering radii. It
follows that the list is valid if
$2d_{\max} \le m(r_c+s)-r_c+\min(0,m-1)\,2\rho_{\max}$, where
$\rho_{\max}$ is the largest unit radius. The expression uses the widest
pair on contraction and the base reach on expansion. A negative margin
requires a rebuild even when every anchor is stationary.

The constraint bound on a bond of length $\ell$ is
$\ell\sqrt{1+\tau}$, where the position solver checks the relative squared
length with tolerance $\tau$. The unit uses the largest member bound and
also covers its coordinates at the initial force evaluation or restart.
The numerical margin follows D424. A rigid water's bisector site has a
fixed distance from oxygen. A linear virtual site belongs to the unit
only when its frame is in the unit, with a bound from the sum of the
absolute weights times their bond bounds. Other sites are singletons.

The build bins the anchor coordinates with the maximum covering reach and
uses the pair-specific reach for admission. The force and energy kernels
still test the current site distances, with the existing exclusions and
minimum-image convention.

The existing deterministic fold follows each row's search order. On a GPU
the row slot also selects its lane before the lane reduction. The wider
search and the extra entries outside the cutoff can therefore change the
addition order even for identical in-cutoff site pairs. D84's repeatability
applies to each fixed list decomposition. Comparing SITE and ANCHOR to
the bit also requires an order independent of the list, including the
assignment of in-cutoff entries to lanes. Canonical MATRIX summation is
tracked in #409.

The coverage fixture rotates a constrained member through 0.40 nm while
the anchors stay still. A brute-force enumeration gives three pairs
inside the 1.0 nm cutoff at the second configuration. The site list needs
a second build. The anchor list covers the same three pairs with one
build. The negative control uses the anchor displacements with effectively
zero radii and omits the crossing pair.
