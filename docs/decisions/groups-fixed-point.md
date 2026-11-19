# D[groups-fixed-point]

Recorded 2026-11-19.

Proposed for #917. The force reductions of the groups and segmented traversals share one binary scale per evaluation. The runtime chooses the layout from the precision and the D420 device policy. The control file, checkpoint format, and output paths keep their existing interfaces.

## Pair forces and range

Round each Cartesian component once, to nearest with ties to even, before summing across pairs. For a component that rounds to the integer $k$, the other endpoint receives $-k$. Reserve a symmetric signed range so negation always fits. Nonfinite values and out-of-range conversions set a device flag before conversion. An errored evaluation stops with the particle and component. Decode into the pair term's destination once and preserve contributions from other terms.

For each groups traversal, bound the number of contributions to each endpoint using the outer list. Count the entries of its own group and the set mask bits of all incoming entries at its place. Let $d$ be the largest of these bounds, with a minimum of one. For i64 fields, require every rounded magnitude to be strictly below the integer quotient of the symmetric limit by $d$. The triangle inequality bounds every partial sum, including the sums held in registers.

For i32 fields, reserve $H=2^{30}-1$ quanta for ordinary contributions and the same amount for exceptions. A magnitude below $\lfloor H/d\rfloor$ is ordinary. For every other magnitude up to $H$, add its absolute integer value to a separate i64 counter at both endpoints for that component. A magnitude above $H$ or a counter above $H$ sets the range flag. A successful traversal therefore has absolute partial sums of at most $2H$ in any order. This admits an isolated large pair force even when a particle has many neighbors. D113 starts the flag copy after the writers and checks it before using the force. The diagnostic identifies an offending particle and component.

For packed fields, encode the signed partial sums as $x+2^{32}y$ modulo $2^{64}$. Both the construction and accumulation use unsigned modular arithmetic. Decode the signed low word first, subtract its sign extension modulo $2^{64}$, then take the upper word as signed. Concatenating two independent bit patterns would leak a carry into the upper field. The bound above applies independently to both fields. The other layouts under evaluation are three i32s and three i64s. The floating vector alternative requires an aligned four-component destination and a supporting architecture. It is a candidate for ordinary mixed precision only.

## Selecting a scale

For scale $S$, rounding perturbs one component by at most $1/(2S)$. If particle $i$ receives at most $d_i$ contributions, the norm of its three-component perturbation is bounded by $\sqrt{3}d_i/(2S)$. Summing the squares gives the conservative relative RMS bound

```math
\Delta_{\rm quant} \le
\frac{\sqrt{3\sum_i d_i^2}}{2 S\sqrt{\sum_i \lVert\mathbf F_i\rVert^2}}.
```

With uncorrelated uniform rounding errors the corresponding RMS estimate replaces $3\sum_i d_i^2$ in the numerator by $\sum_i d_i$. The estimate guides the sweep. It does not certify a force or a range. For each precision, choose the smallest power of two that satisfies its accuracy budget and test the guard against physical pair forces and incident counts. A lower bound on scale from accuracy above the upper bound from range rules out that field width for the tested regime. The final scales and the D420 entries await the CPU f64 and device force measurements, the range probes, and the timings.

## Weighted sites in the segmented traversal

**Settled on PR #921.** In quanta, let $Q=q(F_s)$ be the rounded site-force component. Member $i$ has rounded share $a_i=\mathrm{round}(w_i Q)$, with ties to even. The correction is $Q-\sum_i a_i$. For a site force of $11/8$ quanta and two equal weights, $Q=1$, both shares round to zero, and the first member receives the remaining quantum.

The maintainer selected the member with largest absolute site weight to receive the residual, with ties broken by topology order. Quantize each site-force component once. Apply the site weights to that integer and round each weighted share to nearest with ties to even. Add the difference between the rounded site force and the sum of the rounded member forces to the selected member. This applies to negative weights and weights above one. The integer member forces then sum exactly to the quantized site force. Apply the rule before summing separate pairs so scheduling cannot change the result.

For $k$ weighted members, the unrounded shares sum to the integer site force. Each share rounds by at most half a quantum, so their residual is bounded by $k/2$ quanta per component and site, including for signed weights. At scale $S$, this is $k/(2S)$ in force units, or 1.5 quanta for a three-atom frame. If the selected member is a distance $r$ from the molecular center, the torque from one corrected component is at most $rk/(2S)$. Simultaneous corrections in three components give the vector bound $\sqrt{3}rk/(2S)$.

Include the residual alongside pair rounding when selecting the scale and checking the physical-particle range. For each member and component, sum the applicable bounds over its incident site contributions in a step before combining particles in the RMS budget. Include the rounding of each weighted share separately from the residual on its recipient. For the torque budget, sum the applicable lever-arm bounds over those contributions. The equilibrated NVE drift gate tests the resulting scale.
