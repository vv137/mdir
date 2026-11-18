# D[groups-fixed-point]

Recorded 2026-11-19.

Proposed for #917. The force reductions of the groups and segmented traversals share one binary scale per evaluation. The runtime chooses the layout from the precision and the D420 device policy. The control file, checkpoint format, and output paths keep their existing interfaces.

## Pair forces and range

Round each Cartesian component once, to nearest with ties to even, before summing across pairs. For a component that rounds to the integer $k$, the other endpoint receives $-k$. Reserve a symmetric signed range so negation always fits. Nonfinite values and out-of-range conversions set a device flag before conversion. An errored evaluation stops with the particle and component. Decode into the pair term's destination once and preserve contributions from other terms.

For an integer field of width $b$, let the symmetric limit be $2^{b-1}-1$. A bound on each individual addition alone cannot detect overflow of their sum. One candidate guard counts incident candidate pairs for each endpoint at build time, including every site-pair contribution a masked unit pair can generate. Divide the symmetric limit by that count, rounding down. Require the rounded magnitude of every contribution to fit the resulting bound for both endpoints. The triangle inequality then bounds every partial sum in every legal reduction order. The counts must cover the lifetime of the outer list. A conservative refusal identifies the first recorded offending endpoint and component before its force is consumed or a checkpoint is committed. Start the flag copy after the writers, and finish it at the first dependent host boundary (D113).

For packed fields, encode the signed partial sums as $x+2^{32}y$ modulo $2^{64}$. Both the construction and accumulation use unsigned modular arithmetic. Decode the signed low word first, subtract its sign extension modulo $2^{64}$, then take the upper word as signed. Concatenating two independent bit patterns would leak a carry into the upper field. The bound above applies independently to both fields. The other layouts under evaluation are three i32s and three i64s. The floating vector alternative requires an aligned four-component destination and a supporting architecture. It is a candidate for ordinary mixed precision only.

## Selecting a scale

For scale $S$, rounding perturbs one component by at most $1/(2S)$. If particle $i$ receives at most $d_i$ contributions, the norm of its three-component perturbation is bounded by $\sqrt{3}d_i/(2S)$. Summing the squares gives the conservative relative RMS bound

```math
\Delta_{\rm quant} \le
\frac{\sqrt{3\sum_i d_i^2}}{2 S\sqrt{\sum_i \lVert\mathbf F_i\rVert^2}}.
```

With uncorrelated uniform rounding errors the corresponding RMS estimate replaces $3\sum_i d_i^2$ in the numerator by $\sum_i d_i$. The estimate guides the sweep. It does not certify a force or a range. For each precision, choose the smallest power of two that satisfies its accuracy budget and test the guard against physical pair forces and incident counts. A lower bound on scale from accuracy above the upper bound from range rules out that field width for the tested regime. The final scales and the D420 entries await the CPU f64 and device force measurements, the range probes, and the timings.

## Weighted sites in the segmented traversal

D421 distributes a site's force to physical particles through its weights. Independent rounding of the weighted outputs can change their sum. For a half-and-half site receiving one integer quantum, both outputs round to zero with ties to even. Per-record redistribution also makes the result depend on how pairs are partitioned. The proposed extension apportions the already quantized site force per pair. Round the weighted components of all but one contributing member. Give the last member the exact integer residual. Choose the member with largest absolute weight to receive that residual, breaking equal-weight ties by the topology's member order. Include the residual in the physical-particle range guard and its extra rounding error in the scale study. This choice can break equal-weight symmetry by one quantum and requires a maintainer decision.
