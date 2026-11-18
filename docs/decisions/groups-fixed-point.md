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

The maintainer selected the member with largest absolute site weight to receive the residual, with ties broken by topology order. For each Cartesian component round the site force and each weighted force to integers at the same scale. Add the difference between the rounded site force and the sum of the rounded member forces to the selected member. This applies to negative weights and weights above one. The integer member forces then sum exactly to the quantized site force. Apply the rule before summing separate pairs so scheduling cannot change the result.

The requested error budget is one force quantum per component and site for the correction. A correction along one component contributes at most its magnitude times the selected member's distance from the molecular center to the torque magnitude. Include this term alongside pair rounding in the scale study and check the equilibrated NVE drift. The one-quantum requirement needs a further choice for general frames. Four weights of one quarter and a component of two quanta give four exact halves. Ties-to-even rounds all four to zero and the selected member receives two quanta. If all three component corrections are each bounded by one quantum, the vector torque bound also requires their Euclidean norm, which can reach the square root of three times one quantum.
