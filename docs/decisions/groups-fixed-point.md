# D[groups-fixed-point]

Recorded 2026-11-19.

Proposed for #917. The force reductions of the groups and segmented traversals share one binary scale per evaluation. The runtime chooses the layout from the precision and the D420 device policy. The control file, checkpoint format, and output paths keep their existing interfaces.

## Pair forces and range

Round each Cartesian component once, to nearest with ties to even, before summing across pairs. For a component that rounds to the integer $k$, the other endpoint receives $-k$. Reserve a symmetric signed range so negation always fits. Nonfinite values and out-of-range conversions set a device flag before conversion. An errored evaluation stops with the particle and component. Decode into the pair term's destination once and preserve contributions from other terms.

**Outer-list degree bound (maintainer ruling, November 19, 2026).** Once an outer build succeeds, count each set bit at both endpoints. Store the largest degree on the device. Every inner list is a subset of its outer list, so the maximum outer degree bounds all inner degrees until the next build. At pruning the warp only writes the new masks. The debug check independently walks the outer masks on the host and compares every count, including padded places, and the maximum.

Let $M$ be the maximum magnitude of a rounded pair-force component at this evaluation and $B=2^{b-1}-1$ the bound of a symmetric signed field. For unit weights, the triangle inequality bounds every partial and complete per-particle sum by $d_{\max}M$. The warp reduces component magnitudes locally before updating the evaluation maximum. Compare the maximum with the device-resident bound before publishing the decoded forces. The corresponding site bound is $\max_i(A_iM+C_i)$, with the absolute-weight coefficient and rounding allowance accumulated over the outer list. Modular addition makes a short-range sum independent of the order of identical integer contributions. In mixed precision the PME charge spread uses floating-point atomic additions. Two runs from the same Large TIP3P checkpoint already differed in kinetic energy at step 30100. The fixed-point path alone does not make trajectories bitwise reproducible.

With the earlier outer-list certificate at the trial mixed scale of 4096, the check refused the equilibrated large TIP3P checkpoint. The outer list of zero-based particle 309873 contains 705 distinct partners (18 as group member and 687 as entry). Its quota is 3,046,076 quanta. Of 303 distinct partners evaluated inside the cutoff, particle 309859 gave the largest absolute z component, 4,650,232 quanta. The signed z sum of the 303 contributions was −3,453,987 quanta. This is a conservative range refusal for a representable sum.

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

## Disposition of the prototype

The maintainer decided to close #921 without merging. The measured Large TIP3P relative force error against the CPU reference at scale 64 was $1.3\\times10^{-4}$, above the mixed target. The estimated step-time improvement ceiling was 5–6%. The planned November 21 timing comparison was canceled before it ran.

The prototype counts both endpoints after a successful outer build. The JAC recount agreed with the host at all 24,832 places (9,568,342 endpoints). Omitting the group endpoint gave zero at place 0 where the host counted 90. The earlier inner-mask design counted member bits during pruning and gathered entry populations through a reverse index. On Large TIP3P the gather alone cost about 604 µs per pruning and the reverse-index fill about 9.3 ms per outer build. With that design, fusion reduced the prune and force kernels from 729.8 to 697.3 µs per ordinary pruning at trial scale 1024.

The intended scale policy selects a power of two at each outer build and checks the integer inequality $d_{\\max}\\,\\mathrm{roundEven}(sF_{\\mathrm{cap}})\\le B$ for unit weights. The cap margin is at least three times the largest observed component. Even the interim observation of 5706.77 kJ mol⁻¹ nm⁻¹ requires a binary cap of at least 32768. At that cap the initial outer maxima of 469, 543 and 774 give mixed scales of 128, 64 and 64 for JAC, Cellulose and Large TIP3P, respectively.

The archived implementation still uses a fixed production scale. Its uncached GPU regression injects a higher speculative scale, rejects that evaluation on the device, clears its private force accumulation and replays at the ordinary scale. The forces agreed bitwise with the unfused evaluation. Omitting replay produced zero in place of the expected +1 and −1 quanta. Production scale selection and its final timing gates were outstanding at closure.
