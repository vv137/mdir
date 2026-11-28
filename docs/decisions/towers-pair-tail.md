# D[towers-pair-tail]

Recorded 2026-11-28.

**On the certified GPU path of the TOWERS force loop, a central pair region converts its force with one radial factor and one range test. The loop selects the factor, not the three components, for a rejected candidate. On the moving dispatch it multiplies the factor by the power-of-two word scale before the displacement. One sum of squares screens the three scaled components, and the six endpoint comparisons run only when the sum is not below $2^{62}$. Accepted words and failure votes are those of the component tests.** Issue #954 (force-only kernels: shared subexpressions after differentiation), after D430.

## 1. The account of the pair loop

Nsight Compute counters are not available to the jobs on the RTX 4090 nodes (the driver restricts profiling to administrators). The account therefore uses the SASS of the production kernel and ablations of the integrated kernel: each variant removes one piece of work and gives wrong forces, and only its kernel time is read. The times are means of three passes in alternating order, Nsight Systems, 299 force-only launches each, RTX 4090, pinned launch shape.

| Variant | JAC (µs) | Trp-cage ff19SB+OPC (µs) |
|---|---:|---:|
| Main force kernel | 32.9 | 100.5 |
| No source-endpoint shared atomics | 27.0 | 89.9 |
| No range test of the scaled components | 29.3 | 91.1 |
| No pair evaluation (fixed words, all other work) | 20.9 | 53.7 |

- JAC: 256 threads, 80 registers (the kernel is compiled for three resident blocks), 20.5 KB static and 4.6 KB dynamic shared memory, no local memory.
- Trp-cage: 512 threads, one block per SM, 128 registers, 47.6 KB static shared memory, no local memory. Other launch shapes with more resident warps were slower: 256 threads at two blocks took 163 µs, 512 threads at two blocks 187 µs (64 registers with stack spills).
- The candidate loop of JAC has 145 SASS instructions on its static path: 12 shared atomics (three for the source endpoint, three at a target change, six for the rare coarse quotient), six shared loads, one global load of the radial cubic, one reciprocal. The six comparisons and the predicate merge of the range test, three selects and six multiplies formed the conversion tail.

The largest item, the source-endpoint atomics, has no change here. The three components of one endpoint go to three 32-bit words. A 64-bit shared add would carry two of them, but on this architecture it compiles to a compare-and-swap loop, and the sources of a warp's lanes differ, so a warp-level sum does not apply.

## 2. What changes

A pair region is central when its force yield is `arith.mulf` of `vector.broadcast` of an f32 factor $s$ and the displacement argument. On the certified GPU path (direct pairs, not exclusion cancellation or wide replay):

1. A rejected candidate selects $s = 0$ instead of three zero components. The displacement is finite, so each component is $\pm 0$ and converts to the word 0 with no vote, as before.
2. The moving dispatch passes the word scale $w = 2^{18}$ (`forceWordScale`). The loop computes $x_a = d_a\,\mathrm{fl}(s w)$. Multiplication by a power of two is exact unless it overflows or leaves the normal range, so $\mathrm{fl}(d_a\,\mathrm{fl}(s w)) = \mathrm{fl}(\mathrm{fl}(d_a s)\,w)$, the word of main. Other entries keep $(d_a s) w$.
3. $n = \mathrm{fl}(x_1^2 + \mathrm{fl}(x_2^2 + \mathrm{fl}(x_3^2)))$ with fused multiply-adds. Rounding to nearest is monotone and the addends are nonnegative. A component of magnitude at least $2^{31}$ therefore gives $n \ge 2^{62}$, and a NaN or an infinity gives a NaN or an infinity. So $n < 2^{62}$ implies that every component lies in $(-2^{31}, 2^{31})$, and the conversion is accepted. Otherwise the six comparisons $-2^{31} \le x_a < 2^{31}$ decide, which keeps $-2^{31}$ accepted and $+2^{31}$ and NaN rejected.

The words and votes are those of main whenever $\mathrm{fl}(s w)$ is finite. If it overflows ($\lvert s\rvert > 2^{110}$), a component becomes an infinity, or a NaN where $d_a = 0$, and the pair fails conversion. Main fails such a pair too unless every $\lvert d_a s\rvert < 2^{13}$, which needs a distance below $2^{-97}$ nm. The change can therefore only add a vote, and an outstanding vote selects the exact wide replay, so the published force is the same. The exclusion cancellation keeps the product order of main, so its debit matches the direct vote under the same condition.

## 3. Validation

To be completed with the measurements of the pull request: force words against main, forces against the double references, NVE drift, and timings.
