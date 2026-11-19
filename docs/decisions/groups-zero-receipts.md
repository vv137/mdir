# D[groups-zero-receipts]

Recorded 2026-11-19.

At the end of a GROUPS round, the lane holding a real entry tests its
accumulated receipts in the destination precision. If every component of
every output equals zero, it skips the entry's global additions. For a
single force vector this suppresses three atomics. Either sign of zero
passes the zero test. A NaN requires the additions to run.

The predicate applies to the direct global flush in both the standalone
and fused prune-and-force loops. Members still reduce their contributions,
and the block-local accumulation path retains its existing flush.
The receipt test uses the result of the rotation, so it needs no branch
around the interaction. Issue #917 records the exploratory timings for
the combined change. The isolated guard requires its own comparison.
There is no control-file key.

## Validation

The four-particle GPU test compares every force component against two
evaluations with unconditional receipt additions. The interacting pair has
$x$ forces $+1$ and $-1$, and the other ten components are zero. All twelve
matched bitwise, including on the CPU oracle. Inverting the receipt guard
changed one component by 1.

Mixed precision, RTX 4090, 10,000 NPT steps per run. Each job interleaved
main and the receipt guard on each input. The second job reversed both
orders. Each time is the 10% trimmed mean of stationary reporting intervals.
Every run recorded 10,001 PME gathers, including the initial force
evaluation. The final column compares the means of the two jobs.

| Input | Main / guard, forward (ms/step) | Main / guard, reverse (ms/step) | Step time reduction |
|---|---:|---:|---:|
| JAC | 0.16095 / 0.15657 | 0.16148 / 0.15798 | 2.45% |
| Cellulose | 1.23017 / 1.19395 | 1.24610 / 1.20700 | 3.04% |
| Large TIP3P | 1.00678 / 0.97230 | 1.01243 / 0.97633 | 3.50% |

The lowered-program corpus covers 277 controls and 1,106 cases per
architecture. On both sm_89 and sm_120, 1,018 outputs matched the base
(including 488 identical diagnostics). The same 88 GPU cases differed.
Re-emitting all 88 on each architecture and comparing the embedded kernel
images separately showed identical host IR, kernel names, and every
kernel image outside the pair loops.
