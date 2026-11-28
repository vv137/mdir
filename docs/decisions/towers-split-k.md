# D[towers-split-k]

Recorded 2026-11-28.

Issue #1043.

`[execution.towers_shape].split` accepts 1, 2, 4, or 8. The moving
dispatch and the setup trial use one source block per part per tower,
with the same exclusion schedule. The retry diagnostic allocates one
counter per source block of the largest candidate that uses it. Splits
4 and 8 require an explicit split pin. Other unpinned launch fields
remain eligible for the setup trial. AUTO proposes splits 1 and 2.
The measured 2 fs sweep favored split 1 or 2 on whole-step rate and
force-plus-finish time in every case.

At split $k$, part $p$ takes chunks $p, p+k, \dots$ of 32 sources. The
chunk assignment from D[towers-split-interleave] gives every logical
source exactly one owner. Its mapping is increasing within a part, so
the home-target range remains contiguous in part-local indices. Every
block stages its target plate independently. Larger splits reduce the
work on the longest-running blocks at the cost of more plate staging.

The same Newton pair contributes the same two fixed-point force words
under each split at a fixed height. The partition changes the order of
integer additions. The same published words result when the sums fit,
including after certified replay.

## Fixed-split measurements

The RTX 4090 sweep uses Trp-cage ff19SB+OPC, Trp-cage ff14SB+TIP3P,
TIP3P 10k, JAC, and Cellulose in both NVT and NPT at a 2 fs timestep. Each case starts from
the same frame and keeps its calibrated height, thread count, residency,
panel, stripe, record capacity, pack, and screen across all four arms.
Three repeats run in ascending split order and three in descending
order. A repeat pairs all four splits on the same device. The
whole-step rate uses the latter half of 6,000 steps, or 2,000 for
Cellulose. Separate 1,000-step traces measure the direct-force launch
span and the sum of the force and finish times per step.

Instrumented runs count screen passes by source part and time source
and target staging through the staging barrier. The sum of block
staging times measures staging work. Blocks can overlap in time, so the
force launch span comes from the uninstrumented trace.

On Trp-cage ff19SB+OPC NVT, the six matched RTX 4090 repeats gave
force launch spans of 129.00 ± 3.63, 67.86 ± 0.93, 71.24 ± 0.88,
and 75.73 ± 1.58 µs for splits 1, 2, 4, and 8, respectively. Split 2
was fastest and took 26.66 µs longer than the 41.2 µs even-spread
estimate in #1043. Whole-step rates were 892.8 ± 19.6, 1251.8 ± 46.6,
1186.6 ± 84.4, and 1166.2 ± 71.1 ns/day. The values give means and
sample standard deviations.

## Persistent work queue

The prototype builds one queue item per tower and 32-source chunk.
Persistent blocks claim items with an atomic ticket. On Trp-cage
ff19SB+OPC NVT at 2 fs, six RTX 4090 repeats in opposite arm orders gave
102.12 ± 3.73 µs/step for queue construction, force, and finish, versus
70.01 ± 1.63 µs/step for force and finish at split 2. Queue construction
took 2.26 ± 0.10 µs/step. The force launch spans were 96.75 ± 3.52 µs
and 66.52 ± 1.53 µs. The whole-step rates were 947.62 ± 21.30 ns/day
and 1146.42 ± 24.64 ns/day. The values give the mean and sample standard
deviation. The measured slowdown ruled out the persistent queue.

## Validation

On an 8,602-particle Trp-cage ff19SB+OPC frame, each split produced
4,136,552 direct pairs and 25,806 identical force words on an RTX 4090.
The exhaustive closest-image oracle used binary64 outside a $10^{-4}$ Å
shell and independently reconstructed the binary32 staged displacement
and D159 cutoff for the 177 pairs in the shell. All four splits had
zero missing, extra, or duplicated pairs. The strict binary64 cutoff
alone includes five additional pairs in the shell.

The GPU regression checks all four splits against periodic all-pairs
force words on a sheared frame, with packed and binary64 geometry, both
insertion orders, and certified replay. The selector regression checks
that both record capacities are timed with split 4 and split 8.
