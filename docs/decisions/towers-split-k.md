# D[towers-split-k]

Recorded 2026-11-28.

Issue #1043.

`[execution.towers_shape].split` accepts 1, 2, 4, or 8. The moving
dispatch and the setup trial use one source block per part per tower,
with the same exclusion schedule. The retry diagnostic allocates one
counter per source block of the largest candidate that uses it.

At split $k$, part $p$ takes chunks $p, p+k, \\dots$ of 32 sources. The
chunk assignment from D[towers-split-interleave] gives every logical
source exactly one owner. Its mapping is increasing within a part, so
the home-target range remains contiguous in part-local indices. Every
block stages its target plate independently. Larger splits reduce the
work on the longest-running blocks at the cost of more plate staging.

The same Newton pair contributes the same two fixed-point force words
under each split at a fixed height. The partition changes the order of
integer additions. The same published words result when the sums fit,
including after certified replay.
