# D[towers-excl-tail]

Recorded at its merge.

**Compact large GPU exclusion lists and place them among the tower splits.**
When at least 128 four-row blocks can be filled, each lane processes up to
four pairs, a block width apart. The grid reserves the smaller of half the
tower splits and four splits per exclusion block for the final tower work.
The remaining tower splits precede the exclusion group. The threshold keeps
parallelism for smaller lists, which retain the tower-first, one-row layout.
The threshold and spacing are empirical choices for the benchmarked GPU.
Logical tower indices, fixed-point contributions, and replay rules are
independent of the physical block order.
