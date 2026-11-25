# D[distribute-pair-lowering]

Recorded at its merge.

The default distributed CPU driver sends the owned and explicit-image
ghost rows of each rank-local domain through the ordinary MATRIX builder
and host pair lowering. Independent kernels on the same neighborhood
share the local coordinates, gathered parameters, and candidate list.
The host policy chooses the lane width from the computed precision.
OpenMP divides the center rows among the threads of the rank.

The halo owns the list. Reuse requires the same rows and a displacement
within half the pair skin for every row. A migration creates a new halo.
Unique and midpoint traversals filter by global identity. Midpoint
ownership is evaluated at the current positions even on a list hit.
The two-ended force traversal adds the exchanged contribution to each
neighbor and returns ghost contributions to their owners.

Deterministic distributed runs retain the canonical image calculation,
Q40 force terms, Q32 observables, and checked wide sums of ruling B.
