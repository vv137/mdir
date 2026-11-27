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

For midpoint vector loops whose force and observable sums all permit
reordering, each chunk of center rows accumulates into its own force
buffer. The scratch contains both owned and ghost endpoints. A second
loop sums each destination in chunk order before halo return. This avoids
a scalar atomic addition per active neighbor and force component.

In a native reporting interval the plain inner steps and the subsequent
reporting step share a Verlet epoch when their particle set, cell,
constraint relations, halo reach, and halo construction mode match. The
inner loop completes before the reporting step uses its cache. The same
displacement test runs at both sites. An invariant exclusion relation
has one directory version within its common loop scope, so alternating
the two sites preserves its projection on a live halo.
