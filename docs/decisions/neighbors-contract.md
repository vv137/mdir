# D[neighbors-contract]

Recorded at its merge.

**The layout contract of a neighbor structure is a parameter of `!mdrt.neighbors`: what its build guarantees about multiplicity (traversal, images, exclusions) and organization (`none`, `grouped_by(G)`, `sorted_within(G, K)`, `stable(G, K)`, as in D409). Each consumer states its requirement from its policy and its algebra, and the verifier refuses a consumer whose requirement the type does not meet. Cost choices stay attributes of the build. `md_exec.weaken_neighbors` weakens a guarantee. No program changes.** Issue #881, step 1 of #877 (the maintainer's rulings of 2026-11-14), part of #558.

Draft: see the pull request for the design under review.
