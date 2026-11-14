### Changed

- **A neighbor structure's type states what its build guarantees** (D[neighbors-contract], #881, part of #877). `!mdrt.neighbors` carries the traversal (`directed` or `unique`), the images, the encoding of the exclusions, and the organization of its lists (`grouped_by`, `stable`, as in D409), and the verifier refuses a loop over pairs, or a build of triplets or edges, whose requirement the type does not meet. `md_exec.weaken_neighbors` weakens a guarantee. No program changes.
