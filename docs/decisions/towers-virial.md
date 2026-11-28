# D[towers-virial]

Recorded at its merge.

**Reduce the cost of the fused TOWERS force and direct virial traversal on
pressure-coupled steps.** Issue #1039.

The fused traversal must retain the certified integer force words. Its
virial contains each nonexcluded pair within the cutoff once, and the
topology terms supply their own contributions. The control file, output
formats, and pressure coupling schedule retain their existing behavior.

## Exclusions and reduction

Each target atom has an undirected exclusion window anchored at the least
of its own ID and its excluded partners' IDs. The first 31 partner IDs in
that window have exact bits. The high bit records whether there are
excluded partners beyond the window. A worker loads the anchor and mask
alongside the target fields it already carries across a record. A source
in the window needs a bit test. Outside the window, a clear high bit
proves that the source is physical. A set high bit uses the existing
forward index and sorted exclusion list. Both paths test membership before
adding the pair's virial. This preserves the exclusion of even very large
or nonfinite provisional interactions.

With $\mathbf d$ the periodic displacement and $\mathbf F$ the pair
force, the direct term is the sum of $\mathbf d \otimes \mathbf F$
over physical pairs inside the cutoff. The reduction keeps the binary64
thread sums, reduces each warp, then reduces the warp sums in one warp.
The block can reuse the tile's shared force planes after their last use.
It publishes one atomic addition per populated tensor component. The
exclusion blocks continue to cancel integer force words and conversion
votes in their original arithmetic.
