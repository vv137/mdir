# D[towers-virial]

Recorded at its merge.

**Reduce the cost of the fused TOWERS force and direct virial traversal on
pressure-coupled steps.** Issue #1039.

The fused traversal must retain the certified integer force words. Its
virial contains each nonexcluded pair within the cutoff once, and the
topology terms supply their own contributions. The control file, output
formats, and pressure coupling schedule retain their existing behavior.
