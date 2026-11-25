# D[bonded-exclusion-fusion]

Recorded at its merge.

Launch exclusion corrections and packed valence work in separate block
ranges of one grid for eligible mixed force-only steps. Short exclusion
lists use direct atomics. Long lists reuse the valence blocks' shared-memory
layout for hash keys and force subtotals. Each block takes only one path.
The existing hash clear and flush barriers apply within an exclusion block.

This launch choice is independent of checked Q18 valence accumulation.
With separate kernels, two opposite-order RTX 4090 comparisons measured
whole-step differences from the default of 0 and +2 µs on JAC and −3 and
+1 µs on Cellulose. Both systems met the ±3 µs target in those runs.
In a matched Cellulose trace, separate kernels totaled 89.469 µs and the
combined grid took 90.953 µs. Fusion gives one recurring launch. The
measurements leave its whole-step speed advantage unresolved.

[Work-unit ownership and publication](bonded-workunits.md)
