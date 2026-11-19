# D[groups-zero-receipts]

Recorded 2026-11-19.

At the end of a GROUPS round, the lane holding a real entry tests its
accumulated receipts in the destination precision. If every component of
every output equals zero, it skips the entry's global additions. For a
single force vector this suppresses three atomics. Either sign of zero
passes the zero test. A NaN requires the additions to run.

The predicate applies to the direct global flush in both the standalone
and fused prune-and-force loops. Members still reduce their contributions,
and the block-local accumulation path retains its existing flush.
The receipt test uses the result of the rotation, so it needs no branch
around the interaction. Issue #917 records the exploratory timings for
the combined change. The isolated guard requires its own comparison.
There is no control-file key.
