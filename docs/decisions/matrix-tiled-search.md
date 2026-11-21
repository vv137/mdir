# D[matrix-tiled-search]

Recorded 2026-11-21.

Compare the existing GPU matrix search (A, one target particle per warp)
with a cell-cooperative scalar search (B). One block owns a reference
cell. Each warp owns up to sixteen consecutive targets, and all warps
share batches of eight source positions loaded into block memory. Dense
target cells are processed in successive block-wide batches. Source
cells and particles retain the baseline traversal order. The scalar
test uses the same minimum-image calculation, rounding margin, and
strict comparison as A. Exclusions occupy the same self-sentinel slots.

Measure B against A on the same rebuilds, checking every ordered row.
The comparison includes launch geometry, register use and row writes as
well as load reuse. Measure the distance-test share before deciding how
much C (an MMA screen with the exact scalar recheck) could save. There
are no new control-file keys.
