# D[matrix-search]

Recorded 2026-11-21.

**Bound the distance from a wrapped particle to a candidate cell before reading its particles in the orthorhombic GPU neighbor matrix search.** Part of #924. The minimum periodic distances to the y and z slabs of a row bound the x interval that can intersect the search sphere. Clip the x runs to that interval with slack for rounding, then use the existing f32 distance test and exclusion handling.

The experiment compares the bound at the existing width and at a finer width. The reference row order and contents are validation requirements. The choice of width and whether to keep the bound depend on the timings. There is no new control-file key.
