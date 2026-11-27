# D[distribute-tuple-cost]

Recorded at its merge.

**Assign eligible bonded tuples by measured cost.**

The floating-point MPI midpoint driver times bonded kernels separately
from tuple routing and constraint projection. At a Verlet rebuild it
uses those timings to pair neighboring domains above and below the
mean cost [Phillips2020]. A donor offers a stable sample of its topology
indices to one recipient. The fraction grows with the observed cost
difference and is capped at one quarter of the donor's tuples.

A tuple moves only when the recipient covers every explicit member image
with the existing halo. The coverage test reserves a full skin and
roundoff margin for movement since the halo was built. Each recipient is
paired with at most one donor. The existing topology order, force
transpose, and constraint owners apply on the resulting layout.
Fixed-point trajectories use midpoint tuple ownership.

[Phillips2020]: ../references.md#phillips2020
