# D[distribute-pme-rank-groups]

Recorded at its merge.

**Disjoint particle and PME pencil ranks.**

If an MPI CPU run explicitly chooses a particle domain count below the
transport size and a PME rank count equal to the remaining ranks, the
rank ranges are disjoint [Hess2008]. The first range owns the spatial
particle rows, and the second owns the mesh pencils. A spread and
gather still run on the particle side and exchange charges and mesh
potential with the second range. FFT and influence work run on the
mesh range. Native identity intervals and their collectives span the
transport. The particle and mesh stages use the existing ordered
execution schedule.

The same control fields also support overlapping rank ranges.
Comparisons reserve the same allocation and record both rank layouts.

[Hess2008]: ../references.md#hess2008
