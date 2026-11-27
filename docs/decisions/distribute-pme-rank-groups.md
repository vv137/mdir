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
With disjoint ranges, a cyclic launcher rank order can spread the mesh
ranks evenly across nodes. On four nodes in mixed Cellulose NVT at 2 fs
with 24 particle and eight mesh ranks, spreading two mesh ranks per node
reduced the mean step time from 70.705 to 62.302 ms over steps 48–96.
Both arms used eight eight-thread ranks per node with core binding.

[Hess2008]: ../references.md#hess2008
