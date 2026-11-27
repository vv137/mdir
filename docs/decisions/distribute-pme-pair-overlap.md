# D[distribute-pme-pair-overlap]

Recorded at its merge.

**Overlap reciprocal PME with local pair forces on disjoint MPI ranks.**

After the charge exchange, dedicated mesh ranks execute both pencil FFTs
and the reciprocal convolution while particle ranks evaluate their local
midpoint force rows. The local pair loop and the pencil collectives use
disjoint ranks, following the separate reciprocal-space rank schedule of
[Hess2008]. The full-transport observable and potential exchanges take
place after both phases. The observable reduction still visits the
same rank contributions in communicator order.

The compiler uses this schedule when the mesh operation is immediately
followed by a local pair-force operation and the mesh outputs are first
used after it. A shared-rank mesh call completes synchronously.
The force and charge rounding rules apply to both schedules.

[Hess2008]: ../references.md#hess2008
