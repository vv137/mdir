# D[distribute-auto-balance]

Recorded at its merge.

**Automatic particle-domain load balancing.**

The floating-point MPI CPU driver with one midpoint domain per rank starts
with uniform planes. At an existing Verlet epoch rebuild, it compares the
rank-local time spent in real-space pair and other force work, after
subtracting MPI arrival waits. It uses at least sixteen step markers per
sample. It averages load across each slab and moves the planes on the
axis with the largest peak-to-mean slab load above 1.05. The target planes
divide the piecewise-constant estimate of force cost equally. Each move
takes one quarter of the estimated displacement, limited to five percent
of the original slab width per update and twenty percent from the initial
position. This follows the measured domain resizing of [Hess2008].

The plain and reporting paths share the measured load and cut generation.
A retained layout adopts a new generation before its next use.
The new layout rebuilds particle ownership, halo coverage and midpoint
ranges together. The reciprocal mesh retains its own layout and
communicates charge and potential through geometric intersections. The
run log records the choice and the measured mean arrival wait. Fixed-point
execution uses uniform planes to preserve its replay guarantees. The
fixed-cell requirement and control-file schema remain in force.

[Hess2008]: ../references.md#hess2008
