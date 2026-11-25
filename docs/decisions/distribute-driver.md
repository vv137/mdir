# D[distribute-driver]

The CPU control-file driver lowers the fixed-cell velocity Verlet program
onto domains. The potential rewrite supplies the force and energy kernels.
The step state carries positions, velocities, forces, masses and the real
per-particle parameters used by the force. The driver retains its output
schedule. At a checkpoint segment that reorders particles, the domain loop
starts after that permutation. Collected rows use the same particle-set
identity as the other inputs to the output calls.

The keys and default choices follow Section 13 of
[ML2](../distributed-ml2.md). The acceptance comparisons are CPU NVE with
finite-range LJ and a truncated Coulomb/LJ mixture, with both mixed and
double output storage. The reference executor computes its domain kernels
in double. Fixed-grid repeatability and agreement with a different grid
are separate comparisons because each grid determines a floating-point
sum order.
