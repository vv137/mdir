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

Log every candidate grid score, selected geometry, import size and source
of the traversal and transport. Sample occupancy, migrations, directed halo
bytes and phase times at the energy cadence. Record optional rank traces
under the prefix in `MDIR_DIST_STEP_TRACE`. Checkpoints keep their physical
format and atomic commit. Domain and ghost records belong to the manifest.
Initial draws are indexed by particle ID at step zero. Bitwise continuation
across grids remains open because pair and partial sum orders vary.
