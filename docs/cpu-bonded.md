# Topology transfer in the Cartesian CPU reference executor

D[cpu-bonded]. Implementation in progress; depends on D[cpu-temporal].

`mdir-cpu-lj --bonds=FILE` adds harmonic terms to the existing LJ calculation.
Each nonempty file row is `id_i id_j k r0`; the potential is
$U_b=\tfrac12 k(r-r_0)^2$. IDs must exist, endpoints must differ, unordered
pairs must be unique, and `k` and `r0` must be finite and positive. Distances
use the existing fixed orthorhombic minimum-image convention, independently
of the LJ cutoff. No automatic exclusions or 1-4 scaling are introduced.
This is an explicit additive test Hamiltonian, not a molecular force-field
import format. No production TOML keys or output files change.

The owner of the smaller endpoint ID evaluates each bond once. A separate
ID-indexed topology map gathers its remote endpoint, even beyond the spatial
halo. Forward positions and reverse force contributions use the same epoch's
route. The larger endpoint's contribution returns to its state owner; local
contributions are added locally. The final force is not ready until both LJ
center subsets and all bonded contributions are applied. Migration rebuilds
the owner directory, local bond indices, and topology routes before reuse.

The first implementation replicates the bond table and gathers the ID-owner
directory at epoch construction. It uses a generated CPU bond kernel and
separate per-bond output buffers to avoid concurrent endpoint scatter races.
This is a correctness implementation, not a scalable directory algorithm.
The existing restricted `md_dist` plan will explicitly schedule topology
transfer, bond dispatch, and reverse completion; general semantic contribution
verification remains a separate task.

Validation covers mixed/double, synchronous/nonblocking transport, 1/2/8
ranks, remote partners outside the LJ cutoff, repeated bond endpoints,
periodic boundaries, moving ownership, and invalid input. The independent
oracle checks energy, forces, virial, finite differences, and final NVE state.
Angles, dihedrals, constraints, production topology import/exclusions, and
GPU transport remain unsupported.
