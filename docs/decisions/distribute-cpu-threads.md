# D[distribute-cpu-threads]

Recorded at its merge.

CPU domain kernels visit the rank's local topology tuples in stable domain
order. Each tuple retains its original index for parameter lookup. Fixed
chunks of a domain's tuples compute independent energy subtotals, combined
in chunk order. Force kernels stage the member forces and gather each
destination's contributions in topology and member order. Constraint groups
write disjoint rows and combine their convergence flags with bitwise OR.

A snapshot used to distribute a field copies the particle identities of the
local domains. A tuple parameter snapshot uses the same local tuple order
as its kernel. Partitioning a replicated input still reads the complete
position field. Distributed field copies and zero fills visit only the
local domain range.

During migration, each rank evaluates coordinates and constraint centers
on their previous owners. Changed domain assignments update the replicated
identity map. Field payloads travel to the receiving ranks. Each rank
numbers its new local identities in ascending order using fixed-block
counts and prefix offsets. Halo construction can request the remote row
maps and positions after the migration.
