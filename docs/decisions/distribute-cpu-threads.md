# D[distribute-cpu-threads]

Recorded at its merge.

CPU domain kernels visit the rank's local topology tuples in stable domain
order. Each tuple retains its original index for parameter lookup. Fixed
chunks of a domain's tuples compute independent energy subtotals, combined
in chunk order. Parallel force kernels stage the member forces and gather each
destination's contributions in topology and member order. A serial force kernel adds directly to each member row in that same order,
skipping the temporary member-force buffer and both incidence passes.
Constraint groups
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

The halo plan stores the peer stream offset of each crossing row.
Packing and forward delivery write disjoint rows. The transpose groups
the existing stream by destination, preserving its order within each
owned row for both floating-point and checked wide sums.

Within a Verlet list, identities and exclusions determine the static
candidate rows. The midpoint test still uses the current coordinates on
each evaluation, since a midpoint can cross a domain face before the list
is rebuilt. Selected entries retain their original list order.

An epoch remains valid only if the maximum squared displacement stays
within the existing skin bound. Independent particle checks reduce by
maximum. Sorted exclusion keys live across serial steps while their
topology relation is invariant. A permutation establishes a new relation
and a new sorted table.

The mesh uses independent particle blocks to mark occupied planes and bin
fixed-point charges. A plane visits the bins in particle order. Overlap
packets have prescribed offsets for their periodic intersections. Fourier
lines have private work buffers and reuse pocketfft plans of the same
dimensions. Spectrum planes form separate energy and virial subtotals,
which the rank combines in increasing plane order.

Migration initializes owner and constraint metadata by contiguous blocks
within the same parallel region that assigns the new owners. Each worker
first writes the output pages of its block. Redistribution and particle
maps assign independent domain rows to workers, including the kick and
drift maps. Each domain builds its own tuple-force incidence lists in
topology order. The rank compacts their destination offsets in domain order.
For a two-member tuple the owner midpoint uses the minimum-image distance
from the lower identity, which also determines both periodic images.
A transpose gather writes each active destination component
once after summing its incident ghost entries in stream order.

Ranks gather ownership changes as one packed stream in rank order when
MPI counts and offsets fit. A second gather shares the new domain row
counts. Each packed observable exchanges all domain partials in a single
collective and adds them in domain order. One-rank halo transfers use the local copy and transpose kernels
directly. Mesh ranks begin the charge exchange before processing the
subgrid intersections they own.

With MPI transport and no explicit particle grid, the driver uses the
communicator size as the particle-domain count and logs the choice.
The launcher determines process placement. For threaded PME the launch
recommendation is one rank per NUMA node, with its OpenMP threads bound
within the node. An explicit domain count or factor grid overrides the
particle grid independently of the mesh pencil count.

For distributed PME, each domain uses one OpenMP team to find its charge
footprint and another to bin particles and spread charge. A single worker
scans the block counts in order. Each output plane has one writer. Both
teams use a particle-row threshold of 512. Mesh overlap transfers execute
serially below 128 intersecting rows.

Particle halos pack one contiguous stream per nonempty peer and field
direction using row offsets assigned when the epoch first uses the
schedule. Each rank posts receives and sends before copying local ghosts.
For a force transpose, it sums destinations with only local contributions
during transfer and completes the other destinations after the wait. Each
destination retains the original order of its contributions.

The MPI lowering packs consecutive forward copies of the same halo into
one peer message. An intervening distribution can produce the next field.
A consumer or a new halo ends the batch. The packet holds the components
of each field in program order at each scheduled row.
Return packets can span independent tuple-force producers until a result
is first read. Every component still gathers the original stream for its
owned row, including checked wide-word additions in deterministic mode.
