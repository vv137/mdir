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
as its kernel. MPI partitioning reads an identity interval and a disjoint
interval of each constraint relation. The initial registration routes
group assignments to identity intervals and then to spatial owners.

A migrating group carries its center, members and original topology index.
Each rank merges incoming records with survivors in domain and identity
order. Each new row refers to an old local field slot or a peer receive
slot. Fields use the same permutation. Halo discovery distributes sites
whose periodic images can enter the receiver's boxes. Receivers assign
ghost rows in identity and image order, then send their requested source
rows to the exporters. Cell bounds, domain counts and reporting summaries
remain collective.

The MPI driver evaluates disjoint intervals of bonded topology. A temporary
directory partitions current site coordinates by identity. A topology
interval requests its own member sites, computes the interaction midpoint
and routes the tuple and its image shifts to the midpoint owner. Parameter
lookup uses the original tuple index. Constraints instead enumerate group
centers on their spatial owners. When the native integrator requests a
complete field, identity intervals assemble consecutive output ranges.

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


Layout and halo handles own their row-count and identity arrays. Lowered
epochs borrow views of these arrays for the lifetime of the handles.
A migrated layout can share frozen constraint membership until a caller
adds a group. Each rank materializes image translations only for its own
ghost rows. With one local domain, surviving identities are already in
order, so the rank sorts only incoming identities and merges the two runs.

The midpoint matrix stores three candidate classes in original neighbor
order. A componentwise coordinate interval extends half the list skin
around each owned or unshifted ghost coordinate. Applying interval
endpoints to the actual midpoint arithmetic bounds its cell. A pair whose
interval misses the domain is dropped. A pair wholly inside is tagged for
direct inclusion. Other candidates retain the per-step midpoint test.
Leaving any coordinate interval refreshes the classification, as does a
list rebuild or an exclusion change.

The midpoint pair kernel assigns runs of eight consecutive rows from each
of sixteen bands in turn. A static worker interval then visits both the
heavily populated owned centers and the shorter ghost rows. The final
incomplete cycle uses the original row order.

The CPU PME driver reports the communicator size and the number of ranks
sharing the reporting node, including runs with an explicit particle grid.
It recommends binding a rank to each NUMA domain and placing its threads
on cores with close affinity. At the same particle-domain count, the
one-node JAC step measured 38.72 ms with one rank per NUMA domain,
59.87 ms with two ranks, and 72.37 ms with one rank. The corresponding
Factor IX measurements were 127.07, 196.50, and 235.56 ms at 2 fs.

For eight equal row slices, interleaving reduced the peak selected-neighbor
count relative to an equal share from 3.16 to 1.17 in JAC, from 2.40 to
1.12 in Factor IX, and from 1.74 to 1.05 in Cellulose. The comparison sums
the peak and ideal counts separately over the same domain evaluations.

A tuple relation retains its member-to-directory plan while its SSA input
is invariant. Remote identities are sorted and requested once per peer.
On each geometry evaluation the peer returns one site per requested identity.
Workers expand the results into disjoint member slots and resolve the received
midpoint records into disjoint tuple rows. Parameter and force order are preserved.

In a floating midpoint pair loop, selection reads the current packed
explicit-image positions and discards Verlet-shell neighbors beyond the
physical cutoff plus a roundoff guard. It repeats this test at every
evaluation, including iterations that reuse a neighbor list. The pair
kernel applies its own precision-specific cutoff to the retained rows.
Deterministic execution keeps the original lane membership and sums.

Inside an MPI native integration loop, the particle state uses a separate
particle set for the rank's contiguous interval of identities. Kicks, drifts,
constraint updates, and their loop-carried copies use that extent. The layout
routes spatially owned values to interval owners for collection. Distribution
reads the same schedule in reverse. Group registration requests coordinates
for its disjoint interval of constraint tuples. Scalar reporting combines the
interval sums. Deterministic sums retain their wide integer representation
until all intervals have contributed. A complete-field consumer at an output
boundary materializes particle-set order.

Each rank reads an interval of exclusion tuples. It sends each unordered pair
to the directory rank of the smaller identity. A new halo requests its owned
and ghost identities and keeps the returned keys whose other endpoint is also
present. An immutable relation version identifies this projection to the
candidate matrix.

The midpoint candidate builder bins the explicit owned and ghost image
coordinates in Cartesian cells. Each center visits intersecting bins and
rejects identities below its own before testing distance. The final midpoint
test still chooses the owner of the pair. Constraint projections retain the
group ownership layout across a fixed-cell integration segment. Each solve
wraps current coordinates on that layout, so updated positions do not require
a fresh ownership exchange between projections.
