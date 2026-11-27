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
is rebuilt. Deterministic execution retains the original list order.

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
For comparison, `MDIR_DIST_HALO_OVERLAP=0` waits immediately after
posting the same peer streams, then visits the same two sets of rows.

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
Workers read the requested sites through their planned slots and resolve
the received midpoint records into disjoint tuple rows. Parameter and force order are preserved.

In a floating midpoint pair loop, neighbors whose midpoint ownership is
stable over the Verlet bounds form a prefix of each row. The prefix uses
a smaller Verlet skin, one quarter of the list skin. A position check
refreshes it before either endpoint can move more than half that smaller
skin. Neighbors with ambiguous midpoint ownership are tested each
evaluation. The force kernel applies its physical cutoff to every
retained pair. Both radial tests include the binary32 roundoff guard.
Deterministic execution preserves the original row and lane order.

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
to the directory rank of the smaller identity. Directory ranks own consecutive
identity intervals. A new halo requests sorted owned and ghost identities.
The replies concatenate in key order. A returned pair whose second endpoint
is absent cannot match any local candidate. An immutable relation version identifies this projection to the
candidate matrix.

The midpoint candidate builder bins the explicit owned and ghost image
coordinates in Cartesian cells. Each center visits intersecting bins and
rejects identities below its own before testing distance. The final midpoint
test still chooses the owner of the pair. Constraint projections retain the
group ownership layout across a fixed-cell integration segment. Each solve
wraps current coordinates on that layout, so updated positions do not require
a fresh ownership exchange between projections.

A group layout retains the stable topology order, parameter indices, and
owned member rows for each disjoint constraint relation. Every projection
holds a reference while it uses this schedule. On a single MPI rank the
native and spatial permutations copy directly between row views, including
integer conversion or position wrapping when required.

The checkpoint execution fingerprint retains the configured
`[execution] threads` value by maintainer decision. Changing the configured
count changes that fingerprint and its dependent checksum. The physical
checkpoint datasets remain bitwise equal across team sizes.

At a fixed configuration, the energy, force, and virial kernels can use
the same read-only local tuple schedule. The lowering reuses a table only
within its defining block, for the same relation, halo, and position
field. Constraint row numbers refer to the group layout and require no
position-field match. Each kernel still reads its own current geometry.

An MPI tuple packet identifies the input tuple and its midpoint domain.
Each owned member needs only its nonnegative row. A ghost in one of
the neighboring periodic images uses a negative base-three encoding of
its three translations. The receiver reads the particle identity from
the immutable input tuple and resolves the image to its local ghost row.
If any sender needs a larger translation, all ranks use the full
identity and three signed translations for that exchange. Complemented
destination counts communicate this choice in the existing count exchange.

Midpoint selection uses the exact binary64 coordinate thresholds for
entering and leaving each domain along each axis. Bisection over ordered
binary64 numerators evaluates the original division, wrapping and cell
index at each probe. The resulting half-open interval serves both the
Verlet membership bounds and the individual tests of ambiguous pairs.

Reporting consumers in a block share the complete view of each distinct
native field. Several constraint relations can therefore read the same
positions, velocities, or forces after a single interval collection per
field. A new field value or a different execution block has its own view.

Each reported native particle sum gathers one value per rank for each
observable component in a single collective. The fold still visits
ranks in communicator order, including the checked wide-word addition
for a deterministic observable.

The driver distributes an immutable field without a periodic image once
per field and layout within a block. When both operands are invariant
over an enclosing step loop, their owned rows are available before
that loop. Distributions of evolving fields still follow their inputs.

Disjoint constraint sets read the same uncorrected particle field at
their own members. The distributed driver applies D83 while the native
tuple operations still carry the union proof. This lets multiple group
projections read a single distributed copy of the input.

Mixed PME spread multiplies the rounded charge weights along the second
and third stencil axes in binary32. Mixed interpolation uses binary32
products of inverse grid points, weights, and derivatives. The rounded
product enters the component sum in stencil order. The final reciprocal
force still uses the original charge and box scale.

Floating constraint observations can reuse a preceding disjoint tuple
projection for the same relation and block when the observation reads
only gathered fields. Each group is evaluated on its owner and the
domain subtotals are combined in domain order. An observation that
uses an internal coordinate keeps its own geometry.

In floating-point execution, a constraint correction can be added on
its group layout when the incoming particle field already has an
unwrapped view in those owned rows. Consecutive corrections add in
the original order. Native consumers collect the sum at the point
where they need it. An imaged position view cannot supply the
unwrapped operand.

A rank can derive wrapped positions from an unwrapped owned field on
the same layout. If a native position field needs both forms in a
block, one distribution supplies the unwrapped rows. Wrapping writes
a separate buffer before the trajectory epoch aligns its periodic
images. The reciprocal mesh and unwrapped constraint consumers keep
the original coordinates.

The tuple geometry plan stores a slot per member. During midpoint
assignment, each worker reads coordinates, domain, and row directly
from the local directory or the unique replies for its peer requests.
Only the members of the current tuple need temporary geometry. Repeated
members of the same site share its reply across tuples.

When a floating field returns from identity order to the same owned
layout without an image change, the original owned field supplies that
consumer. A remaining native consumer still triggers a collection.
Fixed-point sums retain their explicit field materialization.

In mixed precision the owned spatial transport delivers binary64 charge
values. When every use of its isolated scalar copy is a pair input, the
copy rounds each value once to binary32. The pair kernel reads binary32
charges directly. A scalar with a native consumer keeps its assigned
precision. Explicit midpoint images still form displacements in binary64
before the force calculation rounds them.

For a Verlet interval, the exact midpoint numerator is half the sum of
the two coordinates and the rounded relative image shift. Its extrema
occur at the respective lower and upper coordinate bounds. A binary64
error margin covers both endpoint evaluation and the subtraction sequence
used by the pair kernel. A near-face or extreme-scale result falls back
to the current-coordinate ownership check. This proves stable ownership
for more pairs while preserving force coverage at a rounded face.

Immutable bonded relations in the same loop block share remote site
requests when their rows use the same coordinate snapshot. The first
schedule gathers the union of remote particle identities. Each relation
maps its members into the reply in its original tuple order. The next
snapshot refreshes the coordinates even if the allocator reused an
address. Loop-local relations use a plan with their own lifetime.

A midpoint rebuild records the largest actual raw row. Both filtered
matrices need at most that many entries per row. A later rebuild can
grow them before generating new candidates. The general neighbor builder
continues to supply its allocated width when it has no measured maximum.

A retained partition used only for disjoint constraint groups and owned
fields assigns each group to the native identity interval of its first
member when each rank has one domain. Ungrouped atoms keep their native
owner. A physical halo, migration or other spatial use requires spatial
ownership. The original identity carries every solved field back to its
caller, including members crossing an identity interval boundary.

When each rank owns one spatial domain and the real PME pencil grid
subdivides the physical x and y grid, assign each column of physical z
ranks the real pencils inside its x/y tile. Apply the same rank
permutation to all three FFT layouts. Partly occupied or incompatible
pencil grids keep their original assignment.

The midpoint membership filter copies current unshifted positions while
checking the existing Verlet coordinate bounds. It stores each row's
periodic shift on classification refresh. The ambiguous-neighbor scan
reads these two values from the same row record and evaluates the original
binary64 midpoint expression and half-open domain test in neighbor order.

The mixed gather computes the independent derivative products along the
contiguous stencil axis as SIMD lanes. At a periodic boundary, the
potential values come from the tail and head of the same mesh row.
Each component still adds the rounded products in stencil order.

The mixed Gaussian spectrum forms each axis factor once per local mode
index and combines the rounded factors in binary32. The complex
influence multiplication also uses binary32 inputs. Mode powers,
virial terms, and the ordered plane sums keep their existing precision.
The prolate radial factor still depends on the full wave vector.
