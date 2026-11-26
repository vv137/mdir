# D[distribute-gpu-node]

Recorded at its merge.

Stage 1 selects GPU MATRIX pairs with `[execution] target = "GPU"` and
`[distribute] transport = "nccl"`. Build with MPI and `MDIR_ENABLE_NCCL`.
Use one domain per MPI rank. The product of the domain grid dimensions
must equal the communicator size. Directed traversal evaluates both
orientations. Unique traversal returns ghost forces to the owners. Midpoint traversal needs a moving ownership filter for each
pair. Stage 1 diagnoses a midpoint or deterministic GPU request at setup.
Deterministic GPU execution requires the integer force and observable
sums of ruling B.

The default `[distribute] placement = "node"` groups domains into compact
blocks on each shared-memory node. Minimize the predicted bytes on faces
between nodes over possible rectangular blocks, breaking ties in grid
axis order. MPI shared-memory communicators supply local rank numbers.
Select the GPU by local rank when all local GPUs are visible, or use
ordinal zero when each process sees its own assigned GPU. Verify distinct
device identities within each shared-memory communicator.

Use global communicator ranks in halo and reverse plans. Exchange the
NCCL bootstrap ID with MPI. Pack device rows by peer, group matching sends
and receives, and order them with the pair kernel on CUDA streams. NCCL
selects the link between peers. Host migration can rebuild ownership and
the halo plan. Log directed face bytes per step by node locality, counting
each send once. In stage 1 the host step state supplies the local field before each
pack and consumes its result after each device unpack. Each rank runs the
MATRIX pair terms on its GPU. Bonded tuple evaluation, migration and the
outer step still use host state. MPI reduces the scalar observables.
Reverse-force bytes and time are accounted separately.
The first validation targets fixed-cell finite-range Coulomb/LJ, including
reaction field. Reciprocal-space decomposition is outside this stage.

Mixed MATRIX gathers the imaged search coordinates and their translations in
full precision. Restore the original coordinates before narrowing and subtracting
them, then apply the image translation in kernel precision, as in the host
MATRIX traversal. This preserves cutoff membership across periodic faces.

### Stage 2

The rank's packed Verlet state retains immutable field allocations. Packing or
unpacking a state changes ownership of references. Migration constructs new
fields in the new row order. For an unshifted field an unchanged ownership
map retains the original field. On a crossing, a device gather exports
the outgoing rows, MPI exchanges them, and a device kernel merges the
arrivals with the local survivors in global identity order. Each field can hold a device copy of the local
domain's rows. A kick or drift reads the device fields, writes a new device
field and leaves its host copy unmaterialized. A coverage kernel checks the
owned displacements and MPI combines the flags. Device force slices feed
force combination, subsequent kicks and owned particle reductions. A host
consumer materializes the rank's local slice on demand.

For a neighborhood whose only position consumers are pair traversals, NCCL
packs from device storage and unpacks translated ghosts directly after the
owned positions. The pair kernel borrows the resulting field. For scalar
pair parameters, each halo retains the combined owned and ghost rows by
copy site and immutable input identity. All ranks agree on reuse before
skipping the exchange. Integer particle kinds are converted on the device.
The halo releases the stored parameters with its row maps.

An exclusion-free packed Verlet epoch may reuse its device MATRIX counts,
order and entries. Relative to the reference positions that define the halo,
an individual row moves by at most half the halo margin. Between two force
evaluations in that epoch each of two rows can therefore move by a full
margin. Building candidates at the cutoff plus twice the margin covers this
relative displacement. The pair traversal applies its original cutoff.
Periodic bin keys index the search, and explicit imaged coordinates determine
candidate distances. This keeps a second periodic copy of a particle out
of the list when that particular image lies beyond its reach. The halo owns
the entry allocation across temporary force evaluations and releases it
when its row map expires.

When the grid cells are wider than twice the halo reach, an unconstrained
packed Verlet rebuild classifies owners and faces on the device. A row
farther than the reach from every face can reach only its owner. A row
within the halo reach of a face can also belong to the neighboring
domain's halo. For edges and corners the squared distance is the sum
of the squared distances to the participating faces. The GPU reports
only rows that change owner or qualify for a neighboring halo. The host
exchanges identity, owner and direction bits, then orders the rows by
identity and periodic image. The runtime wraps physical positions on
the device and retains immutable numbering when no ownership changes.

With one domain and directed double-precision pair geometry, ownership is
fixed. A device pass counts periodic images per block of 256 owned rows.
The host prefix sum of the block counts assigns disjoint output ranges.
A second device pass emits source rows and image translations in the same
identity and image order as the general halo. Forward copies gather
periodic images directly from local device memory.

For other grids and constrained systems the host plans ownership and halo
membership from coordinates. A topology schedule shares positions for its
member identities. Host tuple kernels gather the owned and ghost rows they
read. An immutable field records which rows the host has already received.
The native constrained integrator and output can still materialize global
fields at partition and collection boundaries.

### Overlap of position exchange and directed forces

For a cached, exclusion-free directed MATRIX with multiple domains, the export map divides the centers into two sets. Each
source of a remote or periodic ghost is a boundary center. All other
owned rows are interior centers. At the rebuild an interior center is
farther than the halo reach from the faces. The two half-skin displacement
allowances together bound how much closer it and a remote neighbor can
move before the next rebuild.

The communication stream waits on the position producer's event, packs
outgoing triples, exchanges them with peers, scatters translated ghosts,
and records completion. The compute stream can run the interior force
rows using a separate buffer with owned positions and initialized padding.
This keeps the cell-order gather from reading incoming ghost storage.
The boundary pass waits on the completion event before reading the full
position field. Both passes use the epoch's original MATRIX entries and
write disjoint owned force rows. A cold list requires complete geometry
for its search. The two partial observable sums are added in interior, then boundary order.

### Ownership over a Verlet epoch

After drift, the packed integrator checks the displacements against the
reference positions that defined its halo. While the reference covers
the current positions, it carries the same layout, halo and reference
into the force call. A face crossing within that margin changes the
position sent by the original owner. When coverage expires, the rebuild
assigns new owners, redistributes fields in identity order and enumerates
the canonical images for the new halo.

A native constraint projection also holds a layout over successive calls
at its force site. Its reference is the complete configuration at the last
partition. The site tests each minimum-image displacement against half
the skin. On a miss, it partitions the current positions and assigns each
constraint group to its common owner. On a hit, it uses those same member
rows for the solve. The cache belongs to the enclosing loop with invariant
cell and group relations. A continuation creates a new cache.

The constrained fixed-cell GPU Verlet loop carries the owned position,
velocity, force, mass and potential parameters. At the force neighborhood
it tests coverage and, on a rebuild, redistributes each live field by
particle identity. Position and velocity projections read the members
of each disjoint group on their common owner. Each rank compacts its
bonded and constraint incidences in original topology order. Original
tuple identities address the device parameter arrays. The ordinary device tuple lowering evaluates
these local relations. Constraint failure flags are agreed across ranks
and virial terms reduce per-domain partials. Only the live global fields
consumed outside the packed loop are collected at its boundary.

The disjoint solve and its output correction use the integrator's
working precision. Narrowing this local correction in mixed mode loses
accuracy for positions close to a periodic face. Bonded tuples retain
the ordinary force precision.

With one domain per rank, one all-gather publishes the floating-point
observable partials. Each receiver maps ranks back to domain IDs before
the existing ordered sum. At a device face rebuild, one count exchange
and one variable-length all-gather replace the count and payload
broadcasts from each rank. Receivers process the lists in rank order and
sort each sender's migrating identities as before. Large displacements
outside the MPI collective's integer range use chunked broadcasts.

### Row lists for a covered directed traversal

At a halo rebuild, each owned row gets a slot in the interior list or
the boundary list. Every exported row, including a periodic self-image
source, belongs to the boundary. Once the device builds the neighbor
matrix, one kernel converts the cell-ordered matrix places into the two
launch lists. Each covered phase indexes its own list and reduces its
observable scratch over that list's length. The interior evaluates
against owned coordinates while the communication stream fills the
ghost rows. The boundary waits for the halo event. A cold matrix takes
a full traversal. For an eligible directed run of at least 32 steps,
the default `pair_overlap = "auto"` runs two disposable 32-step trials
from the same input on the production layout. The second half of each
trial determines the choice when energy reports bracket that interval.
Otherwise the trial uses elapsed integration time. Rank zero broadcasts
the two times. The production log records both and the selected traversal.
Short runs use the unsplit traversal. Explicit `"split"` and `"unsplit"`
values bypass measurement. A single rank always uses its self-image path.

### Asynchronous rebuild face exchange

Each rank starts a nonblocking all-gather of its record count, then maps
local row indices to particle identities. Once the lengths arrive, it
posts the peer receives into disjoint rank-ordered slices, posts all
sends, and waits for completion before migration consumes the records.
Peer payloads exceeding the MPI count range use matching chunk rounds.
The wait timer records the two completion waits. Earlier rebuilds timed
arrival barriers before blocking collectives, so the two measurements
include different amounts of communication progress. Packing, transfer,
and scatter are ordered on the same CUDA stream. A synchronous caller
waits after the scatter.

On eight V100s, the mean maximum-rank rebuild wait across two liquid
runs changed from 13.169 to 14.517 ms. The corresponding migration
phase changed from 25.777 to 21.710 ms. Each run contributed 19 rebuild
intervals between steps 103 and 203. The change in the wait timer above
precludes attributing the migration reduction to a shorter measured wait.

### Device scheduling of bonded tuples

At a rebuild, the rank uploads the global topology adjacency and an
epoch-specific lookup from particle identity to owner row and local
images. For each new position field, NCCL gathers the unshifted owner
coordinates into a padded device buffer. This exchange lets every rank
determine the canonical midpoint owner even when one of the members is
absent from its halo. A schedule kernel selects the owner and addresses
the corresponding imaged rows. Each local particle counts its active
incidences in original topology order. A block prefix sum and a second
particle pass write compressed rows directly into device memory. The
bonded kernels consume those rows and use the original tuple numbers to
index the parameters. A constraint group uses its common owner and
reuses its incidence rows for the lifetime of the layout.

Device events order the joined owned and ghost fields, tuple evaluation,
and reverse force exchange. Immutable parameter fields upload once per
field lifetime. The schedule, coordinates, and tuple forces stay on
the device during covered steps. Scalar observations and constraint
failure flags still have host consumers. The padded coordinate gather
communicates all owners' positions for each new bonded position field.

### Mixed-precision comparisons between backends

For a scalar component $X=\sum_i X_i$, model the result on backend $b$ as
$\widehat X_b=\sum_i(X_i+e_{bi})$. The reduction-order allowance assigns
each backend $|e_{bi}|\le 2\epsilon_{32}|X_i|$, with
$\epsilon_{32}=2^{-23}$. Applying the triangle inequality to the two
results gives $|\widehat X_1-\widehat X_2|\le
4\epsilon_{32}\sum_i|X_i|$. The coefficient four specifies the
validation budget. The worst-case error of an arbitrary serial f32 sum
also depends on its length.

Use the greater of this allowance and the original field-specific
absolute bound for each scalar of a mixed GPU versus CPU MATRIX
comparison. For each energy frame, sum the magnitudes of the reference
potential terms. For the virial, take the magnitude of each term's
signed virial trace, including the constraint contribution when
present. For force, sum the magnitudes of the force terms at the same
particle and Cartesian component. For positions and velocities the
checkpoint supplies a single reference component. Apply the minimum
image to the position difference. Double comparisons and distributed
versus ordinary GPU comparisons use the original absolute bounds.
The comparison script checks every frame or coordinate against its own
bound and reports the largest error-to-bound ratio.

In the charged liquid's initial frame, CPU mixed single-term replays
give LJ and Coulomb virials of $-4297730.897035$ and $3095703.864428$
kcal/mol. Their absolute contributions give a bound of $3.525464$
kcal/mol. The one-rank mixed GPU difference from the CPU MATRIX trace
of $-1202027.032770$ kcal/mol was $0.521918$ kcal/mol.

## Per-step cost of distributed dynamics

On a V100, the original million-particle liquid averaged 11.138 ms/step
on four devices. Nineteen of 102 timed intervals rebuilt the halo.
In two allocations the other intervals averaged 4.77 and 4.52 ms,
while the rebuild intervals averaged 46.73 and 37.49 ms. On the
eligible device rebuild path, each owner now compares the new face mask
with the previous mask and exchanges changed records. A zero mask
removes a former face member. Migrating rows carry the previous mask
in their new owner order.

For double-precision NCCL potentials with deterministic execution
disabled, two single-use pair energies added over the same neighborhood
can share a differentiated pair walk. Both operands must use the same
positions, cell and exchange convention. Each retains its own scalar
energy formula.
