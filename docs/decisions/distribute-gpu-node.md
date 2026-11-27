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

A V-rescale thermostat step uses the same coverage decision. Each domain
sums kinetic energy over its owned velocities and applies the resulting
global scale factor to those rows. Thermostat applications reuse the
current halo and MATRIX while the positions remain covered.

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
values bypass measurement. A single rank uses the unsplit traversal. The physical-row case described below can omit its self exchange.

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

For the directed double-precision path without exclusions, each rank
materializes only its incident halo routes. Global route counts remain
available to the observer. Import order follows global identity and the
existing periodic image order. Other halo consumers retain full maps.

Independent sums over owned particles can share a domain traversal
and field views when their region inputs dominate the traversal.
Each observable keeps its own accumulation and domain reduction.

Device migration computes one peer schedule for the ownership change.
Each live field uses that schedule to pack outgoing rows, exchange them
with their new owners, and reorder the received rows on the device.

The sorted exclusion table follows the lifetime of its input relation.
A halo retains the mapped exclusion pairs for each table and domain
until its layout expires. Repeated evaluations can borrow those pairs.

MPI wait timing covers arrival skew and completion inside the blocking
payload collective. A one-rank communicator can return its local
coverage predicate or maximum immediately.

Owned observable fields and imported force slices use the executor's
compute stream. An event orders the producer before the consumer.
The synchronous packed-halo path also orders the halo stream after the
position update through an event.

A relation used in a successor block creates its table in the consumer's
scope. When a loop replaces a relation between halo rebuilds, its expired
maps are reclaimed on lookup or at halo release.

Two independent V100 allocations per layout gave the following double-precision
means in ms/step for production steps 103 through 204.

| Workload | One GPU | Two GPUs | Four GPUs | Two nodes of four GPUs |
|---|---:|---:|---:|---:|
| Charged liquid, 1,048,576 particles | 8.347 | 9.257 | 7.521 | 9.053 |
| JAC-RF | 28.335 | 29.328 | 29.578 | 37.812 |

Ordinary one-device liquid MATRIX measured 5.845 ms/step. One distributed
device took 1.428 times as long. Four devices gave a 1.110-fold
speedup over one distributed device. The eight-device time was 1.204
times the four-device time. All 16 original precision and layout
comparisons and all four mixed cross-backend comparisons passed at
their original bounds. The 53 focused regressions passed.

### Directed halo rebuilds

On four V100s the two runs above had 19 rebuilds in 102 intervals each.
The rebuilds averaged 27.84 and 27.09 ms. Migration accounted for 11.79
and 11.46 ms, and halo construction for 6.34 and 6.28 ms. Covered
intervals averaged 3.43 and 3.46 ms including final output, and 2.10
and 2.09 ms when that interval was dropped.

For the directed face path, apply the exchanged owner and direction
changes to a device snapshot indexed by particle identity. Each rank
keeps per-direction counts for communication and reporting. A changed
owner splits the sorted identity lists of the sending and receiving
domains. Copy the intervening runs. Describe each local run by its
new first row and old first row, with a separate slot for each arrival.
The GPU expands these spans into the gather map for all migrating fields.
A host consumer looking up a crossing row searches the sorted domain
list by identity. The old layout remains valid for the
fields still being redistributed.

Device prefix sums over blocks of identities assign owned rows, ghost
rows, and offsets within each nonempty peer message. The device fills
the send indices, receive indices, and periodic shifts directly. It also
writes the next face classification in local owned order. Device message
slots preserve the destination's identity and image order even when
physical rank placement permutes the geometric domains.

### Measured halo reach

A numeric `[energy] pairlist_distance` fixes the halo reach. For directed
NCCL runs without topology, `"AUTO"` can measure that reach on the allocated
layout. An explicit request searches for runs of at least 204 steps. An
omitted reach searches for runs of at least 4,080 steps. Each candidate
uses a disposable 204-step trajectory with the production energy cadence,
capped at 102 steps between reports. The run log gives each skin and its
measured time, and the manifest records the candidate times and selected
reach. The search precedes automatic selection of pair overlap.

In two one-device liquid runs, increasing the skin from 0.15 to 0.60 Å
increased the mean gap between successive rebuilds from 5.22 to 20.25
steps. The 0.60 Å runs took 7.552 and 7.697 ms/step. A further increase
to 1.20 Å gave 9.049 and 9.050 ms/step. The wider list costs time on
every covered step. The search measures the entire distributed step so
that this cost enters the choice alongside the cost of rebuilding.

A periodic face with the same sending and receiving rank uses a
device-to-device copy between the packed message buffers. Remote sends
and receives retain their peer offsets and ordering. Forward coordinate
translation and reverse force accumulation use the same row maps.

The displacement predicate retains its four-byte result for the NCCL
runtime lifetime. Each call clears it before testing local rows and reads
it before voting on coverage. Shutdown releases it. On one V100, two
2,104-step runs at a 0.15 Å skin reached 4,661.375 MiB used within the
first 1,000 steps and stayed at or below that high-water mark to the end.

For an excluded GPU MATRIX, store the candidate rows, traversal order
and counts with the mapped exclusion relation for that halo and domain.
A changed relation has its own map. A different cutoff declines reuse.
Releasing an expired map releases its cached list. The list radius
includes twice the halo margin so that a list first built during the
epoch remains valid through its last covered evaluation.

Submit the local and inter-node peer groups inside a common NCCL group
when the plan uses both link classes. Each peer still receives one
contiguous slice per field. The joint launch is charged to the outer
halo phase. Gather interval clocks before formatting the replicated
counts. The writer emits the complete interval in one log call.

A pair-only epoch retains sparse changes in particle owner and face mask.
When its predecessor releases its last reference, the runtime applies
those changes to the retired host arrays. A successor rebuild can take
the updated arrays and reuse the retired identity storage if it fits.
An outstanding reader or another branch keeps its original snapshot.
A rebuild without a spare copies the current tables on demand.
The device classifier reuses its count and record storage across rebuilds
and frees the allocation at transport shutdown.

With multiple domains, when fused particle observables each feed one reduction in the same
block, store their components in adjacent columns of a single partial
row per domain. Share the rows once. Each reduction reads its own
columns and accumulates in the original domain order. The fixed-point
path retains its word representation and communication sequence.

At a 0.15 Å skin, four V100 devices rebuilt the liquid halo 19 times
in 102 steps at 2 fs. The rebuild change lowered the ratio of mean
rebuild to covered time from 4.59 to 2.82 against its immediate parent
in the same allocation. The means include final output in its measured
interval. The full production step improved by 6.5 percent.

The four-device comparison of sparse snapshot reuse at a 0.15 Å skin
gave 3.485 ms per covered interval and 10.335 ms per rebuild. The
corresponding parent means were 3.998 and 11.849 ms. Both rebuilt
19 times in 102 production steps at 2 fs.

On the directed device face path each rank merges only its own sorted
identities. All ranks still know every particle owner and every domain
size. A global identity view reconstructs the remote rows from that
map in increasing identity order, in the existing allocation. Collective
field output and the host fallback request those rows before reading
them. A checkpoint needs them only on its writer.

On four V100 devices the fixed-skin liquid rebuilt 19 times in 102 steps. The rank-local numbering gave 8.627 ms per rebuild, 3.593 ms per covered interval including final output, and 3.525 ms per production step. A 204-step checkpoint comparison agreed exactly in identities, positions, velocities and forces.

For constrained multi-device NCCL halos, enclose the physical domain in
the central box and group its owned sites by their nearest periodic
image relative to the domain center. Each axis has three possible
classifications, giving at most 27 boxes. An image is a ghost candidate
if its distance from any box is less than the halo reach. The original
identity and image ordering determines all surviving rows. Domain
diagnostics still report the outer enclosing box.

In the floating-point distributed GPU executor with tuple terms, delay
an observable reduction and its scalar arithmetic until their first
uses in the same block. At a reduction, pack independent partials that
are already available on that rank into a row. One collective shares
all components. Each observable then adds its own column in the
original domain order. The fixed-point executor keeps its word sums.

On the 23,558-site JAC-RF case, two 2 fs runs on each layout averaged
24.532 ms/step on four V100 devices and 21.802 on eight. The respective
matched controls were 28.354 and 33.454 ms/step. The return traffic
across nodes fell from 34.763 to 13.882 MB/step.

The 90,906-atom Factor IX benchmark used the same 2 fs finite-range
controls. The two-run means in ms/step were 2.098 for ordinary MATRIX
and 76.287, 61.760, 54.835, and 50.320 at one, two, four, and eight
distributed GPUs, respectively. Both repetitions at every layout
agreed with the ordinary potential energy and scalar virial across all
205 frames within $2\times10^{-4}$ and $2\times10^{-3}$ kcal/mol.

Floating-point NCCL particle maps that produce a three-component field
from owned scalar and vector rows can borrow those inputs on the device.
Each rank executes the pointwise program over its own count. The result
is copied on the producer stream into the new owned field. This keeps
constraint corrections on the device between tuple projections.

For a floating-point NCCL MATRIX, build the ordinary exclusion incidence
over the mapped local row set once per halo, domain and live exclusion
relation. The halo owns the device buffer. A covered step borrows it
directly, and a replacement relation receives its own incidence even
within the same halo. Releasing a map releases its incidence.

All tuple sites in a rank use the same device table of particle identities,
owners, local row indices, and periodic images for a given halo. Constraint
groups use the corresponding owned-only table on the layout. Both tables
expire with the object that defines their row numbering.

On a single domain, a scheduled bonded interaction has all its members
on the local rank. The kernel uses the layout's owned particle rows and
periodic bond geometry. A packed unique neighborhood with exclusions can
likewise construct its periodic MATRIX over owned rows when its consumers
use the complete set of centers and return ghost contributions only to
the matching reverse sum. The identity test retains one orientation of
each pair. The pair list covers twice the per-particle displacement
allowance of the packed state.

A tuple relation owns its global member indices and adjacency list from
its first device use to the end of its SSA lifetime. A new rank layout
maps local rows against these immutable arrays. Each mapped plan retains
its relation, so replacing a relation at an evaluation site rebuilds the
plan even when the layout is unchanged.

A stationary packed layout can reuse the local tuple numbering after
migration when every particle retains its owner. If all consumers of an
imaged coordinate use owned positions and periodic bond or pair geometry,
the lowered program omits that halo copy entirely.

Where tuples and explicit MATRIX pairs share positions, both consumers
read one completed exchange. The halo owns the periodic translations.
A device kernel adds these translations to raw ghost coordinates for the
pair list, while tuple kernels read the raw coordinates.

In one domain the sorted row of a particle is its global identity.
When every image consumer takes physical rows, a new epoch can retain
this mapping, wrap the coordinates, and publish a halo with zero ghosts.
The displacement test and neighbor-list rebuild continue to run.

An outlined rank-local evaluation borrows immutable device inputs.
Precision assignment first isolates the storage and kernel roles. After
those roles are fixed, it removes an input copy only if the two field
types agree and the copy kernel returns its argument directly. Storage
assignment then protects the borrowed buffer from in-place writes.

A cached single-domain periodic MATRIX applies the global identity test
while compacting its candidate neighbors. Every row is local and the
physical row index equals the particle identity. The builder therefore
stores each eligible pair once, in increasing identity order, before
the force or observable evaluation. Each consumer reads that list
directly. The uncached single-domain evaluation still applies the
identity test at traversal.

An excluded pair on a single physical domain has the same two row
indices after every rebuild. The exclusion relation owns these members
and their device incidence while the particle count is unchanged. Each
halo keeps its own neighbor list. A new exclusion relation has its own
members and incidence, even when it appears within the same halo.

When every packed physical neighborhood uses the state's verified position
field, the single-domain MATRIX can search at the epoch reference
coordinates. The candidate radius is the cutoff plus the halo margin.
The pair evaluation uses current positions. For other cached layouts the
search still uses current positions with twice that margin.

### Mixed trajectory acceptance

The mixed gate uses ordinary MATRIX runs of each protein at 2 fs with
identical input coordinates, velocities, dynamics seed and reaction-field
parameters. The original calibration used two reference runs and 19
additional pairs in the ordinary neighbor order. Reversing only the
neighbor slots of the ordinary JAC-RF run produced a 1.275243 kcal/mol
virial difference, above that calibration's 1.098365 kcal/mol limit.

Keep the original 40 ordinary histories and add two runs each in the
original order, reversed order and eight seeded orders. In each seeded
order, four rounds compose bijections of each row's occupied slots. In
one round slot $i$ is paired with $(p-i)\bmod n$, where $p$ depends on the
row and seed and $n$ is the row length. Both partners use the same
decision, derived from the smaller index. The cutoff test, exclusions,
pair expressions, and lane reduction structure are identical in all
orders. Separate compile caches keep the generated variants distinct.

For each quantity $Q$ (potential energy and scalar virial), collect
$|O_{a,Q}(t)-O_{b,Q}(t)|$ over all unordered pairs of distinct ordinary
histories and all frames $t$ from 0 through 204. The 60 histories give
362,850 differences per quantity. The bound is 1.25 times the empirical
99.95th percentile of those differences, using the higher sample when
the quantile falls between samples. The percentile and margin were fixed
before comparing the distributed candidate. Frames and differences that
share histories are correlated. This is an empirical tolerance for
the tested RF datasets and orders.

| System | Potential 99.95th percentile | Virial 99.95th percentile | Potential bound | Virial bound |
|---|---:|---:|---:|---:|
| JAC-RF | 0.103113 | 1.499141 | 0.128891 | 1.873926 |
| Factor IX | 1.279922 | 9.642828 | 1.599903 | 12.053535 |

All four values for each system are in kcal/mol. The exact bounds are
stored in the calibration summaries.

Two distributed repetitions pass when both potential and virial
differences from the original two ordinary reference histories stay
within their respective bounds at every frame. The checker reports
the repetition and frame of peak error for each quantity.

```sh
python scripts/validation/distributed/trajectory.py --system factorix distributed1.dat distributed2.dat
```

Use `--system jac-rf` for JAC-RF. The reference data store the ordinary
potential and virial for each frame. Passing the original energy files
with `--reference` verifies their hashes and order. The checker exits
with status 1 if either quantity exceeds its bound. The double limits
remain the same.

To reproduce the calibration, use
`scripts/validation/distributed/calibrate-orders.py overlay SOURCE` on
a disposable copy of the source and build it. For each label printed by
`calibrate-orders.py orders`, set `MDIR_CALIBRATION_ORDER` to that label
and use a compile cache specific to the system and label. Run the
ordinary mixed deck twice, storing `energy.dat` under
`LABEL/1/` and `LABEL/2/`. Place the original references followed by the
38 same-order controls under `existing/01/` through `existing/40/`.
Run `calibrate-orders.py summarize HISTORIES` to generate the bounds and
input digests. The `--calibration` and `--reference` options accept a
regenerated summary and an explicit ordinary reference pair.

### Large reaction-field benchmarks

Cellulose has 408,609 atoms and STMV has 1,067,095. Use the prepared
Amber topology and restart from `cellulose_nve` and `stmv_npt_4fs`.
The latter retains its benchmark masses. Both runs use the restart
velocities and orthorhombic cell, periodic NVE velocity Verlet at 2 fs,
rigid water and analytic constraints on bonds to hydrogen. The reaction-field
dielectric is zero, the cutoff is 8 Å, and the list distance is 10 Å.
The control generator takes a prepared directory and a precision.
Omitting the domain count selects ordinary MATRIX. The preparation
interpreter needs ParmEd.

```sh
python scripts/benchmarks/amber/bench.py --work inputs prepare --parmed-python python3
python scripts/validation/distributed/large-rf.py inputs/cellulose_nve --precision double --domains 4 > run.toml
```

The following V100 times are ms/step over the last 102 of 204 steps.
Each cell represents one unprofiled trajectory. The distributed runs
use unique traversal with one NCCL rank per device. Eight devices span
two nodes.

| System | Precision | Ordinary | One | Two | Four | Eight | One/ordinary |
|---|---|---:|---:|---:|---:|---:|---:|
| Cellulose | double | 8.353 | 18.730 | 51.202 | 40.912 | 49.859 | 2.242 |
| Cellulose | mixed | 6.578 | 16.562 | 51.271 | 40.093 | 48.175 | 2.518 |
| STMV | double | 21.657 | 48.129 | 124.950 | 120.330 | 133.408 | 2.222 |
| STMV | mixed | 16.518 | 42.704 | 126.084 | 121.413 | 136.338 | 2.585 |

For Cellulose double the two ranks held 0.886 and 0.888 ghost rows per
owned row. The pair phase rose from 8.211 ms/step on one rank to
21.296 and 21.535 ms/step. The two migration clocks averaged 4.510 and
4.593 ms/step. Halo construction took 5.166 and 5.217 ms/step.
On eight ranks halo construction ranged from 13.47 to 15.07 ms/step.
For STMV double the pair phase rose from 23.494 ms/step on one rank to
50.092 and 51.179 ms/step on two.

Separate Cellulose double traces measured 51.15 kernel launches per
step for ordinary execution, 176.40 on one distributed rank and
279.61 on each of two ranks. The corresponding device-to-host transfer
counts were 2.13, 29.13 and 30.51. At two ranks, rank zero uploaded
22.19 MB/step. The owned and ghost rows together accounted for 385,403
identities at step 103. Four 64-bit copies at that size transfer 12.33 MB.

At two domains, both precision modes exchange a complete set of coordinates
to assign bonded tuples by midpoint. In double precision, rank zero
contributed 4.909 MB per step for Cellulose and 12.845 MB for STMV to
the NCCL all-gather. It also sent 13.440 and 37.582 MB per step,
respectively, in peer messages. The Cellulose rank called six MPI
all-gathers and five MPI all-reduces per step. STMV called six of each.
These are application payloads in decimal megabytes.

The following double-precision Cellulose trials used the same allocation
for each A/B/B/A sequence. The times are ms/step. The identity upload
must precede use on the compute stream. Omitting that dependency caused
a potential error of 15,235.74 kcal/mol at the first halo rebuild.

| Two-domain change | A | B | B | A | Reduction in mean time |
|---|---:|---:|---:|---:|---:|
| Reuse the halo identity buffer with a stream wait | 50.250 | 48.269 | 47.517 | 50.394 | 4.8% |
| Reuse the tuple incidence for the same topology and coordinates | 51.930 | 46.915 | 47.992 | 50.789 | 7.6% |
| Select unique pairs while building the explicit list | 51.412 | 43.893 | 43.350 | 50.218 | 14.2% |

For each of these changes, Cellulose double stayed within 0.000003
kcal/mol of ordinary for both quantities over 205 frames. The
two-domain Factor IX mixed potential and virial errors were
3.173127 / 10.307203 for identity reuse, 3.045264 / 10.219408 for
incidence reuse, and 2.799096 / 9.671699 kcal/mol for unique-list
construction. Each exceeded the corresponding calibrated limits.

For STMV on two GPUs, unique-list construction reduced the double mean
from 136.504 to 108.519 ms/step and the mixed mean from 125.758 to
100.770. The one-GPU distributed baselines were 48.129 and 42.704.
On one GPU, retaining the original bonded member tables in device
memory reduced the double time from 18.737 to 16.764 on Cellulose and
from 47.704 to 45.047 on STMV. The one-GPU goals were 12.530 and
32.486 ms/step. Factor IX mixed still reached 3.006331 kcal/mol in
potential and 10.191476 kcal/mol in virial with the retained tables.

On eight GPUs, assigning a destination halo to each rank and exchanging
its ordered identity and image records reduced repeated classification.
For Cellulose double, the mean halo-build time over all eight ranks fell
from 13.405 to 3.970 ms/step and the driver time from 46.614 to 38.476.
For STMV double, the rank mean fell from 52.679 to 12.284 and the driver
time from 136.732 to 95.887. In mixed precision the STMV driver time fell
from 136.515 to 93.787. The eight-rank protein check exceeded the mixed
limits on Factor IX (3.191403 / 11.512909 kcal/mol) and the virial limit
on JAC-RF (1.383398 kcal/mol). The measured runtime changes remain
experimental.

With this halo implementation, four GPUs took 38.072 and 37.836 ms/step
on Cellulose in double and mixed precision, compared with 38.476 and
38.571 on eight. Four GPUs took 92.949 and 93.310 on STMV, compared
with 95.887 and 93.787 on eight. Sharing classification improved the
four-domain run as well. Neither system achieved the four-to-eight
target in these measurements.

### One-domain half MATRIX for reaction-field dynamics

An NCCL unique traversal with one domain and constrained topology can
use the single-device integrator. No pair has endpoints on different
devices. Keep input row order and build a periodic half MATRIX. Each
accepted pair contributes to both endpoint forces. The specialization
selects built-in RF terms without a thermostat or user pair, triplet,
or model terms.

Each comparison used 205 frames at 2 fs. A is the parent, B is the
half MATRIX, and ordinary is the mean of its two allocation-matched
runs. Times are ms/step over steps 102 through 204.

| System | Precision | Ordinary | A1 | B1 | B2 | A2 |
|---|---|---:|---:|---:|---:|---:|
| JAC-RF | Mixed | 0.670 | 3.367 | 0.686 | 0.688 | 3.368 |
| JAC-RF | Double | 0.824 | 3.309 | 0.791 | 0.797 | 3.332 |
| Factor IX | Mixed | 1.645 | 5.234 | 1.792 | 1.850 | 5.255 |
| Factor IX | Double | 2.067 | 5.610 | 2.088 | 2.090 | 5.582 |
| Cellulose | Mixed | 6.579 | 16.194 | 7.207 | 7.204 | 16.217 |
| Cellulose | Double | 8.366 | 18.585 | 8.411 | 8.411 | 18.214 |
| STMV | Mixed | 16.494 | 42.295 | 18.289 | 18.286 | 42.076 |
| STMV | Double | 21.692 | 47.509 | 22.230 | 22.072 | 47.562 |

The mixed candidate's largest potential and virial differences from
the frozen ordinary references were 0.060534 and 0.908409 kcal/mol on
JAC-RF and 0.923649 and 9.009296 kcal/mol on Factor IX. Both passed the
order-aware bounds. Across the four double cases, the largest
differences from the respective ordinary runs were 0.0000011 kcal/mol
for potential and 0.0000061 kcal/mol for virial. The candidate passed
all 53 focused tests and all 20 numerical comparisons.
The liquid controls also passed. The one-GPU-to-ordinary ratio was
1.211, the one-to-four-GPU speedup was 2.781, and the four-to-eight
speedup was 1.072. The fixed-skin rebuild-to-covered time ratio was
2.247.

### Mixed coordinate images and constraint projection

At 2 fs, a one-domain RF run differs from ordinary mixed before its first
kick. The table gives distributed minus ordinary energy in kcal/mol at
the input coordinates. Bond, angle, dihedral and both 1–4 energies
agreed to the six decimals printed. The first and last columns evaluate
the full potential and the molecular virial independently of the term
energy listing.

| System | Full potential | Lennard-Jones | RF Coulomb pairs | RF exclusions | Molecular virial |
|---|---:|---:|---:|---:|---:|
| Factor IX | −0.032784 | −0.003287 | −0.032591 | +0.005268 | −0.095854 |
| JAC-RF | +0.005400 | +0.000700 | +0.004550 | −0.000140 | +0.012220 |

The ordinary mixed MATRIX converts coordinates to binary32 before
subtracting them and applying the periodic translation. Distribution
first reduces each input coordinate into the primary cell in binary64.
Translation by a lattice vector and conversion to binary32 can change
the low bits of the resulting pair displacement. To measure that effect,
apply exactly the distributed initial translation to the ordinary input
in memory. Compare the resulting ordinary force with the distributed
force at the same physical frame. The entries are RMS Cartesian force
differences in kJ/mol/nm. Frame one uses one ordinary checkpoint,
serialized once as a common coordinate input for both evaluations.

| System | Input frame, original / translated ordinary | Frame one, original / translated ordinary |
|---|---:|---:|
| Factor IX | 0.007771 / 0.0001950 | 0.007753 / 0.0001960 |
| JAC-RF | 0.0008845 / 0.0001961 | 0.0009628 / 0.0001950 |

On Factor IX the input translation reproduced the distributed
excluded-pair force to an RMS of $2.29\times10^{-8}$ kJ/mol/nm.
The corresponding residuals for the Lennard-Jones and real-space
Coulomb pair forces were $2.81\times10^{-5}$ and $1.80\times10^{-4}$.
The ordinary MATRIX accumulates center forces by neighbor row and
combines the lanes of each row. Unique traversal also adds the
opposite force at the neighbor. Both runs use floating-point
accumulation. The deterministic fixed-point policy remains a separate
execution request.

A second comparison supplied identical full ordinary forces, positions
and velocities to both velocity-Verlet runs. The updated position
isolates the first half kick, drift and position projection. The
maximum Cartesian differences after that projection were
$9.09\times10^{-8}$ nm on Factor IX and $2.34\times10^{-8}$ nm on
JAC-RF. A prototype preserving the input images and using the ordinary
mixed constraint kernel arithmetic with a binary64 correction reduced
those maxima to $1.78\times10^{-15}$ and $8.88\times10^{-16}$ nm.
Changing either feature alone left larger differences.

The tests in the following subsections used the original repetition-only
mixed gate. They retain that acceptance result for each historical run.

### One domain with a periodic half MATRIX

With one physical domain every pair endpoint and bonded tuple is local.
A trial kept the input identity order, built a periodic MATRIX containing
only neighbors with a higher identity, and added each force to both
endpoints. This let the constrained step use the local velocity-Verlet
and projection kernels. The transport still initialized for the requested
rank. The same-allocation means below cover the second half of each
204-step RF run at 2 fs.

| System | Precision | Existing one-domain step (ms) | Local unique step (ms) |
|---|---|---:|---:|
| JAC-RF | double | 3.383 | 0.795 |
| Factor IX | double | 5.610 | 2.100 |
| Cellulose | double | 18.360 | 8.409 |
| Cellulose | mixed | 16.378 | 7.210 |
| STMV | double | 47.708 | 22.096 |
| STMV | mixed | 42.402 | 18.300 |

The first mixed repetition pair passed the frozen gate on both proteins.
The all-frame potential and virial peaks were 0.882819 and 4.075258
kcal/mol for Factor IX, and 0.064784 and 0.996674 for JAC-RF.
With ordinary double as the reference, both corresponding protein
quantities stayed within 0.0000011 kcal/mol.

In paired unprofiled measurements the local/ordinary ratios were 1.009
and 1.097 for Cellulose double and mixed, and 1.024 and 1.108 for STMV.
The CUDA traces counted one extra launch per step in each local run.
Both modes made 2.127 device-to-host transfers and 4.127 synchronization
calls per step. The earlier one-domain Cellulose double trace had 29.127
device-to-host transfers and 61.382 synchronization calls per step.
Over 102 intervals, both modes copied 204 reduction buffers of 8,192
bytes and 13 eight-byte values to the host. The latter count matches the
number of neighbor rebuilds. The extra launch was a particle loop in
the Cellulose double trace.

An independent build and repetition pair reproduced the performance
gain. JAC-RF again passed, with peaks of 0.063485 and 0.913002 kcal/mol.
Factor IX passed potential at 0.845988 but exceeded the frozen virial
limit of 7.742251 at 9.594057 kcal/mol. The single-domain specialization
therefore did not qualify for retention.

These RF measurements used kernel launches. The existing graph replay
implements the mixed TOWERS, Gaussian PME, V-rescale path and requires
an ordinary step between output boundaries. The validation trajectory
uses MATRIX, RF, NVE and an energy write every step. Graph replay would
require a separate implementation covering those output and rebuild
boundaries.

### Physical rows on two domains

On two domains, retaining the global IDs on the compute stream for the
lifetime of a physical halo reduced Cellulose double from 37.663 to
35.670 ms/step. Moving the ID comparison into periodic list construction
reduced the pair phase from 11.845 to 8.486 ms in a separate paired run.
The full step averaged 35.820 and 32.378 ms. The maximum potential and
virial errors of both variants were within 0.0000021 kcal/mol of
ordinary double. Factor IX mixed exceeded its virial bound with each
change, reaching 8.637098 and 8.898177 kcal/mol, respectively.

A separate trial selected each bonded tuple's midpoint owner on first
use in a physical halo, when the runtime verifies that all members are
present. The unshifted rows and reverse destinations remain valid
through that halo's lifetime. Reusing that incidence across coordinate
updates reduced the Cellulose double other-term phase from 10.068 to
6.109 ms/step on two domains. The STMV mixed full step fell from
92.638 to 81.816 ms. Both remained slower than the single-domain
specialization.

The independent local Factor IX repeat first exceeded its virial
bound at frames 189 and 196 in the two trajectories. The corresponding
initial virial errors were −0.004552 and −0.004585 kcal/mol. This
late failure can coexist with agreement of the first projected
coordinates to machine precision.

With both executors widening force destinations while retaining
binary32 pair expressions, their full step-zero force checkpoints
agreed in every stored component for both proteins. Independent
single-term diagnostics retained small differences. The force-storage
change still exceeded the JAC-RF mixed bounds over 205 frames
(0.104037 potential and 1.663530 virial). Step-zero agreement and
late-trajectory acceptance measure distinct properties.

Tracing the two-domain Cellulose double run showed that stream-owned
identities reduced rank-zero host-to-device traffic from 13.997 to
8.384 MB/step. The compact list still launched 278.461 kernels and
made 30.382 device-to-host transfers and 61.382 synchronization calls
per step on each rank. Of the downloads, 28 per step returned
8,192-byte buffers of rank-local reduction results. Rank zero also
copied 279.374 MB/step between device buffers.

Running ordinary and local unique with the same widened force storage
gave identical printed kinetic energies at all 205 frames. Their
maximum potential/virial differences were 0.006042/0.006086 kcal/mol
for Factor IX and 0.002244/0.003460 kcal/mol for JAC-RF. In the latter
case even widened ordinary failed against the frozen original mixed
reference (0.104828/1.661923 kcal/mol). This isolates sensitivity to
force rounding independently of exchange.

With corrected image and projection arithmetic, a second test of halo
sharing compared two mixed trajectories on eight devices against the
frozen ordinary references. The unshared runs passed for both proteins.
Sharing raised the Factor IX maximum potential difference from 0.695913
to 1.355350 kcal/mol, above the 1.223086 bound. On JAC-RF the maximum
virial difference rose from 0.724733 to 1.108238 kcal/mol, above the
1.098365 bound. Retain the unshared schedule.

The scoped one-domain half MATRIX passed all 20 numerical comparisons,
53 focused tests and four liquid timing gates. Liquid took 5.859 ms/step
ordinary, 7.087 on one device, 2.555 on four and 2.407 on eight.
The fixed four-device rebuild/covered ratio was 2.551. The independent
Factor IX mixed failure remains decisive for that implementation.

The same executable passed all double protein comparisons at two, four
and eight devices. Factor IX mixed exceeded both limits at each count.
Its potential differences were 2.981, 2.596 and 3.356 kcal/mol and its
virial differences were 10.644, 10.218 and 10.607 kcal/mol. JAC-RF
mixed exceeded the potential limit on two devices and passed both
limits on four and eight.

A one-domain trial retained ordinary's binary32 row and lane summation
order while evaluating each pair once. It stored signed endpoint
contributions at reciprocal MATRIX slots. Its step-zero full-force RMS
differences were 0.00001271 kJ/mol/nm for JAC-RF and 0.00001014 for
Factor IX, compared with 0.00001152 and 0.00001108 in ordinary repeats.
Both mixed trajectories passed the frozen comparison. Per-step times
were 15.637 and 58.208 ms, compared with 0.710 and 1.643 for ordinary.

The compact periodic list produced the following mean ms/step on one
executable. It failed the one-to-two and four-to-eight scaling targets
in both precisions on both large systems.

| System | Precision | One GPU | Two GPUs | Four GPUs | Eight GPUs |
|---|---|---:|---:|---:|---:|
| Cellulose | double | 18.843 | 32.314 | 28.252 | 35.876 |
| Cellulose | mixed | 16.218 | 30.521 | 27.346 | 35.468 |
| STMV | double | 47.943 | 84.189 | 78.615 | 104.814 |
| STMV | mixed | 42.597 | 77.228 | 77.769 | 112.082 |

In Cellulose mixed the ordered trial stored seven binary32 components
per MATRIX slot and cleared 10.251 GB per step. The same approach
exhausted device memory on STMV in both precisions. A block-local sort
at each rebuild indexed reciprocal slots for a parallel lookup.
Writing only the canonical endpoint and storing each component plane
contiguously avoided the clear and cut Cellulose mixed to 110.341
ms/step. That candidate passed both protein mixed checks.

Caching only the RF force coefficient and energy reduced Cellulose
mixed to 31.412 ms/step. Reconstructing force and diagonal virial in
the endpoint kernel allowed the compiler to fuse both products into
their lane sums. The frozen virial bound was crossed at step 198 on
JAC-RF and 189 on Factor IX. Explicitly rounding the six products
before adding them reduced full-force RMS error from 0.00013042 to
0.00001328 kJ/mol/nm on JAC-RF and from 0.00013299 to 0.00001177
on Factor IX. Both rounded mixed trajectories passed. Cellulose mixed
still took 33.893 ms/step, compared with 6.576 for ordinary.

With the rounded two-scalar cache, one-device double took 2.493 ms/step
on JAC-RF and 9.094 on Factor IX, exceeding the 5.60 limit on Factor IX.
Rounded STMV took 78.901 ms/step mixed and 441.259 double. Both large
systems exceeded the 1.5-times-ordinary limit in both precisions.

At eight domains a diagnostic rebuilt every destination's list locally
at the same coordinates and checked the complete identity, image and
row sequence against the shared result. The check agreed on every rank
at all 22 JAC-RF and 24 Factor IX rebuilds on both node placements. On
two nodes this covered 2,228,560 JAC-RF rows and 5,224,040 Factor IX
rows per rank. The tested membership and ordering therefore agree even
though the independent shared trajectories exceeded the mixed bounds.

### Open Stage 8f items

The fused tuple scheduler remains uncommitted. In the bracket on two Tesla V100-SXM2 GPUs, the STMV mixed NVE candidate reduced the mean step
time from 65.375 to 64.126 ms. Across 205 frames, its largest paired
virial difference against the radix parent was 92.128140 kcal/mol at
step 204 in the second pair.
The within-build maxima were 36.437599 kcal/mol for the radix parent
and 39.669605 kcal/mol for the fused candidate. The largest paired
potential difference was 5.771644 kcal/mol. This STMV deck has no
frozen mixed NVE bound, and the paired virial difference exceeds the
observed repeat variation for both builds. Numerical qualification
remains open.

The STMV mixed NVT smoke run on one RTX 4090 exposed an open CUDA
teardown defect. Both the parsed-input reuse candidate
and its parent wrote all 21 reports through step 200 in both AUTO
and numeric-distance runs, then exited with status 1. Both numeric
replays traced CUDA driver status 4 (`CUDA_ERROR_DEINITIALIZED`) to
`mdrtNCCLReleaseScratch` in `mdrtNCCLFinish`, called from the MPI
exit handler. A diagnostic candidate relink called `mdrtNCCLFinish`
before CLI exit and replayed through step 200 with exit status 0.
The diagnostic change remains outside the production builds.

AUTO pair-list calibration for STMV mixed-precision NVT aborted on
two Tesla V100-SXM2 GPUs with 32 GiB each. The original log reports an out-of-memory abort on rank
1 and omits the active distance. A traced replay used the same
deck and executable on two matching V100s. It reproduced the rank-1
failure at 12 Å, after completing 9.5, 10 and 11 Å and before trying
13 Å. The backtrace places the CUDA allocation failure in
`mdrtNCCLAllocate` during `mdrtDistAdvance`.

AUTO should record and skip a candidate when its device-memory
requirement exceeds the capacity available to any rank, and choose
among the candidates that fit. If none fit, it should report a clear
failure. The current failure aborts the MPI communicator from inside
the runtime after the trial has started. Safe recovery needs a
collectively agreed capacity check or isolation of the trial, with a
regression test that rejects an oversized candidate and still selects a
viable distance. This remains open.

### Ordinary execution for one NCCL rank

With one domain, all interaction endpoints are local. The CLI keeps
the NCCL request in the manifest and runs the ordinary GPU executor.
The ordinary path selects the neighbor structure and accumulation
mode. It can also run in deterministic mode. The grid must still
match the rank count. Runs on multiple ranks retain their domain
traversal and exchange.

### Device memory preflight on the full build

Each rank estimates device demand from the initial geometry and includes
headroom for concurrent MATRIX and tuple storage. On two 32 GiB V100
GPUs, the full build completed STMV numeric runs at 12 and 13 Å in
double precision and at 13 Å in mixed precision. The largest sampled
per-rank usages were 16,718, 20,692, and 16,820 MiB, respectively,
against initial estimates of 28,295.9, 33,696.3, and 24,975.7 MiB.
The estimate still screens out the double-precision 13 Å candidate
on these GPUs. The earlier mixed 13 Å allocation failure reproduced
in the diagnostic overlay with allocation tracing disabled. The full
build also completed the mixed 13 Å deck with allocation tracing.


### Distributed AUTO buffer pins

Record the measured device-buffer peaks for each admitted reach and each
rank. Export the winning rank tables with the numeric reach through
`mdir pin` and the automatic choices in the manifest. Preallocate the
named arenas when replaying these choices. Retired list storage returns
to its retained arena and growth beyond the pin is logged. Runs without
a buffer pin keep demand allocation.

Distributed halo fallback, observer, energy, and coupling transfers use
page-locked DMA staging. Enqueue copies on the stream that orders their
producer and consumer. Host consumers that need values immediately wait
for that stream before accessing the staging buffer.
