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
Destroying the halo invalidates the cached device matrix.

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

For other grids and constrained systems the host plans ownership and halo
membership from coordinates. The current implementation also runs
the topology tuple and constraint paths. A host tuple or output consumer can
require a local field slice. The constrained native integrator can also
materialize global fields. Forward peer exchanges complete before the pair
evaluation starts.
