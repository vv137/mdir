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
each send once. The host step state supplies the local field before each
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
