# D[distribute-gpu-node]

Recorded at its merge.

Stage 1 selects GPU MATRIX pairs with `[execution] target = "GPU"` and
`[distribute] transport = "nccl"`. Build with MPI and `MDIR_ENABLE_NCCL`.
Use one domain per MPI rank. The product of the domain grid dimensions
must equal the communicator size. Directed traversal evaluates both
orientations. Unique and midpoint traversals return ghost forces to the
owners. GPU deterministic execution is refused at setup until it can
implement the integer force and observable sums of ruling B.

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
each send once. Reverse-force bytes and time are accounted separately.
The first validation targets fixed-cell finite-range Coulomb/LJ, including
reaction field. Reciprocal-space decomposition is outside this stage.
