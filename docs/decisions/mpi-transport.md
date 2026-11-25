# D[mpi-transport]

Recorded at its merge.

CPU ranks use MPI for the D310 halo operations. Configure with
`-DMDIR_ENABLE_MPI=ON` and request `transport = "mpi"` inside
`[distribute]`. The default build and the default transport use the
in-process executor. Rank zero writes the trajectory, checkpoint, energy,
log records and manifest. Every rank participates in the checkpoint stop
decision.

The rank intervals partition the global domain sequence. A rank computes
pair contributions for its own centers. Halo copy packs the live overlap
for each peer. Halo sum returns endpoint contributions and applies them
in the in-process executor's ghost order. The scalar reduction shares
each domain partial before summing in global domain order. These choices
preserve the floating-point order for a fixed grid and traversal as ranks
change. Interior work and communication overlap remain open in #481 and
#536. The layout and the collected rows still use replicated storage.

Use the launcher's communicator order for the rank map. Rank zero merges
energy-interval and final tables. Rank-local traces require the prefix on
every rank. Time collective arrival with a barrier on the communicator.
The reported wait includes the barrier latency and is subtracted from its
enclosing operation. Trace and timing collectives are excluded from phase
clocks. Checkpoint-private fields gather on rank zero. A field also needed
by the next segment must be shared with its consumers. The restart format
stores no rank metadata. The fixed-grid reduction preserves domain order
when the rank count changes. Changing the grid can change the results.
