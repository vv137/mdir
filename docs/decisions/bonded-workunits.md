# D[bonded-workunits]

Recorded at its merge.

Use the moving TOWERS force representation for atom-local bonded work
units on a GPU. Round each Cartesian force contribution to the nearest
signed 64-bit word, ties to even, at $2^{18}$ words per kJ mol⁻¹ nm⁻¹.
Check the conversion, the shared sum and the global sum. A failed check
ends the run before integration consumes the force. Term energies and
virials retain their physical units and use ordered binary64 reductions.

Partition bonds, angles, dihedrals, CMAP and 1–4 pairs in topology order.
An interval takes only its members' forces. The owner of the first slot
records energy and virial at the original term index. The experimental
GPU strategy is selected with `MDIR_TUPLE_SUMS=workunits`. Two
opposite-order timing runs found longer steps on all four benchmark
systems, so the existing direct/shared-hash selector remains the default.
[Implementation and measurements](../device-tuples.md#12-topology-order-work-units)
