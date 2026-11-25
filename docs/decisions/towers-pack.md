# D[towers-pack]

Recorded at its merge.

Use an eight-key-per-lane bitonic network for GPU cells containing 129–256
atoms. The two Trp-cage input frames each have 45 cells at HALF height.
Their cell populations range from 131 to 164 and from 168 to 208. The
previous per-key rank scan therefore visited every key for every atom in
both frames. At 256 slots the network compares 128 pairs at each of 36
stages within one warp.

Insertion computes a fresh quarter-cell key on every evaluation. An atom
can change quarters while staying in its cell, so an unchanged set of atom
IDs is insufficient to reuse the previous ordering. The unsigned compound
key gives each atom a unique rank even when several atoms share a quarter.
The gather uses this rank for coordinates, types, stable IDs and inverse
maps. Cells with more than 256 atoms retain the dense rank scan. The force traversal consumes the same spans.

The GPU emits separate pack kernels so the eight-key network has its own
register allocation. A warp leader knows the new cell count after reserving
its group's slots. It raises a device maximum when that count exceeds the
previous maximum. For a grid whose peak can fall on either side of 128,
the compact and wide launches test the maximum on the simulation stream.
The compact launch saves a snapshot, and the wide launch resets the
accumulator for the next insertion. Exactly one launch packs the grid.
If there are at most 128 atoms, only the compact launch is needed. If
there are more than 128 atoms per cell on average, the maximum must exceed
128 and only the wide launch is needed. Both single-launch paths also
save and reset the maximum.

The rebuild test compares both insertion directions against independently
sorted keys, checks cell-relative coordinates and enumerates pair ownership.
Its dense fixture exercises both the eight-key network and the larger-cell
fallback.
