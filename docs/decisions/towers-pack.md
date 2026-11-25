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
maps. Cells with more than 256 atoms retain the dense rank scan. Control
files, storage and the force traversal use the existing interfaces.

The rebuild test compares both insertion directions against independently
sorted keys, checks cell-relative coordinates and enumerates pair ownership.
Its dense fixture exercises both the eight-key network and the larger-cell
fallback.
