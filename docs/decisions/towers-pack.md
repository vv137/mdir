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

Setup chooses one pack specialization for the entire run. After the first
private rebuild at the resolved height, it reads each cell's own span
count at the synchronization used to check overflow. It includes empty
cells in the mean. AUTO chooses WIDE when the maximum exceeds 128 and
COMPACT otherwise. The compact kernel uses up to four keys per lane and
rank-scans larger cells. The wide kernel also handles smaller populations.
A later population change affects cost only.

The threshold targets the first cell size at which the wide network can
save work. In the two dense input frames the median sorting interval fell
from 14,987 to 4,373 and from 21,720 to 4,405 SM cycles. A frame with only
one dense cell may see a smaller benefit. The physics-neutral
`[execution] towers_shape.pack` field accepts `"AUTO"`, `"COMPACT"`, or
`"WIDE"`. The setup log and manifest record the choice, the selection
rule, and both populations. Pinned launches and cache hits still inspect
the starting frame. Programs without an eligible trial region use the
input-coordinate bin counts and report their source.

The rebuild test compares both insertion directions against independently
sorted keys, checks cell-relative coordinates and enumerates pair ownership.
Its dense fixture includes cells above both network limits.

On an RTX 4090 the compact and wide pack entries use 48 and 72 registers
per thread. With the earlier guarded dispatch, in two reversed-order runs at 2 fs, Trp-cage CHARMM36m saved
7.68 and 8.15 µs per step, and ff19SB + OPC saved 13.95 and 14.78 µs.
All six benchmark deck inputs were slower in both orders. TIP3P 10k cost
an additional 1.87 and 2.24 µs per step. On its fixed input keys, adding
the guarded wide launch increased the event-bracketed pack interval from
4.99 to 6.14 µs. Across the eight fixed frames, all 4,905,552 Q18 force
words and their published components matched the reference bitwise.

On the same TIP3P frame, a CUDA graph IF/ELSE node with a device
condition took 10.24 µs including the condition-setting kernel and
9.22 µs with that kernel outside the interval. Two guarded entries took
6.14 µs in either measurement order.
