# Groups of Neighbors for M1: Design

Status: accepted (2026-10-01), D89; G1 done, G2 under way (the loop in
the default mode is in MDIR, behind `neighbor_structure = "GROUPS"`). It
replaces the
tile structure of D82 ([tiles-m1.md](tiles-m1.md)), whose prototype did not
beat the neighbor matrix. The keys in brackets are those of
[references.md](references.md).

This document describes a second kind of neighbor structure on a device,
`groups`: sixteen particles that share one list of neighbors, over which a
loop computes each pair once. The neighbor matrix
([neighbors-m0.md](neighbors-m0.md)) stays: on the CPU, for kernels that
cannot be computed once a pair, in the deterministic comparison of the
two, and on a device where it is faster.

## 1. Why

Measured on Cellulose (408,609 atoms, RTX 3090,
`scripts/experiments/neighbor-structures`, README):

| | Neighbor matrix | Groups of 16 |
|---|---|---|
| Entries of the list at a reach of 9 Å | 128.7 million, one per particle and neighbor | 8.8 million, one per group and neighbor |
| The loop over pairs, reach 9 Å | 1245 to 1267 µs, bound by the loads of its entries | 1013 to 1034 µs, bound by its arithmetic |
| A build, reach 10 Å | 8.0 ms in MDIR: 5.0 ms for the tests of the candidates, 1.6 for writing the matrix | |

A matrix writes and reads an entry for each particle and neighbor; each
pair is computed twice, once for each particle. Groups write an entry for
each group and neighbor, a sixteenth as many, and compute each pair once.
The published layout is that of [SalomonFerrer2013] (Section 3.3); the
cluster pair lists of [Pall2013] have the same aim.

**What the gain rests on** (`groups.cu`, Cellulose, reach 9 Å, GPU 1,
2026-10-01; groups of 16 against the matrix in the same places):

| Order of the places | erfc | Slots within the cutoff | Groups | Matrix |
|---|---|---|---|---|
| Cells of a third of the reach, x first (a build of MDIR now) | `erfcf` | 23.8 % | 1751 µs | 1284 µs |
| The same | shares the exponential | 23.8 % | 1410 to 1416 µs | 1270 to 1305 µs |
| Compact: columns of 64 halved in x and y into groups of 16, sorted by z | `erfcf` | 29.7 % | 1614 µs | 1236 µs |
| The same | shares the exponential | 29.7 % | 1162 to 1209 µs (996 to 1034 in `supercluster.cu`, not yet explained) | 1196 to 1331 µs |

So the loop over groups gains only with both a compact order of the places
and a cheaper erfc: with the exact `erfcf` of the kernels now it is slower
than the matrix, and with the order of a build now its boxes are 12 Å long
and it is slower as well. Trivial acceptance within 0.75 R of the box (the
build of Section 5 as first written) made the loop 2.5 times slower in the
order of cells and changed little in the compact order: the build tests
every pair against the reach instead. The compact order makes the matrix
faster too (5 to 16 %). The gain of the build, a sixteenth of the entries
and no test of candidates far from a box, is still to be measured
(stage G1).

## 2. The structure

**Places and groups.** A build sorts the particles into cells and gives
each a place in the order of the cells (D86); the positions that the loops
read are gathered in that order. Group g is the places 16 g to 16 g + 15;
the last group is padded with empty places.

**Entries.** The list of group g holds entries (q, m): a place q and a mask
m of 16 bits, bit u for the particle at place 16 g + u. The list holds
every pair {p, q} with |x_p − x_q| ≤ R at the build, once:

- for q in a later group, in the list of g = ⌊p/16⌋ with the bits of the
  particles p of g that pair with q;
- for q in group g itself, in the list of g, with the bits of the places
  before q.

A bit is cleared for an excluded pair and for an empty place. The build
may set bits of pairs farther apart than R (Section 5); the loop tests the
cutoff of each pair.

**Sizes.** No size is fixed. A list is held in blocks of 64 entries,
which its group takes from one pool as it fills them (with an atomic
addition); block b is block `ordinals[b]` of the list of group `units[b]`,
and a block is the unit of work of the loop. The runtime holds the buffers
(`mdrtGroupsCreate`, `mdrtGroupsBuffer`) and sizes them from an estimate:
places for a quarter more than the particles and a chunk, blocks for as
many entries as a row of the matrix for each group and one block more. A
build that finds a buffer too small makes it a quarter larger than it
needs (`mdrtGroupsGrow`) and is made again; the run stops only if the
device has no memory left. Measured: a fixed list of twice the width of
a row stopped a run of Cellulose under NPT when one group needed 2521
entries of 1248, where the longest list is usually 740 to 900; with the
pool, Cellulose takes at most 153,517 of 280,929 blocks (144 MB for the
entries and the masks, against 510 MB for the fixed lists), and a run of
`test/Driver/groups-gpu.test` that starts with almost no room grows its
buffers and gives the energies of the matrix to every digit.

**Validity.** The build tests distances in f32 against the reach widened
by 3e-6 of the sum of the edges of the cell, as the build of the matrix
does: far more than the rounding of the positions to f32 (half an ulp of
the edge a coordinate) can move a distance, so that every pair within R in
f64 is in the lists (`test/Runtime/neighbors-groups-gpu.mlir` checks this
against every pair in f64).

The test of D80 decides when to build again, unchanged: the
structure is valid while
2 max_i |x_i − m ⊙ x_ref,i| ≤ min(m) R − r_c. A pair that the list leaves
out was farther than R apart at the build, and so is farther than r_c now
(D80). Every pair within the cutoff is in the list: the list is a superset
of the pairs within the cutoff, and the cutoff test of the loop removes the
others, as for the matrix ([ops-m0.md](ops-m0.md), Section 9.2).

## 3. The loop: each pair once

The loop implements Section 9.3 of [ops-m0.md](ops-m0.md): for each
unordered pair {i, j} in the list and within the cutoff,

```text
a_i += k(i, j)
a_j += s · k(i, j)        s = +1 for a symmetric kernel, −1 for an antisymmetric one
S   += k_S(i, j)          weight 1 (the matrix counts each pair twice, weight ½)
```

This is legal only for kernels with an exchange contract
([ops-m0.md](ops-m0.md), Section 4.7): the forces of a potential are
antisymmetric (derived by differentiation), its energy and virial
symmetric. `md_exec.pair_for` carries the contract of each of its
destinations and sums (Section 6); a loop with a destination whose kernel
has none keeps the matrix.

**On a device.** The list of a group is cut into units of work of up to
64 entries (the unit of the loop, not of the structure); a warp takes a
unit, 32 entries at a time. Lanes u
and u + 16 hold particle 16 g + u; each lane loads one entry, and the
entries turn within their half-warp, one lane a step, for 16 steps, with
the value accumulated for them, so that each of the 16 particles meets
each of the 32 entries once. The values of the entries go back to their
own places with an atomic addition each, those of the group once a unit.
With a warp for a whole group, JAC (1,568 groups) had too few warps: 125 µs
against 75 for the matrix at 9 Å; with units of 64 entries, 69 µs, and
Cellulose 1109 µs against 1200 (`groups.cu`, 2026-10-01).

**Order of the sums.** The atomic additions make the sums depend on the
order of the threads. In the default mode they are additions in the type
of the destination (D84). In the deterministic mode they are integer
additions in fixed point (D70, [LeGrand2013]), whose sum does not depend on
the order: the value of an entry, summed over the 16 steps of a chunk in
registers, is converted and added; the value of a particle of the group is
converted and added for each chunk, so its sum does not depend on the
order of the chunks either; the entries of a group are in a fixed order
(Section 5), so the sum of a chunk in registers is the same from run to
run.

**Destinations.** A loop that adds each pair once adds into its
destination; it cannot overwrite it (the `overwrite` of
`md_exec.pair_for`). Its destination is zeros or the destination of the
term before (`md-exec-accumulate-destinations`), and the lowering clears a
destination that would be overwritten before the loop.

**Fusion.** A loop over groups is not a loop over the particles: it is
not fused with loops over tuples into runs of rows (`convert-md-exec-to-gpu`,
D83). Two loops over pairs fuse as before when both run over groups; a
fused loop whose destinations are not all symmetric or antisymmetric keeps
the matrix.

**In MDIR** (`convert-md-exec-to-gpu`, `lowerGroupPairFor`): a warp a unit
of work, as many warps as particles at most, a warp taking the units w,
w + warps, ...; the sums of a warp go to the scratch of a particle and are
reduced as those of the matrix. The positions and the fields that the
kernel reads are gathered in the order of the places first; an empty place
takes those of particle 0, which no bit reads.

The kernel is bound by the pipe of loads and shuffles (Nsight Compute, 92%
of its peak on Cellulose), so what it reads and turns a pair is what it
costs: the coefficients of Lennard-Jones are one load of 8 bytes, a table of
vectors (D96), and the mask does not turn with the entry, each lane taking
its bits by ballots at the start of a round (D97); a pair takes two loads
(with the radial table of D94) and eight shuffles.

## 4. What the loop reads

The positions are read in the order of the places, as the matrix reads them
(D86), and the entries hold places. No minimum image is taken (D95): the
build puts each group in a frame of its own, its first particle where it
is wrapped and the others at their places relative to it, and stores for
each place the whole cells that move its particle, as it is kept, into
that frame (ten bits an axis), and in the mask of each entry the cells
that move the entry from the frame of its own group into that of the group
whose list holds it (three bits an axis, −2 to 2, in bits 16 to 24). The
gather adds the shift of the place to the positions as they are stored,
in f64, and converts the sum (D101); the kernel adds that of the entry,
and a displacement is then a subtraction. The kernel is the semantic kernel of the loop,
lowered as for the matrix; the lowering adds the second accumulation and
the atomic additions.

## 5. The build on a device

The build sorts the particles into a compact order (Section 1): groups of
16 that fill a near-cube, so that their boxes are small. On a device: a
counting sort by column in x-y (of the width of 64 particles) and bin
along z; then a warp a chunk of 64 of a column, sorting by x with a bitonic
network and each half by y, which gives 4 groups of up to 16; the places of
a short group stay empty. A Z curve cut into sixteens gives boxes across
its jumps as long as the cell (Section 1 of the README of the
experiments). For each group:
its bounding box; the particles of the cells within R of the box, as
candidates; a candidate farther than R from the box is dropped, and the
others are tested against each particle of the group, the bit set for a
pair within R. The excluded pairs of the group are compared with the
candidates as in the build of the matrix. A warp builds the list of one
group, scanning its cells in a fixed order, so the order of the entries of
a group does not depend on the threads; the entries are written group by
group, each group reserving its room with an atomic addition, and the loop
takes the groups in any order.

A lane tests its candidate against the box only; the candidates near it
go to a queue of the warp, which takes them 32 at a time, a lane each,
against the particles of the group (D99): tested where they lay, a third
of the lanes worked.

Trivial acceptance, which takes a candidate near the box for all the
group without testing, was measured and dropped (Section 1).

Measured (`build.cu`, reach 9 Å, GPU 1): the lists of Cellulose in 1.48 ms
and of JAC in 116 µs, with the order about 0.1 ms more, against 8.0 ms for
a build of the matrix of 10 Å on Cellulose. What it took: a lane a
candidate, testing it against the 16 particles of the group (taking the
survivors two at a time took 53 ms); the partners of the excluded pairs of
a group sorted, and a binary search for each candidate (a linear scan cost
1 ms of 2.4; a group with more partners than the memory of its warp holds,
256, reads the rows of the excluded pairs instead, D106); the candidate relative to the center of the box, once, and
the particles of the group too (1.76 to 1.48 ms). A box is computed
relative to the first particle of the group, in the minimum image.

## 6. In the IR

- `md_exec.NeighborKind` gains `groups`: `md_exec.empty_neighbors
  kind(groups)`, and the refresh builds that kind.
- `md_exec.pair_for` gains `exchange`, one of `symmetric`,
  `antisymmetric`, `none` for each destination and each sum, from the
  `exchange` of the `md.gather_relation` or `md.sum_relation` it came from;
  the fusion of loops keeps them.
- The policy `(unique, atomic)` of `md_exec.pair_for`, which M0 declared
  and rejected, becomes legal over a structure of kind `groups`, with every
  destination and sum symmetric or antisymmetric. The weights of the sums
  are then 1.
- A pass chooses the kind of each structure and the policy of its loops on
  a device (the kind stays `matrix` on the CPU).

## 6.1 Measured in MDIR

Mixed precision, reach 10 Å (the benchmark settings), RTX 3090 at 300 W,
one run each, `nsys` (2026-10-01). The loop over pairs of a step (forces
only) and all the work of the device over the run:

| | Matrix | Groups | Matrix, D90 | Groups, D90 |
|---|---|---|---|---|
| Loop over pairs, Cellulose | 2134 µs | 2097 µs | 1856 µs | 1687 µs |
| Loop over pairs, JAC | 113 µs | 113 µs | 102 µs | 92 µs |
| The device over 500 steps of Cellulose | 2471 ms | 2081 ms | 2325 ms | 1876 ms |
| A build of Cellulose | about 1.0 ms in the kernel of the search, 10 builds per 100 steps | about 0.35 ms over its kernels | | |

With the exact `erfcf` the loops are even, as Section 1 found; with the
approximation of D90 the groups gain 9 % on Cellulose and 10 % on JAC. The
gain of the whole run comes mostly from the build. In double precision the
run of `test/Driver/groups-gpu.test` gives the energies of the matrix to
every digit of the log over 100 steps; JAC under NPT (400 steps) builds as
often, and its conserved energy drifts as much (−7.5 and −7.7 kcal/mol).

## 7. Stages

| Stage | What | Checked by |
|---|---|---|
| G0 | The build prototyped standalone (`scripts/experiments/neighbor-structures`) against the build of the matrix | Its time on Cellulose and JAC, before it is written as a template |
| G1, done but the matrix in the compact order | The compact order of the places (for the matrix as well); the structure and its build on a device; the exchange contracts on `md_exec.pair_for` | The pairs of the list within the cutoff are those of the matrix, on JAC and Cellulose; the matrix in the compact order against the order of cells |
| G2, under way: the default mode and the erfc of D90 done; the deterministic mode to come | The loop over groups, each pair once, both modes; the cheaper erfc of the direct sum under `fast_math`, with the accuracy of the force near the cutoff checked (it is what lets the loop over groups gain, Section 1) | Forces, energies, and virials against the matrix (to the rounding; to the bit in the deterministic mode, where the matrix sums in fixed point too); conservation over runs |
| G3 | Groups by default on a device for the loops that allow them | The Amber suite against pmemd.cuda and GROMACS |
| G4 | The reach of the list: a skin of 1 Å, once builds are cheap | Rates and intervals between builds |
| G5, under way: the table of D94 and the frames of D95 done | The arithmetic of the kernel (a table for the erfc, no minimum image, the order of the entries) | The prototype first |

## 8. Open questions

- The size of a group: 16 measured best among 16 and 32 on Cellulose; 8
  is untried.
- Why the same loop over groups of 16 measures 996 to 1034 µs in
  `supercluster.cu` and 1162 to 1209 in `groups.cu`.
- The size of a unit of work (64 and 128 entries measured; Section 3).
