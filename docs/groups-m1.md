# Groups of Neighbors for M1: Design

Status: accepted (2026-10-01), D89; stage G1 under way. It replaces the
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

**Validity.** The test of D80 decides when to build again, unchanged: the
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

**On a device.** A warp takes a group and 32 entries at a time. Lanes u
and u + 16 hold particle 16 g + u; each lane loads one entry, and the
entries turn within their half-warp, one lane a step, for 16 steps, with
the value accumulated for them, so that each of the 16 particles meets
each of the 32 entries once. The values of the entries go back to their
own places with an atomic addition each, those of the group once for the
whole list.

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

## 4. What the loop reads

The positions are read in the order of the places, as the matrix reads them
(D86), and the entries hold places. The minimum image is taken for each
pair as for the matrix. The kernel is the semantic kernel of the loop,
lowered as for the matrix; the lowering adds the second accumulation and
the atomic additions.

## 5. The build on a device

The build sorts the particles into a compact order (Section 1): groups of
16 that fill a near-cube, so that their boxes are small. For each group:
its bounding box; the particles of the cells within R of the box, as
candidates; a candidate farther than R from the box is dropped, and the
others are tested against each particle of the group, the bit set for a
pair within R. The excluded pairs of the group are compared with the
candidates as in the build of the matrix. A warp builds the list of one
group, scanning its cells in a fixed order, so the order of the entries of
a group does not depend on the threads; the entries are written group by
group, each group reserving its room with an atomic addition, and the loop
takes the groups in any order.

Trivial acceptance, which takes a candidate near the box for all the
group without testing, was measured and dropped (Section 1).

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

## 7. Stages

| Stage | What | Checked by |
|---|---|---|
| G0 | The build prototyped standalone (`scripts/experiments/neighbor-structures`) against the build of the matrix | Its time on Cellulose and JAC, before it is written as a template |
| G1 | The compact order of the places (for the matrix as well); the structure and its build on a device; the exchange contracts on `md_exec.pair_for` | The pairs of the list within the cutoff are those of the matrix, on JAC and Cellulose; the matrix in the compact order against the order of cells |
| G2 | The loop over groups, each pair once, both modes; the cheaper erfc of the direct sum under `fast_math`, with the accuracy of the force near the cutoff checked (it is what lets the loop over groups gain, Section 1) | Forces, energies, and virials against the matrix (to the rounding; to the bit in the deterministic mode, where the matrix sums in fixed point too); conservation over runs |
| G3 | Groups by default on a device for the loops that allow them | The Amber suite against pmemd.cuda and GROMACS |
| G4 | The reach of the list: a skin of 1 Å, once builds are cheap | Rates and intervals between builds |
| G5 | The arithmetic of the kernel (a table for the erfc, the order of the entries) | The prototype first |

## 8. Open questions

- The size of a group: 16 measured best among 16 and 32 on Cellulose; 8
  is untried.
- Why the same loop over groups of 16 measures 996 to 1034 µs in
  `supercluster.cu` and 1162 to 1209 in `groups.cu`.
- Whether small systems (JAC) keep the matrix, whose loop reads a list that
  fits the cache: groups of 32 took 289 µs on JAC against 104 for the
  matrix (too few warps); groups of 16 are unmeasured there.
