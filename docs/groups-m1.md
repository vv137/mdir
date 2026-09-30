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
the order.

## 4. What the loop reads

The positions are read in the order of the places, as the matrix reads them
(D86), and the entries hold places. The minimum image is taken for each
pair as for the matrix. The kernel is the semantic kernel of the loop,
lowered as for the matrix; the lowering adds the second accumulation and
the atomic additions.

## 5. The build on a device

For each group: its bounding box; the particles of the cells within R of
the box, as candidates; a candidate within 0.75 R of the box is taken with
all the bits of the group, one farther than R from the box is dropped, and
one between is tested against each particle of the group. The excluded
pairs of the group are compared with the candidates as in the build of the
matrix. The entries are written group by group; their number is not known
before, so each group reserves its room with an atomic addition, and the
loop takes the groups in any order.

Trivial acceptance sets bits of pairs up to about 1.5 R apart; they cost
the loop a test each and keep the build cheap. Its threshold is a
parameter to measure (Section 8).

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
| G1 | The structure and its build on a device; the exchange contracts on `md_exec.pair_for` | The pairs of the list within the cutoff are those of the matrix, on JAC and Cellulose |
| G2 | The loop over groups, each pair once, both modes | Forces, energies, and virials against the matrix (to the rounding; to the bit in the deterministic mode, where the matrix sums in fixed point too); conservation over runs |
| G3 | Groups by default on a device for the loops that allow them | The Amber suite against pmemd.cuda and GROMACS |
| G4 | The reach of the list: a skin of 1 Å, once builds are cheap | Rates and intervals between builds |
| G5 | The arithmetic of the kernel (the erfc of the direct sum, the order of the entries) | The prototype first |

## 8. Open questions

- The size of a group: 16 measured best among 16 and 32 on Cellulose; 8
  is untried.
- The threshold of trivial acceptance, against the slots of the loop.
- Whether small systems (JAC) keep the matrix, whose loop reads a list that
  fits the cache.
