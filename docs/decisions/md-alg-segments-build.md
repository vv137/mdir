# D[md-alg-segments-build]

Recorded at its merge.

**Slice S2 of D417. The producer of a list of units is the groups build of D404 over units.**
- A unit is a copy of a taken kind or a particle of the rest, at its reference point. A group holds one kind.
- Each segment $s(f, g)$ is searched with its covering reach $R_s = R + e_f + e_g$, into records of its own, put in the order of the segments.
- The covering reach is one helper, which the build, the pruning, and the coverage query share.
- `convert-md-exec-to-gpu` lowers the list's empty list and its refresh: the unit tables, the test of the reference points, the build, and the pruning of a dual list.
- The coverage query has a case for each segment that a loop visits.
- `md-exec-reuse-neighbors` moves the build out of the loop over steps.
- The serial definition of D417 runs over the lists that the device builds, with `traverse-oracle=serial`. The SIMT schedule stays slice S3's.

Issue #894, after D418 (the maintainer's option (b) on #894).

## 1. The units and their reference points

The type `!mdrt.segmented_pairs` holds the $K$ taken kinds of D403. The particles of no taken kind are the rest, kind $K$.

| | A unit of taken kind $f < K$ | A unit of the rest |
|---|---|---|
| Unit | one tuple of the kind's relation, its $m_f$ frame atoms | one particle |
| Reference particle | the anchor, the first frame atom | its anchor from the (member, anchor) pairs of #799, or itself |
| Covering radius $e$ | the kind's `extent` | the build's `radius` $\rho$, or 0 without anchors |
| `order` at its place (S1's layout) | the slot of its first member in `members` | the particle |

The tables are topology, so they are formed once, on the host, when the empty list is lowered.
- The rest comes first, in the order of its particles. Then the copies of each taken kind, from $K - 1$ to 0, in the order of the relation.
- The rank of a unit is 0 for the rest and $K - f$ for kind $f$. The excluded pairs become the ragged rows of RaggedRows.h on the host.

**The compact order.** The build keys each unit by rank, then by column, so that a chunk of 64 places, and so a group of 16, holds one kind. Two choices were measured on the proxies of #894's step 1 and on the six real systems:
- **One plan of the columns for all the units, and the columns cut along z into cubes of their width.** A kind is sparse where the others are: a solute among water, water about a solute, or scattered ions. A chunk taken from a whole column of such a kind spans the cell along z.
  - Its group's box is then as wide as the brick, so the images kernel of D115 runs for it, over windows of the whole grid.
  - Before the cubes, the images kernel took 400 to 730 µs a build on the solvated systems. With them it takes a few µs.
  - The rest's cubes give a scattered ion a chunk of its own, which costs places, not time.
- **The kinds in descending rank in the places.** The groups' rule takes a candidate at place $q$ only if $q \ge 16g$. In this order, that rule alone visits each segment $s(f, g)$, $g \le f$, from the groups of kind $f$.

## 2. The segments

- **The covering reach** is `md_exec::getSegmentCoveringReach(R, e_f, e_g)` $= R + e_f + e_g$, added in this order, $e_f$ the radius of the larger kind (D417, Section 1).
  - The lowering computes it for each segment and passes it to the build.
  - The pruning takes it with the inner reach $r_c + \text{prune\_skin}$.
  - The coverage query checks each requirement against it.
- **The search.** The candidates of each rank are in cells of their own, of half the largest $R_s$. A group of kind $f$ searches, for each rank of the partners from its own on, the cells of that rank, with its segment's reach, widened as D404 widens the groups' reach. Its entries go into records of that segment alone.
- **Exclusions** apply only in segment $(K, K)$. A taken kind has no exclusion outside its molecule (D405). The test of a group of a taken kind, or of a candidate of a taken kind, takes none.
- **The records.** The list kernels take blocks of 64 entries in the order of their atomics, each named by its group and segment, $gS + s$. A counting sort by segment then gives the records in the order of the segments, as S1's serial definition reads them:
  - `segments[s]` is the first record of segment $s$;
  - `counts[w]` is the count of record $w$;
  - `units[w]` is its group.

  No order is kept within a segment, since the blocks came in the order of their atomics. A stable sort within the bins took 14 to 63 ms a build.
- **Images.** The images kernel of D115, with D418's restriction to the axes that need them, continues each list of a group in a segment. The shifts of the places and the entries are those that S1's serial definition reads.
- **The stop of the reach against the cell** is #795's. At every build, the largest covering reach must be at most half the least face width of the cell, or the run stops with both lengths.
- **The groups' instance keeps D404's bytes.** The build over units has its own name, `mdrt_gpu_build_segmented_pairs_k<K>` (with `_triclinic`). The helpers that both builds share are taken once in a module.

## 3. The refresh, the pruning, and the reuse

- **The test.** A refresh tests the units' reference points against those of the last build (#799). Its limit is half of $\min(m)(R + 2e) - (r_c + 2e)$, $e$ the largest radius. This is D80's limit for the groups when $e = 0$, and the least half-margin of the segments under a barostat.
- **The pruning.** With `prune_skin` below the skin the list is dual (D114).
  - The refresh tests the same points against the last pruning, with the inner reach.
  - A warp a record re-tests each entry at the positions of the reference points in the frames of their groups, against its segment's inner reach. The entries it keeps fill the front of the inner record.
  - The records that keep entries are the active units of D268.
- **`md-exec-reuse-neighbors`** turns a build of a list of units in a loop over steps into an empty list before the loop and a refresh in it. With its `prune-skin` below the skin, the refresh is dual.

## 4. Coverage (D357)

`verifySegmentCoverage(op, list, cutoff, segments)` walks the list to its build or refresh, through D357's boundaries. For each segment that the loop visits it requires

$$r_{c,\text{loop}} + e_f + e_g \le \texttt{getSegmentCoveringReach}(r_{c,\text{list}}, e_f, e_g),$$

with $e$ from the type's kinds and from the radius of the list's empty list or build. A failure names the segment and both lengths. A covering reach that leaves out a radius fails `test/Conversion/MDExecToGPU/traverse-segments-serial-invalid.mlir` (the mutant, tried once and reverted).

## 5. Measurements

### Setup

All runs are on an RTX 5090, by the timing protocol: two jobs of two rounds each, interleaved, the second round reversed. The six systems are those of #894's step 1, as real unit systems.

- **Units.** The list replays a trajectory of 2000 steps of `main`'s general GROUPS run of each system, with its integrator, cutoff, skin, and pruned distance. A harness refreshes the list at every frame, in the form that `md-exec-reuse-neighbors` gives a loop over steps: an empty list, then a refresh with its own test.
- **#795.** The binary at 004a0735, `water.template` on, 10,000 steps.
- **main.** 75d22231, the general groups build over all particles, 10,000 steps.

| System | Build | Builds every, steps | Prunes every, steps | List a step, µs | Test a step, µs | A build, µs | A pruning, µs | Slowest build kernel a call, µs (registers) |
|---|---|---|---|---|---|---|---|---|
| rd-opc | #795 | 58.2 (57.8–58.5) | 8.7 (8.6–8.7) | 8.57 (8.51–8.65) | in the writer | 397.8 (396.1–399.6) | 15.1 (14.9–15.2) | 219.7 (168) |
| rd-opc | main | 21.1 (21.0–21.3) | 2.8 (2.8–2.8) | 20.75 (20.72–20.78) | in the integrator | 216.8 (215.9–217.4) | 29.7 (29.6–29.8) | 151.0 (96) |
| rd-opc | units | 57.1 (57.1–57.1) | 8.6 (8.6–8.6) | 3.13 (3.13–3.14) | 4.50 (4.50–4.51) | 135.8 (135.6–136.3) | 6.5 (6.5–6.5) | 89.0 (128) |
| rd-tip3p | #795 | 53.6 (53.2–54.1) | 8.6 (8.5–8.6) | 9.19 (9.13–9.26) | in the writer | 392.4 (392.1–392.8) | 16.0 (15.8–16.1) | 212.0 (168) |
| rd-tip3p | main | 16.9 (16.8–16.9) | 2.8 (2.8–2.8) | 19.71 (19.61–19.81) | in the integrator | 183.6 (182.4–184.3) | 24.7 (24.7–24.8) | 133.9 (96) |
| rd-tip3p | units | 54.1 (54.1–54.1) | 8.6 (8.6–8.6) | 3.43 (3.43–3.43) | 4.45 (4.44–4.46) | 145.4 (145.4–145.5) | 6.4 (6.4–6.4) | 97.4 (128) |
| ubq-opc | #795 | 56.7 (56.5–56.8) | 8.5 (8.5–8.6) | 8.74 (8.61–8.90) | in the writer | 95.1 (94.9–95.3) | 60.3 (59.3–61.3) | 66.2 (167) |
| ubq-opc | main | 20.5 (20.4–20.5) | 2.8 (2.8–2.8) | 22.52 (22.44–22.60) | in the integrator | 242.1 (241.7–242.6) | 30.3 (30.1–30.4) | 170.0 (64) |
| ubq-opc | units | 55.6 (55.6–55.6) | 8.5 (8.5–8.5) | 3.62 (3.61–3.62) | 4.51 (4.50–4.51) | 152.3 (151.9–152.7) | 7.4 (7.4–7.5) | 97.9 (72) |
| ubq-tip3p | #795 | 52.3 (52.1–52.6) | 8.5 (8.4–8.5) | 9.61 (9.56–9.69) | in the writer | 96.1 (96.0–96.3) | 65.9 (65.3–66.3) | 66.7 (167) |
| ubq-tip3p | main | 16.5 (16.4–16.6) | 2.8 (2.8–2.8) | 19.58 (19.55–19.63) | in the integrator | 181.3 (180.9–181.6) | 24.1 (24.0–24.2) | 129.1 (64) |
| ubq-tip3p | units | 52.6 (52.6–52.6) | 8.5 (8.5–8.5) | 3.68 (3.68–3.69) | 4.46 (4.45–4.46) | 149.3 (149.0–149.7) | 7.2 (7.2–7.2) | 97.3 (72) |
| opc-large | #795 | 52.4 (52.1–52.9) | 8.2 (8.1–8.3) | 5.26 (5.22–5.30) | in the writer | 143.1 (142.9–143.5) | 20.7 (20.6–20.9) | 49.3 (168) |
| opc-large | main | 18.3 (18.2–18.4) | 2.8 (2.8–2.8) | 27.24 (27.17–27.38) | in the integrator | 270.8 (270.2–271.3) | 34.5 (34.4–34.6) | 196.1 (64) |
| opc-large | units | 54.1 (54.1–54.1) | 8.4 (8.4–8.4) | 3.74 (3.73–3.74) | 4.57 (4.54–4.66) | 136.1 (135.8–136.3) | 10.2 (10.2–10.3) | 85.5 (72) |
| charmm-tip3p | #795 | 52.0 (51.8–52.4) | 8.4 (8.4–8.5) | 11.83 (11.76–11.87) | in the writer | 130.4 (130.0–130.9) | 78.6 (77.4–79.5) | 93.9 (167) |
| charmm-tip3p | main | 16.5 (16.4–16.6) | 2.8 (2.8–2.8) | 33.24 (33.18–33.30) | in the integrator | 316.4 (314.5–317.9) | 39.3 (39.1–39.4) | 201.2 (64) |
| charmm-tip3p | units | 52.6 (52.6–52.6) | 8.5 (8.5–8.5) | 5.04 (5.02–5.07) | 4.49 (4.45–4.57) | 200.7 (199.6–202.2) | 10.5 (10.4–10.6) | 142.4 (72) |

### What it shows

- **Build frequency.** The units rebuild and prune as often as #795 does, within 1%. Both use the same reference points, the same skin, and one mobility group. `main` rebuilds every 16–21 steps and prunes every 2.8, since its hydrogens move.
- **The list a step** is the build plus the pruning: 3.1–5.0 µs, against #795's 5.3–11.8 µs.
- **The test of validity** runs as three kernels of its own, about 4.5 µs a step. #795 does it in its writer, and so does S4a; its gate is in D417, Section 11.
- **Which kernel dominates a build.** The list kernel takes 98 µs of a 152 µs build on ubq-opc, and 142 µs of 201 µs on charmm-tip3p. Its cells are larger there, since the reach is 1.5 nm.
  - The places of the cells of the candidates are ranked a thread a place (StableRank, #231). By insertion, a thread a cell, that sort took 63 µs (ubq-opc) and 77 µs (charmm-tip3p) a build.
- **Registers.** The list kernel takes 128 registers in a triclinic cell and 72 in an orthorhombic one. #795's search kernels take 167–168.
- **Entries at the first frame** (all frames in the direct check):

  | System | Units | #795 |
  |---|---|---|
  | rd-opc | 229,931 | 202,919 |
  | ubq-opc | 285,909 | 236,129 |
  | charmm-tip3p | 466,603 | 379,608 |

  - The pairs listed beyond the covering reach are 0.01–0.02% of the pairs within it.
  - The entries are more than #795's because the cubes leave groups partly filled: on ubq-opc 13,056 places hold 7,397 units.
  - Cubes of two widths took 5–12% fewer entries, but on rd-opc a build took 360 µs instead of 135 (`kUnitCubeHeight`).

## 6. Not in this slice

- The SIMT schedule (S3).
- The test of validity in the loop that writes the positions, the writer, and the check of the extents (S4a).
- The route, the key, and `mdir check` (S4b).
