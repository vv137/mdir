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

On an RTX 5090, the six systems of #894's step 1 as real unit systems. Each list replays a trajectory of 2000 steps of `main`'s general GROUPS run of the same system, its integrator, cutoff, and skin: the harness refreshes it at every frame, with its own test. The results are in PR #897; they are added here at the gate.

## 6. Not in this slice

- The SIMT schedule (S3).
- The test of validity in the loop that writes the positions, the writer, and the check of the extents (S4a).
- The route, the key, and `mdir check` (S4b).
