# D[md-alg-segments-simt]

Recorded 2026-11-17.

**Slice S3 of D417. The SIMT schedule of `md_alg.traverse_pairs` over a list of units (`!mdrt.segmented_pairs`), called in place by `convert-md-exec-to-gpu`, as a second schedule chosen at compile time by the type of the list. D415's schedule of the groups is not touched. The schedule's parameters are an entry of the tuning policies of D420.** Issue #902, after D420 (the maintainer's order of 2026-11-16).

Status: design, before the implementation.

## 1. The schedule

`kernels::lowerTraversePairsSegmentsSIMT` (`lib/Conversion/MDExecKernels/TraverseSegmentsSIMT.cpp`), a file of its own, so that the groups' schedule in `TraversePairs.cpp` keeps its bytes by construction. `Lowering::lowerTraversePairs` calls it for an op over a `!mdrt.segmented_pairs` list, through a `TraverseHost` of its own (the list's records, segment offsets, and table of members). No runtime flag chooses between the two schedules.

- **Work.** One launch for every segment that the op visits. A warp takes one record of at most 64 entries; a record lies in one segment (D419), so the segment of a warp, and with it the shapes of its two units and its region, is uniform. The record's segment is found from the segment offsets; each segment that the op visits is a branch of its own, and a segment that it does not visit has no code.
- **Lanes.** Lanes $u$ and $u + 16$ hold the unit at place $16g + u$ of the record's group: its anchor in the frame of its group, the offsets $\mathbf h_k$ of its members, and the fields of its members. The entries of the record are taken 32 at a time; the partner of each lane's entry is loaded once.
- **The rotation.** The partner's anchor (with the entry's shift added once), its offsets, its fields, its mask, and the accumulators of its places turn through the half-warp in $L$ shuffle steps. The bits of the pairs of a lane's unit are kept as in D415.
- **The sites (R2).** The offset $\mathbf c_p$ of each place of the lane's unit is formed once, from the kind's weights $\mathsf W$ in the type. The partner's places are formed from its rotating offsets at each step, as #795 measured faster than rotating them; either way the extra point is formed in the kernel and never loaded.
- **Site pairs.** For each site pair of the segment's table the schedule clones only that slice of the region, behind its own guard $r_m^2 < r_c^2$. The constants of a slice are those of the region (D405's generator writes a taken kind's charges and types as constants). A site pair that the table does not hold emits no instruction.
- **Writes (rule A).** A partner's place forces are written once per entry, after its rotation: $\mathsf W^{\mathsf T}$, then one atomic addition per member. The lane's unit is written once per record, its two half-warp copies added first.
- **Sums.** The warp and block reductions of D415, then `finishSums`.
- **The rest with itself.** Segment $(K, K)$ has two shapes $m = n = 1$. It takes the same code of this schedule, with no offsets and no transpose; it is not D415's body, which S5 unifies.
- **Frames.** Until S4a's writer packs the frames, a launch before the schedule writes, for each place, the anchor moved by the place's shift in f64 and narrowed to f32, and the offsets $\mathbf h_k = \mathbf x_k - \mathbf x_0$ formed in f64 and narrowed (D417, Section 4.3). It is timed apart from the evaluation.
- **The pruning in the loop** (`prune(G, R)`, D284): each record is pruned against its segment's inner reach, and the survivors of a row are re-chunked into full records (U10), as #795 does.

## 2. The parameters (D420)

A new table, `SegmentsPolicy`, in `DevicePolicy.h`: the block size, the lanes of the rotation (the width of `rotate(L)`), and the groups from which the loop prunes. Its default entry takes #795's values (128 threads, 16 lanes) with the measurement of this slice and the architectures that it covered; an architecture that it did not cover takes the entry as a fallback. `convert-md-exec-to-gpu` reads the entry for its `arch`; no literal of the schedule depends on the device.

## 3. Refusals

Each with a negative case in `test/Conversion/MDExecToGPU/traverse-segments-invalid.mlir`: the buffer form, the outer records of a dual list, $K > 2$, a site type other than f32, a stable visit, `rotate(L)` other than the policy's widths that the schedule implements, and a segment beyond its build's reach (D357).

## 4. Gates

As D417, Section 8, S3: the contract in D415's five modes; the mechanism table as structural checks of the PTX; registers, instructions, and time a call of every kernel next to #795's at 745c9f12; the timing protocol on #795's six systems; the stop rule; the sanitizer tier; the corpus on `sm_89` and `sm_120`.
