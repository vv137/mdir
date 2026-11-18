# D[md-alg-segments-writer]

Recorded 2026-11-18.

**Slice S4a of D417. The writer of a list of units: one kernel a step, after the loop that writes the positions, packs the f32 frames of the units (D95; D417, Section 4.3), tests the units' reference points against the least half-margin of the segments (one mobility group, D419, Section 3), and checks that every place of every unit lies within its covering radius (the convex rule of D417, Section 3). Its three results are flags of D113, which the host reads where the refresh needs them, after the launches of the reciprocal sum (#795's U3). The refresh no longer tests or reads back, and the loops over the list read the writer's frames.** Issue #905, after D421 (the maintainer: "do S4a after the train", 2026-11-18). The test fused into the writer is a requirement of S4a (the leader's ruling on #897).

Status: design, before the implementation.

## 1. The op

```mlir
%moved, %stale = md_exec.write_unit_frames %l at(%x, %cell)
                   cutoff(0.9) skin(0.1) prune_skin(0.05)
                   : !sp, !md.field<@atoms, 3 x f64>
%l1 = md_exec.refresh_segmented_pairs %l at(%x, %cell) moved(%moved) stale(%stale)
        cutoff(0.9) skin(0.1) prune_skin(0.05) : !sp, !md.field<@atoms, 3 x f64>
```

- `md_exec.write_unit_frames` (new) takes a list of units and the positions and cell of a step. It writes the list's frames at those positions and returns:
  - `moved`: some reference point has moved, since the list was built, by more than the least half-margin of the segments, $\min_s \bigl(m(R + E_s) - (r_c + E_s)\bigr)/2$ with $E_s = e_f + e_g$ (D419, Section 3), or the list has not been built;
  - `stale`: the same against the last pruning of a dual list, with the inner reach;
  - it stops the run where a place of a unit lies beyond its covering radius (Section 3).
- `md_exec.refresh_segmented_pairs` takes `moved` and `stale` as `md_exec.refresh_neighbors` does (D80), and then makes no test of its own.
- `md-exec-expose-validity` makes the writer for each refresh of a list of units in the value form, in the refresh's block, right after the op that gives the refresh's positions (or at the start of the block), with the refresh's reaches. A refresh of an empty list builds whatever has moved and takes no writer. Nothing changes for a refresh of groups.

## 2. The writer's kernel

`convert-md-exec-to-gpu` lowers the writer to one kernel, a thread a unit, launched only once the list has been built (a host flag; the first refresh builds):
- **Frames.** For unit $u$ at its place $p$ (`placeOf`): the anchor $\mathbf x_0$ moved by the place's shift in f64 and narrowed to f32, and the offsets $\mathbf h_k = \mathbf x_k - \mathbf x_0$ of its members, formed in f64 as the minimum image of the bond and narrowed (D417, Section 4.3: the writer establishes that the members are stored whole about the anchor). The fields of the members that the loops over the list read (the charges and types of the rest), found from the traversals that take the refresh's result, are packed with them. The buffers belong to the list (the runtime's arena, which grows with the places; ABI 3), so the loops read them and pack nothing; after a build the refresh packs every place again, empty places as zeros.
- **The test.** The unit's reference point (its anchor, or the anchor of a member of the rest, #799) against the configuration of the last build, scaled with the cell (D80), in f64, against the least half-margin; with a dual list, against the last pruning too. A member of the rest whose reference point is another particle's is not tested: that particle's unit tests it.
- **Flags.** `moved`, `stale`, and the extent's failure are three flags of D113, set by plain stores (a mapped word takes no atomics). `replaceWithFlags` waits for them before their first use, the refresh, which follows the reciprocal sum's launches in the step: U3 of #795 by construction.

## 3. The check of the extents

Every place of a unit must lie within the covering radius $e$ of its reference point at the positions that the loops read (R5); the list's reaches assume it (D417, Section 1).
- **Taken kinds, the convex rule.** A site whose weights all lie in $[0, 1]$ is a convex combination of the frame atoms, so $\lVert \mathbf c_p \rVert \le \max_k \lVert \mathbf h_k \rVert$: checking the frame atoms, $\lVert \mathbf h_k \rVert \le e_f$, covers it. A site with a weight outside $[0, 1]$ is formed and checked explicitly. Which sites need it is decided at compile time from `#md.rigid_kind`.
- **The rest.** A member anchored to another particle is checked against $\rho$, its distance from its anchor (#795's U1).
- **Where.** In the writer, which holds every member of the unit after the last position constraint of the step, at the positions that the loops read. D417's table of Section 3 placed the check in the constraint kernel; the writer runs after it and before every loop over the list, so the check covers each step's position constraint without a second reading of the members, and the constraint kernels of programs without a taken kind keep their bytes. A unit beyond its radius stops the run with a message naming it; a stop of the constraints of the same step takes precedence (D278).

## 4. Refusals

Each with a negative case: a refresh whose `moved` does not come from a writer at its positions, its cell, and its reaches; a writer over a list that is not a list of units; positions not in f64.

## 5. Gates

As D417, Section 8, S4a, and the ruling on #897: the short trajectories of #795 (units against general, both integrators, NVE and NPT); the probes of the check of the extents; the coverage check of S2 along those trajectories; the corpus by D406's method on `sm_89` and `sm_120`; the sanitizer tier; S4a's gate on #795's six systems by the timing protocol; the step profile of D421 again.
