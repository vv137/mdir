# D[md-alg-segments-simt]

Recorded 2026-11-17.

**Slice S3 of D417. The SIMT schedule of `md_alg.traverse_pairs` over a list of units (`!mdrt.segmented_pairs`), called in place by `convert-md-exec-to-gpu` as a second schedule chosen at compile time by the type of the list. D415's schedule of the groups is not touched. The schedule's block and rotation are an entry of the tuning policies of D420. A test-only route of the builder (`MDIR_RIGID_UNITS_TEST`) forms the programs that run on it, so that the real loop over steps, the contract, and the timings are measured; S4b replaces it by `[performance] rigid_units`.** Issue #902, after D420 (the maintainer's order of 2026-11-16). The maintainer's rulings on #903: the test route (option (a), 2026-11-17); after the first per-kernel numbers met the stop rule, round 1 (#795's tunings of the slices and the pruning in the loop) and, if a gap remains, round 2 (the entries), one round each.

## 1. The schedule

`kernels::lowerTraversePairsSegmentsSIMT` (`lib/Conversion/MDExecKernels/TraverseSegmentsSIMT.cpp`) is a file of its own, so that the groups' schedule in `TraversePairs.cpp` keeps its bytes by construction. `Lowering::lowerTraversePairs` calls it for an op over a `!mdrt.segmented_pairs` list, with the list's buffers (`kernels::SegmentLists`). No runtime flag chooses between the two schedules.

**Work.** One launch for every segment that the op visits. A warp takes one record of at most 64 entries of one segment, found from the segment offsets, so the segment of a warp, and with it the shapes of its two units and its region, is uniform. Each segment that the op visits is a branch of its own; another has no code. The warps stride over the records that the device counts, as many as particles at most, so no count is read back.

**Lanes.** Lanes $u$ and $u + 16$ hold the unit at place $16g + u$ of the record's group: its anchor, the offsets $\mathbf h_k$ of its members, and the fields of its members that a slice reads. The entries are taken 32 at a time.

**The rotation.** The partner's anchor (with the entry's shift added once, in f32), its offsets, its fields, its mask, and the accumulators of its places turn through the half-warp in 16 shuffle steps. A turn tests bit $u$ of the turning mask; the lane's own unit must hold a unit.

**The sites (R2).** The offset $\mathbf c_p$ of each place of the lane's unit is formed once from the kind's weights $\mathsf W$ in the type, the offsets of equal weight summed first (D417, Section 4.3). The partner's places are formed from its turning offsets at each step, as #795 does. The extra point is formed in the kernel; its stored position is never loaded.

**Site pairs.** For each site pair of the segment's table the schedule emits only that slice of the region, behind its own guard $r_m^2 < r_c^2$. A site pair that the table does not hold emits no instruction. As in #795's schedule (round 1):
- the pairs go in an order where consecutive pairs add to different places, so that neighboring guards are independent;
- the bounded reads of a slice's tables (the radial tables of D94) are issued before the guard of the pair before it;
- the next-to-last pair, where it reads a table and is otherwise pure, is computed in every lane and kept by a select.

The second and third apply where the region adds to no sum: in the kernel of the steps that write the energy they took the registers above #795's (163 against 160), as #795's sums keep their guards.

**Writes (rule A).** A partner's place accumulators reach its members by $\mathsf W^{\mathsf T}$ once per entry, after its rotation, one atomic addition a member, every member's particle read before the first atomic. The lane's unit is written once per record, its two half-warp copies added first.

**Sums.** A sum that every slice computes in a narrower type and widens to yield it is summed in that type and widened once (D415's rule). The warp and block reductions of D415, then `finishSums`. `md-exec-fuse-loops` makes the sums of the segments that are only added together one sum: D405's generator gives one sum a segment, and three energies and three virials took the kernel of the energy steps to 168 registers.

**The rest with itself.** Segment $(K, K)$ has two shapes $m = n = 1$. It takes the same code of this schedule, with no offsets and no transpose. It is not D415's body; S5 makes the two one emitter.

**Frames.** Until S4a's writer packs them, a launch before the schedule writes, a thread a place: the anchor moved by the place's shift in the stored type and narrowed to f32, and the offsets $\mathbf h_k = \mathbf x_k - \mathbf x_0$ formed in the stored type and narrowed (D417, Section 4.3), with the fields of the members that the slices read. The kind of a place comes from the places where the ranks begin, which the build now keeps in a buffer of the list (`rankPlaces`, $K + 2$ values).

**The pruning in the loop** (round 1, D284's mechanism over units). Where the first traversal that takes a dual refresh's result is at its positions and in its cell, the refresh moves the reference points into their frames and leaves the pruning to the traversal. At a step where the refresh prunes, the traversal runs a second kernel:
- a warp a record walks its row, the records of its group in its segment, from the row's first, prunes each against its segment's inner reach, and numbers the survivors in that order;
- it keeps the 64 survivors from 64 times its place in the row, so that the row's first inner records hold full batches and the last the rest (#795's U10);
- it writes its count, counts its record as active, and evaluates its own records.

The records of a row are contiguous because the build's counting sort of the records now takes the row, $s G + g$, as its key (D419 took the segment). The segments keep their order and their offsets, so the serial definition and the coverage checks read the lists as before.

## 2. The parameters (D420)

A table of its own, `SegmentsPolicy` in `DevicePolicy.h`: `blockSize` (128) and `rotationLanes` (16), #795's values, and `columnUnits` (128), the units of a column of the plan of the build over units (D419), which the cubes cut along z. Only the build over units reads `columnUnits`; the groups build keeps D404's 64. It was swept over 64, 96, 128, 192, and 256 on the six systems of #795 (round 2, Section 7): at 64 the groups of units held 55 to 66% of their places, at 128 70 to 85%, and 128 gave the least time a call of the loop over pairs on four systems and one within 0.3% of the least on the other two. The default entry cites the timing protocol on RTX 4090s and covers `sm_89`; every other architecture takes it as a fallback. The option `segments-column-units` overrides it. `convert-md-exec-to-gpu` reads the entry for its `arch`. The op records the rotation that it was lowered with (`rotate(16)`), and the module's record of the policies (D408) has an entry for each traversal, which the log's line of the policies prints: `md_alg.traverse_pairs{determinism=unordered strategy=segments rotate(16) launch=(?,128,1)}`.

## 3. The test route

`MDIR_RIGID_UNITS_TEST=<k>[,<k>]` (the maintainer's ruling (a) on #903, `docs/environment.md`). With it, `TermNonbonded` builds the primary pair term with D405's generator (`buildKindPairsTerm`) over the rigid kinds of D403 that it names, as `mdir check` numbers them:
- the kinds' copies are tuple sets of the program (`kind_<f>`), the segments logical domains declared without tuples;
- the rest's anchors (#799) are a tuple set `kind_anchors`: a hydrogen whose bond to a heavy atom the constraints keep, and the hydrogens of a rigid water that no kind takes, with the radius of the longest such length;
- a kind's covering radius is the largest distance of a place from its anchor, from the constrained distances of the frame, plus $10^{-4}$ nm (#795's numerical margin);
- the Lennard-Jones coefficients of a taken kind are constants (E3): its types are constants of its slices, and the route refuses tunables, so the tables are fixed for the run.

It refuses, naming each, the CPU target, a precision other than `MIXED`, the deterministic mode, a minimization, a cell that is not periodic, `[free_energy]`, LJPME, a kernel of PME other than the Gaussian, frame, tunable, or derivative programs, a system without a topology, and a kind that the system lacks or that is named twice (`test/Driver/rigid-units-test-invalid.test`). There is no key; S4b replaces the variable and removes it, with no alias. Without it no program changes.

## 4. The fixes of D419's lowering

The route was the first to lower a real program over a list of units. Each fix is a commit of its own with a test that fails without it:

| Fix | What failed | Test |
|---|---|---|
| The uncarrying of a list from a loop over steps that carries other values | `convert-md-exec-to-gpu` erased a yield the builder had not made: every loop over steps of a program aborted | `test/Conversion/MDExecToGPU/segments-program-lower.mlir` (`@carried`) |
| The relations of a list of units on the host | a program's tuples are on the host (D253); the lowering refused them as host buffers | the same test (`@host`) |
| The renumbered members of the kinds, the exclusions, and the anchors | with `device-tuples` (D257) the host value of a renumbering holds the members in the order of the files; the list named the wrong particles wherever the program reorders them (on the dipeptide, Lennard-Jones 4.5e6 against 579 kcal/mol at the start) | `test/Integration/md-alg-segments-contract-gpu.test` |
| The tables that the regions of the segments read | the serial definition collected the captures of the groups' region alone, and read a table of the device on the host | the same test |
| A build of a list of units in place | the build outside a loop over steps (the evaluation before the first step) stayed a build, which the lowering refuses; it becomes an empty list and a refresh, as a build of groups does | `segments-program-lower.mlir` (`@in_place`) |

## 5. The virial of a site

**Which part each route reports.** The log's rows give the virial and the pressure with the constraints' part; the barostat takes the pressure (D264). The line "the diagonal of the virial at the start, without the constraints" gives the part of the forces alone. On the large OPC box at the start:

| | Without the constraints | With them (the row of step 0) |
|---|---:|---:|
| general, double | 242,385.43 | -27,460.89 |
| general, mixed | 242,385.39 | -27,460.98 |
| units, mixed | 204,025.23 | -27,460.93 |
| #795 | 204,025.37 | -27,460.52 |

**Why the parts differ.** An extra point of Amber's water is the function $\boldsymbol\phi$ of its frame $\mathbf x_s = \mathbf x_O + a\,(\mathbf u + \mathbf v)/\lvert\mathbf u + \mathbf v\rvert$, which is not linear in the frame. On the constrained frame it equals $\sum_k (\mathsf W)_{sk}\mathbf x_k$ with D403's constant weights. The general path moves the site's force $\mathbf F_s$ to the frame atoms by the Jacobian, $\mathbf F_k \mathrel{+}= \mathsf J_k^{\mathsf T}\mathbf F_s$, $\mathsf J_k = \partial\boldsymbol\phi/\partial\mathbf x_k$ (D298); the units and #795 by the weights, $\mathbf F_k \mathrel{+}= (\mathsf W)_{sk}\mathbf F_s$ (rule A). The difference is $\Delta\mathbf F_k = (\mathsf J_k^{\mathsf T} - (\mathsf W)_{sk}\mathsf I)\,\mathbf F_s$.
- Under a rigid motion of the frame, a translation $\delta\mathbf x_k = \mathbf t$ or a rotation $\delta\mathbf x_k = \boldsymbol\omega \times \mathbf x_k$, the site moves with it: $\delta\mathbf x_s = \sum_k \mathsf J_k\,\delta\mathbf x_k = \sum_k (\mathsf W)_{sk}\,\delta\mathbf x_k$, because $\boldsymbol\phi$ commutes with rigid motions and $\sum_k (\mathsf W)_{sk}\mathbf x_k = \mathbf x_s$ on the frame. So $\Delta\mathbf F$ does no work on a rigid motion: $\sum_k \Delta\mathbf F_k = 0$, and it exerts no torque.
- A force on the three atoms of a rigid frame that does no work on its rigid motions lies in the span of the gradients of its three constraints. SETTLE's forces of constraint therefore take it up: they differ by $-\Delta\mathbf F$ between the two routes, and the constrained motion, the positions after each step, is the same.
- The virial of the forces, $\sum_k \mathbf x_k \otimes \mathbf F_k$, differs by $\sum_k (\mathbf x_k - \mathbf x_O)\otimes\Delta\mathbf F_k$ (the sum of $\Delta\mathbf F$ is zero), and the virial of the constraints by its negative. The total, and the pressure, are the same.

Where the site is linear in its frame, as GROMACS's three-atom site of the mixture, $\mathsf J_k = (\mathsf W)_{sk}\mathsf I$ and neither part changes.

**What reads a part.** The pressure, the barostat, the virial column of the log, and the observables take the total. Only the line of the start reports the part without the constraints, whose split changes with the route; the change is in its meaning, not in a value that a run depends on. The contract therefore compares the total virial with the general path (Section 7).

## 6. Refusals

Each has a negative case (`test/Conversion/MDExecToGPU/traverse-segments-simt-invalid.mlir`): site pairs in a type other than f32; a field of 64 bits that a slice reads; a rotation other than the 16 lanes that the schedule holds; more than two kinds. The refusals before the schedule (the outer records of a dual list, the inner records of a list that is not, a segment beyond its build's reach) are those of the serial definition, and `traverse-segments-serial-invalid.mlir` now runs them under the schedule too. A stable visit and the pruning attribute of the groups cannot reach the schedule (no build of a list of units states `stable`, and the pass sets the attribute for the groups alone); they are assertions.

## 7. Gates

**The contract** (D415's five modes, mixed precision, the bound of D417, Section 6, with $n = 4096$; `test/Integration/md-alg-segments-contract-gpu.test`): the dodecahedron of TIP3P (outer and dual), the mixture of TIP3P, OPC, ions, and molecules of the rest ($K = 2$), and the dipeptide in OPC and in TIP3P. The largest difference is 0.0022 of its bound; the potential and the total virial against the general run 0.0041 of theirs. **D415's mutation** (the serial definition ignoring the sign of an antisymmetric destination) fails it: 3,720 times the bound on the dodecahedron (a force of 960.1, $3.28\times10^5$ against 88.1), while the reversed visit, which has the same mutation, stays within its bound. Tried once and reverted.

**The mechanism table** as structural checks (`test/Driver/segments-schedule-ptx.test`): no lookup in the windows of two waters (E3; with the folding removed, the test fails), 12 loads of f64 positions in the packing and none of a stored site, one guard for each site pair of a kernel's tables (3, 13, 15), and 53 index shuffles of the rotation.

**Refusals**: each a case of `traverse-segments-simt-invalid.mlir`, `traverse-segments-serial-invalid.mlir` (also under the schedule), and `test/Driver/rigid-units-test-invalid.test`.

**Sanitizer** (`test/Sanitizer/segments-traverse-gpu.test`, with `-Dsanitize=1`): memcheck, racecheck, and initcheck on the dipeptide in OPC and the rigid dodecahedron, which prune in the loop. initcheck found reads of the places of a group that the packing had left unwritten (11,140 on the dipeptide), read before the masks select them; the packing writes zeros there now. The test held the device for 72 s (RTX 4090).

**Performance** (RTX 4090, `sm_89`; the RTX 5090 that the first measurements used went down during the protocol, and its preliminary numbers are on the PR, not here). The protocol: two jobs on two devices, #795 at 745c9f12, `main` at bab3f1fe, and the units interleaved, two rounds each, the second reversed; wall runs of 200,000 steps (100 intervals; the start trimmed by tenths until the mean of the rest moves by less than 0.3%) and captures of 60,000 steps (trimmed alike, to 0.5%); every run became stationary. The evaluation a call, units against #795, paired within a job and round:

| System | Round 1 | Round 2 (`columnUnits` 128) | Wall, round 2 |
|---|---:|---:|---:|
| rd-opc | 0.930 (0.924–0.938) | 0.885 (0.879–0.889) | 1.031 |
| rd-tip3p | 1.063 (1.054–1.074) | 1.027 (1.017–1.035) | 1.081 |
| ubq-opc | 1.014 (1.004–1.026) | 0.878 (0.873–0.885) | 1.027 |
| ubq-tip3p | 1.073 (1.068–1.079) | 0.999 (0.991–1.007) | 1.055 |
| opc-large | 1.073 (1.015–1.132) | 0.996 (0.945–1.043) | 1.047 |
| charmm-tip3p | 1.062 (1.050–1.079) | 0.937 (0.933–0.945) | 1.007 |

Round 1 ported #795's tunings and the pruning in the loop; round 2 the column units. On rd-tip3p the gap remains beyond the spread after round 2, so the stop rule applies (the PR, `needs-decision`). Per kernel on rd-tip3p: the loop of a step, 72 registers, 1,176 SASS instructions, 36.57 µs a call, against #795's 96, 2,448, and 37.25; the loop of the steps that prune (one in 8.6), 96 registers and 49.78 µs. On ubiquitin in OPC: 96 registers, 1,288 instructions, 38.22 µs, against 128, 2,456, and 44.98. The kernels a step are fewer than #795's on every system; the lists cost 8 to 9 µs a step against #795's 11 to 16.

**The step** (nsys, the middle 80% of the steps):

| | #795 | `main` | units |
|---|---:|---:|---:|
| ubq-opc NVE: a step, µs | 151.2 | 174.2 | 168.5 |
| device idle a step, µs | 19.5 | 18.4 | 38.9 |
| kernels a step (launches) | 26.9 | 27.4 | 28.9 |
| graph launches; streams | 0; 1 | 0; 1 | 0; 1 |
| blocking copies a step (µs) | 0.06 (10.0) | 0.10 (9.4) | 1.04 (60.1) |
| overlap of direct and reciprocal sums | 0 | 0 | 0 |
| integration and constraints a step, µs | 22.6 | 22.6 | 22.6 |
| ubq-opc NPT (C-rescale): a step, µs | 436.5 | 274.6 | 276.6 |
| blocking copies a step under NPT (µs) | 0.83 (215.5) | 0.88 (38.3) | 2.06 (86.7) |

The units' copy of every step is the refresh's test of D419, whose reduction the host reads; S4a's test in the writer removes it. Under NPT the readbacks after a change of the cell come on top, in `main` as in the units. S3's code adds no copy. Candidates for issues, not opened: the test of the refresh on the device (S4a; 19 µs a step of idle device on ubiquitin in OPC, about 13% of the step); the reciprocal sum on a second stream beside the direct sum (D87, off by default; at most the smaller of the two, 34 µs a step on ubiquitin, if they overlapped fully); a graph of the step's 27 to 29 launches (the idle time between kernels that remains with the copy gone, about 19 µs a step on #795 as on `main`).

**The rest with itself.** Segment $(0, 0)$, the rest with the rest, runs this schedule's body, not D415's: one schedule over every segment. Making the groups schedule the instance $K = 0$ is S5; the measurement here does not decide whether it comes before or after S4.

**Corpus** (D406, `check.py compare ... --stages lowered --targets cpu,gpu --kinds run,segments --precisions mixed,double --execution neighbor_structure=GROUPS --instances`, `MDIR_GPU_BINARY=ptx`, against `main`'s tree): see the PR for `sm_89` and `sm_120` at the head.

## 8. Not in this slice

- The validity in the writer, the packing of the frames there, and the check of the extents (S4a).
- The route's key, its refusals in `mdir check`, and its records (S4b).
- One emitter for the groups and the units (S5).
