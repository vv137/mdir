# D[matrix-no-host-wait]

Recorded 2026-11-21.

## Device decisions for matrix and group neighbor lists

Part of #924 (M2). This proposal changes GPU scheduling internally. It introduces no control keys, file formats, or overwrite behavior.

### Current host decisions

| Producer | Host use |
|---|---|
| `lib/Conversion/MDExecKernels/MatrixBuild.cpp: MatrixBuild::emitBuildResult` | Copies two i32 words and waits. In `lib/Conversion/MDExecToGPU/MDExecToGPU.cpp`, `Lowering::emitBuild` sends the invalid-position count through `emitStopNotNumbers` and compares the largest untruncated row count with the allocated width. On overflow it calls `runtime/mdrt_cuda.c: mdrtMatrixGrow`, reacquires the entries and repeats the build. |
| `lib/Conversion/MDExecToGPU/MDExecToGPU.cpp: Lowering::finishBuild` | Both structures snapshot positions on the device. The host reads no coordinates. It records the build cell, validity, build count and age. `runtime/mdrt_cuda.c: mgpuMemcpy` queues D2D without its host-memory completion wait. For the inner GROUPS reference, `Lowering::emitPruneFrames` saves positions inside `emitFramePositions`. |
| `lib/Conversion/MDExecToGPU/MDExecToGPU.cpp: Lowering::replaceWithFlags` | In `runtime/mdrt_cuda.c`, `mdrtFlagStart` records the producer event. `mdrtFlagFinish` waits at the first use and reads and clears the first mapped word. `mdrtFlagTake` consumes siblings after that wait. Unmapped words use a staged copy and event. The host ORs each nonzero flag with its initial condition. `Lowering::lowerRefreshNeighbors` selects reuse or rebuild. Dual GROUPS selects outer build plus prune, inner prune alone, or reuse, including fused pruning. |
| `lib/Conversion/MDExecKernels/GroupsBuild.cpp: GroupsBuild::emitBuild` | Copies five i32 words and waits. `Lowering::emitGroupsBuild` checks places and blocks against capacity, calls `mdrtGroupsGrow` and retries if needed. Completed-build statistics use block count, capacity, longest list, excluded-partner spill count and growth history. The fifth word feeds the invalid-position stop. Places, longest list and blocks also determine launch bounds, block-index storage and fused-prune selection in `getGroupsShape`, `emitGroupsBlockIndex` and `takesFusedPruning`. Excluded-partner spill has its own exact fallback. |

### Proposed scheduling and exactness

Decide on the device from the current coordinates before their first pair traversal. Combine the existing movement and initial invalidation bits per reference, join all producers with stream dependencies, then consume and clear once. Use a fixed submission sequence with device-conditional build and prune regions. Gate all scratch clears, scans, packing, reference updates and accounting. Preallocate common-path scratch. For GROUPS, launch over capacity and use device sizes for indexing, block-index extent checks and fused-prune selection.

Preserve first-build, cell-change and explicit policy semantics. For displacement-based reuse in an orthogonal cell, let $m$ be the least axis scale since the build, $R$ its reach and $d$ the largest displacement from a scaled reference. An omitted pair's current distance is bounded below by $mR-2d$. Reuse requires this bound to cover the cutoff with the existing boundary convention. A negative margin forces a build even at zero displacement. Retain triclinic validity and image checks. Apply each test to its own outer or inner reference and cell. Snapshot the original position values and matching cell only upon successful build or prune, before any subsequent displacement producer. Keep row order and force accumulation order.

After search, test untruncated required sizes and invalid-coordinate status on the device. Publish only complete lists. On failure latch the first refresh, phase, required sizes and diagnostic in a generation-tagged record. For rare recovery, submit bounded batches ending before any unguarded host-visible effect. Guard all dependent work, including integration, side-stream work, RNG advances, counters and output. This preserves the exact failed configuration and completed prefix while queued suffix work becomes inert. Poll completion asynchronously and drain status before exposing results or returning. On overflow grow storage, refresh descriptors, retry that build on the saved configuration and submit the unexecuted suffix once. On fatal status surface the existing diagnostic. Recovery requires demonstrating that every dependent consumer observes the latch.

### Validation planned

- Force matrix overflow after a successful narrow build with an interacting partner beyond the stored prefix. Force GROUPS place, block and block-index capacity exhaustion separately. Compare recovered entries, forces, counters and state with ample initial storage. Inject an invalid coordinate at the final refresh of a batch and verify the diagnostic precedes output.
- Move a pair from outside build reach into cutoff in one step immediately after reuse. Cover threshold neighbors, periodic images, box contraction, triclinic invalidation, shared flags, inner-only prune and simultaneous outer/inner invalidation in separate and fused pruning. Verify reference preservation on reuse and zero missing pairs against CPU all-pairs enumeration at every force evaluation.
- Compare short deterministic MATRIX and GROUPS trajectories with the base on the same GPU in mixed and double precision. Require bitwise coordinates, velocities, forces and energies and identical build/prune steps. Check RNG continuation across mid-batch recovery. Compare forces with the independent CPU oracle at existing precision-specific tolerances.
- Inspect lowered IR and Nsight Systems traces for common-path status copies and neighbor event waits. Report per-build idle, per-step wait and step time on TIP3P 10k, TIP3P 20k and JAC for both structures on RTX 4090. Interleave base and feature in two jobs with reversed order, trimmed stationary means and recorded clocks. Run the sanitizer tier and full suite at the final head.
