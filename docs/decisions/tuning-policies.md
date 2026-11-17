# D[tuning-policies]

Recorded at its merge.

**The thresholds and launch parameters of the device lowering are the entries of one table for each strategy, after the dispatch policies of CUB: the pruning in the loop over pairs (D284), the scatter-accumulates (D282, D283, D291), and the launches of the loops. A table is keyed by the architecture of the target and by a device class. Each entry cites the measurement behind it: decision, systems, devices, and date. An architecture without an entry takes the table's default. The choosers and the lowering read only the selected entries, and no threshold stays a literal in a lowering. The string test of `getPipeline()` on `cubin-chip=sm_12` is gone: the pruning table has an entry for the major version 12, so every `sm_12x` target takes 20,480 groups as before, the RTX 5090 included. No value changes, and the environment overrides keep their names and effects.** Step 1 of #433 (the maintainer's order of 2026-11-16), after S2 of D417 and before S3, which takes its parameters from an entry of its own from its first commit.

## 1. The tables

`include/mdir/Dialect/MDExec/DevicePolicy.h` and `lib/Dialect/MDExec/DevicePolicy.cpp` hold them. Each table has one record type for its strategy:

| Table | Record | Default entry | Other entries | Read by |
|---|---|---|---|---|
| Pruning in the loop (D284) | `PruningPolicy`: `fusedGroups` | 4096 (RTX 4090, `sm_89`) | major version 12: 20,480 (RTX PRO 6000 Blackwell Max-Q, `sm_120`) | `getPipeline()`, which writes `fuse-pruning-groups=N` as before |
| Scatter-accumulates (D282, D283, D291) | `ScatterPolicy`: `tileMinPoints`, `tileEdge`, `hashLeast`, `hashProbes`, `pairHashLeast` | 500,000, 8, 100,000, 8, 0 (RTX 4090 and RTX PRO 6000) | none | `md-exec-choose-scatter` |
| Launches | `LaunchPolicy`: `blockSize`, `rowLanes`, `stageBlockSize`, `blockMemoryBytes`, the lanes of PME's gathering | 128, 16, 256, 48 KiB, 4 lanes up to order 4 and 8 above (not measured as a choice) | none | `md-exec-choose-scatter` (the slots of a table), `convert-md-exec-to-gpu`, `md-exec-lower-stage-bodies`, the precondition checks of `Scatter.h` |

## 2. The key and the selection

- **The architecture** is the name that the target states, `sm_<major><minor>` with an optional suffix of letters (`sm_120a`). It is parsed into a major and a minor version. A name that does not parse, and no name (PTX for the default architecture), take the default entries.
- **The device class** is a string, empty for every device of the architecture. The RTX 5090 and the RTX PRO 6000 are both `sm_120`, and D284's 20,480 was measured on the PRO 6000 Max-Q. Every entry has the empty class for now, and nothing selects another: no key, option, or variable is added. A later step adds the classes that measurements tell apart, and the input that selects them.
- **The order** is: an entry of the architecture and the class, then one of the architecture, then one of its major version, then the default entry. The pruning's entry of the major version 12 is what `contains("cubin-chip=sm_12")` matched: `sm_120`, `sm_120a`, `sm_121`, and `sm_121a`.
- **A fallback** is a selection of the default entry for an architecture that the default was not measured on. The explain remarks of `md-exec-choose-scatter`, the only chooser that has them, say so on a choice made by a value of the scatter table: `(the default of the policy: sm_80 was not measured)`, or `no architecture given`.

## 3. How the passes read it

`getPipeline()` takes the architecture from the options of `mdir-gpu-lower-to-nvvm` (`getTargetArchitecture`, the value of `cubin-chip=`). It passes it as the option `arch` of `md-exec-choose-scatter`, `convert-md-exec-to-gpu`, and `md-exec-lower-stage-bodies` when there is one. Each option of these passes that a table now gives defaults to -1, "the entry for `arch`": `tile-edge`, `tile-min-points`, `hash-least`, `hash-probes`, `pair-hash-least`, and both `block-size` and `row-lanes`. An explicit value overrides the entry, as before. The 48 KiB of block memory is a parameter of `checkTile` and `checkHash` (`Scatter.h`), and the chooser and the lowering pass the same entry. `kernels::resolveTemplateParameters` takes the gathering's lanes from the lowering's entry.

## 4. The overrides

They keep their names and effects (docs/environment.md). `MDIR_FUSED_PRUNE=off` and `always` still give no option and `fuse-pruning-groups=0`. `MDIR_PME_SPREAD`, `MDIR_TUPLE_SUMS`, and `MDIR_PAIR_SUMS` still give `tile=`, `tile-edge=`, `hash=`, and `pair-hash=`. `MDIR_GPU_ARCH` names the architecture, and so the entries that it selects.

## 5. Not moved

These stay where they are. They are the geometry of one emitter or template, or a capability of the device, not a tuning:

- the 1024 parts of a reduction (`md_exec::maxReductionParts`, D364);
- the threads of the tile kernel for each edge (`getTileThreads`, the template's `TILE_THREADS`);
- the 256 threads of the counting kernels and the blocks of 1024 counts of the scans (`emitDeviceScan`, `lowerNonzero`);
- the rows of the batched bodies' reductions (`kRowsBlock`, `kWarp` in `StageBodies.cpp`);
- `kPairHashReach` (D291), which is a count of particles;
- the warp width of 32 in the NVIDIA emitters, which is a capability and comes from the target description of #558.

## 6. Validation

The gate is the corpus of D406, identical between the base and this branch, on `sm_89` and on `sm_120`. The PR names the runs and which run covers which threshold. A lit test checks the entries for each architecture through the printed pipeline and the remarks of the chooser, including 20,480 on `sm_120` and `sm_121a`, 4096 on `sm_89`, `sm_80`, and no architecture, and the fallback remark.
