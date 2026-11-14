# D[device-pool-split]

Recorded at its merge.

**The runtime's pool of device memory is a best-fit allocator with splitting and immediate coalescing [[Wilson1995]](../references.md#wilson1995), over the allocations that it takes from the driver. A request takes the smallest idle piece that holds it, cut at its size rounded up to 256 bytes. A freed piece merges at once with the idle pieces next to it in its own driver allocation. Only a wholly idle allocation goes back to the driver.** Issue #890, after D411 (#876) and D414 (#885).

## 1. The rule

- **Arenas.** A request that no idle piece can hold takes an allocation
  of its own size from the driver (`cuMemAlloc`), as before. That
  allocation is an *arena*. No other driver call is added, and the
  step's requests are served from the pool without one.
- **Best fit.** A request of $s$ bytes takes the idle piece of the
  smallest extent $e \ge s$. Among pieces of one extent it takes the last
  in the list of idle pieces. A piece of exactly $s$ bytes is therefore
  taken first, as the pool took before it split.
- **Splitting.** Let $u$ be $s$ rounded up to a multiple of $A = 256$
  bytes. If $e - u \ge A$, the request takes the first $u$ bytes and the
  rest stays idle. Otherwise the request takes the whole piece, and its
  last $e - s$ bytes are padding. The driver aligns an allocation to at
  least 256 bytes, and every cut is at a multiple of $A$ from the start
  of the arena, so every piece is aligned to $A$.
- **Coalescing.** A freed piece merges at once with the idle piece that
  ends where it begins and the one that begins where it ends, if they are
  in its arena. It never merges with a piece of another arena, even one
  that the driver placed right after it: the driver frees an allocation
  only whole. Two idle pieces of one arena are therefore never next to
  each other.
- **Return to the driver.** Only a whole arena goes back. An idle piece
  of an arena that still holds a block in use stays in the pool until
  the whole arena is idle.

## 2. What goes back to the driver

- **The budget** (`MDIR_DEVICE_POOL_BYTES`, half of the device by
  default). An arena that becomes wholly idle goes back to the driver if
  the idle pieces, it included, would exceed the budget. Let $S$ be the
  bytes of the wholly idle arenas that the pool keeps. Each was kept when
  the idle bytes, which include it, were within the budget, and no arena
  joins $S$ otherwise. So $S$ never exceeds the budget. Once no block is
  in use, the idle bytes are $S$, within the budget, as before. While a
  block of an arena is in use, its idle pieces cannot go back, and they
  may take the idle bytes past the budget.
- **A trim** (`mdir.trim_device_memory()`, `mdrtDeviceTrim`, and the
  retry after the driver refuses a request) waits for the device, then
  returns every wholly idle arena and the plans of cuFFT. The idle pieces
  of an arena that holds a block in use stay.
- **A buffer that a growth replaces** (D312, amended). A buffer of the
  runtime's neighbor structures that is an arena of its own goes back to
  the driver at once, as before. A buffer that is a piece of a larger
  arena cannot go back alone. It returns to the pool, where a smaller
  request may take it, and its arena follows the budget's rule.
- **Measured against the rule of D312 unchanged.** A variant made these
  buffers arenas of their own: such a request took an idle arena of
  exactly its size or a new one, never a piece, so that a growth always
  gave the old buffer back. It held more, as the buffers of the groups
  could not use the idle pieces. On D411's four programs its peak with the
  pool was 995,656, 3,162,836, 30,678,656, and 1,442,513,008 bytes,
  against 985,452, 3,162,836, 30,062,108, and 1,426,518,020 for the rule
  above. At STMV that is 16.0 MB more, a slack of 25.1 MB against 9.1. On
  the input of `Driver/pool-retention-gpu.test` it held 75 and 78 arenas
  against 70 and 73 (below). It was not kept.

### The test of the pool's growth with the run

`Driver/pool-retention-gpu.test` compares the blocks that the driver
holds (taken less returned) after a short and a long run of the
dipeptide with groups, with a slack of 2. Its short run was of 100
steps. That run ends before the run's peak in use, which is reached
between steps 100 and 300 (1.0 and 1.1 MiB). The blocks held rise there
once, and then stay, three runs each on an RTX 4090:

| Steps | 100 | 300 | 1,000 | 3,000 | 10,000 |
|---|---:|---:|---:|---:|---:|
| `main`, blocks | 95 | 97 | 97 | 97 | 97 |
| this change, arenas | 70 | 73 | 73 | 73 | 73 |
| the variant of D312 unchanged (above), arenas | 75 | 78 | 78 | 78 | 78 |

From 100 steps, this change and the variant are past the slack by one (3
against 2). The short run is now of 1,000 steps, after the rise. Both
trees then hold the same count in the short and the long run (3,000
steps), and the slack of 2 stays. The defect that the test guards against (D312) left
blocks in the pool at each build, and grew with the run past 1,000
steps (783 MiB at STMV after 1,000 steps, more after). It was not run
again here.

## 3. The order of the streams

**The rule.** A freed piece, or a part of one, is handed only to work
issued after the free. That work runs after the work that used the
piece:
- on the same stream, by the order of the stream;
- on a stream that the runtime has changed to since, through the event
  that the change records on the old stream and the new stream waits
  for (`mdrtUseStream`, `mdrtRestoreStream`).

The premises:
- All allocations and frees are issued on the runtime's one stream. The
  stream operand of `mgpuMemAlloc` and `mgpuMemFree` is null in the
  lowered code.
- The second stream of D87 allocates and frees nothing (premise P4,
  checked by the lowering).
- A buffer that a step uses is not freed before its last use is issued
  (D412, D414).

**Merged pieces.** Every byte of a merged piece was freed before the
request that takes it was issued. The order above therefore covers each
part of the merged piece, and no event per piece is needed.

**The test** (`test/Runtime/device-blocks-gpu.test`, `streams`):
1. On stream A, a host function that sleeps half a second holds back a
   write of 1 to the first half of an arena. That half is then freed.
   The other half was freed before, with no work pending.
2. The two halves merge into the whole arena.
3. The runtime changes to stream B. A request of the whole arena takes
   the merged piece, and B writes 2 to all of it.
4. After both streams, every value is 2.

A pool that handed the piece out with no order between the streams lets
A's late write land last. The mutant that skips the event wait of the
change of stream fails the test: the values are 1 and 2 (Section 7).

## 4. The accounts

- **In use** is the sum of the bytes asked for by the blocks in use, as
  before. So the peak in use is unchanged to the byte.
- **With the pool** is the bytes of the arenas that the driver holds.
  That is the bytes in use, the idle pieces, and the padding of the
  pieces in use. Before, it was the bytes in use and the idle blocks, the
  same quantity, since a block was its own allocation and had no
  padding.
- **The breakdown** (`MDIR_DEVICE_MEMORY=2`) lists the blocks in use and
  the idle pieces at each peak, and adds a line of the padding at the
  peak with the pool. The summary adds the requests split from a larger
  piece. `mdrtDeviceMemoryCounts` counts the idle pieces ("idle") and
  the arenas ("driver blocks").

## 5. Peak device memory

**How it was measured.** `mdir run` of D411's four control files (200
steps, `[output] energy_interval = 0`, mixed precision) on an RTX 4090,
with `MDIR_DEVICE_MEMORY=2`, built from `main` 9c10a859 and from this
change, alternated in one job. The bytes are the sums of the runtime's
tables of the blocks at each peak: in use, idle, and padding. `main`'s
numbers are those of D414's change (#885, in `main` since) to the byte.

| | Dodecahedron, groups | Dodecahedron, matrix | JAC, groups | STMV, groups |
|---|---:|---:|---:|---:|
| peak in use, both | 933,696 | 3,120,456 | 28,768,536 | 1,417,374,424 |
| peak with the pool, `main` | 1,147,080 | 3,254,356 | 32,743,444 | 1,509,967,188 |
| peak with the pool, this change | 985,452 | 3,162,836 | 30,062,108 | 1,426,518,020 |
| **saved with the pool** | **161,628 (14.09%)** | **91,520 (2.81%)** | **2,681,336 (8.19%)** | **83,449,168 (5.53%)** |
| slack, `main` (with the pool − in use) | 213,384 (18.60%) | 133,900 (4.11%) | 3,974,908 (12.14%) | 92,592,764 (6.13%) |
| slack, this change | 51,756 (5.25%) | 42,380 (1.34%) | 1,293,572 (4.30%) | 9,143,596 (0.64%) |
| of which idle pieces | 66,104 | 57,868 | 1,420,212 | 11,816,360 |
| of which padding | 1,492 | 352 | 2,360 | 3,444 |
| requests split from a larger piece | 941 | 271 | 792 | 437 |
| arenas the driver holds at the end, `main` / this change | 64 / 37 | 49 / 38 | 105 / 63 | 110 / 71 |
| the driver's use of the device at its most, `main` / this change (MiB) | 407.1 / 407.1 | 411.1 / 411.1 | 451.1 / 449.1 | 2,303.1 / 2,205.1 |

The percentages of the slack are of the peak with the pool of the same
column. The slack of a column can be smaller than its idle pieces: at the
peak with the pool the bytes in use are below the peak in use.

**The slack that is left.** The peak with the pool is reached at a
request that takes a new arena. The request takes one only if no idle
piece holds it. So every idle piece at that peak is smaller than the
request that set it. At STMV the idle pieces at that peak are six of
1.07 to 3.20 MB and smaller ones of 1.31 MB in all. Two idle pieces of
one arena are never next to each other, so each of these lies between
blocks in use, or between a block in use and the end of its arena. It
can serve only a smaller request, or merge once a neighbor is freed. The
padding is at most 255 bytes a piece in use.

**The driver's use of the device** moves in steps of 2 MiB. It falls by
98 MiB at STMV and by 2 MiB at JAC, and the dodecahedra stay within one
step.

**Where the saving comes from.** On `main` the slack is the setup's
buffers of the device tuples, the per-step buffers of other sizes (D411),
and the blocks that D414 releases after the setup and that no later
request matched: 25.7 MB of them at STMV (D414, Section 2). These now
serve smaller requests. The idle bytes at the peak with the pool fall
from 92.6 MB to 11.8 MB at STMV.

## 6. The same kernels, the same trajectories

- **The kernels and the host's code.** `mdir emit --stage=lowered` with
  D406's settings (`MDIR_GPU_BINARY=ptx MDIR_GPU_ARCH=sm_89
  MDIR_COMPILE_CACHE=off MDIR_SKIN_CACHE=off MDIR_SKIN_SEARCH=never`) on
  D414's ten programs: the four of Section 5 in mixed and in double
  precision, FactorIX NVE with the groups, and JAC NVE with the matrix.
  The lowered module of `main` and of this change is the same to the
  byte for every program, its 131 to 408 `gpu.binary` ops (the PTX of
  the kernels) and its host's code alike. The change is in the runtime
  alone.
- **The trajectories.** The five checkpoints were run in the
  deterministic mode with the compile cache off, `main` and this change
  alternated in one job on an RTX 4090: JAC NVE, Factor IX NPT, and
  Cellulose NVE of the Amber suite (200 steps each), the ala3
  minimization, and argon (200 steps, on the GPU). Every dataset of the
  final checkpoint is identical to the bit, 5 of 5: the positions,
  velocities, box, forces where written, and the fingerprints. The
  addresses of the buffers differ, since requests are now served from
  pieces of larger arenas (Section 5, the splits), and no result changes.

## 7. Tests and mutations

`test/Runtime/device-blocks-gpu.test` drives the pool through the
runtime's functions (`Inputs/device_blocks.py`). Every piece that a
scenario takes is checked to be aligned to 256 bytes. Each scenario fails
on a pool that breaks the property it names:

| Scenario | Checks | Refusal it tests |
|---|---|---|
| `split` | a request served from a larger idle arena, the next piece 1,024 bytes after a request of 1,000 bytes | |
| `tail` | a rest of 200 bytes is not cut off: the request takes the whole piece | a split below the alignment |
| `best` | of idle arenas of 16, 64, and 32 KiB, a request of 10 KiB takes the 16 KiB one | a piece that is not the best fit |
| `coalesce` | a freed piece merges with the idle pieces on both sides, and the arena serves a request of its full size without the driver | |
| `arenas` | two arenas that the driver placed next to each other never merge; a request of both takes a third | merging across arenas |
| `trim` | a trim returns the idle arena and keeps the one with a piece in use, then returns it once the piece is freed | a trim of a partly used arena |
| `pieces` (budget of 1 MiB) | an idle piece of an arena in use stays past the budget; the arena goes back once wholly idle | a piece returned alone |
| `off` (budget of 0) | a freed arena goes back at once; nothing is kept or split | |
| `replaced` | a growth returns a buffer that is an arena of its own to the driver, and a piece of a larger arena to the pool | a piece returned to the driver alone (D312) |
| `streams` | the order of the streams (Section 3) | a reuse before the work on another stream is done |

The earlier scenarios (`registry`, `budget`, `retry`) pass unchanged.
`test/Sanitizer/device-blocks-gpu.test` runs six of the scenarios under
memcheck with the check of leaks, and `streams` under racecheck.

**Mutations** (local, not committed). Each mutant was built as the
runtime's library alone and run against every RUN line of the test, on
an RTX 4090. The library built from this change in the same way passes
every line.

| Mutant | Fails |
|---|---|
| no coalescing | `split`, `coalesce`, `trim`, `pieces`, `replaced`, `streams` |
| no rounding of the cut to 256 bytes | `split` |
| the event wait of a change of stream skipped | `streams` (values 1 and 2) |
| a split that leaves a rest below 256 bytes | `tail` |
| a trim of a partly used arena | `trim` |
| merging across arenas | `arenas`, `best`, `trim`, `pieces`, and the earlier `registry`, `budget`, `retry` |
| first fit in place of best fit | `best` |
| a replaced piece returned to the driver alone | `replaced` |
| the budget applied to a piece of an arena in use | `pieces` |

## 8. compute-sanitizer

- **The tier** (`lit -Dsanitize=1 test/Sanitizer`, with the interpreters
  of PyTorch and h5py): 24 passed, 1 unsupported (JAX is not installed),
  on an RTX 4090. The job held the device for 4.8 minutes.
- **D411's four programs, with the default pool**, on an RTX 4090:
  memcheck on all four (STMV for 25 steps, the others for 200), 0 errors;
  racecheck on the dodecahedra and JAC (200 steps), 0 hazards. racecheck
  of STMV had not finished its first step after 20 minutes and was
  stopped; its kernels are those of JAC's kinds, and this change touches
  no kernel. The job held the device for 29 minutes, 20 of them in that
  run.
- memcheck sees an access beyond a driver allocation, not beyond a piece
  of one. An access past a buffer's end would have been past its own
  allocation on `main`, where D414's runs with the pool off found none.

## 9. Timing

**How it was measured.** The rules of the timing protocol: JAC NVE with
the groups (100,000 steps, a row of the log every 1,000) and STMV with
the groups (20,000 steps, a row every 200), mixed precision, on RTX 4090s
with `gpu-run --idle`, in six jobs. Each run's rate is the mean of its
windows after the first quarter. The mean after half of the windows
differs from it by at most 0.07% at JAC and 0.19% at STMV, so the rate is
stationary there; the mean of all the windows differs by up to 0.22% and
0.62%, as the card warms at the start of a run. The trees ran in alternating orders, ABBA and BAAB, so a drift of
the card through a job cancels. Two further jobs with ABAB orders, run
first, put every branch run after a `main` run and are not counted; they
gave the same sign.

| Job | JAC `main` (ms/step) | JAC branch / `main` | STMV `main` (ms/step) | STMV branch / `main` | spread of `main`'s runs, JAC / STMV |
|---|---:|---:|---:|---:|---:|
| 1 | 0.1384 | 1.0019 | 4.0302 | 1.0016 | 0.51% / 0.18% |
| 2 | 0.1356 | 1.0012 | 3.9984 | 1.0017 | 0.15% / 0.33% |
| 3 | 0.1382 | 1.0024 | 4.0301 | 1.0015 | 0.43% / 0.18% |
| 4 | 0.1351 | 1.0017 | 3.9884 | 1.0043 | 0.15% / 0.03% |
| 5 | 0.1383 | 1.0026 | 4.0276 | 1.0017 | 0.29% / 0.17% |
| 6 | 0.1337 | 1.0013 | 3.9683 | 1.0012 | 0.07% / 0.26% |

**The result.** The branch is slower by 0.12 to 0.26% at JAC and by 0.12
to 0.43% at STMV, 0.2% on average for both. That is within the spread of
`main`'s own runs in a job (up to 0.51%) and well within the spread
between cards (3.5% at JAC, 1.6% at STMV), but it has the same sign in
every job, so it is a real cost and not noise.

**Where it comes from.**
- **Not the pool's own work.** With `MDRT_PROFILE`, the step's calls of
  `mgpuMemAlloc` and `mgpuMemFree` (counted together, from runs of
  2,000 and 4,000 steps) are 12.3 a step at JAC and 15.5 at STMV, on
  both trees. Their host time is 0.6 µs a step at JAC against 0.4 on
  `main`, and 0.85 to 1.05 µs at STMV against 0.55: 50 to 70 ns a call
against 35, beside steps of 135 µs and 4 ms. No
  driver call is made in the step: the arenas that the driver holds are
  the same after 2,000 and 4,000 steps (71 at STMV), and the requests
  split from a larger piece grow by 1.8 a step.
- **The placement of the buffers.** Jobs 3 and 4 also ran this change
  with the fit limited to pieces of exactly the request's size, the same
  code with the addresses of `main`. It ran at 0.9991 and 1.0014 of
  `main` at STMV and 1.0011 and 1.0009 at JAC. Jobs 5 and 6 ran it with
  the cuts at 2 MiB instead of 256 bytes: 1.0024 and 1.0009 at STMV,
  1.0039 and 1.0028 at JAC, so the alignment of the pieces is not the
  cause. The time of each kernel (STMV, 2,000 steps, `MDRT_WAIT`, ABBA)
  shows the spreading of PME's charges 1.7% slower on the branch (210.6
  and 210.8 µs a call against 206.9 and 207.4), about 3.5 µs of the
  step. The other kernels differ within the spread of `main`'s runs.
  Which buffers of the spreading sit where, and why that costs, is not
  found.
