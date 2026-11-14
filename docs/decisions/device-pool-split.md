# D[device-pool-split]

Recorded at its merge.

**The runtime's pool of device memory is a best-fit allocator with splitting and immediate coalescing [[Wilson1995]](../references.md#wilson1995), over the allocations that it takes from the driver. A request takes the smallest idle piece that holds it, cut at its size rounded up to 256 bytes. A freed piece merges at once with the idle pieces next to it in its own driver allocation. Only a wholly idle allocation goes back to the driver.** Issue #890, after D411 (#876) and D414 (#885).

The validation is in progress; its sections follow.

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
change of stream fails the test (Section 5).

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
numbers are D414's to the byte.

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
