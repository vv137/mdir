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
