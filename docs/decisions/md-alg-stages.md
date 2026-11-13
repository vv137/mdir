# D[md-alg-stages]

Recorded 2026-11-13.

**Step 2 of the `md_alg` layer (D406): three stage ops of a neighbor build, each written by hand at several sites of `main`, become typed ops with verifiers and a serial definition: `md_alg.sort_by_key` (a counting sort whose key a region computes), `md_alg.append` (entries appended to lists held in blocks taken from a pool, with a capacity), and `md_alg.grid_plan` (the counts of cells of a grid along the edges of the cell). The hand-written copies move onto them, and the PTX of every kernel stays that of the base.** Issue #864, part of #558, #604, and #817 (item 1).

The rules of D406 hold unchanged: an op's meaning names no warp, lane, or
block; the serial lowering to `scf` is the definition, and the SIMT
lowering is tested against it by the op's contract; the parameters of the
SIMT lowering are an optional execution group that the result does not
depend on, which the caller gives until a policy pass fills it from the
target (#433, #558; the review of #862 named this step: the group then
becomes an attribute that the pass fills, and the callers stop passing
it); options are compile-time, and an unused one emits no code.

## The ops

### `md_alg.sort_by_key`

```mlir
md_alg.sort_by_key %n bins %bins keys %key counts %held start %start
    cursor %cursor into %order by %ids
    launch(%grid_n, %block, %c1)
    scan chunked scratch %sums : memref<?xi32, 1>
        launch(%chunks, %chunk, %grid_chunks, %block, %c1)
    sort insertion launch(%grid_cells, %block, %c1)
    key {
  ^bb0(%i: index):
    ...
    md_alg.bin %k : index
  } : memref<?xi32, 1>, ...
```

The items are `[0, n)`; each has a key, a bin in `[0, bins)`, or none.

- **The key.** With the region `key`, the region runs once for each item,
  with the item as its argument, and assigns its key by executing exactly
  one `md_alg.bin %k` (the bin `k`, an i32 or an index, which must lie in
  `[0, bins)`) or `md_alg.no_bin` (no bin); `keys[i]` is then `k`, or −1.
  Without the region the keys are given in `keys`; a negative key is no
  bin, and with `every_key` every key is a bin (a precondition, not
  checked).
- **Counts.** `counts[b]` holds zeros on entry (a precondition: the op
  does not clear them, so that the clear can share a launch with other
  work, as at the sites) and the number of items of bin `b` afterwards.
- **Start and cursor.** `start[b]` = the sum of `counts[c]` for `c < b`,
  for `b` in `[0, bins]`; `cursor[b]` = `start[b + 1]` afterwards.
- **The order.** `out[start[b], start[b + 1])` holds the items of bin `b`
  ascending, or ascending by `by[item]` where `by` is given (the values of
  `by` within a bin must be distinct). No other element of `out` is
  written; no other buffer of the op is written.
- **What the region may do.** It computes the key of its item and may
  write side outputs, which the op declares (`writes(%wrapped, ...)`):
  every write of the region is to a declared side output, as a memref or
  through an address taken of one (`memref.extract_aligned_pointer_as_index`);
  it neither reads nor writes `keys`, `counts`, `start`, `cursor`, `out`,
  or the scatter buffer of the sort. It runs once for each item, in no
  fixed order and concurrently: a side output is written at places of its
  item, or by atomics whose sum does not depend on the order.
- **Contract:** ordered, stable: every lowering gives the serial
  lowering's `keys`, `counts`, `start`, `cursor`, `out`, and side outputs,
  bitwise.
- **Execution** (the SIMT lowering): `launch(grid, block, one)`, a thread
  an item, for the key and for the scatter; the scan's strategy, scratch,
  and launch, as for `md_alg.scan`; the sort's strategy and launch, as for
  `md_alg.sort_bins` (`stable_rank` takes the buffer that the scatter
  fills, `buffer %b`, and no `by`). The SIMT lowering is the key kernel
  followed by the code of `md_alg.scan`, `md_alg.scatter_to_bins` (with
  `every_key` where the region names no `md_alg.no_bin`), and
  `md_alg.sort_bins`; the serial lowering is a loop over the items running
  the region, then the serial definitions of those three.

Verifier rules: the execution group whole or absent, with the counts of
launch values and the scratch type of `md_alg.scan`; `stable_rank` with a
buffer and without `by`, `insertion` without a buffer; `every_key` only
without the region; the region one block of one index argument; on every
path through it exactly one `md_alg.bin` or `md_alg.no_bin`, directly in
it or in the branches of `scf.if` (not in a loop or another region); a
bin of i32 or index; no use of the op's own buffers inside it; every write
inside it to a declared side output; the side outputs none of the op's
buffers; `out` none of `keys`, `counts`, `start`, `cursor`, `by`.

### `md_alg.append`

```mlir
%count1, %last1 = md_alg.append %keep values(%place, %bits) into(%entries, %masks)
    list(%count, %last) owner %g names(%units, %ordinals)
    pool %sizes[%c2] capacity %capacity block 64
    lane(%lane, %below)
    : i32, i32, memref<?xi32, 1>, memref<?xi32, 1>, ...
```

Appends one entry (the `values`, one to each buffer of `into`) for each
item that keeps one, to a list held in blocks of `block` entries. The list
has `count` entries; `last` is the block that holds its last one (any
value while `count` is a multiple of `block`). A block is taken from the
pool, a counter at `pool[at]`, when the list's next entry begins one.

- **Serially, for each item in order:** if `keep`: with `s = count`,
  `k = s / block`, `j = s mod block`: if `j = 0`, the next block
  `b = pool[at]` is taken (`pool[at]` incremented), named
  (`units[b] = owner`, `ordinals[b] = k`) if `b < capacity`, and becomes
  `last`; the values are written at `last * block + j` if
  `last < capacity`; `count` increments. The results are `count` and
  `last` afterwards.
- **A batch.** The items that execute the op together are a batch, in an
  order, with the same `count` and `last`; its result is that of its
  items appended one at a time in that order. Serially a batch is one
  item; in the SIMT lowering it is the 32 lanes of a warp, in the order of
  the lanes, which execute the op convergently.
- **Capacity contract.**
  - `pool[at]` counts every block that is begun, whether or not it is below
    the capacity: afterwards it holds its value before plus the blocks
    that the appends began, the full demand, so that a caller can size
    the buffers and build again.
  - `count` is the full number of entries of the list, whether or not they
    were stored.
  - A block below the capacity is named and holds its entries; a block at
    or beyond it is neither named nor written. No element of the `into`
    buffers at or beyond `capacity * block`, and none of `units` or
    `ordinals` at or beyond `capacity`, is written, nor any slot of a
    block past the list's last entry.
  - What a caller may read: where `pool[at]` after every append is at
    most the capacity, every list whole (the blocks named by its owner, in
    the order of their ordinals, holding its `count` entries); otherwise
    only the counter and the counts, since which lists lost blocks depends
    on the order in which concurrent lists took them.
- **Order.** The entries of a list are in the order of its appends: the
  batches in the order in which they execute, and within a batch the order
  of the batch. The numbers of the blocks are not fixed: lists that run
  concurrently interleave their takes. *Contract:* the lists, the counts,
  and the counter bitwise; the numbers of the blocks as any distinct
  numbers, compared after the lists are read out through their names.
- **Execution:** `lane(%lane, %below)`, the lane's number in its warp and
  the mask of the lanes below it, which the kernels compute once and share.

Verifier rules: one buffer of `into` for each value, at least one;
`block` a power of two; `lane` and `below` together; `into`, `units`,
`ordinals`, and `pool` distinct. The SIMT lowering requires the execution
group and a `block` of at least 32 (so that a batch begins at most two
blocks).

### `md_alg.grid_plan`

```mlir
%nx, %ny, %nz = md_alg.grid_plan %lx, %ly, %lz width %w bounds [3.0, 256.0] : f64
%cx, %cy = md_alg.grid_plan %lx, %ly volume %v occupancy 64.0 of %n
    bounds [1.0, 256.0] : f64
```

The count of cells along each of the given edges `L_a`, as i64:
`min(max(floor(L_a / w), least), most)`, with `maxnumf` and `minnumf`, so
that a length or width that is not a number gives `least` and an infinite
ratio `most` (#168). The width is given, or follows from a target
occupancy: `w = (occupancy / (n / V))^{1/3}`, the edge of a cube that
holds `occupancy` of the `n` items in the volume `V` on average.

- *Contract:* bitwise; one lowering, on the host, which the serial and the
  SIMT passes share.
- Verifier rules: at least one edge; `least` and `most` whole numbers with
  `1 <= least <= most`; a width or an occupancy, not both; an occupancy
  with its volume and population, and positive.

The grid plan moves what the sites compute alike: the counts from a
width, clamped. What the sites choose differently stays with them: the
matrix's width from its cost of a search (`mdrt_gpu_cell_width`), the
groups' widths of a column (an occupancy of 64) and of the grid of the
candidates (half the reach).

## Sites

To be filled in with the implementation.

## What this enables

To be filled in with the implementation.

## Validation

To be filled in with the implementation.
