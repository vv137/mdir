// The md_alg dialect: the collection algorithms of the device lowering,
// target-neutral, each defined by a serial lowering (D406, #861).
//
// See docs/device-primitives.md, Section 4, and
// docs/decisions/md-alg.md for the design.

#ifndef MDIR_DIALECT_MDALG_MDALGOPS_TD
#define MDIR_DIALECT_MDALG_MDALGOPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def MDAlg_Dialect : Dialect {
  let name = "md_alg";
  let summary = "Collection algorithms, target-neutral, defined serially.";
  let description = [{
    The `md_alg` dialect states what a collection algorithm computes, below
    `md_exec` and above the device lowering: a scan, a scatter to bins, a
    sort within bins, and the accessors of ragged rows and of a periodic
    grid (D406, #861). An op's meaning names no warp, lane, or block.
    Its attributes state semantics and contracts: what the result is and
    how it is ordered.

    Each op has a serial lowering to `scf` (`convert-md-alg-to-scf`),
    which is its definition and the oracle of its tests, and a SIMT
    lowering (`convert-md-alg-to-gpu`) that calls the emitters of the
    device primitives (docs/device-primitives.md, Section 4). A test
    compares the two on random and boundary inputs, by the op's contract:
    bitwise for an ordered result, as multisets after a canonical sort for
    an unordered one.

    The parameters of the SIMT lowering (the strategy, the scratch buffer,
    and the launch geometry) are an optional part of a collection op. The
    result does not depend on them: the serial lowering ignores them and
    the SIMT lowering requires them. Until a policy pass fills them from
    the target (#433, #558), the caller gives them.

    The ops are created and lowered inside `convert-md-exec-to-gpu`, in the
    templates of the kernels and in the build of the groups of neighbors.
  }];
  let cppNamespace = "::mdir::md_alg";
}

class MDAlg_Op<string mnemonic, list<Trait> traits = []>
    : Op<MDAlg_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Enumerations
//===----------------------------------------------------------------------===//

def MDAlg_ScanStrategy
    : I32EnumAttr<"ScanStrategy", "how the SIMT lowering scans", [
        // Blocks of 256 threads, 4 counts a thread, sum their counts; one
        // block scans the sums; each block scans its counts from the sum
        // before it. Three launches.
        I32EnumAttrCase<"ReduceThenScan", 0, "reduce_then_scan">,
        // Blocks claim tiles in order and take the sum before them by
        // decoupled look-back (Merrill and Garland 2016). A launch that
        // clears the state, and one that scans.
        I32EnumAttrCase<"SinglePass", 1, "single_pass">,
        // A thread sums each chunk, one thread scans the sums, a thread
        // writes each chunk's offsets. For a few thousand counts.
        I32EnumAttrCase<"Chunked", 2, "chunked">
      ]> {
  let cppNamespace = "::mdir::md_alg";
}

def MDAlg_SortStrategy
    : I32EnumAttr<"SortStrategy", "how the SIMT lowering sorts within bins", [
        // A thread a bin sorts the bin in place by insertion. For bins of a
        // few items.
        I32EnumAttrCase<"Insertion", 0, "insertion">,
        // A thread an item counts the smaller items of its bin and writes
        // the item at that place of the bin in another buffer. For bins of
        // many items.
        I32EnumAttrCase<"StableRank", 1, "stable_rank">
      ]> {
  let cppNamespace = "::mdir::md_alg";
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

def MDAlg_I32Buffer : MemRefRankOf<[I32], [1]>;
def MDAlg_RealBuffer : MemRefRankOf<[F32, F64], [1]>;
def MDAlg_I32OrIndex : AnyTypeOf<[I32, Index]>;

//===----------------------------------------------------------------------===//
// Collection algorithms
//===----------------------------------------------------------------------===//

def MDAlg_ScanOp : MDAlg_Op<"scan", [AttrSizedOperandSegments]> {
  let summary = "The exclusive sums of i32 counts";
  let description = [{
    ```text
    start[i]  = counts[0] + ... + counts[i - 1]   for i in [0, n]
    cursor[i] = start[i]                           for i in [0, n)
    ```

    in i32 arithmetic, which wraps (two's complement). `start` holds
    `n + 1` elements, the sum of all at `n`; `cursor`, if given, `n`.
    No other element is written.

    *Contract:* ordered integer: every lowering gives the serial lowering's
    values, bitwise.

    *Execution* (the SIMT lowering, not the meaning): `strategy`, the
    `scratch` buffer, and the `launch` values.
    - `reduce_then_scan`: `scratch` holds the sums of the blocks, at least
      `blocks` i32; `launch(blocks, threads, one)` with `blocks` at least
      ceil(n / 1024) and 1, and `threads` = 256.
    - `single_pass`: `scratch` holds the tile counter and the descriptors of
      the tiles, `blocks + 1` i64, used by one scan at a time;
      `launch(blocks, threads, one)` as above.
    - `chunked`: `scratch` holds the sums of the chunks, `chunks` i32;
      `launch(chunks, chunk, grid, block, one)`. It requires the cursor.

    ```mlir
    md_alg.scan %counts[%n] into %start cursor %cursor
        strategy reduce_then_scan scratch %sums : memref<?xi32, 1>
        launch(%blocks, %threads, %c1)
        : memref<?xi32, 1>, memref<?xi32, 1>, memref<?xi32, 1>
    ```
  }];
  let arguments = (ins
    Arg<MDAlg_I32Buffer, "", [MemRead]>:$counts,
    Index:$n,
    Arg<MDAlg_I32Buffer, "", [MemWrite]>:$start,
    Arg<Optional<MDAlg_I32Buffer>, "", [MemWrite]>:$cursor,
    OptionalAttr<MDAlg_ScanStrategy>:$strategy,
    Arg<Optional<MemRefRankOf<[I32, I64], [1]>>, "", [MemRead, MemWrite]>:$scratch,
    Variadic<Index>:$launch
  );
  let assemblyFormat = [{
    $counts `[` $n `]` `into` $start (`cursor` $cursor^)?
    (`strategy` $strategy^)?
    (`scratch` $scratch^ `:` type($scratch))?
    (`launch` `(` $launch^ `)`)?
    attr-dict `:` type($counts) `,` type($start) (`,` type($cursor)^)?
  }];
  let hasVerifier = 1;
}

def MDAlg_ScatterToBinsOp : MDAlg_Op<"scatter_to_bins"> {
  let summary = "Each item to a slot of its bin, taken from a cursor";
  let description = [{
    For each item `i` in `[0, n)` with the key `k = keys[i]`: if `k` is not
    negative (or for every key with `every_key`), `i` is written to
    `out[s]` for a slot `s` taken from `cursor[k]`, which is incremented.
    An item with a negative key has no bin and is skipped.

    Afterwards, with `c_k` the items of key `k` and `cursor0` the cursor
    before: `cursor[k] = cursor0[k] + c_k`, the items of key `k` are
    `out[cursor0[k], cursor0[k] + c_k)`, and no other element of `out` is
    written. The spans must lie in `out` and not overlap, as the exclusive
    sums of the counts of the keys give them (`md_alg.scan` with a
    cursor); this is a precondition, not checked.

    *Contract:* unordered: the order of the items within a bin is not
    fixed. Every lowering gives the serial lowering's items of each bin as
    a multiset, its cursor bitwise, and leaves the same elements of `out`
    untouched. The serial lowering writes each bin's items ascending.

    *Execution*: `launch(grid, block, one)`, a thread an item.

    ```mlir
    md_alg.scatter_to_bins %keys[%n] cursor %cursor into %order
        launch(%grid, %block, %c1)
        : memref<?xi32, 1>, memref<?xi32, 1>, memref<?xi32, 1>
    ```
  }];
  let arguments = (ins
    Arg<MDAlg_I32Buffer, "", [MemRead]>:$keys,
    Index:$n,
    Arg<MDAlg_I32Buffer, "", [MemRead, MemWrite]>:$cursor,
    Arg<MDAlg_I32Buffer, "", [MemWrite]>:$out,
    UnitAttr:$every_key,
    Variadic<Index>:$launch
  );
  let assemblyFormat = [{
    $keys `[` $n `]` `cursor` $cursor `into` $out
    (`launch` `(` $launch^ `)`)?
    attr-dict `:` type($keys) `,` type($cursor) `,` type($out)
  }];
  let hasVerifier = 1;
}

def MDAlg_SortBinsOp : MDAlg_Op<"sort_bins", [AttrSizedOperandSegments]> {
  let summary = "The items of each bin sorted ascending, stably";
  let description = [{
    The items (i32) of each of the `bins` bins of `start` (`bins + 1`
    offsets), `items[start[b], start[b + 1])`, sorted ascending by the item,
    or by `by[item]` where `by` is given; items of equal value keep their
    order. In place, or, with `into`, written to the same places of `out`,
    `items` unchanged. No other element is written.

    *Contract:* stable sort: every lowering gives the serial lowering's
    values, bitwise.

    *Execution*: `strategy` and `launch(grid, block, one)`.
    - `insertion`: a thread a bin, in place; `grid` blocks hold `bins`
      threads. It takes no `into`.
    - `stable_rank`: a thread an item, into `out`; `grid` blocks hold
      `start[bins]` threads. It requires `keys`, the bin of each item
      (`keys[item]`, as the scatter took it), and the items of a bin
      distinct; it takes no `by`.

    ```mlir
    md_alg.sort_bins %start[%bins] items %order by %number
        strategy insertion launch(%grid, %block, %c1)
        : memref<?xi32, 1>, memref<?xi32, 1> by memref<?xi32, 1>
    ```
  }];
  let arguments = (ins
    Arg<MDAlg_I32Buffer, "", [MemRead]>:$start,
    Index:$bins,
    Arg<MDAlg_I32Buffer, "", [MemRead, MemWrite]>:$items,
    Arg<Optional<MDAlg_I32Buffer>, "", [MemRead]>:$by,
    Arg<Optional<MDAlg_I32Buffer>, "", [MemWrite]>:$out,
    Arg<Optional<MDAlg_I32Buffer>, "", [MemRead]>:$keys,
    OptionalAttr<MDAlg_SortStrategy>:$strategy,
    Variadic<Index>:$launch
  );
  let assemblyFormat = [{
    $start `[` $bins `]` `items` $items (`by` $by^)? (`into` $out^)?
    (`keys` $keys^)? (`strategy` $strategy^)?
    (`launch` `(` $launch^ `)`)?
    attr-dict `:` type($start) `,` type($items)
    (`by` type($by)^)? (`into` type($out)^)? (`keys` type($keys)^)?
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Stages (D[md-alg-stages], #864)
//===----------------------------------------------------------------------===//

def MDAlg_SortByKeyOp : MDAlg_Op<"sort_by_key", [
    AttrSizedOperandSegments, NoTerminator,
    AllTypesMatch<["keys", "counts", "start", "cursor", "out"]>
  ]> {
  let summary = "A counting sort of items by a key that a region computes";
  let description = [{
    The items `[0, n)`, each with a key: a bin in `[0, bins)`, or none.

    - The key: with the region `key`, which runs once for each item with
      the item as its argument, by exactly one `md_alg.bin %k` (the bin
      `k`, which must lie in `[0, bins)`) or `md_alg.no_bin` on every path
      through it; `keys[i]` is then `k`, or -1. Without the region the
      keys are given in `keys`: a negative key is no bin, and with
      `every_key` every key is a bin (a precondition, not checked).
    - `counts[b]` holds the number of items of bin `b` afterwards. The op
      clears the counts first; with `cleared` the caller has cleared them
      (so that the clear can share a launch with other work), and the op
      does not clear them again.
    - `start[b]` = the sum of `counts[c]` for `c < b`, for `b` in
      `[0, bins]`; `cursor[b]` = `start[b + 1]` afterwards.
    - `out[start[b], start[b + 1])` holds the items of bin `b` ascending, or
      ascending by `by[item]` where `by` is given (distinct within a bin).
      No other element of `out` is written.

    The region computes the key of its item from any memory but the op's
    own buffers, and may write two kinds of side output, which the op
    declares, so that the result does not depend on the order of the items:
    - a slot of its item in a buffer of `writes`: a `memref.store` whose
      first index is the item, or the item times a constant other than zero
      plus a constant, and whose other indices are constants;
    - an addition to a counter of `adds`: `memref.atomic_rmw addi`, or
      `llvm.atomicrmw add` at an address taken of the counter
      (`memref.extract_aligned_pointer_as_index`) plus a constant, which
      sum in any order to the same integers.
    Every other write is refused. The region runs once for each item, in no
    fixed order and concurrently.

    *Contract:* ordered and stable: every lowering gives the serial
    lowering's `keys`, `counts`, `start`, `cursor`, `out`, and side outputs,
    bitwise.

    *Execution* (the SIMT lowering, not the meaning): `launch(grid, block,
    one)`, a thread an item, for the key and for the scatter (and a thread
    a bin of blocks of `block` for the clear); the scan's
    strategy, scratch, and launch, as for `md_alg.scan`; the sort's strategy
    and launch, as for `md_alg.sort_bins`, where `stable_rank` takes the
    buffer that the scatter fills (`buffer`). The SIMT lowering is the
    kernel of the key followed by the code of `md_alg.scan`,
    `md_alg.scatter_to_bins` (with `every_key` where the region names no
    `md_alg.no_bin`), and `md_alg.sort_bins`.

    ```mlir
    md_alg.sort_by_key %n bins %cells keys %key counts %held start %start
        cursor %cursor into %order by %ids : memref<?xi32, 1> cleared
        launch(%grid_n, %block, %c1)
        scan chunked scratch %sums : memref<?xi32, 1>
            launch(%chunks, %chunk, %grid_chunks, %block, %c1)
        sort insertion launch(%grid_cells, %block, %c1)
        key {
    ^bb0(%i: index):
      ...
      md_alg.bin %k : index
    } : memref<?xi32, 1>
    ```
  }];
  let arguments = (ins
    Index:$n,
    Index:$bins,
    Arg<MDAlg_I32Buffer, "", [MemRead, MemWrite]>:$keys,
    Arg<MDAlg_I32Buffer, "", [MemRead, MemWrite]>:$counts,
    Arg<MDAlg_I32Buffer, "", [MemWrite]>:$start,
    Arg<MDAlg_I32Buffer, "", [MemRead, MemWrite]>:$cursor,
    Arg<MDAlg_I32Buffer, "", [MemRead, MemWrite]>:$out,
    Arg<Optional<MDAlg_I32Buffer>, "", [MemRead]>:$by,
    Arg<Variadic<AnyMemRef>, "", [MemRead, MemWrite]>:$writes,
    Arg<Variadic<AnyMemRef>, "", [MemRead, MemWrite]>:$adds,
    UnitAttr:$cleared,
    UnitAttr:$every_key,
    Variadic<Index>:$launch,
    OptionalAttr<MDAlg_ScanStrategy>:$scan_strategy,
    Arg<Optional<MemRefRankOf<[I32, I64], [1]>>, "", [MemRead, MemWrite]>:$scan_scratch,
    Variadic<Index>:$scan_launch,
    OptionalAttr<MDAlg_SortStrategy>:$sort_strategy,
    Arg<Optional<MDAlg_I32Buffer>, "", [MemRead, MemWrite]>:$sort_buffer,
    Variadic<Index>:$sort_launch
  );
  let regions = (region MaxSizedRegion<1>:$key);
  let assemblyFormat = [{
    $n `bins` $bins `keys` $keys `counts` $counts `start` $start
    `cursor` $cursor `into` $out (`by` $by^ `:` type($by))?
    (`writes` `(` $writes^ `:` type($writes) `)`)?
    (`adds` `(` $adds^ `:` type($adds) `)`)?
    (`cleared` $cleared^)? (`every_key` $every_key^)? `:` type($keys)
    (`launch` `(` $launch^ `)`)?
    (`scan` $scan_strategy^ `scratch` $scan_scratch `:` type($scan_scratch)
     `launch` `(` $scan_launch `)`)?
    (`sort` $sort_strategy^ (`buffer` $sort_buffer^ `:` type($sort_buffer))?
     `launch` `(` $sort_launch `)`)?
    (`key` $key^)?
    attr-dict
  }];
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

def MDAlg_BinOp : MDAlg_Op<"bin"> {
  let summary = "The bin of the item of a key region";
  let description = [{
    In the `key` region of `md_alg.sort_by_key`: the item's bin is `bin`,
    which must lie in `[0, bins)`. Exactly one `md_alg.bin` or
    `md_alg.no_bin` runs on every path through the region.

    ```mlir
    md_alg.bin %k : index
    ```
  }];
  let arguments = (ins MDAlg_I32OrIndex:$bin);
  let assemblyFormat = "$bin attr-dict `:` type($bin)";
  let hasVerifier = 1;
}

def MDAlg_NoBinOp : MDAlg_Op<"no_bin"> {
  let summary = "The item of a key region has no bin";
  let description = [{
    In the `key` region of `md_alg.sort_by_key`: the item has no bin; its
    key is -1 and it is in no bin of the order.

    ```mlir
    md_alg.no_bin
    ```
  }];
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def MDAlg_AppendOp : MDAlg_Op<"append", [AttrSizedOperandSegments]> {
  let summary = "Entries appended to a list held in blocks from a pool";
  let description = [{
    Appends one entry (the `values`, one to each buffer of `into`) for each
    item that keeps one (`keep`), to a list held in blocks of `block`
    entries. The list has `count` entries; `last` is the block that holds
    its last one (any value while `count` is a multiple of `block`). A
    block is taken from the pool, the counter `pool[at]`, when the list's
    next entry begins one.

    Serially, for each item: if `keep`, with `s = count`, `k = s / block`,
    and `j = s mod block`: if `j = 0` the next block `b = pool[at]` is taken
    (`pool[at]` incremented), named (`units[b] = owner`,
    `ordinals[b] = k`) if `b < capacity`, and becomes `last`; the values
    are written at `last * block + j` if `last < capacity`; `count`
    increments. The results are `count` and `last` afterwards.

    The items that execute the op together are a batch, in an order, with
    the same `count` and `last`; its result is that of its items appended
    one at a time in that order. Serially a batch is one item; in the SIMT
    lowering it is the lanes of a warp, in the order of the lanes, which
    execute the op convergently.

    *Capacity:* `pool[at]` counts every block that is begun, below the
    capacity or not, and `count` every entry, stored or not: afterwards they
    hold the full demand, so that a caller can size its buffers and build
    again. A block at or beyond the capacity is neither named nor written;
    no element of `into` at or beyond `capacity * block`, none of `units`
    or `ordinals` at or beyond `capacity`, and no slot of a block past the
    list's last entry is written. A caller may read the lists (the blocks
    named by an owner, in the order of their ordinals) only where `pool[at]`
    after every append is at most the capacity; otherwise only the counter
    and the counts, since which lists lost blocks depends on the order in
    which concurrent lists took them.

    *Contract:* the entries of a list are in the order of its appends (the
    batches in the order in which they execute, the items of a batch in its
    order): the lists, the counts, and the counter bitwise. The numbers of
    the blocks are not fixed, since concurrent lists interleave their
    takes: any distinct numbers, compared after the lists are read out
    through their names.

    *Execution* (the SIMT lowering, not the meaning): `lane(lane, below)`,
    the lane's number in its warp and the mask of the lanes below it, which
    a kernel computes once; and a `block` of at least 32, so that a batch
    begins at most two blocks. The serial lowering ignores the first and
    takes any power of two.

    ```mlir
    %count1, %last1 = md_alg.append %keep values(%place, %bits)
        into(%entries, %masks : memref<?xi32, 1>, memref<?xi32, 1>)
        list(%count, %last) owner %g names(%units, %ordinals)
        pool %sizes[%c2] capacity %capacity block 64 lane(%lane, %below)
        : memref<?xi32, 1>, memref<?xi32, 1>, memref<5xi32, 1>
    ```
  }];
  let arguments = (ins
    I1:$keep,
    Variadic<I32>:$values,
    Arg<Variadic<MDAlg_I32Buffer>, "", [MemWrite]>:$into,
    I32:$count,
    I32:$last,
    I32:$owner,
    Arg<MDAlg_I32Buffer, "", [MemWrite]>:$units,
    Arg<MDAlg_I32Buffer, "", [MemWrite]>:$ordinals,
    Arg<MemRefRankOf<[I32], [1]>, "", [MemRead, MemWrite]>:$pool,
    Index:$at,
    Index:$capacity,
    I64Attr:$block,
    Optional<I32>:$lane,
    Optional<I32>:$below
  );
  let results = (outs I32:$next_count, I32:$next_last);
  let assemblyFormat = [{
    $keep `values` `(` $values `)` `into` `(` $into `:` type($into) `)`
    `list` `(` $count `,` $last `)` `owner` $owner
    `names` `(` $units `,` $ordinals `)` `pool` $pool `[` $at `]`
    `capacity` $capacity `block` $block
    (`lane` `(` $lane^ `,` $below `)`)?
    attr-dict `:` type($units) `,` type($ordinals) `,` type($pool)
  }];
  let hasVerifier = 1;
}

def MDAlg_GridPlanOp : MDAlg_Op<"grid_plan", [
    Pure, AttrSizedOperandSegments
  ]> {
  let summary = "The counts of the cells of a grid along edges of a cell";
  let description = [{
    The count of cells along each of the edges `lengths` (f32 or f64, all
    operands and bounds of one type, as a template is instantiated), as
    i64:
    `min(max(floor(L / w), least), most)` with `maxnumf` and `minnumf`, so
    that a length or width that is not a number gives `least` and an
    infinite ratio `most` (#168). The width `w` is given, or follows from a
    target occupancy of the `population` items in the `volume`:
    `w = (occupancy / (population / volume))^(1/3)`, the edge of a cube
    that holds `occupancy` items on average.

    *Contract:* bitwise: there is one lowering, on the host, which the
    serial and the SIMT passes share.

    ```mlir
    %nx, %ny, %nz = md_alg.grid_plan %lx, %ly, %lz : f64, f64, f64
        width %w : f64 bounds [3.0 : f64, 256.0 : f64] -> i64, i64, i64
    %cx, %cy = md_alg.grid_plan %lx, %ly : f64, f64 volume %v : f64
        occupancy 64.0 : f64 of %n bounds [1.0 : f64, 256.0 : f64]
        -> i64, i64
    ```
  }];
  let arguments = (ins
    Variadic<AnyTypeOf<[F32, F64]>>:$lengths,
    Optional<AnyTypeOf<[F32, F64]>>:$width,
    Optional<AnyTypeOf<[F32, F64]>>:$volume,
    OptionalAttr<AnyAttrOf<[F32Attr, F64Attr]>>:$occupancy,
    Optional<Index>:$population,
    AnyAttrOf<[F32Attr, F64Attr]>:$least,
    AnyAttrOf<[F32Attr, F64Attr]>:$most
  );
  let results = (outs Variadic<I64>:$counts);
  let assemblyFormat = [{
    $lengths `:` type($lengths) (`width` $width^ `:` type($width))?
    (`volume` $volume^ `:` type($volume) `occupancy` $occupancy
     `of` $population)?
    `bounds` `[` $least `,` $most `]` attr-dict `->` type($counts)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Accessors
//===----------------------------------------------------------------------===//

def MDAlg_RaggedRowOp : MDAlg_Op<"ragged_row"> {
  let summary = "A row of a buffer of ragged rows";
  let description = [{
    The element where the first entry of the row `row` begins, and the
    number of its entries, in a buffer of ragged rows of entries of
    `entry_size` values (D314; RaggedRows.h): `first` = `n + 2 +
    entry_size * o_row` and `count` = `o_{row+1} - o_row`. With
    `checked`, a row past the buffer's rows, or any row of a buffer of no
    elements, is a row of no entries; without it, `row` must be a row of the
    buffer.

    *Contract:* per item; every lowering gives the same values.

    ```mlir
    %first, %count = md_alg.ragged_row %rows[%i] entry %c4 checked
        : memref<?xi32, 1>
    ```
  }];
  let arguments = (ins
    Arg<MDAlg_I32Buffer, "", [MemRead]>:$rows,
    Index:$row,
    Index:$entry_size,
    UnitAttr:$checked
  );
  let results = (outs Index:$first, Index:$count);
  let assemblyFormat = [{
    $rows `[` $row `]` `entry` $entry_size (`checked` $checked^)?
    attr-dict `:` type($rows)
  }];
}

def MDAlg_RaggedEntryOp : MDAlg_Op<"ragged_entry"> {
  let summary = "A value of an entry of a row of ragged rows";
  let description = [{
    `rows[first + k * entry_size + column]`: the value `column` of the
    entry `k` of the row that begins at the element `first`
    (`md_alg.ragged_row`).

    *Contract:* per item; every lowering gives the same value.

    ```mlir
    %v = md_alg.ragged_entry %rows[%first, %k, %c1] entry %c4
        : memref<?xi32, 1>
    ```
  }];
  let arguments = (ins
    Arg<MDAlg_I32Buffer, "", [MemRead]>:$rows,
    Index:$first,
    Index:$k,
    Index:$column,
    Index:$entry_size
  );
  let results = (outs I32:$result);
  let assemblyFormat = [{
    $rows `[` $first `,` $k `,` $column `]` `entry` $entry_size
    attr-dict `:` type($rows)
  }];
}

def MDAlg_WrapOp : MDAlg_Op<"wrap", [
    Pure,
    AllTypesMatch<["start", "offset", "size", "result"]>
  ]> {
  let summary = "The sum of two coordinates of a periodic axis, wrapped";
  let description = [{
    `(start + offset) mod size` for `start` and `offset` in `[0, size)`, with
    one subtraction: the axis of a periodic grid that a lane reads (D325;
    GatherEmitter.h). The type is i32 or index. With `signed`, the
    comparison with `size` is signed, as a site with i32 coordinates
    compared them; it applies to i32 alone.

    *Contract:* per item; every lowering gives the same value.

    ```mlir
    %z = md_alg.wrap %start, %lane, %size : i32
    ```
  }];
  let arguments = (ins
    MDAlg_I32OrIndex:$start,
    MDAlg_I32OrIndex:$offset,
    MDAlg_I32OrIndex:$size,
    UnitAttr:$signed_compare
  );
  let results = (outs MDAlg_I32OrIndex:$result);
  let assemblyFormat = [{
    $start `,` $offset `,` $size (`signed` $signed_compare^)? attr-dict `:` type($result)
  }];
  let hasVerifier = 1;
}

def MDAlg_GridReadOp : MDAlg_Op<"grid_read", [
    TypesMatchWith<"the result is an element of the grid", "grid", "result",
                   "::llvm::cast<::mlir::MemRefType>($_self).getElementType()">,
    AllTypesMatch<["base", "z"]>
  ]> {
  let summary = "A point of a row of a grid";
  let description = [{
    `grid[base + z]`, where `base` is the first point of a row and `z` a
    wrapped coordinate along it (`md_alg.wrap`) (D325; GatherEmitter.h).
    The coordinates are i32 or index; with i32 only the final address
    becomes an index. With `active`, a lane where it is false reads
    `grid[base]`, so that its address stays in the grid. The result has the
    grid's element type.

    *Contract:* per item; every lowering gives the same value.

    ```mlir
    %p = md_alg.grid_read %phi[%base + %z] : memref<?xf32, 1>, i32
    %q = md_alg.grid_read %phi[%base + %z] if %use : memref<?xf32, 1>, i32
    ```
  }];
  let arguments = (ins
    Arg<MDAlg_RealBuffer, "", [MemRead]>:$grid,
    MDAlg_I32OrIndex:$base,
    MDAlg_I32OrIndex:$z,
    Optional<I1>:$active
  );
  let results = (outs AnyTypeOf<[F32, F64]>:$result);
  let assemblyFormat = [{
    $grid `[` $base `+` $z `]` (`if` $active^)? attr-dict `:` type($grid) `,`
    type($base)
  }];
}

#endif // MDIR_DIALECT_MDALG_MDALGOPS_TD
