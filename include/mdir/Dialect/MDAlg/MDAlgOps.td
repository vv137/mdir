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
