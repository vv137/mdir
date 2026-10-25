// The emitters of compaction, scans, and counting sorts on a device
// (D296, docs/device-primitives.md, Section 4).
// Portability: docs/device-primitives.md, Section 6 (D340).
//
// A compaction keeps the items whose predicate holds and packs them in their
// order: item i goes to place r(i), the number of kept items before it. At
// the level of a warp, the lanes vote on the predicate (`ballot`) and each
// lane counts the votes of the lanes below its own (`popc` of the vote under
// a mask): this is CUB's warp-level select, a WarpScan over flags done in
// the bits of one vote [CUB]. The sites keep what is their own: how their
// threads take their items, where the kept items go, and what a round
// carries to the next. Every site calls the same emitters, so that a change
// of the primitive is made once.

#ifndef MDIR_CONVERSION_COMPACTIONEMITTER_H
#define MDIR_CONVERSION_COMPACTIONEMITTER_H

#include "mlir/IR/Builders.h"

namespace mdir {
namespace kernels {

//===----------------------------------------------------------------------===//
// The calls of the templates
//===----------------------------------------------------------------------===//

/// The prefix of the names by which a template calls a primitive:
/// `func.call @mdir.primitive.warp_rank(%ballot, %below) : (i32, i32) ->
/// i32`, with a declaration of the callee in the template.
inline constexpr llvm::StringLiteral kPrimitivePrefix = "mdir.primitive.";

/// Replaces every call of a primitive under `root` with the code of its
/// emitter, where the call stands, and removes the declarations of the
/// primitives. A primitive that does not exist, or a call whose operands or
/// results the primitive does not have, is an error on the call. The
/// lowering calls this on each template it parses; the pass
/// `expand-device-primitives` calls it on a module.
mlir::LogicalResult expandPrimitiveCalls(mlir::Operation *root);

//===----------------------------------------------------------------------===//
// Warp compaction
//===----------------------------------------------------------------------===//

/// The mask of the lanes below `lane` (i32, in [0, 32)) in a warp of 32,
/// (1 << lane) - 1, which emitWarpRank takes. A site that votes in a loop
/// emits it once, before the loop. `one`, the i32 1 where the site has it,
/// is used in place of a constant of the emitter's own.
mlir::Value emitLanesBelow(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::Value lane, mlir::Value one = {});

/// The vote of the warp on `predicate` (i1), as the i32 whose bit l is the
/// predicate of lane l: `gpu.ballot`. Every lane of the warp takes part.
mlir::Value emitWarpBallot(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::Value predicate);

/// The place of a lane among the lanes whose predicate holds, in the order
/// of the lanes: the votes of `ballot` under `below` (emitLanesBelow),
/// counted. A kept lane writes its item at the count of the round plus this.
mlir::Value emitWarpRank(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Value ballot, mlir::Value below);

/// The number of lanes whose predicate holds: the votes of `ballot`,
/// counted. The count of a round adds it for the next.
mlir::Value emitWarpCount(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value ballot);

//===----------------------------------------------------------------------===//
// Scans
//===----------------------------------------------------------------------===//

/// The inclusive sum of the i32 `value` over the lanes of each segment of
/// `lanes` lanes (2, 4, ..., 32) at and below this one: log2(lanes) rounds,
/// in each of which a lane adds the partial sum of the lane `o` below it
/// (`gpu.shuffle up` by o = 1, 2, 4, ...) if it has one. This is the scan of
/// Hillis and Steele [HillisSteele1986], CUB's WarpScanShfl [CUB]. `lane`
/// (i32) is the place of the lane in its segment, compared with `o` as a
/// signed number or, with `unsignedLane`, as an unsigned one; `zero` is the
/// i32 0 and `width` the i32 width of the shuffles (32), which the site has.
/// Every lane of the warp takes part.
mlir::Value emitWarpScan(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Value value, mlir::Value lane,
                         mlir::Value zero, mlir::Value width, int64_t lanes,
                         bool unsignedLane = false);

/// The exclusive sum of an i32 over the threads of a block before this one,
/// and the sum over the block, in every thread.
struct BlockScan {
  mlir::Value before, total;
};

/// The block scan of `value` (i32): emitWarpScan in each warp, the sums of
/// the warps through block memory, emitWarpScan of those by the first warp,
/// and each thread's sum before it from its warp's offset. This is CUB's
/// BlockScanWarpScans [CUB]. `shared` is a buffer of i32 in block memory
/// with one element for each warp of the block and one for the total
/// (`memref<9xi32>` for blocks of 256 threads); `thread` is the index of
/// the thread in the block. Every thread of the block reaches it: it holds
/// two barriers.
BlockScan emitBlockScan(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value value, mlir::Value shared,
                        mlir::Value thread);

/// The exclusive sums of the `n` i32 of `counts` into `start` (n + 1, the
/// sum of all at n) and, unless it is null, into `cursor` (n): blocks of
/// `threads` = 256 threads, each thread 4 counts, sum their 1024 counts
/// into `sums` (emitBlockScan); one block scans the sums of the blocks;
/// then each block scans its counts from the sum before it: a
/// reduce-then-scan in three passes over the counts. CUB's device scan is
/// a single pass with decoupled look-back [CUB], which is #420.
/// The caller gives `blocks`, at least ceil(n / 1024) and 1, `sums`, a
/// buffer of `blocks` i32 on the device, `threads`, and `one`, the index
/// 1 of the launches' other dimensions. Three launches.
void emitDeviceScan(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value counts, mlir::Value start, mlir::Value cursor,
                    mlir::Value n, mlir::Value sums, mlir::Value blocks,
                    mlir::Value threads, mlir::Value one);

/// An alternative device scan of the same i32 counts, with one read of each
/// count and one output launch after a launch that clears `state`. Blocks
/// claim tiles dynamically and obtain their offsets by decoupled look-back
/// (Merrill and Garland, NVR-2016-002). `state` is a device buffer of
/// `blocks + 1` i64 words: the first is the tile counter, the rest pack a
/// status and a sum. It must not be shared by concurrent scans. The caller
/// owns the buffer. `blocks`, `threads`, and `one` have the same meaning as
/// in emitDeviceScan.
void emitSinglePassScan(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value counts, mlir::Value start,
                        mlir::Value cursor, mlir::Value n, mlir::Value state,
                        mlir::Value blocks, mlir::Value threads,
                        mlir::Value one);

/// The loop of one thread that turns the `count` i32 of `sums` into their
/// exclusive sums where they are, and stores the sum of all in `out` at
/// `at`: the scan of the sums of chunks in the chunked scan, and of
/// `md_exec.nonzero`'s counts of chunks.
void emitSerialScan(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value sums, mlir::Value count, mlir::Value out,
                    mlir::Value at);

/// The chunked scan: the exclusive sums of the `n` i32 of `counts` into
/// `start` (n + 1, the sum of all at n) and `cursor` (n). A thread sums
/// each chunk of `chunk` counts into `sums` (`chunks` of them); one thread
/// scans those (emitSerialScan); a thread writes the offsets of each chunk
/// from its sum before it. Three launches of `block` threads; `grid` holds
/// `chunks` threads, and `one` is the index 1. It suits a few thousand
/// counts, as the cells of a build have.
void emitChunkedScan(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::Value counts, mlir::Value start, mlir::Value cursor,
                     mlir::Value sums, mlir::Value n, mlir::Value chunks,
                     mlir::Value chunk, mlir::Value grid, mlir::Value block,
                     mlir::Value one);

//===----------------------------------------------------------------------===//
// Counting sorts
//===----------------------------------------------------------------------===//
//
// A counting sort of n items by keys in [0, K): a histogram of the keys
// (the site counts, with one device atomic for each item, where it
// computes the keys), the exclusive scan of the counts (emitDeviceScan or
// emitChunkedScan, into `start` and `cursor`), the scatter of each item to
// the next slot of its key (emitCursorScatter), then an order within each
// key, since the order of the atomics is not fixed: emitSortBins, by
// insertion, or emitRankInBins, the stable rank. The rank is the variant
// for keys with many items (#231) and for the partitions to come (#38).

/// A relaxed atomic addition of the i32 `value` to `buffer[index]` at the
/// scope of the device; returns the old value.
mlir::Value emitDeviceAtomicAdd(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value buffer, mlir::Value index,
                                mlir::Value value);

/// The scatter of a counting sort: a thread for each of the `n` items
/// takes the next slot of its key `keys[i]` (i32) from `cursor` with an
/// atomic and writes `i` there in `out`. An item whose key is negative has
/// no bin and is skipped, unless `everyKey`, where every key is a bin. One
/// launch of `block` threads on `grid` blocks; `one` is the index 1.
void emitCursorScatter(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::Value keys, mlir::Value cursor, mlir::Value out,
                       mlir::Value n, mlir::Value grid, mlir::Value block,
                       mlir::Value one, bool everyKey);

/// The items (i32) of each of the `n` bins of `start` (n + 1 offsets) in
/// `items` sorted ascending by insertion, a thread a bin: by the item, or by
/// `by[item]` where `by` is given. Equal values do not move past each other.
/// It suits bins of a few items.
void emitSortBins(mlir::OpBuilder &builder, mlir::Location loc,
                  mlir::Value start, mlir::Value items, mlir::Value n,
                  mlir::Value grid, mlir::Value block, mlir::Value one,
                  mlir::Value by = {});

/// The stable rank of a counting sort: a thread for each of the
/// `start[bins]` items of `items` (i32, in the bins of `start`, by the key
/// `keys[item]`) counts the items of its bin that are smaller, and writes
/// its item at that place of the bin in `ranked`. The items of a bin are
/// distinct. It suits bins of many items (#231).
void emitRankInBins(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value items, mlir::Value keys, mlir::Value start,
                    mlir::Value ranked, mlir::Value bins, mlir::Value grid,
                    mlir::Value block, mlir::Value one);

} // namespace kernels
} // namespace mdir

#endif // MDIR_CONVERSION_COMPACTIONEMITTER_H
