// The lowerings of md_alg.traverse_pairs (D415): the SIMT
// schedule of the groups of neighbors (D89, docs/neighbor-groups.md),
// called in place by convert-md-exec-to-gpu through `TraverseHost` (the
// maintainer's ruling on #887; #888 frees it), and the serial loop over
// the entries, the oracle.

#ifndef MDIR_CONVERSION_TRAVERSEPAIRS_H
#define MDIR_CONVERSION_TRAVERSEPAIRS_H

#include "mdir/Dialect/MDAlg/MDAlgOps.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include <functional>

namespace mdir::kernels {

/// The inner list of a dual list of groups (D114), as the runtime holds it:
/// each block of the outer list pruned into the same block, its entries at
/// the front, and their number.
struct InnerLists {
  mlir::Value entries, masks, counts;
};

/// The buffers of a structure of groups, as the runtime holds them now.
struct GroupBuffers {
  mlir::Value order, counts, entries, masks, units, ordinals, shifts;
};

/// What the pruning of a dual list keeps besides the inner list
/// (D268): the block of each group's list by its number in the
/// list, a row of the blocks of the longest list for each group, made at
/// each build; the units of work that hold entries of the inner list (the
/// active units), in the order of the blocks, and their number; and the
/// counts of the active units in runs of 32 and of 1024 blocks.
struct PruneBuffers {
  mlir::Value blockOf, activeUnits, activeCount, runSums, wideSums;
};

/// What the kernels of a pruning take, made on the host: the buffers, the
/// frames of the places, the groups and the blocks of the longest list, the
/// square of the widened reach and the cell in f32, and the blocks of the
/// outer list with their runs of 32 and 1024.
struct PruneContext {
  GroupBuffers buffers;
  InnerLists inner;
  PruneBuffers prune;
  mlir::Value placed, groups, stride, reach2, box32, blocks, runs, wideRuns;
  bool triclinic = false;
};

/// What the SIMT schedule of md_alg.traverse_pairs takes from the pass that
/// calls it in place (convert-md-exec-to-gpu): the structure that the op
/// reads, the naming of kernels, the device's buffers, the pruning of a
/// dual list, and the global sums.
class TraverseHost {
public:
  virtual ~TraverseHost() = default;

  /// The number of threads of a block.
  virtual int64_t getBlockSize() = 0;
  /// The block memory that a strategy's kernel may take (the launch
  /// policy's `blockMemoryBytes`, DevicePolicy.h).
  virtual int64_t getBlockMemoryBytes() = 0;
  /// Names the kernel `launch` and brings the constants and the vectors it
  /// uses into it.
  virtual void bringIn(mlir::gpu::LaunchOp launch) = 0;
  /// A launch of a thread for each of `count` items.
  virtual void
  launchOver(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value count,
             llvm::function_ref<void(mlir::OpBuilder &, mlir::Value)> body) = 0;
  /// A buffer of the device.
  virtual mlir::Value createDeviceBuffer(mlir::OpBuilder &builder,
                                         mlir::Location loc,
                                         mlir::MemRefType type,
                                         mlir::ValueRange sizes) = 0;

  /// The structure: the sizes on the host (the places, the longest list,
  /// the blocks), whether a refresh keeps it dual, and whether its last
  /// refresh pruned it (on the host).
  virtual mlir::Value getSizes() = 0;
  virtual bool isDual() = 0;
  virtual mlir::Value getPrunedNow() = 0;
  /// Its buffers, as the runtime holds them where `builder` is.
  virtual GroupBuffers getGroupsBuffers(mlir::OpBuilder &builder,
                                        mlir::Location loc) = 0;
  virtual InnerLists getInnerLists(mlir::OpBuilder &builder,
                                   mlir::Location loc) = 0;
  virtual PruneBuffers getPruneBuffers(mlir::OpBuilder &builder,
                                       mlir::Location loc) = 0;
  /// The positions of the places in the frames that the last pruning left.
  virtual mlir::Value getGroupsPlaced(mlir::OpBuilder &builder,
                                      mlir::Location loc) = 0;

  /// The pruning in the loop (D284): whether the loop prunes, on the host,
  /// from `groups` groups; that it counts one; what its kernels take; the
  /// warp of one group; the active units after; and the check of the
  /// pruning after the loop, where it is asked for (D268).
  virtual mlir::Value takesFusedPruning(mlir::OpBuilder &builder,
                                        mlir::Location loc,
                                        int64_t groups) = 0;
  virtual void countFusedPruning(mlir::OpBuilder &builder,
                                 mlir::Location loc) = 0;
  virtual PruneContext getPruneContext(mlir::OpBuilder &builder,
                                       mlir::Location loc, mlir::Value box,
                                       double reach) = 0;
  virtual mlir::Value emitPruneGroup(mlir::OpBuilder &b, mlir::Location loc,
                                     const PruneContext &ctx,
                                     mlir::Value group, mlir::Value lane) = 0;
  virtual void emitActiveUnits(mlir::OpBuilder &builder, mlir::Location loc,
                               const PruneContext &ctx) = 0;
  virtual void emitFusedPruningCheck(mlir::OpBuilder &builder,
                                     mlir::Location loc,
                                     mlir::Value positions,
                                     mlir::Value cell) = 0;

  /// Replaces the results of `op` with its initial values plus the sums of
  /// the `count` rows of `scratch`.
  virtual mlir::LogicalResult finishSums(mlir::Operation *op,
                                         mlir::OpBuilder &builder,
                                         mlir::ValueRange reduce,
                                         mlir::ValueRange scratch,
                                         mlir::Value count) = 0;
};

/// The SIMT schedule of the groups (D89, D95, D97, D114, D268, D284, D291):
/// a warp for each unit of work, lanes u and u + 16 holding the particle
/// at place 16 g + u, the entries turning through the half-warps.
/// Refuses what the schedule cannot keep: a stable visit, the combination
/// `owner`, a block that is not whole warps, and the buffer form.
mlir::LogicalResult lowerTraversePairsSIMT(md_alg::TraversePairsOp op,
                                           TraverseHost &host);

/// The buffers of a list of units (D417, D419), as the runtime holds them
/// now: the order of the places as the traversal reads it (the slot of a
/// unit's first member in `members`, or the particle of the rest), their
/// shifts, the records in the order of the segments (entries, masks,
/// groups, counts), the first record of each segment, the members of the
/// units of the taken kinds, and the place where each rank begins (the
/// rest rank 0, kind f rank K - f, then the end of the places). A dual
/// list adds its inner records and their active records.
struct SegmentLists {
  mlir::Value order, shifts, entries, masks, groups, counts, offsets;
  mlir::Value members, rankPlaces;
  InnerLists inner;
  mlir::Value active, activeCount;
  bool dual = false;
};

/// The pruning of a dual list of units in the loop that takes its refresh
/// (D284; D417, Section 6), from the pass that calls the schedule: whether
/// the refresh pruned at this step (`now`, on the host), what the kernels
/// take, made on the host (`prepare`), the warp's pruning of a record, the
/// survivors numbered from a cursor and those in [low, high) written to the
/// inner records from an entry on (`pruneRecord`, the number kept), a
/// record's count as an active one (`recordActive`), and the active units
/// after the launch (`finish`).
struct SegmentsFusedPruning {
  mlir::Value now;
  std::function<void(mlir::OpBuilder &)> prepare;
  std::function<mlir::Value(mlir::OpBuilder &, mlir::Value record,
                            mlir::Value lane, mlir::Value base,
                            mlir::Value cursor, mlir::Value low,
                            mlir::Value high)>
      pruneRecord;
  std::function<void(mlir::OpBuilder &, mlir::Value record, mlir::Value kept)>
      recordActive;
  std::function<void(mlir::OpBuilder &)> finish;
};

/// The SIMT schedule of a list of units (D[md-alg-segments-simt], D417,
/// Section 6): a warp for each record of one segment, lanes u and u + 16
/// holding the unit at place 16 g + u, the partners' frames, fields, masks,
/// and the accumulators of their places turning through the half-warps;
/// one guard and one slice of the region for each site pair of a segment's
/// table; W^T at the write. `blockSize` and `rotation` are the entry of the
/// tuning policy (D420). Refuses what the schedule does not take, by name.
mlir::LogicalResult lowerTraversePairsSegmentsSIMT(
    md_alg::TraversePairsOp op, TraverseHost &host, const SegmentLists &lists,
    int64_t blockSize, int64_t rotation,
    const SegmentsFusedPruning *fused = nullptr);

/// The buffers that the serial definition of md_alg.traverse_pairs reads,
/// all where the loop runs: the layout of the groups, the inner list and
/// its active units with `inner`, the positions, the fields of `ins`, and
/// the destinations; `places` and `units` are index values, `cell` the
/// vector of the cell.
struct SerialLists {
  GroupBuffers groups;
  InnerLists inner;
  mlir::Value active, activeCount;
  bool dual = false;
  /// Over the segments of a list of units (D417): the
  /// offsets of each segment's units of work, (K+1)(K+2)/2 + 1 of them, and
  /// the members of each unit of a taken kind, which its place names by
  /// the position of its first member.
  mlir::Value segments, members;
  mlir::Value places, units;
  mlir::Value positions, cell;
  llvm::SmallVector<mlir::Value> ins, outs;
  /// The buffers that the region reads from outside it (tables), and the
  /// copies it reads them through where the loop runs.
  llvm::SmallVector<std::pair<mlir::Value, mlir::Value>> captures;
};

/// How the serial definition runs: in the order of the entries or in
/// reverse (`reverse-unordered-visits`, D409); with the magnitudes of the
/// terms in place of the terms, for the bound of a test; or with no term.
struct SerialOptions {
  bool reverse = false;
  bool magnitudes = false;
  /// The test's baseline: no pair contributes (what the rest of the
  /// program gives without the traversal).
  bool omitted = false;
};

/// Emits the serial definition of `op` at `builder` over `lists`, and
/// returns the initial values of `op` plus its sums. The destinations of
/// `lists` are added to, or overwritten where `op` says so.
llvm::SmallVector<mlir::Value>
emitTraversePairsSerial(mlir::OpBuilder &builder, md_alg::TraversePairsOp op,
                        const SerialLists &lists, SerialOptions options = {});

/// The serial definition of `op` over the segments of a list of units
/// (D417, Section 6), which emitTraversePairsSerial calls.
llvm::SmallVector<mlir::Value>
emitTraverseSegmentsSerial(mlir::OpBuilder &builder,
                           md_alg::TraversePairsOp op,
                           const SerialLists &lists, SerialOptions options);

} // namespace mdir::kernels

#endif // MDIR_CONVERSION_TRAVERSEPAIRS_H
