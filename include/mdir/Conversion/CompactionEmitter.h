// The emitters of compaction, scans, and counting sorts on a device
// (D[compaction-emitters], docs/device-primitives.md, Section 4).
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
// Warp compaction
//===----------------------------------------------------------------------===//

/// The mask of the lanes below `lane` (i32, in [0, 32)) in a warp of 32,
/// (1 << lane) - 1, which emitWarpRank takes. A site that votes in a loop
/// emits it once, before the loop.
mlir::Value emitLanesBelow(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::Value lane);

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

} // namespace kernels
} // namespace mdir

#endif // MDIR_CONVERSION_COMPACTIONEMITTER_H
