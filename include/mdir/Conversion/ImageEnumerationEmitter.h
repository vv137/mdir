// Pair-local periodic image choices for reduced cells [AllenTildesley2017].
#ifndef MDIR_CONVERSION_IMAGEENUMERATIONEMITTER_H
#define MDIR_CONVERSION_IMAGEENUMERATIONEMITTER_H

#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace mdir::kernels {
enum class PairImageStrategy { Pass, ConditionalNeighbors };

/// Emit one or eight scalar payloads in pass-first c/b/a traversal order.
/// rawCartesian is vector<3xf64>; cell is vector<3xf64> or vector<6xf64>.
/// paddedReach is f64. The caller must establish the reduced-cell, reach,
/// finite-intermediate, and rounding preconditions for the chosen strategy.
/// Arithmetic requires gradual underflow and no approximate/fast math.
/// Each following axis is re-centered for its higher-axis prefix.
/// visit receives a vector<3xf64> integer-valued image query inside the
/// valid leaf's scf.if; it returns the same scalar type as missingPayload.
/// Visitors own membership tests and payload semantics. Inactive leaves
/// return missingPayload without invoking the visitor at runtime. Ordering
/// or aggregation beyond this local traversal belongs to the caller.
llvm::SmallVector<mlir::Value> emitPairImageChoices(
    mlir::OpBuilder &, mlir::Location, mlir::Value rawCartesian,
    mlir::Value cell, mlir::Value paddedReach, PairImageStrategy,
    mlir::Value missingPayload,
    llvm::function_ref<mlir::Value(mlir::OpBuilder &, mlir::Value)> visit);
} // namespace mdir::kernels
#endif
