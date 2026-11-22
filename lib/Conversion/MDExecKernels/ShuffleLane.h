// Shuffle a scalar between lanes, preserving the validity result.
#pragma once

#include "mlir/Dialect/GPU/IR/GPUDialect.h"

namespace mdir::kernels::detail {
inline auto shuffleLane(mlir::OpBuilder &b, mlir::Location loc,
                        mlir::gpu::ShuffleMode mode, mlir::Value value,
                        mlir::Value offset, mlir::Value width) {
  auto op = mlir::gpu::ShuffleOp::create(b, loc, value, offset, width, mode);
  return std::pair<mlir::Value, mlir::Value>{op.getShuffleResult(),
                                             op.getValid()};
}
} // namespace mdir::kernels::detail
