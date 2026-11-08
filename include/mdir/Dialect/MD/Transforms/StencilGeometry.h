#ifndef MDIR_DIALECT_MD_TRANSFORMS_STENCILGEOMETRY_H
#define MDIR_DIALECT_MD_TRANSFORMS_STENCILGEOMETRY_H
#include "mlir/IR/Builders.h"

namespace mdir::md {
// Row-cell convention: x = s H. Returns row-major H^{-1} and det(H).
struct StencilCellInverse {
  llvm::SmallVector<mlir::Value, 9> entries;
  mlir::Value determinant;
};
StencilCellInverse emitStencilCellInverse(mlir::OpBuilder &builder,
                                         mlir::Location loc,
                                         mlir::Value matrix);
mlir::Value emitStencilCellMatrix(mlir::OpBuilder &builder,
                                  mlir::Location loc, mlir::Value cell);
// The continuous geometry rule for one declared stencil operand. It clones
// and differentiates the source weight body, not integer index selection.
mlir::FailureOr<mlir::Value> differentiateStencilGeometry(
    mlir::OpBuilder &builder, mlir::Operation *source, unsigned operand,
    mlir::Value seed, bool reverse);
}
#endif
