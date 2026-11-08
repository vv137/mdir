#ifndef MDIR_DIALECT_MD_TRANSFORMS_GRIDKERNELS_H
#define MDIR_DIALECT_MD_TRANSFORMS_GRIDKERNELS_H

#include "mlir/IR/Builders.h"

namespace mlir { class DialectRegistry; }
namespace mdir::md {
void registerGridDerivatives(mlir::DialectRegistry &registry);
// Grid complex scalars use two real components in structured kernels.
mlir::Type getGridKernelType(mlir::Type element);
// Transposes the kernel linearization under the declared domain pairing.
mlir::FailureOr<mlir::Value> differentiateGridKernel(
    mlir::OpBuilder &builder, mlir::Operation *source, unsigned operand,
    mlir::Value seed, bool reverse);
}
#endif
