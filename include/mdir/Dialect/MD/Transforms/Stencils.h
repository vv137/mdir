#ifndef MDIR_DIALECT_MD_TRANSFORMS_STENCILS_H
#define MDIR_DIALECT_MD_TRANSFORMS_STENCILS_H
#include "mlir/IR/Builders.h"
namespace mlir { class DialectRegistry; }
namespace mdir::md {
// Clones the weight body; no second assignment/interpolation formula.
mlir::Value cloneStencil(mlir::OpBuilder &builder, mlir::Operation *source,
                         mlir::Value data, bool transpose);
void registerStencilDerivatives(mlir::DialectRegistry &registry);
// The ordinary evaluator has not admitted a cell-dependent stencil chain.
// Check the energy's cell activity, not only uses of its primary positions.
mlir::LogicalResult checkStencilVirialActivity(mlir::Operation *function,
                                              mlir::Value cell,
                                              mlir::Value energy,
                                              mlir::LocationAttr expandedWindow);
}
#endif
