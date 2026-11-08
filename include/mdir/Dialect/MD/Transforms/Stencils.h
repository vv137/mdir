#ifndef MDIR_DIALECT_MD_TRANSFORMS_STENCILS_H
#define MDIR_DIALECT_MD_TRANSFORMS_STENCILS_H
#include "mlir/IR/Builders.h"
namespace mlir { class DialectRegistry; }
namespace mdir::md {
// Clones the weight body; no second assignment/interpolation formula.
mlir::Value cloneStencil(mlir::OpBuilder &builder, mlir::Operation *source,
                         mlir::Value data, bool transpose);
void registerStencilDerivatives(mlir::DialectRegistry &registry);
}
#endif
