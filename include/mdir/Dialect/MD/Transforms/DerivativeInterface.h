#ifndef MDIR_DIALECT_MD_TRANSFORMS_DERIVATIVEINTERFACE_H
#define MDIR_DIALECT_MD_TRANSFORMS_DERIVATIVEINTERFACE_H

#include "mlir/IR/OpDefinition.h"

namespace mlir { class DialectRegistry; }
namespace mdir { namespace md {
class ScalarDerivative;
enum class DerivativeOperand { Differentiable, Structural };
void registerDerivativeInterfaces(mlir::DialectRegistry &registry);
} }

#include "mdir/Dialect/MD/Transforms/DerivativeInterface.h.inc"
#endif
