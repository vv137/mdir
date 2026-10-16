// The declared transposes of linear ops, which md-differentiate carries
// adjoints through (D[ml2-distribute]).

#ifndef MDIR_DIALECT_MD_TRANSFORMS_ADJOINTINTERFACE_H
#define MDIR_DIALECT_MD_TRANSFORMS_ADJOINTINTERFACE_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir { class DialectRegistry; }
namespace mdir { namespace md {
/// Attaches the transposes of the transfers of md_dist: G and G^T, the
/// distribution of a field and its collection, and the sum of the partial
/// sums of the domains.
void registerAdjointInterfaces(mlir::DialectRegistry &registry);
} }

#include "mdir/Dialect/MD/Transforms/AdjointInterface.h.inc"
#endif
