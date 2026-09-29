// Types of the md dialect.

#ifndef MDIR_DIALECT_MD_MDTYPES_H
#define MDIR_DIALECT_MD_MDTYPES_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "mdir/Dialect/MD/MDEnums.h"

#define GET_TYPEDEF_CLASSES
#include "mdir/Dialect/MD/MDTypes.h.inc"

namespace mlir {
class Operation;
} // namespace mlir

namespace mdir {
namespace md {

/// Verifies that the fields in the signature `type` of the function `op`
/// have the reference precision: `f64` is the only floating-point type at
/// the semantic level.
mlir::LogicalResult verifyReferencePrecision(mlir::Operation *op,
                                             mlir::FunctionType type);

} // namespace md
} // namespace mdir

#endif // MDIR_DIALECT_MD_MDTYPES_H
