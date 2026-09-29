// Ops of the mdrt dialect.

#ifndef MDIR_DIALECT_MDRT_MDRTOPS_H
#define MDIR_DIALECT_MDRT_MDRTOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mdir/Dialect/MD/MDTypes.h"
#include "mdir/Dialect/MDRT/MDRTTypes.h"

namespace mdir {
namespace mdrt {

/// The type of the buffer that holds a field: `memref<?x3xT>` for a field
/// with 3 components, `memref<?xT>` for a field with 1.
mlir::MemRefType getBufferType(md::FieldType field);

} // namespace mdrt
} // namespace mdir

#define GET_OP_CLASSES
#include "mdir/Dialect/MDRT/MDRTOps.h.inc"

#endif // MDIR_DIALECT_MDRT_MDRTOPS_H
