// Ops of the mdrt dialect.

#ifndef MDIR_DIALECT_MDRT_MDRTOPS_H
#define MDIR_DIALECT_MDRT_MDRTOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mdir/Dialect/MD/MDTypes.h"
#include "mdir/Dialect/MDRT/MDRTTypes.h"

namespace mdir {
namespace mdrt {

/// The type of the buffer that holds a field: `memref<?x3xT>` for a field
/// with 3 components, `memref<?xT>` for a field with 1.
mlir::MemRefType getBufferType(md::FieldType field);

/// Returns true if a buffer of the type `buffer` can hold a field of the
/// type `field`. The element types may differ if both are floating-point
/// types: the buffer then states the type that the field is stored in.
bool canHold(mlir::Type buffer, md::FieldType field);

/// The type of the buffer that holds the members of the tuples of
/// `relation`, the relation of a tuple set: `memref<?xkxi32>` for tuples of
/// `k` members.
mlir::MemRefType getMembersType(md::RelationType relation);

/// The attribute that marks a loop whose iterations are segments of a run:
/// stretches between two checkpoints.
inline llvm::StringRef getSegmentAttrName() { return "mdrt.segment"; }

} // namespace mdrt
} // namespace mdir

#define GET_OP_CLASSES
#include "mdir/Dialect/MDRT/MDRTOps.h.inc"

#endif // MDIR_DIALECT_MDRT_MDRTOPS_H
