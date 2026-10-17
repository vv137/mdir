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

/// The type of the buffer of the host that takes `number`, a number that
/// mdrt.host_call yields: memref<1xf64> for an f64, memref<kxf64> for a
/// vector of k f64; null for another type.
mlir::MemRefType getHostNumberType(mlir::Type number);

/// The arguments of the function of mdrt.host_call that an input of the
/// type `input` takes: four for the pairs of images of `md_exec.build_edges`
/// (its slots, counts, offsets, and the number of their build,
/// D277), one for any other.
unsigned getHostArgumentCount(mlir::Type input);

/// The index of the first argument of the function of `call` that takes
/// its input `index`, or, for an index beyond the inputs, that takes its
/// field of `outs` or its number of that position after the inputs.
class HostCallOp;
unsigned getHostArgumentIndex(HostCallOp call, unsigned index);

/// The type of the buffer that holds the members of the tuples of
/// `relation`, the relation of a tuple set: `memref<?xkxi32>` for tuples of
/// `k` members.
mlir::MemRefType getMembersType(md::RelationType relation);

/// The type of the buffer that holds a table: of its rank, with a dynamic
/// size in each dimension.
mlir::MemRefType getTableBufferType(md::TableType table);

/// The attribute that marks a loop whose iterations are segments of a run:
/// stretches between two checkpoints.
inline llvm::StringRef getSegmentAttrName() { return "mdrt.segment"; }

/// The attribute that marks a loop whose neighbor structures begin empty
/// rather than continue those built before it: the loop of steps after the
/// evaluation at the start of a run, which is a segment of its own
/// (D218).
inline llvm::StringRef getFreshAttrName() { return "mdrt.fresh"; }

/// The attribute of an argument of a function that the function never
/// writes (D258): on an argument of the entry of a
/// program, a buffer of the model that the caller hands without a copy; on
/// an argument of a declared function of the runtime, a buffer that the
/// runtime only reads. `check-read-only-arguments` proves the first by the
/// second and fails the compile where it cannot.
inline llvm::StringRef getReadOnlyAttrName() { return "mdrt.read_only"; }

} // namespace mdrt
} // namespace mdir

#define GET_OP_CLASSES
#include "mdir/Dialect/MDRT/MDRTOps.h.inc"

#endif // MDIR_DIALECT_MDRT_MDRTOPS_H
