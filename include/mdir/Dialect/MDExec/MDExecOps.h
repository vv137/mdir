// Ops of the md_exec dialect.

#ifndef MDIR_DIALECT_MDEXEC_MDEXECOPS_H
#define MDIR_DIALECT_MDEXEC_MDEXECOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mdir/Dialect/MD/MDCoordinates.h"
#include "mdir/Dialect/MD/MDTypes.h"
#include "mdir/Dialect/MDRT/MDRTTypes.h"

#include "mdir/Dialect/MDExec/MDExecEnums.h.inc"

namespace mdir {
namespace md_exec {

/// Returns true if `type` belongs to the value form: a field, or a
/// structure that the storage form does not have.
bool isValueFormType(mlir::Type type);

/// Returns true if `type` is the type of a buffer that holds a field:
/// `memref<?x3xT>` or `memref<?xT>`.
bool isBufferType(mlir::Type type);

/// Returns true if `type` is the type of a buffer in `scratch`: a buffer
/// with one value per particle, a number or a fixed number of them.
bool isScratchType(mlir::Type type);

/// The type of the buffer in `scratch` that holds one value of the type
/// `value`, a number or a vector, per particle. `like` is a buffer of the
/// same particles: the buffer is where that one is.
mlir::MemRefType getScratchType(mlir::Type value, mlir::MemRefType like);

/// Returns true if `type` is the type of a buffer that holds the members
/// of tuples: `memref<?xkxi32>`.
bool isMembersType(mlir::Type type);

/// Returns true if `type` is the type of a buffer that holds the tuples of
/// each particle: `memref<?x?xi32>`.
bool isIncidenceBufferType(mlir::Type type);

/// The type of the buffer that holds the tuples of each particle, on the
/// host or on the device.
mlir::MemRefType getIncidenceBufferType(mlir::MLIRContext *context,
                                        mlir::Attribute memorySpace);

/// The number of values that the buffer of the tuples of each particle
/// holds for one tuple of `arity` members: the number of the tuple, the
/// place of the particle, and the members.
inline int64_t getIncidenceEntrySize(int64_t arity) { return arity + 2; }

/// The type of one value of a field inside a kernel, for the type of the
/// field or of the buffer that holds it.
mlir::Type getKernelValueType(mlir::Type fieldOrBuffer);

} // namespace md_exec
} // namespace mdir

#define GET_OP_CLASSES
#include "mdir/Dialect/MDExec/MDExecOps.h.inc"

#endif // MDIR_DIALECT_MDEXEC_MDEXECOPS_H
