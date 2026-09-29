// Ops of the md_exec dialect.

#ifndef MDIR_DIALECT_MDEXEC_MDEXECOPS_H
#define MDIR_DIALECT_MDEXEC_MDEXECOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mdir/Dialect/MD/MDTypes.h"
#include "mdir/Dialect/MDRT/MDRTTypes.h"

#include "mdir/Dialect/MDExec/MDExecEnums.h.inc"

namespace mdir {
namespace md_exec {

/// Returns true if `type` is the type of a buffer that holds a field:
/// `memref<?x3xT>` or `memref<?xT>`.
bool isBufferType(mlir::Type type);

/// The type of one value of a field inside a kernel, for the type of the
/// field or of the buffer that holds it.
mlir::Type getKernelValueType(mlir::Type fieldOrBuffer);

} // namespace md_exec
} // namespace mdir

#define GET_OP_CLASSES
#include "mdir/Dialect/MDExec/MDExecOps.h.inc"

#endif // MDIR_DIALECT_MDEXEC_MDEXECOPS_H
