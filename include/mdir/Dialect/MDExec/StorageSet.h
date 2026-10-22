// Set identity on storage-form buffers and its lowering boundary.

#ifndef MDIR_DIALECT_MDEXEC_STORAGESET_H
#define MDIR_DIALECT_MDEXEC_STORAGESET_H

#include "mlir/IR/BuiltinOps.h"

namespace mdir::md_exec {

/// Check set-preserving casts, then erase identity layouts before lowering.
mlir::LogicalResult eraseStorageSets(mlir::ModuleOp module);

} // namespace mdir::md_exec

#endif // MDIR_DIALECT_MDEXEC_STORAGESET_H
