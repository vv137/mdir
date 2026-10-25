// Set identity on storage-form buffers and its lowering boundary.

#ifndef MDIR_DIALECT_MDEXEC_STORAGESET_H
#define MDIR_DIALECT_MDEXEC_STORAGESET_H

#include "mlir/IR/BuiltinOps.h"

namespace mdir::md_exec {

/// Verify every cast into set-tagged storage in a typed-storage module.
mlir::LogicalResult verifyStorageSetCasts(mlir::ModuleOp module);

/// Check set-preserving casts, then erase identity layouts before lowering.
mlir::LogicalResult eraseStorageSets(mlir::ModuleOp module);

} // namespace mdir::md_exec

#endif // MDIR_DIALECT_MDEXEC_STORAGESET_H
