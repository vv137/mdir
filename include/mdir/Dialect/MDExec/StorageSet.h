// Set identity on storage-form buffers and its lowering boundary.

#ifndef MDIR_DIALECT_MDEXEC_STORAGESET_H
#define MDIR_DIALECT_MDEXEC_STORAGESET_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/StringRef.h"

namespace mdir::md_exec {

inline constexpr llvm::StringLiteral kInputSetBoundaryAttrName =
    "md_exec.input_set_boundary";

/// A declared external input gains a storage-set identity at this cast.
bool hasInputSetBoundary(mlir::memref::CastOp cast);
void markInputSetBoundary(mlir::memref::CastOp cast);
mlir::LogicalResult verifyInputSetBoundaryAttribute(mlir::Operation *op,
                                                     mlir::Attribute value);

/// Verify every cast into set-tagged storage in a typed-storage module.
mlir::LogicalResult verifyStorageSetCasts(mlir::ModuleOp module);

/// Check set-preserving casts, then erase identity layouts before lowering.
mlir::LogicalResult eraseStorageSets(mlir::ModuleOp module);

} // namespace mdir::md_exec

#endif // MDIR_DIALECT_MDEXEC_STORAGESET_H
