// Dispatch one mixed directed pair region over a freshly rebuilt tower grid.
#pragma once
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

namespace mdir::kernels {
void markTowersExclusions(mlir::ModuleOp module);
using TowersRowLoop = llvm::function_ref<void(
    mlir::OpBuilder &, mlir::Location, mlir::Value,
    llvm::function_ref<void(mlir::OpBuilder &, mlir::Value)>)>;
using TowersTransfer = llvm::function_ref<void(
    mlir::OpBuilder &, mlir::Location, mlir::Value, mlir::Value)>;
mlir::LogicalResult lowerTowersPair(md_exec::PairForOp pair,
                                    mlir::Value positions, bool masked,
                                    unsigned ordinal, TowersRowLoop rows,
                                    TowersTransfer transfer);
} // namespace mdir::kernels
