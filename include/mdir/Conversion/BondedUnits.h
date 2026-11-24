#pragma once
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

namespace mdir::kernels {
mlir::gpu::LaunchOp emitBondedUnits(mlir::OpBuilder &builder,
                                    llvm::ArrayRef<md_exec::TupleForOp> ops,
                                    mlir::Value blocks,
                                    mlir::Value instructions, mlir::Value place,
                                    mlir::Value box, mlir::Value inverse,
                                    mlir::Value forceWords,
                                    mlir::ValueRange observerTerms, int mode);
}
