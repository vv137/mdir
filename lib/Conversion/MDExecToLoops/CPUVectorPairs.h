#ifndef MDIR_CPU_VECTOR_PAIRS_H
#define MDIR_CPU_VECTOR_PAIRS_H
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mlir/IR/Builders.h"
namespace mdir {
mlir::LogicalResult checkCPUSIMDKernel(md_exec::PairForOp op);
llvm::SmallVector<mlir::Value>
emitCPUSIMDPairs(mlir::OpBuilder &b, md_exec::PairForOp op, mlir::Value counts,
                 mlir::Value entries, mlir::Value box, mlir::Value inverse,
                 mlir::Value central, int64_t width);
} // namespace mdir
#endif
