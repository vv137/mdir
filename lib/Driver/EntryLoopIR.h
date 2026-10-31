// Synchronous native loop bodies consume their actual region arguments.
#ifndef MDIR_LIB_DRIVER_ENTRYLOOPIR_H
#define MDIR_LIB_DRIVER_ENTRYLOOPIR_H
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/Error.h"
namespace mdir::driver::builder {
// Frame inputs may use header parameters, but never replaced positions/cell state.
llvm::Error checkFrameInputs(mlir::Operation &, mlir::ValueRange forbidden);
llvm::Expected<mlir::scf::ForOp> buildEntryLoop(
    mlir::OpBuilder &, llvm::StringRef scope, mlir::Value lower,
    mlir::Value upper, mlir::Value step, mlir::ValueRange initial,
    llvm::function_ref<llvm::Expected<llvm::SmallVector<mlir::Value>>(
        mlir::OpBuilder &, mlir::Value, mlir::ValueRange)> body);
}
#endif
