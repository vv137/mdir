// A validated private theta buffer; no caller input is modified or read back.
#ifndef MDIR_LIB_DRIVER_FRAMEBATCHINPUTIR_H
#define MDIR_LIB_DRIVER_FRAMEBATCHINPUTIR_H
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/Error.h"
namespace mdir::driver::builder {
llvm::Expected<mlir::func::FuncOp> buildFrameBatchParameterReader(mlir::OpBuilder &);
}
#endif
