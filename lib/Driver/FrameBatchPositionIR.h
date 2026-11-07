// Validate device positions and normalize their physical type in reusable storage.
#ifndef MDIR_LIB_DRIVER_FRAMEBATCHPOSITIONIR_H
#define MDIR_LIB_DRIVER_FRAMEBATCHPOSITIONIR_H
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Error.h"
namespace mdir::driver::builder {
/// Arguments: batch bytes, row index, particle count, input scalar width,
/// reusable row bytes, initial model positions, device status. Returns a
/// fresh N by 3 state-precision descriptor. Invalid coordinates are replaced
/// before neighbor indexing, while the status retains the original failure.
llvm::Expected<mlir::func::FuncOp> buildFrameBatchPositionReader(
    mlir::OpBuilder &, mlir::MemRefType bytes, mlir::Type state,
    bool acceptsF32, bool hasBaseline);
}
#endif
