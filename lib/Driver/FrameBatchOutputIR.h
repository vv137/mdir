// Output writes on the caller's device stream; no host result buffer.
#ifndef MDIR_LIB_DRIVER_FRAMEBATCHOUTPUTIR_H
#define MDIR_LIB_DRIVER_FRAMEBATCHOUTPUTIR_H
#include "mlir/IR/Value.h"
#include "llvm/Support/Error.h"
namespace mlir { class OpBuilder; }
namespace mdir::driver::builder {
/// Status is one device i64: INT64_MAX means success, otherwise
/// (frame << 16) | reason. Atomic minimum selects the first failing frame.
llvm::Error buildFrameBatchStatus(mlir::OpBuilder &, mlir::Value status);
/// Force and destination rows are N by 3; identities maps reordered atoms
/// back to input order. f32 forces can widen exactly to f64, never narrow.
llvm::Error buildFrameBatchPositionOutput(mlir::OpBuilder &, mlir::Value forces,
    mlir::Value identities, mlir::Value destination, mlir::Value frame,
    mlir::Value status);
llvm::Error buildFrameBatchScalarOutput(mlir::OpBuilder &, mlir::Value scalar,
    mlir::Value destination, mlir::Value frame, mlir::Value status,
    unsigned reason);
/// Transpose and negate W, then subtract the isotropic correction on the
/// diagonal, as Output::getFrameStrain does. Destination is K by 9 f64.
llvm::Error buildFrameBatchStrainOutput(mlir::OpBuilder &, mlir::Value virial,
    mlir::Value correction, mlir::Value destination, mlir::Value frame,
    mlir::Value status);
}
#endif
