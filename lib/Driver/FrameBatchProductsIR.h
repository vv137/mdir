// Device contractions without materializing a K by M Jacobian.
#ifndef MDIR_LIB_DRIVER_FRAMEBATCHPRODUCTSIR_H
#define MDIR_LIB_DRIVER_FRAMEBATCHPRODUCTSIR_H
#include "mlir/IR/Value.h"
namespace mlir { class OpBuilder; }
namespace mdir::driver::builder {
void buildFrameBatchProductStart(mlir::OpBuilder &, mlir::Value vjp);
void buildFrameBatchRowProducts(mlir::OpBuilder &, mlir::Value row,
    mlir::Value tangent, mlir::Value cotangent, mlir::Value jvp,
    mlir::Value vjp, mlir::Value frame, mlir::Value status);
}
#endif
