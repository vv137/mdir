// Typed state kernels shared by dynamics and minimization programs.
#ifndef MDIR_LIB_DRIVER_DYNAMICSIR_H
#define MDIR_LIB_DRIVER_DYNAMICSIR_H
#include "mlir/IR/Value.h"
namespace mlir { class OpBuilder; }
namespace mdir::driver::builder {
mlir::Value buildDescentPositions(mlir::OpBuilder &, mlir::Value positions,
                                   mlir::Value gradient, mlir::Value distance,
                                   mlir::Value count, mlir::Value tiny);
struct ConstrainedTrialIR { mlir::Value positions, intact; };
ConstrainedTrialIR buildConstrainedTrial(mlir::OpBuilder &, mlir::Value candidate,
                                          mlir::Value previous,
                                          mlir::Value broken, mlir::Value zero);
mlir::Value buildTrialEnergy(mlir::OpBuilder &, mlir::Value energy, mlir::Value intact);
} // namespace mdir::driver::builder
#endif
