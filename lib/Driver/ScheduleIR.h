// Typed loop counters and minimization convergence state.
#ifndef MDIR_LIB_DRIVER_SCHEDULEIR_H
#define MDIR_LIB_DRIVER_SCHEDULEIR_H
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include <cstdint>
namespace mlir { class OpBuilder; }
namespace mdir::driver::builder {
struct LoopIterationIR { mlir::Value index, extent; };
mlir::Value buildStepNumber(mlir::OpBuilder &, llvm::ArrayRef<LoopIterationIR>,
                              mlir::Value current, mlir::Value outerPeriod,
                              mlir::Value period, mlir::Value one, mlir::Value start);
struct MinimizationStopIR { mlir::Value memory, flagIndex, stepIndex, notDone; };
MinimizationStopIR buildMinimizationStop(mlir::OpBuilder &, mlir::Value start, int64_t steps);
struct MinimizationBoundsIR { mlir::Value steps, rows; };
MinimizationBoundsIR buildMinimizationBounds(mlir::OpBuilder &, const MinimizationStopIR &,
                                               mlir::Value steps, mlir::Value zero, mlir::Value one);
mlir::Value buildMinimizationFrameBound(mlir::OpBuilder &, const MinimizationStopIR &,
                                          mlir::Value step, mlir::Value zero, mlir::Value one);
}
#endif
