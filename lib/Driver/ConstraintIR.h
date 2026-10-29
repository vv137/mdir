// Typed inputs for constraint reductions at their existing kernel sites.
#ifndef MDIR_LIB_DRIVER_CONSTRAINTIR_H
#define MDIR_LIB_DRIVER_CONSTRAINTIR_H

#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/ArrayRef.h"
#include <utility>

namespace mlir {
class OpBuilder;
}
namespace mdir::driver::builder {
struct ConstraintVirialInputs {
  mlir::Value relation, positions, cell, change, masses, timeStep, virial;
};
struct ConstraintTraceInputs {
  mlir::Value relation, positions, cell, descent, forces, velocities, masses,
      trace;
};
mlir::Value buildConstraintVirial(mlir::OpBuilder &,
                                  const ConstraintVirialInputs &, unsigned arity);
mlir::Value buildConstraintTrace(mlir::OpBuilder &, const ConstraintTraceInputs &,
                                 unsigned arity, bool axes);
struct ConstraintVelocityInputs {
  mlir::Value relation, positions, cell, velocities, masses;
  mlir::ValueRange parameters;
  mlir::Value timeStep, virial;
};
struct ConstraintVelocityResult {
  mlir::Value velocities, virial;
};
ConstraintVelocityResult buildShakeVelocities(
    mlir::OpBuilder &, const ConstraintVelocityInputs &, unsigned arity);
ConstraintVelocityResult buildSettleVelocities(
    mlir::OpBuilder &, const ConstraintVelocityInputs &);
struct ConstraintForceInputs {
  mlir::Value relation, positions, cell, velocities, forces, masses, virial;
};
mlir::Value buildConstraintForceVirial(
    mlir::OpBuilder &, const ConstraintForceInputs &, unsigned arity,
    llvm::ArrayRef<std::pair<unsigned, unsigned>> bonds);
} // namespace mdir::driver::builder
#endif
