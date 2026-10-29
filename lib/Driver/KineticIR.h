// Typed particle reductions for kinetic observables and coupling.
#ifndef MDIR_LIB_DRIVER_KINETICIR_H
#define MDIR_LIB_DRIVER_KINETICIR_H

#include "mlir/IR/Value.h"

namespace mlir {
class OpBuilder;
}
namespace mdir::driver::builder {
mlir::Value buildKineticVector(mlir::OpBuilder &, mlir::Value velocities,
                               mlir::Value masses);
mlir::Value buildKineticEnergy(mlir::OpBuilder &, mlir::Value velocities,
                               mlir::Value masses);
mlir::Value buildMomentum(mlir::OpBuilder &, mlir::Value velocities,
                          mlir::Value masses);
mlir::Value buildForceSquare(mlir::OpBuilder &, mlir::Value forces,
                             mlir::Value masses);
mlir::Value buildKineticExcess(mlir::OpBuilder &, mlir::Value forces,
                               mlir::Value masses, mlir::Value dt);
mlir::Value buildCorrectedHalfVelocity(mlir::OpBuilder &, mlir::Value kicked,
                                       mlir::Value corrected,
                                       mlir::Value drifted, mlir::Value interval);
mlir::Value buildHalfStepKinetic(mlir::OpBuilder &, mlir::Value first,
                                  mlir::Value second, mlir::Value masses);
struct SolventKineticInputs {
  mlir::Value relation, positions, cell, full, first, second, masses;
};
mlir::Value buildSolventKinetic(mlir::OpBuilder &, const SolventKineticInputs &);
} // namespace mdir::driver::builder
#endif
