// Typed state kernels shared by dynamics and minimization programs.
#ifndef MDIR_LIB_DRIVER_DYNAMICSIR_H
#define MDIR_LIB_DRIVER_DYNAMICSIR_H
#include "mlir/IR/Value.h"
#include "ConstraintIR.h"
#include "PotentialIR.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <vector>
#include <string>
#include "mlir/IR/TypeRange.h"
namespace mlir { class OpBuilder; }
namespace mdir::dyn { class ProgramOp; }
namespace mdir::driver::builder {
mdir::dyn::ProgramOp buildNativeVerletStep(
    mlir::OpBuilder &, llvm::StringRef name, mlir::TypeRange parameters,
    bool energy, bool leapfrog, bool timed);
mlir::Value buildDescentDirection(mlir::OpBuilder &, mlir::Value forces, mlir::Value masses);
struct DescentConstraintGroupIR {
  unsigned relation, arity;
  std::vector<unsigned> parameters;
  std::vector<ConstraintBond> bonds;
  ConstraintPositionConfig position;
  bool water, settlePosition;
};
struct DescentSiteIR {
  unsigned relation, a, b;
  bool amber, derived;
  std::string potential;
};
mdir::dyn::ProgramOp buildNativeDescent(
    mlir::OpBuilder &, mlir::TypeRange parameters, double particles, bool timed,
    llvm::ArrayRef<DescentConstraintGroupIR> groups = {},
    llvm::ArrayRef<DescentSiteIR> sites = {}, bool hosted = false,
    llvm::function_ref<EnergyResults(mlir::OpBuilder &, const PotentialStateIR &,
                                    EnergyResults)> complete = {});
mlir::Value buildDescentPositions(mlir::OpBuilder &, mlir::Value positions,
                                   mlir::Value gradient, mlir::Value distance,
                                   mlir::Value count, mlir::Value tiny);
struct ConstrainedTrialIR { mlir::Value positions, intact; };
ConstrainedTrialIR buildConstrainedTrial(mlir::OpBuilder &, mlir::Value candidate,
                                          mlir::Value previous,
                                          mlir::Value broken, mlir::Value zero);
mlir::Value buildTrialEnergy(mlir::OpBuilder &, mlir::Value energy, mlir::Value intact);
mlir::Value buildKick(mlir::OpBuilder &, mlir::Value velocities, mlir::Value forces,
                         mlir::Value masses, mlir::Value interval);
mlir::Value buildDrift(mlir::OpBuilder &, mlir::Value positions,
                          mlir::Value velocities, mlir::Value interval);
struct DriftScaleIR { mlir::Value oneMinus, inverse; };
DriftScaleIR buildDriftScale(mlir::OpBuilder &, mlir::Value scale);
mlir::Value buildVelocityScaling(mlir::OpBuilder &, mlir::Value velocities,
                                   mlir::Value scale);
mlir::Value buildFieldDifference(mlir::OpBuilder &, mlir::Value first, mlir::Value second);
} // namespace mdir::driver::builder
#endif
