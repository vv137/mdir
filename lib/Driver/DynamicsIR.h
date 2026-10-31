// Typed state kernels shared by dynamics and minimization programs.
#ifndef MDIR_LIB_DRIVER_DYNAMICSIR_H
#define MDIR_LIB_DRIVER_DYNAMICSIR_H
#include "mlir/IR/Value.h"
#include "mdir/Dialect/Dyn/DynOps.h"
#include "llvm/Support/Error.h"
#include "ConstraintIR.h"
#include "PotentialIR.h"
#include "NoiseIR.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <vector>
#include <string>
#include "mlir/IR/TypeRange.h"
namespace mlir { class OpBuilder; }
namespace mdir::dyn { class ProgramOp; }
namespace mdir::driver::builder {
inline bool measuresHalfSteps(bool constrained, bool brownian) {
  return constrained && !brownian;
}
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
struct StepProgramIR {
  std::string name;
  bool energy = false, virial = false, scales = false, measured = false, solvent = false;
};
struct StepNoiseIR {
  bool active = false, brownian = false;
  NoiseIRConfig parameters{};
  double friction = 0.0;
};
std::string describeStepCombination(bool leapfrog, const StepNoiseIR &,
    bool constrained, bool withEnergy, bool measured, bool sites);
llvm::Expected<mdir::dyn::ProgramOp> buildNativeStep(
    mlir::OpBuilder &, const StepProgramIR &, mlir::TypeRange parameters,
    bool leapfrog, bool timed, llvm::ArrayRef<DescentSiteIR> sites = {},
    bool referenceScale = false, bool hosted = false,
    llvm::function_ref<EnergyResults(mlir::OpBuilder &, const PotentialStateIR &,
                                    EnergyResults, mlir::Value)> complete = {},
    StepNoiseIR noise = {}, llvm::ArrayRef<DescentConstraintGroupIR> groups = {},
    bool separateTests = false);
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
