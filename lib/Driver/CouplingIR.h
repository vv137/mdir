// Typed group inputs for molecular pressure and center scaling.
#ifndef MDIR_LIB_DRIVER_COUPLINGIR_H
#define MDIR_LIB_DRIVER_COUPLINGIR_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
class OpBuilder;
}
namespace mdir::driver::builder {
struct RigidGroupIR {
  mlir::Value relation;
  unsigned arity;
};
struct MolecularDiagonalInputs {
  mlir::Value diagonal, positions, forces, cell, masses;
  llvm::ArrayRef<RigidGroupIR> groups;
};
struct GroupScalingInputs {
  mlir::Value positions, scale, oneMinusScale, cell, masses;
  llvm::ArrayRef<RigidGroupIR> groups;
};
// totalMass is the existing vector broadcast, absent when COM is retained.
struct GroupKineticInputs {
  mlir::Value positions, velocities, forces, cell, masses, timeStep, totalMass;
  llvm::ArrayRef<RigidGroupIR> groups;
};
struct GroupKineticResult {
  mlir::Value kinetic, forceTerm, half;
};
mlir::Value buildMolecularDiagonal(mlir::OpBuilder &,
                                   const MolecularDiagonalInputs &);
mlir::Value buildGroupScaling(mlir::OpBuilder &, const GroupScalingInputs &);
GroupKineticResult buildGroupKinetic(mlir::OpBuilder &, const GroupKineticInputs &);
struct CenterMotionInputs {
  mlir::Value velocities, masses, totalMass, half;
};
struct CenterMotionResult {
  mlir::Value momentum, velocity, energy;
};
CenterMotionResult buildCenterMotion(mlir::OpBuilder &, const CenterMotionInputs &);
mlir::Value buildAxesWithoutCenter(mlir::OpBuilder &, mlir::Value kinetic,
                                     mlir::Value momentum, mlir::Value center,
                                     mlir::Value half);
struct ThermostatIRInputs {
  mlir::Value totalEnergy, centerEnergy, optimal, step, seed, target, freedom, decay;
};
struct ThermostatIRResult {
  mlir::Value factor, factorSquared, heat;
};
ThermostatIRResult buildThermostat(mlir::OpBuilder &, const ThermostatIRInputs &,
                                    bool noseHoover);
mlir::Value buildCoupledVelocities(mlir::OpBuilder &, mlir::Value velocities,
                                     mlir::Value center, mlir::Value factor,
                                     mlir::Value inverse, mlir::Value bath);
} // namespace mdir::driver::builder
#endif
