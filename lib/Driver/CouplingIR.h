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
} // namespace mdir::driver::builder
#endif
