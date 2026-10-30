// Typed physical state at potential evaluation boundaries.
#ifndef MDIR_LIB_DRIVER_POTENTIALIR_H
#define MDIR_LIB_DRIVER_POTENTIALIR_H
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/SmallVector.h"
namespace mlir { class OpBuilder; }
namespace mdir::driver::builder {
struct PotentialStateIR {
  mlir::Value positions, cell;
  mlir::ValueRange fields;
  mlir::Value time;
};
llvm::SmallVector<mlir::Value> potentialArguments(const PotentialStateIR &);
struct EnergyRequests {
  bool energy, forces, virial;
};
struct EnergyResults {
  mlir::Value energy, forces, virial;
};
EnergyResults buildEnergyEvaluation(mlir::OpBuilder &, const PotentialStateIR &,
                                      const EnergyRequests &);
mlir::Value buildSimulationTime(mlir::OpBuilder &, mlir::Value step, mlir::Value interval);
} // namespace mdir::driver::builder
#endif
