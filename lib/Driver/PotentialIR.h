// Typed physical state at potential evaluation boundaries.
#ifndef MDIR_LIB_DRIVER_POTENTIALIR_H
#define MDIR_LIB_DRIVER_POTENTIALIR_H
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLFunctionalExtras.h"
namespace mlir { class OpBuilder; class MLIRContext; }
namespace mdir::driver { struct Program; }
namespace mdir::md { class PotentialOp; }
namespace mdir::driver::builder {
mdir::md::PotentialOp buildScalarPotential(mlir::OpBuilder &, llvm::StringRef name,
    mlir::TypeRange inputs, llvm::function_ref<mlir::Value(mlir::OpBuilder &, mlir::ValueRange)> body);
llvm::SmallVector<mlir::Value> buildCentroidCoordinates(mlir::OpBuilder &, mlir::ValueRange, size_t groups, unsigned arity);
// Parameter fields, tables and relations in their declared semantic order.
llvm::SmallVector<mlir::Type> buildParameterTypes(mlir::MLIRContext &, const Program &);
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
