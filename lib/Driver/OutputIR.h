// Typed state and owned tables for observable output construction.
#ifndef MDIR_LIB_DRIVER_OUTPUTIR_H
#define MDIR_LIB_DRIVER_OUTPUTIR_H

#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/ArrayRef.h"
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mlir {
class OpBuilder;
}
namespace mdir::driver::builder {
struct OutputStateIR {
  mlir::Value positions, cell;
  mlir::ValueRange fields;
  mlir::Value time;
};
struct FreeEnergyOutputConfig {
  std::vector<std::vector<double>> lambdas;
  size_t state;
  unsigned baseArgument;
};
void buildFreeEnergyOutput(mlir::OpBuilder &, const OutputStateIR &,
                            mlir::Value step, mlir::Value zeroIndex,
                            mlir::Value oneIndex,
                            const FreeEnergyOutputConfig &);
struct ObservableConstantIR {
  double value;
  std::optional<unsigned> tunable;
};
struct ObservableOutputTermIR {
  std::string potential;
  std::vector<ObservableConstantIR> constants;
  // A negative derivative argument requests energy; the second member is
  // the output slot, in the user's observe order.
  std::vector<std::pair<int64_t, unsigned>> columns;
};
void buildObservablesOutput(mlir::OpBuilder &, const OutputStateIR &,
                             mlir::Value step, size_t count,
                             unsigned baseArgument,
                             llvm::ArrayRef<ObservableOutputTermIR>,
                             mlir::ValueRange tunables);
struct TunableGradientOutputConfig {
  std::vector<unsigned> scalarArguments, fieldArguments;
};
void buildTunableGradientOutput(mlir::OpBuilder &, const OutputStateIR &,
                                 mlir::Value asked, mlir::Value zeroIndex,
                                 mlir::Value oneIndex, mlir::Value ids,
                                 mlir::ValueRange constants,
                                 mlir::ValueRange seeds,
                                 const TunableGradientOutputConfig &);
void buildWriteSolvent(mlir::OpBuilder &, mlir::Value sums);
mlir::Value buildTermEnergyBuffer(mlir::OpBuilder &, const OutputStateIR &,
                                    llvm::ArrayRef<std::string> potentials,
                                    size_t count);
void buildWriteTerms(mlir::OpBuilder &, mlir::Value memory);
} // namespace mdir::driver::builder
#endif
