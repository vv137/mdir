// Typed state and owned tables for observable output construction.
#ifndef MDIR_LIB_DRIVER_OUTPUTIR_H
#define MDIR_LIB_DRIVER_OUTPUTIR_H

#include "mlir/IR/ValueRange.h"
#include "PotentialIR.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mlir {
class OpBuilder;
}
namespace mdir::driver::builder {
struct FreeEnergyOutputConfig {
  std::vector<std::vector<double>> lambdas;
  size_t state;
  unsigned baseArgument;
};
void buildFreeEnergyOutput(mlir::OpBuilder &, const PotentialStateIR &,
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
void buildObservablesOutput(mlir::OpBuilder &, const PotentialStateIR &,
                             mlir::Value step, size_t count,
                             unsigned baseArgument,
                             llvm::ArrayRef<ObservableOutputTermIR>,
                             mlir::ValueRange tunables);
struct TunableGradientOutputConfig {
  std::vector<unsigned> scalarArguments, fieldArguments;
};
void buildTunableGradientOutput(mlir::OpBuilder &, const PotentialStateIR &,
                                 mlir::Value asked, mlir::Value zeroIndex,
                                 mlir::Value oneIndex, mlir::Value ids,
                                 mlir::ValueRange constants,
                                 mlir::ValueRange seeds,
                                 const TunableGradientOutputConfig &);
/// The product of a frame (D381, #726): the derivatives of
/// @tunable in its slots and fields, and those of its tangent along
/// `tangent`, the cotangent of the gradient in the positions, handed over
/// as the derivative of D230 is: the scalars of the energy and then those
/// of the tangent in one call of mdrtWriteTunableGradient, and the fields
/// of the energy and then those of the tangent. With `positions`, also the
/// forces of the energy and those of its tangent, through
/// mdrtWriteFrameProductForces (#771, D383); `spread`, where given, takes
/// each of them from the places of the virtual sites to their atoms first
/// (D384), and `curvature`, where given, adds to the spread forces of the
/// tangent the curvature of the placement, from the forces of the energy
/// before their spread (#771).
void buildTunableProductOutput(
    mlir::OpBuilder &, const PotentialStateIR &, mlir::Value tangent,
    mlir::Value ids, mlir::ValueRange constants, mlir::ValueRange seeds,
    const TunableGradientOutputConfig &, bool positions = false,
    llvm::function_ref<mlir::Value(mlir::OpBuilder &, mlir::Value)> spread = nullptr,
    llvm::function_ref<mlir::Value(mlir::OpBuilder &, mlir::Value, mlir::Value)>
        curvature = nullptr);
void buildWriteSolvent(mlir::OpBuilder &, mlir::Value sums);
mlir::Value buildTermEnergyBuffer(mlir::OpBuilder &, const PotentialStateIR &,
                                    llvm::ArrayRef<std::string> potentials,
                                    size_t count);
void buildWriteTerms(mlir::OpBuilder &, mlir::Value memory);
} // namespace mdir::driver::builder
#endif
