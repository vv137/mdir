// Typed stochastic cell strain and persistent barostat state.
#ifndef MDIR_LIB_DRIVER_BAROSTATIR_H
#define MDIR_LIB_DRIVER_BAROSTATIR_H
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/SmallVector.h"
#include <array>
namespace mlir { class OpBuilder; }
namespace mdir::driver::builder {
enum class StrainCoupling { Isotropic, SemiIsotropic, Anisotropic };
struct StrainIRInputs {
  mlir::Value kinetic, diagonal, boxMemory;
  mlir::ValueRange edgeIndices;
  mlir::Value constantVirial, third, barUnit, unitVector;
  mlir::Value seed, step, target, thermal, rate;
  std::array<mlir::Value, 3> beta;
  mlir::Value half, tension;
};
struct StrainIRResult {
  mlir::Value volume, constantTrace, constantDiagonal, logScale, scale, inverse;
  llvm::SmallVector<mlir::Value, 6> edges;
};
StrainIRResult buildBarostatStrain(mlir::OpBuilder &, const StrainIRInputs &,
                                    StrainCoupling);
void buildStoreBarostatState(mlir::OpBuilder &, mlir::Value memory,
                              mlir::Value diagonal, mlir::Value groups,
                              mlir::Value kinetic);
mlir::Value buildCell(mlir::OpBuilder &, mlir::ValueRange edges);
mlir::Value buildReferenceScale(mlir::OpBuilder &, mlir::ValueRange edges,
                                 mlir::Value referenceEdges);
struct VirialDiagonalIR {
  mlir::Value vector, xx, yy, zz;
};
VirialDiagonalIR buildVirialDiagonal(mlir::OpBuilder &, mlir::Value virial);
mlir::Value buildVirialTrace(mlir::OpBuilder &, mlir::Value virial);
mlir::Value buildSum3(mlir::OpBuilder &, mlir::Value vector);
} // namespace mdir::driver::builder
#endif
