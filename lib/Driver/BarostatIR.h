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
mlir::Value buildScaleKinetic(mlir::OpBuilder &, mlir::Value squaredFactor, mlir::Value kinetic);
mlir::Value buildScaledGroupKinetic(mlir::OpBuilder &, mlir::Value squaredFactor,
                                     mlir::Value kinetic, mlir::Value forceTerm);
mlir::Value buildVirialWork(mlir::OpBuilder &, mlir::Value virial, mlir::Value logScale);
struct MeanVirialWorkInputs {
  mlir::Value constantVirial, newVolume, third, groupsAfter, workBefore, half, logScale;
};
mlir::Value buildMeanVirialWork(mlir::OpBuilder &, const MeanVirialWorkInputs &);
mlir::Value buildKineticScaleChange(mlir::OpBuilder &, mlir::Value inverse,
                                     mlir::Value unit, mlir::Value kinetic);
mlir::Value buildTrotterBath(mlir::OpBuilder &, mlir::Value inverse, mlir::Value unit,
                              mlir::Value kinetic, mlir::Value work, mlir::Value bath);
mlir::Value buildOneMinusScale(mlir::OpBuilder &, mlir::Value unit, mlir::Value scale);
mlir::Value buildFirstOrderVirialWork(mlir::OpBuilder &, mlir::Value groups,
                                       mlir::Value constantTrace, mlir::Value third,
                                       mlir::Value oneMinusScale);
mlir::Value buildBathAfterWork(mlir::OpBuilder &, mlir::Value work,
                                mlir::Value kineticChange, mlir::Value bath);
struct ExactBathInputs {
  mlir::ValueRange edges;
  mlir::Value unit, oldVolume, constantEnergy, newEnergy, oldEnergy, kineticChange, bath;
};
mlir::Value buildExactBath(mlir::OpBuilder &, const ExactBathInputs &);
struct BarostatParameters {
  std::array<double, 3> axisCompressibility;
  double pressure, compressibility, rate, thermal, zCompressibility, tension;
};
struct BarostatParameterValues {
  std::array<mlir::Value, 3> axisCompressibility;
  mlir::Value pressure, compressibility, rate, thermal;
  mlir::Value bar, three, third, unit, unitVector, halfVector, zCompressibility, tension;
};
BarostatParameterValues buildBarostatParameters(mlir::OpBuilder &, const BarostatParameters &);
} // namespace mdir::driver::builder
#endif
