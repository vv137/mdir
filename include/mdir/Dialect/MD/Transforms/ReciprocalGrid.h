#ifndef MDIR_DIALECT_MD_TRANSFORMS_RECIPROCALGRID_H
#define MDIR_DIALECT_MD_TRANSFORMS_RECIPROCALGRID_H
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/IR/Builders.h"

namespace mdir::md {
inline constexpr llvm::StringLiteral kReciprocalBeta = "md.reciprocal_beta";
inline constexpr llvm::StringLiteral kReciprocalCoulomb = "md.reciprocal_coulomb";
mlir::LogicalResult verifyGaussianReciprocalKernel(MapGridOp product);
mlir::LogicalResult verifyReciprocalEnergyKernel(SumGridOp energy);
struct ReciprocalGridGraph {
  ScatterStencilOp assignment;
  mlir::Value matrix;
  llvm::SmallVector<mlir::Operation *> order;
};
struct ReciprocalGridDerivative {
  ScatterStencilOp assignment;
  mlir::Value positionGradient;
  mlir::Value chargeGradient;
  mlir::Value virial;
};
// Structural and activity proof of the admitted primitive graph; no marker
// licenses differentiation, fusion or a resident strategy.
mlir::FailureOr<ReciprocalGridGraph> inspectReciprocalGrid(SumGridOp root);
mlir::FailureOr<ReciprocalGridDerivative> reverseReciprocalGrid(
    mlir::OpBuilder &builder, SumGridOp root);
}
#endif
