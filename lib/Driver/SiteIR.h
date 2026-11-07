// Native virtual-site geometry shared by placement and its work potential.
#ifndef MDIR_LIB_DRIVER_SITEIR_H
#define MDIR_LIB_DRIVER_SITEIR_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
class OpBuilder;
}

namespace mdir::driver::builder {
mlir::Value buildSitePlacement(mlir::OpBuilder &, bool amber, mlir::Value toJ,
                               mlir::Value toK, mlir::Value a, mlir::Value b);
mlir::Value buildAddChange(mlir::OpBuilder &, mlir::Value positions,
                            mlir::Value change);
struct SitePlacementInputs { mlir::Value relation, positions, cell, a, b; };
mlir::Value buildPlacedSites(mlir::OpBuilder &, const SitePlacementInputs &, bool amber);
struct SiteSpreadInputs {
  mlir::Value relation, positions, cell, a, b, forces, virial;
};
struct SiteSpreadResult {
  mlir::Value forces, virial;
};
/// The tangent of the placement of the sites (D[force-matching-sites],
/// #771): the derivative of the placement along the tangent, which replaces
/// the entry of each site of `relation`; for linear sites that of
/// `(1 - a - b) x_i + a x_j + b x_k`, for those of `amber` that of
/// `x_i + a s / |s|` at the positions. A tangent is no position, so the
/// members' tangents are read as they are, never through the minimum
/// image. `forces` of the inputs holds the tangent; `virial` is not read.
mlir::Value buildPlacedSiteTangents(mlir::OpBuilder &, const SiteSpreadInputs &,
                                    bool amber);
SiteSpreadResult buildSiteSpread(mlir::OpBuilder &, const SiteSpreadInputs &,
                                  bool amber, bool derived,
                                  llvm::StringRef potential);
} // namespace mdir::driver::builder
#endif
