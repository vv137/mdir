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
struct SiteSpreadInputs {
  mlir::Value relation, positions, cell, a, b, forces, virial;
};
struct SiteSpreadResult {
  mlir::Value forces, virial;
};
SiteSpreadResult buildSiteSpread(mlir::OpBuilder &, const SiteSpreadInputs &,
                                  bool amber, bool derived,
                                  llvm::StringRef potential);
} // namespace mdir::driver::builder
#endif
