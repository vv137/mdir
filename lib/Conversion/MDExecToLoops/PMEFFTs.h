// Host PME line transforms, parallel over disjoint planes or column blocks.
#ifndef MDIR_CONVERSION_MDEXECTOLOOPS_PMEFFTS_H
#define MDIR_CONVERSION_MDEXECTOLOOPS_PMEFFTS_H

#include "mlir/IR/BuiltinOps.h"

namespace mdir::kernels {
inline std::string hostFFTName(mlir::Type grid, llvm::StringRef transform) {
  return ((grid.isF64() ? "mdrtFFT64" : "mdrtFFT32") + transform).str();
}
void emitHostPMEFFTs(mlir::ModuleOp module);
} // namespace mdir::kernels
#endif
