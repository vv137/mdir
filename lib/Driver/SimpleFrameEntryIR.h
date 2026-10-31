// Native entry construction for the plain frame-evaluation family.
#ifndef MDIR_LIB_DRIVER_SIMPLEFRAMEENTRYIR_H
#define MDIR_LIB_DRIVER_SIMPLEFRAMEENTRYIR_H
#include "EntryBoundaryIR.h"
#include <cstdint>
namespace mdir::driver::builder {
struct SimpleFrameEntryIRConfig {
  bool branchesStart = false, givenForces = false;
  llvm::SmallVector<int64_t> counts, periods;
};
llvm::Expected<mlir::func::FuncOp> buildSimpleFrameEntry(
    mlir::OpBuilder &, const Program &, const SimpleFrameEntryIRConfig &);
}
#endif
