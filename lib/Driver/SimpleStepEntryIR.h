// Complete native entry for a bare fixed-cell Verlet step loop.
#ifndef MDIR_LIB_DRIVER_SIMPLESTEPENTRYIR_H
#define MDIR_LIB_DRIVER_SIMPLESTEPENTRYIR_H
#include "EntryBoundaryIR.h"
#include <cstdint>
namespace mdir::dyn { class ProgramOp; }
namespace mdir::driver::builder {
llvm::Expected<mlir::func::FuncOp> buildSimpleStepEntry(
    mlir::OpBuilder &, const Program &, mdir::dyn::ProgramOp step, int64_t count);
}
#endif
