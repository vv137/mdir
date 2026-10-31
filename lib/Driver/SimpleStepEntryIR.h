// Complete native entry for a bare fixed-cell Verlet step loop.
#ifndef MDIR_LIB_DRIVER_SIMPLESTEPENTRYIR_H
#define MDIR_LIB_DRIVER_SIMPLESTEPENTRYIR_H
#include "EntryBoundaryIR.h"
#include <cstdint>
#include <optional>
namespace mdir::dyn { class ProgramOp; }
namespace mdir::driver::builder {
struct SimpleStepEntryIRConfig {
  int64_t plainSteps;
  std::optional<int64_t> energyIntervals;
  bool periodic = true;
  std::optional<int64_t> frameIntervals;
  bool framesAtEnergies = false;
};
llvm::Expected<mlir::func::FuncOp> buildSimpleStepEntry(
    mlir::OpBuilder &, const Program &, mdir::dyn::ProgramOp step, mdir::dyn::ProgramOp energy,
    const SimpleStepEntryIRConfig &);
}
#endif
