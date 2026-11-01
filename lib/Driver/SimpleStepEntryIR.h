// Complete native entry for a bare fixed-cell Verlet step loop.
#ifndef MDIR_LIB_DRIVER_SIMPLESTEPENTRYIR_H
#define MDIR_LIB_DRIVER_SIMPLESTEPENTRYIR_H
#include "EntryBoundaryIR.h"
#include "PotentialIR.h"
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
namespace mdir::dyn { class ProgramOp; }
namespace mdir::driver::builder {
struct SimpleStepEntryIRConfig {
  int64_t plainSteps;
  std::optional<int64_t> energyIntervals;
  bool periodic = true;
  std::optional<int64_t> frameIntervals;
  bool framesAtEnergies = false;
  std::optional<int64_t> checkpointIntervals;
  bool topology = false;
  std::vector<std::string> terms;
  bool needsCoordinateOutput = false;
  bool timed = false;
};
llvm::Expected<mlir::func::FuncOp> buildSimpleStepEntry(
    mlir::OpBuilder &, const Program &, mdir::dyn::ProgramOp step, mdir::dyn::ProgramOp energy,
    const SimpleStepEntryIRConfig &,
    llvm::function_ref<void(mlir::OpBuilder &, const PotentialStateIR &, mlir::Value step)>
        coordinateOutput = {});
}
#endif
