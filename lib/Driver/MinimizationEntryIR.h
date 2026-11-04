// Complete native construction of a standalone minimization controller.
#ifndef MDIR_LIB_DRIVER_MINIMIZATIONENTRYIR_H
#define MDIR_LIB_DRIVER_MINIMIZATIONENTRYIR_H
#include "EntryBoundaryIR.h"
#include "EntryEvaluationIR.h"
#include <array>
#include <optional>
namespace mdir::driver::builder {
struct MinimizationEntryIRConfig {
  bool segments = false;
  int64_t partCounts = 9;
  bool timed = false, stepMemory = false, restart = false, stops = true;
  bool topology = false, changesCell = false, scalesEveryStep = false;
  unsigned cellEdges = 3, storedTerms = 0;
  bool needsPostEvaluation = false, needsHostedEvaluation = false;
  int64_t steps = 1, energyPeriod = 1, framePeriod = 0, checkpointPeriod = 0;
  double firstSize = 0.0, cap = 0.0;
  std::optional<std::array<unsigned, 3>> referenceEdges;
  std::vector<DescentSiteIR> sites;
  std::vector<DescentConstraintGroupIR> constraints;
  std::vector<std::string> terms;
};
llvm::Expected<mlir::func::FuncOp> buildMinimizationEntry(mlir::OpBuilder &, const Program &,
    mdir::dyn::ProgramOp descend, const MinimizationEntryIRConfig &,
    llvm::function_ref<EntryCompletionIR(mlir::OpBuilder &, const PotentialStateIR &,
                                        EnergyResults, mlir::Value scale)> complete = {});
}
#endif
