// Native entry construction for the plain frame-evaluation family.
#ifndef MDIR_LIB_DRIVER_SIMPLEFRAMEENTRYIR_H
#define MDIR_LIB_DRIVER_SIMPLEFRAMEENTRYIR_H
#include "EntryBoundaryIR.h"
#include "DynamicsIR.h"
#include "OutputIR.h"
#include <optional>
#include <cstdint>
#include <string>
#include <vector>
namespace mdir::driver::builder {
struct FrameConstraintIR {
  unsigned relation, arity;
  llvm::SmallVector<unsigned> parameters;
  bool water;
};
struct FrameCompletionIR {
  EnergyResults physical;
  llvm::SmallVector<mlir::Value> storedTerms;
};
struct SimpleFrameEntryIRConfig {
  bool branchesStart = false, givenForces = false, timed = false;
  unsigned cellEdges = 3;
  bool topology = false;
  unsigned storedTerms = 0;
  std::optional<FreeEnergyOutputConfig> freeEnergy;
  std::vector<ObservableOutputTermIR> observables;
  size_t observableCount = 0;
  unsigned observableBaseArgument = 0;
  // Ordinals in Program.startValues, resolved from semantic control metadata.
  std::vector<unsigned> observableTunables;
  std::vector<std::string> terms;
  std::vector<DescentSiteIR> sites;
  std::vector<FrameConstraintIR> constraints;
  llvm::SmallVector<int64_t> counts, periods;
};
llvm::Expected<mlir::func::FuncOp> buildSimpleFrameEntry(
    mlir::OpBuilder &, const Program &, const SimpleFrameEntryIRConfig &,
    llvm::function_ref<FrameCompletionIR(mlir::OpBuilder &, const PotentialStateIR &,
                                        EnergyResults)> complete = {});
}
#endif
