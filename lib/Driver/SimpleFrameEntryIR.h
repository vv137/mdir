// Native entry construction for the plain frame-evaluation family.
#ifndef MDIR_LIB_DRIVER_SIMPLEFRAMEENTRYIR_H
#define MDIR_LIB_DRIVER_SIMPLEFRAMEENTRYIR_H
#include "EntryBoundaryIR.h"
#include "EntryLoopIR.h"
#include "EntryConstraintIR.h"
#include "EntryEvaluationIR.h"
#include "DynamicsIR.h"
#include "EntryOutputIR.h"
#include "BarostatIR.h"
#include "FrameBatchParameterPlan.h"
#include <optional>
#include <cstdint>
#include <string>
#include <vector>
namespace mdir::driver::builder {
struct FrameThermostatIR {
  int64_t seed;
  double freedom, target, decay;
};
struct SimpleFrameEntryIRConfig {
  // Selected by the internal device-resident batch caller of #138 PR 1.
  bool deviceBatch = false;
  bool batchOnDevice = false;
  std::optional<FrameBatchParameterPlan> batchParameters;
  bool branchesStart = false, givenForces = false, timed = false;
  unsigned cellEdges = 3;
  bool stepMemory = false, leapfrog = false, brownian = false, coupled = false;
  std::optional<FrameThermostatIR> thermostat;
  bool barostat = false, changesCell = false, scalesEveryStep = false, removesMotion = false;
  std::optional<BarostatParameters> barostatParameters;
  std::optional<std::array<unsigned, 3>> referenceEdges;
  bool topology = false;
  unsigned storedTerms = 0;
  EntryOutputIRConfig output;
  std::vector<std::string> terms;
  std::vector<DescentSiteIR> sites;
  std::vector<EntryConstraintIR> constraints;
  llvm::SmallVector<int64_t> counts, periods;
};
struct FrameBatchRowIR {
  mlir::Value index, positions;
  llvm::SmallVector<mlir::Value> edges;
};
/// One native loop over canonical contiguous position bytes and the cells
/// copied once for a batch. Views advance the data pointer, keeping an
/// identity layout for field imports without copying positions.
llvm::Expected<mlir::scf::ForOp> buildFrameBatchRows(
    mlir::OpBuilder &, mlir::Value positionBytes, mlir::Value cells,
    mlir::Value particles, mlir::Type stateType, unsigned cellEdges,
    llvm::function_ref<llvm::Error(mlir::OpBuilder &, const FrameBatchRowIR &)>);
llvm::Expected<mlir::func::FuncOp> buildSimpleFrameEntry(
    mlir::OpBuilder &, const Program &, const SimpleFrameEntryIRConfig &,
    llvm::function_ref<EntryCompletionIR(mlir::OpBuilder &, const PotentialStateIR &,
                                        EnergyResults, mlir::Value scale)> complete = {},
    llvm::function_ref<void(mlir::OpBuilder &, const PotentialStateIR &, mlir::Value step)>
        coordinateOutput = {});
}
#endif
