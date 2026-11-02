// Complete native entry for the fixed-cell dynamics schedule family.
#ifndef MDIR_LIB_DRIVER_SIMPLESTEPENTRYIR_H
#define MDIR_LIB_DRIVER_SIMPLESTEPENTRYIR_H
#include "EntryBoundaryIR.h"
#include "BarostatIR.h"
#include <array>
#include "EntryConstraintIR.h"
#include "EntryEvaluationIR.h"
#include "PotentialIR.h"
#include "EntryOutputIR.h"
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
  bool triclinic = false;
  EntryOutputIRConfig output;
  bool needsPostEvaluation = false;
  unsigned storedTerms = 0;
  bool stepMemory = false, leapfrog = false, brownian = false;
  std::vector<DescentSiteIR> sites;
  std::vector<EntryConstraintIR> constraints;
  bool restart = false, recomputeForces = false;
  bool needsHostedEvaluation = false;
  std::optional<int64_t> couplingIntervals;
  bool removesMotion = true, rescales = false, noseHoover = false, bathParameters = false;
  int64_t seed = 0;
  double freedom = 0.0, targetKinetic = 0.0, decay = 0.0;
  mdir::dyn::ProgramOp couplingStep;
  bool segments = false;
  int64_t partCounts = 9;
  bool barostat = false, exactBarostat = false;
  bool trotter = false, countsAfterScaling = false;
  mdir::dyn::ProgramOp pressureStep, scaledEnergyStep;
  StrainCoupling strainCoupling = StrainCoupling::Isotropic;
  std::optional<BarostatParameters> barostatParameters;
  std::optional<unsigned> barostatConstant, barostatEnergyConstant;
  std::optional<std::array<unsigned, 3>> referenceEdges;
};
llvm::Expected<mlir::func::FuncOp> buildSimpleStepEntry(
    mlir::OpBuilder &, const Program &, mdir::dyn::ProgramOp step, mdir::dyn::ProgramOp energy,
    const SimpleStepEntryIRConfig &,
    llvm::function_ref<EntryCompletionIR(mlir::OpBuilder &, const PotentialStateIR &,
                                        EnergyResults, mlir::Value scale)> complete = {},
    llvm::function_ref<void(mlir::OpBuilder &, const PotentialStateIR &, mlir::Value step)>
        coordinateOutput = {});
}
#endif
