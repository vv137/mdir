// Site placement precedes entry reordering; force spreading precedes completion.
#include "EntryEvaluationIR.h"
#include "SiteIR.h"
using namespace mlir;
using namespace mdir::driver::builder;
Value mdir::driver::builder::buildEntrySites(OpBuilder &builder,
    const PotentialStateIR &state, llvm::ArrayRef<DescentSiteIR> sites) {
  Value positions = state.positions;
  for (const auto &site : sites)
    positions = buildPlacedSites(builder,
        {state.fields[site.relation], positions, state.cell,
         state.fields[site.a], state.fields[site.b]}, site.amber);
  return positions;
}
EntryCompletionIR mdir::driver::builder::buildEntryEvaluation(OpBuilder &builder,
    const PotentialStateIR &state, EnergyRequests requests, llvm::ArrayRef<DescentSiteIR> sites,
    llvm::function_ref<EntryCompletionIR(OpBuilder &, const PotentialStateIR &, EnergyResults, Value)> complete,
    Value scale, Value siteCell) {
  auto result = buildEnergyEvaluation(builder, state, requests);
  for (const auto &site : sites) {
    auto spread = buildSiteSpread(builder,
        {state.fields[site.relation], state.positions, siteCell ? siteCell : state.cell,
         state.fields[site.a], state.fields[site.b], result.forces, result.virial},
        site.amber, site.derived, site.potential);
    result.forces = spread.forces;
    result.virial = spread.virial;
  }
  return complete ? complete(builder, state, result, scale) : EntryCompletionIR{result, {}};
}
