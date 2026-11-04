// Native output rows retain each output's evaluation and time-conversion order.
#include "EntryOutputIR.h"
#include "mlir/IR/Builders.h"
using namespace mlir;
using namespace mdir::driver::builder;
void mdir::driver::builder::buildEntryOutputs(OpBuilder &builder,
    const EntryOutputPointIR &point, const EntryOutputIRConfig &config) {
  auto rowState = [&] {
    PotentialStateIR state = point.physical;
    if (!point.initial && point.timed)
      state.time = buildSimulationTime(builder, point.step, point.interval);
    return state;
  };
  if (config.freeEnergy)
    buildFreeEnergyOutput(builder, rowState(), point.step, point.zero, point.one, *config.freeEnergy);
  if (!config.observables.empty()) {
    SmallVector<Value> tunables;
    for (unsigned slot : config.observableTunables) tunables.push_back(point.startValues[slot]);
    buildObservablesOutput(builder, rowState(), point.step, config.observableCount,
        config.observableBaseArgument, config.observables, tunables);
  }
  if (point.initial && config.gradient) {
    const auto &gradient = *config.gradient;
    SmallVector<Value> constants, seeds;
    for (unsigned slot : gradient.constants) constants.push_back(point.startValues[slot]);
    if (gradient.seeds) seeds.assign(gradient.seeds, point.physical.fields[*gradient.seedField]);
    buildTunableGradientOutput(builder, point.physical, point.startValues[gradient.request],
        point.zero, point.one, gradient.output.fieldArguments.empty() ? Value{} : point.identities,
        constants, seeds, gradient.output);
  }
}
