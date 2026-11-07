#include "FrameBatchTailIR.h"
#include "BuilderImpl.h"
#include "mdir/Driver/Model.h"
#include <cmath>
using namespace mdir::driver;
using namespace mdir::driver::builder;
llvm::Expected<FrameBatchTailPlan> mdir::driver::builder::makeFrameBatchTailPlan(
    const TermInputs &context, const Program &program, const model::TunableSet &set,
    llvm::ArrayRef<int64_t> offsets) {
  FrameBatchTailPlan plan;
  const auto &control = context.getControl();
  const auto &system = context.getSystem();
  plan.cutoff = control.cutoffDistance;
  plan.fixedFactor = context.getShiftFixedFactor();
  plan.shifted = control.truncation == Truncation::Shift;
  plan.initialCorrections = {program.dispersionEnergy, program.dispersionFixedEnergy,
      program.gradientShiftEnergy, program.gradientShiftFixedEnergy, program.dispersionVirial};
  if (!system.topology) return plan;
  double n = system.getNumParticles();
  double ordered = n * (n - 1) - 2.0 * system.topology->exclusions.size();
  double volume = system.box[0] * system.box[1] * system.box[2] /
      (units::length * units::length * units::length);
  plan.factor = 4.0 * M_PI / volume * (ordered > 0 ? n * n / ordered : 0) * units::energy;
  for (auto [index, pairs] : llvm::enumerate(system.pairTails)) {
    if (pairs.empty() || !context.includesPair(index)) continue;
    auto expression = Expression::parse(control.pairs[index].expression, control.functions);
    if (!expression) return expression.takeError();
    FrameBatchTailPlan::Term term(expression->prepareIR());
    for (auto [tunable, entry] : llvm::enumerate(set.tunables)) {
      if (entry.kind != model::TunableSet::Entry::PairConstant || entry.termIndex != index ||
          entry.map.empty() || entry.map[0] < 0) continue;
      int64_t raw = -1;
      for (auto [slot, gradient] : llvm::enumerate(program.gradientSlots))
        if (gradient.tunable == tunable) raw = slot;
      term.parameters.push_back({entry.parameter, offsets[tunable] + entry.map[0], raw});
    }
    if (term.parameters.empty()) continue;
    for (const auto &pair : pairs) term.pairs.push_back({pair.values, pair.count});
    for (auto [column, observed] : llvm::enumerate(control.observables))
      if (observed.term == control.pairs[index].name)
        term.observed.push_back({static_cast<unsigned>(column), observed.constant});
    auto initial = getCustomPairTail(context, index, {});
    if (!initial) return initial.takeError();
    term.initial = {initial->energy, initial->virial, initial->shift, initial->shiftFixed};
    plan.terms.push_back(std::move(term));
  }
  return plan;
}
