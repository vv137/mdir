#include "FrameBatchParameterPlan.h"
#include "llvm/ADT/STLExtras.h"
#include <limits>
using namespace mdir;
using namespace mdir::driver;
using namespace mdir::driver::builder;

llvm::Expected<FrameBatchParameterPlan>
mdir::driver::builder::makeFrameBatchParameterPlan(const Program &program,
    const model::TunableSet &set, const Control &control, const System &system) {
  FrameBatchParameterPlan plan;
  plan.types = set.typeSigma.size();
  plan.typeSigma = set.typeSigma;
  plan.typeEpsilon = set.typeEpsilon;
  plan.fixedPairs = set.fixedPairs;
  plan.sigmaMap.assign(plan.types, -1);
  plan.epsilonMap.assign(plan.types, -1);
  plan.sigmaPairs.assign(size_t(plan.types) * plan.types, -1);
  plan.epsilonPairs.assign(size_t(plan.types) * plan.types, -1);
  auto fail = [](const llvm::Twine &message) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
        "device frame parameter plan: " + message);
  };
  if (plan.typeEpsilon.size() != plan.types ||
      (!plan.fixedPairs.empty() && plan.fixedPairs.size() != size_t(plan.types) * plan.types))
    return fail("inconsistent type metadata");
  for (const auto &entry : set.tunables) {
    if (entry.entries > size_t(std::numeric_limits<int64_t>::max() - plan.parameters))
      return fail("theta size overflows int64");
    plan.offsets.push_back(plan.parameters);
    plan.parameters += entry.entries;
  }
  auto resource = [&](FrameBatchMappedResource::Kind kind, unsigned index,
                      unsigned field, size_t size) -> FrameBatchMappedResource & {
    for (auto &candidate : plan.mapped)
      if (candidate.kind == kind && candidate.resource == index && candidate.field == field)
        return candidate;
    plan.mapped.push_back({kind, index, field, std::vector<int64_t>(size, -1)});
    return plan.mapped.back();
  };
  auto bindField = [&](llvm::StringRef name) -> FrameBatchMappedResource * {
    for (auto [index, field] : llvm::enumerate(program.fields))
      if (field.name == name)
        return &resource(FrameBatchMappedResource::Field, index, 0, field.values.size());
    return nullptr; // A provably unread parameter has no physical resource.
  };
  auto bindTuple = [&](llvm::StringRef name, llvm::StringRef parameter)
      -> FrameBatchMappedResource * {
    for (auto [index, tuple] : llvm::enumerate(program.tupleSets))
      if (tuple.name == name)
        for (auto [field, values] : llvm::enumerate(tuple.fields))
          if (values.name == parameter)
            return &resource(FrameBatchMappedResource::TupleField, index, field, values.values.size());
    return nullptr;
  };
  auto bindTable = [&](llvm::StringRef name) -> FrameBatchMappedResource * {
    for (auto [index, table] : llvm::enumerate(program.tables))
      if (table.name == name)
        return &resource(FrameBatchMappedResource::Table, index, 0, table.values.size());
    return nullptr;
  };
  for (auto [tunable, entry] : llvm::enumerate(set.tunables)) {
    std::vector<int64_t> mapped;
    for (int64_t index : entry.map) {
      if (index < -1 || (index >= 0 && size_t(index) >= entry.entries))
        return fail("invalid resolved map of '" + entry.name + "'");
      mapped.push_back(index < 0 ? -1 : plan.offsets[tunable] + index);
    }
    FrameBatchMappedResource *target = nullptr;
    using Entry = model::TunableSet::Entry;
    switch (entry.kind) {
    case Entry::Charge:
      target = bindField("q");
      break;
    case Entry::DihedralK:
      target = bindTuple("dihedrals", "k");
      break;
    case Entry::TupleParameter:
      if (!system.topology || entry.termIndex >= system.topology->tupleTerms.size())
        return fail("invalid tuple term index");
      target = bindTuple("custom_" + system.topology->tupleTerms[entry.termIndex].name,
                         entry.parameter);
      break;
    case Entry::PairConstant:
      if (entry.termIndex >= control.pairs.size())
        return fail("invalid pair term index");
      for (auto [column, constant] : llvm::enumerate(control.tunableConstants)) {
        if (constant.first != entry.termIndex || constant.second != entry.parameter)
          continue;
        if (mapped.size() != 1)
          return fail("a pair constant must have one resolved site");
        if (auto table = bindTable("tunable_constants"))
          table->map[column] = mapped[0];
        for (auto [slot, value] : llvm::enumerate(program.startValues))
          if (mapped[0] >= 0 &&
              (value.name == "tg_c" + std::to_string(column) ||
               value.name == "obv_" + control.pairs[entry.termIndex].name + "_" + entry.parameter))
            plan.scalars.push_back({static_cast<unsigned>(slot), mapped[0]});
      }
      break;
    case Entry::Sigma:
    case Entry::Epsilon: {
      auto &types = entry.kind == Entry::Sigma ? plan.sigmaMap : plan.epsilonMap;
      if (mapped.size() != types.size())
        return fail("a per-type map has the wrong number of sites");
      for (size_t i = 0; i != mapped.size(); ++i)
        if (mapped[i] >= 0) types[i] = mapped[i];
      if (entry.kind == Entry::Sigma) {
        plan.mixesSigma = true;
        plan.geometricSigma = entry.mixing == Mixing::Geometric;
      } else {
        plan.mixesEpsilon = true;
      }
      break;
    }
    case Entry::SigmaPair:
    case Entry::EpsilonPair: {
      auto &pairs = entry.kind == Entry::SigmaPair ? plan.sigmaPairs : plan.epsilonPairs;
      if (mapped.size() != size_t(plan.types) * (plan.types + 1) / 2)
        return fail("a pair map has the wrong number of sites");
      size_t site = 0;
      for (size_t a = 0; a != plan.types; ++a)
        for (size_t b = a; b != plan.types; ++b, ++site)
          if (mapped[site] >= 0)
            pairs[a * plan.types + b] = pairs[b * plan.types + a] = mapped[site];
      break;
    }
    }
    if (target) {
      if (target->map.size() != mapped.size())
        return fail("resolved sites differ from the parameter resource");
      for (size_t i = 0; i != mapped.size(); ++i)
        if (mapped[i] >= 0) target->map[i] = mapped[i];
    }
  }
  // A pair override needs no combining rule. Its map writes both symmetric
  // entries of the physical table; the seed weights stay constant.
  if (!plan.mixesSigma && llvm::any_of(plan.sigmaPairs, [](int64_t i) { return i >= 0; }))
    if (auto table = bindTable("lj_sigma")) table->map = plan.sigmaPairs;
  if (!plan.mixesEpsilon && llvm::any_of(plan.epsilonPairs, [](int64_t i) { return i >= 0; }))
    if (auto table = bindTable("lj_epsilon")) table->map = plan.epsilonPairs;
  auto allocate = [&](size_t count) -> llvm::Expected<int64_t> {
    if (count > size_t(std::numeric_limits<int64_t>::max() - plan.rawValues))
      return fail("raw derivative size overflows int64");
    int64_t base = plan.rawValues;
    plan.rawValues += count;
    return base;
  };
  if (auto scalars = allocate(program.gradientSlots.size()); !scalars)
    return scalars.takeError();
  size_t particles = system.getNumParticles();
  for (const auto &field : program.gradientFields) {
    auto values = allocate(particles);
    if (!values) return values.takeError();
    plan.fieldValues.push_back(*values);
  }
  if (!program.gradientChargeFixed.empty()) {
    if (program.gradientChargeFixed.size() != particles ||
        program.gradientChargeVolume.size() != particles)
      return fail("inconsistent charge correction sizes");
    auto charge = allocate(particles);
    if (!charge) return charge.takeError();
    plan.chargeValues = *charge;
  }
  for (const auto &field : program.gradientFields) {
    auto sums = allocate(field.rows.size());
    if (!sums) return sums.takeError();
    plan.typeSums.push_back(field.rows.empty() ? -1 : *sums);
  }
  std::vector<std::vector<std::pair<int64_t, double>>> rows(plan.parameters);
  auto contribution = [&](unsigned tunable, size_t site, int64_t source,
                          double weight) -> llvm::Error {
    if (tunable >= set.tunables.size() || site >= set.tunables[tunable].map.size() ||
        source < 0 || source >= plan.rawValues)
      return fail("invalid derivative site or raw index");
    int64_t entry = set.tunables[tunable].map[site];
    if (entry >= 0)
      rows[plan.offsets[tunable] + entry].push_back({source, weight});
    return llvm::Error::success();
  };
  // Preserve applyTunableChainRule's phases and contribution order for each
  // output entry; interleaving independent output entries is unnecessary.
  for (auto [index, slot] : llvm::enumerate(program.gradientSlots))
    if (llvm::Error error = contribution(slot.tunable, slot.site, index, 1))
      return std::move(error);
  bool groupsTypes = false;
  for (auto [index, field] : llvm::enumerate(program.gradientFields)) {
    if (field.tunable >= set.tunables.size())
      return fail("invalid gradient field tunable");
    for (auto [site, particle] : field.sites) {
      if (particle >= particles) return fail("invalid gradient field particle");
      if (llvm::Error error = contribution(field.tunable, site,
              plan.fieldValues[index] + particle, 1))
        return std::move(error);
    }
    if (plan.chargeValues >= 0 &&
        set.tunables[field.tunable].kind == model::TunableSet::Entry::Charge)
      for (size_t particle = 0; particle != particles; ++particle)
        if (llvm::Error error = contribution(field.tunable, particle,
                plan.chargeValues + particle, 1))
          return std::move(error);
    if (!field.rows.empty()) groupsTypes = true;
    for (auto [type, row] : llvm::enumerate(field.rows))
      if (row.site >= 0)
        if (llvm::Error error = contribution(field.tunable, row.site,
                plan.typeSums[index] + type, row.scale))
          return std::move(error);
  }
  plan.rowOffsets.push_back(0);
  for (const auto &row : rows) {
    for (auto [source, weight] : row) {
      plan.sourceIndices.push_back(source);
      plan.weights.push_back(weight);
    }
    if (plan.sourceIndices.size() > size_t(std::numeric_limits<int64_t>::max()))
      return fail("CSR contribution count overflows int64");
    plan.rowOffsets.push_back(plan.sourceIndices.size());
  }
  if (groupsTypes) {
    if (!system.topology || system.topology->types.size() != particles)
      return fail("type sums need one topology type per particle");
    std::vector<std::vector<int64_t>> byType(plan.types);
    for (size_t particle = 0; particle != particles; ++particle) {
      unsigned type = system.topology->types[particle];
      if (type >= plan.types) return fail("invalid topology type");
      byType[type].push_back(particle);
    }
    plan.typeOffsets.push_back(0);
    for (const auto &group : byType) {
      llvm::append_range(plan.particlesByType, group);
      plan.typeOffsets.push_back(plan.particlesByType.size());
    }
  }
  return plan;
}
