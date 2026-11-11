// Recognizes constant-parameter rigid water without relying on residue names.
#include "mdir/Driver/WaterTemplate.h"
#include "mdir/Driver/Selection.h"
#include "ConstraintIR.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <tuple>

using namespace mdir::driver;

namespace {
struct Components {
  std::vector<unsigned> parent;
  explicit Components(size_t n) : parent(n) {
    std::iota(parent.begin(), parent.end(), 0);
  }
  unsigned root(unsigned i) {
    while (i != parent[i]) {
      parent[i] = parent[parent[i]];
      i = parent[i];
    }
    return i;
  }
  void join(unsigned a, unsigned b) {
    a = root(a); b = root(b);
    if (a != b) parent[std::max(a, b)] = std::min(a, b);
  }
};

bool sameModel(const WaterTemplateModel &a, const WaterTemplateModel &b) {
  return std::tie(a.sites, a.charges, a.types, a.distanceOH, a.distanceHH,
                  a.distanceOS) ==
         std::tie(b.sites, b.charges, b.types, b.distanceOH, b.distanceHH,
                  b.distanceOS);
}
struct Candidate {
  WaterTemplateModel model;
  std::vector<std::array<unsigned, 4>> instances;
};
} // namespace

llvm::Expected<WaterTemplatePlan>
mdir::driver::matchWaterTemplate(const Control &control, const System &system) {
  WaterTemplatePlan plan;
  if (!system.topology) {
    plan.solute.resize(system.getNumParticles());
    std::iota(plan.solute.begin(), plan.solute.end(), 0);
    return plan;
  }
  const Topology &t = *system.topology;
  const size_t n = t.getNumParticles(), nt = t.getNumTypes();
  Components components(n);
  for (const auto &b : t.bonds) components.join(b.i, b.j);
  // Prepared topologies have removed the constrained water bonds.
  std::map<unsigned, const Topology::Settle *> settles;
  for (const auto &s : t.settles) {
    components.join(s.oxygen, s.oxygen + 1);
    components.join(s.oxygen, s.oxygen + 2);
    settles.emplace(s.oxygen, &s);
  }
  std::map<unsigned, const Topology::VirtualSite *> sites;
  for (const auto &s : t.virtualSites) {
    components.join(s.site, s.i);
    components.join(s.site, s.j);
    components.join(s.site, s.k);
    sites.emplace(s.site, &s);
  }
  for (const auto &s : t.shakes)
    for (unsigned h : s.hydrogens) components.join(s.center, h);
  std::map<unsigned, std::vector<unsigned>> molecules;
  for (unsigned i = 0; i != n; ++i) molecules[components.root(i)].push_back(i);

  std::vector<bool> custom(n, false);
  for (size_t term = 0; term != control.pairs.size(); ++term) {
    if (term >= system.pairGroups.size() || system.pairGroups[term].empty()) {
      std::fill(custom.begin(), custom.end(), true);
      break;
    }
    for (const auto &group : system.pairGroups[term])
      for (size_t i = 0; i != n; ++i) custom[i] = custom[i] || group[i];
  }
  for (const auto &p : control.particleParameters) {
    if (!p.selection.empty()) {
      auto selected = selectParticles(p.selection, t);
      if (!selected) return selected.takeError();
      for (size_t i = 0; i != n; ++i) custom[i] = custom[i] || (*selected)[i];
    } else if (!p.particles.empty()) {
      for (unsigned i : p.particles) custom[i] = true;
    } else {
      std::fill(custom.begin(), custom.end(), true);
    }
  }

  std::set<std::pair<unsigned, unsigned>> exclusions(t.exclusions.begin(),
                                                    t.exclusions.end());
  std::vector<bool> crossExcluded(n, false);
  for (auto [i, j] : t.exclusions)
    if (components.root(i) != components.root(j))
      crossExcluded[i] = crossExcluded[j] = true;
  std::vector<Candidate> models;
  for (const auto &[root, members] : molecules) {
    unsigned oxygens = 0, hydrogens = 0, other = 0;
    for (unsigned i : members) {
      oxygens += t.atomicNumbers[i] == 8;
      hydrogens += t.atomicNumbers[i] == 1;
      other += t.atomicNumbers[i] != 8 && t.atomicNumbers[i] != 1 && t.masses[i] != 0;
    }
    if (oxygens != 1 || hydrogens != 2 || other) continue;
    ++plan.candidates;
    auto reject = [&](const char *reason) { ++plan.mismatches[reason]; };
    if (members.size() != 3 && members.size() != 4) {
      reject("site count"); continue;
    }
    unsigned o = members[0];
    if (members[1] != o + 1 || members[2] != o + 2 ||
        t.atomicNumbers[o] != 8 || t.atomicNumbers[o + 1] != 1 ||
        t.atomicNumbers[o + 2] != 1 ||
        (members.size() == 4 && members[3] != o + 3)) {
      reject("atom order"); continue;
    }
    auto found = settles.find(o);
    if (found == settles.end()) {
      reject("flexible or partially constrained water"); continue;
    }
    if (control.ljpme) { reject("LJ-PME"); continue; }
    if (llvm::any_of(members, [&](unsigned i) { return custom[i]; })) {
      reject("custom pair or particle parameter"); continue;
    }
    if (llvm::any_of(members, [&](unsigned i) {
          return i < system.alchemical.size() && system.alchemical[i];
        })) {
      reject("free-energy selection"); continue;
    }
    if (llvm::any_of(members, [&](unsigned i) { return crossExcluded[i]; })) {
      reject("intermolecular exclusion"); continue;
    }
    bool complete = true;
    for (size_t a = 0; a != members.size(); ++a)
      for (size_t b = a + 1; b != members.size(); ++b)
        complete &= exclusions.count({members[a], members[b]}) != 0;
    if (!complete) { reject("incomplete intramolecular exclusions"); continue; }

    WaterTemplateModel model;
    model.sites = members.size();
    bool hydrogenLJ = false, siteLJ = false;
    for (unsigned s = 0; s != model.sites; ++s) {
      unsigned i = members[s], type = t.types[i];
      model.charges[s] = t.charges[i]; model.types[s] = type;
      if (s)
        for (size_t j = 0; j != nt; ++j) {
          bool lj = t.epsilon[type * nt + j] != 0.0 ||
                    t.epsilon[j * nt + type] != 0.0;
          if (s < 3) hydrogenLJ |= lj; else siteLJ |= lj;
        }
    }
    if (hydrogenLJ) { reject("LJ on the hydrogens"); continue; }
    if (siteLJ) { reject("LJ on the extra point"); continue; }
    model.distanceOH = found->second->distanceOH;
    model.distanceHH = found->second->distanceHH;
    if (model.sites == 4) {
      auto site = sites.find(o + 3);
      if (site == sites.end() || site->second->kind != Topology::VirtualSite::AmberWater ||
          site->second->i != o ||
          !((site->second->j == o + 1 && site->second->k == o + 2) ||
            (site->second->j == o + 2 && site->second->k == o + 1)) ||
          t.masses[o + 3] != 0.0) {
        reject("extra-point definition"); continue;
      }
      model.distanceOS = site->second->a;
      double bisector = 2.0 * std::sqrt(model.distanceOH * model.distanceOH -
                                     0.25 * model.distanceHH * model.distanceHH);
      model.alpha = model.distanceOS / bisector;
    }
    unsigned oxygenType = t.types[o];
    model.sigma = t.sigma[oxygenType * nt + oxygenType];
    model.epsilon = t.epsilon[oxygenType * nt + oxygenType];
    model.extent = std::max(model.distanceOH, model.distanceOS) + 1e-3 * units::length;
    auto group = llvm::find_if(models, [&](const Candidate &c) {
      return sameModel(c.model, model);
    });
    if (group == models.end()) {
      models.push_back({model, {}}); group = std::prev(models.end());
    }
    group->instances.push_back({o, o + 1, o + 2, model.sites == 4 ? o + 3 : o});
  }
  const Candidate *best = nullptr;
  for (const auto &model : models)
    if (!best || model.instances.size() > best->instances.size()) best = &model;
  std::vector<bool> matched(n, false);
  if (best) {
    plan.model = best->model; plan.instances = best->instances;
    for (const auto &model : models)
      if (&model != best) plan.mismatches["other water template"] += model.instances.size();
    for (const auto &instance : plan.instances)
      for (unsigned s = 0; s != plan.model.sites; ++s) matched[instance[s]] = true;
  }
  for (unsigned i = 0; i != n; ++i)
    if (!matched[i]) plan.solute.push_back(i);
  // Heavy-atom anchors of the solute (#799): a hydrogen whose bond the
  // constraints keep belongs to the unit of its heavy atom. Unmatched
  // rigid waters are units of their oxygen. Every other solute particle,
  // a stored extra point included, is its own anchor with radius zero.
  std::vector<int> anchorOf(n, -1), dependents(n, 0);
  auto anchor = [&](unsigned member, unsigned center, double length) {
    if (matched[member] || matched[center] || anchorOf[member] >= 0 ||
        anchorOf[center] >= 0 || dependents[member] || dependents[center] == 4)
      return;
    anchorOf[member] = int(center);
    ++dependents[center];
    plan.anchored.push_back({member, center});
    plan.anchorLength = std::max(plan.anchorLength, length);
  };
  for (const auto &s : t.shakes)
    for (size_t k = 0; k != s.hydrogens.size(); ++k)
      anchor(s.hydrogens[k], s.center, s.lengths[k]);
  for (const auto &s : t.settles) {
    anchor(s.oxygen + 1, s.oxygen, s.distanceOH);
    anchor(s.oxygen + 2, s.oxygen, s.distanceOH);
  }
  for (unsigned i : plan.solute)
    if (t.atomicNumbers[i] == 1 && t.masses[i] != 0 && anchorOf[i] < 0)
      ++plan.unconstrainedHydrogens;
  return plan;
}

double mdir::driver::getAnchorRadius(const WaterTemplatePlan &plan, bool doublePrecision) {
  return plan.anchored.empty() ? 0.0 : plan.anchorLength *
      (1.0 + builder::getConstraintPositionTolerance(doublePrecision));
}

std::string mdir::driver::getWaterTemplateModeReason(const Control &control) {
  if (control.target != Target::GPU) return "the target is the CPU";
  if (control.precision != Precision::Mixed) return "MIXED precision is required";
  if (control.deterministic) return "the deterministic mode";
  if (control.minimize) return "the program minimizes instead of taking dynamics steps";
  if (control.frames || control.frameProduct || control.tunableGradient ||
      control.dependentTerms || control.tunables || !control.observedTunables.empty() ||
      llvm::any_of(control.observables, [](const Control::Observable &o) {
        return !o.constant.empty();
      }))
    return "frame, tunable or derivative programs keep the general path";
  return {};
}

llvm::Expected<WaterTemplateSelection>
mdir::driver::selectWaterTemplate(const Control &control, const System &system,
                                  size_t measuredParticleFloor) {
  WaterTemplateSelection result;
  if (control.waterTemplate == WaterTemplateChoice::Off) {
    result.reason = "OFF was given";
    return result;
  }
  result.reason = getWaterTemplateModeReason(control);
  if (!result.reason.empty()) {
    if (control.waterTemplate == WaterTemplateChoice::On)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
          "'water.template = \"ON\"' cannot apply: %s", result.reason.c_str());
    return result;
  }
  auto matched = matchWaterTemplate(control, system);
  if (!matched) return matched.takeError();
  result.plan = std::move(*matched);
  if (result.reason.empty() && result.plan.instances.empty())
    result.reason = "no rigid water matches the template";
  if (result.reason.empty() && result.plan.unconstrainedHydrogens)
    result.reason = std::to_string(result.plan.unconstrainedHydrogens) +
        " solute hydrogens have no constrained bond to a heavy atom (the "
        "solute's units need [constraints] hydrogen_bonds)";
  if (!result.reason.empty()) {
    if (control.waterTemplate == WaterTemplateChoice::On)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
          "'water.template = \"ON\"' cannot apply: %s", result.reason.c_str());
    return result;
  }
  if (control.waterTemplate == WaterTemplateChoice::Auto) {
    if (result.plan.model.sites != 4) {
      result.reason = "AUTO keeps three-site water on the general path";
      return result;
    }
    if (system.getNumParticles() < measuredParticleFloor) {
      result.reason = "below AUTO's measured particle-count floor";
      return result;
    }
  }
  result.enabled = true;
  result.reason = control.waterTemplate == WaterTemplateChoice::On
                      ? "ON was given" : "four-site water above AUTO's measured floor";
  return result;
}
