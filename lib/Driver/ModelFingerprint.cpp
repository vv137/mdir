// The fingerprint of a model that no control file gave (D172,
// D223): the entries that the control file of the same model
// writes, so that a run of either front end continues the other's.
//
// See docs/python-checkpoints.md.

#include "mdir/Driver/Fingerprint.h"
#include "mdir/Driver/Model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <tuple>

using namespace mdir;
using namespace mdir::model;
using driver::Fingerprint;
using driver::getFingerprintNumber;
using driver::getFingerprintString;

namespace {

/// A number converted to the units of the control file, as the text that a
/// file which writes it with fifteen digits gives: 0.9 nm is "9" Å, not the
/// 9.000000000000002 that the division leaves.
std::string number(double value) {
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "%.15g", value);
  return getFingerprintNumber(std::strtod(buffer, nullptr));
}

std::string flag(bool value) { return value ? "true" : "false"; }

const char *truncationModifier(driver::Truncation truncation) {
  switch (truncation) {
  case driver::Truncation::Shift:
    return "POTENTIAL_SHIFT";
  case driver::Truncation::ForceSwitch:
    return "FORCE_SWITCH";
  case driver::Truncation::PowerForceSwitch:
    return "POWER_FORCE_SWITCH";
  case driver::Truncation::SquaredDistanceSwitch:
    return "SQUARED_DISTANCE_SWITCH";
  default:
    return nullptr;
  }
}

bool switches(driver::Truncation truncation) {
  return truncation == driver::Truncation::Switch ||
         truncation == driver::Truncation::ForceSwitch ||
         truncation == driver::Truncation::PowerForceSwitch ||
         truncation == driver::Truncation::SquaredDistanceSwitch;
}

std::string describePairTerm(const driver::PairTerm &term) {
  std::string text = "{name=" + getFingerprintString(term.name) +
                     ",expression=" + getFingerprintString(term.expression) +
                     ",constants={";
  std::vector<std::pair<std::string, double>> constants(term.constants.begin(),
                                                        term.constants.end());
  std::sort(constants.begin(), constants.end());
  for (size_t k = 0; k != constants.size(); ++k)
    text += (k ? "," : "") + constants[k].first + "=" +
            getFingerprintNumber(constants[k].second);
  text += "},groups=[";
  for (size_t k = 0; k != term.groups.size(); ++k)
    text += (k ? "," : "") + getFingerprintString(term.groups[k]);
  return text + "]}";
}

std::string describeTupleTerm(const driver::TupleTerm &term) {
  std::string text = "{name=" + getFingerprintString(term.name) +
                     ",expression=" + getFingerprintString(term.expression) +
                     ",arity=" + std::to_string(term.arity) + ",particles=";
  std::string particles;
  for (unsigned particle : term.particles)
    particles += std::to_string(particle) + ",";
  text += driver::getFingerprintHash(particles) + ",parameters={";
  std::vector<std::pair<std::string, std::vector<double>>> parameters(
      term.parameters.begin(), term.parameters.end());
  std::sort(parameters.begin(), parameters.end());
  for (size_t k = 0; k != parameters.size(); ++k)
    text += (k ? "," : "") + parameters[k].first + "=" +
            driver::getNumbersHash(parameters[k].second);
  return text + "}}";
}

std::string describeExternalTerm(const driver::ExternalTerm &term) {
  std::string text = "{name=" + getFingerprintString(term.name) +
                     ",expression=" + getFingerprintString(term.expression) +
                     ",selection=" + getFingerprintString(term.selection) +
                     ",particles=";
  std::string particles;
  for (unsigned particle : term.particles)
    particles += std::to_string(particle) + ",";
  text += driver::getFingerprintHash(particles) + ",constants={";
  std::vector<std::pair<std::string, double>> constants(term.constants.begin(),
                                                        term.constants.end());
  std::sort(constants.begin(), constants.end());
  for (size_t k = 0; k != constants.size(); ++k)
    text += (k ? "," : "") + constants[k].first + "=" +
            getFingerprintNumber(constants[k].second);
  text += "},scaling=";
  text += term.scaling == driver::ExternalTerm::Scaling::Cell
              ? "CELL"
              : term.scaling == driver::ExternalTerm::Scaling::None ? "NONE"
                                                                     : "UNSET";
  return text + "}";
}

} // namespace

std::string mdir::model::describeTunables(const TunableSet &set) {
  std::string text;
  for (size_t k = 0; k != set.tunables.size(); ++k) {
    const TunableSet::Entry &entry = set.tunables[k];
    std::string map;
    for (int64_t site : entry.map)
      map += std::to_string(site) + ",";
    text += "name=" + getFingerprintString(entry.name) +
            ",parameter=" + getFingerprintString(entry.parameter) +
            ",term=" + getFingerprintString(entry.term) +
            ",entries=" + std::to_string(entry.entries) +
            ",mixing=" +
            (entry.mixing == driver::Mixing::Geometric ? "geometric"
                                                       : "arithmetic") +
            ",map=" + driver::getFingerprintHash(map) + "\n";
  }
  return text;
}

Fingerprint mdir::model::getFingerprint(const System &s,
                                        const Integrator &integrator,
                                        const Ensemble &ensemble,
                                        const Execution &execution,
                                        const GivenSettings &given,
                                        const PreparedModel &prepared) {
  Fingerprint fingerprint;
  // A value longer than the control file's fingerprint shows is its hash,
  // as there.
  auto add = [&](const char *group, std::string name, std::string value) {
    if (value.size() > 160)
      value = driver::getFingerprintHash(value);
    fingerprint.push_back({group, std::move(name), std::move(value)});
  };
  auto has = [](const std::set<std::string> &names, const char *name) {
    return names.count(name) != 0;
  };
  const double length = driver::units::length;
  const double atm = 1.01325;

  // Physics: [energy], [pme], [constraints], [restraints], [boundary].
  // Lengths in Å.
  if (has(given.system, "cutoff") || s.cutoff != 12.0 * length)
    add("physics", "[energy] cutoff", number(s.cutoff / length));
  // The control file switches from the cutoff, which is no switch, unless
  // it gives 'switch_distance'.
  if (switches(s.truncation) &&
      (has(given.system, "switch_distance") || s.switchDistance != s.cutoff))
    add("physics", "[energy] switch_distance",
        number(s.switchDistance / length));
  if (const char *modifier = truncationModifier(s.truncation))
    add("physics", "[energy] lennard_jones_modifier",
        getFingerprintString(modifier));
  if (has(given.system, "electrostatics") ||
      s.electrostatics != Electrostatics::Cutoff)
    add("physics", "[energy] electrostatics",
        getFingerprintString(s.electrostatics == Electrostatics::PME
                                 ? "PME"
                                 : "CUTOFF"));
  if (has(given.system, "coulomb_modifier") ||
      s.coulombModifier != CoulombModifier::None)
    add("physics", "[energy] coulomb_modifier",
        getFingerprintString(s.coulombModifier ==
                                     CoulombModifier::PotentialShift
                                 ? "POTENTIAL_SHIFT"
                                 : "NONE"));
  if (has(given.system, "dispersion"))
    add("physics", "[energy] dispersion_correction",
        getFingerprintString(s.dispersion ==
                                     driver::DispersionCorrection::None
                                 ? "NONE"
                                 : "ENERGY_PRESSURE"));
  if (has(given.system, "pme_alpha"))
    add("physics", "[pme] beta", number(s.pmeAlpha * length));
  if (has(given.system, "pme_tolerance"))
    add("physics", "[pme] tolerance", number(s.pmeTolerance));
  if (has(given.system, "pme_spacing"))
    add("physics", "[pme] max_spacing", number(s.pmeSpacing / length));
  if (has(given.system, "pme_grid"))
    add("physics", "[pme] grid",
        "[" + getFingerprintNumber(s.pmeGrid[0]) + "," +
            getFingerprintNumber(s.pmeGrid[1]) + "," +
            getFingerprintNumber(s.pmeGrid[2]) + "]");
  if (has(given.system, "pme_order"))
    add("physics", "[pme] order", getFingerprintNumber(s.pmeOrder));
  if (has(given.system, "rigid_hydrogen_bonds"))
    add("physics", "[constraints] hydrogen_bonds",
        flag(s.rigidHydrogenBonds));
  if (has(given.system, "rigid_water") || has(given.system, "flexible_water"))
    add("physics", "[constraints] rigid_water", flag(s.rigidWater));
  if (has(given.system, "water_residues")) {
    std::string text = "[";
    for (size_t k = 0; k != s.waterResidues.size(); ++k)
      text += (k ? "," : "") + getFingerprintString(s.waterResidues[k]);
    add("physics", "[constraints] water_residues", text + "]");
  }
  if (has(given.system, "periodic") || !s.periodic)
    add("physics", "[boundary] type",
        getFingerprintString(s.periodic ? "PERIODIC" : "NONE"));
  if (!s.restraints.empty()) {
    // An array of tables, its keys in the order of their names; the
    // constant in kcal/mol/Å².
    double unit = driver::units::energy / (length * length);
    std::string text = "[";
    for (size_t k = 0; k != s.restraints.size(); ++k) {
      const System::Restraint &r = s.restraints[k];
      text += (k ? ",{" : "{");
      text += "force_constant=" + number(r.forceConstant / unit);
      if (r.scaling == driver::ReferenceScaling::All)
        text += ",reference_scaling=" + getFingerprintString("ALL");
      text += ",selection=" + getFingerprintString(r.selection) + "}";
    }
    add("physics", "[[restraints]]", text + "]");
  }
  // What has no key of the control file, in the units of the model.
  if (!s.pairTerms.empty()) {
    std::string text;
    for (const driver::PairTerm &term : s.pairTerms)
      text += describePairTerm(term);
    add("physics", "[python] pair_terms", driver::getFingerprintHash(text));
  }
  if (!s.tupleTerms.empty()) {
    std::string text;
    for (const driver::TupleTerm &term : s.tupleTerms)
      text += describeTupleTerm(term);
    add("physics", "[python] tuple_terms", driver::getFingerprintHash(text));
  }
  if (!s.externalTerms.empty()) {
    std::string text;
    for (const driver::ExternalTerm &term : s.externalTerms)
      text += describeExternalTerm(term);
    add("physics", "[python] external_terms",
        driver::getFingerprintHash(text));
  }
  if (!prepared.tunables.empty()) {
    // The declarations and the values the model begins with; a
    // continuation takes the values of the checkpoint.
    std::string text = describeTunables(prepared.tunables);
    for (const std::vector<double> &values : prepared.tunables.values)
      text += driver::getNumbersHash(values) + "\n";
    add("physics", "[python] tunables", driver::getFingerprintHash(text));
  }
  if (!s.topologyFilesHash.empty())
    add("physics", "the files of the topology", s.topologyFilesHash);
  add("physics", "the masses", driver::getNumbersHash(prepared.system.masses));
  if (!s.restraints.empty())
    add("physics", "the reference of the restraints",
        driver::getNumbersHash(prepared.system.referencePositions));

  // Coupling: [dynamics] but 'steps', [ensemble], [thermostat], [barostat].
  if (has(given.integrator, "method"))
    add("coupling", "[dynamics] integrator",
        getFingerprintString(integrator.method ==
                                     driver::Integrator::Leapfrog
                                 ? "LEAPFROG"
                                 : integrator.method ==
                                           driver::Integrator::Brownian
                                       ? "BROWNIAN"
                                       : "VELOCITY_VERLET"));
  if (has(given.integrator, "timestep"))
    add("coupling", "[dynamics] time_step", number(integrator.timestep));
  if (has(given.ensemble, "seed"))
    add("coupling", "[dynamics] seed",
        getFingerprintNumber(static_cast<double>(ensemble.seed)));
  bool thermostat = ensemble.kind != EnsembleKind::NVE;
  bool barostat = ensemble.kind == EnsembleKind::NPT;
  // With a thermostat the control file removes the motion of the center of
  // mass at its interval unless it says otherwise.
  int64_t comAbsent = thermostat ? ensemble.couplingPeriod : 0;
  if (has(given.ensemble, "com_period") || ensemble.comPeriod != comAbsent)
    add("coupling", "[dynamics] center_of_mass_interval",
        getFingerprintNumber(static_cast<double>(ensemble.comPeriod)));
  if (has(given.ensemble, "kind") || thermostat)
    add("coupling", "[ensemble] ensemble",
        getFingerprintString(barostat     ? "NPT"
                             : thermostat ? "NVT"
                                          : "NVE"));
  if (has(given.ensemble, "temperature"))
    add("coupling", "[ensemble] temperature", number(ensemble.temperature));
  if (has(given.ensemble, "pressure"))
    add("coupling", "[ensemble] pressure", number(ensemble.pressure / atm));
  if (thermostat) {
    add("coupling", "[thermostat] method", getFingerprintString("V-RESCALE"));
    if (has(given.ensemble, "tau_t"))
      add("coupling", "[thermostat] time_constant", number(ensemble.tauT));
    if (has(given.ensemble, "coupling_period") ||
        ensemble.couplingPeriod != 10)
      add("coupling", "[thermostat] interval",
          getFingerprintNumber(static_cast<double>(ensemble.couplingPeriod)));
  }
  if (barostat) {
    add("coupling", "[barostat] method", getFingerprintString("C-RESCALE"));
    if (has(given.ensemble, "tau_p"))
      add("coupling", "[barostat] time_constant", number(ensemble.tauP));
    if (has(given.ensemble, "compressibility"))
      add("coupling", "[barostat] compressibility",
          number(ensemble.compressibility * atm));
  }

  // Execution: [execution], and the reach of the neighbor structures.
  // The control file takes 'pairlist_distance' 1.5 Å beyond the cutoff.
  if (has(given.system, "pairlist_distance") ||
      std::fabs(s.pairlistDistance - (s.cutoff + 1.5 * length)) > 1e-12)
    add("execution", "[energy] pairlist_distance",
        number(s.pairlistDistance / length));
  // A dual list, where there is one, as the key of the control file; none
  // (0) has no key there (D245).
  if (s.prunedDistance != 0.0)
    add("execution", "[energy] pruned_distance",
        number(s.prunedDistance / length));
  if (has(given.execution, "neighbor_structure"))
    add("execution", "[execution] neighbor_structure",
        getFingerprintString(execution.neighborStructure ==
                                     driver::NeighborStructure::Groups
                                 ? "GROUPS"
                                 : "MATRIX"));
  if (has(given.execution, "target"))
    add("execution", "[execution] target",
        getFingerprintString(execution.target == driver::Target::GPU ? "GPU"
                                                                     : "CPU"));
  if (has(given.execution, "precision"))
    add("execution", "[execution] precision",
        getFingerprintString(
            execution.precision == driver::Precision::Mixed    ? "MIXED"
            : execution.precision == driver::Precision::Single ? "SINGLE"
                                                               : "DOUBLE"));
  if (has(given.execution, "threads"))
    add("execution", "[execution] threads",
        getFingerprintNumber(static_cast<double>(execution.threads)));
  if (has(given.execution, "deterministic"))
    add("execution", "[execution] deterministic",
        flag(execution.deterministic));
  if (has(given.execution, "reorder"))
    add("execution", "[execution] spatial_order", flag(execution.reorder));
  if (has(given.execution, "fast_math"))
    add("execution", "[execution] fast_math", flag(execution.fastMath));
  // The capacity that was given, as the control file writes the key; the
  // estimate (0) has no key there (D227).
  if (has(given.execution, "neighbor_capacity") &&
      execution.neighborCapacity > 0)
    add("execution", "[execution] neighbor_capacity",
        getFingerprintNumber(static_cast<double>(execution.neighborCapacity)));

  std::stable_sort(fingerprint.begin(), fingerprint.end(),
                   [](const driver::FingerprintEntry &a,
                      const driver::FingerprintEntry &b) {
                     return std::tie(a.group, a.name) <
                            std::tie(b.group, b.name);
                   });
  return fingerprint;
}
