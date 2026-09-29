// The control file of a run.

#include "mdir/Driver/Control.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Path.h"

#define TOML_EXCEPTIONS 0
#define TOML_ENABLE_FORMATTERS 0
#include "toml.hpp"

using namespace mdir::driver;
using llvm::Error;
using llvm::StringRef;

namespace {

/// Reads the tables of a control file and reports what is wrong with them,
/// with the line.
class Reader {
public:
  Reader(StringRef path, Control &control) : path(path), control(control) {}

  Error read(const toml::table &root);

private:
  Error fail(const toml::node &node, const llvm::Twine &message) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(), "%s:%u: %s", path.str().c_str(),
        static_cast<unsigned>(node.source().begin.line),
        message.str().c_str());
  }

  /// Verifies that `table` has no keywords but `known`. A keyword in
  /// `planned` is reported with the milestone that brings it.
  Error checkKeywords(
      const toml::table &table, StringRef name,
      std::initializer_list<StringRef> known,
      std::initializer_list<std::pair<StringRef, StringRef>> planned = {});

  Error readReal(const toml::table &table, StringRef key, double &value);
  Error readPositive(const toml::table &table, StringRef key, double &value);
  Error readCount(const toml::table &table, StringRef key, int64_t &value,
                  int64_t least);
  Error readBool(const toml::table &table, StringRef key, bool &value);
  Error readString(const toml::table &table, StringRef key,
                   std::string &value);
  Error readPath(const toml::table &table, StringRef key, std::string &value);

  /// Reads a keyword whose value names one of `choices`, without regard to
  /// case.
  template <typename T>
  Error readChoice(const toml::table &table, StringRef key, T &value,
                   std::initializer_list<std::pair<StringRef, T>> choices);

  Error readEnergy(const toml::table &table);
  Error readPair(const toml::table &table);
  Error readOverride(const toml::table &table);
  Error readType(const toml::table &table);
  Error readDynamics(const toml::table &table);
  Error readEnsemble(const toml::table &table);
  /// Sets the periods of the removal of the motion of the center of mass
  /// and of the thermostat that were not given, and checks them.
  Error resolveCoupling();
  Error readBoundary(const toml::table &table);
  Error readExecution(const toml::table &table);

  StringRef path;
  Control &control;
};

} // namespace

static StringRef toRef(std::string_view text) {
  return StringRef(text.data(), text.size());
}

Error Reader::checkKeywords(
    const toml::table &table, StringRef name,
    std::initializer_list<StringRef> known,
    std::initializer_list<std::pair<StringRef, StringRef>> planned) {
  for (auto &&[key, node] : table) {
    StringRef keyword = toRef(key.str());
    if (llvm::is_contained(known, keyword))
      continue;
    for (auto [future, milestone] : planned)
      if (future == keyword)
        return fail(node, "'" + keyword + "' in [" + name +
                              "] is not supported yet; it is planned for " +
                              milestone);
    return fail(node, "unknown keyword '" + keyword + "' in [" + name + "]");
  }
  return Error::success();
}

Error Reader::readReal(const toml::table &table, StringRef key,
                       double &value) {
  const toml::node *node = table.get(std::string_view(key));
  if (!node)
    return Error::success();
  // A number without a fraction is a number all the same.
  if (auto real = node->value<double>()) {
    if (node->is_number()) {
      value = *real;
      return Error::success();
    }
  }
  return fail(*node, "expected a number for '" + key + "'");
}

Error Reader::readPositive(const toml::table &table, StringRef key,
                           double &value) {
  if (Error error = readReal(table, key, value))
    return error;
  const toml::node *node = table.get(std::string_view(key));
  if (node && !(value > 0.0))
    return fail(*node, "expected a positive number for '" + key + "'");
  return Error::success();
}

Error Reader::readCount(const toml::table &table, StringRef key,
                        int64_t &value, int64_t least) {
  const toml::node *node = table.get(std::string_view(key));
  if (!node)
    return Error::success();
  if (!node->is_integer())
    return fail(*node, "expected an integer for '" + key + "'");
  value = *node->value<int64_t>();
  if (value < least)
    return fail(*node, "expected at least " + llvm::Twine(least) + " for '" +
                           key + "'");
  return Error::success();
}

Error Reader::readBool(const toml::table &table, StringRef key, bool &value) {
  const toml::node *node = table.get(std::string_view(key));
  if (!node)
    return Error::success();
  if (!node->is_boolean())
    return fail(*node, "expected true or false for '" + key + "'");
  value = *node->value<bool>();
  return Error::success();
}

Error Reader::readString(const toml::table &table, StringRef key,
                         std::string &value) {
  const toml::node *node = table.get(std::string_view(key));
  if (!node)
    return Error::success();
  if (!node->is_string())
    return fail(*node, "expected a string for '" + key + "'");
  value = *node->value<std::string>();
  return Error::success();
}

Error Reader::readPath(const toml::table &table, StringRef key,
                       std::string &value) {
  if (Error error = readString(table, key, value))
    return error;
  if (value.empty() || llvm::sys::path::is_absolute(value))
    return Error::success();
  llvm::SmallString<256> full(llvm::sys::path::parent_path(path));
  llvm::sys::path::append(full, value);
  value = std::string(full);
  return Error::success();
}

template <typename T>
Error Reader::readChoice(
    const toml::table &table, StringRef key, T &value,
    std::initializer_list<std::pair<StringRef, T>> choices) {
  const toml::node *node = table.get(std::string_view(key));
  if (!node)
    return Error::success();
  std::string text;
  if (Error error = readString(table, key, text))
    return error;
  std::string names;
  for (auto [name, choice] : choices) {
    if (StringRef(text).equals_insensitive(name)) {
      value = choice;
      return Error::success();
    }
    names += (names.empty() ? "" : ", ") + name.str();
  }
  return fail(*node, "expected one of " + names + " for '" + key +
                         "', got '" + text + "'");
}

//===----------------------------------------------------------------------===//
// Tables
//===----------------------------------------------------------------------===//

Error Reader::readPair(const toml::table &table) {
  PairTerm term;
  if (Error error = readString(table, "name", term.name))
    return error;
  if (Error error = readString(table, "expression", term.expression))
    return error;
  if (term.expression.empty())
    return fail(table, "expected an 'expression' in [[energy.pair]]");
  if (term.name.empty())
    term.name = "pair" + std::to_string(control.pairs.size());

  auto readMixing = [&](const toml::node &node, StringRef text,
                        Mixing &mixing) -> Error {
    if (text.equals_insensitive("arithmetic"))
      mixing = Mixing::Arithmetic;
    else if (text.equals_insensitive("geometric"))
      mixing = Mixing::Geometric;
    else if (text.equals_insensitive("product"))
      mixing = Mixing::Product;
    else
      return fail(node, "expected 'arithmetic', 'geometric', or 'product', "
                        "got '" +
                            text + "'");
    return Error::success();
  };

  for (auto &&[key, node] : table) {
    StringRef keyword = toRef(key.str());
    if (keyword == "name" || keyword == "expression")
      continue;
    if (keyword == "dispersion_corr") {
      if (Error error = readChoice<DispersionCorrection>(
              table, "dispersion_corr", term.dispersion,
              {{"NONE", DispersionCorrection::None},
               {"EPRESS", DispersionCorrection::EnergyPressure}}))
        return error;
      continue;
    }
    if (keyword == "mixing") {
      // The name of a rule, or a rule for each parameter.
      if (auto rules = node.as_table()) {
        for (auto &&[parameter, rule] : *rules) {
          if (!rule.is_string())
            return fail(rule,
                        "expected 'arithmetic', 'geometric', or 'product'");
          Mixing mixing;
          if (Error error = readMixing(rule, *rule.value<std::string>(),
                                       mixing))
            return error;
          term.mixing[toRef(parameter.str())] = mixing;
        }
        continue;
      }
      if (!node.is_string())
        return fail(node, "expected a string or a table for 'mixing'");
      std::string rule = *node.value<std::string>();
      if (StringRef(rule).equals_insensitive("lorentz-berthelot")) {
        term.mixing["sigma"] = Mixing::Arithmetic;
        term.mixing["epsilon"] = Mixing::Geometric;
      } else if (StringRef(rule).equals_insensitive("geometric")) {
        term.mixing["sigma"] = Mixing::Geometric;
        term.mixing["epsilon"] = Mixing::Geometric;
      } else {
        return fail(node, "expected 'lorentz-berthelot', 'geometric', or a "
                          "table with a rule for each parameter, got '" +
                              rule + "'");
      }
      continue;
    }
    // Any other keyword names a number that the expression uses.
    if (!node.is_number())
      return fail(node, "expected a number for '" + keyword + "'");
    term.constants.push_back({keyword.str(), *node.value<double>()});
  }
  control.pairs.push_back(std::move(term));
  return Error::success();
}

Error Reader::readOverride(const toml::table &table) {
  PairOverride entry;
  if (Error error = readString(table, "pair", entry.term))
    return error;
  const toml::node *types = table.get("types");
  const toml::array *names = types ? types->as_array() : nullptr;
  if (!names || names->size() != 2 || !(*names)[0].is_string() ||
      !(*names)[1].is_string())
    return fail(types ? *types : static_cast<const toml::node &>(table),
                "expected 'types' with the names of two types in "
                "[[energy.nbfix]]");
  entry.first = *(*names)[0].value<std::string>();
  entry.second = *(*names)[1].value<std::string>();
  for (auto &&[key, node] : table) {
    StringRef keyword = toRef(key.str());
    if (keyword == "pair" || keyword == "types")
      continue;
    if (!node.is_number())
      return fail(node, "expected a number for '" + keyword + "'");
    entry.parameters.push_back({keyword.str(), *node.value<double>()});
  }
  if (entry.parameters.empty())
    return fail(table, "expected at least one parameter in [[energy.nbfix]]");
  control.overrides.push_back(std::move(entry));
  return Error::success();
}

Error Reader::readType(const toml::table &table) {
  ParticleType type;
  if (Error error = readString(table, "name", type.name))
    return error;
  if (type.name.empty())
    return fail(table, "expected a 'name' in [[energy.type]]");
  if (Error error = readPositive(table, "mass", type.mass))
    return error;
  if (!table.contains("mass"))
    return fail(table, "expected a 'mass' in [[energy.type]]");
  for (const ParticleType &other : control.types)
    if (other.name == type.name)
      return fail(table, "the type '" + type.name + "' is defined twice");

  for (auto &&[key, node] : table) {
    StringRef keyword = toRef(key.str());
    if (keyword == "name" || keyword == "mass")
      continue;
    if (!node.is_number())
      return fail(node, "expected a number for '" + keyword + "'");
    type.parameters.push_back({keyword.str(), *node.value<double>()});
  }
  control.types.push_back(std::move(type));
  return Error::success();
}

Error Reader::readEnergy(const toml::table &table) {
  if (Error error = checkKeywords(
          table, "energy",
          {"switchdist", "cutoffdist", "pairlistdist", "vdw_force_switch",
           "vdw_shift", "pair", "type", "nbfix", "dispersion_corr",
           "electrostatic", "pme_alpha", "pme_alpha_tol", "pme_ngrid_x",
           "pme_ngrid_y", "pme_ngrid_z", "pme_max_spacing", "pme_nspline",
           "pme_shift", "pme_influence"},
          {{"forcefield", "M1"},
           {"dielec_const", "M1"}}))
    return error;

  if (Error error = readPositive(table, "cutoffdist", control.cutoffDistance))
    return error;
  // Without a distance to switch from, nothing is switched.
  control.switchDistance = control.cutoffDistance;
  if (Error error = readPositive(table, "switchdist", control.switchDistance))
    return error;
  control.pairlistDistance = control.cutoffDistance + 1.5;
  if (Error error =
          readPositive(table, "pairlistdist", control.pairlistDistance))
    return error;

  if (control.switchDistance > control.cutoffDistance)
    return fail(table, "'switchdist' exceeds 'cutoffdist'");
  if (control.pairlistDistance < control.cutoffDistance)
    return fail(table, "'pairlistdist' is less than 'cutoffdist'");

  bool forceSwitch = false, shift = false;
  if (Error error = readBool(table, "vdw_force_switch", forceSwitch))
    return error;
  if (Error error = readBool(table, "vdw_shift", shift))
    return error;
  if (forceSwitch && shift)
    return fail(table,
                "'vdw_force_switch' and 'vdw_shift' exclude each other");
  bool switches = control.switchDistance < control.cutoffDistance;
  if (shift)
    control.truncation = Truncation::Shift;
  else if (forceSwitch && switches)
    control.truncation = Truncation::ForceSwitch;
  else if (forceSwitch)
    return fail(table, "'vdw_force_switch' needs a 'switchdist' that is "
                       "less than 'cutoffdist'");
  else if (switches)
    control.truncation = Truncation::Switch;
  else
    control.truncation = Truncation::None;

  auto readArray = [&](StringRef key,
                       Error (Reader::*read)(const toml::table &)) -> Error {
    const toml::node *node = table.get(std::string_view(key));
    if (!node)
      return Error::success();
    const toml::array *array = node->as_array();
    if (!array)
      return fail(*node, "expected [[energy." + key + "]]");
    for (const toml::node &element : *array) {
      const toml::table *entry = element.as_table();
      if (!entry)
        return fail(element, "expected [[energy." + key + "]]");
      if (Error error = (this->*read)(*entry))
        return error;
    }
    return Error::success();
  };
  if (Error error = readArray("type", &Reader::readType))
    return error;
  if (Error error = readArray("pair", &Reader::readPair))
    return error;
  if (Error error = readArray("nbfix", &Reader::readOverride))
    return error;

  // With a topology, the correction for the dispersion is for the whole
  // run, and the electrostatics are a cutoff or particle mesh Ewald.
  bool hasTopology = control.hasTopology();
  for (StringRef key : {"dispersion_corr", "electrostatic", "pme_alpha",
                        "pme_alpha_tol", "pme_ngrid_x", "pme_ngrid_y",
                        "pme_ngrid_z", "pme_max_spacing", "pme_nspline",
                        "pme_shift", "pme_influence"})
    if (!hasTopology && table.contains(std::string_view(key)))
      return fail(*table.get(std::string_view(key)),
                  "'" + key + "' in [energy] is for a run from a topology; "
                  "without one, give the terms in [[energy.pair]]");
  if (Error error = readChoice<DispersionCorrection>(
          table, "dispersion_corr", control.topologyDispersion,
          {{"NONE", DispersionCorrection::None},
           {"EPRESS", DispersionCorrection::EnergyPressure}}))
    return error;
  int electrostatic = 0;
  if (Error error = readChoice<int>(table, "electrostatic", electrostatic,
                                    {{"CUTOFF", 0}, {"PME", 1}}))
    return error;
  control.pme = electrostatic == 1;
  for (StringRef key : {"pme_alpha", "pme_alpha_tol", "pme_ngrid_x",
                        "pme_ngrid_y", "pme_ngrid_z", "pme_max_spacing",
                        "pme_nspline", "pme_shift", "pme_influence"})
    if (!control.pme && table.contains(std::string_view(key)))
      return fail(*table.get(std::string_view(key)),
                  "'" + key + "' is for 'electrostatic = \"PME\"'");
  if (Error error = readPositive(table, "pme_alpha", control.pmeAlpha))
    return error;
  if (Error error =
          readPositive(table, "pme_alpha_tol", control.pmeAlphaTolerance))
    return error;
  if (!(control.pmeAlphaTolerance < 1.0))
    return fail(*table.get("pme_alpha_tol"),
                "expected a tolerance less than 1 for 'pme_alpha_tol'");
  static const char *const gridKeys[] = {"pme_ngrid_x", "pme_ngrid_y",
                                         "pme_ngrid_z"};
  for (int k = 0; k != 3; ++k)
    if (Error error = readCount(table, gridKeys[k], control.pmeGrid[k], 8))
      return error;
  if (Error error =
          readPositive(table, "pme_max_spacing", control.pmeMaxSpacing))
    return error;
  if (Error error = readCount(table, "pme_nspline", control.pmeOrder, 4))
    return error;
  if (control.pmeOrder != 4 && control.pmeOrder != 6 &&
      control.pmeOrder != 8)
    return fail(*table.get("pme_nspline"),
                "expected 4, 6, or 8 for 'pme_nspline'");
  if (Error error = readBool(table, "pme_shift", control.pmeShift))
    return error;
  if (Error error = readChoice<bool>(table, "pme_influence",
                                     control.pmeOptimal,
                                     {{"OPTIMAL", true}, {"SPME", false}}))
    return error;
  if (hasTopology && control.truncation != Truncation::None)
    return fail(table, "a run from a topology takes a plain cutoff: "
                       "'switchdist' equal to 'cutoffdist', and no "
                       "'vdw_shift' or 'vdw_force_switch'");

  // A topology gives the types and the terms.
  if (control.hasTopology()) {
    if (!control.types.empty() || !control.pairs.empty() ||
        !control.overrides.empty())
      return fail(table, "[[energy.type]], [[energy.pair]], and "
                         "[[energy.nbfix]] are for a system without a "
                         "topology; the topology gives them");
    return Error::success();
  }
  if (control.types.empty())
    return fail(table, "expected at least one [[energy.type]]");
  if (control.pairs.empty())
    return fail(table, "expected at least one [[energy.pair]]");
  return Error::success();
}

Error Reader::readDynamics(const toml::table &table) {
  if (Error error = checkKeywords(
          table, "dynamics",
          {"integrator", "timestep", "nsteps", "eneout_period",
           "crdout_period", "rstout_period", "nbupdate_period", "iseed",
           "comm_period", "thermostat_period"},
          {{"velout_period", "M1"},
           {"stoptr_period", "M1"},
           {"elec_long_period", "M2"},
           {"barostat_period", "M1"},
           {"annealing", "M1"}}))
    return error;

  if (Error error = readChoice<Integrator>(
          table, "integrator", control.integrator,
          {{"VVER", Integrator::VelocityVerlet},
           {"LEAP", Integrator::Leapfrog}}))
    return error;
  if (Error error = readPositive(table, "timestep", control.timestep))
    return error;
  if (Error error = readCount(table, "nsteps", control.numSteps, 0))
    return error;
  if (Error error = readCount(table, "eneout_period", control.energyPeriod, 0))
    return error;
  if (Error error = readCount(table, "crdout_period", control.framePeriod, 0))
    return error;
  if (Error error =
          readCount(table, "rstout_period", control.checkpointPeriod, 0))
    return error;
  if (Error error =
          readCount(table, "nbupdate_period", control.rebuildPeriod, 0))
    return error;
  // Unset until [ensemble] is read; -1 stands for a period not given.
  control.comPeriod = control.thermostatPeriod = -1;
  if (Error error = readCount(table, "comm_period", control.comPeriod, 0))
    return error;
  if (Error error =
          readCount(table, "thermostat_period", control.thermostatPeriod, 0))
    return error;
  int64_t seed = static_cast<int64_t>(control.seed);
  if (Error error = readCount(table, "iseed", seed, 0))
    return error;
  control.seed = static_cast<uint64_t>(seed);

  if (control.rebuildPeriod != 0)
    return fail(*table.get("nbupdate_period"),
                "'nbupdate_period' is not supported yet; without it, a "
                "neighbor structure is rebuilt when it is no longer valid");

  // Each period is a multiple of the one inside it.
  auto checkMultiple = [&](StringRef outer, int64_t large, StringRef inner,
                           int64_t small) -> Error {
    if (large == 0 || small == 0 || large % small == 0)
      return Error::success();
    return fail(table, "'" + outer + "' is not a multiple of '" + inner +
                           "'");
  };
  if (Error error = checkMultiple("crdout_period", control.framePeriod,
                                  "eneout_period", control.energyPeriod))
    return error;
  if (Error error = checkMultiple("rstout_period", control.checkpointPeriod,
                                  "crdout_period", control.framePeriod))
    return error;
  if (Error error = checkMultiple("rstout_period", control.checkpointPeriod,
                                  "eneout_period", control.energyPeriod))
    return error;
  for (auto [name, period] :
       {std::pair<StringRef, int64_t>{"eneout_period", control.energyPeriod},
        {"crdout_period", control.framePeriod},
        {"rstout_period", control.checkpointPeriod}})
    if (Error error = checkMultiple("nsteps", control.numSteps, name, period))
      return error;
  return Error::success();
}

Error Reader::resolveCoupling() {
  int64_t &com = control.comPeriod, &thermostat = control.thermostatPeriod;
  if (!control.thermostat && thermostat > 0)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "%s: 'thermostat_period' is given, but there is no thermostat",
        path.str().c_str());
  // The thermostat acts every 10 steps and the motion of the center of mass
  // is removed with it, unless one period is given: then both take it.
  // Without a thermostat the motion is removed only if 'comm_period' asks.
  if (control.thermostat) {
    if (thermostat < 0)
      thermostat = com > 0 ? com : 10;
    if (thermostat == 0)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "%s: 'thermostat_period' is 0",
                                     path.str().c_str());
    if (com < 0)
      com = thermostat;
    if (com != 0 && com != thermostat)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "%s: 'comm_period' differs from 'thermostat_period'; in M1 the "
          "motion of the center of mass is removed when the thermostat acts, "
          "or never",
          path.str().c_str());
  } else {
    thermostat = 0;
    if (com < 0)
      com = 0;
  }

  // Coupling acts at the end of the step that completes a period, so the
  // periods of output and the number of steps are multiples of it.
  int64_t period = control.getCouplingPeriod();
  if (period == 0)
    return Error::success();
  for (auto [name, value] :
       {std::pair<StringRef, int64_t>{"nsteps", control.numSteps},
        {"eneout_period", control.energyPeriod},
        {"crdout_period", control.framePeriod},
        {"rstout_period", control.checkpointPeriod}})
    if (value % period != 0)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "%s: '%s' is not a multiple of the period of coupling, %lld steps "
          "('comm_period' or 'thermostat_period')",
          path.str().c_str(), name.str().c_str(), (long long)period);
  return Error::success();
}

Error Reader::readEnsemble(const toml::table &table) {
  if (Error error = checkKeywords(table, "ensemble",
                                  {"ensemble", "temperature", "thermostat",
                                   "tau_t"},
                                  {{"pressure", "M1"},
                                   {"tau_p", "M1"},
                                   {"gamma_t", "M1"},
                                   {"isotropy", "M1"}}))
    return error;
  int ensemble = 0;
  if (Error error = readChoice<int>(table, "ensemble", ensemble,
                                    {{"NVE", 0}, {"NVT", 1}, {"NPT", 2}}))
    return error;
  if (ensemble == 2)
    return fail(*table.get("ensemble"),
                "'ensemble = \"NPT\"' is not supported yet; it is planned "
                "for M1");
  int thermostat = 0;
  if (Error error = readChoice<int>(table, "thermostat", thermostat,
                                    {{"NO", 0}, {"BUSSI", 1}}))
    return error;
  if (ensemble == 1 && thermostat != 1)
    return fail(table, "'ensemble = \"NVT\"' needs 'thermostat = \"BUSSI\"', "
                       "the thermostat of M1");
  if (ensemble == 0 && thermostat != 0)
    return fail(table, "a thermostat needs 'ensemble = \"NVT\"'");
  control.thermostat = thermostat == 1;
  if (Error error = readPositive(table, "tau_t", control.tauT))
    return error;
  if (Error error = readReal(table, "temperature", control.temperature))
    return error;
  if (control.temperature < 0.0)
    return fail(*table.get("temperature"),
                "expected a temperature that is not negative");
  return Error::success();
}

Error Reader::readBoundary(const toml::table &table) {
  if (Error error = checkKeywords(
          table, "boundary",
          {"type", "box_size_x", "box_size_y", "box_size_z"},
          {{"domain_x", "M2"}, {"domain_y", "M2"}, {"domain_z", "M2"}}))
    return error;
  int type = 0;
  if (Error error = readChoice<int>(table, "type", type, {{"PBC", 0}}))
    return error;
  // With a topology, the box is that of the file of coordinates.
  const char *keys[3] = {"box_size_x", "box_size_y", "box_size_z"};
  for (int i = 0; i != 3; ++i) {
    if (control.hasTopology()) {
      if (table.contains(keys[i]))
        return fail(table, llvm::Twine("'") + keys[i] +
                               "' is not needed: the box comes from the "
                               "file of coordinates");
      continue;
    }
    if (Error error = readPositive(table, keys[i], control.box[i]))
      return error;
    if (!table.contains(keys[i]))
      return fail(table, llvm::Twine("expected '") + keys[i] +
                             "' in [boundary]");
  }
  return Error::success();
}

Error Reader::readExecution(const toml::table &table) {
  if (Error error = checkKeywords(table, "execution",
                                  {"target", "threads", "precision",
                                   "neighbor_width", "fast_math",
                                   "reorder"}))
    return error;
  if (Error error = readChoice<Target>(
          table, "target", control.target,
          {{"cpu", Target::CPU}, {"gpu", Target::GPU}}))
    return error;
  if (Error error = readCount(table, "threads", control.threads, 1))
    return error;
  if (Error error = readChoice<Precision>(
          table, "precision", control.precision,
          {{"single", Precision::Single},
           {"mixed", Precision::Mixed},
           {"double", Precision::Double}}))
    return error;
  if (Error error =
          readCount(table, "neighbor_width", control.neighborWidth, 1))
    return error;
  if (Error error = readBool(table, "fast_math", control.fastMath))
    return error;
  return readBool(table, "reorder", control.reorder);
}

Error Reader::read(const toml::table &root) {
  if (Error error = checkKeywords(
          root, "the control file",
          {"input", "output", "energy", "dynamics", "ensemble", "boundary",
           "execution", "constraints"},
          {{"selection", "M1"},
           {"restraints", "M1"},
           {"minimize", "M1"},
           {"remd", "M3"}}))
    return error;

  auto getTable = [&](StringRef name, bool required,
                      const toml::table *&table) -> Error {
    const toml::node *node = root.get(std::string_view(name));
    table = node ? node->as_table() : nullptr;
    if (node && !table)
      return fail(*node, "expected a table [" + name + "]");
    if (!node && required)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "%s: expected a table [%s]",
                                     path.str().c_str(),
                                     name.str().c_str());
    return Error::success();
  };

  const toml::table *table;
  if (Error error = getTable("input", /*required=*/true, table))
    return error;
  if (Error error = checkKeywords(
          *table, "input",
          {"pdbfile", "rstfile", "prmtopfile", "ambcrdfile", "grotopfile",
           "grocrdfile", "groinclude", "grodefine"},
          {{"psffile", "M2"}, {"topfile", "M2"}, {"parfile", "M2"}}))
    return error;
  if (Error error = readPath(*table, "grotopfile", control.gromacsTopologyFile))
    return error;
  if (Error error =
          readPath(*table, "grocrdfile", control.gromacsCoordinateFile))
    return error;
  for (StringRef key : {"groinclude", "grodefine"}) {
    const toml::node *node = table->get(std::string_view(key));
    if (!node)
      continue;
    const toml::array *array = node->as_array();
    if (!array)
      return fail(*node, "expected a list of strings for '" + key + "'");
    for (const toml::node &element : *array) {
      if (!element.is_string())
        return fail(element, "expected a list of strings for '" + key + "'");
      std::string value = *element.value<std::string>();
      if (key == "groinclude") {
        // Relative to the control file, as the other paths are.
        llvm::SmallString<256> full(llvm::sys::path::parent_path(path));
        if (llvm::sys::path::is_absolute(value))
          full = value;
        else
          llvm::sys::path::append(full, value);
        control.gromacsIncludes.push_back(std::string(full));
      } else {
        control.gromacsDefines.push_back(value);
      }
    }
  }
  if (control.gromacsTopologyFile.empty() !=
      control.gromacsCoordinateFile.empty())
    return fail(*table, "expected 'grotopfile' and 'grocrdfile' together");
  if (Error error = readPath(*table, "pdbfile", control.pdbFile))
    return error;
  if (Error error = readPath(*table, "rstfile", control.restartInput))
    return error;
  if (Error error = readPath(*table, "prmtopfile", control.prmtopFile))
    return error;
  if (Error error =
          readPath(*table, "ambcrdfile", control.amberCoordinateFile))
    return error;
  if (control.prmtopFile.empty() != control.amberCoordinateFile.empty())
    return fail(*table, "expected 'prmtopfile' and 'ambcrdfile' together");
  int sources = !control.pdbFile.empty() + !control.prmtopFile.empty() +
                !control.gromacsTopologyFile.empty();
  if (sources > 1)
    return fail(*table, "expected one of 'pdbfile', 'prmtopfile', and "
                        "'grotopfile'");
  if (sources == 0)
    return fail(*table, "expected a 'pdbfile', a 'prmtopfile', or a "
                        "'grotopfile' in [input]");

  if (Error error = getTable("output", /*required=*/false, table))
    return error;
  if (table) {
    if (Error error = checkKeywords(*table, "output", {"dcdfile", "rstfile"},
                                    {{"xtcfile", "M0"},
                                     {"dcdvelfile", "M1"}}))
      return error;
    if (Error error = readPath(*table, "dcdfile", control.dcdFile))
      return error;
    if (Error error = readPath(*table, "rstfile", control.restartOutput))
      return error;
  }

  if (Error error = getTable("energy", /*required=*/true, table))
    return error;
  if (Error error = readEnergy(*table))
    return error;

  if (Error error = getTable("dynamics", /*required=*/true, table))
    return error;
  if (Error error = readDynamics(*table))
    return error;

  if (Error error = getTable("ensemble", /*required=*/false, table))
    return error;
  if (table)
    if (Error error = readEnsemble(*table))
      return error;

  if (Error error = resolveCoupling())
    return error;

  if (Error error = getTable("constraints", /*required=*/false, table))
    return error;
  if (table) {
    if (Error error = checkKeywords(*table, "constraints",
                                    {"rigid_bond", "fast_water",
                                     "settle_residues"},
                                    {{"shake_tolerance", "M1"},
                                     {"shake_iterations", "M1"}}))
      return error;
    if (Error error = readBool(*table, "rigid_bond", control.rigidBonds))
      return error;
    if (Error error = readBool(*table, "fast_water", control.fastWater))
      return error;
    for (StringRef key : {"fast_water", "rigid_bond"}) {
      bool on = key == "fast_water" ? control.fastWater : control.rigidBonds;
      if (on && control.integrator != Integrator::VelocityVerlet)
        return fail(*table->get(std::string_view(key)),
                    "constraints need 'integrator = \"VVER\"'; with "
                    "leapfrog they are planned for M1");
    }
    if (const toml::node *node = table->get("settle_residues")) {
      const toml::array *array = node->as_array();
      if (!array)
        return fail(*node, "expected a list of residue names for "
                           "'settle_residues'");
      control.settleResidues.clear();
      for (const toml::node &element : *array) {
        if (!element.is_string())
          return fail(element, "expected a list of residue names for "
                               "'settle_residues'");
        control.settleResidues.push_back(*element.value<std::string>());
      }
    }
    control.statesFlexible =
        table->contains("fast_water") && !control.fastWater;
  }

  if (Error error = getTable("boundary", /*required=*/true, table))
    return error;
  if (Error error = readBoundary(*table))
    return error;

  if (Error error = getTable("execution", /*required=*/false, table))
    return error;
  if (table)
    if (Error error = readExecution(*table))
      return error;

  if (control.checkpointPeriod != 0 && control.restartOutput.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "%s: 'rstout_period' is given, but [output] names no 'rstfile'",
        path.str().c_str());
  if (control.checkpointPeriod == 0 && !control.restartOutput.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "%s: [output] names an 'rstfile', but 'rstout_period' is not given",
        path.str().c_str());
  if (control.framePeriod != 0 && control.dcdFile.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "%s: 'crdout_period' is given, but [output] names no 'dcdfile'",
        path.str().c_str());
  return Error::success();
}

llvm::Expected<Control> mdir::driver::readControl(StringRef path) {
  toml::parse_result result = toml::parse_file(std::string_view(path));
  if (!result) {
    const toml::parse_error &error = result.error();
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(), "%s:%u: %s", path.str().c_str(),
        static_cast<unsigned>(error.source().begin.line),
        std::string(error.description()).c_str());
  }

  Control control;
  Reader reader(path, control);
  if (Error error = reader.read(result.table()))
    return std::move(error);
  return control;
}

std::string mdir::driver::getControlTemplate() {
  return R"TOML([input]
pdbfile = "system.pdb"          # positions; the name of an atom is its type
# rstfile = "earlier.h5"        # the state that the run continues from

[output]
dcdfile = "run.dcd"             # trajectory of positions
# rstfile = "run.h5"            # checkpoint, with rstout_period

[energy]
cutoffdist       = 12.0         # cutoff (Å)
switchdist       = 10.0         # where switching begins (Å); the cutoff: none
pairlistdist     = 13.5         # reach of the neighbor structures (Å)
vdw_force_switch = false        # switch the force instead of the energy
vdw_shift        = false        # shift the energy to zero at the cutoff

[[energy.pair]]
name       = "lj"
expression = "4*epsilon*((sigma/r)^12 - (sigma/r)^6)"
mixing     = "lorentz-berthelot"

[[energy.type]]
name    = "AR"
mass    = 39.95                 # amu
epsilon = 0.2385                # kcal/mol
sigma   = 3.4                   # Å

[dynamics]
integrator    = "VVER"          # VVER, LEAP
timestep      = 0.001           # ps
nsteps        = 100
eneout_period = 10              # steps between energies in the log; 0: none
crdout_period = 0               # steps between frames; 0: none
rstout_period = 0               # steps between checkpoints; 0: none
iseed         = 314159          # seed of the velocities and the thermostat
# comm_period = 0               # steps between removals of the motion of
#                               # the center of mass; 0: none
# thermostat_period = 10        # steps between actions of the thermostat

[ensemble]
ensemble    = "NVE"             # NVE, NVT
temperature = 298.15            # of the velocities and the bath (K)
# thermostat = "BUSSI"          # with NVT: stochastic velocity rescaling
# tau_t      = 1.0              # time of the thermostat (ps)

[boundary]
type       = "PBC"              # PBC
box_size_x = 40.0               # Å
box_size_y = 40.0
box_size_z = 40.0

[execution]
target    = "cpu"               # cpu, gpu
threads   = 1                   # for the target cpu
precision = "double"            # single, mixed, double
fast_math = true                # allow rewrites that change rounding
reorder   = true                # keep the particles in the order of their positions
# neighbor_width = 160          # neighbors per particle; default: estimated
)TOML";
}
