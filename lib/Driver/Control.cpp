// The control file of a run.

#include "mdir/Driver/Control.h"
#include "mdir/Driver/Expression.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Path.h"

#include <algorithm>
#include <cmath>
#include <optional>

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
  /// [[energy.function]]: a tabulated function (D138).
  Error readFunction(const toml::table &table);
  /// [[energy.bond]], [[energy.angle]], or [[energy.dihedral]]: a term over
  /// tuples of `arity` particles (D136).
  Error readTupleTerm(const toml::table &table, unsigned arity);
  Error readBond(const toml::table &table) { return readTupleTerm(table, 2); }
  Error readAngle(const toml::table &table) { return readTupleTerm(table, 3); }
  Error readDihedral(const toml::table &table) {
    return readTupleTerm(table, 4);
  }
  Error readOverride(const toml::table &table);
  Error readType(const toml::table &table);
  Error readDynamics(const toml::table &table);
  Error readMinimize(const toml::table &table);
  Error readEnsemble(const toml::table &table);
  Error readInput(const toml::table &table);
  Error checkIntervals(const toml::table &table);
  Error readPME(const toml::table &table);
  Error readOutput(const toml::table &table);
  Error readThermostat(const toml::table &table);
  Error readBarostat(const toml::table &table);

  /// Sets the periods of the removal of the motion of the center of mass
  /// and of the thermostat that were not given, and checks them.
  Error resolveCoupling();
  Error readBoundary(const toml::table &table);
  Error readExecution(const toml::table &table);

  StringRef path;
  Control &control;
  /// NVE, NVT, or NPT, from [ensemble]: 0, 1, or 2.
  int ensembleKind = 0;
  /// [output], which the intervals of output are checked against.
  const toml::table *outputTable = nullptr;
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
    // Offer only a close, unambiguous match. The same list that validates
    // the table supplies both the suggestion and its supported keywords.
    StringRef nearest;
    unsigned bestDistance = 3;
    bool tied = false;
    std::string names;
    for (StringRef candidate : known) {
      if (!names.empty())
        names += ", ";
      names += candidate.str();
      unsigned distance = keyword.edit_distance(
          candidate, /*AllowReplacements=*/true, /*MaxEditDistance=*/2);
      if (distance < bestDistance) {
        nearest = candidate;
        bestDistance = distance;
        tied = false;
      } else if (distance == bestDistance) {
        tied = true;
      }
    }
    std::string message =
        ("unknown keyword '" + keyword + "' in [" + name + "]").str();
    if (!nearest.empty() && !tied)
      message += ("; did you mean '" + nearest + "'?").str();
    message += "\nvalid keywords: " + names;
    message += "\ntry 'mdir template md' or 'mdir template amber' for a "
               "control file with the supported keywords";
    return fail(node, message);
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

Error Reader::readTupleTerm(const toml::table &table, unsigned arity) {
  static const char *const kinds[] = {"", "", "bond", "angle", "dihedral"};
  std::string kind = std::string("[[energy.") + kinds[arity] + "]]";
  TupleTerm term;
  term.arity = arity;
  if (Error error = readString(table, "name", term.name))
    return error;
  if (Error error = readString(table, "expression", term.expression))
    return error;
  if (term.name.empty())
    return fail(table, "expected a 'name' in " + kind);
  if (!llvm::all_of(term.name,
                    [](char c) { return llvm::isAlnum(c) || c == '_'; }))
    return fail(*table.get("name"), "a name of letters, digits, and '_', "
                                    "not '" + term.name + "'");
  for (const TupleTerm &other : control.tupleTerms)
    if (other.name == term.name)
      return fail(*table.get("name"),
                  "another term has the name '" + term.name + "'");
  if (term.expression.empty())
    return fail(table, "expected an 'expression' in " + kind);

  // The centers of groups (D139): masks of Amber, one for each member of
  // the one tuple.
  if (const toml::node *node = table.get("groups")) {
    if (table.contains("particles"))
      return fail(*node, "a term takes 'particles' or 'groups', not both");
    const toml::array *masks = node->as_array();
    if (!masks || masks->size() != arity ||
        !llvm::all_of(*masks, [](const toml::node &e) { return e.is_string(); }))
      return fail(*node, "expected 'groups' as " + llvm::Twine(arity) +
                             " masks of particles");
    for (const toml::node &mask : *masks)
      term.groups.push_back(*mask.value<std::string>());
    std::string weighting = "MASS";
    if (Error error = readString(table, "weighting", weighting))
      return error;
    if (weighting != "MASS" && weighting != "NONE")
      return fail(*table.get("weighting"),
                  "expected 'weighting' as \"MASS\" or \"NONE\"");
    term.massWeighted = weighting == "MASS";
  } else if (table.contains("weighting")) {
    return fail(*table.get("weighting"),
                "'weighting' is for the centers of 'groups'");
  }

  // The tuples: lists of the numbers of their particles, from 1.
  const toml::node *node = table.get("particles");
  const toml::array *tuples = node ? node->as_array() : nullptr;
  if (!term.isCentroid() && (!tuples || tuples->empty()))
    return fail(node ? *node : static_cast<const toml::node &>(table),
                "expected 'particles' in " + kind + ": lists of " +
                    llvm::Twine(arity) + " particle numbers, from 1, or "
                    "'groups'");
  const toml::array none;
  for (const toml::node &element : term.isCentroid() ? none : *tuples) {
    const toml::array *tuple = element.as_array();
    if (!tuple || tuple->size() != arity)
      return fail(element, "expected a list of " + llvm::Twine(arity) +
                               " particle numbers");
    std::vector<unsigned> members;
    for (const toml::node &member : *tuple) {
      std::optional<int64_t> number = member.value<int64_t>();
      if (!member.is_integer() || !number || *number < 1)
        return fail(member, "expected a particle number, from 1");
      if (llvm::is_contained(members, static_cast<unsigned>(*number - 1)))
        return fail(member, "a particle appears twice in a tuple");
      members.push_back(static_cast<unsigned>(*number - 1));
    }
    term.particles.insert(term.particles.end(), members.begin(),
                          members.end());
  }
  size_t count = term.size();

  // Every other keyword is a parameter: a number for all tuples, or a list
  // of one for each.
  for (auto &&[key, value] : table) {
    StringRef keyword = toRef(key.str());
    if (keyword == "name" || keyword == "expression" ||
        keyword == "particles" || keyword == "groups" ||
        keyword == "weighting")
      continue;
    std::vector<double> values;
    if (std::optional<double> number = value.value<double>()) {
      values.assign(count, *number);
    } else if (const toml::array *list = value.as_array()) {
      for (const toml::node &element : *list) {
        std::optional<double> number = element.value<double>();
        if (!number)
          return fail(element, "expected a number");
        values.push_back(*number);
      }
      if (values.size() != count)
        return fail(value, "'" + keyword + "' has " +
                               llvm::Twine(values.size()) +
                               " values, and there are " +
                               llvm::Twine(count) + " tuples");
    } else {
      return fail(value, "expected a number, or a list of one for each "
                         "tuple, for the parameter '" +
                             keyword + "'");
    }
    term.parameters.push_back({keyword.str(), std::move(values)});
  }

  // The expression uses the coordinate and the parameters only.
  auto expression = Expression::parse(term.expression, control.functions);
  if (!expression)
    return fail(*table.get("expression"),
                llvm::toString(expression.takeError()));
  // The energy of centers is computed outside the loops over tuples, where
  // the tables of the functions are not read.
  if (term.isCentroid() && expression->callsTabulated())
    return fail(*table.get("expression"),
                "a term over the centers of groups cannot call a tabulated "
                "function yet");
  // Between the centers of two groups, the components of the vector from
  // the first to the second as well.
  bool components = term.isCentroid() && arity == 2;
  // The time `t` in ps, as a reference that moves (D145).
  if (llvm::any_of(term.parameters,
                   [](const auto &p) { return p.first == "t"; }))
    return fail(*table.get("t"), "'t' is the time, not a parameter");
  for (const std::string &name : expression->getNames()) {
    if (name == "t") {
      control.usesTime = true;
      continue;
    }
    if (name != term.getVariable() &&
        !(components && (name == "dx" || name == "dy" || name == "dz")) &&
        !llvm::any_of(term.parameters,
                      [&](const auto &p) { return p.first == name; }))
      return fail(*table.get("expression"),
                  "the expression uses '" + name + "', which is neither '" +
                      term.getVariable() + "'" +
                      (components ? ", 'dx', 'dy', 'dz'" : "") +
                      ", the time 't', nor a parameter of the term");
  }
  control.tupleTerms.push_back(std::move(term));
  return Error::success();
}

Error Reader::readFunction(const toml::table &table) {
  if (Error error = checkKeywords(table, "[energy.function]",
                                  {"name", "values", "min", "max",
                                   "periodic"}))
    return error;
  TabulatedFunction function;
  if (Error error = readString(table, "name", function.name))
    return error;
  if (function.name.empty() ||
      !(llvm::isAlpha(function.name.front()) || function.name.front() == '_') ||
      !llvm::all_of(function.name,
                    [](char c) { return llvm::isAlnum(c) || c == '_'; }))
    return fail(table, "expected a 'name' of letters, digits, and '_' in "
                       "[[energy.function]]");
  if (Expression::isFunction(function.name) ||
      llvm::any_of(control.functions, [&](const TabulatedFunction &other) {
        return other.name == function.name;
      }))
    return fail(*table.get("name"), "the function '" + function.name +
                                        "' exists already");
  const toml::node *values = table.get("values");
  const toml::array *list = values ? values->as_array() : nullptr;
  if (!list || list->size() < 2)
    return fail(values ? *values : static_cast<const toml::node &>(table),
                "expected 'values', a list of at least two numbers");
  for (const toml::node &value : *list) {
    std::optional<double> number = value.value<double>();
    if (!number || !std::isfinite(*number))
      return fail(value, "expected a number in 'values'");
    function.values.push_back(*number);
  }
  for (StringRef key : {"min", "max"})
    if (!table.contains(std::string_view(key)))
      return fail(table, "expected '" + key + "' in [[energy.function]]");
  if (Error error = readReal(table, "min", function.min))
    return error;
  if (Error error = readReal(table, "max", function.max))
    return error;
  if (!(function.min < function.max))
    return fail(*table.get("max"), "'max' is not greater than 'min'");
  if (Error error = readBool(table, "periodic", function.periodic))
    return error;
  if (function.periodic && (function.values.size() < 4 ||
                            function.values.front() != function.values.back()))
    return fail(*values, "a periodic function needs at least four values, "
                         "and the last equal to the first");
  control.functions.push_back(std::move(function));
  return Error::success();
}

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
    if (keyword == "dispersion_correction") {
      if (Error error = readChoice<DispersionCorrection>(
              table, "dispersion_correction", term.dispersion,
              {{"NONE", DispersionCorrection::None},
               {"ENERGY_PRESSURE", DispersionCorrection::EnergyPressure}}))
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
    if (keyword == "groups") {
      const toml::array *list = node.as_array();
      if (!list || list->size() != 2 ||
          !llvm::all_of(*list, [](const toml::node &e) { return e.is_string(); }))
        return fail(node, "expected 'groups' as two masks of particles");
      for (const toml::node &mask : *list)
        term.groups.push_back(*mask.value<std::string>());
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
                "[[energy.pair_override]]");
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
    return fail(table,
                "expected at least one parameter in [[energy.pair_override]]");
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

Error Reader::readInput(const toml::table &table) {
  if (Error error = checkKeywords(
          table, "input",
          {"topology", "coordinates", "format", "checkpoint", "include_paths",
           "defines", "parameters"},
          {}))
    return error;
  std::string topology, coordinates;
  if (Error error = readPath(table, "topology", topology))
    return error;
  if (Error error = readPath(table, "coordinates", coordinates))
    return error;
  if (Error error = readPath(table, "checkpoint", control.restartInput))
    return error;
  if (coordinates.empty())
    return fail(table, "expected 'coordinates' in [input]");

  // The format of the files: given, or, with "AUTO", the default, from
  // their extensions.
  enum class Format { Unknown, Amber, Gromacs, Charmm, PDB };
  Format format = Format::Unknown;
  if (Error error = readChoice<Format>(table, "format", format,
                                       {{"AUTO", Format::Unknown},
                                        {"AMBER", Format::Amber},
                                        {"GROMACS", Format::Gromacs},
                                        {"CHARMM", Format::Charmm},
                                        {"PDB", Format::PDB}}))
    return error;
  if (format == Format::Unknown) {
    StringRef extension =
        llvm::sys::path::extension(topology.empty() ? coordinates : topology);
    if (extension.equals_insensitive(".prmtop") ||
        extension.equals_insensitive(".parm7"))
      format = Format::Amber;
    else if (extension.equals_insensitive(".top"))
      format = Format::Gromacs;
    else if (extension.equals_insensitive(".psf"))
      format = Format::Charmm;
    else if (topology.empty() && extension.equals_insensitive(".pdb"))
      format = Format::PDB;
    else
      return fail(table, "cannot tell the format from the name '" +
                             (topology.empty() ? coordinates : topology) +
                             "'; give 'format' in [input]: \"AMBER\", "
                             "\"GROMACS\", \"CHARMM\", or \"PDB\"");
  }
  if (format == Format::PDB && !topology.empty())
    return fail(table, "a run from a PDB file takes no 'topology': the "
                       "terms are in [energy]");
  if (format != Format::PDB && topology.empty())
    return fail(table, "expected 'topology' in [input]");
  switch (format) {
  case Format::Amber:
    control.prmtopFile = topology;
    control.amberCoordinateFile = coordinates;
    break;
  case Format::Gromacs:
    control.gromacsTopologyFile = topology;
    control.gromacsCoordinateFile = coordinates;
    break;
  case Format::Charmm:
    control.charmmStructureFile = topology;
    control.charmmCoordinateFile = coordinates;
    break;
  case Format::PDB:
  case Format::Unknown:
    control.pdbFile = coordinates;
    break;
  }

  // The files of the parameters of a PSF, relative to the control file.
  if (const toml::node *node = table.get("parameters")) {
    if (format != Format::Charmm)
      return fail(*node, "'parameters' is for a PSF of CHARMM");
    const toml::array *array = node->as_array();
    if (!array || array->empty())
      return fail(*node, "expected a list of the files of topology, "
                         "parameters, and streams for 'parameters'");
    for (const toml::node &element : *array) {
      if (!element.is_string())
        return fail(element, "expected a list of file names for "
                             "'parameters'");
      std::string value = *element.value<std::string>();
      llvm::SmallString<256> full(llvm::sys::path::parent_path(path));
      if (llvm::sys::path::is_absolute(value))
        full = value;
      else
        llvm::sys::path::append(full, value);
      control.charmmParameterFiles.push_back(std::string(full));
    }
  } else if (format == Format::Charmm) {
    return fail(table, "a PSF needs 'parameters' in [input]: the files of "
                       "topology, parameters, and streams, in the order "
                       "CHARMM reads them");
  }

  for (StringRef key : {"include_paths", "defines"}) {
    const toml::node *node = table.get(std::string_view(key));
    if (!node)
      continue;
    if (format != Format::Gromacs)
      return fail(*node, "'" + key + "' is for a GROMACS topology");
    const toml::array *array = node->as_array();
    if (!array)
      return fail(*node, "expected a list of strings for '" + key + "'");
    for (const toml::node &element : *array) {
      if (!element.is_string())
        return fail(element, "expected a list of strings for '" + key + "'");
      std::string value = *element.value<std::string>();
      if (key == "include_paths") {
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
  return Error::success();
}

Error Reader::readOutput(const toml::table &table) {
  if (Error error = checkKeywords(
          table, "output",
          {"trajectory", "trajectory_format", "checkpoint", "energy_interval",
           "trajectory_interval", "checkpoint_interval", "pull_coordinates"},
          {}))
    return error;
  if (Error error = readPath(table, "trajectory", control.trajectoryFile))
    return error;
  // The format: from the extension of the name (AUTO, the default), or as
  // the file says (D141).
  enum class Format { Auto, DCD, XTC };
  Format format = Format::Auto;
  if (Error error = readChoice<Format>(
          table, "trajectory_format", format,
          {{"AUTO", Format::Auto}, {"DCD", Format::DCD}, {"XTC", Format::XTC}}))
    return error;
  if (format == Format::Auto && !control.trajectoryFile.empty()) {
    StringRef extension = llvm::sys::path::extension(control.trajectoryFile);
    if (extension.equals_insensitive(".dcd"))
      format = Format::DCD;
    else if (extension.equals_insensitive(".xtc"))
      format = Format::XTC;
    else
      return fail(*table.get("trajectory"),
                  "the format of the trajectory '" + control.trajectoryFile +
                      "' is not known from its extension; name a file that "
                      "ends in '.dcd' or '.xtc', or give 'trajectory_format'");
  }
  control.trajectoryFormat =
      format == Format::XTC ? TrajectoryFormat::XTC : TrajectoryFormat::DCD;
  if (Error error = readPath(table, "checkpoint", control.restartOutput))
    return error;
  if (Error error = readPath(table, "pull_coordinates", control.pullFile))
    return error;
  if (Error error =
          readCount(table, "energy_interval", control.energyPeriod, 0))
    return error;
  if (Error error =
          readCount(table, "trajectory_interval", control.framePeriod, 0))
    return error;
  if (Error error =
          readCount(table, "checkpoint_interval", control.checkpointPeriod, 0))
    return error;
  outputTable = &table;
  return Error::success();
}

Error Reader::readEnergy(const toml::table &table) {
  if (Error error = checkKeywords(
          table, "energy",
          {"cutoff", "switch_distance", "pairlist_distance",
           "pruned_distance", "rebuild_interval", "lennard_jones_modifier", "coulomb_modifier", "reaction_field_dielectric", "implicit_solvent", "solvent_dielectric", "solute_dielectric", "surface_area_energy", "pair", "bond", "angle", "dihedral", "function", "type",
           "pair_override", "dispersion_correction", "electrostatics"},
          {}))
    return error;

  if (Error error = readPositive(table, "cutoff", control.cutoffDistance))
    return error;
  // Without a distance to switch from, nothing is switched.
  control.switchDistance = control.cutoffDistance;
  if (Error error =
          readPositive(table, "switch_distance", control.switchDistance))
    return error;
  control.pairlistDistance = control.cutoffDistance + 1.5;
  if (Error error =
          readPositive(table, "pairlist_distance", control.pairlistDistance))
    return error;

  // Opt-in, not a default: a structure rebuilt at a fixed interval may miss
  // pairs within the cutoff (D88). The run warns.
  if (Error error =
          readCount(table, "rebuild_interval", control.rebuildPeriod, 0))
    return error;

  if (control.switchDistance > control.cutoffDistance)
    return fail(table, "'switch_distance' exceeds 'cutoff'");
  if (control.pairlistDistance < control.cutoffDistance)
    return fail(table, "'pairlist_distance' is less than 'cutoff'");
  // A dual list (D114): the inner list, pruned from the structure of
  // 'pairlist_distance', with the reach 'pruned_distance'.
  if (Error error =
          readPositive(table, "pruned_distance", control.prunedDistance))
    return error;
  if (control.prunedDistance != 0.0 &&
      !(control.prunedDistance > control.cutoffDistance &&
        control.prunedDistance < control.pairlistDistance))
    return fail(table, "'pruned_distance' is not between 'cutoff' and "
                       "'pairlist_distance'");
  if (control.prunedDistance != 0.0 && control.rebuildPeriod > 0)
    return fail(table, "'pruned_distance' and 'rebuild_interval' do not go "
                       "together");

  enum class Modifier { None, PotentialShift, ForceSwitch, PowerForceSwitch };
  Modifier modifier = Modifier::None;
  if (Error error = readChoice<Modifier>(
          table, "lennard_jones_modifier", modifier,
          {{"NONE", Modifier::None},
           {"POTENTIAL_SHIFT", Modifier::PotentialShift},
           {"FORCE_SWITCH", Modifier::ForceSwitch},
           {"POWER_FORCE_SWITCH", Modifier::PowerForceSwitch}}))
    return error;
  bool switches = control.switchDistance < control.cutoffDistance;
  if ((modifier == Modifier::ForceSwitch ||
       modifier == Modifier::PowerForceSwitch) &&
      !switches)
    return fail(table, "a force switch needs a 'switch_distance' that is less "
                       "than 'cutoff'");
  if (modifier == Modifier::PotentialShift)
    control.truncation = Truncation::Shift;
  else if (modifier == Modifier::ForceSwitch)
    control.truncation = Truncation::ForceSwitch;
  else if (modifier == Modifier::PowerForceSwitch)
    control.truncation = Truncation::PowerForceSwitch;
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
  // The functions first, so that the expressions of the terms may call
  // them wherever they stand in the file.
  if (Error error = readArray("function", &Reader::readFunction))
    return error;
  if (Error error = readArray("type", &Reader::readType))
    return error;
  if (Error error = readArray("pair", &Reader::readPair))
    return error;
  if (Error error = readArray("bond", &Reader::readBond))
    return error;
  if (Error error = readArray("angle", &Reader::readAngle))
    return error;
  if (Error error = readArray("dihedral", &Reader::readDihedral))
    return error;
  if (!control.tupleTerms.empty() && !control.hasTopology())
    return fail(*table.get(control.tupleTerms.front().arity == 2   ? "bond"
                           : control.tupleTerms.front().arity == 3 ? "angle"
                                                                   : "dihedral"),
                "terms over tuples need a topology, whose particles they "
                "number");
  if (Error error = readArray("pair_override", &Reader::readOverride))
    return error;

  // With a topology, the correction for the dispersion is for the whole
  // run, and the electrostatics are a cutoff or particle mesh Ewald.
  bool hasTopology = control.hasTopology();
  for (StringRef key :
       {"dispersion_correction", "electrostatics", "coulomb_modifier",
        "reaction_field_dielectric", "implicit_solvent",
        "solvent_dielectric", "solute_dielectric", "surface_area_energy"})
    if (!hasTopology && table.contains(std::string_view(key)))
      return fail(*table.get(std::string_view(key)),
                  "'" + key + "' in [energy] is for a run from a topology; "
                  "without one, give the terms in [[energy.pair]]");
  control.topologyDispersionGiven = table.contains("dispersion_correction");
  if (Error error = readChoice<DispersionCorrection>(
          table, "dispersion_correction", control.topologyDispersion,
          {{"NONE", DispersionCorrection::None},
           {"ENERGY_PRESSURE", DispersionCorrection::EnergyPressure}}))
    return error;
  int electrostatic = 0;
  if (Error error = readChoice<int>(
          table, "electrostatics", electrostatic,
          {{"CUTOFF", 0}, {"PME", 1}, {"REACTION_FIELD", 2}}))
    return error;
  control.pme = electrostatic == 1;
  control.reactionField = electrostatic == 2;
  // The reaction field (D140): the permittivity beyond the cutoff, 1 or
  // more, or 0 for a conductor, as the file must say.
  if (control.reactionField) {
    const toml::node *node = table.get("reaction_field_dielectric");
    if (!node)
      return fail(table, "expected 'reaction_field_dielectric' with "
                         "'electrostatics = \"REACTION_FIELD\"': the "
                         "relative permittivity beyond the cutoff, or 0 for "
                         "a conductor");
    if (Error error = readReal(table, "reaction_field_dielectric",
                               control.reactionFieldDielectric))
      return error;
    if (!(control.reactionFieldDielectric == 0.0 ||
          control.reactionFieldDielectric >= 1.0))
      return fail(*node, "'reaction_field_dielectric' is 1 or more, or 0 "
                         "for a conductor");
  } else if (const toml::node *node = table.get("reaction_field_dielectric")) {
    return fail(*node, "'reaction_field_dielectric' is for "
                       "'electrostatics = \"REACTION_FIELD\"'");
  }
  // Generalized Born (D144), with the Coulomb of a cutoff, as the
  // continuum stands for the solvent.
  if (Error error = readChoice<Control::ImplicitSolvent>(
          table, "implicit_solvent", control.implicitSolvent,
          {{"NONE", Control::ImplicitSolvent::None},
           {"OBC1", Control::ImplicitSolvent::OBC1},
           {"OBC2", Control::ImplicitSolvent::OBC2}}))
    return error;
  bool born = control.implicitSolvent != Control::ImplicitSolvent::None;
  for (StringRef key :
       {"solvent_dielectric", "solute_dielectric", "surface_area_energy"})
    if (!born && table.contains(std::string_view(key)))
      return fail(*table.get(std::string_view(key)),
                  "'" + key + "' is for 'implicit_solvent'");
  if (born) {
    if (control.pme || control.reactionField)
      return fail(table, "generalized Born takes the Coulomb of a plain "
                         "cutoff, 'electrostatics = \"CUTOFF\"'; its "
                         "continuum stands for the solvent");
    if (Error error = readPositive(table, "solvent_dielectric",
                                   control.solventDielectric))
      return error;
    if (Error error = readPositive(table, "solute_dielectric",
                                   control.soluteDielectric))
      return error;
    if (Error error = readReal(table, "surface_area_energy",
                               control.surfaceAreaEnergy))
      return error;
    if (control.surfaceAreaEnergy < 0.0)
      return fail(*table.get("surface_area_energy"),
                  "expected 0 or a positive 'surface_area_energy'");
  }
  if (!control.pme && table.contains("coulomb_modifier"))
    return fail(*table.get("coulomb_modifier"),
                "'coulomb_modifier' is for 'electrostatics = \"PME\"'");
  if (Error error = readChoice<bool>(table, "coulomb_modifier",
                                     control.pmeShift,
                                     {{"NONE", false},
                                      {"POTENTIAL_SHIFT", true}}))
    return error;
  // A topology's Lennard-Jones takes the modifier on its own, not its
  // Coulomb; a switch of the potential is for terms in the control file.
  if (hasTopology && control.truncation == Truncation::Switch)
    return fail(table, "a run from a topology switches its Lennard-Jones "
                       "only with a 'lennard_jones_modifier' "
                       "(\"FORCE_SWITCH\" or \"POWER_FORCE_SWITCH\"); "
                       "without one, 'switch_distance' must equal 'cutoff'");
  if (hasTopology && control.truncation != Truncation::None &&
      control.topologyDispersion != DispersionCorrection::None)
    return fail(table, "the correction for the dispersion needs a plain "
                       "cutoff; with a 'lennard_jones_modifier', give "
                       "'dispersion_correction = \"NONE\"'");
  if (!hasTopology && control.truncation == Truncation::PowerForceSwitch)
    return fail(table, "'lennard_jones_modifier = \"POWER_FORCE_SWITCH\"' "
                       "switches the powers of the Lennard-Jones of a "
                       "topology; for terms in the control file use "
                       "\"FORCE_SWITCH\"");

  // A topology gives the types and the terms. A pair term adds to them,
  // with the parameters of the topology (D137).
  if (control.hasTopology()) {
    if (!control.types.empty() || !control.overrides.empty())
      return fail(table, "[[energy.type]] and [[energy.pair_override]] are "
                         "for a system without a topology; the topology "
                         "gives them");
    const toml::array *pairs = table.get("pair") ? table.get("pair")->as_array()
                                                 : nullptr;
    if (pairs && control.truncation == Truncation::PowerForceSwitch)
      return fail(*pairs, "a pair term is truncated as the whole of its "
                          "energy, which \"POWER_FORCE_SWITCH\" does not "
                          "say; use \"FORCE_SWITCH\"");
    for (auto [index, term] : llvm::enumerate(control.pairs)) {
      const toml::node &node = (*pairs)[index];
      if (!term.mixing.empty() ||
          term.dispersion != DispersionCorrection::None)
        return fail(node, "with a topology a pair term takes the parameters "
                          "of the topology, not 'mixing' or "
                          "'dispersion_correction'");
      if (!llvm::all_of(term.name, [](char c) {
            return llvm::isAlnum(c) || c == '_';
          }))
        return fail(node, "a name of letters, digits, and '_', not '" +
                              term.name + "'");
      auto expression = Expression::parse(term.expression, control.functions);
      if (!expression)
        return fail(node, llvm::toString(expression.takeError()));
      static const char *const known[] = {
          "r", "q1", "q2", "sigma", "epsilon", "sigma1", "sigma2",
          "epsilon1", "epsilon2", "coulomb", "t"};
      for (const std::string &name : expression->getNames())
        if (!llvm::is_contained(known, StringRef(name)) &&
            !llvm::any_of(term.constants,
                          [&](const auto &c) { return c.first == name; }))
          return fail(node, "the expression uses '" + name +
                                "', which is not r, q1, q2, sigma, epsilon, "
                                "sigma1, sigma2, epsilon1, epsilon2, coulomb, "
                                "the time t, or a constant of the term");
      if (llvm::is_contained(expression->getNames(), "t"))
        control.usesTime = true;
      // The energy of a pair cannot depend on which particle comes first.
      // No structure of the expression shows that in general, so it is
      // tested at a few points, and the kernel states the symmetry as
      // asserted.
      const double samples[][7] = {{3.7, 0.41, -0.83, 3.1, 1.7, 0.21, 0.07},
                                   {5.3, -0.62, 0.35, 2.2, 3.4, 0.13, 0.48},
                                   {8.9, 0.97, 0.12, 1.3, 2.6, 0.92, 0.35}};
      for (const double *s : samples) {
        llvm::StringMap<double> values, swapped;
        for (const auto &[name, value] : term.constants)
          values[name] = swapped[name] = value;
        values["coulomb"] = swapped["coulomb"] = 332.0637;
        values["r"] = swapped["r"] = s[0];
        values["sigma"] = swapped["sigma"] = 0.5 * (s[3] + s[4]);
        values["epsilon"] = swapped["epsilon"] = std::sqrt(s[5] * s[6]);
        const char *pairs[][2] = {
            {"q1", "q2"}, {"sigma1", "sigma2"}, {"epsilon1", "epsilon2"}};
        for (auto [k, names] : llvm::enumerate(pairs)) {
          values[names[0]] = swapped[names[1]] = s[1 + 2 * k];
          values[names[1]] = swapped[names[0]] = s[2 + 2 * k];
        }
        double a = expression->evaluate(values);
        double b = expression->evaluate(swapped);
        if (std::isnan(a) && std::isnan(b))
          continue;
        if (!(std::abs(a - b) <=
              1e-12 * std::max({std::abs(a), std::abs(b), 1e-300})))
          return fail(node, "the energy of a pair must not change when its "
                            "two particles are exchanged (q1 with q2, sigma1 "
                            "with sigma2, epsilon1 with epsilon2), and this "
                            "expression does");
      }
    }
    return Error::success();
  }
  for (const PairTerm &term : control.pairs)
    if (!term.groups.empty())
      return fail(table, "'groups' of a pair term need a topology, whose "
                         "particles the masks select");
  if (control.types.empty())
    return fail(table, "expected at least one [[energy.type]]");
  if (control.pairs.empty())
    return fail(table, "expected at least one [[energy.pair]]");
  return Error::success();
}

Error Reader::readPME(const toml::table &table) {
  if (Error error = checkKeywords(table, "pme",
                                  {"tolerance", "beta", "max_spacing", "grid",
                                   "order", "influence"},
                                  {}))
    return error;
  if (Error error = readPositive(table, "beta", control.pmeAlpha))
    return error;
  if (Error error =
          readPositive(table, "tolerance", control.pmeAlphaTolerance))
    return error;
  if (!(control.pmeAlphaTolerance < 1.0))
    return fail(*table.get("tolerance"),
                "expected a tolerance less than 1 for 'tolerance'");
  if (const toml::node *node = table.get("grid")) {
    const toml::array *grid = node->as_array();
    if (!grid || grid->size() != 3)
      return fail(*node, "expected three numbers of points for 'grid'");
    for (int k = 0; k != 3; ++k) {
      std::optional<int64_t> points = (*grid)[k].value<int64_t>();
      if (!points || *points < 8)
        return fail(*node, "expected numbers of points of 8 or more for "
                           "'grid'");
      control.pmeGrid[k] = *points;
    }
  }
  if (Error error =
          readPositive(table, "max_spacing", control.pmeMaxSpacing))
    return error;
  if (Error error = readCount(table, "order", control.pmeOrder, 4))
    return error;
  if (control.pmeOrder != 4 && control.pmeOrder != 6 &&
      control.pmeOrder != 8)
    return fail(*table.get("order"), "expected 4, 6, or 8 for 'order'");
  if (Error error = readChoice<bool>(table, "influence", control.pmeOptimal,
                                     {{"OPTIMAL", true}, {"SPME", false}}))
    return error;
  return Error::success();
}

/// Checks that the intervals of output nest: each is a multiple of the one
/// inside it, and the number of steps a multiple of each. A checkpoint is
/// written where an interval between frames and one between energies ends;
/// frames and energies nest either way: if frames are the more frequent,
/// the energies are computed at each frame and the log shows those of its
/// interval.
Error Reader::checkIntervals(const toml::table &table) {
  const toml::table &at = outputTable ? *outputTable : table;
  auto checkMultiple = [&](StringRef outer, int64_t large, StringRef inner,
                           int64_t small) -> Error {
    if (large == 0 || small == 0 || large % small == 0)
      return Error::success();
    std::string hint;
    if (outer != "steps")
      hint = "; the output of '" + outer.str() +
             "' is written where an interval of '" + inner.str() +
             "' ends, so '" + inner.str() + "' must divide it";
    return fail(outer == "steps" ? table : at,
                "'" + outer + "' is not a multiple of '" + inner + "'" +
                    hint);
  };
  if (control.framePeriod >= control.energyPeriod ||
      control.framePeriod == 0) {
    if (Error error =
            checkMultiple("trajectory_interval", control.framePeriod,
                          "energy_interval", control.energyPeriod))
      return error;
  } else if (Error error =
                 checkMultiple("energy_interval", control.energyPeriod,
                               "trajectory_interval", control.framePeriod)) {
    return error;
  }
  if (Error error =
          checkMultiple("checkpoint_interval", control.checkpointPeriod,
                        "trajectory_interval", control.framePeriod))
    return error;
  if (Error error =
          checkMultiple("checkpoint_interval", control.checkpointPeriod,
                        "energy_interval", control.energyPeriod))
    return error;
  for (auto [name, period] :
       {std::pair<StringRef, int64_t>{"energy_interval",
                                      control.energyPeriod},
        {"trajectory_interval", control.framePeriod},
        {"checkpoint_interval", control.checkpointPeriod}})
    if (Error error = checkMultiple("steps", control.numSteps, name, period))
      return error;
  return Error::success();
}

Error Reader::readDynamics(const toml::table &table) {
  if (Error error = checkKeywords(
          table, "dynamics",
          {"integrator", "time_step", "steps", "seed",
           "center_of_mass_interval"},
          {}))
    return error;

  if (Error error = readChoice<Integrator>(
          table, "integrator", control.integrator,
          {{"VELOCITY_VERLET", Integrator::VelocityVerlet},
           {"LEAPFROG", Integrator::Leapfrog}}))
    return error;
  if (Error error = readPositive(table, "time_step", control.timestep))
    return error;
  if (Error error = readCount(table, "steps", control.numSteps, 0))
    return error;
  // Unset until [thermostat] and [barostat] are read; -1 stands for an
  // interval not given.
  control.comPeriod = control.thermostatPeriod = -1;
  control.barostatPeriod = -1;
  if (Error error =
          readCount(table, "center_of_mass_interval", control.comPeriod, 0))
    return error;
  int64_t seed = static_cast<int64_t>(control.seed);
  if (Error error = readCount(table, "seed", seed, 0))
    return error;
  control.seed = static_cast<uint64_t>(seed);
  return checkIntervals(table);
}

Error Reader::readMinimize(const toml::table &table) {
  if (Error error = checkKeywords(table, "minimize",
                                  {"method", "steps", "initial_step"}, {}))
    return error;
  enum class Method { SteepestDescent };
  Method method = Method::SteepestDescent;
  if (Error error = readChoice<Method>(
          table, "method", method,
          {{"STEEPEST_DESCENT", Method::SteepestDescent}}))
    return error;
  control.minimize = true;
  if (Error error = readCount(table, "steps", control.numSteps, 0))
    return error;
  if (Error error = readPositive(table, "initial_step", control.minimizeStep))
    return error;
  if (control.energyPeriod == 0)
    return fail(outputTable ? *outputTable : table,
                "a minimization writes energies: 'energy_interval' may not "
                "be 0");
  if (control.framePeriod != 0 &&
      control.framePeriod % control.energyPeriod != 0)
    return fail(outputTable ? *outputTable : table,
                "'trajectory_interval' is not a multiple of "
                "'energy_interval'");
  if (control.numSteps % control.energyPeriod != 0)
    return fail(table, "'steps' is not a multiple of 'energy_interval'");
  if (control.framePeriod != 0 && control.numSteps % control.framePeriod != 0)
    return fail(table, "'steps' is not a multiple of 'trajectory_interval'");
  return Error::success();
}

Error Reader::resolveCoupling() {
  int64_t &com = control.comPeriod, &thermostat = control.thermostatPeriod;
  if (!control.thermostat && thermostat > 0)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "%s: [thermostat] gives 'interval', but there is no thermostat",
        path.str().c_str());
  // The thermostat acts every 10 steps and the motion of the center of mass
  // is removed with it, unless one period is given: then both take it.
  // Without a thermostat the motion is removed only if
  // 'center_of_mass_interval' asks. Langevin dynamics acts in every step
  // and does not keep the momentum, so the motion is removed only if asked;
  // 'interval' then paces the removal and the barostat, every 10 steps by
  // default where there is one (D135).
  if (control.isLangevin()) {
    if (thermostat < 0)
      thermostat = com > 0 ? com : control.barostat ? 10 : 0;
    if (com < 0)
      com = 0;
    if (thermostat == 0 && control.barostat)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "%s: 'interval' in [thermostat] is 0",
                                     path.str().c_str());
    if (com != 0 && com != thermostat)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "%s: 'center_of_mass_interval' differs from 'interval' in "
          "[thermostat]; in M1 the motion of the center of mass is removed "
          "when the barostat acts, or never",
          path.str().c_str());
  } else if (control.thermostat) {
    if (thermostat < 0)
      thermostat = com > 0 ? com : 10;
    if (thermostat == 0)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "%s: 'interval' in [thermostat] is 0",
                                     path.str().c_str());
    if (com < 0)
      com = thermostat;
    if (com != 0 && com != thermostat)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "%s: 'center_of_mass_interval' differs from 'interval' in "
          "[thermostat]; in M1 the "
          "motion of the center of mass is removed when the thermostat acts, "
          "or never",
          path.str().c_str());
  } else {
    thermostat = 0;
    if (com < 0)
      com = 0;
  }

  // The barostat acts when the thermostat does (D72).
  int64_t &barostat = control.barostatPeriod;
  if (!control.barostat && barostat > 0)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "%s: [barostat] gives 'interval', but there is no barostat",
        path.str().c_str());
  if (!control.barostat)
    barostat = 0;
  else if (barostat < 0)
    barostat = thermostat;
  else if (barostat != thermostat)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "%s: 'interval' in [barostat] differs from that in [thermostat]; in "
        "M1 the barostat acts when the thermostat does",
        path.str().c_str());

  // Coupling acts at the end of the step that completes a period, so the
  // periods of output and the number of steps are multiples of it.
  int64_t period = control.getCouplingPeriod();
  if (period == 0)
    return Error::success();
  for (auto [name, value] :
       {std::pair<StringRef, int64_t>{"steps", control.numSteps},
        {"energy_interval", control.energyPeriod},
        {"trajectory_interval", control.framePeriod},
        {"checkpoint_interval", control.checkpointPeriod}})
    if (value % period != 0)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "%s: '%s' is not a multiple of the interval of coupling, %lld "
          "steps ('center_of_mass_interval' or 'interval' in [thermostat])",
          path.str().c_str(), name.str().c_str(), (long long)period);
  return Error::success();
}

Error Reader::readEnsemble(const toml::table &table) {
  if (Error error = checkKeywords(table, "ensemble",
                                  {"ensemble", "temperature", "pressure"},
                                  {{"gamma_t", "M1"}}))
    return error;
  if (Error error = readChoice<int>(table, "ensemble", ensembleKind,
                                    {{"NVE", 0}, {"NVT", 1}, {"NPT", 2}}))
    return error;
  if (ensembleKind != 2 && table.contains("pressure"))
    return fail(*table.get("pressure"),
                "'pressure' is for 'ensemble = \"NPT\"'");
  if (Error error = readReal(table, "pressure", control.pressure))
    return error;
  if (Error error = readReal(table, "temperature", control.temperature))
    return error;
  if (control.temperature < 0.0)
    return fail(*table.get("temperature"),
                "expected a temperature that is not negative");
  return Error::success();
}

Error Reader::readThermostat(const toml::table &table) {
  if (Error error = checkKeywords(table, "thermostat",
                                  {"method", "time_constant", "friction",
                                   "interval"},
                                  {}))
    return error;
  int method = -1;
  if (Error error = readChoice<int>(table, "method", method,
                                    {{"V-RESCALE", 0}, {"LANGEVIN", 1}}))
    return error;
  if (method < 0)
    return fail(table, "expected 'method' in [thermostat]: \"V-RESCALE\", "
                       "stochastic velocity rescaling, or \"LANGEVIN\", "
                       "Langevin dynamics");
  control.thermostat = true;
  if (method == 1) {
    control.thermostatMethod = ThermostatMethod::Langevin;
    if (const toml::node *node = table.get("time_constant"))
      return fail(*node, "'time_constant' is for \"V-RESCALE\"; Langevin "
                         "dynamics takes 'friction', in 1/ps");
    if (!table.get("friction"))
      return fail(table, "expected 'friction' in [thermostat], in 1/ps");
    control.friction = 0.0;
    if (Error error = readPositive(table, "friction", control.friction))
      return error;
  } else if (const toml::node *node = table.get("friction")) {
    return fail(*node, "'friction' is for \"LANGEVIN\"; stochastic velocity "
                       "rescaling takes 'time_constant', in ps");
  }
  if (Error error = readPositive(table, "time_constant", control.tauT))
    return error;
  if (Error error =
          readCount(table, "interval", control.thermostatPeriod, 0))
    return error;
  return Error::success();
}

Error Reader::readBarostat(const toml::table &table) {
  if (Error error = checkKeywords(
          table, "barostat",
          {"method", "time_constant", "compressibility", "coupling", "work",
           "interval", "compressibility_z", "surface_tension", "surfaces"},
          {}))
    return error;
  int method = -1;
  if (Error error = readChoice<int>(table, "method", method,
                                    {{"C-RESCALE", 0}}))
    return error;
  if (method < 0)
    return fail(table, "expected 'method' in [barostat]: \"C-RESCALE\", "
                       "stochastic cell rescaling");
  control.barostat = true;
  if (Error error = readPositive(table, "time_constant", control.tauP))
    return error;
  if (Error error =
          readPositive(table, "compressibility", control.compressibility))
    return error;
  if (Error error = readChoice<BarostatWork>(
          table, "work", control.barostatWork,
          {{"TROTTER", BarostatWork::Trotter},
           {"TROTTER_FIRST_ORDER", BarostatWork::TrotterFirstOrder},
           {"EXACT", BarostatWork::Exact},
           {"FIRST_ORDER", BarostatWork::FirstOrder}}))
    return error;
  int coupling = 0;
  if (Error error = readChoice<int>(table, "coupling", coupling,
                                    {{"ISOTROPIC", 0}, {"SEMI_ISOTROPIC", 1}}))
    return error;
  control.semiIsotropic = coupling == 1;
  // The keys of semi-isotropic coupling (D119).
  for (const char *key : {"compressibility_z", "surface_tension", "surfaces"})
    if (const toml::node *node = table.get(key); node && coupling != 1)
      return fail(*node, "'" + llvm::Twine(key) +
                             "' needs 'coupling = \"SEMI_ISOTROPIC\"'");
  control.compressibilityZ = control.compressibility;
  if (Error error =
          readReal(table, "compressibility_z", control.compressibilityZ))
    return error;
  if (const toml::node *node = table.get("compressibility_z");
      node && control.compressibilityZ < 0.0)
    return fail(*node, "expected 0 or a positive number for "
                       "'compressibility_z'");
  if (Error error = readReal(table, "surface_tension", control.surfaceTension))
    return error;
  if (Error error = readCount(table, "surfaces", control.surfaces, 1))
    return error;
  if (control.semiIsotropic &&
      control.barostatWork == BarostatWork::FirstOrder)
    return fail(*table.get("work"),
                "'work = \"FIRST_ORDER\"' counts the work from the trace of "
                "the virial of the step with twice the internal kinetic "
                "energy, which holds for the trace only; with "
                "'coupling = \"SEMI_ISOTROPIC\"' use \"TROTTER\", "
                "\"TROTTER_FIRST_ORDER\", or \"EXACT\"");
  if (Error error = readCount(table, "interval", control.barostatPeriod, 0))
    return error;
  return Error::success();
}

Error Reader::readBoundary(const toml::table &table) {
  if (Error error = checkKeywords(table, "boundary", {"type", "box"}, {}))
    return error;
  int type = 0;
  if (Error error = readChoice<int>(table, "type", type,
                                    {{"PERIODIC", 0}, {"NONE", 1}}))
    return error;
  const toml::node *node = table.get("box");
  // Without a periodic cell (D142) the run takes a cell around the
  // particles that no image reaches, from their positions; the cell of a
  // file of coordinates is ignored. What needs a periodic cell is refused.
  if (type == 1) {
    control.periodic = false;
    if (node)
      return fail(*node, "a run without a periodic cell takes no 'box'");
    if (control.pme)
      return fail(table, "particle mesh Ewald needs a periodic cell; "
                         "without one, give 'electrostatics = \"CUTOFF\"' "
                         "or \"REACTION_FIELD\"");
    if (control.barostat)
      return fail(table, "a barostat needs a periodic cell");
    if (control.topologyDispersionGiven &&
        control.topologyDispersion != DispersionCorrection::None)
      return fail(table, "the correction for the dispersion takes a "
                         "density, which a run without a periodic cell has "
                         "not; give 'dispersion_correction = \"NONE\"'");
    control.topologyDispersion = DispersionCorrection::None;
    for (const PairTerm &term : control.pairs)
      if (term.dispersion != DispersionCorrection::None)
        return fail(table, "the pair term '" + term.name + "' corrects for "
                           "the dispersion, which takes a density that a "
                           "run without a periodic cell has not");
    return Error::success();
  }
  // With a topology, the box is that of the file of coordinates, except
  // for CHARMM, whose coordinates have none.
  if (control.hasTopology() && control.charmmStructureFile.empty()) {
    if (node)
      return fail(*node, "'box' is not needed: the box comes from the file "
                         "of coordinates");
    return Error::success();
  }
  if (!node)
    return fail(table, control.charmmStructureFile.empty()
                           ? "expected 'box' in [boundary], the edges of the "
                             "cell in Å"
                           : "expected 'box' in [boundary], the edges of the "
                             "cell in Å: a coordinate file of CHARMM has no "
                             "cell");
  const toml::array *box = node->as_array();
  bool charmm = !control.charmmStructureFile.empty();
  if (!box || (box->size() != 3 && !(charmm && box->size() == 6)))
    return fail(*node, charmm ? "expected the three edges of the cell for "
                                "'box', in Å, and its angles α, β, γ in "
                                "degrees if they are not right angles"
                              : "expected the three edges of the cell for "
                                "'box'");
  for (int i = 0; i != 3; ++i) {
    std::optional<double> edge = (*box)[i].value<double>();
    if (!edge || !(*edge > 0.0))
      return fail(*node, "expected positive edges of the cell for 'box'");
    control.box[i] = *edge;
  }
  for (size_t i = 3; i < box->size(); ++i) {
    std::optional<double> angle = (*box)[i].value<double>();
    if (!angle || !(*angle > 0.0 && *angle < 180.0))
      return fail(*node, "expected angles between 0 and 180 degrees for "
                         "'box'");
    control.angles[i - 3] = *angle;
  }
  return Error::success();
}

Error Reader::readExecution(const toml::table &table) {
  if (Error error = checkKeywords(table, "execution",
                                  {"target", "threads", "precision",
                                   "neighbor_capacity", "fast_math",
                                   "spatial_order", "deterministic",
                                   "neighbor_structure"}))
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
          readCount(table, "neighbor_capacity", control.neighborWidth, 1))
    return error;
  if (Error error = readBool(table, "fast_math", control.fastMath))
    return error;
  if (Error error = readBool(table, "deterministic", control.deterministic))
    return error;
  // Groups of 16 that share a list of neighbors, each pair once, where the
  // loops allow them (D89): on a device, and not yet in the deterministic
  // mode.
  if (Error error = readChoice<NeighborStructure>(
          table, "neighbor_structure", control.neighborStructure,
          {{"MATRIX", NeighborStructure::Matrix},
           {"GROUPS", NeighborStructure::Groups}}))
    return error;
  if (control.neighborStructure == NeighborStructure::Groups &&
      (control.target != Target::GPU || control.deterministic))
    return fail(table, "'neighbor_structure = \"GROUPS\"' needs 'target = "
                       "\"GPU\"' and not 'deterministic'");
  return readBool(table, "spatial_order", control.reorder);
}

Error Reader::read(const toml::table &root) {
  if (Error error = checkKeywords(
          root, "the control file",
          {"input", "output", "energy", "pme", "dynamics", "minimize",
           "ensemble", "thermostat", "barostat",
           "boundary", "execution", "constraints", "restraints"},
          {}))
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
  if (Error error = readInput(*table))
    return error;

  if (Error error = getTable("output", /*required=*/false, table))
    return error;
  if (table)
    if (Error error = readOutput(*table))
      return error;

  if (Error error = getTable("energy", /*required=*/true, table))
    return error;
  if (Error error = readEnergy(*table))
    return error;
  if (Error error = getTable("pme", /*required=*/false, table))
    return error;
  if (table && !control.pme)
    return fail(*table, "[pme] is for 'electrostatics = \"PME\"' in "
                        "[energy]");
  if (table)
    if (Error error = readPME(*table))
      return error;

  // A run minimizes the energy or follows the dynamics.
  if (Error error = getTable("minimize", /*required=*/false, table))
    return error;
  if (table) {
    if (root.contains("dynamics"))
      return fail(*root.get("dynamics"),
                  "expected [dynamics] or [minimize], not both");
    if (Error error = readMinimize(*table))
      return error;
  } else {
    if (Error error = getTable("dynamics", /*required=*/true, table))
      return error;
    if (Error error = readDynamics(*table))
      return error;
  }

  if (Error error = getTable("ensemble", /*required=*/false, table))
    return error;
  const toml::table *ensembleTable = table;
  if (table)
    if (Error error = readEnsemble(*table))
      return error;
  // The coupling to the bath: [thermostat] for NVT and NPT, [barostat] for
  // NPT as well.
  const toml::table *thermostat, *barostat;
  if (Error error = getTable("thermostat", /*required=*/false, thermostat))
    return error;
  if (Error error = getTable("barostat", /*required=*/false, barostat))
    return error;
  if (control.minimize && (thermostat || barostat))
    return fail(thermostat ? *thermostat : *barostat,
                "a minimization has no thermostat or barostat");
  static const char *const names[] = {"NVE", "NVT", "NPT"};
  if (thermostat && ensembleKind == 0)
    return fail(*thermostat, "a [thermostat] needs 'ensemble = \"NVT\"' or "
                             "\"NPT\" in [ensemble]");
  if (barostat && ensembleKind != 2)
    return fail(*barostat,
                "a [barostat] needs 'ensemble = \"NPT\"' in [ensemble]");
  if (!thermostat && ensembleKind != 0)
    return fail(*ensembleTable, llvm::Twine("'ensemble = \"") +
                                    names[ensembleKind] +
                                    "\"' needs a [thermostat]");
  if (!barostat && ensembleKind == 2)
    return fail(*ensembleTable,
                "'ensemble = \"NPT\"' needs a [barostat]");
  if (thermostat)
    if (Error error = readThermostat(*thermostat))
      return error;
  if (barostat)
    if (Error error = readBarostat(*barostat))
      return error;

  if (!control.minimize)
    if (Error error = resolveCoupling())
      return error;

  if (Error error = getTable("constraints", /*required=*/false, table))
    return error;
  if (table) {
    if (Error error = checkKeywords(*table, "constraints",
                                    {"hydrogen_bonds", "rigid_water",
                                     "water_residues", "analytic_bonds"},
                                    {}))
      return error;
    if (Error error =
            readBool(*table, "hydrogen_bonds", control.rigidBonds))
      return error;
    if (Error error = readBool(*table, "rigid_water", control.fastWater))
      return error;
    if (Error error = readBool(*table, "analytic_bonds", control.analyticBonds))
      return error;
    if (const toml::node *node = table->get("water_residues")) {
      const toml::array *array = node->as_array();
      if (!array)
        return fail(*node, "expected a list of residue names for "
                           "'water_residues'");
      control.settleResidues.clear();
      for (const toml::node &element : *array) {
        if (!element.is_string())
          return fail(element, "expected a list of residue names for "
                               "'water_residues'");
        control.settleResidues.push_back(*element.value<std::string>());
      }
    }
    control.statesFlexible =
        table->contains("rigid_water") && !control.fastWater;
  }

  if (const toml::node *node = root.get("restraints")) {
    const toml::array *array = node->as_array();
    if (!array)
      return fail(*node, "expected [[restraints]]");
    for (const toml::node &element : *array) {
      const toml::table *entry = element.as_table();
      if (!entry)
        return fail(element, "expected [[restraints]]");
      if (Error error = checkKeywords(
              *entry, "restraints",
              {"selection", "force_constant", "reference_scaling"}, {}))
        return error;
      Control::Restraint restraint;
      if (Error error = readString(*entry, "selection", restraint.selection))
        return error;
      if (restraint.selection.empty())
        return fail(*entry, "expected a 'selection' in [[restraints]]");
      if (Error error =
              readPositive(*entry, "force_constant", restraint.forceConstant))
        return error;
      if (restraint.forceConstant == 0.0)
        return fail(*entry, "expected a 'force_constant' in [[restraints]]");
      if (Error error = readChoice<ReferenceScaling>(
              *entry, "reference_scaling", restraint.scaling,
              {{"CENTER", ReferenceScaling::Center},
               {"ALL", ReferenceScaling::All}}))
        return error;
      control.restraints.push_back(restraint);
    }
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

  if (control.prunedDistance != 0.0 &&
      control.neighborStructure != NeighborStructure::Groups)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "%s: 'pruned_distance' keeps a dual list, which needs "
        "'neighbor_structure = \"GROUPS\"'",
        path.str().c_str());
  if (control.checkpointPeriod != 0 && control.restartOutput.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "%s: 'checkpoint_interval' is given, but [output] names no "
        "'checkpoint'",
        path.str().c_str());
  // A minimization writes its checkpoint at the end.
  if (control.minimize && !control.restartOutput.empty())
    control.checkpointPeriod = control.numSteps;
  if (control.checkpointPeriod == 0 && !control.restartOutput.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "%s: [output] names a 'checkpoint', but 'checkpoint_interval' is "
        "not given",
        path.str().c_str());
  if (control.framePeriod != 0 && control.trajectoryFile.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "%s: 'trajectory_interval' is given, but [output] names no "
        "'trajectory'",
        path.str().c_str());
  // The coordinates of the terms over centers, at every energy (D145).
  if (!control.pullFile.empty()) {
    if (llvm::none_of(control.tupleTerms,
                      [](const TupleTerm &term) { return term.isCentroid(); }))
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "%s: [output] names 'pull_coordinates', but no term is over the "
          "centers of groups",
          path.str().c_str());
    if (control.energyPeriod == 0 || control.minimize)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "%s: 'pull_coordinates' are written at the energies of a run of "
          "dynamics, which needs 'energy_interval'",
          path.str().c_str());
  }
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
coordinates = "system.pdb"      # positions; the name of an atom is its type
# checkpoint = "earlier.h5"     # the state that the run begins from, at its
#                               # step

[output]
trajectory          = "run.dcd" # positions, in DCD or XTC (.xtc)
# checkpoint        = "run.h5"  # the state, with checkpoint_interval;
#                               # mdir run --continue goes on from it
energy_interval     = 10        # steps between energies in the log; 0: none
trajectory_interval = 0         # steps between frames; 0: none
# checkpoint_interval = 0       # steps between checkpoints

[energy]
cutoff            = 12.0        # Å
switch_distance   = 10.0        # where switching begins (Å); the cutoff: none
pairlist_distance = 13.5        # reach of the neighbor structures (Å)
# pruned_distance = 0           # reach of the inner list of a dual list,
                                # pruned from the structure; needs GROUPS (Å)
# rebuild_interval = 0          # 0: rebuild when a particle has moved half
                                # the skin (default); N: every N steps, not
                                # tested between; may miss pairs (opt-in)
# lennard_jones_modifier = "NONE"  # NONE, POTENTIAL_SHIFT, FORCE_SWITCH,
                                   # POWER_FORCE_SWITCH (a topology only)

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
integrator = "VELOCITY_VERLET"  # VELOCITY_VERLET, LEAPFROG
time_step  = 0.001              # ps
steps      = 100                # of the run; --continue runs to them
seed       = 314159             # of the velocities and the thermostat
# center_of_mass_interval = 0   # steps between removals of the motion of
#                               # the center of mass; 0: none

[ensemble]
ensemble    = "NVE"             # NVE, NVT (with [thermostat])
temperature = 298.15            # of the velocities and the bath (K)

# With 'ensemble = "NVT"':
# [thermostat]
# method        = "V-RESCALE"   # stochastic velocity rescaling, or
#                               # "LANGEVIN", Langevin dynamics
# time_constant = 1.0           # ps, with V-RESCALE
# friction      = 1.0           # 1/ps, with LANGEVIN
# interval      = 10            # steps between its actions

[boundary]
type = "PERIODIC"
box  = [40.0, 40.0, 40.0]       # edges of the cell (Å)

[execution]
target        = "CPU"           # CPU, GPU
threads       = 1               # for the target CPU
precision     = "DOUBLE"        # SINGLE, MIXED, DOUBLE
fast_math     = true            # allow rewrites that change rounding
spatial_order = true            # keep the particles in the order of their
                                # positions
deterministic = false           # sums in an order the threads do not decide:
                                # the same bits from run to run
# neighbor_capacity = 160       # neighbors per particle at first; grows
# neighbor_structure = "MATRIX" # MATRIX, GROUPS: groups of 16 that share
                                # a list, each pair once (GPU only)
)TOML";
}

std::string mdir::driver::getAmberControlTemplate() {
  return R"TOML([input]
topology    = "system.prmtop"   # of Amber (tleap, ParmEd)
coordinates = "system.inpcrd"   # and the box; the reference of restraints
# format    = "AUTO"            # AUTO (from the names), AMBER, GROMACS, PDB
# checkpoint = "earlier.h5"     # the state that the run begins from, at its
#                               # step; one of a minimization gives only
#                               # positions

[output]
trajectory          = "run.dcd" # positions, in DCD or XTC (.xtc)
checkpoint          = "run.h5"  # the state; mdir run --continue goes on
#                               # from it, and the one before is run.h5.prev
energy_interval     = 5000      # steps between energies in the log
trajectory_interval = 5000      # steps between frames
checkpoint_interval = 50000     # steps between checkpoints

[energy]
cutoff            = 9.0         # of the direct terms (Å)
pairlist_distance = 10.0        # reach of the neighbor structures (Å)
# pruned_distance = 0           # reach of the inner list of a dual list,
                                # pruned from the structure; needs GROUPS (Å)
# rebuild_interval = 0          # 0: rebuild when a particle has moved half
                                # the skin (default); N: every N steps, not
                                # tested between; may miss pairs (opt-in)
electrostatics    = "PME"       # PME, CUTOFF
coulomb_modifier  = "POTENTIAL_SHIFT"  # NONE, POTENTIAL_SHIFT: the direct
                                       # sum shifted to zero at the cutoff
# dispersion_correction = "ENERGY_PRESSURE"  # NONE, ENERGY_PRESSURE

# Particle mesh Ewald; every entry has a default.
# [pme]
# tolerance   = 1.0e-5          # erfc(β r_c), which gives β
# beta        = 0.35            # β (1/Å), instead
# max_spacing = 1.2             # largest spacing of the grid (Å)
# grid        = [48, 48, 48]    # the grid, instead
# order       = 4               # of the B-splines: 4, 6, 8
# influence   = "SPME"          # SPME, OPTIMAL (as sander)

[dynamics]
integrator = "VELOCITY_VERLET"  # VELOCITY_VERLET, LEAPFROG (velocities
                                # half a step behind)
time_step  = 0.002              # ps
steps      = 500000             # of the run; --continue runs to them
seed       = 314159             # of the velocities and the coupling
# center_of_mass_interval = 10  # steps between removals of the motion of
#                               # the center of mass: with a thermostat,
#                               # when it acts

[ensemble]
ensemble    = "NPT"             # NVE, NVT (with [thermostat]), NPT (with
                                # [thermostat] and [barostat])
temperature = 300.0             # of the velocities and the bath (K)
pressure    = 1.0               # atm, with NPT

[thermostat]
method        = "V-RESCALE"     # stochastic velocity rescaling, or
                                # "LANGEVIN", Langevin dynamics
time_constant = 0.5             # ps, with V-RESCALE
# friction    = 1.0             # 1/ps, with LANGEVIN
interval      = 10              # steps between its actions

[barostat]
method        = "C-RESCALE"     # stochastic cell rescaling
time_constant = 2.0             # ps
# compressibility = 4.56e-5     # 1/atm (4.5e-5 /bar)
# coupling = "ISOTROPIC"        # ISOTROPIC; SEMI_ISOTROPIC: x and y
#                               # together, z on its own
# compressibility_z = 4.56e-5   # 1/atm, of z with SEMI_ISOTROPIC (0 keeps
#                               # the height); compressibility by default
# surface_tension = 0.0         # dyn/cm, of each surface normal to z,
# surfaces        = 2           # with SEMI_ISOTROPIC
# work     = "TROTTER"          # TROTTER: the scaling within the drift of
#                               # a step, its energy from the virials before
#                               # and after; TROTTER_FIRST_ORDER: from the
#                               # one before, a virial less; EXACT: the energy of each
#                               # scaling from the scaled positions, whose
#                               # forces the next step takes; FIRST_ORDER:
#                               # from the virial, as GROMACS does
# interval = 10                 # steps between its actions: those of the
#                               # thermostat

[constraints]
hydrogen_bonds = true           # SHAKE and RATTLE on the bonds of hydrogen
analytic_bonds = false          # checked quadratic solve for one-bond groups
rigid_water    = true           # rigid waters: SETTLE in DOUBLE,
                                # M-SHAKE on their three distances below
# water_residues = ["WAT"]      # names of the residues of rigid water

[boundary]
type = "PERIODIC"               # the box is that of the coordinates;
                                # NONE: no periodic cell

[execution]
target    = "GPU"               # CPU, GPU
precision = "MIXED"             # SINGLE, MIXED, DOUBLE
# threads = 1                   # for the target CPU
# neighbor_structure = "MATRIX"  # MATRIX, GROUPS: groups of 16 that share
                                # a list, each pair once

# A minimization instead of dynamics: steepest descent.
# [minimize]
# method       = "STEEPEST_DESCENT"
# steps        = 2000
# initial_step = 0.1            # Å

# Restraints to the positions of 'coordinates', any number of them.
# [[restraints]]
# selection         = "!:WAT & !@H*"  # a mask of Amber
# force_constant    = 10.0            # kcal/mol/Å²
# reference_scaling = "CENTER"        # or "ALL", under a barostat
)TOML";
}
