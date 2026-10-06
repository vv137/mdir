// Tunable parameters of a model (D[python-tunable],
// docs/python-tunable.md): their declarations resolved against a prepared
// model, and their values put into it, from which the builder computes the
// values of the program and every quantity derived from them.
#include "mdir/Driver/Model.h"
#include "llvm/ADT/StringExtras.h"
#include <cmath>
#include <set>

using namespace mdir;
using namespace mdir::model;

static llvm::Error input(const llvm::Twine &s) {
  return llvm::make_error<ModelError>(ModelError::Input, s.str());
}

static std::string number(double value) {
  char text[32];
  std::snprintf(text, sizeof text, "%.10g", value);
  return text;
}

/// The σ of a pair of types from those of the two types.
static double mixSigma(driver::Mixing mixing, double a, double b) {
  return mixing == driver::Mixing::Geometric ? std::sqrt(a * b) : 0.5 * (a + b);
}

/// Whether `value` is `rule` within 1e-6 of the larger: the eight digits of
/// the coefficients of an Amber topology leave its tables within 2e-7 of
/// the combining rule, while an NBFIX differs by percents.
static bool follows(double value, double rule) {
  return std::fabs(value - rule) <= 1.0e-6 * std::max(std::fabs(value), std::fabs(rule));
}

llvm::Expected<TunableSet>
mdir::model::resolveTunables(const System &model, driver::Control &control,
                             driver::System &system) {
  TunableSet set;
  if (model.tunables.empty())
    return set;
  const driver::Topology &topology = *system.topology;
  size_t types = topology.getNumTypes();
  for (size_t a = 0; a != types; ++a) {
    set.typeSigma.push_back(topology.sigma[a * types + a]);
    set.typeEpsilon.push_back(topology.epsilon[a * types + a]);
  }
  std::set<std::string> names;
  std::set<std::tuple<int, unsigned, std::string>> parameters;
  for (const Tunable &tunable : model.tunables) {
    std::string where = "the tunable '" + tunable.name + "'";
    llvm::StringRef name = tunable.name;
    if (name.empty() || !(llvm::isAlpha(name.front()) || name.front() == '_') ||
        !llvm::all_of(name, [](char c) { return llvm::isAlnum(c) || c == '_'; }))
      return input("a tunable needs a name of letters, digits, and '_', "
                   "beginning with a letter or '_'; found '" + tunable.name + "'");
    if (!names.insert(tunable.name).second)
      return input("two tunables are named '" + tunable.name + "'");
    TunableSet::Entry entry;
    entry.name = tunable.name;
    entry.parameter = tunable.parameter;
    entry.term = tunable.term;
    entry.mixing = tunable.mixing;
    if (entry.mixing != driver::Mixing::Arithmetic &&
        entry.mixing != driver::Mixing::Geometric)
      return input(where + ": the mixing of σ is arithmetic or geometric");
    if (entry.mixing != driver::Mixing::Arithmetic && tunable.parameter != "sigma")
      return input(where + ": a mixing rule is for \"sigma\" only");
    // The values of the sites in the model.
    std::vector<double> sites;
    if (tunable.term.empty()) {
      if (tunable.parameter == "charge") {
        entry.kind = TunableSet::Entry::Charge;
        entry.unit = "e";
        sites = topology.charges;
      } else if (tunable.parameter == "sigma") {
        entry.kind = TunableSet::Entry::Sigma;
        entry.unit = "nm";
        sites = set.typeSigma;
      } else if (tunable.parameter == "epsilon") {
        entry.kind = TunableSet::Entry::Epsilon;
        entry.unit = "kJ/mol";
        sites = set.typeEpsilon;
      } else {
        return input(where + ": the parameter is \"charge\", \"sigma\", or "
                     "\"epsilon\", or a constant or parameter of the term "
                     "given as `term`; found \"" + tunable.parameter + "\"");
      }
    } else {
      bool found = false;
      for (auto [k, term] : llvm::enumerate(control.pairs)) {
        if (term.name != tunable.term)
          continue;
        for (const auto &[constant, value] : term.constants)
          if (constant == tunable.parameter) {
            entry.kind = TunableSet::Entry::PairConstant;
            entry.termIndex = k;
            sites = {value};
            found = true;
          }
        if (!found)
          return input(where + ": the pair term '" + tunable.term +
                       "' has no constant '" + tunable.parameter + "'");
      }
      for (auto [k, term] : llvm::enumerate(topology.tupleTerms)) {
        if (found || term.name != tunable.term)
          continue;
        for (const auto &[parameter, values] : term.parameters)
          if (parameter == tunable.parameter) {
            entry.kind = TunableSet::Entry::TupleParameter;
            entry.termIndex = k;
            sites = values;
            found = true;
          }
        if (!found)
          return input(where + ": the tuple term '" + tunable.term +
                       "' has no parameter '" + tunable.parameter + "'");
      }
      if (!found)
        return input(where + ": no pair or tuple term is named '" +
                     tunable.term + "'");
      entry.unit = "the units of the expression of '" + tunable.term + "'";
    }
    unsigned termKey = tunable.term.empty() ? 0 : entry.termIndex;
    if (!parameters.insert({entry.kind, termKey, tunable.parameter}).second)
      return input(where + ": the parameter \"" + tunable.parameter + "\"" +
                   (tunable.term.empty() ? "" : " of '" + tunable.term + "'") +
                   " is declared by another tunable");

    // The map from the sites to the entries.
    if (tunable.map.empty()) {
      for (size_t i = 0; i != sites.size(); ++i)
        entry.map.push_back(static_cast<int64_t>(i));
    } else {
      if (tunable.map.size() != sites.size())
        return input(where + ": the map has " + llvm::Twine(tunable.map.size()) +
                     " entries, and the parameter has " + llvm::Twine(sites.size()) +
                     " sites");
      entry.map = tunable.map;
    }
    int64_t most = -1;
    for (int64_t k : entry.map) {
      if (k < -1)
        return input(where + ": an entry of the map is below -1");
      most = std::max(most, k);
    }
    if (most < 0)
      return input(where + ": the map takes no site");
    entry.entries = static_cast<size_t>(most) + 1;
    std::vector<int64_t> first(entry.entries, -1);
    for (size_t i = 0; i != entry.map.size(); ++i)
      if (entry.map[i] >= 0 && first[entry.map[i]] < 0)
        first[entry.map[i]] = static_cast<int64_t>(i);
    for (size_t k = 0; k != entry.entries; ++k)
      if (first[k] < 0)
        return input(where + ": no site takes the entry " + llvm::Twine(k) +
                     " of the map; the entries are 0 to M - 1, each taken");

    // The initial values: those given, or the common value of the sites of
    // each entry.
    std::vector<double> values;
    if (!tunable.values.empty()) {
      if (tunable.values.size() != entry.entries)
        return input(where + ": `values` has " + llvm::Twine(tunable.values.size()) +
                     " entries, and the map " + llvm::Twine(entry.entries));
      values = tunable.values;
    } else {
      for (size_t k = 0; k != entry.entries; ++k)
        values.push_back(sites[first[k]]);
      for (size_t i = 0; i != entry.map.size(); ++i)
        if (entry.map[i] >= 0 && sites[i] != values[entry.map[i]])
          return input(where + ": the sites " + llvm::Twine(first[entry.map[i]]) +
                       " and " + llvm::Twine(i) + " of the entry " +
                       llvm::Twine(entry.map[i]) + " differ in the model (" +
                       number(values[entry.map[i]]) + " and " +
                       number(sites[i]) + "); give `values`");
    }
    if (entry.kind == TunableSet::Entry::PairConstant)
      control.tunableConstants.push_back({entry.termIndex, tunable.parameter});
    set.tunables.push_back(std::move(entry));
    set.values.push_back(std::move(values));
  }
  if (llvm::Error error = checkTunableValues(set, set.values))
    return std::move(error);

  // σ and ε of the pairs of types follow the combining rule from those of
  // the types, which the table of the model must already do.
  for (const TunableSet::Entry &entry : set.tunables) {
    if (entry.kind != TunableSet::Entry::Sigma &&
        entry.kind != TunableSet::Entry::Epsilon)
      continue;
    bool sigma = entry.kind == TunableSet::Entry::Sigma;
    for (size_t a = 0; a != types; ++a)
      for (size_t b = 0; b != types; ++b) {
        double epsilon = topology.epsilon[a * types + b];
        double value, rule;
        if (sigma) {
          // σ of a pair without Lennard-Jones does not matter.
          if (epsilon == 0.0)
            continue;
          value = topology.sigma[a * types + b];
          rule = mixSigma(entry.mixing, set.typeSigma[a], set.typeSigma[b]);
        } else {
          value = epsilon;
          rule = std::sqrt(set.typeEpsilon[a] * set.typeEpsilon[b]);
        }
        if (!follows(value, rule))
          return input("the tunable '" + entry.name + "': " +
                       (sigma ? "σ" : "ε") + " of the types '" +
                       topology.typeNames[a] + "' and '" + topology.typeNames[b] +
                       "' is " + number(value) + ", and the combining rule "
                       "of the two types gives " + number(rule) +
                       "; a table with such pairs (NBFIX) cannot take per-type "
                       "values");
      }
  }
  for (const auto &tails : system.pairTails)
    set.pairTails.push_back(!tails.empty());
  control.tunables = true;
  if (llvm::Error error = applyTunables(set, set.values, control, system))
    return std::move(error);
  return set;
}

llvm::Error mdir::model::checkTunableValues(
    const TunableSet &set, const std::vector<std::vector<double>> &values) {
  if (values.size() != set.tunables.size())
    return input("expected the values of " + llvm::Twine(set.tunables.size()) +
                 " tunables");
  for (auto [entry, given] : llvm::zip_equal(set.tunables, values)) {
    if (given.size() != entry.entries)
      return input("the tunable '" + entry.name + "' takes shape (" +
                   llvm::Twine(entry.entries) + ",); found (" +
                   llvm::Twine(given.size()) + ",)");
    for (double value : given) {
      if (!std::isfinite(value))
        return input("the tunable '" + entry.name + "' takes finite values");
      if ((entry.kind == TunableSet::Entry::Sigma ||
           entry.kind == TunableSet::Entry::Epsilon) && value < 0.0)
        return input("the tunable '" + entry.name + "' takes values of at "
                     "least 0 (" + entry.unit + ")");
    }
  }
  return llvm::Error::success();
}

llvm::Error mdir::model::applyTunables(
    const TunableSet &set, const std::vector<std::vector<double>> &values,
    driver::Control &control, driver::System &system) {
  if (set.empty())
    return llvm::Error::success();
  if (llvm::Error error = checkTunableValues(set, values))
    return error;
  auto topology = std::make_shared<driver::Topology>(*system.topology);
  size_t types = topology->getNumTypes();
  std::vector<double> sigma = set.typeSigma, epsilon = set.typeEpsilon;
  bool mixesSigma = false, mixesEpsilon = false;
  driver::Mixing mixing = driver::Mixing::Arithmetic;
  for (auto [entry, theta] : llvm::zip_equal(set.tunables, values)) {
    auto take = [&](std::vector<double> &target) {
      for (size_t i = 0; i != entry.map.size(); ++i)
        if (entry.map[i] >= 0)
          target[i] = theta[entry.map[i]];
    };
    switch (entry.kind) {
    case TunableSet::Entry::Charge:
      take(topology->charges);
      break;
    case TunableSet::Entry::Sigma:
      take(sigma);
      mixesSigma = true;
      mixing = entry.mixing;
      break;
    case TunableSet::Entry::Epsilon:
      take(epsilon);
      mixesEpsilon = true;
      break;
    case TunableSet::Entry::PairConstant:
      for (auto &[name, value] : control.pairs[entry.termIndex].constants)
        if (name == entry.parameter && entry.map[0] >= 0)
          value = theta[entry.map[0]];
      break;
    case TunableSet::Entry::TupleParameter:
      for (auto &[name, list] : topology->tupleTerms[entry.termIndex].parameters)
        if (name == entry.parameter)
          take(list);
      break;
    }
  }
  // The table of the pairs of types by the combining rule: Lorentz (or the
  // geometric mean) for σ, Berthelot for ε.
  for (size_t a = 0; a != types; ++a)
    for (size_t b = 0; b != types; ++b) {
      if (mixesSigma)
        topology->sigma[a * types + b] = mixSigma(mixing, sigma[a], sigma[b]);
      if (mixesEpsilon)
        topology->epsilon[a * types + b] = std::sqrt(epsilon[a] * epsilon[b]);
    }
  system.topology = std::move(topology);
  // The tails of the pair terms (D209), whose classes of particles and
  // values follow the charges, the table, and the constants.
  if (llvm::Error error = driver::recollectPairTails(control, system))
    return input("the new values: " + llvm::toString(std::move(error)));
  for (size_t k = 0; k != set.pairTails.size() && k != system.pairTails.size(); ++k)
    if (set.pairTails[k] != !system.pairTails[k].empty())
      return input("at the new values the tail of the pair term '" +
                   control.pairs[k].name + "' " +
                   (set.pairTails[k] ? "diverges, and the correction for the "
                                       "dispersion would leave it out"
                                     : "converges, and the correction for the "
                                       "dispersion would take it in") +
                   "; that changes the program: compile it with these values");
  return llvm::Error::success();
}
