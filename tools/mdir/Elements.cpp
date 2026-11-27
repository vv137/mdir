// The elements of a topology as `mdir check` reports them (#469,
// D[gromacs-hmr-elements]); see Elements.h and docs/check.md.

#include "Elements.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <tuple>

using namespace mdir;
using namespace mdir::driver;
using ElementSource = Topology::ElementSource;

namespace {
/// The symbols of the elements, by atomic number.
const char *const symbols[] = {
    "",   "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na",
    "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",
    "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
    "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag",
    "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr",
    "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
    "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am",
    "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh",
    "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"};
constexpr int numSymbols = sizeof(symbols) / sizeof(symbols[0]);

/// The standard atomic weights, in amu, of the elements 1 to 56 (IUPAC
/// 2021, abridged to five significant digits; for Tc the mass number of
/// its longest-lived isotope), which the masses of groups are compared
/// with.
const double weights[] = {
    0.0,    1.008,  4.0026, 6.94,   9.0122, 10.81,  12.011, 14.007,
    15.999, 18.998, 20.180, 22.990, 24.305, 26.982, 28.085, 30.974,
    32.06,  35.45,  39.95,  39.098, 40.078, 44.956, 47.867, 50.942,
    51.996, 54.938, 55.845, 58.933, 58.693, 63.546, 65.38,  69.723,
    72.630, 74.922, 78.971, 79.904, 83.798, 85.468, 87.62,  88.906,
    91.224, 92.906, 95.95,  98.0,   101.07, 102.91, 106.42, 107.87,
    112.41, 114.82, 118.71, 121.76, 127.60, 126.90, 131.29, 132.91,
    137.33};
constexpr int numWeights = sizeof(weights) / sizeof(weights[0]);

/// The names of ion types that do not begin with the symbol of their
/// element: those of the CHARMM force fields, and the calcium and the older
/// ions of Amber.
const std::pair<const char *, int> ionTypes[] = {
    {"SOD", 11}, {"POT", 19}, {"CES", 55}, {"RUB", 37}, {"CAL", 20},
    {"LIT", 3},  {"BAR", 56}, {"CLA", 17}, {"CAD", 48}, {"C0", 20},
    {"IP", 11},  {"IM", 17}};

/// The atomic numbers that the letters of a type name may stand for: the
/// one-letter and the two-letter symbol it begins with, after any digits
/// (`2C` of ff14SB), and that of a known ion type of this name. Empty where
/// the name begins with no symbol.
std::vector<int> getNameElements(llvm::StringRef name) {
  std::string upper = name.trim().upper();
  std::vector<int> found;
  for (auto [type, number] : ionTypes)
    if (upper == type)
      found.push_back(number);
  llvm::StringRef letters = llvm::StringRef(upper).drop_while(
      [](char c) { return !llvm::isAlpha(c); });
  for (size_t length : {1, 2}) {
    if (letters.size() < length || !llvm::isAlpha(letters[length - 1]))
      continue;
    for (int number = 1; number != numSymbols; ++number)
      if (llvm::StringRef(symbols[number]).upper() ==
          letters.take_front(length))
        found.push_back(number);
  }
  return found;
}

const char *getSourceText(ElementSource source) {
  switch (source) {
  case ElementSource::File:
    return "from the file";
  case ElementSource::Name:
    return "from the name and the mass";
  case ElementSource::Mass:
    return "from the mass";
  case ElementSource::None:
    return "none in the file";
  }
  return "";
}

/// Where the format keeps the element, for the hints of the warnings.
std::string getWhere(const Control &control) {
  if (!control.gromacsTopologyFile.empty())
    return "the at.num column of its line in [ atomtypes ]";
  if (!control.charmmStructureFile.empty())
    return "the element at the end of its MASS line";
  return "the ATOMIC_NUMBER section of the prmtop";
}
} // namespace

std::string mdir::tool::getElementSymbol(int number) {
  if (number < 0)
    return "?";
  if (number == 0)
    return "none";
  if (number < numSymbols)
    return symbols[number];
  return "Z" + std::to_string(number);
}

const char *mdir::tool::getElementSourceKey(ElementSource source) {
  switch (source) {
  case ElementSource::File:
    return "file";
  case ElementSource::Name:
    return "name_and_mass";
  case ElementSource::Mass:
    return "mass";
  case ElementSource::None:
    return "none";
  }
  return "";
}

tool::ElementReport mdir::tool::diagnoseElements(const Control &control,
                                                 const Topology &topology) {
  ElementReport report;
  size_t count = topology.getNumParticles();
  const Topology::ElementOrigins &origins = topology.elementOrigins;
  bool hasSources = origins.sources.size() == count;
  bool hasTypes = origins.atomTypes.size() == count;
  auto sourceOf = [&](size_t i) {
    return hasSources ? origins.sources[i] : ElementSource::File;
  };
  auto typeOf = [&](size_t i) -> std::string {
    if (hasTypes && origins.atomTypes[i] < origins.atomTypeNames.size())
      return origins.atomTypeNames[origins.atomTypes[i]];
    return topology.typeNames[topology.types[i]];
  };
  // The atomic number as the run takes it: that of the topology, and for a
  // particle without one a hydrogen as the reader of GROMACS and the bonds
  // of hydrogen take it (isHydrogenWithoutNumber).
  auto numberOf = [&](size_t i) {
    int number = i < topology.atomicNumbers.size() ? topology.atomicNumbers[i]
                                                   : -1;
    if (number >= 0)
      return number;
    return topology.masses[i] > 0.0 &&
                   isHydrogenWithoutNumber(topology.masses[i],
                                           topology.atomNames[i])
               ? 1
               : -1;
  };

  std::map<std::pair<int, ElementSource>, size_t> counts;
  for (size_t i = 0; i != count; ++i) {
    ++counts[{numberOf(i), sourceOf(i)}];
    ++report.sources[static_cast<unsigned>(sourceOf(i))];
  }
  for (auto [key, n] : counts)
    report.counts.push_back({key.first, key.second, n});

  for (const Topology::Shake &shake : topology.shakes)
    report.hydrogenBonds += shake.hydrogens.size();
  report.hydrogenGroups = topology.shakes.size();
  report.rigidWaters = topology.settles.size();

  // The bonds of each particle: those of the terms, those that SHAKE keeps,
  // and those of the rigid waters, whose terms are gone.
  std::vector<std::vector<unsigned>> bonded(count);
  auto bond = [&](unsigned a, unsigned b) {
    if (a < count && b < count) {
      bonded[a].push_back(b);
      bonded[b].push_back(a);
    }
  };
  for (const Topology::Bond &b : topology.bonds)
    bond(b.i, b.j);
  for (const Topology::Shake &shake : topology.shakes)
    for (unsigned h : shake.hydrogens)
      bond(shake.center, h);
  std::vector<bool> settled(count, false);
  for (const Topology::Settle &settle : topology.settles) {
    bond(settle.oxygen, settle.oxygen + 1);
    bond(settle.oxygen, settle.oxygen + 2);
    for (unsigned k = 0; k != 3; ++k)
      if (settle.oxygen + k < count)
        settled[settle.oxygen + k] = true;
  }

  std::string where = getWhere(control);
  // Each warning names a type; the text names the first `listed` types of
  // each kind and counts the rest.
  const size_t listed = 8;
  auto describe = [&](size_t i) {
    int number = numberOf(i);
    if (number < 0)
      return std::string("no element (none in the file, and neither below "
                         "1.2 amu nor named H below 4.1 amu)");
    return getElementSymbol(number) + " (" + getSourceText(sourceOf(i)) +
           ")";
  };
  auto addRest = [&](const char *code, size_t types, const char *what) {
    if (types > listed)
      report.warnings.push_back(
          {code, llvm::formatv("elements: and {0} more types with {1}",
                               types - listed, what)
                     .str()});
  };

  // A heavy atom and its hydrogens keep their total mass when the masses
  // of the hydrogens are repartitioned: a group whose mass differs from
  // the standard atomic weights of its elements has a wrong element. A
  // united atom of C, N, O, or S carries one to three hydrogens more.
  struct Group {
    size_t particles = 0;
    double low = 0.0, high = 0.0;
    size_t first = 0;
  };
  std::map<std::pair<std::string, int>, Group> groups;
  std::vector<std::string> groupOrder;
  for (size_t i = 0; i != count; ++i) {
    int number = numberOf(i);
    double mass = topology.masses[i];
    if (number <= 1 || number >= numWeights || !(mass > 0.0))
      continue;
    double sum = mass, reference = weights[number];
    for (unsigned j : bonded[i])
      if (numberOf(j) == 1) {
        sum += topology.masses[j];
        reference += weights[1];
      }
    double difference = sum - reference;
    if (std::fabs(difference) <= 0.1)
      continue;
    bool united = false;
    if (number == 6 || number == 7 || number == 8 || number == 16)
      for (int k = 1; k <= 3; ++k)
        united |= std::fabs(difference - k * weights[1]) <= 0.1;
    if (united)
      continue;
    auto [entry, added] = groups.insert({{typeOf(i), number}, Group{}});
    Group &group = entry->second;
    if (added) {
      group.low = group.high = difference;
      group.first = i;
    }
    ++group.particles;
    group.low = std::min(group.low, difference);
    group.high = std::max(group.high, difference);
  }
  {
    std::vector<std::pair<size_t, std::pair<std::string, int>>> order;
    for (auto &[key, group] : groups)
      order.push_back({group.first, key});
    llvm::sort(order);
    for (auto [index, entry] : llvm::enumerate(order)) {
      if (index == listed)
        break;
      const Group &group = groups[entry.second];
      std::string range =
          group.low == group.high
              ? llvm::formatv("{0:F3} amu", group.low).str()
              : llvm::formatv("{0:F3} to {1:F3} amu", group.low, group.high)
                    .str();
      report.warnings.push_back(
          {"element_group_mass",
           llvm::formatv(
               "elements: {0} group{1} of a particle of the type '{2}', "
               "taken for {3}, and the hydrogens bonded to it differ by {4} "
               "from the standard atomic weights of their elements; "
               "repartitioning the masses of hydrogens keeps the mass of "
               "such a group, so an element is likely wrong: check the "
               "atomic number of the type '{2}' and of the hydrogens bonded "
               "to it, in {5}",
               group.particles, group.particles == 1 ? "" : "s",
               entry.second.first, describe(group.first), range, where)
               .str()});
    }
    addRest("element_group_mass", groups.size(),
            "groups whose mass differs from the standard atomic weights");
  }

  // A particle with one bond and a mass below 4.1 amu that the run does
  // not take for a hydrogen: 'hydrogen_bonds = true' leaves its bond free.
  // SETTLE keeps those of the rigid waters, whatever their elements.
  struct Light {
    size_t particles = 0;
    double low = 0.0, high = 0.0;
    size_t first = 0;
  };
  std::map<std::string, Light> lights;
  for (size_t i = 0; i != count; ++i) {
    double mass = topology.masses[i];
    if (bonded[i].size() != 1 || !(mass > 0.0) || mass >= 4.1 ||
        numberOf(i) == 1 || settled[i])
      continue;
    auto [entry, added] = lights.insert({typeOf(i), Light{}});
    Light &light = entry->second;
    if (added) {
      light.low = light.high = mass;
      light.first = i;
    }
    ++light.particles;
    light.low = std::min(light.low, mass);
    light.high = std::max(light.high, mass);
  }
  {
    std::vector<std::pair<size_t, std::string>> order;
    for (auto &[type, light] : lights)
      order.push_back({light.first, type});
    llvm::sort(order);
    for (auto [index, entry] : llvm::enumerate(order)) {
      if (index == listed)
        break;
      const Light &light = lights[entry.second];
      std::string range =
          light.low == light.high
              ? llvm::formatv("{0:F3} amu", light.low).str()
              : llvm::formatv("{0:F3} to {1:F3} amu", light.low, light.high)
                    .str();
      bool none = numberOf(light.first) < 0 &&
                  sourceOf(light.first) == ElementSource::None;
      report.warnings.push_back(
          {"element_light_atom",
           llvm::formatv(
               "elements: {0} particle{1} of the type '{2}' {3} one bond and "
               "a mass of {4}, below 4.1 amu, but {5} taken for {6}, not "
               "hydrogen, so 'hydrogen_bonds = true' leaves {7} bond{1} "
               "free; if {8} hydrogen{1}, give the type '{2}' the atomic "
               "number 1 in {9}",
               light.particles, light.particles == 1 ? "" : "s", entry.second,
               light.particles == 1 ? "has" : "have", range,
               light.particles == 1 ? "is" : "are", describe(light.first),
               light.particles == 1 ? "its" : "their",
               light.particles == 1 ? "it is a" : "they are",
               none ? std::string(
                          "an at.num column of [ atomtypes ]")
                    : where)
               .str()});
    }
    addRest("element_light_atom", lights.size(),
            "light particles of one bond that are not taken for hydrogens");
  }

  // The element that the file gives and the one whose symbol begins the
  // name of the type (its bonded type in GROMACS). Only elements given by
  // the file: the inferred ones come from the names already.
  std::map<std::string, std::pair<size_t, size_t>> names;
  for (size_t i = 0; i != count; ++i) {
    int number = numberOf(i);
    if (sourceOf(i) != ElementSource::File || number <= 0)
      continue;
    std::string label =
        hasTypes && origins.atomTypes[i] < origins.atomTypeLabels.size()
            ? origins.atomTypeLabels[origins.atomTypes[i]]
            : typeOf(i);
    std::vector<int> candidates = getNameElements(label);
    if (candidates.empty() || llvm::is_contained(candidates, number))
      continue;
    auto [entry, added] = names.insert({typeOf(i), {0, i}});
    ++entry->second.first;
  }
  {
    std::vector<std::pair<size_t, std::string>> order;
    for (auto &[type, entry] : names)
      order.push_back({entry.second, type});
    llvm::sort(order);
    for (auto [index, entry] : llvm::enumerate(order)) {
      if (index == listed)
        break;
      auto [particles, first] = names[entry.second];
      std::string label =
          hasTypes && origins.atomTypes[first] < origins.atomTypeLabels.size()
              ? origins.atomTypeLabels[origins.atomTypes[first]]
              : entry.second;
      std::string symbolsOfName;
      for (int candidate : getNameElements(label))
        symbolsOfName += (symbolsOfName.empty() ? "" : " or ") +
                         getElementSymbol(candidate);
      int number = numberOf(first);
      report.warnings.push_back(
          {"element_type_name",
           llvm::formatv(
               "elements: the file gives the type '{0}' ({1} particle{2}) "
               "the atomic number {3} ({4}), but the name '{5}' begins with "
               "the symbol of {6}; if the name is right, correct the atomic "
               "number in {7}",
               entry.second, particles, particles == 1 ? "" : "s", number,
               getElementSymbol(number), label, symbolsOfName, where)
               .str()});
    }
    addRest("element_type_name", names.size(),
            "names that disagree with their elements");
  }
  return report;
}

std::string mdir::tool::describeElementCounts(const ElementReport &report) {
  // By element, whatever the source, in the order of the atomic numbers,
  // and those without an element last.
  std::map<int, size_t> byNumber;
  for (const ElementReport::Count &count : report.counts)
    byNumber[count.number <= 0 ? 1000 - count.number : count.number] +=
        count.count;
  std::string text;
  for (auto [key, n] : byNumber) {
    int number = key >= 1000 ? 1000 - key : key;
    text += (text.empty() ? "" : ", ") + getElementSymbol(number) + " " +
            std::to_string(n);
  }
  return text;
}

std::string mdir::tool::describeElementSources(const ElementReport &report) {
  return llvm::formatv("{0} from the file, {1} from a fallback ({2} from the "
                       "name and the mass, {3} from the mass, {4} none in "
                       "the file)",
                       report.sources[0], report.getFallbacks(),
                       report.sources[1], report.sources[2],
                       report.sources[3])
      .str();
}

std::string mdir::tool::describeConstraintKinds(const ElementReport &report) {
  return llvm::formatv("{0} bonds of hydrogen in {1} groups, {2} rigid "
                       "waters",
                       report.hydrogenBonds, report.hydrogenGroups,
                       report.rigidWaters)
      .str();
}

llvm::json::Object mdir::tool::makeElementsJSON(const ElementReport &report) {
  llvm::json::Array counts;
  for (const ElementReport::Count &count : report.counts)
    counts.push_back(llvm::json::Object{
        {"element", getElementSymbol(count.number)},
        {"atomic_number",
         count.number >= 0 ? llvm::json::Value(count.number)
                           : llvm::json::Value(nullptr)},
        {"source", getElementSourceKey(count.source)},
        {"particles", count.count}});
  llvm::json::Object sources;
  for (unsigned k = 0; k != 4; ++k)
    sources[getElementSourceKey(static_cast<ElementSource>(k))] =
        report.sources[k];
  return llvm::json::Object{{"counts", std::move(counts)},
                            {"sources", std::move(sources)},
                            {"fallback", report.getFallbacks()}};
}
