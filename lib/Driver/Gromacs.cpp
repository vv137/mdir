// Readers of the formats of GROMACS: topologies (.top, .itp) and
// coordinates (.gro).
//
// The readers are written from a specification of the formats, with no
// code of GROMACS. The terms are converted to the forms of
// docs/conventions.md, which are those of GROMACS for the terms that M1
// supports: harmonic bonds and angles with ½ k, periodic dihedrals, and
// Lennard-Jones in σ and ε.

#include "mdir/Driver/Cell.h"
#include "mdir/Driver/Topology.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>

using namespace mdir::driver;
using llvm::StringRef;

namespace {

constexpr double radiansPerDegree = M_PI / 180.0;
constexpr double kjPerKcal = 4.184;

std::string show(double value) {
  char text[32];
  std::snprintf(text, sizeof(text), "%g", value);
  return text;
}

//===----------------------------------------------------------------------===//
// Preprocessor
//===----------------------------------------------------------------------===//

/// A line of the topology after the preprocessor, with where it came from.
struct Line {
  std::string text;
  std::string file;
  unsigned number;
  /// Whether `_FF_AMBER_LEAP_ATOM_REORDERING` is defined here: the force
  /// field asks for the atoms of dihedrals in the order of LEaP.
  bool leapOrder = false;
};

class Preprocessor {
public:
  Preprocessor(std::vector<std::string> includePath,
               llvm::ArrayRef<std::string> defines)
      : includePath(std::move(includePath)) {
    // NAME or NAME=VALUE, as -D of grompp.
    for (const std::string &define : defines) {
      auto [name, value] = StringRef(define).split('=');
      macros.push_back({name.str(), value.str()});
    }
  }

  /// Reads `path` and the files that it includes into `lines`.
  llvm::Error run(StringRef path, std::vector<Line> &lines);
  std::vector<std::string> sourceFiles;

private:
  llvm::Error readFile(StringRef path, std::vector<Line> &lines);
  std::string substitute(std::string line) const;
  llvm::Expected<std::string> resolve(StringRef name, StringRef from);

  std::vector<std::string> includePath;
  /// Macros in the order of their definition, which substitution follows.
  std::vector<std::pair<std::string, std::string>> macros;
  unsigned depth = 0;
};

} // namespace

static llvm::Error fail(StringRef file, unsigned line,
                        const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), "%s:%u: %s",
                                 file.str().c_str(), line,
                                 message.str().c_str());
}

static bool isWordCharacter(char c) {
  return llvm::isAlnum(c) || c == '_';
}

std::string Preprocessor::substitute(std::string line) const {
  // Whole words, one macro after another in the order of definition; a
  // macro with no value is not substituted.
  for (auto &[name, value] : macros) {
    if (value.empty())
      continue;
    std::string result;
    size_t start = 0;
    while (true) {
      size_t found = line.find(name, start);
      if (found == std::string::npos)
        break;
      size_t end = found + name.size();
      bool before = found == 0 || !isWordCharacter(line[found - 1]);
      bool after = end == line.size() || !isWordCharacter(line[end]);
      if (before && after) {
        result += line.substr(start, found - start) + value;
      } else {
        result += line.substr(start, end - start);
      }
      start = end;
    }
    result += line.substr(start);
    line = std::move(result);
  }
  return line;
}

llvm::Expected<std::string> Preprocessor::resolve(StringRef name,
                                                  StringRef from) {
  auto candidate = [](StringRef directory, StringRef file) {
    llvm::SmallString<256> path(directory);
    llvm::sys::path::append(path, file);
    return std::string(path);
  };
  if (llvm::sys::path::is_absolute(name))
    return llvm::sys::fs::exists(name) ? name.str() : std::string();
  // The directory of the file that includes, then the include path, then
  // the directories of GMXLIB.
  std::vector<std::string> directories = {
      llvm::sys::path::parent_path(from).str()};
  directories.insert(directories.end(), includePath.begin(),
                     includePath.end());
  if (auto library = llvm::sys::Process::GetEnv("GMXLIB")) {
    llvm::SmallVector<StringRef> parts;
    StringRef(*library).split(parts, ':', -1, false);
    for (StringRef part : parts)
      directories.push_back(part.str());
  }
  for (const std::string &directory : directories) {
    std::string path = candidate(directory, name);
    if (llvm::sys::fs::exists(path))
      return path;
  }
  return std::string();
}

llvm::Error Preprocessor::readFile(StringRef path, std::vector<Line> &lines) {
  if (++depth > 64)
    return fail(path, 0, "the includes are nested more than 64 deep");
  auto file = llvm::MemoryBuffer::getFile(path);
  if (!file)
    return fail(path, 0, "cannot read the file: " + file.getError().message());

  if (!llvm::is_contained(sourceFiles, path.str()))
    sourceFiles.push_back(path.str());

  // The state of each open conditional of this file: active, inactive, or
  // inside an inactive one.
  enum State { Active, Inactive, Ignored };
  std::vector<State> conditionals;
  auto isActive = [&]() {
    return conditionals.empty() || conditionals.back() == Active;
  };

  llvm::SmallVector<StringRef> physical;
  (*file)->getBuffer().split(physical, '\n');
  for (auto [index, raw] : llvm::enumerate(physical)) {
    unsigned number = index + 1;
    // The line ends at the first carriage return.
    StringRef text = raw.take_until([](char c) { return c == '\r'; });
    StringRef trimmed = text.ltrim();
    if (trimmed.starts_with("#")) {
      StringRef rest = trimmed.drop_front().ltrim();
      StringRef directive = rest.take_until([](char c) {
        return llvm::isSpace(c);
      });
      StringRef argument = rest.drop_front(directive.size()).trim();
      if (directive == "ifdef" || directive == "ifndef") {
        if (!isActive()) {
          conditionals.push_back(Ignored);
          continue;
        }
        if (argument.empty())
          return fail(path, number, "#" + directive + " without a name");
        // The name is the whole rest of the line.
        bool defined = llvm::any_of(
            macros, [&](auto &macro) { return macro.first == argument; });
        bool active = directive == "ifdef" ? defined : !defined;
        conditionals.push_back(active ? Active : Inactive);
        continue;
      }
      if (directive == "else") {
        if (conditionals.empty())
          return fail(path, number, "#else without #ifdef in this file");
        if (conditionals.back() != Ignored)
          conditionals.back() =
              conditionals.back() == Active ? Inactive : Active;
        continue;
      }
      if (directive == "endif") {
        if (conditionals.empty())
          return fail(path, number, "#endif without #ifdef in this file");
        conditionals.pop_back();
        continue;
      }
      if (!isActive())
        continue;
      if (directive == "include") {
        if (argument.empty() ||
            (!argument.starts_with("\"") && !argument.starts_with("<")))
          return fail(path, number,
                      "expected a file name in \"\" or <> after #include");
        StringRef name = argument.drop_front().take_until(
            [](char c) { return c == '"' || c == '<' || c == '>'; });
        auto resolved = resolve(name, path);
        if (!resolved)
          return resolved.takeError();
        if (resolved->empty())
          return fail(path, number,
                      "the included file '" + name +
                          "' is not found next to this file, in the include "
                          "path, or in GMXLIB");
        if (llvm::Error error = readFile(*resolved, lines))
          return error;
        continue;
      }
      if (directive == "define") {
        if (argument.empty())
          return fail(path, number, "#define without a name");
        StringRef name =
            argument.take_until([](char c) { return llvm::isSpace(c); });
        std::string value = argument.drop_front(name.size()).trim().str();
        auto found = llvm::find_if(
            macros, [&](auto &macro) { return macro.first == name; });
        if (found != macros.end())
          found->second = value;
        else
          macros.push_back({name.str(), value});
        continue;
      }
      if (directive == "undef") {
        if (argument.empty())
          return fail(path, number, "#undef without a name");
        llvm::erase_if(macros,
                       [&](auto &macro) { return macro.first == argument; });
        continue;
      }
      return fail(path, number, "an unknown directive #" + directive);
    }
    if (!isActive())
      continue;
    bool leapOrder = llvm::any_of(macros, [](auto &macro) {
      return macro.first == "_FF_AMBER_LEAP_ATOM_REORDERING";
    });
    lines.push_back({substitute(text.str()), path.str(), number, leapOrder});
  }
  --depth;
  return llvm::Error::success();
}

llvm::Error Preprocessor::run(StringRef path, std::vector<Line> &lines) {
  std::vector<Line> physical;
  if (llvm::Error error = readFile(path, physical))
    return error;

  // Join the lines that end in a backslash, then strip the comments.
  for (size_t i = 0; i < physical.size(); ++i) {
    Line line = physical[i];
    while (true) {
      StringRef text = StringRef(line.text).rtrim(" \t");
      if (!text.ends_with("\\") || i + 1 == physical.size())
        break;
      line.text = text.drop_back().str() + " " + physical[++i].text;
    }
    std::string text = line.text;
    size_t comment = text.find(';');
    if (comment != std::string::npos)
      text.erase(comment);
    line.text = StringRef(text).trim().str();
    if (!line.text.empty())
      lines.push_back(std::move(line));
  }
  return llvm::Error::success();
}

//===----------------------------------------------------------------------===//
// The topology
//===----------------------------------------------------------------------===//

namespace {

/// A type of `[ atomtypes ]`.
struct AtomType {
  std::string name;
  std::string bondedType;
  int atomicNumber = -1;
  double mass = 0.0, charge = 0.0;
  /// The particle type: A for an atom, V or D for a virtual site, S for a
  /// shell.
  char particle = 'A';
  /// σ and ε, or c6 and c12 with the combination rule 1.
  double v = 0.0, w = 0.0;
};

/// Parameters of bonded types, with the types that they are for.
struct BondedType {
  std::vector<std::string> types;
  int function;
  std::vector<double> parameters;
};

struct Atom {
  std::string type;
  std::string name;
  std::string residue;
  std::string residueNumber;
  double charge, mass;
  /// Of the particle type V or D: a virtual site.
  bool virtualSite = false;
  const Line *line = nullptr;
};

struct Interaction {
  std::vector<unsigned> atoms;
  /// The atoms in the order of the file, by which the types are looked up,
  /// when the order of LEaP has changed `atoms`; empty otherwise.
  std::vector<unsigned> written;
  int function;
  std::vector<double> parameters;
  const Line *line;
};

struct MoleculeType {
  std::string name;
  int nrexcl = 0;
  std::vector<Atom> atoms;
  std::vector<Interaction> bonds, pairs, angles, dihedrals, settles,
      virtualSites, cmaps;
  std::vector<std::pair<unsigned, unsigned>> exclusions;
};

class TopologyReader {
public:
  TopologyReader(StringRef path, std::vector<std::string> include,
                 llvm::ArrayRef<std::string> defines)
      : path(path), preprocessor(std::move(include), defines) {}

  llvm::Expected<Topology> read();

private:
  llvm::Error fail(const Line &line, const llvm::Twine &message) {
    return ::fail(line.file, line.number, message);
  }

  llvm::Error readLine(const Line &line, StringRef section);
  llvm::Error readDefaults(const Line &line,
                           llvm::ArrayRef<std::string> tokens);
  llvm::Error readAtomType(const Line &line,
                           llvm::ArrayRef<std::string> tokens);
  llvm::Error readBondedType(const Line &line,
                             llvm::ArrayRef<std::string> tokens,
                             unsigned count, StringRef section);
  llvm::Error readDihedralType(const Line &line,
                               llvm::ArrayRef<std::string> tokens);
  llvm::Error readPairType(const Line &line,
                           llvm::ArrayRef<std::string> tokens,
                           StringRef section);
  llvm::Error readAtom(const Line &line, llvm::ArrayRef<std::string> tokens);
  llvm::Error readInteraction(const Line &line,
                              llvm::ArrayRef<std::string> tokens,
                              unsigned count, std::vector<Interaction> &into,
                              StringRef section);
  llvm::Error readExclusions(const Line &line,
                             llvm::ArrayRef<std::string> tokens);

  llvm::Error build(Topology &topology);
  llvm::Error expandMolecule(const MoleculeType &type, Topology &topology,
                             unsigned offset);
  /// The entry of `[ dihedraltypes ]` for a dihedral, in the order of the
  /// file: the first with the most types that are not wildcards, in either
  /// direction. For functions 1 and 9, the first term of a block.
  struct DihedralMatch {
    const std::vector<BondedType> *list = nullptr;
    size_t index = 0;
    /// The entry matches the atoms in the reverse order.
    bool reversed = false;
  };
  DihedralMatch matchDihedral(const MoleculeType &molecule,
                              const Interaction &dihedral) const;
  /// Puts the atoms of a dihedral in the order that LEaP gives them.
  void orderLikeLeap(const MoleculeType &molecule, Interaction &dihedral);
  const AtomType *findType(StringRef name) const;
  /// The σ and ε of a pair of types, from the rule and the overrides.
  std::pair<double, double> getPair(unsigned a, unsigned b) const;
  /// Converts V and W of the file to σ and ε.
  llvm::Error toSigmaEpsilon(const Line *line, double v, double w,
                             double &sigma, double &epsilon) const;

  std::string path;
  Preprocessor preprocessor;
  std::vector<Line> lines;

  // [ defaults ]
  bool hasDefaults = false;
  int combinationRule = 2;
  bool generatePairs = false;
  double fudgeLJ = 1.0, fudgeQQ = 1.0;

  std::vector<AtomType> atomTypes;
  std::map<std::string, unsigned> atomTypeIndex;
  std::map<std::string, std::vector<BondedType>> bondedTypes;
  /// [ cmaptypes ]: the five types of each map and its grid, in kJ/mol.
  struct CMapType {
    std::vector<std::string> types;
    std::vector<double> grid;
    const Line *line;
  };
  std::vector<CMapType> cmapTypes;
  unsigned cmapResolution = 0;
  /// The place in the topology of each map of `cmapTypes` that is used.
  std::map<unsigned, unsigned> usedCMaps;
  /// Overrides of the nonbonded pairs, and the pair types, by the two
  /// indices of the types, the lower first.
  std::map<std::pair<unsigned, unsigned>, std::pair<double, double>>
      nonbondParams, pairTypes;
  std::vector<MoleculeType> moleculeTypes;
  std::vector<std::pair<unsigned, long>> molecules;
  bool sawMoleculeType = false;
  /// The types and the atoms of the impropers that kept the order of the
  /// file under the order of LEaP, by function, over all molecule types.
  std::map<int, std::set<std::array<std::string, 4>>> leapImproperTypes;
  std::map<int, std::set<std::array<unsigned, 4>>> leapImproperAtoms;
};

} // namespace

/// Splits a line on blanks.
static std::vector<std::string> tokenize(StringRef text) {
  llvm::SmallVector<StringRef> parts;
  text.split(parts, ' ', -1, false);
  std::vector<std::string> tokens;
  for (StringRef part : parts) {
    llvm::SmallVector<StringRef> pieces;
    part.split(pieces, '\t', -1, false);
    for (StringRef piece : pieces)
      tokens.push_back(piece.str());
  }
  return tokens;
}

static bool readReal(StringRef token, double &value) {
  return !token.trim().getAsDouble(value);
}

static bool readInteger(StringRef token, long &value) {
  return !token.trim().getAsInteger(10, value);
}

/// The canonical name of a section: lower case, without `-` and `_`.
static std::string canonicalSection(StringRef name) {
  std::string result;
  for (char c : name.trim())
    if (c != '-' && c != '_')
      result += llvm::toLower(c);
  return result;
}

const AtomType *TopologyReader::findType(StringRef name) const {
  auto found = atomTypeIndex.find(name.str());
  return found == atomTypeIndex.end() ? nullptr : &atomTypes[found->second];
}

llvm::Error TopologyReader::toSigmaEpsilon(const Line *line, double v,
                                           double w, double &sigma,
                                           double &epsilon) const {
  if (combinationRule != 1) {
    if (v < 0.0)
      return line ? ::fail(line->file, line->number,
                           "a negative sigma (a type without dispersion) is "
                           "not supported")
                  : llvm::createStringError(llvm::inconvertibleErrorCode(),
                                            "a negative sigma is not "
                                            "supported");
    sigma = v;
    epsilon = w;
    return llvm::Error::success();
  }
  // c6 = 4 ε σ⁶, c12 = 4 ε σ¹².
  if (v == 0.0 && w == 0.0) {
    sigma = epsilon = 0.0;
    return llvm::Error::success();
  }
  if (v <= 0.0 || w <= 0.0) {
    std::string message = "c6 = " + show(v) + " and c12 = " + show(w) +
                          " are not a Lennard-Jones pair";
    return line ? ::fail(line->file, line->number, message)
                : llvm::createStringError(llvm::inconvertibleErrorCode(),
                                          "%s", message.c_str());
  }
  sigma = std::pow(w / v, 1.0 / 6.0);
  epsilon = v * v / (4.0 * w);
  return llvm::Error::success();
}

//===----------------------------------------------------------------------===//
// Sections
//===----------------------------------------------------------------------===//

llvm::Error TopologyReader::readDefaults(const Line &line,
                                         llvm::ArrayRef<std::string> t) {
  if (hasDefaults)
    return fail(line, "a second line of [ defaults ]");
  hasDefaults = true;
  if (t.size() < 2)
    return fail(line, "expected nbfunc and comb-rule in [ defaults ]");
  StringRef function = t[0];
  if (function != "1" && !function.equals_insensitive("LJ"))
    return fail(line, "only the Lennard-Jones potential (nbfunc 1) is "
                      "supported, not '" +
                          function + "'");
  StringRef rule = t[1];
  if (rule == "1" || rule.equals_insensitive("Geometric"))
    combinationRule = 1;
  else if (rule == "2" || rule.equals_insensitive("Arithmetic"))
    combinationRule = 2;
  else if (rule == "3" || rule.equals_insensitive("GeomSigEps"))
    combinationRule = 3;
  else
    return fail(line, "unknown comb-rule '" + rule + "'");
  if (t.size() > 2)
    generatePairs = !t[2].empty() && llvm::toLower(t[2][0]) == 'y';
  if (t.size() > 3 && !readReal(t[3], fudgeLJ))
    return fail(line, "cannot read fudgeLJ '" + t[3] + "'");
  if (t.size() > 4 && !readReal(t[4], fudgeQQ))
    return fail(line, "cannot read fudgeQQ '" + t[4] + "'");
  double power = 12.0;
  if (t.size() > 5 && (!readReal(t[5], power) || power != 12.0))
    return fail(line, "only the repulsion exponent 12 is supported");
  return llvm::Error::success();
}

llvm::Error TopologyReader::readAtomType(const Line &line,
                                         llvm::ArrayRef<std::string> t) {
  if (!hasDefaults)
    return fail(line, "[ atomtypes ] before [ defaults ]");
  if (t.size() < 6)
    return fail(line, "too few fields in [ atomtypes ]");
  auto isLetter = [](const std::string &token) {
    return token.size() == 1 && llvm::isAlpha(token[0]);
  };
  // Which of the bonded type and the atomic number are there, from the
  // place of the particle type.
  AtomType type;
  type.name = t[0];
  size_t next = 1;
  if (t.size() > 5 && isLetter(t[5])) {
    type.bondedType = t[1];
    long number;
    if (!readInteger(t[2], number))
      return fail(line, "cannot read the atomic number '" + t[2] + "'");
    type.atomicNumber = number;
    next = 3;
  } else if (isLetter(t[3])) {
    next = 1;
  } else {
    long number;
    if (readInteger(t[1], number))
      type.atomicNumber = number;
    else
      type.bondedType = t[1];
    next = 2;
  }
  if (type.bondedType.empty())
    type.bondedType = type.name;
  if (t.size() < next + 5)
    return fail(line, "too few fields in [ atomtypes ]");
  if (!readReal(t[next], type.mass) || !readReal(t[next + 1], type.charge))
    return fail(line, "cannot read the mass and the charge in [ atomtypes ]");
  StringRef particle = t[next + 2];
  if (particle.size() != 1 ||
      !llvm::is_contained(StringRef("ANSBVD"), llvm::toUpper(particle[0])))
    return fail(line, "unknown particle type '" + particle + "'");
  type.particle = llvm::toUpper(particle[0]);
  if (!readReal(t[next + 3], type.v) || !readReal(t[next + 4], type.w))
    return fail(line, "cannot read the Lennard-Jones parameters in "
                      "[ atomtypes ]");
  auto found = atomTypeIndex.find(type.name);
  if (found != atomTypeIndex.end()) {
    atomTypes[found->second] = type;
    return llvm::Error::success();
  }
  atomTypeIndex[type.name] = atomTypes.size();
  atomTypes.push_back(type);
  return llvm::Error::success();
}

llvm::Error TopologyReader::readBondedType(const Line &line,
                                           llvm::ArrayRef<std::string> t,
                                           unsigned count,
                                           StringRef section) {
  if (t.size() < count + 1)
    return fail(line, "too few fields in [ " + section + " ]");
  long function;
  if (!readInteger(t[count], function))
    return fail(line, "cannot read the function '" + t[count] + "'");
  BondedType entry;
  entry.types.assign(t.begin(), t.begin() + count);
  entry.function = function;
  for (size_t k = count + 1; k < t.size(); ++k) {
    double value;
    if (!readReal(t[k], value))
      return fail(line, "cannot read the parameter '" + t[k] + "'");
    entry.parameters.push_back(value);
  }
  std::string key = section.str() + ":" + std::to_string(function);
  std::vector<BondedType> &list = bondedTypes[key];
  // The last definition wins.
  auto same = [&](const BondedType &other) {
    std::vector<std::string> reversed(entry.types.rbegin(),
                                      entry.types.rend());
    return other.types == entry.types || other.types == reversed;
  };
  auto found = llvm::find_if(list, same);
  if (found != list.end())
    *found = entry;
  else
    list.push_back(entry);
  return llvm::Error::success();
}

llvm::Error TopologyReader::readDihedralType(const Line &line,
                                             llvm::ArrayRef<std::string> t) {
  auto isDigit = [](const std::string &token) {
    return token.size() == 1 && llvm::isDigit(token[0]);
  };
  BondedType entry;
  size_t first;
  if (t.size() >= 3 && isDigit(t[2])) {
    entry.function = t[2][0] - '0';
    // Two types: the outer atoms for function 2, the middle ones otherwise.
    if (entry.function == 2)
      entry.types = {t[0], "X", "X", t[1]};
    else
      entry.types = {"X", t[0], t[1], "X"};
    first = 3;
  } else if (t.size() >= 5 && isDigit(t[4])) {
    entry.function = t[4][0] - '0';
    entry.types.assign(t.begin(), t.begin() + 4);
    first = 5;
  } else {
    return fail(line, "expected two or four types in [ dihedraltypes ]");
  }
  for (size_t k = first; k < t.size(); ++k) {
    double value;
    if (!readReal(t[k], value))
      return fail(line, "cannot read the parameter '" + t[k] + "'");
    entry.parameters.push_back(value);
  }
  // Functions 1 and 9 share a table; adjacent lines of the same types make
  // the terms of one multi-term dihedral.
  int table = entry.function == 9 ? 1 : entry.function;
  std::vector<BondedType> &list =
      bondedTypes["dihedraltypes:" + std::to_string(table)];
  if (entry.function == 9 && !list.empty() && list.back().types == entry.types) {
    list.push_back(entry);
    return llvm::Error::success();
  }
  auto found = llvm::find_if(list, [&](const BondedType &other) {
    std::vector<std::string> reversed(entry.types.rbegin(),
                                      entry.types.rend());
    return other.types == entry.types || other.types == reversed;
  });
  if (found != list.end()) {
    if (entry.function == 9 && found->parameters != entry.parameters)
      return fail(line, "a second block of parameters for the dihedral type "
                        "of these types");
    // Replace the whole block by this entry.
    size_t start = found - list.begin();
    size_t end = start + 1;
    while (end < list.size() && list[end].types == found->types)
      ++end;
    list.erase(list.begin() + start, list.begin() + end);
    list.insert(list.begin() + start, entry);
    return llvm::Error::success();
  }
  list.push_back(entry);
  return llvm::Error::success();
}

llvm::Error TopologyReader::readPairType(const Line &line,
                                         llvm::ArrayRef<std::string> t,
                                         StringRef section) {
  if (t.size() < 5)
    return fail(line, "too few fields in [ " + section + " ]");
  const AtomType *a = findType(t[0]), *b = findType(t[1]);
  if (!a || !b)
    return fail(line, "unknown atom type '" + (a ? t[1] : t[0]) + "'");
  long function;
  if (!readInteger(t[2], function))
    return fail(line, "cannot read the function '" + t[2] + "'");
  if (section == "nonbond_params" && function != 1)
    return fail(line, "only Lennard-Jones (function 1) is supported in "
                      "[ nonbond_params ]");
  double v, w;
  if (!readReal(t[3], v) || !readReal(t[4], w))
    return fail(line, "cannot read the parameters in [ " + section + " ]");
  unsigned i = atomTypeIndex[t[0]], j = atomTypeIndex[t[1]];
  auto key = std::make_pair(std::min(i, j), std::max(i, j));
  (section == "nonbond_params" ? nonbondParams : pairTypes)[key] = {v, w};
  return llvm::Error::success();
}

llvm::Error TopologyReader::readAtom(const Line &line,
                                     llvm::ArrayRef<std::string> t) {
  if (moleculeTypes.empty())
    return fail(line, "[ atoms ] before [ moleculetype ]");
  MoleculeType &molecule = moleculeTypes.back();
  if (t.size() < 6)
    return fail(line, "too few fields in [ atoms ]");
  long number;
  if (!readInteger(t[0], number) ||
      number != static_cast<long>(molecule.atoms.size()) + 1)
    return fail(line, "the atoms are not numbered from 1 without gaps");
  const AtomType *type = findType(t[1]);
  if (!type)
    return fail(line, "unknown atom type '" + t[1] + "'");
  // A force field defines types for shells that a system may not use.
  if (type->particle != 'A' && type->particle != 'V' && type->particle != 'D')
    return fail(line, "the atom type '" + t[1] + "' is of the particle type " +
                          llvm::Twine(type->particle) +
                          "; only atoms and virtual sites are supported");
  Atom atom;
  atom.virtualSite = type->particle != 'A';
  atom.line = &line;
  atom.type = t[1];
  atom.residueNumber = t[2];
  atom.residue = t[3];
  atom.name = t[4];
  atom.charge = type->charge;
  atom.mass = type->mass;
  if (t.size() > 6 && !readReal(t[6], atom.charge))
    atom.charge = type->charge;
  if (t.size() > 7 && !readReal(t[7], atom.mass))
    atom.mass = type->mass;
  if (t.size() > 8 && t[8] != t[1])
    return fail(line, "a topology of the state B (free energy) is not "
                      "supported");
  if (atom.virtualSite && atom.mass != 0.0)
    return fail(line, "the virtual site '" + atom.name + "' has a mass of " +
                          show(atom.mass) + "; a virtual site has none");
  molecule.atoms.push_back(atom);
  return llvm::Error::success();
}

llvm::Error TopologyReader::readInteraction(const Line &line,
                                            llvm::ArrayRef<std::string> t,
                                            unsigned count,
                                            std::vector<Interaction> &into,
                                            StringRef section) {
  if (moleculeTypes.empty())
    return fail(line, "[ " + section + " ] before [ moleculetype ]");
  MoleculeType &molecule = moleculeTypes.back();
  if (t.size() < count)
    return fail(line, "too few fields in [ " + section + " ]");
  Interaction interaction;
  interaction.line = &line;
  for (unsigned k = 0; k != count; ++k) {
    long index;
    if (!readInteger(t[k], index) || index < 1 ||
        index > static_cast<long>(molecule.atoms.size()))
      return fail(line, "the atom index '" + t[k] + "' in [ " + section +
                            " ] is out of range");
    if (llvm::is_contained(interaction.atoms, index - 1))
      return fail(line, "an atom appears twice in [ " + section + " ]");
    interaction.atoms.push_back(index - 1);
  }
  long function = 1;
  size_t next = count;
  if (t.size() > count) {
    if (readInteger(t[count], function))
      next = count + 1;
    else
      function = 1;
  }
  interaction.function = function;
  for (size_t k = next; k < t.size(); ++k) {
    double value;
    if (!readReal(t[k], value))
      break;
    interaction.parameters.push_back(value);
  }
  into.push_back(std::move(interaction));
  return llvm::Error::success();
}

llvm::Error TopologyReader::readExclusions(const Line &line,
                                           llvm::ArrayRef<std::string> t) {
  if (moleculeTypes.empty())
    return fail(line, "[ exclusions ] before [ moleculetype ]");
  MoleculeType &molecule = moleculeTypes.back();
  long first;
  if (t.empty() || !readInteger(t[0], first) || first < 1 ||
      first > static_cast<long>(molecule.atoms.size()))
    return llvm::Error::success();
  for (size_t k = 1; k < t.size(); ++k) {
    long other;
    if (!readInteger(t[k], other))
      break;
    if (other < 1 || other > static_cast<long>(molecule.atoms.size()))
      return fail(line, "the atom index " + t[k] + " in [ exclusions ] is "
                                                   "out of range");
    if (other != first)
      molecule.exclusions.push_back(
          {std::min(first, other) - 1, std::max(first, other) - 1});
  }
  return llvm::Error::success();
}

llvm::Error TopologyReader::readLine(const Line &line, StringRef section) {
  std::vector<std::string> t = tokenize(line.text);
  if (section == "defaults")
    return readDefaults(line, t);
  if (section == "atomtypes")
    return readAtomType(line, t);
  if (sawMoleculeType && (section == "bondtypes" || section == "angletypes" ||
                          section == "dihedraltypes" ||
                          section == "pairtypes" ||
                          section == "nonbondparams"))
    return fail(line, "[ " + section + " ] after a [ moleculetype ]");
  if (section == "bondtypes")
    return readBondedType(line, t, 2, "bondtypes");
  // Parameters that a system uses only with constraints, which come later,
  // and parameters of implicit solvent, which M1 does not have.
  if (section == "constrainttypes")
    return readBondedType(line, t, 2, "constrainttypes");
  if (section == "implicitgenbornparams" ||
      section == "implicitsurfaceparams")
    return llvm::Error::success();
  if (section == "angletypes")
    return readBondedType(line, t, 3, "angletypes");
  if (section == "dihedraltypes")
    return readDihedralType(line, t);
  if (section == "pairtypes")
    return readPairType(line, t, "pairtypes");
  if (section == "nonbondparams")
    return readPairType(line, t, "nonbond_params");
  if (section == "moleculetype") {
    sawMoleculeType = true;
    if (t.size() < 2)
      return fail(line, "expected a name and nrexcl in [ moleculetype ]");
    MoleculeType molecule;
    molecule.name = t[0];
    long nrexcl;
    if (!readInteger(t[1], nrexcl) || nrexcl < 0)
      return fail(line, "cannot read nrexcl '" + t[1] + "'");
    molecule.nrexcl = nrexcl;
    for (const MoleculeType &other : moleculeTypes)
      if (other.name == molecule.name)
        return fail(line, "the molecule type '" + molecule.name +
                              "' is defined twice");
    moleculeTypes.push_back(std::move(molecule));
    return llvm::Error::success();
  }
  if (section == "atoms")
    return readAtom(line, t);
  if (moleculeTypes.empty() &&
      llvm::is_contained({"bonds", "pairs", "angles", "dihedrals",
                          "exclusions", "settles", "virtualsites3",
                          "cmap"},
                         section))
    return fail(line, "[ " + section + " ] before [ moleculetype ]");
  if (section == "bonds")
    return readInteraction(line, t, 2, moleculeTypes.back().bonds, "bonds");
  if (section == "pairs")
    return readInteraction(line, t, 2, moleculeTypes.back().pairs, "pairs");
  if (section == "angles")
    return readInteraction(line, t, 3, moleculeTypes.back().angles, "angles");
  if (section == "dihedrals") {
    MoleculeType &molecule = moleculeTypes.back();
    if (llvm::Error error =
            readInteraction(line, t, 4, molecule.dihedrals, "dihedrals"))
      return error;
    if (line.leapOrder)
      orderLikeLeap(molecule, molecule.dihedrals.back());
    return llvm::Error::success();
  }
  if (section == "exclusions")
    return readExclusions(line, t);
  if (section == "cmaptypes") {
    // Five bonded types, a function, the numbers of points along φ and ψ,
    // and the grid, φ the slower index.
    long function, nx, ny;
    if (t.size() < 8 || !readInteger(t[5], function) ||
        !readInteger(t[6], nx) || !readInteger(t[7], ny))
      return fail(line, "expected five types, a function, and two numbers "
                        "of points in [ cmaptypes ]");
    if (function != 1)
      return fail(line, "only the function 1 of [ cmaptypes ] is supported");
    if (nx != ny || nx < 4 || nx % 2 != 0)
      return fail(line, "a map of " + t[6] + " by " + t[7] +
                            " points is not supported; the numbers must be "
                            "equal and even");
    if (cmapResolution != 0 && nx != static_cast<long>(cmapResolution))
      return fail(line, "the maps have different numbers of points");
    cmapResolution = nx;
    CMapType type;
    type.types.assign(t.begin(), t.begin() + 5);
    type.line = &line;
    for (size_t k = 8; k < t.size(); ++k) {
      double value;
      if (!readReal(t[k], value))
        return fail(line, "cannot read the value '" + t[k] +
                              "' in [ cmaptypes ]");
      type.grid.push_back(value);
    }
    if (static_cast<long>(type.grid.size()) != nx * ny)
      return fail(line, "expected " + llvm::Twine(nx * ny) +
                            " values in [ cmaptypes ], got " +
                            llvm::Twine(type.grid.size()));
    cmapTypes.push_back(std::move(type));
    return llvm::Error::success();
  }
  if (section == "cmap") {
    MoleculeType &molecule = moleculeTypes.back();
    if (llvm::Error error =
            readInteraction(line, t, 5, molecule.cmaps, "cmap"))
      return error;
    if (molecule.cmaps.back().function != 1)
      return fail(line, "only the function 1 of [ cmap ] is supported");
    return llvm::Error::success();
  }
  if (section == "virtualsites3") {
    // The site, the three atoms it is built from, a function, and the
    // parameters of the function.
    MoleculeType &molecule = moleculeTypes.back();
    if (llvm::Error error = readInteraction(line, t, 4, molecule.virtualSites,
                                            "virtual_sites3"))
      return error;
    const Interaction &site = molecule.virtualSites.back();
    if (site.function != 1)
      return fail(line, "only the function 1 of [ virtual_sites3 ], a linear "
                        "combination of three atoms, is supported");
    if (site.parameters.size() != 2)
      return fail(line, "expected the two weights a and b in "
                        "[ virtual_sites3 ]; weights that grompp would "
                        "compute from the constraints are not supported");
    // The particle that the section builds is a virtual site whatever the
    // particle type of its atom type says, as grompp takes it: the OPC of
    // amber19sb.ff gives its site the type A. It has no mass.
    if (site.atoms[0] < molecule.atoms.size()) {
      Atom &built = molecule.atoms[site.atoms[0]];
      if (!built.virtualSite && built.mass != 0.0)
        return fail(line, "the virtual site '" + built.name + "' has a mass "
                          "of " + show(built.mass) + "; a virtual site has "
                          "none");
      built.virtualSite = true;
    }
    return llvm::Error::success();
  }
  if (section == "settles") {
    // The oxygen, a function, and the distances O–H and H–H.
    MoleculeType &molecule = moleculeTypes.back();
    long oxygen, function;
    double oh, hh;
    if (t.size() < 4 || !readInteger(t[0], oxygen) ||
        !readInteger(t[1], function) || !readReal(t[2], oh) ||
        !readReal(t[3], hh))
      return fail(line, "expected an atom, a function, and two distances in "
                        "[ settles ]");
    if (oxygen < 1 || oxygen + 2 > static_cast<long>(molecule.atoms.size()))
      return fail(line, "the atoms of [ settles ] are out of range");
    Interaction settle;
    settle.atoms = {static_cast<unsigned>(oxygen - 1)};
    settle.function = function;
    settle.parameters = {oh, hh};
    settle.line = &line;
    molecule.settles.push_back(std::move(settle));
    return llvm::Error::success();
  }
  if (section == "system")
    return llvm::Error::success();
  if (section == "molecules") {
    if (t.size() < 2)
      return fail(line, "expected a name and a count in [ molecules ]");
    long count;
    if (!readInteger(t[1], count) || count < 0)
      return fail(line, "cannot read the count '" + t[1] + "'");
    // Case-sensitive first, then without regard to case if that is unique.
    std::vector<unsigned> exact, loose;
    for (auto [index, molecule] : llvm::enumerate(moleculeTypes)) {
      if (molecule.name == t[0])
        exact.push_back(index);
      if (StringRef(molecule.name).equals_insensitive(t[0]))
        loose.push_back(index);
    }
    unsigned index;
    if (exact.size() == 1)
      index = exact.front();
    else if (loose.size() == 1)
      index = loose.front();
    else
      return fail(line, "no single molecule type is named '" + t[0] + "'");
    if (moleculeTypes[index].atoms.empty())
      return fail(line, "the molecule type '" + t[0] + "' has no atoms");
    if (count > 0)
      molecules.push_back({index, count});
    return llvm::Error::success();
  }
  return fail(line, "the section [ " + section + " ] is not supported");
}

//===----------------------------------------------------------------------===//
// The system
//===----------------------------------------------------------------------===//

std::pair<double, double> TopologyReader::getPair(unsigned a,
                                                  unsigned b) const {
  auto found = nonbondParams.find({std::min(a, b), std::max(a, b)});
  if (found != nonbondParams.end())
    return found->second;
  const AtomType &ta = atomTypes[a], &tb = atomTypes[b];
  switch (combinationRule) {
  case 1:
    return {std::sqrt(ta.v * tb.v), std::sqrt(ta.w * tb.w)};
  case 2:
    return {0.5 * (ta.v + tb.v), std::sqrt(ta.w * tb.w)};
  default:
    return {std::sqrt(ta.v * tb.v), std::sqrt(ta.w * tb.w)};
  }
}

/// The bonded types of `atoms` of `molecule`.
static std::vector<std::string>
getBondedTypes(const MoleculeType &molecule,
               const std::vector<AtomType> &types,
               const std::map<std::string, unsigned> &index,
               const std::vector<unsigned> &atoms) {
  std::vector<std::string> result;
  for (unsigned atom : atoms)
    result.push_back(
        types[index.at(molecule.atoms[atom].type)].bondedType);
  return result;
}

TopologyReader::DihedralMatch
TopologyReader::matchDihedral(const MoleculeType &molecule,
                              const Interaction &dihedral) const {
  DihedralMatch match;
  int table = dihedral.function == 9 ? 1 : dihedral.function;
  auto found = bondedTypes.find("dihedraltypes:" + std::to_string(table));
  if (found == bondedTypes.end())
    return match;
  const std::vector<BondedType> &list = found->second;
  std::vector<std::string> types = getBondedTypes(
      molecule, atomTypes, atomTypeIndex,
      dihedral.written.empty() ? dihedral.atoms : dihedral.written);
  std::vector<std::string> reversed(types.rbegin(), types.rend());
  int best = -1;
  for (size_t e = 0; e != list.size(); ++e) {
    for (bool backward : {false, true}) {
      const std::vector<std::string> &candidate = backward ? reversed : types;
      int exact = 0;
      bool matches = true;
      for (int k = 0; k != 4; ++k) {
        if (list[e].types[k] == "X")
          continue;
        if (list[e].types[k] != candidate[k]) {
          matches = false;
          break;
        }
        ++exact;
      }
      if (matches && exact > best) {
        best = exact;
        match.list = &list;
        match.index = e;
        match.reversed = backward;
      }
    }
    // Skip the other terms of a block.
    while (table == 1 && e + 1 < list.size() &&
           list[e + 1].types == list[e].types)
      ++e;
  }
  return match;
}

// The order of LEaP, as grompp gives it when the force field defines
// `_FF_AMBER_LEAP_ATOM_REORDERING`: with the types of the matching entry, a
// blank for a wildcard, a dihedral with a force constant is reversed when its
// first type sorts after its last, or the two are equal and the second sorts
// after the third. An improper whose types or atoms came before has the
// three atoms around the third, the central one, sorted by type and then by
// index; the first of its types keeps its order. Which atoms are the outer
// ones changes the angle and so the energy of an improper.
void TopologyReader::orderLikeLeap(const MoleculeType &molecule,
                                   Interaction &dihedral) {
  DihedralMatch match = matchDihedral(molecule, dihedral);
  if (!match.list)
    return;
  const BondedType &entry = (*match.list)[match.index];
  std::array<std::pair<std::string, unsigned>, 4> atoms;
  for (int k = 0; k != 4; ++k) {
    const std::string &type = entry.types[match.reversed ? 3 - k : k];
    atoms[k] = {type == "X" ? " " : type, dihedral.atoms[k]};
  }
  if (entry.parameters.size() > 1 && entry.parameters[1] != 0.0 &&
      (atoms[0].first > atoms[3].first ||
       (atoms[0].first == atoms[3].first && atoms[1].first > atoms[2].first)))
    std::reverse(atoms.begin(), atoms.end());
  if (dihedral.function == 2 || dihedral.function == 4) {
    std::array<std::string, 4> types;
    std::array<unsigned, 4> indices;
    for (int k = 0; k != 4; ++k)
      std::tie(types[k], indices[k]) = atoms[k];
    auto &seenTypes = leapImproperTypes[dihedral.function];
    auto &seenAtoms = leapImproperAtoms[dihedral.function];
    if (!seenTypes.count(types) && !seenAtoms.count(indices)) {
      seenTypes.insert(types);
      seenAtoms.insert(indices);
    } else {
      std::array<std::pair<std::string, unsigned>, 3> outer = {
          atoms[0], atoms[1], atoms[3]};
      std::sort(outer.begin(), outer.end());
      atoms[0] = outer[0];
      atoms[1] = outer[1];
      atoms[3] = outer[2];
    }
  }
  std::vector<unsigned> ordered;
  for (auto &[type, index] : atoms)
    ordered.push_back(index);
  if (ordered != dihedral.atoms) {
    dihedral.written = dihedral.atoms;
    dihedral.atoms = std::move(ordered);
  }
}

llvm::Error TopologyReader::expandMolecule(const MoleculeType &molecule,
                                           Topology &topology,
                                           unsigned offset) {
  auto bonded = [&](const Interaction &interaction) {
    return getBondedTypes(molecule, atomTypes, atomTypeIndex,
                          interaction.atoms);
  };
  auto lookup = [&](StringRef table,
                    const std::vector<std::string> &types)
      -> const BondedType * {
    auto found = bondedTypes.find(table.str());
    if (found == bondedTypes.end())
      return nullptr;
    std::vector<std::string> reversed(types.rbegin(), types.rend());
    for (const BondedType &entry : found->second)
      if (entry.types == types || entry.types == reversed)
        return &entry;
    return nullptr;
  };
  auto isHydrogen = [&](unsigned atom) {
    const AtomType *type = findType(molecule.atoms[atom].type);
    if (type->atomicNumber >= 0)
      return type->atomicNumber == 1;
    return molecule.atoms[atom].mass < 1.2;
  };

  // The terms that MDIR takes as terms given by expressions (D136), in the
  // units of the control file: Å, radians, kcal/mol. One for each function.
  auto termOf = [&](StringRef name, StringRef expression, unsigned arity,
                    std::initializer_list<const char *> parameters)
      -> TupleTerm & {
    for (TupleTerm &term : topology.tupleTerms)
      if (term.name == name)
        return term;
    TupleTerm term;
    term.name = name.str();
    term.expression = expression.str();
    term.arity = arity;
    for (const char *parameter : parameters)
      term.parameters.push_back({parameter, {}});
    topology.tupleTerms.push_back(std::move(term));
    return topology.tupleTerms.back();
  };

  // Bonds: function 1, ½ k (r − b0)², with b0 then k; function 2, the
  // quartic bond of GROMOS, ¼ k (r² − b0²)², a term given by an expression.
  for (const Interaction &bond : molecule.bonds) {
    if (bond.function != 1 && bond.function != 2)
      return fail(*bond.line,
                  "only bonds of functions 1 and 2 are supported, not " +
                      llvm::Twine(bond.function));
    std::vector<double> p = bond.parameters;
    if (p.empty()) {
      const BondedType *entry = lookup(
          "bondtypes:" + std::to_string(bond.function), bonded(bond));
      if (!entry)
        return fail(*bond.line, "no [ bondtypes ] for this bond");
      p = entry->parameters;
    }
    if (p.size() != 2 && p.size() != 4)
      return fail(*bond.line, "expected 2 parameters of a bond");
    if (p.size() == 4 && (p[2] != p[0] || p[3] != p[1]))
      return fail(*bond.line, "a bond of the state B (free energy) is not "
                              "supported");
    if (bond.function == 2) {
      // b0 in nm to Å, k in kJ/(mol nm⁴) to kcal/(mol Å⁴).
      TupleTerm &term = termOf("gromos_bond", "0.25*kb*(r^2 - b0^2)^2", 2,
                               {"b0", "kb"});
      term.particles.push_back(offset + bond.atoms[0]);
      term.particles.push_back(offset + bond.atoms[1]);
      term.parameters[0].second.push_back(p[0] * 10.0);
      term.parameters[1].second.push_back(p[1] / kjPerKcal * 1e-4);
      continue;
    }
    Topology::Bond term;
    term.i = offset + bond.atoms[0];
    term.j = offset + bond.atoms[1];
    term.r0 = p[0];
    term.k = p[1];
    term.hydrogen = isHydrogen(bond.atoms[0]) || isHydrogen(bond.atoms[1]);
    topology.bonds.push_back(term);
  }

  // Angles: function 1, ½ k (θ − θ0)², with θ0 in degrees then k; function
  // 5 adds ½ k_UB (r₁₃ − r13)² between the outer atoms (Urey–Bradley), with
  // r13 then k_UB after them.
  for (const Interaction &angle : molecule.angles) {
    int function = angle.function;
    if (function != 1 && function != 2 && function != 5)
      return fail(*angle.line,
                  "only angles of functions 1, 2, and 5 are supported, not " +
                      llvm::Twine(function));
    std::vector<double> p = angle.parameters;
    if (p.empty()) {
      const BondedType *entry =
          lookup("angletypes:" + std::to_string(function), bonded(angle));
      if (!entry)
        return fail(*angle.line, "no [ angletypes ] for this angle");
      p = entry->parameters;
    }
    size_t count = function == 5 ? 4 : 2;
    if (p.size() != count && p.size() != 2 * count)
      return fail(*angle.line, "expected " + llvm::Twine(count) +
                                   " parameters of an angle of function " +
                                   llvm::Twine(function));
    if (function == 2) {
      // The angle of GROMOS in its cosine, ½ k (cos θ − cos θ0)², a term
      // given by an expression: θ0 in degrees to radians, k in kJ/mol to
      // kcal/mol.
      TupleTerm &term = termOf("gromos_angle",
                               "0.5*ka*(cos(theta) - cos(t0))^2", 3,
                               {"t0", "ka"});
      for (int a = 0; a != 3; ++a)
        term.particles.push_back(offset + angle.atoms[a]);
      term.parameters[0].second.push_back(p[0] * radiansPerDegree);
      term.parameters[1].second.push_back(p[1] / kjPerKcal);
      continue;
    }
    Topology::Angle term;
    term.i = offset + angle.atoms[0];
    term.j = offset + angle.atoms[1];
    term.k = offset + angle.atoms[2];
    term.theta0 = p[0] * radiansPerDegree;
    term.force = p[1];
    topology.angles.push_back(term);
    if (function == 5 && p[3] != 0.0)
      topology.ureyBradleys.push_back(
          {term.i, term.k, /*force=*/p[3], /*r0=*/p[2]});
  }

  // Dihedrals: functions 1, 4, and 9, k (1 + cos(n φ − φ0)), with φ0 in
  // degrees, then k, then n; function 2, the harmonic improper
  // ½ k (ξ − ξ0)², with ξ0 in degrees then k.
  // Ryckaert-Bellemans dihedrals (function 3), sum over n of C_n cos^n psi
  // with psi = phi - 180 degrees, and Fourier dihedrals (function 5), are
  // terms given by expressions (D136), in kcal/mol, one for each function.
  for (const Interaction &dihedral : molecule.dihedrals) {
    int function = dihedral.function;
    if (function != 1 && function != 2 && function != 3 && function != 4 &&
        function != 5 && function != 9)
      return fail(*dihedral.line,
                  "only periodic dihedrals (functions 1, 4, and 9), "
                  "harmonic impropers (function 2), and dihedrals of "
                  "Ryckaert and Bellemans (3) and Fourier (5) are supported, "
                  "not " +
                      llvm::Twine(function));
    if (function == 3 || function == 5) {
      std::vector<double> p = dihedral.parameters;
      if (p.empty()) {
        DihedralMatch match = matchDihedral(molecule, dihedral);
        if (!match.list)
          return fail(*dihedral.line,
                      "no [ dihedraltypes ] for this dihedral");
        p = (*match.list)[match.index].parameters;
      }
      size_t count = function == 3 ? 6 : 4;
      if (p.size() != count && p.size() != 2 * count)
        return fail(*dihedral.line, "expected " + llvm::Twine(count) +
                                        " parameters of this dihedral");
      if (llvm::all_of(llvm::ArrayRef<double>(p).take_front(count),
                       [](double c) { return c == 0.0; }))
        continue;
      TupleTerm &term =
          function == 3
              ? termOf("ryckaert_bellemans",
                       "c0 + c1*p + c2*p^2 + c3*p^3 + c4*p^4 + c5*p^5; "
                       "p = -cos(theta)",
                       4, {"c0", "c1", "c2", "c3", "c4", "c5"})
              : termOf("fourier",
                       "0.5*(f1*(1 + cos(theta)) + f2*(1 - cos(2*theta)) + "
                       "f3*(1 + cos(3*theta)) + f4*(1 - cos(4*theta)))",
                       4, {"f1", "f2", "f3", "f4"});
      for (int a = 0; a != 4; ++a)
        term.particles.push_back(offset + dihedral.atoms[a]);
      for (size_t c = 0; c != count; ++c)
        term.parameters[c].second.push_back(p[c] / kjPerKcal);
      continue;
    }
    if (function == 2) {
      std::vector<double> p = dihedral.parameters;
      if (p.empty()) {
        DihedralMatch match = matchDihedral(molecule, dihedral);
        if (!match.list)
          return fail(*dihedral.line,
                      "no [ dihedraltypes ] for this dihedral");
        p = (*match.list)[match.index].parameters;
      }
      if (p.size() != 2 && p.size() != 4)
        return fail(*dihedral.line,
                    "expected 2 parameters of a harmonic improper");
      if (p[1] == 0.0)
        continue;
      Topology::HarmonicImproper term;
      term.i = offset + dihedral.atoms[0];
      term.j = offset + dihedral.atoms[1];
      term.k = offset + dihedral.atoms[2];
      term.l = offset + dihedral.atoms[3];
      term.xi0 = p[0] * radiansPerDegree;
      term.force = p[1];
      topology.harmonicImpropers.push_back(term);
      continue;
    }
    std::vector<std::vector<double>> terms;
    if (!dihedral.parameters.empty()) {
      terms.push_back(dihedral.parameters);
    } else {
      // The entry with the most types that are not wildcards, and, in the
      // table of functions 1 and 9, the terms that follow it with the same
      // types.
      DihedralMatch match = matchDihedral(molecule, dihedral);
      if (match.list) {
        const std::vector<BondedType> &list = *match.list;
        terms.push_back(list[match.index].parameters);
        if (function != 4)
          for (size_t e = match.index + 1;
               e < list.size() && list[e].types == list[match.index].types; ++e)
            terms.push_back(list[e].parameters);
      }
      if (terms.empty())
        return fail(*dihedral.line, "no [ dihedraltypes ] for this dihedral");
    }
    for (const std::vector<double> &p : terms) {
      if (p.size() != 3 && p.size() != 6)
        return fail(*dihedral.line, "expected 3 parameters of a dihedral");
      double n = p[2];
      if (std::fabs(n - std::round(n)) > 0.01)
        return fail(*dihedral.line, "the multiplicity " + show(n) +
                                        " is not a whole number");
      if (p[1] == 0.0 && (p.size() == 3 || p[4] == 0.0))
        continue;
      Topology::Dihedral term;
      term.i = offset + dihedral.atoms[0];
      term.j = offset + dihedral.atoms[1];
      term.k = offset + dihedral.atoms[2];
      term.l = offset + dihedral.atoms[3];
      term.phase = p[0] * radiansPerDegree;
      term.force = p[1];
      term.n = static_cast<int>(std::round(n));
      term.improper = function == 4;
      topology.dihedrals.push_back(term);
    }
  }

  // Pairs three bonds apart: function 1, with σ and ε from the line, from
  // [ pairtypes ], or from the nonbonded pair scaled by fudgeLJ.
  for (const Interaction &pair : molecule.pairs) {
    if (pair.function != 1)
      return fail(*pair.line, "only pairs of function 1 are supported, not " +
                                  llvm::Twine(pair.function));
    unsigned a = atomTypeIndex.at(molecule.atoms[pair.atoms[0]].type);
    unsigned b = atomTypeIndex.at(molecule.atoms[pair.atoms[1]].type);
    double v, w;
    bool scaled = false;
    if (pair.parameters.size() >= 2) {
      v = pair.parameters[0];
      w = pair.parameters[1];
    } else if (!pair.parameters.empty()) {
      return fail(*pair.line, "expected 2 parameters of a pair");
    } else {
      auto found = pairTypes.find({std::min(a, b), std::max(a, b)});
      if (found != pairTypes.end()) {
        std::tie(v, w) = found->second;
      } else if (generatePairs) {
        std::tie(v, w) = getPair(a, b);
        scaled = true;
      } else {
        return fail(*pair.line, "no [ pairtypes ] for this pair, and "
                                "gen-pairs is no");
      }
    }
    Topology::Pair term;
    term.i = offset + pair.atoms[0];
    term.j = offset + pair.atoms[1];
    if (llvm::Error error =
            toSigmaEpsilon(pair.line, v, w, term.sigma, term.epsilon))
      return error;
    // With the rule 1 fudgeLJ scales c6 and c12, which scales ε alike.
    term.scaleLJ = scaled ? fudgeLJ : 1.0;
    term.scaleCoulomb = fudgeQQ;
    topology.pairs.push_back(term);
  }

  for (const Interaction &settle : molecule.settles)
    topology.settles.push_back({offset + settle.atoms[0],
                                settle.parameters[0], settle.parameters[1]});

  // Each correction map takes the first [ cmaptypes ] whose five types
  // are those of its atoms in their order. A type of the form `T-R`
  // is the bonded type T in the residue R, or in any residue for `T-*`,
  // as amber19sb.ff writes them.
  for (const Interaction &cmap : molecule.cmaps) {
    std::vector<std::string> types = bonded(cmap);
    auto matches = [&](StringRef wanted, unsigned place) {
      if (wanted == types[place])
        return true;
      size_t dash = wanted.rfind('-');
      if (dash == StringRef::npos)
        return false;
      StringRef residue = wanted.drop_front(dash + 1);
      return wanted.take_front(dash) == types[place] &&
             (residue == "*" ||
              residue == molecule.atoms[cmap.atoms[place]].residue);
    };
    int found = -1;
    for (auto [index, type] : llvm::enumerate(cmapTypes))
      if (llvm::all_of(llvm::seq(0u, 5u), [&](unsigned place) {
            return matches(type.types[place], place);
          })) {
        found = index;
        break;
      }
    if (found < 0)
      return fail(*cmap.line, "no [ cmaptypes ] for the types " +
                                  llvm::join(types, " "));
    auto [entry, inserted] =
        usedCMaps.insert({static_cast<unsigned>(found),
                          static_cast<unsigned>(topology.cmapGrids.size())});
    if (inserted) {
      topology.cmapResolution = cmapResolution;
      topology.cmapGrids.push_back(cmapTypes[found].grid);
    }
    const std::vector<unsigned> &a = cmap.atoms;
    topology.cmaps.push_back({offset + a[0], offset + a[1], offset + a[2],
                              offset + a[3], offset + a[4], entry->second});
  }

  // Each virtual site is built once, from atoms that are not sites.
  std::vector<int> built(molecule.atoms.size(), 0);
  for (const Interaction &site : molecule.virtualSites) {
    unsigned s = site.atoms[0];
    if (!molecule.atoms[s].virtualSite)
      return fail(*site.line, "the atom " + llvm::Twine(s + 1) +
                                  " of [ virtual_sites3 ] is not of the "
                                  "particle type V or D");
    for (unsigned k = 1; k != 4; ++k)
      if (molecule.atoms[site.atoms[k]].virtualSite)
        return fail(*site.line, "a virtual site built from another is not "
                                "supported");
    if (built[s]++)
      return fail(*site.line, "the virtual site " + llvm::Twine(s + 1) +
                                  " is built twice");
    Topology::VirtualSite term;
    term.kind = Topology::VirtualSite::Linear;
    term.site = offset + s;
    term.i = offset + site.atoms[1];
    term.j = offset + site.atoms[2];
    term.k = offset + site.atoms[3];
    term.a = site.parameters[0];
    term.b = site.parameters[1];
    topology.virtualSites.push_back(term);
  }
  for (auto [local, atom] : llvm::enumerate(molecule.atoms))
    if (atom.virtualSite && !built[local])
      return fail(*atom.line, "the virtual site " + llvm::Twine(local + 1) +
                                  " of the molecule type '" + molecule.name +
                                  "' is not built by a [ virtual_sites3 ]");

  // Exclusions: the atoms within nrexcl bonds, and those of the section.
  size_t count = molecule.atoms.size();
  std::vector<std::vector<unsigned>> neighbors(count);
  for (const Interaction &bond : molecule.bonds) {
    neighbors[bond.atoms[0]].push_back(bond.atoms[1]);
    neighbors[bond.atoms[1]].push_back(bond.atoms[0]);
  }
  std::set<std::pair<unsigned, unsigned>> excluded(
      molecule.exclusions.begin(), molecule.exclusions.end());
  for (unsigned i = 0; i != count; ++i) {
    std::vector<int> distance(count, -1);
    std::vector<unsigned> frontier = {i};
    distance[i] = 0;
    for (int d = 1; d <= molecule.nrexcl; ++d) {
      std::vector<unsigned> next;
      for (unsigned atom : frontier)
        for (unsigned other : neighbors[atom])
          if (distance[other] < 0) {
            distance[other] = d;
            next.push_back(other);
          }
      frontier = std::move(next);
    }
    for (unsigned j = i + 1; j != count; ++j)
      if (distance[j] > 0)
        excluded.insert({i, j});
  }
  for (auto [i, j] : excluded)
    topology.exclusions.push_back({offset + i, offset + j});
  return llvm::Error::success();
}

llvm::Error TopologyReader::build(Topology &topology) {
  if (molecules.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "%s: [ molecules ] names no molecule",
                                   path.c_str());

  // The types of the system: those that its atoms use, in the order of
  // [ atomtypes ].
  std::vector<int> used(atomTypes.size(), -1);
  for (auto [index, count] : molecules)
    for (const Atom &atom : moleculeTypes[index].atoms)
      used[atomTypeIndex.at(atom.type)] = 0;
  std::vector<unsigned> typesOfSystem;
  for (unsigned t = 0; t != atomTypes.size(); ++t)
    if (used[t] >= 0) {
      used[t] = typesOfSystem.size();
      typesOfSystem.push_back(t);
      topology.typeNames.push_back(atomTypes[t].name);
    }
  unsigned numTypes = typesOfSystem.size();
  topology.sigma.assign(numTypes * numTypes, 0.0);
  topology.epsilon.assign(numTypes * numTypes, 0.0);
  for (unsigned a = 0; a != numTypes; ++a)
    for (unsigned b = 0; b != numTypes; ++b) {
      auto [v, w] = getPair(typesOfSystem[a], typesOfSystem[b]);
      if (llvm::Error error =
              toSigmaEpsilon(nullptr, v, w, topology.sigma[a * numTypes + b],
                             topology.epsilon[a * numTypes + b]))
        return error;
    }

  unsigned offset = 0, residue = 0;
  for (auto [index, count] : molecules) {
    const MoleculeType &molecule = moleculeTypes[index];
    for (long copy = 0; copy != count; ++copy) {
      std::string lastName, lastNumber;
      for (auto [local, atom] : llvm::enumerate(molecule.atoms)) {
        const AtomType &type = atomTypes[atomTypeIndex.at(atom.type)];
        topology.atomNames.push_back(atom.name);
        topology.atomicNumbers.push_back(type.atomicNumber);
        topology.masses.push_back(atom.mass);
        topology.charges.push_back(atom.charge);
        topology.types.push_back(used[atomTypeIndex.at(atom.type)]);
        if (local == 0 || atom.residue != lastName ||
            atom.residueNumber != lastNumber) {
          topology.residueNames.push_back(atom.residue);
          topology.residueStarts.push_back(offset + local);
          ++residue;
        }
        topology.residueOf.push_back(residue - 1);
        lastName = atom.residue;
        lastNumber = atom.residueNumber;
      }
      if (llvm::Error error = expandMolecule(molecule, topology, offset))
        return error;
      offset += molecule.atoms.size();
    }
  }
  std::sort(topology.exclusions.begin(), topology.exclusions.end());
  return llvm::Error::success();
}

llvm::Expected<Topology> TopologyReader::read() {
  if (llvm::Error error = preprocessor.run(path, lines))
    return std::move(error);

  std::string section;
  for (const Line &line : lines) {
    StringRef text = line.text;
    if (text.starts_with("[")) {
      StringRef name = text.drop_front().take_until(
          [](char c) { return c == ']'; });
      section = canonicalSection(name);
      // The old name of the section.
      if (section == "dummies3")
        section = "virtualsites3";
      static const std::set<std::string> known = {
          "defaults", "atomtypes", "bondtypes", "angletypes",
          "dihedraltypes", "pairtypes", "nonbondparams", "moleculetype",
          "atoms", "bonds", "pairs", "angles", "dihedrals", "exclusions",
          "settles", "system", "molecules", "constrainttypes",
          "virtualsites3", "cmaptypes", "cmap",
          "implicitgenbornparams", "implicitsurfaceparams"};
      static const std::map<std::string, std::string> planned = {
          {"constraints", "M1"},
          {"virtualsites1", "M2a"},    {"virtualsites2", "M2a"},
          {"virtualsites4", "M2a"},
          {"virtualsitesn", "M2a"},    {"dummies1", "M2a"},
          {"dummies2", "M2a"},
          {"dummies4", "M2a"},         {"dummiesn", "M2a"}};
      auto found = planned.find(section);
      if (found != planned.end())
        return fail(line, "[ " + name.trim() + " ] is not supported yet; it "
                              "is planned for " +
                              found->second);
      if (!known.count(section))
        return fail(line, "the section [ " + name.trim() +
                              " ] is not supported");
      continue;
    }
    if (section.empty())
      continue;
    if (llvm::Error error = readLine(line, section))
      return std::move(error);
  }

  Topology topology;
  topology.sourceFiles = preprocessor.sourceFiles;
  if (llvm::Error error = build(topology))
    return std::move(error);
  return std::move(topology);
}

llvm::Expected<Topology>
mdir::driver::readGromacsTopology(StringRef path,
                                  llvm::ArrayRef<std::string> includePath,
                                  llvm::ArrayRef<std::string> defines) {
  return TopologyReader(path, includePath, defines).read();
}

//===----------------------------------------------------------------------===//
// Coordinates
//===----------------------------------------------------------------------===//

llvm::Error mdir::driver::readGromacsCoordinates(StringRef path,
                                                 Topology &topology) {
  auto file = llvm::MemoryBuffer::getFile(path);
  if (!file)
    return fail(path, 0, "cannot read the file: " + file.getError().message());
  llvm::SmallVector<StringRef> lines;
  (*file)->getBuffer().split(lines, '\n');
  for (StringRef &line : lines)
    line = line.take_until([](char c) { return c == '\r'; });
  if (lines.size() < 3)
    return fail(path, 0, "expected a title, the number of atoms, and atoms");

  long count;
  if (lines[1].trim().take_while([](char c) {
                          return llvm::isDigit(c) || c == '-';
                        })
          .getAsInteger(10, count))
    return fail(path, 2, "expected the number of atoms");
  if (count != static_cast<long>(topology.getNumParticles()))
    return fail(path, 2, "the file has " + llvm::Twine(count) +
                             " atoms, and the topology " +
                             llvm::Twine(topology.getNumParticles()));
  if (lines.size() < static_cast<size_t>(count) + 3)
    return fail(path, lines.size(), "the file ends before the box");

  // The width of a field: the distance between the first decimal points.
  StringRef first = lines[2];
  size_t d1 = first.find('.'), d2 = first.find('.', d1 + 1),
         d3 = first.find('.', d2 + 1);
  if (d3 == StringRef::npos)
    return fail(path, 3, "a coordinate without a decimal point");
  size_t width = d2 - d1;
  if (d3 - d2 != width)
    return fail(path, 3, "the decimal points of x, y, and z are not evenly "
                         "spaced");

  topology.positions.clear();
  std::vector<double> velocities;
  bool hasVelocities = false;
  for (long i = 0; i != count; ++i) {
    StringRef line = lines[2 + i];
    unsigned number = 3 + i;
    if (line.size() < 20 + 3 * width - 1)
      return fail(path, number, "the line is too short for the coordinates");
    for (int k = 0; k != 3; ++k) {
      double value;
      if (!readReal(line.substr(20 + k * width, width), value))
        return fail(path, number, "cannot read a coordinate in '" + line +
                                      "'");
      topology.positions.push_back(value);
    }
    for (int k = 0; k != 3; ++k) {
      double value = 0.0;
      StringRef field = line.substr(20 + (3 + k) * width, width).trim();
      if (!field.empty() && readReal(field, value))
        hasVelocities = true;
      else
        value = 0.0;
      velocities.push_back(value);
    }
  }
  topology.velocities.clear();
  if (hasVelocities)
    topology.velocities = std::move(velocities);

  std::vector<std::string> box = tokenize(lines[2 + count]);
  if (box.size() < 3)
    return fail(path, 3 + count, "expected the box");
  for (int k = 0; k != 3; ++k)
    if (!readReal(box[k], topology.box[k]) || topology.box[k] < 0.0)
      return fail(path, 3 + count, "cannot read the box");
  // A box of zeros, as GROMACS writes for no periodic cell (D142).
  if (topology.box[0] == 0.0 && topology.box[1] == 0.0 &&
      topology.box[2] == 0.0)
    return llvm::Error::success();
  for (int k = 0; k != 3; ++k)
    if (topology.box[k] == 0.0)
      return fail(path, 3 + count, "cannot read the box");
  // v1(x) v2(y) v3(z) v1(y) v1(z) v2(x) v2(z) v3(x) v3(y): a cell with a
  // along x and b in the x-y plane, as GROMACS keeps it.
  double rest[6] = {0, 0, 0, 0, 0, 0};
  for (size_t k = 3; k < box.size() && k < 9; ++k)
    if (!readReal(box[k], rest[k - 3]))
      return fail(path, 3 + count, "cannot read the box");
  if (rest[0] != 0.0 || rest[1] != 0.0 || rest[3] != 0.0)
    return fail(path, 3 + count, "a cell needs a along x and b in the x-y "
                                 "plane, as GROMACS keeps it: v1(y), v1(z), "
                                 "and v2(z) must be 0");
  Cell cell;
  cell.diagonal = {topology.box[0], topology.box[1], topology.box[2]};
  cell.tilt = {rest[2], rest[4], rest[5]};
  if (llvm::Error error = reduceCell(cell))
    return fail(path, 3 + count, llvm::toString(std::move(error)));
  for (int k = 0; k != 3; ++k)
    topology.tilt[k] = cell.tilt[k];
  return llvm::Error::success();
}
