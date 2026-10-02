// Readers of the formats of CHARMM: protein structure files (PSF), the
// files of topology and parameters (RTF, PRM, and stream files), and
// coordinates (CRD).
//
// The readers are written from the documentation of the formats, with no
// code of CHARMM or of its tools (docs/charmm-m1.md). The terms are
// converted to the forms of docs/conventions.md.

#include "mdir/Driver/Topology.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

#include <array>
#include <cmath>
#include <map>
#include <set>

using namespace mdir::driver;
using llvm::StringRef;

namespace {

/// Units of CHARMM in those of MDIR.
constexpr double nmPerAngstrom = 0.1;
constexpr double kjPerKcal = 4.184;
constexpr double radiansPerDegree = M_PI / 180.0;
/// σ = R_min / 2^(1/6).
const double sigmaPerRmin = std::pow(2.0, -1.0 / 6.0);

llvm::Error fail(StringRef path, size_t line, const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), "%s:%zu: %s",
                                 path.str().c_str(), line,
                                 message.str().c_str());
}

llvm::Error fail(StringRef path, const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), "%s: %s",
                                 path.str().c_str(), message.str().c_str());
}

bool readReal(StringRef text, double &value) {
  return !text.trim().getAsDouble(value);
}

bool readInteger(StringRef text, long &value) {
  return !text.trim().getAsInteger(10, value);
}

/// The first four letters of a keyword, upper case, as CHARMM matches them.
std::string keyword(StringRef token) { return token.take_front(4).upper(); }

//===----------------------------------------------------------------------===//
// Parameters
//===----------------------------------------------------------------------===//

/// The parameters of the files that a run names, in the units and forms of
/// CHARMM (kcal/mol, Å, degrees; E = K (x − x0)² for bonds, angles,
/// Urey–Bradley terms, and impropers).
struct Parameters {
  struct Bond {
    double k, b0;
  };
  struct Angle {
    double k, theta0, kUB = 0.0, s0 = 0.0;
  };
  struct Dihedral {
    double k;
    int n;
    double delta;
  };
  /// K (ψ − ψ0)² for n = 0, K (1 + cos(n ψ − ψ0)) otherwise.
  struct Improper {
    double k;
    int n;
    double psi0;
  };
  struct LennardJones {
    // ε (as a positive number) and R_min/2, and those of the pairs three
    // bonds apart.
    double epsilon, rminHalf, epsilon14, rminHalf14;
  };
  struct PairFix {
    double epsilon, rmin, epsilon14, rmin14;
  };
  struct CMap {
    std::array<std::string, 8> types;
    unsigned resolution;
    std::vector<double> grid;
  };

  llvm::StringMap<double> masses;
  llvm::StringMap<int> elements;
  std::map<std::array<std::string, 2>, Bond> bonds;
  std::map<std::array<std::string, 3>, Angle> angles;
  /// The terms of each dihedral type, in the order of the file.
  std::map<std::array<std::string, 4>, std::vector<Dihedral>> dihedrals;
  std::map<std::array<std::string, 4>, Improper> impropers;
  std::vector<CMap> cmaps;
  llvm::StringMap<LennardJones> nonbonded;
  std::map<std::array<std::string, 2>, PairFix> nbfix;
  /// The factor of the Coulomb term of the pairs three bonds apart, from
  /// the options of NONBONDED.
  double e14fac = 1.0;
};

/// A key of two types in a canonical order.
std::array<std::string, 2> key2(StringRef a, StringRef b) {
  return a <= b ? std::array<std::string, 2>{a.str(), b.str()}
                : std::array<std::string, 2>{b.str(), a.str()};
}

/// A key of three types whose outer types are in a canonical order.
std::array<std::string, 3> key3(StringRef a, StringRef b, StringRef c) {
  return a <= c ? std::array<std::string, 3>{a.str(), b.str(), c.str()}
                : std::array<std::string, 3>{c.str(), b.str(), a.str()};
}

/// A key of four types, read forward or backward, whichever sorts first.
std::array<std::string, 4> key4(StringRef a, StringRef b, StringRef c,
                                StringRef d) {
  std::array<std::string, 4> forward{a.str(), b.str(), c.str(), d.str()};
  std::array<std::string, 4> backward{d.str(), c.str(), b.str(), a.str()};
  return std::min(forward, backward);
}

int elementNumber(StringRef symbol) {
  static const std::pair<const char *, int> elements[] = {
      {"H", 1},   {"HE", 2},  {"LI", 3},  {"BE", 4},  {"B", 5},   {"C", 6},
      {"N", 7},   {"O", 8},   {"F", 9},   {"NE", 10}, {"NA", 11}, {"MG", 12},
      {"AL", 13}, {"SI", 14}, {"P", 15},  {"S", 16},  {"CL", 17}, {"AR", 18},
      {"K", 19},  {"CA", 20}, {"MN", 25}, {"FE", 26}, {"CO", 27}, {"NI", 28},
      {"CU", 29}, {"ZN", 30}, {"SE", 34}, {"BR", 35}, {"RB", 37}, {"SR", 38},
      {"CD", 48}, {"I", 53},  {"CS", 55}, {"BA", 56}};
  std::string upper = symbol.upper();
  for (auto [name, number] : elements)
    if (upper == name)
      return number;
  return 0;
}

/// The atomic number of a particle without an element: from its mass.
int numberFromMass(double mass) {
  static const std::pair<double, int> weights[] = {
      {1.008, 1},   {4.0026, 2},  {6.94, 3},    {12.011, 6},  {14.007, 7},
      {15.999, 8},  {18.998, 9},  {22.990, 11}, {24.305, 12}, {30.974, 15},
      {32.06, 16},  {35.45, 17},  {39.098, 19}, {40.078, 20}, {65.38, 30},
      {79.904, 35}, {126.90, 53}, {132.91, 55}};
  if (mass <= 0.0)
    return 0;
  // A hydrogen whose mass has been repartitioned is still a hydrogen.
  if (mass < 4.1)
    return 1;
  int best = 0;
  double distance = 1e30;
  for (auto [weight, number] : weights)
    if (std::fabs(weight - mass) < distance) {
      distance = std::fabs(weight - mass);
      best = number;
    }
  return distance < 0.6 ? best : 0;
}

/// Reads the files of topology, parameters, and streams of CHARMM, in the
/// order given: what a later file defines replaces what an earlier one did.
class ParameterReader {
public:
  explicit ParameterReader(Parameters &parameters) : parameters(parameters) {}

  llvm::Error read(StringRef path);

private:
  enum class Mode { Commands, Topology, Parameters };
  enum class Section {
    None,
    Atoms,
    Bonds,
    Angles,
    Dihedrals,
    Impropers,
    CMap,
    Nonbonded,
    NBFix,
    Skipped
  };

  llvm::Error readParameterLine(StringRef path, size_t number,
                                llvm::ArrayRef<StringRef> t);
  llvm::Error readMass(StringRef path, size_t number,
                       llvm::ArrayRef<StringRef> t);

  llvm::Error readOptions(StringRef path, size_t number,
                          llvm::ArrayRef<StringRef> t);

  Parameters &parameters;
  Section section = Section::None;
  /// The dihedral type of the line before, if it was one: lines of the
  /// same types that follow one another make the terms of one set.
  std::array<std::string, 4> lastDihedral;
  bool afterDihedral = false;
  /// A map of CMAP being read: its numbers so far.
  Parameters::CMap *cmap = nullptr;
  /// Whether the options of NONBONDED go on to the next line.
  bool continued = false;
};

llvm::Error ParameterReader::readMass(StringRef path, size_t number,
                                      llvm::ArrayRef<StringRef> t) {
  // MASS number type mass [element]
  double mass;
  if (t.size() < 4 || !readReal(t[3], mass))
    return fail(path, number, "expected 'MASS number type mass [element]'");
  parameters.masses[t[2].upper()] = mass;
  if (t.size() >= 5)
    if (int element = elementNumber(t[4]))
      parameters.elements[t[2].upper()] = element;
  return llvm::Error::success();
}

llvm::Error ParameterReader::read(StringRef path) {
  auto file = llvm::MemoryBuffer::getFile(path);
  if (!file)
    return fail(path, "cannot read the file: " + file.getError().message());
  StringRef extension = llvm::sys::path::extension(path);
  Mode mode = Mode::Commands;
  if (extension.equals_insensitive(".rtf"))
    mode = Mode::Topology;
  else if (extension.equals_insensitive(".prm") ||
           extension.equals_insensitive(".par") ||
           extension.equals_insensitive(".inp"))
    mode = Mode::Parameters;
  section = Section::None;
  afterDihedral = false;
  continued = false;
  cmap = nullptr;
  // The title of a file or of a block that `read ... card` begins: lines
  // that begin with `*`, up to a line of `*` alone.
  bool title = mode != Mode::Commands;

  llvm::SmallVector<StringRef> lines;
  (*file)->getBuffer().split(lines, '\n');
  for (auto [index, raw] : llvm::enumerate(lines)) {
    size_t number = index + 1;
    StringRef line = raw.split('!').first.rtrim("\r").trim();
    if (title) {
      if (line.starts_with("*")) {
        if (line == "*")
          title = false;
        continue;
      }
      title = false;
    }
    if (line.empty())
      continue;
    llvm::SmallVector<StringRef> t;
    line.split(t, ' ', -1, /*KeepEmpty=*/false);
    for (StringRef &token : t)
      token = token.trim("\t");
    llvm::erase_if(t, [](StringRef token) { return token.empty(); });
    if (t.empty())
      continue;
    std::string word = keyword(t[0]);

    if (mode == Mode::Commands) {
      // `read rtf card [append]` and `read para[meter] card [flex]
      // [append]` begin blocks; the other commands of a stream file are
      // not for a reader of parameters, which stops at the first `return`
      // as CHARMM does when the condition before it holds: the first of
      // two alternatives in a stream file is the one taken.
      if (word == "RETU")
        break;
      if (word == "READ" && t.size() >= 2) {
        std::string what = keyword(t[1]);
        if (what == "RTF") {
          mode = Mode::Topology;
          title = true;
        } else if (what == "PARA") {
          mode = Mode::Parameters;
          section = Section::None;
          afterDihedral = false;
          title = true;
        }
      }
      continue;
    }
    if (word == "END") {
      if (cmap && cmap->grid.size() != cmap->resolution * cmap->resolution)
        return fail(path, number, "a map of CMAP ends before its " +
                                      llvm::Twine(cmap->resolution *
                                                  cmap->resolution) +
                                      " numbers");
      cmap = nullptr;
      mode = Mode::Commands;
      continue;
    }
    if (mode == Mode::Topology) {
      if (word == "MASS")
        if (llvm::Error error = readMass(path, number, t))
          return error;
      continue;
    }
    if (llvm::Error error = readParameterLine(path, number, t))
      return error;
  }
  return llvm::Error::success();
}

llvm::Error ParameterReader::readOptions(StringRef path, size_t number,
                                         llvm::ArrayRef<StringRef> t) {
  // The options of NONBONDED that change the terms: E14FAC.
  if (section != Section::Nonbonded)
    return llvm::Error::success();
  for (size_t k = 0; k + 1 < t.size(); ++k)
    if (keyword(t[k]) == "E14F") {
      if (!readReal(t[k + 1], parameters.e14fac))
        return fail(path, number, "cannot read E14FAC");
    }
  return llvm::Error::success();
}

llvm::Error ParameterReader::readParameterLine(StringRef path, size_t number,
                                               llvm::ArrayRef<StringRef> t) {
  if (continued) {
    continued = t.back() == "-";
    return readOptions(path, number, t);
  }
  bool dihedralLine = false;
  llvm::scope_exit done([&] { afterDihedral = dihedralLine; });
  // A line of numbers continues a map of CMAP.
  if (cmap) {
    double value;
    if (readReal(t[0], value)) {
      for (StringRef token : t) {
        if (!readReal(token, value))
          return fail(path, number, "expected numbers of a map of CMAP");
        cmap->grid.push_back(value);
      }
      if (cmap->grid.size() > cmap->resolution * cmap->resolution)
        return fail(path, number, "too many numbers for a map of CMAP");
      return llvm::Error::success();
    }
    if (cmap->grid.size() != cmap->resolution * cmap->resolution)
      return fail(path, number, "a map of CMAP ends before its " +
                                    llvm::Twine(cmap->resolution *
                                                cmap->resolution) +
                                    " numbers");
    cmap = nullptr;
  }

  std::string word = keyword(t[0]);
  static const std::pair<const char *, Section> sections[] = {
      {"ATOM", Section::Atoms},      {"BOND", Section::Bonds},
      {"ANGL", Section::Angles},     {"THET", Section::Angles},
      {"DIHE", Section::Dihedrals},  {"PHI", Section::Dihedrals},
      {"IMPR", Section::Impropers},  {"IMPH", Section::Impropers},
      {"CMAP", Section::CMap},       {"NONB", Section::Nonbonded},
      {"NBON", Section::Nonbonded},  {"NBFI", Section::NBFix},
      {"HBON", Section::Skipped},    {"THOL", Section::Skipped},
      {"EQUI", Section::Skipped}};
  for (auto [name, kind] : sections)
    if (word == name) {
      section = kind;
      // NONBONDED and HBOND carry options on their line, which may go on.
      continued = t.back() == "-";
      return readOptions(path, number, t);
    }

  auto real = [&](size_t i, double &value) -> llvm::Error {
    if (i >= t.size() || !readReal(t[i], value))
      return fail(path, number, "expected a number in field " +
                                    llvm::Twine(i + 1));
    return llvm::Error::success();
  };
  auto upper = [&](size_t i) { return t[i].upper(); };

  switch (section) {
  case Section::None:
    return fail(path, number, "a line before the first section: '" +
                                  t[0] + "'");
  case Section::Skipped:
    return llvm::Error::success();
  case Section::Atoms:
    if (word == "MASS")
      return readMass(path, number, t);
    return llvm::Error::success();
  case Section::Bonds: {
    Parameters::Bond bond;
    if (llvm::Error error = real(2, bond.k))
      return error;
    if (llvm::Error error = real(3, bond.b0))
      return error;
    parameters.bonds[key2(upper(0), upper(1))] = bond;
    return llvm::Error::success();
  }
  case Section::Angles: {
    Parameters::Angle angle;
    if (llvm::Error error = real(3, angle.k))
      return error;
    if (llvm::Error error = real(4, angle.theta0))
      return error;
    if (t.size() >= 7) {
      if (llvm::Error error = real(5, angle.kUB))
        return error;
      if (llvm::Error error = real(6, angle.s0))
        return error;
    }
    parameters.angles[key3(upper(0), upper(1), upper(2))] = angle;
    return llvm::Error::success();
  }
  case Section::Dihedrals: {
    Parameters::Dihedral term;
    double n;
    if (llvm::Error error = real(4, term.k))
      return error;
    if (llvm::Error error = real(5, n))
      return error;
    if (llvm::Error error = real(6, term.delta))
      return error;
    term.n = static_cast<int>(std::lround(n));
    if (term.n <= 0)
      return fail(path, number, "a dihedral of multiplicity " +
                                    llvm::Twine(term.n) +
                                    " is not supported");
    auto key = key4(upper(0), upper(1), upper(2), upper(3));
    std::vector<Parameters::Dihedral> &terms = parameters.dihedrals[key];
    // Lines of the same types that follow one another make one set of
    // terms; a later set of those types replaces it whole.
    if (!(afterDihedral && key == lastDihedral))
      terms.clear();
    terms.push_back(term);
    lastDihedral = key;
    dihedralLine = true;
    return llvm::Error::success();
  }
  case Section::Impropers: {
    Parameters::Improper improper;
    double n;
    if (llvm::Error error = real(4, improper.k))
      return error;
    if (llvm::Error error = real(5, n))
      return error;
    if (llvm::Error error = real(6, improper.psi0))
      return error;
    improper.n = static_cast<int>(std::lround(n));
    parameters.impropers[key4(upper(0), upper(1), upper(2), upper(3))] =
        improper;
    return llvm::Error::success();
  }
  case Section::CMap: {
    // Eight types, those of the two dihedrals, and the number of points of
    // the grid along each angle.
    long resolution;
    if (t.size() != 9 || !readInteger(t[8], resolution) || resolution < 2)
      return fail(path, number, "expected eight types and the number of "
                                "points of a map of CMAP");
    Parameters::CMap map;
    for (int k = 0; k != 8; ++k)
      map.types[k] = upper(k);
    map.resolution = static_cast<unsigned>(resolution);
    // A later map of the same types replaces an earlier one.
    llvm::erase_if(parameters.cmaps, [&](const Parameters::CMap &other) {
      return other.types == map.types;
    });
    parameters.cmaps.push_back(std::move(map));
    cmap = &parameters.cmaps.back();
    return llvm::Error::success();
  }
  case Section::Nonbonded: {
    // type ignored ε R_min/2 [ignored ε14 R_min/2 14]; ε is negative.
    Parameters::LennardJones lj;
    if (llvm::Error error = real(2, lj.epsilon))
      return error;
    if (llvm::Error error = real(3, lj.rminHalf))
      return error;
    lj.epsilon = std::fabs(lj.epsilon);
    lj.epsilon14 = lj.epsilon;
    lj.rminHalf14 = lj.rminHalf;
    if (t.size() >= 7) {
      if (llvm::Error error = real(5, lj.epsilon14))
        return error;
      if (llvm::Error error = real(6, lj.rminHalf14))
        return error;
      lj.epsilon14 = std::fabs(lj.epsilon14);
    }
    if (t[0].contains('*') || t[0].contains('%'))
      return fail(path, number, "types with wildcards in NONBONDED are not "
                                "supported: '" + t[0] + "'");
    parameters.nonbonded[upper(0)] = lj;
    return llvm::Error::success();
  }
  case Section::NBFix: {
    // type type E_min R_min [E_min,14 R_min,14]
    Parameters::PairFix fix;
    if (llvm::Error error = real(2, fix.epsilon))
      return error;
    if (llvm::Error error = real(3, fix.rmin))
      return error;
    fix.epsilon = std::fabs(fix.epsilon);
    fix.epsilon14 = -1.0;
    fix.rmin14 = -1.0;
    if (t.size() >= 6) {
      if (llvm::Error error = real(4, fix.epsilon14))
        return error;
      if (llvm::Error error = real(5, fix.rmin14))
        return error;
      fix.epsilon14 = std::fabs(fix.epsilon14);
    }
    parameters.nbfix[key2(upper(0), upper(1))] = fix;
    return llvm::Error::success();
  }
  }
  return llvm::Error::success();
}

//===----------------------------------------------------------------------===//
// The protein structure file
//===----------------------------------------------------------------------===//

struct Structure {
  struct Atom {
    std::string segment, residue, residueName, name, type;
    double charge, mass;
  };
  std::vector<Atom> atoms;
  std::vector<std::array<unsigned, 2>> bonds;
  std::vector<std::array<unsigned, 3>> angles;
  std::vector<std::array<unsigned, 4>> dihedrals, impropers;
  std::vector<std::array<unsigned, 8>> crossTerms;
  /// Pairs excluded by the file beyond those of the bonds (!NNB).
  std::vector<std::pair<unsigned, unsigned>> exclusions;
};

llvm::Expected<Structure> readStructure(StringRef path) {
  auto file = llvm::MemoryBuffer::getFile(path);
  if (!file)
    return fail(path, "cannot read the file: " + file.getError().message());
  llvm::SmallVector<StringRef> lines;
  (*file)->getBuffer().split(lines, '\n');
  if (lines.empty() || !lines[0].trim().starts_with("PSF"))
    return fail(path, 1, "a PSF begins with 'PSF'");
  llvm::SmallVector<StringRef> flags;
  lines[0].split(flags, ' ', -1, false);
  bool xplor = llvm::is_contained(flags, "XPLOR");
  for (StringRef flag : flags)
    if (flag == "DRUDE")
      return fail(path, 1, "a PSF of the Drude model is not supported");

  Structure structure;
  size_t at = 1;
  // The next line with `!NAME`: its counts, and the line after it.
  auto header = [&](StringRef name, llvm::SmallVectorImpl<long> &counts,
                    bool required) -> llvm::Expected<bool> {
    for (; at < lines.size(); ++at) {
      StringRef line = lines[at];
      size_t bang = line.find('!');
      if (bang == StringRef::npos)
        continue;
      StringRef label = line.substr(bang + 1).split(':').first.trim();
      if (!label.starts_with(name))
        continue;
      llvm::SmallVector<StringRef> fields;
      line.take_front(bang).split(fields, ' ', -1, false);
      counts.clear();
      for (StringRef field : fields) {
        long value;
        if (!readInteger(field, value))
          return fail(path, at + 1, "cannot read the counts of !" + name);
        counts.push_back(value);
      }
      ++at;
      return true;
    }
    if (required)
      return fail(path, "no !" + name + " section");
    return false;
  };
  // `count` tuples of `arity` atoms numbered from 1, as many to a line as
  // fit.
  auto tuples = [&](StringRef name, long count, unsigned arity,
                    auto &result) -> llvm::Error {
    std::vector<long> values;
    while (static_cast<long>(values.size()) < count * arity) {
      if (at >= lines.size())
        return fail(path, "the file ends in !" + name);
      llvm::SmallVector<StringRef> fields;
      lines[at++].split(fields, ' ', -1, false);
      for (StringRef field : fields) {
        long value;
        if (!readInteger(field.trim(), value))
          return fail(path, at, "cannot read the atoms of !" + name);
        if (value < 1 || value > static_cast<long>(structure.atoms.size()))
          return fail(path, at, "the atom " + llvm::Twine(value) + " of !" +
                                    name + " is out of range");
        values.push_back(value - 1);
      }
    }
    if (static_cast<long>(values.size()) != count * arity)
      return fail(path, at, "!" + name + " has more atoms than its count");
    for (long k = 0; k != count; ++k) {
      typename std::decay_t<decltype(result)>::value_type tuple;
      for (unsigned m = 0; m != arity; ++m)
        tuple[m] = static_cast<unsigned>(values[k * arity + m]);
      result.push_back(tuple);
    }
    return llvm::Error::success();
  };

  llvm::SmallVector<long> counts;
  llvm::Expected<bool> found = header("NTITLE", counts, true);
  if (!found)
    return found.takeError();
  found = header("NATOM", counts, true);
  if (!found)
    return found.takeError();
  long natom = counts.empty() ? -1 : counts[0];
  if (natom < 0)
    return fail(path, at, "cannot read the count of atoms");
  for (long i = 0; i != natom; ++i, ++at) {
    if (at >= lines.size())
      return fail(path, "the file ends in !NATOM");
    // number segment residue residue-name name type charge mass imove ...
    llvm::SmallVector<StringRef> f;
    lines[at].split(f, ' ', -1, false);
    llvm::erase_if(f, [](StringRef s) { return s.trim().empty(); });
    if (f.size() < 8)
      return fail(path, at + 1, "expected the number, segment, residue, "
                                "residue name, name, type, charge, and mass "
                                "of an atom");
    Structure::Atom atom;
    atom.segment = f[1].str();
    atom.residue = f[2].str();
    atom.residueName = f[3].str();
    atom.name = f[4].str();
    atom.type = f[5].upper();
    if (!xplor) {
      long code;
      if (readInteger(f[5], code))
        return fail(path, at + 1, "a PSF with numbers for its types is not "
                                  "supported; write it with 'write psf "
                                  "card xplor'");
    }
    if (!readReal(f[6], atom.charge) || !readReal(f[7], atom.mass))
      return fail(path, at + 1, "cannot read the charge and the mass");
    structure.atoms.push_back(std::move(atom));
  }

  if (!(found = header("NBOND", counts, true)))
    return found.takeError();
  if (llvm::Error error = tuples("NBOND", counts[0], 2, structure.bonds))
    return std::move(error);
  if (!(found = header("NTHETA", counts, true)))
    return found.takeError();
  if (llvm::Error error = tuples("NTHETA", counts[0], 3, structure.angles))
    return std::move(error);
  if (!(found = header("NPHI", counts, true)))
    return found.takeError();
  if (llvm::Error error = tuples("NPHI", counts[0], 4, structure.dihedrals))
    return std::move(error);
  if (!(found = header("NIMPHI", counts, true)))
    return found.takeError();
  if (llvm::Error error =
          tuples("NIMPHI", counts[0], 4, structure.impropers))
    return std::move(error);

  // !NNB: the excluded atoms, then for each atom the index of its last one.
  found = header("NNB", counts, false);
  if (!found)
    return found.takeError();
  if (*found && counts[0] > 0) {
    std::vector<std::array<unsigned, 1>> excluded;
    if (llvm::Error error = tuples("NNB", counts[0], 1, excluded))
      return std::move(error);
    std::vector<long> last;
    while (static_cast<long>(last.size()) < natom && at < lines.size()) {
      llvm::SmallVector<StringRef> fields;
      lines[at++].split(fields, ' ', -1, false);
      for (StringRef field : fields) {
        long value;
        if (!readInteger(field.trim(), value))
          return fail(path, at, "cannot read the pointers of !NNB");
        last.push_back(value);
      }
    }
    long begin = 0;
    for (long i = 0; i != natom; ++i) {
      for (long k = begin; k < last[i]; ++k) {
        unsigned j = excluded[k][0];
        structure.exclusions.push_back(
            {std::min<unsigned>(i, j), std::max<unsigned>(i, j)});
      }
      begin = last[i];
    }
  }

  // Lone pairs are not supported; a file without them says 0 0.
  found = header("NUMLP", counts, false);
  if (!found)
    return found.takeError();
  if (*found && !counts.empty() && counts[0] != 0)
    return fail(path, at, "lone pairs (!NUMLP) are not supported");

  found = header("NCRTERM", counts, false);
  if (!found)
    return found.takeError();
  if (*found)
    if (llvm::Error error =
            tuples("NCRTERM", counts[0], 8, structure.crossTerms))
      return std::move(error);
  return structure;
}

//===----------------------------------------------------------------------===//
// The topology
//===----------------------------------------------------------------------===//

/// Assigns the parameters to the terms of a structure.
class Assigner {
public:
  Assigner(StringRef path, const Structure &structure,
           const Parameters &parameters)
      : path(path), structure(structure), parameters(parameters) {}

  llvm::Expected<Topology> build();

private:
  const std::string &type(unsigned atom) const {
    return structure.atoms[atom].type;
  }
  std::string describe(llvm::ArrayRef<unsigned> atoms) const {
    std::string text;
    for (unsigned atom : atoms)
      text += (text.empty() ? "" : " ") + type(atom) + "(" +
              std::to_string(atom + 1) + ")";
    return text;
  }
  const std::vector<Parameters::Dihedral> *
  findDihedral(const std::array<unsigned, 4> &atoms) const;
  const Parameters::Improper *
  findImproper(const std::array<unsigned, 4> &atoms) const;

  std::string path;
  const Structure &structure;
  const Parameters &parameters;
};

const std::vector<Parameters::Dihedral> *
Assigner::findDihedral(const std::array<unsigned, 4> &a) const {
  // The types of the four atoms, then X for the outer two.
  auto &table = parameters.dihedrals;
  auto found = table.find(key4(type(a[0]), type(a[1]), type(a[2]),
                               type(a[3])));
  if (found != table.end())
    return &found->second;
  found = table.find(key4("X", type(a[1]), type(a[2]), "X"));
  if (found != table.end())
    return &found->second;
  return nullptr;
}

const Parameters::Improper *
Assigner::findImproper(const std::array<unsigned, 4> &a) const {
  // The types of the four atoms, then the patterns with wildcards in the
  // order that CHARMM tries them: A X X D, X B C D, X B C X, X X C D; each
  // in either direction.
  auto &table = parameters.impropers;
  const std::string &t0 = type(a[0]), &t1 = type(a[1]), &t2 = type(a[2]),
                    &t3 = type(a[3]);
  for (auto key : {key4(t0, t1, t2, t3), key4(t0, "X", "X", t3),
                   key4("X", t1, t2, t3), key4("X", t1, t2, "X"),
                   key4("X", "X", t2, t3)}) {
    auto found = table.find(key);
    if (found != table.end())
      return &found->second;
  }
  return nullptr;
}

llvm::Expected<Topology> Assigner::build() {
  Topology topology;
  size_t count = structure.atoms.size();

  // Particles, residues, and the types of Lennard-Jones that they use.
  std::map<std::string, unsigned> typeIndex;
  std::string lastResidue;
  for (auto [i, atom] : llvm::enumerate(structure.atoms)) {
    topology.atomNames.push_back(atom.name);
    topology.masses.push_back(atom.mass);
    topology.charges.push_back(atom.charge);
    auto element = parameters.elements.find(atom.type);
    topology.atomicNumbers.push_back(element != parameters.elements.end()
                                         ? element->second
                                         : numberFromMass(atom.mass));
    std::string residue = atom.segment + ":" + atom.residue;
    if (i == 0 || residue != lastResidue) {
      topology.residueNames.push_back(atom.residueName);
      topology.residueStarts.push_back(i);
      lastResidue = residue;
    }
    topology.residueOf.push_back(topology.residueStarts.size() - 1);
    if (!parameters.nonbonded.count(atom.type))
      return fail(path, "no NONBONDED parameters for the type " + atom.type +
                            " of atom " + llvm::Twine(i + 1));
    auto [entry, added] =
        typeIndex.try_emplace(atom.type, topology.typeNames.size());
    if (added)
      topology.typeNames.push_back(atom.type);
    topology.types.push_back(entry->second);
  }

  // Lennard-Jones of each pair of types: the rule, or NBFIX.
  // E = ε ((R/r)¹² − 2 (R/r)⁶) = 4ε ((σ/r)¹² − (σ/r)⁶), σ = R / 2^(1/6).
  size_t types = topology.typeNames.size();
  topology.sigma.assign(types * types, 0.0);
  topology.epsilon.assign(types * types, 0.0);
  auto pairParameters = [&](StringRef a, StringRef b, bool fourteen,
                            double &sigma, double &epsilon) {
    const Parameters::LennardJones &p = parameters.nonbonded.find(a)->second;
    const Parameters::LennardJones &q = parameters.nonbonded.find(b)->second;
    double rmin = fourteen ? p.rminHalf14 + q.rminHalf14
                           : p.rminHalf + q.rminHalf;
    double e = fourteen ? std::sqrt(p.epsilon14 * q.epsilon14)
                        : std::sqrt(p.epsilon * q.epsilon);
    auto fix = parameters.nbfix.find(key2(a, b));
    if (fix != parameters.nbfix.end()) {
      if (!fourteen) {
        rmin = fix->second.rmin;
        e = fix->second.epsilon;
      } else if (fix->second.rmin14 >= 0.0) {
        rmin = fix->second.rmin14;
        e = fix->second.epsilon14;
      } else {
        rmin = fix->second.rmin;
        e = fix->second.epsilon;
      }
    }
    sigma = rmin * sigmaPerRmin * nmPerAngstrom;
    epsilon = e * kjPerKcal;
  };
  for (size_t a = 0; a != types; ++a)
    for (size_t b = 0; b != types; ++b)
      pairParameters(topology.typeNames[a], topology.typeNames[b], false,
                     topology.sigma[a * types + b],
                     topology.epsilon[a * types + b]);

  auto isHydrogen = [&](unsigned atom) {
    return topology.atomicNumbers[atom] == 1;
  };

  // Bonds: E = K (b − b0)², ½ k (r − r0)² with k = 2K.
  for (const auto &b : structure.bonds) {
    auto found = parameters.bonds.find(key2(type(b[0]), type(b[1])));
    if (found == parameters.bonds.end())
      return fail(path, "no BONDS parameters for " + describe(b));
    Topology::Bond bond;
    bond.i = b[0];
    bond.j = b[1];
    bond.k = 2.0 * found->second.k * kjPerKcal /
             (nmPerAngstrom * nmPerAngstrom);
    bond.r0 = found->second.b0 * nmPerAngstrom;
    bond.hydrogen = isHydrogen(b[0]) || isHydrogen(b[1]);
    topology.bonds.push_back(bond);
  }

  // Angles, and their Urey–Bradley terms between the outer atoms.
  for (const auto &a : structure.angles) {
    auto found =
        parameters.angles.find(key3(type(a[0]), type(a[1]), type(a[2])));
    if (found == parameters.angles.end())
      return fail(path, "no ANGLES parameters for " + describe(a));
    const Parameters::Angle &p = found->second;
    topology.angles.push_back({a[0], a[1], a[2], 2.0 * p.k * kjPerKcal,
                               p.theta0 * radiansPerDegree});
    if (p.kUB != 0.0)
      topology.ureyBradleys.push_back(
          {a[0], a[2],
           2.0 * p.kUB * kjPerKcal / (nmPerAngstrom * nmPerAngstrom),
           p.s0 * nmPerAngstrom});
  }

  // Dihedrals: K (1 + cos(n φ − δ)), each term of the type.
  for (const auto &d : structure.dihedrals) {
    const std::vector<Parameters::Dihedral> *terms = findDihedral(d);
    if (!terms)
      return fail(path, "no DIHEDRALS parameters for " + describe(d));
    for (const Parameters::Dihedral &term : *terms) {
      if (term.k == 0.0)
        continue;
      Topology::Dihedral dihedral;
      dihedral.i = d[0];
      dihedral.j = d[1];
      dihedral.k = d[2];
      dihedral.l = d[3];
      dihedral.force = term.k * kjPerKcal;
      dihedral.n = term.n;
      dihedral.phase = term.delta * radiansPerDegree;
      dihedral.improper = false;
      topology.dihedrals.push_back(dihedral);
    }
  }

  // Impropers: K (ψ − ψ0)², ½ k (ψ − ψ0)² with k = 2K.
  for (const auto &d : structure.impropers) {
    const Parameters::Improper *p = findImproper(d);
    if (!p)
      return fail(path, "no IMPROPER parameters for " + describe(d));
    if (p->k == 0.0)
      continue;
    if (p->n == 0) {
      topology.harmonicImpropers.push_back(
          {d[0], d[1], d[2], d[3], 2.0 * p->k * kjPerKcal,
           p->psi0 * radiansPerDegree});
      continue;
    }
    Topology::Dihedral periodic;
    periodic.i = d[0];
    periodic.j = d[1];
    periodic.k = d[2];
    periodic.l = d[3];
    periodic.force = p->k * kjPerKcal;
    periodic.n = p->n;
    periodic.phase = p->psi0 * radiansPerDegree;
    periodic.improper = true;
    topology.dihedrals.push_back(periodic);
  }

  // CMAP: two dihedrals that share three atoms, φ of i j k l and ψ of
  // j k l m, on the map of their eight types.
  if (!structure.crossTerms.empty()) {
    std::map<size_t, unsigned> used;
    for (const auto &c : structure.crossTerms) {
      if (c[1] != c[4] || c[2] != c[5] || c[3] != c[6])
        return fail(path, "the two dihedrals of the cross-term " +
                              describe(c) + " do not share three atoms");
      std::array<std::string, 8> types;
      for (int k = 0; k != 8; ++k)
        types[k] = type(c[k]);
      auto found = llvm::find_if(parameters.cmaps,
                                 [&](const Parameters::CMap &map) {
                                   return map.types == types;
                                 });
      if (found == parameters.cmaps.end())
        return fail(path, "no CMAP parameters for " + describe(c));
      if (topology.cmapResolution != 0 &&
          topology.cmapResolution != found->resolution)
        return fail(path, "maps of CMAP of different resolutions are not "
                          "supported");
      topology.cmapResolution = found->resolution;
      size_t index = found - parameters.cmaps.begin();
      auto [entry, added] = used.try_emplace(index, topology.cmapGrids.size());
      if (added) {
        std::vector<double> grid;
        for (double value : found->grid)
          grid.push_back(value * kjPerKcal);
        topology.cmapGrids.push_back(std::move(grid));
      }
      topology.cmaps.push_back({c[0], c[1], c[2], c[3], c[7], entry->second});
    }
  }

  // Exclusions: the atoms one, two, and three bonds apart, and those of
  // !NNB; the pairs three bonds apart (and not closer) take their own
  // Lennard-Jones and the Coulomb term times E14FAC.
  std::vector<std::vector<unsigned>> neighbors(count);
  for (const auto &b : structure.bonds) {
    neighbors[b[0]].push_back(b[1]);
    neighbors[b[1]].push_back(b[0]);
  }
  std::set<std::pair<unsigned, unsigned>> excluded(
      structure.exclusions.begin(), structure.exclusions.end());
  std::vector<int> distance(count, -1);
  for (unsigned i = 0; i != count; ++i) {
    std::vector<unsigned> reached = {i}, frontier = {i};
    distance[i] = 0;
    for (int d = 1; d <= 3; ++d) {
      std::vector<unsigned> next;
      for (unsigned atom : frontier)
        for (unsigned other : neighbors[atom])
          if (distance[other] < 0) {
            distance[other] = d;
            next.push_back(other);
            reached.push_back(other);
          }
      frontier = std::move(next);
    }
    for (unsigned j : reached) {
      if (j <= i)
        continue;
      excluded.insert({i, j});
      if (distance[j] == 3) {
        Topology::Pair pair;
        pair.i = i;
        pair.j = j;
        pairParameters(type(i), type(j), true, pair.sigma, pair.epsilon);
        pair.scaleLJ = 1.0;
        pair.scaleCoulomb = parameters.e14fac;
        topology.pairs.push_back(pair);
      }
    }
    for (unsigned j : reached)
      distance[j] = -1;
  }
  topology.exclusions.assign(excluded.begin(), excluded.end());
  return topology;
}

} // namespace

llvm::Expected<Topology>
mdir::driver::readCharmmTopology(StringRef path,
                                 llvm::ArrayRef<std::string> parameterFiles) {
  if (parameterFiles.empty())
    return fail(path, "a PSF needs the files of its parameters ('parameters' "
                      "in [input])");
  Parameters parameters;
  ParameterReader reader(parameters);
  for (const std::string &file : parameterFiles)
    if (llvm::Error error = reader.read(file))
      return std::move(error);
  llvm::Expected<Structure> structure = readStructure(path);
  if (!structure)
    return structure.takeError();
  return Assigner(path, *structure, parameters).build();
}

llvm::Error mdir::driver::readCharmmCoordinates(StringRef path,
                                                Topology &topology) {
  auto file = llvm::MemoryBuffer::getFile(path);
  if (!file)
    return fail(path, "cannot read the file: " + file.getError().message());
  llvm::SmallVector<StringRef> lines;
  (*file)->getBuffer().split(lines, '\n');
  size_t at = 0;
  while (at < lines.size() && lines[at].starts_with("*"))
    ++at;
  long count;
  llvm::SmallVector<StringRef> head;
  if (at < lines.size())
    lines[at].split(head, ' ', -1, false);
  if (head.empty() || !readInteger(head[0], count))
    return fail(path, at + 1, "expected the number of atoms after the title");
  if (static_cast<size_t>(count) != topology.getNumParticles())
    return fail(path, at + 1, "the file has " + llvm::Twine(count) +
                                  " atoms, the topology " +
                                  llvm::Twine(topology.getNumParticles()));
  ++at;
  topology.positions.clear();
  for (long i = 0; i != count; ++i, ++at) {
    if (at >= lines.size())
      return fail(path, "the file ends before atom " + llvm::Twine(i + 1));
    // number residue-number residue name x y z segment residue weight
    llvm::SmallVector<StringRef> f;
    lines[at].split(f, ' ', -1, false);
    llvm::erase_if(f, [](StringRef s) { return s.trim().empty(); });
    double x[3];
    if (f.size() < 7 || !readReal(f[4], x[0]) || !readReal(f[5], x[1]) ||
        !readReal(f[6], x[2]))
      return fail(path, at + 1, "cannot read the position of atom " +
                                    llvm::Twine(i + 1));
    if (f[3].str() != topology.atomNames[i])
      return fail(path, at + 1, "atom " + llvm::Twine(i + 1) + " is " + f[3] +
                                    " here and " + topology.atomNames[i] +
                                    " in the topology");
    for (double v : x)
      topology.positions.push_back(v * nmPerAngstrom);
  }
  return llvm::Error::success();
}
