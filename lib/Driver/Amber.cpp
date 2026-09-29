// Readers of the formats of Amber: topologies (prmtop) and coordinates
// (inpcrd, rst7).
//
// The readers are written from a specification of the formats, with no
// code of Amber or of its tools. The terms are converted to the forms of
// docs/conventions.md.

#include "mdir/Driver/Topology.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/MemoryBuffer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>

using namespace mdir::driver;
using llvm::StringRef;

namespace {

/// A number as a message shows it.
std::string show(double value) {
  char text[32];
  std::snprintf(text, sizeof(text), "%g", value);
  return text;
}

/// Units of Amber in those of MDIR.
constexpr double nmPerAngstrom = 0.1;
constexpr double kjPerKcal = 4.184;
/// The factor of the charges of a prmtop: `q = CHARGE / 18.2223`.
constexpr double chargeFactor = 18.2223;
/// Velocities of a restart are in Å per 1/20.455 ps.
constexpr double velocityFactor = 20.455;

/// A section of a prmtop: the descriptor of `%FORMAT` and the data lines.
struct Section {
  std::string format;
  std::vector<StringRef> lines;
  unsigned line = 0;
};

/// `[count](letter width[.decimals])`, with or without parentheses.
struct Format {
  char letter = 0;
  unsigned width = 0;
};

class Reader {
public:
  explicit Reader(StringRef path) : path(path) {}

  llvm::Expected<Topology> read();

private:
  llvm::Error fail(const llvm::Twine &message) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "%s: %s", path.c_str(),
                                   message.str().c_str());
  }

  llvm::Error split(StringRef text);
  llvm::Expected<Format> parseFormat(StringRef flag, StringRef text);

  /// The items of the section `flag`, `count` of them if `count` is not
  /// -1. A missing section is an error if `required`.
  llvm::Error readStrings(StringRef flag, long count,
                          std::vector<std::string> &values,
                          bool required = true);
  llvm::Error readIntegers(StringRef flag, long count,
                           std::vector<long> &values, bool required = true);
  llvm::Error readReals(StringRef flag, long count,
                        std::vector<double> &values, bool required = true);
  llvm::Error readFields(StringRef flag, long count, char kind,
                         std::vector<std::string> &fields, bool required);

  llvm::Error checkSupported(const std::vector<long> &pointers);
  llvm::Error readTypes(Topology &topology, long count, long numPairs);
  llvm::Error readTerms(Topology &topology, const std::vector<long> &p);
  llvm::Error checkExclusions(const Topology &topology);

  std::string path;
  std::unique_ptr<llvm::MemoryBuffer> buffer;
  std::map<std::string, Section> sections;

  /// Lennard-Jones by the triangle of types that the 1-4 pairs use.
  std::vector<double> tableA, tableB;
  std::vector<long> index;
  std::vector<long> numExcluded, excludedList;
};

} // namespace

//===----------------------------------------------------------------------===//
// Sections
//===----------------------------------------------------------------------===//

llvm::Error Reader::split(StringRef text) {
  llvm::SmallVector<StringRef> lines;
  text.split(lines, '\n');
  bool hasVersion = false;
  Section *current = nullptr;
  bool inData = false;
  for (StringRef line : lines) {
    line = line.rtrim("\r");
    if (line.starts_with("%VERSION")) {
      hasVersion = true;
      continue;
    }
    if (line.starts_with("%FLAG")) {
      std::string name = line.drop_front(5).trim().str();
      if (sections.count(name))
        return fail("the section '" + name + "' appears twice");
      current = &sections[name];
      inData = false;
      continue;
    }
    if (line.starts_with("%COMMENT")) {
      if (inData)
        return fail("a %COMMENT line among the data of a section");
      continue;
    }
    if (line.starts_with("%FORMAT")) {
      if (!current)
        return fail("%FORMAT before the first %FLAG");
      StringRef format = line.drop_front(7).trim();
      if (!format.consume_front("(") || !format.consume_back(")"))
        return fail("cannot read the line '" + line + "'");
      current->format = format.str();
      inData = true;
      continue;
    }
    if (line.starts_with("%"))
      return fail("an unknown control line '" + line + "'");
    if (inData)
      current->lines.push_back(line);
  }
  if (!hasVersion)
    return fail("no %VERSION line: a prmtop of the format before Amber 7 is "
                "not supported; write the file again with tleap or ParmEd");
  return llvm::Error::success();
}

llvm::Expected<Format> Reader::parseFormat(StringRef flag, StringRef text) {
  // `10I8`, `5E16.8`, `20a4`, `8(F9.5)`.
  StringRef rest = text.trim();
  rest = rest.drop_while([](char c) { return llvm::isDigit(c); });
  rest.consume_front("(");
  rest.consume_back(")");
  if (rest.empty())
    return fail("cannot read the format '" + text + "' of " + flag);
  Format format;
  format.letter = llvm::toUpper(rest.front());
  StringRef width = rest.drop_front().take_while(
      [](char c) { return llvm::isDigit(c); });
  if (width.getAsInteger(10, format.width) || format.width == 0 ||
      !llvm::is_contained(StringRef("AIEFD"), format.letter))
    return fail("cannot read the format '" + text + "' of " + flag);
  return format;
}

llvm::Error Reader::readFields(StringRef flag, long count, char kind,
                               std::vector<std::string> &fields,
                               bool required) {
  fields.clear();
  auto found = sections.find(flag.str());
  if (found == sections.end()) {
    if (required)
      return fail("no section " + flag);
    return llvm::Error::success();
  }
  auto format = parseFormat(flag, found->second.format);
  if (!format)
    return format.takeError();
  bool isName = format->letter == 'A';
  if ((kind == 'A') != isName || (kind == 'I') != (format->letter == 'I'))
    return fail("the section " + flag + " has the format '" +
                found->second.format + "', which is not of its kind");

  // Fields of a fixed width, not separated by blanks. Names are counted
  // from the length that the section should have.
  for (StringRef line : found->second.lines) {
    StringRef data = isName ? line : line.rtrim();
    for (size_t start = 0; start < data.size(); start += format->width) {
      if (isName && count >= 0 && static_cast<long>(fields.size()) == count)
        break;
      fields.push_back(data.substr(start, format->width).str());
    }
  }
  if (count >= 0 && static_cast<long>(fields.size()) != count)
    return fail("the section " + flag + " has " +
                llvm::Twine(fields.size()) + " values, and the file needs " +
                llvm::Twine(count));
  return llvm::Error::success();
}

llvm::Error Reader::readStrings(StringRef flag, long count,
                                std::vector<std::string> &values,
                                bool required) {
  if (llvm::Error error = readFields(flag, count, 'A', values, required))
    return error;
  for (std::string &value : values)
    value = StringRef(value).trim().str();
  return llvm::Error::success();
}

llvm::Error Reader::readIntegers(StringRef flag, long count,
                                 std::vector<long> &values, bool required) {
  std::vector<std::string> fields;
  if (llvm::Error error = readFields(flag, count, 'I', fields, required))
    return error;
  values.clear();
  for (const std::string &field : fields) {
    StringRef text = StringRef(field).trim();
    long value = 0;
    if (!text.empty() && text.getAsInteger(10, value))
      return fail("cannot read the integer '" + text + "' in " + flag);
    values.push_back(value);
  }
  return llvm::Error::success();
}

llvm::Error Reader::readReals(StringRef flag, long count,
                              std::vector<double> &values, bool required) {
  std::vector<std::string> fields;
  if (llvm::Error error = readFields(flag, count, 'E', fields, required))
    return error;
  values.clear();
  for (std::string &field : fields) {
    std::string text = StringRef(field).trim().str();
    for (char &c : text)
      if (c == 'D' || c == 'd')
        c = 'E';
    double value = 0.0;
    if (!text.empty() && StringRef(text).getAsDouble(value))
      return fail("cannot read the number '" + text + "' in " + flag);
    values.push_back(value);
  }
  return llvm::Error::success();
}

//===----------------------------------------------------------------------===//
// What MDIR does not support
//===----------------------------------------------------------------------===//

llvm::Error Reader::checkSupported(const std::vector<long> &p) {
  auto has = [&](StringRef flag) { return sections.count(flag.str()) != 0; };
  if (has("CTITLE") || has("FORCE_FIELD_TYPE"))
    return fail("a topology of CHARMM (Chamber) is not supported");
  for (auto &[name, section] : sections) {
    StringRef flag = name;
    if (flag.starts_with("CHARMM_"))
      return fail("a topology of CHARMM (Chamber) is not supported: " + flag);
    if (flag.starts_with("AMOEBA_"))
      return fail("a topology of AMOEBA is not supported: " + flag);
  }
  if (has("CMAP_COUNT"))
    return fail("CMAP terms (as in ff19SB) are not supported yet: "
                "CMAP_COUNT");
  if (has("LENNARD_JONES_CCOEF"))
    return fail("12-6-4 terms are not supported: LENNARD_JONES_CCOEF");
  if ((p.size() > 32 && p[32] > 0) || has("LENNARD_JONES_DCOEF") ||
      has("LENNARD_JONES_DVALUE"))
    return fail("pairwise C4 terms are not supported: POINTERS[32] = " +
                llvm::Twine(p.size() > 32 ? p[32] : 0));
  if (has("POLARIZABILITY") || has("DIPOLE_DAMP_FACTOR"))
    return fail("a polarizable topology is not supported: POLARIZABILITY");
  std::vector<long> ipol;
  if (llvm::Error error = readIntegers("IPOL", -1, ipol, false))
    return error;
  if (!ipol.empty() && ipol.front() > 0)
    return fail("a polarizable topology is not supported: IPOL = " +
                llvm::Twine(ipol.front()));
  if (p[30] > 0)
    return fail("extra points (virtual sites, as in TIP4P-Ew or OPC) are "
                "not supported yet (M2a): NUMEXTRA = " +
                llvm::Twine(p[30]));
  if (p[27] == 0)
    return fail("the topology has no periodic cell: IFBOX = 0");
  if (p[27] != 1)
    return fail("only an orthorhombic cell is supported: IFBOX = " +
                llvm::Twine(p[27]));
  if (p[29] != 0)
    return fail("a solvent cap is not supported: IFCAP = " +
                llvm::Twine(p[29]));
  if (p[20] != 0)
    return fail("a perturbed topology is not supported: IFPERT = " +
                llvm::Twine(p[20]));
  if (p[9] == 1)
    return fail("a topology of LES is not supported: NPARM = 1");
  if (p.size() > 31 && p[31] > 1)
    return fail("a topology of PIMD is not supported: NCOPY = " +
                llvm::Twine(p[31]));
  if (p[12] != p[3] || p[13] != p[5] || p[14] != p[7])
    return fail("constraints in the topology are not supported: NBONA, "
                "NTHETA, or NPHIA differs from MBONA, MTHETA, or MPHIA");
  for (StringRef flag :
       {"BOND_STIFFNESS_PULL_ADJ", "BOND_EQUIL_PULL_ADJ",
        "BOND_STIFFNESS_PRESS_ADJ", "BOND_EQUIL_PRESS_ADJ"})
    if (has(flag))
      return fail("augmented bonds are not supported: " + flag);
  for (auto &[name, section] : sections)
    if (StringRef(name).starts_with("DOUBLE_EXPONENTIAL_") ||
        StringRef(name).starts_with("EXPVDWMODEL_"))
      return fail("another form of the van der Waals term is not "
                  "supported: " +
                  name);

  // A section that MDIR does not know and that holds more than zeros may
  // change the physics.
  static const std::set<std::string> known = {
      "TITLE", "POINTERS", "ATOM_NAME", "CHARGE", "ATOMIC_NUMBER", "MASS",
      "ATOM_TYPE_INDEX", "NUMBER_EXCLUDED_ATOMS", "NONBONDED_PARM_INDEX",
      "RESIDUE_LABEL", "RESIDUE_POINTER", "BOND_FORCE_CONSTANT",
      "BOND_EQUIL_VALUE", "ANGLE_FORCE_CONSTANT", "ANGLE_EQUIL_VALUE",
      "DIHEDRAL_FORCE_CONSTANT", "DIHEDRAL_PERIODICITY", "DIHEDRAL_PHASE",
      "SCEE_SCALE_FACTOR", "SCNB_SCALE_FACTOR", "SOLTY",
      "LENNARD_JONES_ACOEF", "LENNARD_JONES_BCOEF", "BONDS_INC_HYDROGEN",
      "BONDS_WITHOUT_HYDROGEN", "ANGLES_INC_HYDROGEN",
      "ANGLES_WITHOUT_HYDROGEN", "DIHEDRALS_INC_HYDROGEN",
      "DIHEDRALS_WITHOUT_HYDROGEN", "EXCLUDED_ATOMS_LIST", "HBOND_ACOEF",
      "HBOND_BCOEF", "HBCUT", "AMBER_ATOM_TYPE",
      "TREE_CHAIN_CLASSIFICATION", "JOIN_ARRAY", "IROTAT",
      "SOLVENT_POINTERS", "ATOMS_PER_MOLECULE", "BOX_DIMENSIONS",
      "RADIUS_SET", "RADII", "SCREEN", "IPOL", "RESIDUE_NUMBER",
      "RESIDUE_CHAINID", "RESIDUE_ICODE", "ATOM_NUMBER", "ATOM_BFACTOR",
      "ATOM_OCCUPANCY", "ATOM_ALTLOC"};
  for (auto &[name, section] : sections) {
    if (known.count(name))
      continue;
    for (StringRef line : section.lines) {
      StringRef data = line.trim();
      bool zeros = llvm::all_of(data, [](char c) {
        return c == '0' || c == '.' || c == ' ' || c == '+' || c == '-' ||
               c == 'E' || c == 'e';
      });
      if (!zeros)
        return fail("the section " + name + " is not known to MDIR and "
                    "holds data that may change the physics");
    }
  }
  return llvm::Error::success();
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

llvm::Error Reader::readTypes(Topology &topology, long count,
                              long numPairs) {
  if (llvm::Error error =
          readIntegers("NONBONDED_PARM_INDEX", count * count, index))
    return error;
  long triangle = count * (count + 1) / 2;
  if (llvm::Error error = readReals("LENNARD_JONES_ACOEF", triangle, tableA))
    return error;
  if (llvm::Error error = readReals("LENNARD_JONES_BCOEF", triangle, tableB))
    return error;
  std::vector<double> hbondA, hbondB;
  if (llvm::Error error = readReals("HBOND_ACOEF", numPairs, hbondA, false))
    return error;
  if (llvm::Error error = readReals("HBOND_BCOEF", numPairs, hbondB, false))
    return error;

  std::vector<bool> used(count, false);
  for (unsigned type : topology.types)
    used[type] = true;

  topology.sigma.assign(count * count, 0.0);
  topology.epsilon.assign(count * count, 0.0);
  for (long a = 0; a != count; ++a)
    for (long b = 0; b != count; ++b) {
      long entry = index[a * count + b];
      long high = std::max(a, b) + 1, low = std::min(a, b) + 1;
      long canonical = high * (high - 1) / 2 + low;
      if (entry < 0) {
        // A 10-12 pair. With zero coefficients it is a placeholder, as
        // that of TIP3P, and the pair has no Lennard-Jones.
        long slot = -entry - 1;
        if (slot >= static_cast<long>(hbondA.size()))
          return fail("NONBONDED_PARM_INDEX points beyond HBOND_ACOEF");
        if ((hbondA[slot] != 0.0 || hbondB[slot] != 0.0) && used[a] &&
            used[b])
          return fail("10-12 terms are not supported: HBOND_ACOEF or "
                      "HBOND_BCOEF of the types " +
                      topology.typeNames[a] + " and " +
                      topology.typeNames[b] + " is not 0");
        continue;
      }
      // sander takes the 1-4 pairs from the triangle directly; the two
      // agree only for the canonical index.
      if (entry != canonical)
        return fail("NONBONDED_PARM_INDEX is not in the canonical order at "
                    "the types " +
                    llvm::Twine(a + 1) + " and " + llvm::Twine(b + 1) +
                    ", so that the nonbonded pairs and the 1-4 pairs of "
                    "Amber would take different parameters");
      double A = tableA[entry - 1], B = tableB[entry - 1];
      if (A == 0.0 && B == 0.0)
        continue;
      if (A <= 0.0 || B <= 0.0)
        return fail("the Lennard-Jones coefficients of the types " +
                    llvm::Twine(a + 1) + " and " + llvm::Twine(b + 1) +
                    " are not a Lennard-Jones pair: A = " + show(A) +
                    ", B = " + show(B));
      // A = 4 ε σ¹², B = 4 ε σ⁶.
      topology.sigma[a * count + b] =
          std::pow(A / B, 1.0 / 6.0) * nmPerAngstrom;
      topology.epsilon[a * count + b] = B * B / (4.0 * A) * kjPerKcal;
    }
  return llvm::Error::success();
}

//===----------------------------------------------------------------------===//
// Terms
//===----------------------------------------------------------------------===//

llvm::Error Reader::readTerms(Topology &topology,
                              const std::vector<long> &p) {
  long natom = p[0];
  auto atom = [&](long coordinate, unsigned &value) -> llvm::Error {
    coordinate = std::labs(coordinate);
    if (coordinate % 3 != 0 || coordinate / 3 >= natom)
      return fail("an atom index " + llvm::Twine(coordinate) +
                  " of a bonded term is out of range");
    value = coordinate / 3;
    return llvm::Error::success();
  };

  // Bonds: k (r − r0)² in the file.
  std::vector<double> bondK, bondR;
  if (llvm::Error error = readReals("BOND_FORCE_CONSTANT", p[15], bondK))
    return error;
  if (llvm::Error error = readReals("BOND_EQUIL_VALUE", p[15], bondR))
    return error;
  for (auto [flag, count, hydrogen] :
       {std::tuple<StringRef, long, bool>{"BONDS_INC_HYDROGEN", p[2], true},
        {"BONDS_WITHOUT_HYDROGEN", p[3], false}}) {
    std::vector<long> list;
    if (llvm::Error error = readIntegers(flag, 3 * count, list))
      return error;
    for (long n = 0; n != count; ++n) {
      Topology::Bond bond;
      if (llvm::Error error = atom(list[3 * n], bond.i))
        return error;
      if (llvm::Error error = atom(list[3 * n + 1], bond.j))
        return error;
      long type = list[3 * n + 2];
      if (type < 1 || type > p[15])
        return fail("a bond type of " + flag + " is out of range");
      bond.k = 2.0 * bondK[type - 1] * kjPerKcal /
               (nmPerAngstrom * nmPerAngstrom);
      bond.r0 = bondR[type - 1] * nmPerAngstrom;
      bond.hydrogen = hydrogen;
      topology.bonds.push_back(bond);
    }
  }

  // Angles: k (θ − θ0)², with θ0 in radians.
  std::vector<double> angleK, angleT;
  if (llvm::Error error = readReals("ANGLE_FORCE_CONSTANT", p[16], angleK))
    return error;
  if (llvm::Error error = readReals("ANGLE_EQUIL_VALUE", p[16], angleT))
    return error;
  for (auto [flag, count] :
       {std::pair<StringRef, long>{"ANGLES_INC_HYDROGEN", p[4]},
        {"ANGLES_WITHOUT_HYDROGEN", p[5]}}) {
    std::vector<long> list;
    if (llvm::Error error = readIntegers(flag, 4 * count, list))
      return error;
    for (long n = 0; n != count; ++n) {
      Topology::Angle angle;
      if (llvm::Error error = atom(list[4 * n], angle.i))
        return error;
      if (llvm::Error error = atom(list[4 * n + 1], angle.j))
        return error;
      if (llvm::Error error = atom(list[4 * n + 2], angle.k))
        return error;
      long type = list[4 * n + 3];
      if (type < 1 || type > p[16])
        return fail("an angle type of " + flag + " is out of range");
      angle.force = 2.0 * angleK[type - 1] * kjPerKcal;
      angle.theta0 = angleT[type - 1];
      topology.angles.push_back(angle);
    }
  }

  // Dihedrals: k (1 + cos(n φ − φ0)); a negative third index leaves the
  // 1-4 pair out, a negative fourth marks an improper.
  std::vector<double> dihedralK, dihedralN, dihedralPhase, scee, scnb;
  long numTypes = p[17];
  if (llvm::Error error =
          readReals("DIHEDRAL_FORCE_CONSTANT", numTypes, dihedralK))
    return error;
  if (llvm::Error error =
          readReals("DIHEDRAL_PERIODICITY", numTypes, dihedralN))
    return error;
  if (llvm::Error error = readReals("DIHEDRAL_PHASE", numTypes, dihedralPhase))
    return error;
  if (llvm::Error error =
          readReals("SCEE_SCALE_FACTOR", numTypes, scee, false))
    return error;
  if (llvm::Error error =
          readReals("SCNB_SCALE_FACTOR", numTypes, scnb, false))
    return error;
  if (scee.empty())
    scee.assign(numTypes, 1.2);
  if (scnb.empty())
    scnb.assign(numTypes, 2.0);

  long count = topology.getNumTypes();
  std::set<std::pair<unsigned, unsigned>> pairs14;
  for (auto [flag, number] :
       {std::pair<StringRef, long>{"DIHEDRALS_INC_HYDROGEN", p[6]},
        {"DIHEDRALS_WITHOUT_HYDROGEN", p[7]}}) {
    std::vector<long> list;
    if (llvm::Error error = readIntegers(flag, 5 * number, list))
      return error;
    for (long n = 0; n != number; ++n) {
      const long *entry = &list[5 * n];
      if (entry[2] == 0 || entry[3] == 0)
        return fail("a dihedral of " + flag + " has atom 1 in its third or "
                    "fourth place, where the sign of the index is lost");
      Topology::Dihedral dihedral;
      if (llvm::Error error = atom(entry[0], dihedral.i))
        return error;
      if (llvm::Error error = atom(entry[1], dihedral.j))
        return error;
      if (llvm::Error error = atom(entry[2], dihedral.k))
        return error;
      if (llvm::Error error = atom(entry[3], dihedral.l))
        return error;
      long type = entry[4];
      if (type < 1 || type > numTypes)
        return fail("a dihedral type of " + flag + " is out of range");
      double periodicity = std::fabs(dihedralN[type - 1]);
      if (periodicity != std::round(periodicity))
        return fail("a dihedral type has a periodicity that is not a whole "
                    "number: " +
                    show(dihedralN[type - 1]));
      dihedral.force = dihedralK[type - 1] * kjPerKcal;
      dihedral.n = static_cast<int>(periodicity);
      dihedral.phase = dihedralPhase[type - 1];
      dihedral.improper = entry[3] < 0;
      topology.dihedrals.push_back(dihedral);

      // The pair three bonds apart, with the factors of this entry.
      if (entry[2] < 0 || entry[3] < 0)
        continue;
      double e = scee[type - 1], v = scnb[type - 1];
      if (e == 0.0 || v == 0.0)
        return fail("a dihedral that carries a 1-4 pair has a factor of 0 "
                    "in SCEE_SCALE_FACTOR or SCNB_SCALE_FACTOR (type " +
                    llvm::Twine(type) + ")");
      unsigned i = dihedral.i, l = dihedral.l;
      if (!pairs14.insert({std::min(i, l), std::max(i, l)}).second)
        return fail("the 1-4 pair of atoms " + llvm::Twine(i + 1) + " and " +
                    llvm::Twine(l + 1) + " is carried by two dihedrals");
      // sander takes the pair from the triangle of the two types.
      unsigned a = topology.types[i], b = topology.types[l];
      Topology::Pair pair;
      pair.i = i;
      pair.j = l;
      pair.scaleCoulomb = 1.0 / e;
      pair.scaleLJ = 1.0 / v;
      long high = std::max(a, b) + 1, low = std::min(a, b) + 1;
      double A = tableA[high * (high - 1) / 2 + low - 1];
      double B = tableB[high * (high - 1) / 2 + low - 1];
      pair.sigma = pair.epsilon = 0.0;
      if (A > 0.0 && B > 0.0) {
        pair.sigma = std::pow(A / B, 1.0 / 6.0) * nmPerAngstrom;
        pair.epsilon = B * B / (4.0 * A) * kjPerKcal;
      }
      (void)count;
      topology.pairs.push_back(pair);
    }
  }

  // The exclusions of a periodic run: the members of the bonds, the ends of
  // the angles, and the ends of every dihedral.
  std::set<std::pair<unsigned, unsigned>> excluded;
  auto exclude = [&](unsigned i, unsigned j) {
    if (i != j)
      excluded.insert({std::min(i, j), std::max(i, j)});
  };
  for (const Topology::Bond &bond : topology.bonds)
    exclude(bond.i, bond.j);
  for (const Topology::Angle &angle : topology.angles)
    exclude(angle.i, angle.k);
  for (const Topology::Dihedral &dihedral : topology.dihedrals)
    exclude(dihedral.i, dihedral.l);
  topology.exclusions.assign(excluded.begin(), excluded.end());
  return checkExclusions(topology);
}

/// The list of the file must not exclude a pair that the bonded terms do
/// not: a periodic run of sander would drop it, so what the file means is
/// not clear.
llvm::Error Reader::checkExclusions(const Topology &topology) {
  std::set<std::pair<unsigned, unsigned>> known(topology.exclusions.begin(),
                                                topology.exclusions.end());
  size_t next = 0;
  for (size_t i = 0, e = numExcluded.size(); i != e; ++i) {
    for (long n = 0; n != numExcluded[i]; ++n, ++next) {
      if (next >= excludedList.size())
        return fail("EXCLUDED_ATOMS_LIST is shorter than "
                    "NUMBER_EXCLUDED_ATOMS says");
      long j = excludedList[next];
      if (j == 0)
        continue;
      unsigned a = i, b = j - 1;
      if (!known.count({std::min(a, b), std::max(a, b)}))
        return fail("EXCLUDED_ATOMS_LIST excludes the atoms " +
                    llvm::Twine(i + 1) + " and " + llvm::Twine(j) +
                    ", which no bond, angle, or dihedral joins; a periodic "
                    "run of Amber would not exclude them");
    }
  }
  return llvm::Error::success();
}

//===----------------------------------------------------------------------===//
// The topology
//===----------------------------------------------------------------------===//

llvm::Expected<Topology> Reader::read() {
  auto file = llvm::MemoryBuffer::getFile(path);
  if (!file)
    return fail("cannot read the file: " + file.getError().message());
  buffer = std::move(*file);
  if (llvm::Error error = split(buffer->getBuffer()))
    return std::move(error);

  std::vector<long> p;
  if (llvm::Error error = readIntegers("POINTERS", -1, p))
    return std::move(error);
  if (p.size() < 31)
    return fail("POINTERS has " + llvm::Twine(p.size()) +
                " values; at least 31 are needed");
  if (llvm::Error error = checkSupported(p))
    return std::move(error);

  Topology topology;
  long natom = p[0], ntypes = p[1], nres = p[11];
  if (llvm::Error error = readStrings("ATOM_NAME", natom, topology.atomNames))
    return std::move(error);
  std::vector<double> charges;
  if (llvm::Error error = readReals("CHARGE", natom, charges))
    return std::move(error);
  for (double charge : charges)
    topology.charges.push_back(charge / chargeFactor);
  if (llvm::Error error = readReals("MASS", natom, topology.masses))
    return std::move(error);
  std::vector<long> numbers;
  if (llvm::Error error = readIntegers("ATOMIC_NUMBER", natom, numbers, false))
    return std::move(error);
  topology.atomicNumbers.assign(numbers.begin(), numbers.end());

  std::vector<long> types;
  if (llvm::Error error = readIntegers("ATOM_TYPE_INDEX", natom, types))
    return std::move(error);
  std::vector<std::string> typeNames;
  if (llvm::Error error = readStrings("AMBER_ATOM_TYPE", natom, typeNames))
    return std::move(error);
  topology.typeNames.assign(ntypes, "");
  for (long i = 0; i != natom; ++i) {
    if (types[i] < 1 || types[i] > ntypes)
      return fail("ATOM_TYPE_INDEX of atom " + llvm::Twine(i + 1) +
                  " is out of range");
    topology.types.push_back(types[i] - 1);
    // The name of the first atom of a type names the type.
    std::string &name = topology.typeNames[types[i] - 1];
    if (name.empty())
      name = typeNames[i];
    if (StringRef(typeNames[i]).starts_with("EP"))
      return fail("the atom " + llvm::Twine(i + 1) + " is an extra point (" +
                  typeNames[i] + "); virtual sites are not supported yet");
  }
  for (long t = 0; t != ntypes; ++t)
    if (topology.typeNames[t].empty())
      topology.typeNames[t] = "type" + std::to_string(t + 1);

  if (llvm::Error error =
          readStrings("RESIDUE_LABEL", nres, topology.residueNames))
    return std::move(error);
  std::vector<long> starts;
  if (llvm::Error error = readIntegers("RESIDUE_POINTER", nres, starts))
    return std::move(error);
  for (long start : starts) {
    if (start < 1 || start > natom)
      return fail("RESIDUE_POINTER is out of range");
    topology.residueStarts.push_back(start - 1);
  }

  if (llvm::Error error =
          readIntegers("NUMBER_EXCLUDED_ATOMS", natom, numExcluded))
    return std::move(error);
  if (llvm::Error error =
          readIntegers("EXCLUDED_ATOMS_LIST", p[10], excludedList))
    return std::move(error);

  if (llvm::Error error = readTypes(topology, ntypes, p[19]))
    return std::move(error);
  if (llvm::Error error = readTerms(topology, p))
    return std::move(error);
  return std::move(topology);
}

llvm::Expected<Topology> mdir::driver::readAmberTopology(StringRef path) {
  return Reader(path).read();
}

//===----------------------------------------------------------------------===//
// Coordinates
//===----------------------------------------------------------------------===//

llvm::Error mdir::driver::readAmberCoordinates(StringRef path,
                                               Topology &topology) {
  auto fail = [&](const llvm::Twine &message) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(), "%s: %s",
                                   path.str().c_str(),
                                   message.str().c_str());
  };
  auto file = llvm::MemoryBuffer::getFile(path);
  if (!file)
    return fail("cannot read the file: " + file.getError().message());
  StringRef text = (*file)->getBuffer();
  if (text.starts_with("CDF"))
    return fail("a restart in NetCDF is not supported yet; write it in "
                "ASCII");

  llvm::SmallVector<StringRef> all;
  text.split(all, '\n');
  std::vector<StringRef> lines;
  for (StringRef line : all)
    lines.push_back(line.rtrim("\r"));
  while (!lines.empty() && lines.back().trim().empty())
    lines.pop_back();
  if (lines.size() < 2)
    return fail("expected a title and the number of atoms");

  // The number of atoms, then a time and one more number, each optional.
  llvm::SmallVector<StringRef> tokens;
  lines[1].split(tokens, ' ', -1, /*KeepEmpty=*/false);
  long natom = 0;
  if (tokens.empty() || tokens.size() > 3 ||
      tokens.front().getAsInteger(10, natom))
    return fail("cannot read the number of atoms in '" + lines[1] + "'");
  size_t count = topology.getNumParticles();
  if (natom != static_cast<long>(count))
    return fail("the file has " + llvm::Twine(natom) +
                " atoms, and the topology " + llvm::Twine(count));

  // Fields of 12 columns, six on a line.
  auto readLine = [&](StringRef line, std::vector<double> &values) {
    for (size_t start = 0; start < line.size(); start += 12) {
      StringRef field = line.substr(start, 12).trim();
      if (field.empty())
        continue;
      double value;
      if (field.contains('*') || field.getAsDouble(value))
        return false;
      values.push_back(value);
    }
    return true;
  };
  size_t block = (count + 1) / 2;
  size_t data = lines.size() - 2;
  bool hasVelocities, hasBox;
  if (count <= 2)
    return fail("a system of 2 atoms or fewer is not supported: its box "
                "and its velocities cannot be told apart");
  if (data == block + 1) {
    hasVelocities = false;
    hasBox = true;
  } else if (data == 2 * block + 1) {
    hasVelocities = hasBox = true;
  } else if (data == block || data == 2 * block) {
    return fail("the file has no box; MDIR needs a periodic cell");
  } else {
    return fail("the file has " + llvm::Twine(data) + " lines of numbers, "
                "which is not a layout of a file of coordinates");
  }

  auto readBlock = [&](size_t first, std::vector<double> &values,
                       double scale) -> llvm::Error {
    values.clear();
    for (size_t k = 0; k != block; ++k)
      if (!readLine(lines[first + k], values))
        return fail("cannot read the line " + llvm::Twine(first + k + 1) +
                    ": '" + lines[first + k] + "'");
    if (values.size() != 3 * count)
      return fail("expected " + llvm::Twine(3 * count) + " numbers, got " +
                  llvm::Twine(values.size()));
    for (double &value : values)
      value *= scale;
    return llvm::Error::success();
  };
  if (llvm::Error error =
          readBlock(2, topology.positions, nmPerAngstrom))
    return error;
  topology.velocities.clear();
  if (hasVelocities)
    if (llvm::Error error = readBlock(2 + block, topology.velocities,
                                      nmPerAngstrom * velocityFactor))
      return error;

  std::vector<double> box;
  if (hasBox && !readLine(lines.back(), box))
    return fail("cannot read the box '" + lines.back() + "'");
  if (box.size() != 6 && box.size() != 3)
    return fail("expected the lengths and the angles of the box in '" +
                lines.back() + "'");
  if (box.size() == 6)
    for (int k = 3; k != 6; ++k)
      if (std::fabs(box[k] - 90.0) > 1.0e-5 && box[k] != 0.0)
        return fail("only an orthorhombic cell is supported; the box has "
                    "the angles " +
                    show(box[3]) + ", " + show(box[4]) + ", " +
                    show(box[5]));
  for (int k = 0; k != 3; ++k)
    topology.box[k] = box[k] * nmPerAngstrom;
  return llvm::Error::success();
}
