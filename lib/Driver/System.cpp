// The particles of a run.

#include "mdir/Driver/System.h"

#include "mdir/Driver/Cell.h"
#include "mdir/Driver/Selection.h"

#include <algorithm>

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MemoryBuffer.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <map>

using namespace mdir::driver;
using llvm::StringRef;

static llvm::Error findSettles(const Control &control, Topology &topology);
static llvm::Error checkSettles(Topology &topology);
static llvm::Error findShakes(Topology &topology);

/// The system of a topology and a file of coordinates.
static llvm::Expected<System> readTopologySystem(const Control &control) {
  bool charmm = !control.charmmStructureFile.empty();
  llvm::Expected<Topology> topology =
      charmm ? readCharmmTopology(control.charmmStructureFile,
                                  control.charmmParameterFiles)
      : control.prmtopFile.empty()
          ? readGromacsTopology(control.gromacsTopologyFile,
                                control.gromacsIncludes,
                                control.gromacsDefines)
          : readAmberTopology(control.prmtopFile);
  if (!topology)
    return topology.takeError();
  if (llvm::Error error =
          charmm ? readCharmmCoordinates(control.charmmCoordinateFile,
                                         *topology)
          : control.prmtopFile.empty()
              ? readGromacsCoordinates(control.gromacsCoordinateFile,
                                       *topology)
              : readAmberCoordinates(control.amberCoordinateFile, *topology))
    return std::move(error);
  // A coordinate file of CHARMM has no cell; the control file gives it.
  // CHARMM keeps a cell as the symmetric root of its metric, whose frame
  // is turned from the one of MDIR; its positions are turned with it
  // (docs/triclinic-m2.md, Section 1).
  if (charmm) {
    llvm::Expected<Cell> cell =
        makeCell(control.box[0] * 0.1, control.box[1] * 0.1,
                 control.box[2] * 0.1, control.angles[0], control.angles[1],
                 control.angles[2]);
    if (!cell)
      return cell.takeError();
    for (int k = 0; k != 3; ++k) {
      topology->box[k] = cell->diagonal[k];
      topology->tilt[k] = cell->tilt[k];
    }
    if (!cell->isOrthorhombic()) {
      std::array<double, 9> r = getSymmetricFrameRotation(*cell);
      std::vector<double> &x = topology->positions;
      for (size_t i = 0; i + 2 < x.size(); i += 3) {
        double s[3] = {x[i], x[i + 1], x[i + 2]};
        for (int k = 0; k != 3; ++k)
          x[i + k] = s[0] * r[3 * k] + s[1] * r[3 * k + 1] +
                     s[2] * r[3 * k + 2];
      }
    }
  }

  // The waters that SETTLE constrains (D63): those of [ settles ] of
  // GROMACS, and the residues of Amber or CHARMM named in 'water_residues'.
  // A run leaves them flexible only when it says so, and they then need
  // bonds.
  if (control.fastWater) {
    if (!control.prmtopFile.empty() || charmm)
      if (llvm::Error error = findSettles(control, *topology))
        return std::move(error);
  } else if (!topology->settles.empty()) {
    if (!control.statesFlexible)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "the topology has %zu waters with SETTLE; set 'rigid_water = true' "
          "in [constraints] to constrain them, or 'rigid_water = false' to "
          "run them flexible, with their bonds",
          topology->settles.size());
    topology->settles.clear();
  }
  if (llvm::Error error = checkSettles(*topology))
    return std::move(error);
  if (control.rigidBonds)
    if (llvm::Error error = findShakes(*topology))
      return std::move(error);

  System system;
  system.types = topology->types;
  system.masses = topology->masses;
  system.positions = topology->positions;
  system.velocities = topology->velocities;
  system.givenVelocities = !system.velocities.empty();
  system.numConstraints = 3 * topology->settles.size();
  for (const Topology::Shake &shake : topology->shakes)
    system.numConstraints += shake.hydrogens.size();
  if (system.velocities.empty())
    system.velocities.assign(system.positions.size(), 0.0);
  for (int i = 0; i != 3; ++i) {
    system.box[i] = topology->box[i];
    system.tilt[i] = topology->tilt[i];
  }

  // The restraints: their constants add where their selections overlap.
  // Particles without mass, the virtual sites, are placed, not restrained.
  if (!control.restraints.empty())
    system.restraintConstants.assign(system.masses.size(), 0.0);
  for (const Control::Restraint &restraint : control.restraints) {
    auto selected = selectParticles(restraint.selection, *topology);
    if (!selected)
      return selected.takeError();
    size_t count = 0;
    for (size_t i = 0, e = system.masses.size(); i != e; ++i) {
      if (!(*selected)[i] || system.masses[i] == 0.0)
        continue;
      system.restraintConstants[i] += restraint.forceConstant *
                                      units::energy /
                                      (units::length * units::length);
      ++count;
    }
    if (count == 0)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "the restraint of '%s' selects no particle with mass",
          restraint.selection.c_str());
  }

  system.topology = std::make_shared<Topology>(std::move(*topology));
  return std::move(system);
}

/// The settled waters of an Amber topology: every residue that
/// 'water_residues' names, with an oxygen and two hydrogens first, and
/// the distances of the bonds among them, as sander takes them.
static llvm::Error findSettles(const Control &control, Topology &topology) {
  bool charmm = !control.charmmStructureFile.empty();
  const std::string &path =
      charmm ? control.charmmStructureFile : control.prmtopFile;
  auto fail = [&](const llvm::Twine &message) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(), "%s: %s",
                                   path.c_str(), message.str().c_str());
  };
  std::vector<std::string> residues = control.settleResidues;
  if (residues.empty())
    residues = {charmm ? "TIP3" : "WAT"};
  std::map<std::pair<unsigned, unsigned>, double> lengths;
  for (const Topology::Bond &bond : topology.bonds)
    lengths[{std::min(bond.i, bond.j), std::max(bond.i, bond.j)}] = bond.r0;
  std::vector<bool> site(topology.getNumParticles(), false);
  for (const Topology::VirtualSite &s : topology.virtualSites)
    site[s.site] = true;
  size_t count = topology.getNumParticles();
  for (size_t r = 0, e = topology.residueNames.size(); r != e; ++r) {
    StringRef name = StringRef(topology.residueNames[r]).trim();
    if (!llvm::is_contained(residues, name))
      continue;
    unsigned first = topology.residueStarts[r];
    unsigned end = r + 1 < e ? topology.residueStarts[r + 1] : count;
    std::string where =
        ("the residue " + llvm::Twine(r + 1) + " (" + name + ")").str();
    bool water = end - first >= 3 && topology.atomicNumbers[first] == 8 &&
                 topology.atomicNumbers[first + 1] == 1 &&
                 topology.atomicNumbers[first + 2] == 1;
    for (unsigned i = first + 3; i < end; ++i)
      water = water && site[i];
    if (!water)
      return fail(where + " is named in 'water_residues', but it is not "
                          "an oxygen and two hydrogens, with at most virtual "
                          "sites after them");
    unsigned o = first, h1 = first + 1, h2 = first + 2;
    auto length = [&](unsigned a, unsigned b) {
      auto found = lengths.find({a, b});
      return found == lengths.end() ? -1.0 : found->second;
    };
    double oh1 = length(o, h1), oh2 = length(o, h2), hh = length(h1, h2);
    if (oh1 < 0.0 || oh2 < 0.0 || hh < 0.0)
      return fail(where + " lacks one of the bonds O-H1, O-H2, and H1-H2, "
                          "whose lengths SETTLE keeps");
    // sander stops if the two O-H differ by more than 1e-4 Å.
    if (std::fabs(oh1 - oh2) > 1.0e-5)
      return fail(where + " has two different lengths of O-H");
    topology.settles.push_back({o, oh1, hh});
  }
  return llvm::Error::success();
}

/// Checks the settled waters, and drops their bonds and angles, which the
/// constraints keep at their lengths.
static llvm::Error checkSettles(Topology &topology) {
  std::vector<int> water(topology.getNumParticles(), -1);
  for (auto [index, settle] : llvm::enumerate(topology.settles)) {
    unsigned o = settle.oxygen;
    if (o + 2 >= topology.getNumParticles())
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "a water of SETTLE is out of range");
    if (topology.masses[o + 1] != topology.masses[o + 2])
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "the hydrogens of the water of SETTLE at atom %u have different "
          "masses",
          o + 1);
    for (unsigned k = 0; k != 3; ++k)
      water[o + k] = index;
  }
  auto inside = [&](std::initializer_list<unsigned> members) {
    int w = water[*members.begin()];
    return w >= 0 && llvm::all_of(members, [&](unsigned m) {
             return water[m] == w;
           });
  };
  llvm::erase_if(topology.bonds, [&](const Topology::Bond &bond) {
    return inside({bond.i, bond.j});
  });
  llvm::erase_if(topology.angles, [&](const Topology::Angle &angle) {
    return inside({angle.i, angle.j, angle.k});
  });
  llvm::erase_if(topology.ureyBradleys,
                 [&](const Topology::UreyBradley &term) {
                   return inside({term.i, term.k});
                 });
  return llvm::Error::success();
}

/// The bonds of hydrogen that SHAKE keeps, grouped by their heavy atom;
/// their bond terms are dropped. The bonds of the waters of SETTLE are gone
/// already.
static llvm::Error findShakes(Topology &topology) {
  auto fail = [](const llvm::Twine &message) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(), "%s",
                                   message.str().c_str());
  };
  auto isHydrogen = [&](unsigned atom) {
    int number = topology.atomicNumbers[atom];
    return number > 0 ? number == 1 : topology.masses[atom] < 1.2;
  };
  std::map<unsigned, size_t> groupOf;
  std::vector<bool> taken(topology.getNumParticles(), false);
  for (const Topology::Bond &bond : topology.bonds) {
    if (!bond.hydrogen)
      continue;
    bool first = isHydrogen(bond.i), second = isHydrogen(bond.j);
    if (first && second)
      return fail("the hydrogens " +
                  llvm::Twine(std::min(bond.i, bond.j) + 1) + " and " +
                  llvm::Twine(std::max(bond.i, bond.j) + 1) +
                  " are bonded to each other; "
                  "with 'hydrogen_bonds = true' such a water needs "
                  "'rigid_water = true'");
    unsigned heavy = first ? bond.j : bond.i;
    unsigned hydrogen = first ? bond.i : bond.j;
    if (taken[hydrogen])
      return fail("the hydrogen " + llvm::Twine(hydrogen + 1) +
                  " has two bonds that SHAKE would keep");
    taken[hydrogen] = true;
    auto [entry, inserted] =
        groupOf.insert({heavy, topology.shakes.size()});
    if (inserted)
      topology.shakes.push_back({heavy, {}, {}});
    Topology::Shake &group = topology.shakes[entry->second];
    if (group.hydrogens.size() == 3)
      return fail("the atom " + llvm::Twine(heavy + 1) +
                  " has more than three hydrogens; SHAKE takes groups of "
                  "at most three");
    group.hydrogens.push_back(hydrogen);
    group.lengths.push_back(bond.r0);
  }
  for (const Topology::Shake &group : topology.shakes)
    if (taken[group.center])
      return fail("the atom " + llvm::Twine(group.center + 1) +
                  " is a hydrogen of one group of SHAKE and the center of "
                  "another");
  llvm::erase_if(topology.bonds, [&](const Topology::Bond &bond) {
    return bond.hydrogen;
  });
  return llvm::Error::success();
}

llvm::Expected<System> mdir::driver::readSystem(const Control &control) {
  if (control.hasTopology())
    return readTopologySystem(control);
  auto file = llvm::MemoryBuffer::getFile(control.pdbFile);
  if (!file)
    return llvm::createStringError(file.getError(), "cannot read '%s'",
                                   control.pdbFile.c_str());

  System system;
  for (int i = 0; i != 3; ++i)
    system.box[i] = control.box[i] * units::length;

  unsigned number = 0;
  StringRef rest = (*file)->getBuffer();
  while (!rest.empty()) {
    StringRef line;
    std::tie(line, rest) = rest.split('\n');
    ++number;
    if (!line.starts_with("ATOM") && !line.starts_with("HETATM"))
      continue;
    auto fail = [&](const llvm::Twine &message) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "%s:%u: %s", control.pdbFile.c_str(),
                                     number, message.str().c_str());
    };
    if (line.size() < 54)
      return fail("the line is too short to hold the coordinates");

    // Columns 13 to 16 hold the name, 31 to 54 the coordinates.
    StringRef name = line.substr(12, 4).trim();
    unsigned type = 0;
    while (type != control.types.size() && control.types[type].name != name)
      ++type;
    if (type == control.types.size())
      return fail("no [[energy.type]] is named '" + name + "'");

    for (int i = 0; i != 3; ++i) {
      StringRef field = line.substr(30 + 8 * i, 8).trim();
      double value;
      if (field.getAsDouble(value))
        return fail("expected a coordinate, got '" + field + "'");
      system.positions.push_back(value * units::length);
    }
    system.types.push_back(type);
    system.masses.push_back(control.types[type].mass);
  }

  if (system.types.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "%s: the file holds no atoms",
                                   control.pdbFile.c_str());
  if (!control.restraints.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "restraints select particles by the names of a topology; a run from "
        "a PDB file has none");
  system.velocities.assign(system.positions.size(), 0.0);
  return std::move(system);
}

namespace {

/// A generator of random numbers whose sequence is fixed by its definition,
/// not by a library: splitmix64 for the state, xoshiro256** for the numbers.
/// See Steele et al., OOPSLA '14, 453 (2014), and Blackman and Vigna, ACM
/// Trans. Math. Softw. 47(4), 1 (2021); docs/references.md.
class Generator {
public:
  explicit Generator(uint64_t seed) {
    for (uint64_t &word : state) {
      seed += 0x9e3779b97f4a7c15ull;
      uint64_t z = seed;
      z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
      z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
      word = z ^ (z >> 31);
    }
  }

  /// A number in (0, 1].
  double nextUniform() {
    auto rotate = [](uint64_t x, int k) { return (x << k) | (x >> (64 - k)); };
    uint64_t result = rotate(state[1] * 5, 7) * 9;
    uint64_t t = state[1] << 17;
    state[2] ^= state[0];
    state[3] ^= state[1];
    state[1] ^= state[2];
    state[0] ^= state[3];
    state[2] ^= t;
    state[3] = rotate(state[3], 45);
    return (static_cast<double>(result >> 11) + 1.0) * 0x1.0p-53;
  }

  /// A number from the normal distribution, by the method of Box and
  /// Muller, Ann. Math. Stat. 29, 610 (1958).
  double nextNormal() {
    double radius = std::sqrt(-2.0 * std::log(nextUniform()));
    double angle = 2.0 * M_PI * nextUniform();
    return radius * std::cos(angle);
  }

private:
  uint64_t state[4];
};

} // namespace

double mdir::driver::getKineticEnergy(const System &system) {
  double twice = 0.0;
  for (size_t i = 0, e = system.getNumParticles(); i != e; ++i)
    for (int c = 0; c != 3; ++c) {
      double v = system.velocities[3 * i + c];
      twice += system.masses[i] * v * v;
    }
  return 0.5 * twice;
}

/// Removes from the velocities of the atoms of `pairs`, a group of
/// constrained bonds, their parts along the bonds: the impulses along the
/// bonds that solve the linear equations of the constraints, by Gaussian
/// elimination, the matrix being symmetric and positive definite.
static void constrainVelocities(
    System &system, const std::vector<std::array<unsigned, 2>> &pairs) {
  size_t count = pairs.size();
  auto x = [&](unsigned atom, int c) {
    return system.positions[3 * atom + c];
  };
  auto v = [&](unsigned atom, int c) -> double & {
    return system.velocities[3 * atom + c];
  };
  auto inverseMass = [&](unsigned atom) { return 1.0 / system.masses[atom]; };
  std::vector<std::array<double, 3>> unit(count);
  std::vector<double> rhs(count);
  for (size_t c = 0; c != count; ++c) {
    auto [p, q] = pairs[c];
    double norm = 0.0;
    for (int k = 0; k != 3; ++k) {
      unit[c][k] = x(q, k) - x(p, k);
      norm += unit[c][k] * unit[c][k];
    }
    norm = std::sqrt(norm);
    rhs[c] = 0.0;
    for (int k = 0; k != 3; ++k) {
      unit[c][k] /= norm;
      rhs[c] -= unit[c][k] * (v(q, k) - v(p, k));
    }
  }
  // A[c][d] = e_c · (the change of v_q − v_p of c by a unit impulse of d).
  std::vector<std::vector<double>> A(count, std::vector<double>(count));
  for (size_t c = 0; c != count; ++c)
    for (size_t d = 0; d != count; ++d) {
      double dot = 0.0;
      for (int k = 0; k != 3; ++k)
        dot += unit[c][k] * unit[d][k];
      auto [p, q] = pairs[c];
      auto [r, s] = pairs[d];
      double weight = (q == s ? inverseMass(q) : 0.0) -
                      (q == r ? inverseMass(q) : 0.0) -
                      (p == s ? inverseMass(p) : 0.0) +
                      (p == r ? inverseMass(p) : 0.0);
      A[c][d] = dot * weight;
    }
  for (size_t c = 0; c != count; ++c)
    for (size_t r = c + 1; r != count; ++r) {
      double factor = A[r][c] / A[c][c];
      for (size_t d = c; d != count; ++d)
        A[r][d] -= factor * A[c][d];
      rhs[r] -= factor * rhs[c];
    }
  std::vector<double> impulse(count);
  for (size_t c = count; c-- != 0;) {
    double sum = rhs[c];
    for (size_t d = c + 1; d != count; ++d)
      sum -= A[c][d] * impulse[d];
    impulse[c] = sum / A[c][c];
  }
  for (size_t c = 0; c != count; ++c) {
    auto [p, q] = pairs[c];
    for (int k = 0; k != 3; ++k) {
      v(q, k) += impulse[c] * unit[c][k] * inverseMass(q);
      v(p, k) -= impulse[c] * unit[c][k] * inverseMass(p);
    }
  }
}

void mdir::driver::assignVelocities(const Control &control, System &system) {
  size_t count = system.getNumParticles();
  Generator generator(control.seed);
  for (size_t i = 0; i != count; ++i) {
    // A virtual site has no mass and no velocity.
    double width =
        system.masses[i] > 0.0
            ? std::sqrt(units::boltzmann * control.temperature /
                        system.masses[i])
            : 0.0;
    for (int c = 0; c != 3; ++c)
      system.velocities[3 * i + c] = width * generator.nextNormal();
  }
  if (count < 2)
    return;

  // Velocities that the constraints allow: none along a rigid bond.
  if (system.topology) {
    for (const Topology::Settle &settle : system.topology->settles) {
      unsigned o = settle.oxygen;
      constrainVelocities(system, {{o, o + 1}, {o, o + 2}, {o + 1, o + 2}});
    }
    for (const Topology::Shake &shake : system.topology->shakes) {
      std::vector<std::array<unsigned, 2>> pairs;
      for (unsigned hydrogen : shake.hydrogens)
        pairs.push_back({shake.center, hydrogen});
      constrainVelocities(system, pairs);
    }
  }

  // Bring the center of mass to rest.
  double total = 0.0, momentum[3] = {0.0, 0.0, 0.0};
  for (size_t i = 0; i != count; ++i) {
    total += system.masses[i];
    for (int c = 0; c != 3; ++c)
      momentum[c] += system.masses[i] * system.velocities[3 * i + c];
  }
  for (size_t i = 0; i != count; ++i)
    for (int c = 0; c != 3; ++c)
      system.velocities[3 * i + c] -= momentum[c] / total;

  // Scale to the temperature.
  double kinetic = getKineticEnergy(system);
  double target =
      0.5 * system.getDegreesOfFreedom() * units::boltzmann * control.temperature;
  if (kinetic > 0.0) {
    double factor = std::sqrt(target / kinetic);
    for (double &v : system.velocities)
      v *= factor;
  }
}
