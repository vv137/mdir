// The particles of a run.

#include "mdir/Driver/System.h"

#include "mdir/Driver/Cell.h"
#include "mdir/Driver/Selection.h"

#include <algorithm>

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MemoryBuffer.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>

using namespace mdir::driver;
using llvm::StringRef;

static llvm::Error findSettles(const Control &control, Topology &topology);
/// The residues named in 'water_residues', or WAT or TIP3, that are an
/// oxygen and two hydrogens.
static size_t countWaters(const Control &control, const Topology &topology);
static llvm::Error checkSettles(Topology &topology);
static llvm::Error findShakes(Topology &topology);
static llvm::Error findCenters(TupleTerm &term, const Topology &topology);
static llvm::Error placeCell(const Control &control, System &system);
static llvm::Error resolveParticleParameters(const Control &control,
                                             const Topology &topology,
                                             System &system);

/// The radii of mbondi2 [Onufriev2004], the radii of Bondi [Bondi1964] with
/// 1.3 Å for a hydrogen bonded to a nitrogen, in nm, and the screening
/// factors of Hawkins, Cramer, and Truhlar for H, C, N, O, P, and S
/// [Hawkins1996], 0.88 for F, and 0.8 otherwise, the values that Amber's
/// tleap and OpenMM's app layer assign (D152), for a topology without its
/// own.
static void assignBornRadii(Topology &topology) {
  size_t count = topology.getNumParticles();
  std::vector<int> partner(count, -1);
  for (const Topology::Bond &bond : topology.bonds) {
    if (partner[bond.i] < 0)
      partner[bond.i] = static_cast<int>(bond.j);
    if (partner[bond.j] < 0)
      partner[bond.j] = static_cast<int>(bond.i);
  }
  topology.bornRadii.assign(count, 0.0);
  topology.bornScreens.assign(count, 0.0);
  for (size_t i = 0; i != count; ++i) {
    int number = topology.atomicNumbers[i];
    double radius = 1.5, screen = 0.8;
    switch (number) {
    case 1:
      radius = partner[i] >= 0 && topology.atomicNumbers[partner[i]] == 7
                   ? 1.3
                   : 1.2;
      screen = 0.85;
      break;
    case 6:
      radius = 1.7;
      screen = 0.72;
      break;
    case 7:
      radius = 1.55;
      screen = 0.79;
      break;
    case 8:
      radius = 1.5;
      screen = 0.85;
      break;
    case 9:
      radius = 1.5;
      screen = 0.88;
      break;
    case 14:
      radius = 2.1;
      break;
    case 15:
      radius = 1.85;
      screen = 0.86;
      break;
    case 16:
      radius = 1.8;
      screen = 0.96;
      break;
    case 17:
      radius = 1.7;
      break;
    }
    topology.bornRadii[i] = radius * units::length;
    topology.bornScreens[i] = screen;
  }
}

/// The largest screened radius of generalized Born, S (ρ − 0.009 nm), in nm.
static double getLargestScreenedRadius(const Topology &topology) {
  double largest = 0.0;
  for (size_t i = 0, e = topology.bornRadii.size(); i != e; ++i)
    largest = std::max(largest,
                       topology.bornScreens[i] * (topology.bornRadii[i] - 0.009));
  return largest;
}

/// The warnings of the thermostat (D158), for a system from a topology or
/// a PDB file.
static void addThermostatWarnings(const Control &control,
                                  System &system) {
  // A Nose-Hoover chain that acts once a period of N_T steps follows its
  // period tau_T only if tau_T is many periods long, at least 20 (D163a).
  if (control.minimize || !control.isNoseHoover())
    return;
  double period =
      static_cast<double>(control.getCouplingPeriod()) * control.timestep;
  if (control.tauT >= 20.0 * period)
    return;
  char text[256];
  std::snprintf(text, sizeof text,
                "the Nose-Hoover chain acts every %g ps and its "
                "'time_constant' is %g ps, less than 20 times that; give a "
                "smaller 'interval' in [thermostat] or a 'time_constant' of "
                "at least %g ps",
                period, control.tauT, 20.0 * period);
  system.warnings.push_back({"short_thermostat_period", text});
}

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
  if (charmm && control.periodic) {
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
  // The waters of a topology of Amber or CHARMM run flexible unless the
  // control file says how: unlike [ settles ] of GROMACS, the topology does
  // not tell whether they are meant rigid, and a step of 2 fs needs them so.
  size_t flexibleWaters = 0;
  if ((!control.prmtopFile.empty() || charmm) && !control.fastWater &&
      !control.statesFlexible)
    flexibleWaters = countWaters(control, *topology);
  if (control.rigidBonds)
    if (llvm::Error error = findShakes(*topology))
      return std::move(error);

  System system;
  system.warnings = control.warnings;
  // A step longer than 1 fs moves hydrogens too far unless their bonds are
  // constrained (D158).
  if (!control.minimize && control.timestep > 0.00101 &&
      !control.rigidBonds &&
      llvm::is_contained(topology->atomicNumbers, 1))
  {
    char femtoseconds[32];
    std::snprintf(femtoseconds, sizeof femtoseconds, "%g",
                  control.timestep * 1000.0);
    system.warnings.push_back(
        {"long_time_step",
         std::string("a time step of ") + femtoseconds +
             " fs with the bonds of hydrogen free; give 'hydrogen_bonds = "
             "true' in [constraints], or a step of 1 fs"});
  }
  addThermostatWarnings(control, system);
  // Brownian dynamics moves an atom by sqrt(2 k_B T dt / (m gamma)) a step
  // along each axis, and by dt F / (m gamma) with the force: the lightest
  // atoms of constrained bonds move furthest. Beyond 0.005 nm, a twentieth
  // of a bond of hydrogen, the constraints may fail (D163b).
  if (!control.minimize && control.isBrownian() &&
      (control.rigidBonds || !topology->settles.empty())) {
    double lightest = 0.0;
    for (double mass : topology->masses)
      if (mass > 0.0 && (lightest == 0.0 || mass < lightest))
        lightest = mass;
    double kT = units::boltzmann * control.temperature;
    double limit = 0.005;
    double spread =
        std::sqrt(2.0 * kT * control.timestep / (lightest * control.friction));
    if (lightest > 0.0 && spread > limit) {
      char text[320];
      std::snprintf(text, sizeof text,
                    "Brownian dynamics with constraints moves an atom of "
                    "mass %g by %.3g nm a step at random, more than %g nm; "
                    "constraints may fail; give a 'time_step' of at most "
                    "%.3g ps or a larger 'friction'",
                    lightest, spread, limit,
                    limit * limit * lightest * control.friction / (2.0 * kT));
      system.warnings.push_back({"brownian_step", text});
    }
  }
  if (flexibleWaters > 0)
    system.warnings.push_back(
        {"flexible_water",
         std::to_string(flexibleWaters) +
             " waters of the topology run flexible; give 'rigid_water = "
             "true' in [constraints] to hold them rigid with SETTLE, as a "
             "time step of 2 fs needs, or 'rigid_water = false' to keep "
             "them flexible"});
  if (llvm::Error error = resolveParticleParameters(control, *topology, system))
    return std::move(error);
  // The value of the parameter `stem` of each particle, or nothing.
  auto particleValues =
      [&](StringRef stem) -> const std::vector<double> * {
    for (const auto &[name, values] : system.particleParameters)
      if (name == stem)
        return &values;
    return nullptr;
  };

  // The terms over tuples of the control file join those of the topology
  // (D136), once their particles are known to exist.
  for (TupleTerm term : control.tupleTerms) {
    for (unsigned particle : term.particles)
      if (particle >= topology->getNumParticles())
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "the term '%s' names the particle %u, and the topology has %zu",
            term.name.c_str(), particle + 1, topology->getNumParticles());
    if (term.isCentroid())
      if (llvm::Error error = findCenters(term, *topology))
        return std::move(error);
    // A parameter of each particle at a place, `w2`, becomes one of each
    // tuple, the value of its second particle (D165).
    if (!term.isCentroid() && !system.particleParameters.empty()) {
      Expression expression = llvm::cantFail(
          Expression::parse(term.expression, control.functions));
      for (const std::string &name : expression.getNames()) {
        if (name.size() < 2 || !llvm::isDigit(name.back()))
          continue;
        const std::vector<double> *values =
            particleValues(StringRef(name).drop_back());
        unsigned place = name.back() - '1';
        if (!values || place >= term.arity ||
            llvm::any_of(term.parameters,
                         [&](const auto &p) { return p.first == name; }))
          continue;
        std::vector<double> column;
        for (size_t k = 0, e = term.size(); k != e; ++k)
          column.push_back((*values)[term.particles[k * term.arity + place]]);
        term.parameters.push_back({name, std::move(column)});
      }
    }
    topology->tupleTerms.push_back(std::move(term));
  }

  // The terms of the absolute positions (D148): their particles, by a mask
  // or by number, and a value of each parameter for each.
  for (ExternalTerm term : control.externalTerms) {
    if (!term.selection.empty()) {
      auto selected = selectParticles(term.selection, *topology);
      if (!selected)
        return selected.takeError();
      for (size_t i = 0, e = selected->size(); i != e; ++i)
        if ((*selected)[i])
          term.particles.push_back(static_cast<unsigned>(i));
      // A selection of nothing leaves the term without particles: a warning,
      // not an error, since the term adds nothing (D158).
      if (term.particles.empty())
        system.warnings.push_back(
            {"empty_selection", "the selection '" + term.selection +
                                    "' of the term '" + term.name +
                                    "' selects no particle; the term adds "
                                    "nothing"});
      for (const auto &[name, values] : term.parameters)
        if (values.size() != term.particles.size())
          return llvm::createStringError(
              llvm::inconvertibleErrorCode(),
              "the parameter '%s' of the term '%s' has %zu values, and its "
              "selection '%s' selects %zu particles",
              name.c_str(), term.name.c_str(), values.size(),
              term.selection.c_str(), term.particles.size());
    }
    for (unsigned particle : term.particles)
      if (particle >= topology->getNumParticles())
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "the term '%s' names the particle %u, and the topology has %zu",
            term.name.c_str(), particle + 1, topology->getNumParticles());
    // A parameter of each particle, by its name, becomes one of the term
    // (D165).
    if (!system.particleParameters.empty()) {
      Expression expression = llvm::cantFail(
          Expression::parse(term.expression, control.functions));
      for (const std::string &name : expression.getNames())
        if (const std::vector<double> *values = particleValues(name)) {
          std::vector<double> column;
          for (unsigned particle : term.particles)
            column.push_back((*values)[particle]);
          term.parameters.push_back({name, std::move(column)});
        }
    }
    topology->externalTerms.push_back(std::move(term));
  }

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
  if (!control.restraints.empty()) {
    system.restraintConstants.assign(system.masses.size(), 0.0);
    system.restraintScaling.assign(system.masses.size(),
                                   ReferenceScaling::Center);
  }
  for (const Control::Restraint &restraint : control.restraints) {
    auto selected = selectParticles(restraint.selection, *topology);
    if (!selected)
      return selected.takeError();
    size_t count = 0;
    for (size_t i = 0, e = system.masses.size(); i != e; ++i) {
      if (!(*selected)[i] || system.masses[i] == 0.0)
        continue;
      if (system.restraintConstants[i] > 0.0 &&
          system.restraintScaling[i] != restraint.scaling)
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "the restraint of '%s' selects particle %zu, which another "
            "restraint selects with another 'reference_scaling'",
            restraint.selection.c_str(), i + 1);
      system.restraintScaling[i] = restraint.scaling;
      system.restraintConstants[i] += restraint.forceConstant *
                                      units::energy /
                                      (units::length * units::length);
      ++count;
    }
    if (count == 0)
      system.warnings.push_back(
          {"empty_selection", "the restraint of '" + restraint.selection +
                                  "' selects no particle with mass; it "
                                  "restrains nothing"});
  }

  // The interaction groups of the pair terms, by their masks (D137).
  for (const PairTerm &term : control.pairs) {
    std::vector<std::vector<bool>> groups;
    for (const std::string &mask : term.groups) {
      auto selected = selectParticles(mask, *topology);
      if (!selected)
        return selected.takeError();
      if (llvm::none_of(*selected, [](bool b) { return b; }))
        system.warnings.push_back(
            {"empty_selection", "the group '" + mask + "' of the pair term '" +
                                    term.name +
                                    "' selects no particle; the term adds "
                                    "nothing"});
      groups.push_back(std::move(*selected));
    }
    system.pairGroups.push_back(std::move(groups));
    system.pairTermNames.push_back(term.name);
  }

  // Generalized Born (D144) takes the radii and the screening of the
  // topology, or those of the rules of mbondi2 by element (D152).
  if (control.implicitSolvent != Control::ImplicitSolvent::None) {
    if (control.bornRadii == Control::BornRadii::MBondi2)
      assignBornRadii(*topology);
    if (topology->bornRadii.size() != topology->getNumParticles() ||
        topology->bornScreens.size() != topology->getNumParticles())
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "generalized Born needs the radii and the screening of every "
          "particle, the sections RADII and SCREEN of a topology of Amber; "
          "for another topology give 'born_radii = \"MBONDI2\"'");
    // The descreening of a cut integral reaches its cutoff and the screened
    // radius beyond it, which the pairs within the cutoff must hold.
    if (control.bornRadiusCutoff > 0.0) {
      double reach = control.bornRadiusCutoff * units::length +
                     getLargestScreenedRadius(*topology);
      if (reach > control.cutoffDistance * units::length)
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "'born_radius_cutoff' and the largest screened radius reach "
            "%g Å, beyond the cutoff, %g Å",
            reach / units::length, control.cutoffDistance);
    }
    system.bornTermNames.push_back("generalized Born");
    if (control.surfaceAreaEnergy > 0.0)
      system.bornTermNames.push_back("nonpolar surface");
  }

  system.topology = std::make_shared<Topology>(std::move(*topology));
  system.keepsMomentum =
      (!control.isLangevin() && !control.isBrownian()) || control.comPeriod > 0;
  addThermostatWarnings(control, system);
  if (llvm::Error error = placeCell(control, system))
    return std::move(error);
  return std::move(system);
}

/// The parameters of each particle (D165): the entries of the control file,
/// in its order, each over its particles, into a value for every particle
/// of `topology` in `system`. Every value of a parameter comes from here.
static llvm::Error resolveParticleParameters(const Control &control,
                                             const Topology &topology,
                                             System &system) {
  for (const ParticleParameter &entry : control.particleParameters) {
    size_t count = topology.getNumParticles();
    auto found = llvm::find_if(system.particleParameters, [&](const auto &p) {
      return p.first == entry.name;
    });
    if (found == system.particleParameters.end()) {
      system.particleParameters.push_back(
          {entry.name, std::vector<double>(count, std::nan(""))});
      found = std::prev(system.particleParameters.end());
    }
    std::vector<double> &values = found->second;
    if (!entry.values.empty()) {
      if (entry.values.size() != count)
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "the parameter '%s' has %zu values, and the topology has %zu "
            "particles",
            entry.name.c_str(), entry.values.size(), count);
      values = entry.values;
      continue;
    }
    if (!entry.selection.empty()) {
      auto selected = selectParticles(entry.selection, topology);
      if (!selected)
        return selected.takeError();
      size_t chosen = 0;
      for (size_t i = 0; i != count; ++i)
        if ((*selected)[i]) {
          values[i] = entry.value;
          ++chosen;
        }
      if (chosen == 0)
        system.warnings.push_back(
            {"empty_selection", "the selection '" + entry.selection +
                                    "' of the parameter '" + entry.name +
                                    "' selects no particle"});
      continue;
    }
    if (!entry.particles.empty()) {
      for (unsigned particle : entry.particles) {
        if (particle >= count)
          return llvm::createStringError(
              llvm::inconvertibleErrorCode(),
              "the parameter '%s' names the particle %u, and the topology "
              "has %zu",
              entry.name.c_str(), particle + 1, count);
        values[particle] = entry.value;
      }
      continue;
    }
    values.assign(count, entry.value);
  }
  for (const auto &[name, values] : system.particleParameters)
    for (size_t i = 0, e = values.size(); i != e; ++i)
      if (std::isnan(values[i]))
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "no entry of the parameter '%s' gives a value to particle %zu; "
            "give an entry without 'selection' first, for every particle",
            name.c_str(), i + 1);
  return llvm::Error::success();
}

/// The settled waters of an Amber topology: every residue that
/// 'water_residues' names, with an oxygen and two hydrogens first, and
/// the distances of the bonds among them, as sander takes them.
/// The cell of a run. A periodic one is that of the file of coordinates or
/// of the control file, which must give one. Without one (D142), the
/// particles are put in a cell that no image reaches: along each axis the
/// extent of the particles and three times the reach of the neighbor
/// structures, so that an image is farther than that reach while the
/// particles spread less than twice it (Output, checkLimits).
static llvm::Error placeCell(const Control &control, System &system) {
  if (control.periodic) {
    if (!(system.box[0] > 0.0 && system.box[1] > 0.0 && system.box[2] > 0.0))
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "the file of coordinates has no periodic cell; for a run without "
          "one give 'type = \"NONE\"' in [boundary]");
    return llvm::Error::success();
  }
  double reach = control.pairlistDistance * units::length;
  const std::vector<double> &x = system.positions;
  for (int k = 0; k != 3; ++k) {
    double least = x[k], most = x[k];
    for (size_t i = k; i < x.size(); i += 3) {
      least = std::min(least, x[i]);
      most = std::max(most, x[i]);
    }
    system.box[k] = most - least + 3.0 * reach;
    system.tilt[k] = 0.0;
  }
  return llvm::Error::success();
}

/// The groups of a term over their centers (D139): the particles of each
/// mask, their weights, and the particle in the middle of them by number,
/// from which the others are taken in the minimum image. That gives the
/// center only while every particle of the group lies within half the cell
/// of it along each edge. The coordinates of the input must show each
/// within 0.45 of the cell, a tenth of the half to spare, as the minimum
/// image cannot tell a particle beyond the half from one within it.
static llvm::Error findCenters(TupleTerm &term, const Topology &topology) {
  const double *box = topology.box, *tilt = topology.tilt;
  bool periodic = box[0] > 0.0 && box[1] > 0.0 && box[2] > 0.0;
  const std::vector<double> &x = topology.positions;
  for (const std::string &mask : term.groups) {
    auto selected = selectParticles(mask, topology);
    if (!selected)
      return selected.takeError();
    TupleTerm::Center center;
    double total = 0.0;
    for (size_t i = 0, e = selected->size(); i != e; ++i) {
      if (!(*selected)[i])
        continue;
      double weight = term.massWeighted ? topology.masses[i] : 1.0;
      center.members.push_back(static_cast<unsigned>(i));
      center.weights.push_back(weight);
      total += weight;
    }
    if (center.members.empty())
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "the group '%s' of the term '%s' selects no particle",
          mask.c_str(), term.name.c_str());
    if (!(total > 0.0))
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "the group '%s' of the term '%s' has no mass to weigh its center "
          "by; give 'weighting = \"NONE\"'",
          mask.c_str(), term.name.c_str());
    for (double &weight : center.weights)
      weight /= total;
    center.reference = center.members[(center.members.size() - 1) / 2];
    unsigned r = center.reference;
    for (unsigned i : center.members) {
      if (!periodic)
        break;
      // The displacement in fractions of the edges a = (a_x, 0, 0),
      // b = (b_x, b_y, 0), c = (c_x, c_y, c_z), in the minimum image.
      double d[3];
      for (int k = 0; k != 3; ++k)
        d[k] = x[3 * i + k] - x[3 * r + k];
      double fc = d[2] / box[2];
      fc -= std::round(fc);
      double fb = (d[1] - fc * tilt[2]) / box[1];
      fb -= std::round(fb);
      double fa = (d[0] - fc * tilt[1] - fb * tilt[0]) / box[0];
      fa -= std::round(fa);
      double most = std::max({std::abs(fa), std::abs(fb), std::abs(fc)});
      if (most > 0.45)
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "the particle %u of the group '%s' of the term '%s' is %.3f of "
            "the cell along an edge from the particle %u, from which the "
            "center is taken in the minimum image; a group must lie within "
            "0.45 of the cell around it",
            i + 1, mask.c_str(), term.name.c_str(), most, r + 1);
    }
    term.centers.push_back(std::move(center));
  }
  return llvm::Error::success();
}

static size_t countWaters(const Control &control, const Topology &topology) {
  bool charmm = !control.charmmStructureFile.empty();
  std::vector<std::string> residues = control.settleResidues;
  if (residues.empty())
    residues = {charmm ? "TIP3" : "WAT"};
  size_t count = topology.getNumParticles(), waters = 0;
  for (size_t r = 0, e = topology.residueNames.size(); r != e; ++r) {
    if (!llvm::is_contained(residues,
                            StringRef(topology.residueNames[r]).trim()))
      continue;
    unsigned first = topology.residueStarts[r];
    unsigned end = r + 1 < e ? topology.residueStarts[r + 1] : count;
    waters += end - first >= 3 && topology.atomicNumbers[first] == 8 &&
              topology.atomicNumbers[first + 1] == 1 &&
              topology.atomicNumbers[first + 2] == 1;
  }
  return waters;
}

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
  system.keepsMomentum =
      (!control.isLangevin() && !control.isBrownian()) || control.comPeriod > 0;
  addThermostatWarnings(control, system);
  if (llvm::Error error = placeCell(control, system))
    return std::move(error);
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
