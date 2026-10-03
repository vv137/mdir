// `mdir check`: reads the input of a run and prints what it describes,
// without compiling anything.

#include "Commands.h"

#include "mdir/Driver/Cell.h"
#include "mdir/Driver/Control.h"
#include "mdir/Driver/System.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdio>
#include <vector>

using namespace mdir::driver;
using llvm::StringRef;

static int fail(llvm::Error error) {
  llvm::errs() << "mdir: " << llvm::toString(std::move(error)) << "\n";
  return 1;
}

static const char *getName(Integrator integrator) {
  return integrator == Integrator::Leapfrog ? "LEAPFROG" : "VELOCITY_VERLET";
}

static const char *getName(Precision precision) {
  switch (precision) {
  case Precision::Single:
    return "single";
  case Precision::Mixed:
    return "mixed";
  case Precision::Double:
    return "double";
  }
  return "";
}

/// What a topology describes.
static int describeTopology(const Control &control, const System &system) {
  const Topology &topology = *system.topology;
  size_t count = topology.getNumParticles();
  double totalMass = 0.0, totalCharge = 0.0;
  for (size_t i = 0; i != count; ++i) {
    totalMass += topology.masses[i];
    totalCharge += topology.charges[i];
  }
  size_t impropers = 0;
  for (const Topology::Dihedral &dihedral : topology.dihedrals)
    impropers += dihedral.improper;
  size_t hydrogenBonds = 0;
  for (const Topology::Bond &bond : topology.bonds)
    hydrogenBonds += bond.hydrogen;
  double volume = topology.box[0] * topology.box[1] * topology.box[2];

  const std::string &file = !control.prmtopFile.empty()
                               ? control.prmtopFile
                           : !control.charmmStructureFile.empty()
                               ? control.charmmStructureFile
                               : control.gromacsTopologyFile;
  std::printf("topology:           %s\n", file.c_str());
  std::printf("particles:          %zu\n", count);
  std::printf("residues:           %zu\n", topology.residueNames.size());
  std::printf("types:              %zu\n", topology.getNumTypes());
  std::printf("bonds:              %zu, %zu with hydrogen\n",
              topology.bonds.size(), hydrogenBonds);
  std::printf("angles:             %zu\n", topology.angles.size());
  if (!topology.ureyBradleys.empty())
    std::printf("Urey-Bradley terms: %zu\n", topology.ureyBradleys.size());
  std::printf("dihedrals:          %zu, %zu improper\n",
              topology.dihedrals.size(), impropers);
  if (!topology.harmonicImpropers.empty())
    std::printf("harmonic impropers: %zu\n",
                topology.harmonicImpropers.size());
  std::printf("pairs 1-4:          %zu\n", topology.pairs.size());
  std::printf("excluded pairs:     %zu\n", topology.exclusions.size());
  if (!topology.cmaps.empty())
    std::printf("CMAP terms:         %zu, %zu maps\n", topology.cmaps.size(),
                topology.cmapGrids.size());
  if (!topology.virtualSites.empty())
    std::printf("virtual sites:      %zu\n", topology.virtualSites.size());
  std::printf("total charge:       %.6f e\n", totalCharge);
  std::printf("total mass:         %g amu\n", totalMass);
  Cell cell;
  for (int k = 0; k != 3; ++k) {
    cell.diagonal[k] = topology.box[k] / units::length;
    cell.tilt[k] = topology.tilt[k] / units::length;
  }
  if (cell.isOrthorhombic()) {
    std::printf("box:                %g %g %g Å\n", cell.diagonal[0],
                cell.diagonal[1], cell.diagonal[2]);
  } else {
    std::array<double, 6> shape = cell.getLengthsAndAngles();
    std::printf("box:                %g %g %g Å, angles %g %g %g\n",
                shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]);
    std::printf("cell vectors:       a (%g, 0, 0), b (%g, %g, 0), "
                "c (%g, %g, %g) Å\n",
                cell.diagonal[0], cell.tilt[0], cell.diagonal[1],
                cell.tilt[1], cell.tilt[2], cell.diagonal[2]);
  }
  std::printf("density:            %g g/cm³\n",
              totalMass / volume * 1.66053906660e-3);
  std::printf("velocities:         %s\n",
              topology.velocities.empty() ? "no" : "yes");
  std::printf("degrees of freedom: %g\n", system.getDegreesOfFreedom());
  if (!system.restraintConstants.empty()) {
    size_t restrained = 0;
    for (double k : system.restraintConstants)
      restrained += k > 0.0;
    std::printf("restrained:         %zu particles\n", restrained);
  }
  return 0;
}

/// The outputs of the run (D149): each file with its interval and the
/// number of rows, frames, or checkpoints of the whole run, and a warning
/// for each file that exists, which `mdir run` would not write over.
static void describeOutputs(const Control &control) {
  long long steps = control.numSteps;
  auto every = [&](int64_t period, const char *what) {
    std::string text = llvm::formatv("every {0} steps", period).str();
    // A run of dynamics writes the energies at its first step as well; a
    // minimization ends where it converges.
    if (!control.minimize)
      text += llvm::formatv(", {0} {1}", steps / period +
                                             (StringRef(what) == "rows"),
                            what)
                  .str();
    return text;
  };
  std::vector<std::string> written;
  std::printf("outputs:\n");
  if (control.energyPeriod > 0)
    std::printf("  log:        standard output%s%s; a row %s\n",
                control.logFile.empty() ? "" : ", and ",
                control.logFile.c_str(),
                every(control.energyPeriod, "rows").c_str());
  else
    std::printf("  log:        standard output%s%s; no rows\n",
                control.logFile.empty() ? "" : ", and ",
                control.logFile.c_str());
  if (!control.logFile.empty())
    written.push_back(control.logFile);
  if (!control.energyFile.empty()) {
    std::printf("  energy:     %s, %s\n", control.energyFile.c_str(),
                every(control.energyPeriod, "rows").c_str());
    written.push_back(control.energyFile);
  }
  if (!control.pullFile.empty()) {
    std::printf("  pull:       %s, %s\n", control.pullFile.c_str(),
                every(control.energyPeriod, "rows").c_str());
    written.push_back(control.pullFile);
  }
  if (control.framePeriod > 0) {
    std::printf("  trajectory: %s (%s), %s\n", control.trajectoryFile.c_str(),
                control.trajectoryFormat == TrajectoryFormat::XTC ? "XTC"
                                                                 : "DCD",
                every(control.framePeriod, "frames").c_str());
    written.push_back(control.trajectoryFile);
  }
  if (control.checkpointPeriod > 0) {
    if (control.minimize)
      std::printf("  checkpoint: %s, at the end\n",
                  control.restartOutput.c_str());
    else
      std::printf("  checkpoint: %s, %s\n", control.restartOutput.c_str(),
                  every(control.checkpointPeriod, "checkpoints").c_str());
    written.push_back(control.restartOutput);
  }
  for (const std::string &file : written)
    if (llvm::sys::fs::exists(file))
      std::printf("  warning: '%s' exists; mdir run writes over it only with "
                  "--overwrite or --continue\n",
                  file.c_str());
}

int mdir::tool::checkControl(llvm::StringRef controlFile) {
  auto control = readControl(controlFile);
  if (!control)
    return fail(control.takeError());
  auto system = readSystem(*control);
  if (!system)
    return fail(system.takeError());

  if (system->topology) {
    int status = describeTopology(*control, *system);
    describeOutputs(*control);
    return status;
  }

  size_t count = system->getNumParticles();
  std::vector<size_t> perType(control->types.size(), 0);
  double totalMass = 0.0;
  for (size_t i = 0; i != count; ++i) {
    ++perType[system->types[i]];
    totalMass += system->masses[i];
  }
  // The box is held in nm; the control file is in Å.
  double volume = system->box[0] * system->box[1] * system->box[2];

  std::printf("particles:          %zu\n", count);
  for (size_t t = 0, e = control->types.size(); t != e; ++t)
    std::printf("  of type %-10s %zu\n", control->types[t].name.c_str(),
                perType[t]);
  std::printf("total mass:         %g amu\n", totalMass);
  std::printf("box:                %g %g %g Å\n",
              system->box[0] / units::length, system->box[1] / units::length,
              system->box[2] / units::length);
  // amu/nm³ to g/cm³: 1 amu = 1.66053906660e-24 g, 1 nm³ = 1e-21 cm³.
  std::printf("density:            %g g/cm³\n",
              totalMass / volume * 1.66053906660e-3);
  std::printf("degrees of freedom: %g\n", system->getDegreesOfFreedom());
  std::printf("pair terms:         %zu\n", control->pairs.size());
  std::printf("cutoff:             %g Å\n", control->cutoffDistance);
  std::printf("integrator:         %s, %lld steps of %g ps\n",
              getName(control->integrator),
              static_cast<long long>(control->numSteps), control->timestep);
  std::printf("target:             %s, %s precision\n",
              control->target == Target::GPU ? "gpu" : "cpu",
              getName(control->precision));
  describeOutputs(*control);
  return 0;
}
