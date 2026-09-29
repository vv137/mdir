// `mdir check`: reads the input of a run and prints what it describes,
// without compiling anything.

#include "Commands.h"

#include "mdir/Driver/Control.h"
#include "mdir/Driver/System.h"

#include "llvm/Support/raw_ostream.h"

#include <cstdio>
#include <vector>

using namespace mdir::driver;

static int fail(llvm::Error error) {
  llvm::errs() << "mdir: " << llvm::toString(std::move(error)) << "\n";
  return 1;
}

static const char *getName(Integrator integrator) {
  return integrator == Integrator::Leapfrog ? "LEAP" : "VVER";
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

  std::printf("topology:           %s\n", control.prmtopFile.empty()
                                             ? control.gromacsTopologyFile.c_str()
                                             : control.prmtopFile.c_str());
  std::printf("particles:          %zu\n", count);
  std::printf("residues:           %zu\n", topology.residueNames.size());
  std::printf("types:              %zu\n", topology.getNumTypes());
  std::printf("bonds:              %zu, %zu with hydrogen\n",
              topology.bonds.size(), hydrogenBonds);
  std::printf("angles:             %zu\n", topology.angles.size());
  std::printf("dihedrals:          %zu, %zu improper\n",
              topology.dihedrals.size(), impropers);
  std::printf("pairs 1-4:          %zu\n", topology.pairs.size());
  std::printf("excluded pairs:     %zu\n", topology.exclusions.size());
  if (!topology.cmaps.empty())
    std::printf("CMAP terms:         %zu, %zu maps\n", topology.cmaps.size(),
                topology.cmapGrids.size());
  if (!topology.virtualSites.empty())
    std::printf("virtual sites:      %zu\n", topology.virtualSites.size());
  std::printf("total charge:       %.6f e\n", totalCharge);
  std::printf("total mass:         %g amu\n", totalMass);
  std::printf("box:                %g %g %g Å\n",
              topology.box[0] / units::length,
              topology.box[1] / units::length,
              topology.box[2] / units::length);
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

int mdir::tool::checkControl(llvm::StringRef controlFile) {
  auto control = readControl(controlFile);
  if (!control)
    return fail(control.takeError());
  auto system = readSystem(*control);
  if (!system)
    return fail(system.takeError());

  if (system->topology)
    return describeTopology(*control, *system);

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
  return 0;
}
