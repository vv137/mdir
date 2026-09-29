// The particles of a run.

#ifndef MDIR_DRIVER_SYSTEM_H
#define MDIR_DRIVER_SYSTEM_H

#include "mdir/Driver/Control.h"
#include "mdir/Driver/Topology.h"

#include <memory>

#include <vector>

namespace mdir {
namespace driver {

/// Factors between the units of the control file and those inside MDIR.
namespace units {
/// The pressure: atm in one kJ/(mol nm^3).
constexpr double pressure = 16.6053906717 / 1.01325;

/// nm per Å.
constexpr double length = 0.1;
/// kJ/mol per kcal/mol.
constexpr double energy = 4.184;
/// The Boltzmann constant in kJ/(mol K): k_B N_A, exact since the SI of 2019.
/// See Tiesinga et al., Rev. Mod. Phys. 93, 025010 (2021).
constexpr double boltzmann = 0.0083144626181532;
} // namespace units

/// The particles, in the units inside MDIR: nm, ps, amu.
struct System {
  size_t getNumParticles() const { return types.size(); }

  /// The number of degrees of freedom: three per particle with a mass, less
  /// three for the center of mass. The log and the thermostat both take it.
  double getDegreesOfFreedom() const {
    size_t massive = 0;
    for (double mass : masses)
      massive += mass > 0.0;
    return 3.0 * static_cast<double>(massive) - 3.0;
  }

  /// For every particle, the position of its type in `Control::types`.
  std::vector<unsigned> types;
  /// Three numbers per particle.
  std::vector<double> positions;
  std::vector<double> velocities;
  /// Whether the file of coordinates gave the velocities. If not, they are
  /// drawn at the temperature of the control file.
  bool givenVelocities = false;
  std::vector<double> masses;
  /// The edge lengths of the cell.
  double box[3];

  /// The topology that the system was read from, if any. It gives the
  /// terms of the potential.
  std::shared_ptr<Topology> topology;

};

/// Reads the positions from the PDB file of `control`. The name of an atom
/// selects its type.
llvm::Expected<System> readSystem(const Control &control);

/// Gives the particles velocities of the temperature of `control`, from
/// the seed of `control`, with the center of mass at rest.
void assignVelocities(const Control &control, System &system);

/// The kinetic energy of `system`, in kJ/mol.
double getKineticEnergy(const System &system);



} // namespace driver
} // namespace mdir

#endif // MDIR_DRIVER_SYSTEM_H
