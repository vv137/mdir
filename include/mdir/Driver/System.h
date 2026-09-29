// The particles of a run.

#ifndef MDIR_DRIVER_SYSTEM_H
#define MDIR_DRIVER_SYSTEM_H

#include "mdir/Driver/Control.h"

#include <vector>

namespace mdir {
namespace driver {

/// Factors between the units of the control file and those inside MDIR.
namespace units {
/// nm per Å.
constexpr double length = 0.1;
/// kJ/mol per kcal/mol.
constexpr double energy = 4.184;
/// The Boltzmann constant in kJ/(mol K).
constexpr double boltzmann = 0.0083144626181532;
} // namespace units

/// The particles, in the units inside MDIR: nm, ps, amu.
struct System {
  size_t getNumParticles() const { return types.size(); }

  /// The number of degrees of freedom: three per particle, less three for
  /// the center of mass.
  double getDegreesOfFreedom() const {
    return 3.0 * static_cast<double>(getNumParticles()) - 3.0;
  }

  /// For every particle, the position of its type in `Control::types`.
  std::vector<unsigned> types;
  /// Three numbers per particle.
  std::vector<double> positions;
  std::vector<double> velocities;
  std::vector<double> masses;
  /// The edge lengths of the cell.
  double box[3];
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
