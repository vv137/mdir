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
/// A surface tension: bar nm in one dyn/cm (1 dyn/cm is 1e-3 N/m, and
/// 1 bar nm is 1e-4 N/m).
constexpr double dynePerCmToBarNm = 10.0;
/// The Boltzmann constant in kJ/(mol K): k_B N_A, exact since the SI of 2019.
/// See Tiesinga et al., Rev. Mod. Phys. 93, 025010 (2021).
constexpr double boltzmann = 0.0083144626181532;
} // namespace units

/// The particles, in the units inside MDIR: nm, ps, amu.
struct System {
  size_t getNumParticles() const { return types.size(); }

  /// The number of degrees of freedom: three per particle with a mass, less
  /// one for each constraint and three for the center of mass. The log and the thermostat both take it.
  double getDegreesOfFreedom() const {
    size_t massive = 0;
    for (double mass : masses)
      massive += mass > 0.0;
    return 3.0 * static_cast<double>(massive) -
           static_cast<double>(numConstraints) - 3.0;
  }
  /// The number of distances that constraints keep.
  size_t numConstraints = 0;

  /// For every particle, the position of its type in `Control::types`.
  std::vector<unsigned> types;
  /// Three numbers per particle.
  std::vector<double> positions;
  std::vector<double> velocities;
  /// From a checkpoint: the state that the next scaling of a barostat that
  /// scales the cell every step takes its pressure from (D92), or nothing.
  std::vector<double> barostatState;
  /// The positions of the file of coordinates, which restraints hold the
  /// particles to; a checkpoint that the run begins from does not change
  /// them.
  std::vector<double> referencePositions;
  /// The constant k of the restraint of each particle, `k |x − x_ref|²`,
  /// in kJ/mol/nm², or nothing if no particle is restrained.
  std::vector<double> restraintConstants;
  /// Whether the file of coordinates gave the velocities. If not, they are
  /// drawn at the temperature of the control file.
  bool givenVelocities = false;
  std::vector<double> masses;
  /// The edge lengths of the cell.
  double box[3];
  /// Those of the file of coordinates, where a run that continues with a
  /// barostat takes `box` from the checkpoint; 0 if they are those of
  /// `box`. The automatic grid of particle mesh Ewald follows them, so
  /// that the run continues with the grid it began with.
  double inputBox[3] = {0.0, 0.0, 0.0};

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
