// The particles of a run.

#ifndef MDIR_DRIVER_SYSTEM_H
#define MDIR_DRIVER_SYSTEM_H

#include "mdir/Driver/Control.h"
#include "mdir/Driver/Cell.h"
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
/// The Coulomb constant in the units of the control file, kcal Å mol⁻¹ e⁻²,
/// from CODATA 2018 (docs/conventions.md). An expression names it
/// `coulomb`.
constexpr double coulomb = 332.06371329919205;
} // namespace units

class Expression;

/// Whether the tail of the pair energy `expression` beyond `cutoff` (Å),
/// ∫ r² u(r) dr, converges at the values `values`: r³ u(r) must fall by a
/// decade at least from each of 10³, 10⁴, 10⁵ to 10⁶ r_c, or be 0
/// (D209).
bool hasFiniteTail(const Expression &expression,
                   llvm::StringMap<double> values, double cutoff);

/// The particles, in the units inside MDIR: nm, ps, amu.
struct System {
  size_t getNumParticles() const { return types.size(); }

  /// The number of degrees of freedom: three per particle with a mass, less
  /// one for each constraint and three for the center of mass where its
  /// momentum is kept. The log and the thermostat both take it.
  double getDegreesOfFreedom() const {
    size_t massive = 0;
    for (double mass : masses)
      massive += mass > 0.0;
    return 3.0 * static_cast<double>(massive) -
           static_cast<double>(numConstraints) - (keepsMomentum ? 3.0 : 0.0);
  }
  /// For each pair term of a topology, its two interaction groups, a flag
  /// for each particle, or none if it acts on all pairs (D137).
  std::vector<std::vector<std::vector<bool>>> pairGroups;
  /// The names of those terms, for the log.
  std::vector<std::string> pairTermNames;
  /// The tail of each pair term of a topology in the correction for the
  /// dispersion (D209): the pairs of classes of
  /// particles that it counts, each with the values that its expression
  /// reads in the units of the control file (`r` aside) and the number of
  /// such pairs that the topology does not exclude; empty for a term that
  /// the correction leaves out, and for all without the correction.
  struct TailPair {
    llvm::StringMap<double> values;
    double count = 0.0;
  };
  std::vector<std::vector<TailPair>> pairTails;
  /// The weights of each seed of Control::tunableTableSeeds at the values
  /// of the tunables, `w[a * T + b]` (D230).
  std::vector<std::vector<double>> tunableSeedWeights;
  /// The parameters of each particle (D165): a name and a value for every
  /// particle, in the units of the control file, which the pair terms
  /// gather.
  std::vector<std::pair<std::string, std::vector<double>>> particleParameters;
  /// [free_energy] (D161): the particles that it decouples, a flag for
  /// each, or none; and the pairs of them that are not excluded, whose
  /// Coulomb stays at full strength.
  std::vector<bool> alchemical;
  std::vector<std::pair<unsigned, unsigned>> alchemicalPairs;
  /// The names of the terms of generalized Born that follow them (D144).
  std::vector<std::string> bornTermNames;
  /// What the run will do that its user may not intend, by a code and a
  /// message: `mdir run` prints them, and `mdir check` lists them.
  std::vector<std::pair<std::string, std::string>> warnings;
  /// Whether the run keeps the momentum of the center of mass: not under
  /// Langevin dynamics unless its motion is removed (D135).
  bool keepsMomentum = true;
  /// The number of distances that constraints keep.
  size_t numConstraints = 0;
  /// The number of rigid waters that SETTLE keeps, the solvent whose
  /// temperature the log gives apart (D203).
  size_t numSettles = 0;
  /// The degrees of freedom of the rigid waters, 6 each, less their share
  /// of those of the center of mass, in proportion to their number.
  double getSolventDegreesOfFreedom() const {
    size_t massive = 0;
    for (double mass : masses)
      massive += mass > 0.0;
    double free = 3.0 * static_cast<double>(massive) -
                  static_cast<double>(numConstraints);
    if (!(free > 0.0))
      return 0.0;
    return 6.0 * static_cast<double>(numSettles) * getDegreesOfFreedom() /
           free;
  }

  /// For every particle, the position of its type in `Control::types`.
  std::vector<unsigned> types;
  /// Three numbers per particle.
  std::vector<double> positions;
  std::vector<double> velocities;
  /// From a checkpoint: the state that the next scaling of a barostat that
  /// scales the cell every step takes its pressure from (D92), or nothing.
  std::vector<double> barostatState;
  /// From a checkpoint: the state of a Nose-Hoover chain (D163a), or
  /// nothing.
  std::vector<double> thermostatState;
  /// The positions of the file of coordinates, which restraints hold the
  /// particles to; a checkpoint that the run begins from does not change
  /// them.
  std::vector<double> referencePositions;
  /// The constant k of the restraint of each particle, `k |x − x_ref|²`,
  /// in kJ/mol/nm², or nothing if no particle is restrained.
  std::vector<double> restraintConstants;
  /// How the reference of each particle follows a barostat (D124), with
  /// `restraintConstants`.
  std::vector<ReferenceScaling> restraintScaling;
  /// Whether the file of coordinates gave the velocities. If not, they are
  /// drawn at the temperature of the control file.
  bool givenVelocities = false;
  std::vector<double> masses;
  /// The edge lengths of the cell: its diagonal a_x, b_y, c_z, with the
  /// tilts b_x, c_x, c_y of a triclinic one (docs/triclinic-m2.md).
  double box[3];
  double tilt[3] = {0.0, 0.0, 0.0};
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

/// Set a CHARMM cell and rotate positions from its symmetric frame.
void applyCharmmCell(Topology &topology, const Cell &cell);

/// Prepare an owned topology through the same path as file input.
/// The input has been validated before constraints or selections index it.
llvm::Expected<System> prepareTopologySystem(const Control &control,
                                             Topology topology,
                                             bool recognizeWaterResidues);

/// Collects the tails of the pair terms of `system` anew (D209), as
/// prepareTopologySystem does, from the values of its topology and of the
/// pair terms of `control`, which a tunable parameter may have changed
/// (D213); the warnings of `system` are kept as they are.
llvm::Error recollectPairTails(const Control &control, System &system);

/// Gives the particles velocities of the temperature of `control`, from
/// the seed of `control`, with the center of mass at rest.
void assignVelocities(const Control &control, System &system);

/// The kinetic energy of `system`, in kJ/mol.
double getKineticEnergy(const System &system);



} // namespace driver
} // namespace mdir

#endif // MDIR_DRIVER_SYSTEM_H
