// The control file of a run.
//
// See docs/driver-m0.md, Section 1.

#ifndef MDIR_DRIVER_CONTROL_H
#define MDIR_DRIVER_CONTROL_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mdir {
namespace driver {

/// A type of particle: its mass and its parameters, in the units of the
/// control file.
struct ParticleType {
  std::string name;
  double mass = 0.0;
  /// The parameters, in the order of the control file.
  std::vector<std::pair<std::string, double>> parameters;
};

/// How the parameter of a pair follows from those of its two particles:
/// their mean, the square root of their product, or their product.
enum class Mixing { Arithmetic, Geometric, Product };

/// What the correction for the dispersion beyond the cutoff corrects: none,
/// or the energy and the pressure.
enum class DispersionCorrection { None, EnergyPressure };

/// A term of the potential energy over pairs, given by an expression in the
/// distance `r`.
struct PairTerm {
  std::string name;
  std::string expression;
  /// For the parameters of the types that the expression uses.
  llvm::StringMap<Mixing> mixing;
  /// Numbers that the expression uses under a name.
  std::vector<std::pair<std::string, double>> constants;
  /// The correction for what the term leaves out beyond the cutoff. It
  /// suits a term that decays faster than 1/r³, such as dispersion.
  DispersionCorrection dispersion = DispersionCorrection::None;
};

/// Parameters of a pair of types that a pair term takes in place of those
/// that its mixing rules give (NBFIX).
struct PairOverride {
  /// The name of the pair term.
  std::string term;
  /// The two types.
  std::string first, second;
  /// The parameters, in the order of the control file.
  std::vector<std::pair<std::string, double>> parameters;
};

enum class Truncation { None, Shift, Switch, ForceSwitch };
enum class Integrator { VelocityVerlet, Leapfrog };
enum class Target { CPU, GPU };
enum class Precision { Single, Mixed, Double };

/// What a control file says. Lengths are in Å, energies in kcal/mol, times
/// in ps, masses in amu, and temperatures in K.
struct Control {
  // [input]
  std::string pdbFile;
  /// A topology and the coordinates of Amber, in place of `pdbFile` and the
  /// types of [energy].
  std::string prmtopFile;
  std::string amberCoordinateFile;
  /// A topology and the coordinates of GROMACS, the directories that its
  /// includes are looked for in, and the macros defined before it is read.
  std::string gromacsTopologyFile;
  std::string gromacsCoordinateFile;
  std::vector<std::string> gromacsIncludes;
  std::vector<std::string> gromacsDefines;

  bool hasTopology() const {
    return !prmtopFile.empty() || !gromacsTopologyFile.empty();
  }
  std::string restartInput;

  // [output]
  std::string dcdFile;
  std::string restartOutput;

  // [energy]
  double switchDistance = 10.0;
  double cutoffDistance = 12.0;
  double pairlistDistance = 13.5;
  Truncation truncation = Truncation::Switch;
  std::vector<PairTerm> pairs;
  std::vector<ParticleType> types;
  std::vector<PairOverride> overrides;
  /// The correction for the dispersion of a run from a topology.
  /// Particle mesh Ewald for the Coulomb terms of a topology
  /// (docs/pme-m1.md): β in Å⁻¹, or 0 to take it from the tolerance; the
  /// numbers of points of the grid, or 0 to take them from the largest
  /// spacing, in Å; the order of the B-splines; and whether the direct sum
  /// is shifted to 0 at the cutoff.
  bool pme = false;
  double pmeAlpha = 0.0;
  double pmeAlphaTolerance = 1.0e-5;
  int64_t pmeGrid[3] = {0, 0, 0};
  double pmeMaxSpacing = 1.2;
  int64_t pmeOrder = 4;
  bool pmeShift = false;
  /// The influence function: that of Essmann et al. times a factor for
  /// each edge that the aliasing of the B-splines sets, as sander has it,
  /// or that of Essmann et al. alone, as GROMACS has it
  /// (docs/pme-m1.md, Section 1.1).
  bool pmeOptimal = false;

  DispersionCorrection topologyDispersion =
      DispersionCorrection::EnergyPressure;

  // [constraints]
  /// Whether the bonds of hydrogen are constrained (SHAKE, later in M1),
  /// and whether rigid waters are, by SETTLE. `statesFlexible` tells that
  /// the control file says `fast_water = false`, which a topology of
  /// GROMACS with SETTLE needs to run flexible.
  bool rigidBonds = false;
  bool fastWater = false;
  bool statesFlexible = false;
  /// The residues of an Amber topology that SETTLE constrains (D63).
  std::vector<std::string> settleResidues = {"WAT"};

  // [dynamics]
  Integrator integrator = Integrator::VelocityVerlet;
  double timestep = 0.001;
  int64_t numSteps = 100;
  int64_t energyPeriod = 10;
  int64_t framePeriod = 0;
  int64_t checkpointPeriod = 0;
  /// The interval of rebuilds, or 0 for a test of validity at every step.
  int64_t rebuildPeriod = 0;
  /// The interval of the removal of the motion of the center of mass, and
  /// of the thermostat, in steps; 0 for none.
  int64_t comPeriod = 0;
  int64_t thermostatPeriod = 0;
  uint64_t seed = 314159;

  // [ensemble]
  double temperature = 298.15;
  /// Stochastic velocity rescaling (Bussi, Donadio, and Parrinello 2007) at
  /// `temperature` with the time constant `tauT`, in ps.
  bool thermostat = false;
  double tauT = 1.0;
  /// Stochastic cell rescaling (Bernetti and Bussi 2020), isotropic, at
  /// `pressure` in atm with the time constant `tauP` in ps and the
  /// isothermal compressibility `compressibility` in 1/atm.
  bool barostat = false;
  double pressure = 1.0;
  double tauP = 5.0;
  double compressibility = 4.5e-5 * 1.01325;
  int64_t barostatPeriod = 0;

  /// The interval at which the velocities are coupled: the removal of the
  /// motion of the center of mass and the thermostat act there, at the end
  /// of a step. 0 if neither acts.
  /// The two periods are equal where both are not 0.
  int64_t getCouplingPeriod() const {
    return thermostatPeriod ? thermostatPeriod : comPeriod;
  }

  // [boundary]
  double box[3] = {0.0, 0.0, 0.0};

  // [execution]
  Target target = Target::CPU;
  int64_t threads = 1;
  Precision precision = Precision::Double;
  /// The number of neighbors that a neighbor structure holds per particle,
  /// or 0 for an estimate from the density.
  int64_t neighborWidth = 0;
  /// Whether kernels are rewritten in ways that change rounding.
  bool fastMath = true;
  /// Whether the particles are put in the order of their positions.
  bool reorder = true;
};

/// Reads the control file `path`. Paths of files in it are relative to the
/// directory of the control file.
llvm::Expected<Control> readControl(llvm::StringRef path);

/// A control file with every keyword of M0 and its default.
std::string getControlTemplate();

} // namespace driver
} // namespace mdir

#endif // MDIR_DRIVER_CONTROL_H
