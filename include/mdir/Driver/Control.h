// The control file of a run.
//
// See docs/driver-m0.md, Section 1.

#ifndef MDIR_DRIVER_CONTROL_H
#define MDIR_DRIVER_CONTROL_H

#include "mdir/Driver/Expression.h"
#include "mdir/Driver/Topology.h"
#include "mdir/Driver/Trajectory.h"

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
  /// With a topology, two masks: the term then acts only on the pairs of
  /// a particle of one and a particle of the other (D137).
  std::vector<std::string> groups;
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

/// How the Lennard-Jones term is truncated at the cutoff: not at all, shifted
/// to 0, switched, or with its force switched, as a cubic polynomial added to
/// the force (`force_switch` of the IR) or with the force of each inverse
/// power switched on its own [Steinbach1994] (`PowerForceSwitch`, for the
/// Lennard-Jones of a topology only).
enum class Truncation { None, Shift, Switch, ForceSwitch, PowerForceSwitch };
enum class Integrator { VelocityVerlet, Leapfrog };
enum class Target { CPU, GPU };
enum class Precision { Single, Mixed, Double };
enum class NeighborStructure { Matrix, Groups };
enum class BarostatWork { Trotter, TrotterFirstOrder, Exact, FirstOrder };
/// How the reference positions of restraints follow a barostat (D124): their
/// center scales with the cell and the offsets from it stay, or each
/// reference scales with the cell as the positions do.
enum class ReferenceScaling { Center, All };
/// How the thermostat couples the velocities to the bath: stochastic
/// velocity rescaling at the end of a period, or Langevin dynamics in the
/// middle of the drift of every step (D135).
enum class ThermostatMethod { VRescale, Langevin };

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
  /// A protein structure file of CHARMM, its coordinates, and the files of
  /// topology, parameters, and streams that give its parameters, in order.
  std::string charmmStructureFile;
  std::string charmmCoordinateFile;
  std::vector<std::string> charmmParameterFiles;

  bool hasTopology() const {
    return !prmtopFile.empty() || !gromacsTopologyFile.empty() ||
           !charmmStructureFile.empty();
  }
  std::string restartInput;

  // [output]
  std::string trajectoryFile;
  /// The coordinates of the terms over centers of groups at every energy
  /// of the log (D145).
  std::string pullFile;
  /// The format of the trajectory, DCD or XTC (D141).
  TrajectoryFormat trajectoryFormat = TrajectoryFormat::DCD;
  std::string restartOutput;

  // [energy]
  double switchDistance = 10.0;
  double cutoffDistance = 12.0;
  double pairlistDistance = 13.5;
  /// The reach of the inner list of a dual list (D114), between the cutoff
  /// and `pairlistDistance`; 0 keeps one list.
  double prunedDistance = 0.0;
  /// The interval of the rebuilds of the neighbor structures, in steps, or
  /// 0 for a rebuild when the test of validity at every step fails (the
  /// default). NOT A DEFAULT: at an interval, the structures are not
  /// tested between builds and may miss pairs within the cutoff (D88).
  int64_t rebuildPeriod = 0;
  /// The neighbor structure of the loops over pairs on a device: a row a
  /// particle, or groups of 16 that share a list, each pair once, where the
  /// loops allow it (D89).
  NeighborStructure neighborStructure = NeighborStructure::Matrix;
  Truncation truncation = Truncation::Switch;
  std::vector<PairTerm> pairs;
  /// Terms over tuples of the topology given by expressions (D136):
  /// [[energy.bond]], [[energy.angle]], and [[energy.dihedral]].
  std::vector<TupleTerm> tupleTerms;
  /// Whether an expression of a term of the topology takes the time `t`
  /// in ps (D145).
  bool usesTime = false;
  /// Functions of one argument by their values, which every expression may
  /// call (D138): [[energy.function]].
  std::vector<TabulatedFunction> functions;
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
  /// The reaction field beyond the cutoff in place of particle mesh Ewald
  /// (D140), with the relative permittivity of the medium beyond it; 0
  /// for a conductor.
  bool reactionField = false;
  double reactionFieldDielectric = 0.0;
  /// Generalized Born of Onufriev, Bashford, and Case (D144): none, or
  /// OBC I or II, with the relative permittivities of the solvent and the
  /// solute, and the energy per area of the nonpolar term in kcal/mol/Å²,
  /// 0 for none.
  enum class ImplicitSolvent { None, OBC1, OBC2 };
  ImplicitSolvent implicitSolvent = ImplicitSolvent::None;
  double solventDielectric = 78.5;
  double soluteDielectric = 1.0;
  double surfaceAreaEnergy = 0.0;

  DispersionCorrection topologyDispersion =
      DispersionCorrection::EnergyPressure;
  /// Whether the control file gives `dispersion_correction`.
  bool topologyDispersionGiven = false;

  // [constraints]
  /// Whether the bonds of hydrogen are constrained (SHAKE, later in M1),
  /// and whether rigid waters are, by SETTLE. `statesFlexible` tells that
  /// the control file says `rigid_water = false`, which a topology of
  /// GROMACS with SETTLE needs to run flexible.
  bool rigidBonds = false;
  /// Use a checked quadratic solution for isolated one-bond groups.
  bool analyticBonds = false;
  bool fastWater = false;
  bool statesFlexible = false;
  /// The residues of an Amber or CHARMM topology that SETTLE constrains
  /// (D63); empty, those of the format: WAT for Amber, TIP3 for CHARMM.
  std::vector<std::string> settleResidues;

  // [minimize], in place of [dynamics]: steepest descent over `numSteps`
  // steps, the first of which moves no particle farther than
  // `minimizeStep`, in Å. The energies are written every `energyPeriod`
  // steps, a frame every `framePeriod`, and a checkpoint at the end.
  bool minimize = false;
  double minimizeStep = 0.1;

  // [dynamics]
  Integrator integrator = Integrator::VelocityVerlet;
  double timestep = 0.001;
  int64_t numSteps = 100;
  int64_t energyPeriod = 10;
  int64_t framePeriod = 0;
  int64_t checkpointPeriod = 0;
  /// The interval of the removal of the motion of the center of mass, and
  /// of the thermostat, in steps; 0 for none.
  int64_t comPeriod = 0;
  int64_t thermostatPeriod = 0;
  uint64_t seed = 314159;

  // [ensemble]
  double temperature = 298.15;
  /// Stochastic velocity rescaling (Bussi, Donadio, and Parrinello 2007) at
  /// `temperature` with the time constant `tauT`, in ps; or Langevin
  /// dynamics with the friction `friction`, in 1/ps, by the middle scheme
  /// (Zhang et al. 2019; D135).
  bool thermostat = false;
  ThermostatMethod thermostatMethod = ThermostatMethod::VRescale;
  double tauT = 1.0;
  double friction = 0.0;
  bool isLangevin() const {
    return thermostat && thermostatMethod == ThermostatMethod::Langevin;
  }
  /// Stochastic cell rescaling (Bernetti and Bussi 2020), isotropic, at
  /// `pressure` in atm with the time constant `tauP` in ps and the
  /// isothermal compressibility `compressibility` in 1/atm.
  bool barostat = false;
  double pressure = 1.0;
  double tauP = 5.0;
  double compressibility = 4.5e-5 * 1.01325;
  /// Semi-isotropic coupling (D119): x and y scale together, by the strain
  /// of the area from the mean of their pressures, and z by its own, eqs.
  /// (9a) and (9b) of [Bernetti2020] with independent noises. The
  /// compressibility of z `compressibilityZ` in 1/atm (0 keeps the height),
  /// and the tension `surfaceTension` in dyn/cm of each of `surfaces`
  /// surfaces normal to z.
  bool semiIsotropic = false;
  double compressibilityZ = 4.5e-5 * 1.01325;
  double surfaceTension = 0.0;
  int64_t surfaces = 2;
  /// How the barostat integrates a scaling (D77, D92): within the drift of
  /// the last step of a period, the integrator of Trotter type of
  /// [Bernetti2020], with the energy of the scaling from the virials before
  /// and after it (the default); exactly, from the potential energy of the
  /// scaled positions, whose forces the next step then takes; or to first
  /// order in the strain from the virial, keeping the forces of the
  /// positions before the scaling, as GROMACS does. The Trotter type may
  /// count its energy from the virial before the scaling only, to first
  /// order, and skip the virial of the step that scales.
  BarostatWork barostatWork = BarostatWork::Trotter;
  int64_t barostatPeriod = 0;

  /// The interval at which the velocities are coupled: the removal of the
  /// motion of the center of mass and the thermostat act there, at the end
  /// of a step. 0 if neither acts.
  /// The two periods are equal where both are not 0.
  int64_t getCouplingPeriod() const {
    return thermostatPeriod ? thermostatPeriod : comPeriod;
  }
  /// The steps between the energies that the run computes: those of the
  /// log, or those of the frames if they are more frequent, of which the
  /// log then shows every energyPeriod / framePeriod-th.
  int64_t getEnergyLoopPeriod() const {
    if (framePeriod > 0 && energyPeriod > 0 && framePeriod < energyPeriod)
      return framePeriod;
    return energyPeriod;
  }

  // [[restraints]]
  /// A harmonic restraint of the particles that `selection` selects (a mask
  /// of Amber, selectParticles) to their positions in the file of
  /// coordinates: `k |x − x_ref|²` each, with k in kcal/mol/Å².
  struct Restraint {
    std::string selection;
    double forceConstant = 0.0;
    ReferenceScaling scaling = ReferenceScaling::Center;
  };
  std::vector<Restraint> restraints;

  // [boundary]
  /// Whether the cell is periodic. Without one (`type = "NONE"`, D142) the
  /// run takes a cell around the particles that no image reaches.
  bool periodic = true;
  /// The edges of the cell, and for a run from CHARMM's files its angles
  /// α, β, γ in degrees, 90 for a rectangular cell.
  double box[3] = {0.0, 0.0, 0.0};
  double angles[3] = {90.0, 90.0, 90.0};

  // [execution]
  Target target = Target::CPU;
  int64_t threads = 1;
  Precision precision = Precision::Double;
  /// The number of neighbors that a neighbor structure holds per particle
  /// at first, or 0 for an estimate from the density; builds make room for
  /// more.
  int64_t neighborWidth = 0;
  /// Whether kernels are rewritten in ways that change rounding.
  bool fastMath = true;
  /// Whether the particles are put in the order of their positions.
  bool reorder = true;
  /// Whether every sum is added in an order that the threads do not decide,
  /// so that a run gives the same bits on the same binary and hardware
  /// (D84).
  bool deterministic = false;
};

/// Reads the control file `path`. Paths of files in it are relative to the
/// directory of the control file.
llvm::Expected<Control> readControl(llvm::StringRef path);

/// A control file with every keyword of M0 and its default.
std::string getControlTemplate();

/// A control file for a run from an Amber topology, with every keyword that
/// such a run takes: particle mesh Ewald, constraints, the ensembles, a
/// minimization, and restraints.
std::string getAmberControlTemplate();

/// A stage of the standard pipeline from examples/ala3, with placeholder
/// input paths: minimize, nvt, npt, or production. Empty for an unknown kind.
std::string getPipelineControlTemplate(llvm::StringRef kind);

} // namespace driver
} // namespace mdir

#endif // MDIR_DRIVER_CONTROL_H
