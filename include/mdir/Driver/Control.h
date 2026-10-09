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
  /// Whether the term gives `dispersion_correction`: with a topology,
  /// `"NONE"` leaves it out of the correction of [energy], which it
  /// otherwise follows (D209).
  bool dispersionGiven = false;
  /// With a topology, two masks: the term then acts only on the pairs of
  /// a particle of one and a particle of the other (D137), as the
  /// interaction groups of the custom nonbonded force of OpenMM
  /// [Eastman2017].
  std::vector<std::string> groups;
  /// In the Python model, `observe` of the term (D189,
  /// D232): whether the term is observed, and the constants
  /// whose derivatives are; the control file gives Control::observables.
  bool observed = false;
  std::vector<std::string> observe;
};

/// A term of the potential energy over the triplets centered on each
/// particle (D160), given by an expression in the legs `r12` and `r13`
/// from the center, particle 1, to the ends 2 and 3, the far leg `r23`,
/// and the angle `theta` at the center: the three-body term of Stillinger
/// and Weber [StillingerWeber1985] and of mW water [Molinero2009], and the
/// mode `UniqueCentralParticle` of the custom many-particle force of OpenMM
/// [Eastman2017], whose names of the variables it takes.
struct TripletTerm {
  std::string name;
  std::string expression;
  /// The cutoff of each leg from the center, in Å; the far leg is not cut.
  double cutoff = 0.0;
  /// Numbers that the expression uses under a name.
  std::vector<std::pair<std::string, double>> constants;
};

/// A parameter of each particle of a topology (D165), as the per-particle
/// parameters of the custom forces of OpenMM [Eastman2017]: a term takes it
/// by the suffix of a place, `w1` and `w2` in a pair term, `w1` to `wN` in
/// a term over tuples, and `w` itself in a term of the positions. Entries
/// of one name apply in the order of the file, each to its particles.
struct ParticleParameter {
  std::string name;
  /// A value for the particles of `selection`, a mask of Amber, or of
  /// `particles`, their numbers from 0, or for every particle if neither is
  /// given.
  double value = 0.0;
  std::string selection;
  std::vector<unsigned> particles;
  /// Or a value for each particle of the system, in its order.
  std::vector<double> values;
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
/// SquaredDistanceSwitch multiplies the topology Lennard-Jones potential
/// by a cubic in squared distance [Brooks1983].
enum class Truncation {
  None, Shift, Switch, ForceSwitch, PowerForceSwitch, SquaredDistanceSwitch
};
/// Brownian dynamics (D163b) moves the positions only, overdamped.
enum class Integrator { VelocityVerlet, Leapfrog, Brownian };
enum class Target { CPU, GPU };
enum class Precision { Single, Mixed, Double };
enum class NeighborStructure { Matrix, Groups };
enum class BarostatWork { Trotter, TrotterFirstOrder, Exact, FirstOrder };
/// How the reference positions of restraints follow a barostat (D124): their
/// center scales with the cell and the offsets from it stay, or each
/// reference scales with the cell as the positions do.
enum class ReferenceScaling { Center, All };
/// How the thermostat couples the velocities to the bath: stochastic
/// velocity rescaling at the end of a period, Langevin dynamics in the
/// middle of the drift of every step (D135), or a Nose-Hoover chain at the
/// end of a period (D163a).
enum class ThermostatMethod { VRescale, Langevin, NoseHoover };

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

  /// An owned topology supplied by an embedding front end. No file is reopened.
  bool inMemoryTopology = false;
  bool inMemoryCharmm = false;
  /// Set by an embedding front end, not by a key (D213):
  /// whether the model declares tunable parameters, whose values the
  /// program must take at run time rather than as constants; and the
  /// constants of pair terms that are tunable, by the index of the term and
  /// the name of the constant, in the order of the columns of the table
  /// `tunable_constants` that the program reads them from.
  bool tunables = false;
  std::vector<std::pair<unsigned, std::string>> tunableConstants;
  /// The observed constants that a tunable takes, by the name of the term
  /// and of the constant (D232): the program takes their
  /// values as values of its entry (Program::startValues), not as
  /// constants of its text, so that an update does not change it.
  std::vector<std::pair<std::string, std::string>> observedTunables;
  bool isObservedTunable(llvm::StringRef term, llvm::StringRef constant) const {
    for (const auto &[t, c] : observedTunables)
      if (t == term && c == constant)
        return true;
    return false;
  }
  /// Whether the program carries the derivative of the energy in the
  /// tunables (D230, docs/python-gradient.md), and the
  /// declarations that it differentiates, in the order of the tunables.
  bool tunableGradient = false;
  enum class TunableKind {
    Charge,
    Sigma,
    Epsilon,
    SigmaPair,
    EpsilonPair,
    PairConstant,
    TupleParameter
  };
  struct TunableDeclaration {
    std::string name;
    TunableKind kind = TunableKind::Charge;
    /// The index of the term among the pair terms or the tuple terms.
    unsigned termIndex = 0;
    std::string parameter;
  };
  std::vector<TunableDeclaration> tunableDeclarations;
  /// The seeds through which the derivatives in the tables of σ and ε
  /// leave a program as fields of the particles (D230).
  /// The kernel of the Lennard-Jones of `@tunable` takes, for the pair of
  /// the particles i and j of the types a and b, the value of the table
  /// plus d_i w[a, b] + d_j w[b, a], with d a field of zeros and w the
  /// weights of the seed (System::tunableSeedWeights): the derivative of
  /// the energy in d, added over the particles of the type a, is
  /// Σ_b w[a, b] ∂U/∂(table)_ab, the pairs of two particles of one type
  /// counted twice. A seed of a per-type tunable has the derivatives of
  /// the combining rule as its weights, w[a, b] = ∂(table)_ab/∂θ_a, and
  /// its row a gives the site a; a seed of a tunable of the table by pairs
  /// has the weight 1 for one pair of types in each row, and its row gives
  /// that pair, times `scales` (1/2 for a pair of one type), so that as
  /// many seeds are needed as pairs share a type. `sites` holds the site of
  /// each row, or -1.
  struct TunableTableSeed {
    bool sigma = true;
    unsigned tunable = 0;
    std::vector<int64_t> sites;
    std::vector<double> scales;
    /// For a seed of a table by pairs, the type of the other member of the
    /// pair of each row.
    std::vector<int64_t> columns;
  };
  std::vector<TunableTableSeed> tunableTableSeeds;

  bool hasTopology() const {
    return inMemoryTopology || !prmtopFile.empty() || !gromacsTopologyFile.empty() ||
           !charmmStructureFile.empty();
  }
  std::string restartInput;
  /// Set by the driver, not by a key: the run takes the positions,
  /// velocities, and cell of `restartInput`, but evaluates the forces of its
  /// first step, since the checkpoint was written by a run of other physics
  /// or coupling (D172).
  bool restartRecomputes = false;
  /// Set by an embedding front end, not by a key (D196): the
  /// entry takes the counts of its loops as arguments, so that one program
  /// runs any number of steps from any step of the coupling period, and
  /// whether the call begins the run or continues the state that the last
  /// segment left, as a run from `restartInput` continues a checkpoint
  /// (D211).
  bool segments = false;

  // [output] (D149)
  std::string trajectoryFile;
  /// The log in a file, besides the standard output, and the rows of the
  /// log as columns; empty for none.
  std::string logFile;
  /// Optional execution history in JSON Lines (D168).
  std::string manifestFile;
  std::string energyFile;
  /// The terms over centers of groups at every energy of the log (D145).
  std::string pullFile;
  /// dH/dλ and the differences of the energy to the other states of
  /// [free_energy] at every energy of the log (D161).
  std::string freeEnergyFile;
  /// The energies of terms given by expressions and their derivatives in
  /// constants of theirs, at every energy of the log (D189): the file and
  /// the columns, from `observe` of each term, the energy of the term and
  /// then its constants, the terms in the order of the file.
  std::string observablesFile;
  struct Observable {
    std::string term;
    /// Empty for the energy of the term.
    std::string constant;
    /// The line of the term in the control file, which orders the columns.
    int64_t line = 0;
  };
  std::vector<Observable> observables;
  /// The columns of the file of `observables`, after the step: the energy
  /// of each term and its derivatives, in kcal/mol and per unit of the
  /// constant.
  std::vector<std::pair<std::string, std::string>> getObservableColumns() const {
    std::vector<std::pair<std::string, std::string>> columns;
    for (const Observable &observable : observables)
      if (observable.constant.empty())
        columns.push_back({observable.term + ".energy", "kcal/mol"});
      else
        columns.push_back({observable.term + ".d_" + observable.constant,
                           "kcal/mol/" + observable.constant});
    return columns;
  }
  /// The format of the trajectory, DCD or XTC (D141).
  TrajectoryFormat trajectoryFormat = TrajectoryFormat::DCD;
  /// `trajectory_precision = "SINGLE"`: a trajectory in H5MD holds its
  /// positions in f32 rather than f64 (D239).
  bool trajectorySingle = false;
  /// `trajectory_strings = "VARIABLE"`: its `unit` attributes are
  /// variable-length strings rather than fixed-length ones.
  bool trajectoryVariableStrings = false;
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
  /// Terms over the triplets centered on each particle (D160):
  /// [[energy.triplet]].
  std::vector<TripletTerm> triplets;
  /// Terms over tuples of the topology given by expressions (D136):
  /// [[energy.bond]], [[energy.angle]], and [[energy.dihedral]].
  std::vector<TupleTerm> tupleTerms;
  /// What the control file asks that its author may not intend, by a code
  /// and a message (D158); the system adds them to its own warnings.
  std::vector<std::pair<std::string, std::string>> warnings;
  /// Terms of the absolute positions of single particles given by
  /// expressions (D148): [[energy.external]].
  std::vector<ExternalTerm> externalTerms;
  /// Whether an expression of a term of the topology takes the time `t`
  /// in ps (D145).
  bool usesTime = false;

  // [free_energy] (D161)
  /// Alchemical states: the particles that `couple` selects (a mask, or
  /// none) are decoupled from the rest as `lambda_coulomb` and `lambda_vdw`
  /// go from 0, the system of the topology, to 1, while the interactions
  /// within the selection stay; every component `name` is a parameter
  /// `lambda_name` of the expressions as well. The run samples the state
  /// `state`, from 0.
  struct FreeEnergy {
    std::string couple;
    int64_t state = 0;
    /// The soft-core of the Lennard-Jones of the decoupled pairs,
    /// r_A^6 = alpha sigma^6 lambda^power + r^6 [Beutler1994].
    double softCoreAlpha = 0.5;
    int64_t softCorePower = 1;
    /// The components, `coulomb` and `vdw` first, then the others by name,
    /// each with a value for every state.
    std::vector<std::pair<std::string, std::vector<double>>> lambdas;

    size_t getNumStates() const {
      return lambdas.empty() ? 0 : lambdas.front().second.size();
    }
    /// The selection, the soft-core, and every component at every state,
    /// in one line, which a checkpoint records (D161).
    std::string describe() const {
      std::string text = "couple = \"" + couple + "\"; soft_core = " +
                         std::to_string(softCoreAlpha) + ", " +
                         std::to_string(softCorePower);
      for (const auto &[name, values] : lambdas) {
        text += "; " + name + " =";
        for (double value : values)
          text += " " + std::to_string(value);
      }
      return text;
    }
    /// The value of the component `name` at the state `k`; 0 for a
    /// component that the file does not give.
    double get(llvm::StringRef name, size_t k) const {
      for (const auto &[component, values] : lambdas)
        if (component == name)
          return values[k];
      return 0.0;
    }
  };
  bool hasFreeEnergy = false;
  FreeEnergy freeEnergy;
  /// Whether `name` is the parameter `lambda_<component>` of a component of
  /// [free_energy].
  bool isLambda(llvm::StringRef name) const {
    if (!hasFreeEnergy || !name.starts_with("lambda_"))
      return false;
    for (const auto &[component, values] : freeEnergy.lambdas)
      if (name.drop_front(7) == component)
        return true;
    return false;
  }
  /// Functions of one argument by their values, which every expression may
  /// call (D138): [[energy.function]].
  std::vector<TabulatedFunction> functions;
  /// The parameters of each particle (D165).
  std::vector<ParticleParameter> particleParameters;
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
  /// Particle mesh Ewald for the dispersion of the Lennard-Jones of a
  /// topology (D162): the grid sums −c_i c_j / r⁶ with the coefficients
  /// c_i = 2 √ε_i σ_i³ of the types, and the direct terms add c_i c_j
  /// (1 − g(β r)) / r⁶ within the cutoff [Essmann1995, Wennberg2013]. β in
  /// Å⁻¹, or 0 to take it from the tolerance g(β r_c), with g(x) =
  /// exp(−x²) (1 + x² + x⁴/2); the grid, or 0 to take it from the largest
  /// spacing in Å; and the order of the B-splines.
  bool ljpme = false;
  double ljpmeAlpha = 0.0;
  double ljpmeTolerance = 1.0e-3;
  int64_t ljpmeGrid[3] = {0, 0, 0};
  double ljpmeMaxSpacing = 1.2;
  int64_t ljpmeOrder = 4;
  /// The reaction field beyond the cutoff in place of particle mesh Ewald
  /// (D140), with the relative permittivity of the medium beyond it; 0
  /// for a conductor.
  bool reactionField = false;
  double reactionFieldDielectric = 0.0;
  /// Generalized Born (D144, D152): none, the model of Hawkins, Cramer,
  /// and Truhlar, or OBC I or II of Onufriev, Bashford, and Case, with the
  /// relative permittivities of the solvent and the solute, and the energy
  /// per area of the nonpolar term in kcal/mol/Å², 0 for none.
  enum class ImplicitSolvent { None, HCT, OBC1, OBC2 };
  ImplicitSolvent implicitSolvent = ImplicitSolvent::None;
  double solventDielectric = 78.5;
  double soluteDielectric = 1.0;
  double surfaceAreaEnergy = 0.0;
  /// The concentration of a 1:1 salt in mol/L, which screens the solvent
  /// with the Debye length at `temperature`; 0 for none.
  double saltConcentration = 0.0;
  /// The distance in Å beyond which the radii take no descreening, the
  /// integral cut there; 0 for none, all pairs within the cutoff.
  double bornRadiusCutoff = 0.0;
  /// Where the intrinsic radii and the screening factors come from: the
  /// topology (the sections RADII and SCREEN of Amber), or the rules of
  /// mbondi2 by element.
  enum class BornRadii { Topology, MBondi2 };
  BornRadii bornRadii = BornRadii::Topology;

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
  // With `minimizeTolerance` > 0, in kcal/mol/Å, it ends at the first row
  // of the energies whose largest force (without its parts along the
  // constraints) is below it (D219).
  bool minimize = false;
  double minimizeStep = 0.1;
  double minimizeTolerance = 0.0;

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
  /// Brownian dynamics (D163b), with the friction `friction` of [dynamics]
  /// in 1/ps and no [thermostat].
  bool isBrownian() const { return integrator == Integrator::Brownian; }
  /// A Nose-Hoover chain of `chainLength` thermostats (Martyna, Klein, and
  /// Tuckerman 1992; D163a) with the period `tauT`, in ps.
  int64_t chainLength = 3;
  bool isNoseHoover() const {
    return thermostat && thermostatMethod == ThermostatMethod::NoseHoover;
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
  /// Anisotropic coupling (D163c): each axis scales by its own strain, from
  /// its own pressure and noise, with the compressibility of the axis in
  /// `compressibilities`, 1/atm (0 keeps the axis).
  bool anisotropic = false;
  double compressibilities[3] = {4.5e-5 * 1.01325, 4.5e-5 * 1.01325,
                                 4.5e-5 * 1.01325};
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
  /// coordinates: `k |x − x_ref|²` each, with k in kJ/mol/nm². The control
  /// file gives k in kcal/mol/Å²; it is converted where the file is read.
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

/// Validate and resolve cross-field options without parsing a file.
llvm::Error validateControl(Control &control, llvm::StringRef context);
llvm::Error resolveControlCoupling(Control &control, llvm::StringRef context);

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
