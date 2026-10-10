// Builds the program of a run: a module of md and dyn ops.
//
// See docs/driver-m0.md, Section 2.

#ifndef MDIR_DRIVER_BUILDER_H
#define MDIR_DRIVER_BUILDER_H

#include "mdir/Driver/Control.h"
#include "mdir/Driver/System.h"

#include <algorithm>
#include <string>
#include <vector>

namespace mdir {
namespace driver {

/// A floating-point type that a buffer of the host holds.
enum class Element { F32, F64 };

/// The program of a run, and what the host must pass to it.
///
/// The entry function takes, in this order: the buffers of the positions
/// and of the velocities; the buffer of the forces, if `takesForces` is
/// set; the buffer of the masses; one buffer for each field in `fields`;
/// the three edge lengths of the cell, in nm; the time step, in ps; and the
/// number of the step that the run begins after. A minimization in segments
/// then takes the length of its first step, and a program of segments
/// whether the call begins the run. Last come the values of `startValues`,
/// one f64 each, in their order.
struct Program {
  /// The module, as text.
  std::string module;
  /// The name of the entry function.
  std::string entry;

  /// The types that the buffers hold.
  Element state;
  Element force;
  Element mass;
  Element parameter;

  /// Whether the run begins with forces that it is given: a run that
  /// continues an earlier one, with an integrator that carries forces.
  bool takesForces = false;
  /// Whether the state that a checkpoint holds has forces.
  bool writesForces = false;

  /// The parameters that differ between the types, one value per particle,
  /// in the units of the control file.
  struct Field {
    std::string name;
    std::vector<double> values;
    /// Whether the field holds whole numbers, which the program takes as
    /// i32: the types of the particles.
    bool isInteger = false;
  };
  std::vector<Field> fields;

  /// Parameters of the pairs of types that a pair term looks up, in the
  /// units of the control file: `values[a * count + b]` for the types `a`
  /// and `b`.
  /// A table of `count` rows: of pairs of types, symmetric and square,
  /// or of `columns` columns.
  struct Table {
    std::string name;
    unsigned count = 0;
    std::vector<double> values;
    unsigned columns = 0;

    bool isSquare() const { return columns == 0; }
    unsigned getColumns() const { return isSquare() ? count : columns; }
    llvm::StringRef getType() const {
      return isSquare() ? "!table" : "!grid";
    }
  };
  std::vector<Table> tables;

  /// The tuples of a topology: bonds, angles, and the like. `members` has
  /// `arity` particles for each tuple, and each field one value for each
  /// tuple, in the units of MDIR.
  struct TupleSet {
    std::string name;
    unsigned arity = 0;
    std::vector<int32_t> members;
    std::vector<Field> fields;
    /// Whether a tuple is the same read backward, as the members of a bond
    /// or of a dihedral are. Those of a virtual site or a correction map
    /// are not.
    bool reversible = true;
    /// Whether a tuple of two members has an order, as the pairs of a
    /// particle and the anchor of its constraint group have (D110).
    bool oriented = false;

    size_t size() const { return arity ? members.size() / arity : 0; }
    /// Whether no particle is a member of two tuples, as in the groups of
    /// a constraint.
    bool isDisjoint() const {
      std::vector<int32_t> sorted(members);
      std::sort(sorted.begin(), sorted.end());
      return std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end();
    }
    llvm::StringRef getOrientation() const {
      if (arity == 2 && !oriented)
        return "unordered";
      return reversible ? "reversal" : "ordered";
    }
  };
  std::vector<TupleSet> tupleSets;

  /// The correction for the dispersion beyond the cutoff, in kJ/mol, at
  /// the volume of the run: what it adds to the potential energy and to
  /// the trace of the virial.
  double dispersionEnergy = 0.0;
  double dispersionVirial = 0.0;
  /// The part of the estimate of the shift (D210) that does not depend on
  /// the volume, in kJ/mol: the estimate is X (1 - V / (N (4π/3) r_c³))
  /// with X proportional to 1 / V, so X is in `dispersionEnergy` and
  /// -X V / (N (4π/3) r_c³), the particle itself taken out of its
  /// neighbors, is this constant (#224). 0 unless the energy of the run
  /// holds the estimate.
  double dispersionFixedEnergy = 0.0;

  /// Particle mesh Ewald or the reaction field: whether the program has
  /// it, and the constant energy of its Coulomb terms in kJ/mol: for
  /// particle mesh Ewald the self term and the background of a net charge,
  /// with what the background adds to the trace of the virial; for the
  /// reaction field its self term (D140).
  bool pme = false;
  bool reactionField = false;
  double coulombConstantEnergy = 0.0;
  double coulombConstantVirial = 0.0;
  double coulombSelfEnergy = 0.0;
  /// [free_energy] (D161): for each state, the constant energies that
  /// depend on λ, in kJ/mol at the volume of the file: those that do not
  /// depend on the volume (the self term of particle mesh Ewald) and those
  /// proportional to 1 / V (the background of a net charge and the
  /// correction for the dispersion); and for each component of λ their
  /// derivatives at the state of the run.
  std::vector<double> stateFixedEnergies, stateVolumeEnergies;
  std::vector<double> lambdaFixedDerivatives, lambdaVolumeDerivatives;
  /// For each column of `observe`, what the correction for the dispersion
  /// adds to it, in kJ/mol (per unit of the constant) at the volume of the
  /// file, proportional to 1 / V: the tail of an observed pair term and its
  /// derivatives (D209); 0 for the other columns.
  std::vector<double> observableVolumeConstants;
  /// And what it adds that does not depend on the volume: that part of the
  /// estimate of the shift of an observed pair term, and its derivatives.
  std::vector<double> observableFixedConstants;
  /// The derivative of the energy in the tunables
  /// (D230, docs/python-gradient.md): whether the entry
  /// evaluates it when the host asks (the start value `tunable_gradient`);
  /// for each tunable, whether a rule gives its derivative or the program
  /// provably does not read it; and the values that the entry hands the
  /// host (mdrtWriteTunableGradient), each the derivative in one site of a
  /// tunable, with what the host adds to it, in kJ/mol per unit of the
  /// tunable at the volume of the build, proportional to 1 / V: the
  /// derivative of the tail of a pair term and of its shift estimate
  /// (D209, D210).
  bool tunableGradient = false;
  enum class GradientOutcome { Rule, Zero };
  std::vector<GradientOutcome> gradientOutcomes;
  struct GradientSlot {
    unsigned tunable = 0;
    unsigned site = 0;
    double volumeConstant = 0.0;
    /// The part that does not depend on the volume (the estimate of the
    /// shift, #224).
    double fixedConstant = 0.0;
  };
  std::vector<GradientSlot> gradientSlots;
  /// The derivative is that of the potential that the forces sample,
  /// shifted to 0 at the cutoff where the run cuts it (D210). After the
  /// derivatives the entry hands over that energy less the energy that
  /// the run reports, and the host adds the estimate of the shift that the
  /// correction for the dispersion does not hold already, in kJ/mol at the
  /// volume of the build, proportional to 1 / V.
  double gradientShiftEnergy = 0.0;
  /// And its part that does not depend on the volume (#224).
  double gradientShiftFixedEnergy = 0.0;
  /// The fields of the particles that the entry hands the host after the
  /// numbers (mdrtWriteTunableGradientField), each the derivative of the
  /// energy in the value of each particle of a field of zeros through
  /// which the sites of one tunable enter their kernels: the derivative in
  /// the site `first` is the value of the field at the particle `second`
  /// (in the order of the input).
  /// A field of a seed of a table (Control::tunableTableSeeds) gives, for
  /// each type, the site `rows[type].site` (or none, -1) the sum of its
  /// values over the particles of the type, times `scale`, plus what the
  /// host adds, `volumeConstant`: the derivative of the correction for the
  /// dispersion and of its shift estimate, in kJ/mol per unit of the
  /// tunable at the volume of the build, proportional to 1 / V.
  struct GradientField {
    unsigned tunable = 0;
    std::vector<std::pair<uint32_t, uint32_t>> sites;
    struct Row {
      int64_t site = -1;
      double scale = 1.0, volumeConstant = 0.0;
      /// The part that does not depend on the volume (#224).
      double fixedConstant = 0.0;
    };
    std::vector<Row> rows;
  };
  std::vector<GradientField> gradientFields;
  /// The arguments of the potential `@tunable` that the entry asks the
  /// derivative in, each with the tunable (its place among the
  /// declarations) whose derivative it gives, so that an error of the
  /// differentiation names the tunable (#256).
  std::vector<std::pair<unsigned, unsigned>> gradientArguments;
  /// What the host adds to the derivative in the charge of each particle,
  /// in kJ/mol/e: the derivative of the self term of the reaction field or
  /// of particle mesh Ewald, which does not depend on the cell, and that
  /// of the background of a net charge, at the volume of the build,
  /// proportional to 1 / V. Empty unless the charges are a tunable whose
  /// derivative the program carries.
  std::vector<double> gradientChargeFixed, gradientChargeVolume;
  /// β in nm⁻¹ and the numbers of points of the grid, for the log.
  double pmeBeta = 0.0;
  int64_t pmeGrid[3] = {0, 0, 0};
  /// Particle mesh Ewald of the dispersion (D162): whether the program has
  /// it, its self term in kJ/mol, which does not depend on the volume and
  /// which the program does not compute, and β in nm⁻¹ and the grid.
  bool ljpme = false;
  double ljpmeSelfEnergy = 0.0;
  double ljpmeBeta = 0.0;
  int64_t ljpmeGrid[3] = {0, 0, 0};

  /// The skin of the neighbor structures, in nm, and the number of
  /// neighbors that they hold per particle.
  double skin;
  int64_t neighborWidth;
  /// The skin of the inner list of a dual list, in nm; 0 keeps one list.
  double pruneSkin = 0.0;

  /// Whether the program puts the particles in the order of their
  /// positions, and the width of the cells that it orders them by, in nm.
  bool reorders = false;
  double orderWidth = 0.0;

  /// A program of segments (Control::segments, D196): its
  /// entry runs every part of a simulation in one activation
  /// (D215); a loop over parts begins each iteration with
  /// `mdrtPartBoundary`, which hands the host the state where it is and
  /// takes the step and the counts of the loops of the next part. `segmentPeriod` is the period of coupling that an iteration of
  /// its outer loop takes, or 0 if the outer loop is over single steps;
  /// `closingSteps` are the steps of a period after its plain steps.
  int64_t segmentPeriod = 0;
  int64_t closingSteps = 1;

  /// A program with tunable parameters (Control::tunables,
  /// D213): its values are those of its buffers, never
  /// constants of its text. Its entry's `%first_call` is 2 for a call that
  /// evaluates the forces of the state that it is given anew, as the first
  /// call does, without the half kick back of leapfrog.
  bool tunable = false;

  /// The values that depend on the state that the run begins from, its
  /// cell above all, which the entry takes as its last arguments rather
  /// than its text holding them as constants (D227):
  /// two programs that differ only in them have the same text, and the
  /// compile cache serves the second (docs/compile-cache.md). Each is
  /// computed on the host as the constant was, so that a run gives the
  /// same numbers to the bit. They are, in this order and where the
  /// program has them:
  ///
  /// - `tilt_bx`, `tilt_cx`, `tilt_cy`: the tilts of a triclinic cell at
  ///   the start, in nm;
  /// - `rest_edge0` to `rest_edge2`: the edges of the cell that the
  ///   reference positions of the restraints are for, in nm, where a
  ///   barostat scales them;
  /// - `baro_constant`, `baro_energy_constant`: what the barostat adds for
  ///   the constant terms (the correction for the dispersion and the
  ///   background of particle mesh Ewald), the trace of their virial times
  ///   the volume and their energy times the volume, in kJ/mol nm³ at the
  ///   volume of the start; with tunables they depend on their values too
  ///   (D213);
  /// - `bstate0` to `bstate8`: the state that the first scaling of a
  ///   barostat that scales every step takes its pressure from, as the
  ///   checkpoint that the run continues keeps it (D92).
  struct StartValue {
    std::string name;
    double value = 0.0;
  };
  std::vector<StartValue> startValues;
};

/// The terms of a model that a tunable enters
/// (D243, docs/python-frames.md): the Lennard-Jones of
/// the pairs within the cutoff with its correction for the dispersion, if
/// sigma or epsilon is tunable; the electrostatics, whole, if the charges
/// are; the pair terms of which a tunable constant is read by the
/// expression, or which read the charges when these are tunable; and the
/// tuple terms of which a tunable parameter is read.
struct DependentTerms {
  bool lennardJones = false, coulomb = false;
  std::vector<unsigned> pairs, tuples;
  bool hasPair(unsigned index) const {
    return std::find(pairs.begin(), pairs.end(), index) != pairs.end();
  }
  bool hasTuple(unsigned index) const {
    return std::find(tuples.begin(), tuples.end(), index) != tuples.end();
  }
  /// Their names, for the plan of a program: "lennard_jones", "coulomb",
  /// "pair:<name>", "tuple:<name>".
  std::vector<std::string> names;
};
DependentTerms getDependentTerms(const Control &control, const System &system);
/// Makes `control` that of a program of the dependent terms alone: sets
/// Control::dependentTerms, leaves the electrostatics out if no tunable
/// enters them, and keeps the observed columns of the terms that stay.
void keepDependentTerms(Control &control, const System &system);

llvm::Expected<Program> buildProgram(const Control &control,
                                     const System &system);

} // namespace driver
} // namespace mdir

#endif // MDIR_DRIVER_BUILDER_H
