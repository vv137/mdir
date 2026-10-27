// The builder of the program of a run, shared by the files of its parts
// (docs/driver.md, Section 2.12): the class Builder and the helpers
// that more than one part takes. Private to lib/Driver.

#ifndef MDIR_LIB_DRIVER_BUILDERIMPL_H
#define MDIR_LIB_DRIVER_BUILDERIMPL_H

#include "mdir/Driver/Builder.h"
#include "mdir/Driver/Expression.h"
#include "mdir/Driver/Influence.h"
#include "mdir/Driver/ModelTerm.h"
#include "mdir/Driver/NeighborCount.h"
#include "mdir/MLIP/Plugin.h"
#include "mdir/Driver/Profile.h"
#include "mdir/Driver/Prolate.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>

using namespace mdir::driver;
namespace prolate = mdir::prolate;
namespace influence = mdir::influence;

/// The cells and the terms of the tables of the splitting of the prolate
/// kernel (D266): within 1e-14 for kernels in f64
/// (mdir-prolate-test), within 4e-10 for those in f32 with 4 terms, which
/// the loop over pairs reads with fewer loads.
inline std::pair<int64_t, int64_t> getProlateTableShape(Precision precision) {
  if (precision == Precision::Double)
    return {64, 10};
  return {128, 4};
}

/// Whether the direct sum of the prolate kernel in a loop over pairs is a
/// Chebyshev series of r², which a table in f32 takes (D94), rather than
/// the cells of `pme_far` (D280): under `fast_math`, where
/// the kernels compute in f32. A kernel in f64 would evaluate the series
/// itself, for which the cells are faster. `MDIR_PROLATE_CELLS=1` keeps
/// the cells, to compare the two.
inline bool takesProlateSeries(const Control &control) {
  if (!control.fastMath || control.precision == Precision::Double)
    return false;
  const char *cells = std::getenv("MDIR_PROLATE_CELLS");
  return !cells || llvm::StringRef(cells).empty() ||
         llvm::StringRef(cells) == "0";
}

using llvm::StringRef;

inline StringRef getName(Element element) {
  return element == Element::F32 ? "f32" : "f64";
}

/// What a build holds where it has made the arrays of its program from the
/// topology and before it computes anything from them: the fields, the
/// tables, and the tuple sets, with what the builder notes beside them
/// (D256). An array that is empty where the program needs
/// numbers is one that a value of a tunable enters: the build that takes
/// these arrays fills it from its model.
struct mdir::driver::BuildKept {
  std::vector<Program::Field> fields;
  std::vector<Program::Table> tables;
  std::vector<Program::TupleSet> tupleSets;
  std::vector<Program::GradientField> gradientFields;
  /// The tunable parameters of the terms over tuples whose derivatives
  /// `@tunable` gives (D230). The derivative in the
  /// parameter of each tuple leaves the program as a field of the
  /// particles: `@tunable` takes fields of zeros, `%tg_d<c>`, and the
  /// parameter of a tuple is its value plus the value of one of them at one
  /// member of the tuple, the member at the place `place`, so that the
  /// derivative of the energy in that field at that particle is the
  /// derivative in the parameter of the tuple. Tuples that share the
  /// member take different fields: the tuple field `tgc_<parameter>` holds
  /// which, from 0 to `count` - 1, and the fields of the parameter are
  /// `first` and those after it.
  struct TupleGradient {
    unsigned tunable = 0, term = 0;
    std::string parameter;
    unsigned place = 0, count = 0, first = 0;
  };
  std::vector<TupleGradient> tupleGradients;
  bool gradientCharges = false;
  /// The arrays kept and the arrays that a build fills, counted where the
  /// arrays were compared with those of a program (keepArraysOf).
  size_t kept = 0, filled = 0;
};

namespace mdir {
namespace driver {
namespace builder {


/// The Coulomb constant in the units of the control file, kcal Å mol⁻¹ e⁻²,
/// from CODATA 2018 (docs/conventions.md). An expression names it
/// `coulomb`.
constexpr double coulombConstant = units::coulomb;

/// The parameter of a pair from those of its two particles.
inline double mix(Mixing mixing, double a, double b) {
  switch (mixing) {
  case Mixing::Arithmetic:
    return 0.5 * (a + b);
  case Mixing::Geometric:
    return std::sqrt(a * b);
  case Mixing::Product:
    return a * b;
  }
  return 0.0;
}

/// A parameter of the types that an expression uses.
struct Parameter {
  std::string name;
  /// The value, if it is the same for all types.
  bool isUniform = true;
  double value = 0.0;
  /// The position among the fields of the program, if it is not.
  unsigned field = 0;
};

class SiteKernel;
class Builder;

/// The borrowed read-only inputs of a term adapter (D361).
/// No accessor returns the builder or its scheduling state.
class TermInputs {
public:
  virtual ~TermInputs() = default;
  virtual const System &getSystem() const = 0;
  virtual const Control &getControl() const = 0;
  const Topology &getTopology() const { return *getSystem().topology; }

  /// Deterministic indexed names; no process-global counter or registration.
  std::string indexedName(StringRef prefix, size_t index) const {
    return (prefix + llvm::Twine(index)).str();
  }

  /// The terms of the potential of a topology.
  enum Term : unsigned {
    LennardJones = 1,
    Coulomb = 2,
    Bonds = 4,
    Angles = 8,
    Dihedrals = 16,
    LennardJones14 = 32,
    Coulomb14 = 64,
    CMaps = 128,
    CoulombExcluded = 256,
    CoulombReciprocal = 512,
    UreyBradleys = 1024,
    HarmonicImpropers = 2048,
    TupleTerms = 4096,
    PairTerms = 8192,
    GeneralizedBorn = 16384,
    Surface = 32768,
    ExternalTerms = 65536,
    LennardJonesExcluded = 131072,
    LennardJonesReciprocal = 262144,
    /// The Coulomb of the pairs within the selection of [free_energy] that
    /// the reciprocal sum of scaled charges leaves out (D161).
    CoulombWithin = 524288,
    AllTerms = 1048575,
    /// Only what depends on λ (D161): the pairs that the selection
    /// decouples, the Coulomb within it, the reciprocal sum, and the terms
    /// whose expressions take a λ.
    Alchemical = 1048576,
  };
  bool usesLambda(StringRef text) const {
    const Control &control = getControl();
    llvm::Expected<Expression> expression =
        Expression::parse(text, control.functions);
    if (!expression) {
      llvm::consumeError(expression.takeError());
      return false;
    }
    return llvm::any_of(expression->getNames(), [&](const std::string &n) {
      return control.isLambda(n);
    });
  }
};

/// Parameter collection adds only the program's term data collections.
class TermParameters : public TermInputs {
public:
  virtual std::vector<Program::Field> &getFields() = 0;
  virtual std::vector<Program::Table> &getTables() = 0;
  virtual std::vector<Program::TupleSet> &getTupleSets() = 0;
};

/// Emission adds the output stream and explicitly named shared operations.
class TermEmission : public TermParameters {
public:
  virtual llvm::raw_ostream &getOutput() = 0;
  virtual void emitTopologyPotential(StringRef name, unsigned terms,
                                    int tupleTerm = -1, int pairTerm = -1,
                                    int externalTerm = -1) = 0;
  virtual std::string getFieldParameters() const = 0;
  virtual std::string getFieldValues(StringRef prefix = "%p_") const = 0;
  virtual std::string getFieldTypes() const = 0;
  virtual std::string getTimeParameter() const = 0;
  virtual std::string getTimeType() const = 0;
  virtual std::string getTimeValue(StringRef name) const = 0;
  virtual unsigned getNumPotentialArguments() const = 0;
  virtual void emitLambdaConstants(StringRef indent) = 0;
  virtual llvm::StringMap<std::string>
  emitCentroidCoordinates(const TupleTerm &term, StringRef indent,
                          StringRef x, StringRef cell, StringRef relations,
                          StringRef prefix) = 0;
  virtual void bindLambdas(llvm::StringMap<std::string> &values) const = 0;
  virtual void bindObserved(StringRef term,
                            llvm::StringMap<std::string> &values) const = 0;
};

/// One output site; borrowed names remain valid for the hook call.
struct TermOutput {
  StringRef indent, x, cell, prefix, step, time, loopSuffix;
  bool deriveTime = false;
};

/// Hooks for the term kinds already moved out of the central builder
/// (D359). The explicit list fixes their emission order.
class TermBuilder {
public:
  struct PotentialSelection {
    unsigned terms;
    int tupleTerm, pairTerm, externalTerm;
    bool alchemicalOnly;
  };
  virtual ~TermBuilder() = default;
  virtual void collectParameters(TermParameters &) const {}
  virtual void emitTopologyPotential(
      TermEmission &, const PotentialSelection &,
      llvm::function_ref<void(StringRef)>) const {}
  virtual void emitTermPotentials(TermEmission &) const {}
  virtual void appendTermNames(const TermInputs &,
                               std::vector<std::string> &) const {}
  virtual void emitEntry(TermEmission &) const {}
  virtual bool hasCoordinateOutput(const TermInputs &) const { return false; }
  virtual void emitAuxiliaryPotentials(TermEmission &) const {}
  virtual void emitOutput(TermEmission &, const TermOutput &) const {}
  virtual std::pair<unsigned, int> findObservedTerm(const TermInputs &,
                                                  StringRef) const {
    return {0, -1};
  }
  virtual std::optional<double> getObservedValue(const TermInputs &, unsigned,
                                                int, StringRef) const {
    return std::nullopt;
  }
};

llvm::ArrayRef<const TermBuilder *> registerTermBuilders();
const TermBuilder &getExternalTermBuilder();
const TermBuilder &getPullTermBuilder();

class Builder : private TermEmission {
public:
  Builder(const Control &control, const System &system, Program &program)
      : control(control), system(system), program(program),
        os(program.module) {
    if (control.dependentTerms && system.topology)
      dependent = getDependentTerms(control, system);
  }

  llvm::Error build();

  /// The arrays of another build of the same model at other values of its
  /// tunables, which this build takes in place of making them from the
  /// topology (D256); none for a build that makes its own.
  const BuildKept *kept = nullptr;
  /// Where a build that makes only its arrays leaves them, and stops.
  BuildKept *capture = nullptr;

private:
  const System &getSystem() const override { return system; }
  const Control &getControl() const override { return control; }
  std::vector<Program::Field> &getFields() override { return program.fields; }
  std::vector<Program::Table> &getTables() override { return program.tables; }
  std::vector<Program::TupleSet> &getTupleSets() override {
    return program.tupleSets;
  }
  llvm::raw_ostream &getOutput() override { return os; }

  /// The stage of `collectTopology` that makes the arrays of the program
  /// from the topology: every field, table, and tuple set, each one number
  /// or one row for each particle, type, or tuple.
  void collectTopologyArrays();
  /// The stage after it: the constants of the energy that the host adds,
  /// from the arrays and the model.
  llvm::Error collectTopologyConstants();
  /// In its place, for a build that was given the arrays of another: takes
  /// them, fills those that hold the values of the tunables from the model,
  /// and checks that none is left without its numbers.
  llvm::Error adoptArrays();
  /// The arrays that the values of the tunables enter, each from the model,
  /// for both stages: the field of the charges, the tables of the
  /// Lennard-Jones, of the weights of their seeds, and of the tunable
  /// constants, a parameter of a term over tuples, and the products of the
  /// charges of the 1-4 pairs.
  void fillCharges(Program::Field &field) const;
  void fillTypeTable(Program::Table &table) const;
  void fillTupleParameter(Program::Field &field, const TupleTerm &term) const;
  void fillPairProducts(Program::Field &field) const;
  /// With Control::dependentTerms, the terms that the program keeps: every
  /// potential has only these, and the host adds to the energy only what
  /// belongs to them (D243).
  std::optional<DependentTerms> dependent;
  llvm::Error collectParameters();
  llvm::Error emitPotential();
  /// Emits the term over triplets `term` into the potential, and returns
  /// its energy (D160).
  std::string emitTripletTerm(size_t index, const TripletTerm &term);
  /// The parameter of the types that `name` of the expression of `term`
  /// takes for one particle of a triplet, `sigma` for `sigma2`, or an empty
  /// string.
  std::string getTripletParameter(const TripletTerm &term,
                                  StringRef name) const;

  /// For a run from a topology: the fields, the tables, and the tuple sets
  /// of the program, and the potential of the topology.
  llvm::Error collectTopology();

  /// Emits the potential `name` of the terms `terms` of the topology; of
  /// the terms given by expressions over tuples, pairs, or positions, only
  /// the one of index `tupleTerm`, `pairTerm`, or `externalTerm` if it is
  /// not -1.
  void emitTopologyPotential(StringRef name, unsigned terms,
                             int tupleTerm = -1, int pairTerm = -1,
                             int externalTerm = -1) override;
  /// Emits `%u_centroid_<name>`, the energy of the term `term` over the
  /// centers of groups, the term `index` over tuples (D139).
  void emitCentroidTerm(size_t index, const TupleTerm &term);
  /// Emits the coordinates of the term `term` over centers, the term
  /// `index` over tuples, at the positions `x` in the cell `cell`, with
  /// the members of its tuples named `relations` and the name of the set:
  /// `r`, `dx`, `dy`, `dz` in Å for two centers, `theta` in radians
  /// otherwise, mapped to their names, which begin with `prefix`.
  llvm::StringMap<std::string>
  emitCentroidCoordinates(const TupleTerm &term, StringRef indent,
                          StringRef x, StringRef cell, StringRef relations,
                          StringRef prefix) override;
  bool hasCoordinateOutput() const {
    return llvm::any_of(registerTermBuilders(), [&](const TermBuilder *term) {
      return term->hasCoordinateOutput(*this);
    });
  }
  /// Whether the force on each virtual site is spread by the derivative of
  /// its placement (`MDIR_SITE_SPREAD=derived`) or by the transpose written
  /// out in `emitSpreadSites` (`written`), D298.
  bool spreadsSitesDerived() const;
  /// For each set of virtual sites, with the derived spreading, the
  /// potential `@spread_<set>`: Σ_t f_t · p(t), the force on each site
  /// against the place of the site from its owner, whose forces are the
  /// transpose of the derivative of the placement.
  void emitSitePotentials();
  /// Emits what `emitSpreadSites` does for one set of sites, by the
  /// derivative of `@spread_<set>`, and returns the virial as it does.
  std::string emitDerivedSpread(StringRef indent, const Program::TupleSet &set,
                                StringRef x, StringRef current,
                                StringRef next, StringRef relations,
                                StringRef virial, StringRef virialResult);
  /// The number of arguments of a potential, before those of its own.
  unsigned getNumPotentialArguments() const override;
  /// Emits generalized Born (D144), `%u_born` and `%u_surface` as `terms`
  /// asks, and passes their names to `add`.
  void emitBorn(unsigned terms, llvm::function_ref<void(StringRef)> add);
  /// κ of the salt of generalized Born in nm⁻¹, scaled by 0.73, or 0.
  double getDebyeKappa() const;
  /// β, the grid, the influence function, and the constant terms of
  /// particle mesh Ewald (docs/pme.md).
  llvm::Error collectPME();
  /// The parameters and tables of the prolate kernel (D266).
  llvm::Error collectProlate();
  /// The field `name` of the places of the particles whose `values` are not
  /// zero, then −1 − the place of each of the others
  /// (D262), which the reciprocal
  /// sum of PME spreads and gathers alone; none where every value is
  /// nonzero, or where a tunable of one of `kinds` may change the values.
  void collectListed(llvm::StringRef name, llvm::ArrayRef<double> values,
                     llvm::ArrayRef<Control::TunableKind> kinds);
  /// Whether the program has the field `name`.
  bool hasField(llvm::StringRef name) const {
    return llvm::any_of(program.fields, [&](const Program::Field &field) {
      return field.name == name;
    });
  }
  /// Whether more than half the particles are in the list `name` of
  /// collectListed: a device then gathers every particle (`gather_all`),
  /// as the gathering over the list costs more than the others where they
  /// are few and share the points of the grid of their neighbors
  /// (D262).
  bool gathersAll(llvm::StringRef name) const {
    for (const Program::Field &field : program.fields)
      if (field.name == name) {
        size_t listed = llvm::count_if(field.values.get(),
                                       [](double v) { return v >= 0.0; });
        return 2 * listed > field.values.size();
      }
    return false;
  }
  /// The grid of particle mesh Ewald: given, or from the largest spacing in
  /// Å; and the table `name` of the factors of its influence function along
  /// each edge.
  llvm::Error collectMesh(StringRef name, const int64_t (&given)[3],
                          double spacing, int64_t order,
                          PMEInfluence kind, int64_t (&grid)[3]);
  /// β, the grid, the coefficients, and the self term of particle mesh
  /// Ewald for the dispersion (D162).
  llvm::Error collectLJPME();
  void emitPrograms();
  /// Collects the values of Program::startValues, which the entry takes
  /// as arguments (D227).
  void collectStartValues();
  void emitEntry();
  /// Emits `@descend`, one step of steepest descent that moves no particle
  /// farther than `%h`, and the loops of a minimization in the entry.
  void emitDescend();
  /// Emits `result`, an f64: the number of constrained lengths of SETTLE
  /// and SHAKE at the positions `x` whose square is off its target by more
  /// than a tolerance of the precision, or is not a number
  /// (D259).
  void emitConstraintFailures(StringRef indent, StringRef x,
                              StringRef result);
  /// Emits, for each set of constraints (SETTLE, then each set of SHAKE), a
  /// test of its lengths at the positions `x` by the tolerance of
  /// `emitConstraintFailures`: the number of lengths that fail, an f64, or,
  /// with `flag`, whether one fails, an i1 (D278).
  /// The value of a set is named `prefix` and the set's name. Returns, for
  /// each set, its number for `mdrtStopConstraintFailure` (0 for SETTLE, k
  /// for the set of SHAKE of k hydrogens) and the name of its value.
  std::vector<std::pair<int, std::string>>
  emitConstraintTests(StringRef indent, StringRef x, StringRef prefix,
                      bool flag);
  /// Emits, after the evaluation of a step, the stop of the run at the
  /// step `%noise_step` for each set whose test `failures` of
  /// `emitConstraintTests` failed, then the stop that a build of the
  /// neighbor structure of the step recorded, if any
  /// (D278).
  void emitConstraintStops(
      StringRef indent,
      const std::vector<std::pair<int, std::string>> &failures);
  /// Emits `result`, the forces `f` over the masses, 0 for particles
  /// without mass, without their parts along the constraints at the
  /// positions `x`: the direction of steepest descent in the metric of the
  /// masses on the surface of the constraints.
  void emitConstrainedDescent(StringRef indent, StringRef x, StringRef f,
                              StringRef result);
  /// Returns the trace `trace` with that of the virial of the constraints
  /// at the start added, where no step has given their forces: for a group
  /// of particles that the constraints keep rigid, Σ (x_k − x_0) · G_k =
  /// Σ (x_k − x_0) · G⁰_k − 2 K_int, with G⁰ = m P(F/m) − F the forces
  /// that keep the accelerations on the constraints and K_int the kinetic
  /// energy of the motion within the group.
  std::string emitStartConstraintTrace(StringRef x, StringRef f,
                                       StringRef v, StringRef trace,
                                       bool axes = false);
  /// Sets `levels`, the loops of the schedule of a run of dynamics.
  void setSchedule();
  void emitMinimization(StringRef velocities);
  /// Whether the minimization of `mdir run` can end at a row of the
  /// energies before its last step: at the first whose largest force is
  /// below a tolerance (D219), or whose step has shrunk below
  /// the floor (D274). A Python simulation checks both
  /// between its parts, on the host, and its program is that without them.
  bool stopsMinimization() const {
    return control.minimize && !control.segments;
  }
  /// A frame at `step` if a reporter's frame is due (programs of segments).
  void emitFrameIfDue(StringRef indent, StringRef step, StringRef positions,
                      StringRef tag);
  /// Emits the energy of each term at the positions `x`, for the log.
  void emitTerms(StringRef x = "%x0");

  /// Emits the loops of the schedule, from `level` inward, and returns the
  /// values that the loop of `level` results in.
  void emitLevel(unsigned level, StringRef indent);
  /// The loops that emitLevel emits are those of the nest from `nestBegin`
  /// to `nestEnd` (0: to the end of `levels`), which begins with the values
  /// `%x<nestOutside>` and the like, after the step `nestStart`. A program of
  /// segments has a second nest, for a period that ends with a step of
  /// energy (D196).
  unsigned nestBegin = 0, nestEnd = 0;
  std::string nestOutside = "0", nestStart = "%start";
  unsigned getNestEnd() const { return nestEnd ? nestEnd : levels.size(); }
  /// The plain step of energy that may end a segment between the steps that
  /// close periods, after `x`, `v`, and `f`; returns the suffix of its
  /// results (D196).
  std::string emitSegmentEnergyStep(StringRef x, StringRef v, StringRef f);
  /// Whether the cell changes in the run, which a barostat does.
  bool changesCell() const { return control.barostat; }
  /// Whether particles are restrained to reference positions.
  /// Whether the cell is triclinic (docs/triclinic.md).
  bool isTriclinic() const {
    return system.tilt[0] != 0.0 || system.tilt[1] != 0.0 ||
           system.tilt[2] != 0.0;
  }  /// The numbers that the barostat keeps of the cell: the edges, or the
  /// diagonal and the tilts b_x, c_x, c_y of a triclinic cell.
  int getCellSize() const { return isTriclinic() ? 6 : 3; }
  std::string getBoxMemoryType() const {
    return isTriclinic() ? "memref<6xf64>" : "memref<3xf64>";
  }
  /// Emits `name` = the cell of the numbers `prefix`_0 to _2, or to _5 for
  /// a triclinic cell.
  void emitCellOf(StringRef indent, StringRef name, StringRef prefix) {
    os << indent << name
       << (isTriclinic() ? " = md.triclinic_cell " : " = md.orthorhombic_cell ");
    for (int k = 0, e = getCellSize(); k != e; ++k)
      os << (k ? ", " : "") << prefix << "_" << k;
    os << "\n";
  }

  bool hasRestraints() const { return !system.restraintConstants.empty(); }
  /// Whether the reference positions of the restraints follow the cell,
  /// which a barostat changes: they are those of the file times
  /// `scaleName`, the edge of the cell over that of the file.
  bool scalesReference() const { return hasRestraints() && changesCell(); }
  std::string getScaleValue() const {
    return scalesReference() ? ", " + scaleName : "";
  }
  std::string getScaleType() const {
    return scalesReference() ? ", vector<3xf64>" : "";
  }
  std::string getScaleParameter() const {
    return scalesReference() ? ", %rest_scale: vector<3xf64>" : "";
  }
  /// Langevin dynamics (D135): the programs of the steps take the number of
  /// the step that they take and the numbers of the particles, which key the
  /// random numbers of each particle (A13).
  std::string getNoiseParameter() const {
    return needsStepNumber() ? ", %noise_step: i64, %noise_ids: !ids" : "";
  }
  std::string getNoiseType() const {
    return needsStepNumber() ? ", i64, !ids" : "";
  }
  /// Whether the programs of the steps take the number of the step: for the
  /// noise of Langevin dynamics, for the time of terms that take it, and
  /// for the stop of a run whose constraints failed
  /// (D278).
  bool needsStepNumber() const {
    return control.isLangevin() || control.isBrownian() || control.usesTime ||
           (hasConstraints() && !control.minimize);
  }
  /// Terms whose expressions take the time `t` (D145): the potentials take
  /// it in ps as their last argument. A program of a step computes it from
  /// the number of the step, `%time`; the entry has `%time0`, that of the
  /// first step.
  std::string getTimeParameter() const override {
    return control.usesTime ? ", %time: f64" : "";
  }
  std::string getTimeType() const override { return control.usesTime ? ", f64" : ""; }
  std::string getTimeValue(StringRef name) const override {
    return control.usesTime ? (", " + name).str() : "";
  }
  /// [free_energy] (D161): whether the run decouples a selection.
  bool decouples() const { return !system.alchemical.empty(); }
  /// The component `name` of λ at the state of the run.
  double getLambda(StringRef name) const {
    return control.freeEnergy.get(name, control.freeEnergy.state);
  }
  /// The potential `@alchemical` takes the components of λ as its last
  /// arguments, `%lambda_<name>`; the others hold them as constants, the
  /// values of the state of the run, under the same names.
  bool lambdaArguments = false;
  std::string getLambdaParameters() const {
    std::string text;
    if (lambdaArguments)
      for (const auto &[name, values] : control.freeEnergy.lambdas)
        text += ", %lambda_" + name + ": f64";
    return text;
  }
  void emitLambdaConstants(StringRef indent) override {
    if (lambdaArguments)
      return;
    for (const auto &[name, values] : control.freeEnergy.lambdas)
      os << indent << "%lambda_" << name << " = arith.constant "
         << formatReal(getLambda(name)) << " : f64\n";
  }
  /// Binds the parameters `lambda_<name>` of the expressions.
  void bindLambdas(llvm::StringMap<std::string> &values) const override {
    for (const auto &[name, v] : control.freeEnergy.lambdas)
      values["lambda_" + name] = "%lambda_" + name;
  }
  /// Whether the potential being emitted is one whose derivatives enter
  /// free energies and gradients, `@alchemical` and `@observe<k>`
  /// (D210): its pair terms cut at the cutoff without a
  /// shift (`lennard_jones_modifier = "NONE"`, a Coulomb cutoff, the direct
  /// sum of PME without `coulomb_modifier`) are shifted to 0 at the cutoff,
  /// each pair within it less its energy at r_c, as the forces are the
  /// gradient of that potential and the dynamics samples it.
  bool shiftsAtCutoff = false;
  /// The truncation of the pair terms of the potential being emitted.
  Truncation getPairTruncation() const {
    return shiftsAtCutoff && control.truncation == Truncation::None
               ? Truncation::Shift
               : control.truncation;
  }
  /// The term of `@observe<k>` and the constants of it that the potential
  /// takes as its last arguments, `%ob_<name>`, in place of their values
  /// (D189).
  std::string observedTerm;
  std::vector<std::string> observedConstants;
  std::string getObservedParameters() const {
    std::string text;
    for (const std::string &name : observedConstants)
      text += ", %ob_" + name + ": f64";
    return text;
  }
  /// Binds the observed constants of the term `term` to the arguments of
  /// `@observe<k>`, after the term has bound its own values.
  void bindObserved(StringRef term,
                    llvm::StringMap<std::string> &values) const override {
    if (term != observedTerm)
      return;
    for (const std::string &name : observedConstants)
      values[name] = "%ob_" + name;
  }
  /// The potential `@tunable` (D230) takes the tunable
  /// constants of the pair terms as its last arguments, `%tg_c<column>`, in
  /// place of the values of the table `tunable_constants`, so that the
  /// differentiation of parameters gives the derivatives in them (D161).
  bool gradientArguments = false;
  std::string getGradientParameters() const {
    std::string text;
    if (!gradientArguments)
      return text;
    for (size_t k = 0; k != control.tunableConstants.size(); ++k)
      text += ", %tg_c" + std::to_string(k) + ": f64";
    for (size_t c = 0; c != getNumGradientSeeds(); ++c)
      text += ", %tg_d" + std::to_string(c) + ": !real";
    return text;
  }
  /// Whether the program carries the derivative in the charges, the last of
  /// Program::gradientFields then: the derivative of `@tunable` in its
  /// field of the charges itself, where the others are those in fields of
  /// zeros, `%tg_d<c>`.
  bool gradientCharges = false;
  size_t getNumGradientSeeds() const {
    return program.gradientFields.size() - (gradientCharges ? 1 : 0);
  }
  /// The tunable parameters of the terms over tuples whose derivatives
  /// `@tunable` gives (D230). The derivative in the
  /// parameter of each tuple leaves the program as a field of the
  /// particles: `@tunable` takes fields of zeros, `%tg_d<c>`, and the
  /// parameter of a tuple is its value plus the value of one of them at one
  /// member of the tuple, the member at the place `place`, so that the
  /// derivative of the energy in that field at that particle is the
  /// derivative in the parameter of the tuple. Tuples that share the
  /// member take different fields: the tuple field `tgc_<parameter>` holds
  /// which, from 0 to `count` - 1, and the fields of the parameter are
  /// `first` and those after it.
  using TupleGradient = BuildKept::TupleGradient;
  std::vector<TupleGradient> tupleGradients;
  /// For each tunable, whether the energy reads it, and the values that
  /// the entry hands the host (Program::gradientSlots), with the column of
  /// the tunable constant of each.
  llvm::Error collectTunableGradient();
  std::vector<unsigned> gradientColumns;
  /// Emits the evaluation of the derivatives in the tunables at the
  /// positions `x`, in a loop that runs once if the host asks and not at
  /// all otherwise, and their call to the host.
  void emitTunableGradientOutput(StringRef indent, StringRef x, StringRef cell,
                                 StringRef prefix, StringRef time);
  /// The potentials `@observe<k>`, each a term of `[output] observe`: the
  /// kind of the term, its index among the terms of its kind, and the
  /// columns of the term, -1 for its energy or the place of the constant
  /// among `observedConstants` of the potential, with the column of each.
  struct ObservedTerm {
    unsigned kind = 0;
    int index = -1;
    std::string name;
    std::vector<std::string> constants;
    std::vector<std::pair<int, size_t>> columns;
  };
  std::vector<ObservedTerm> getObservedTerms() const;
  /// The value of the constant `name` of the observed term `term`.
  double getObservedValue(const ObservedTerm &term, StringRef name) const {
    if (term.kind == PairTerms)
      for (const auto &[key, value] : control.pairs[term.index].constants)
        if (key == name)
          return value;
    for (const TermBuilder *builder : registerTermBuilders())
      if (auto value =
              builder->getObservedValue(*this, term.kind, term.index, name))
        return *value;
    if (term.kind == TupleTerms)
      for (const auto &[key, values] :
           system.topology->tupleTerms[term.index].parameters)
        if (key == name)
          return values.front();
    llvm_unreachable("Control checks the observed constants");
  }
  /// Emits `@observe<k>` for each term of `[output] observe`.
  void emitObservedPotentials();
  /// Emits the energies and the derivatives of `[output] observe` at the
  /// positions `x` in the cell `cell` and their call to the writer
  /// (D189).
  void emitObservablesOutput(StringRef indent, StringRef x, StringRef cell,
                             StringRef prefix, StringRef step,
                             StringRef time);
  /// Whether the expression `text` takes a component of λ.

  /// Emits dH/dλ of each component and the energy of `@alchemical` at
  /// every state, at the positions `x` in the cell `cell`, and their call
  /// to the writer (D161).
  void emitFreeEnergyOutput(StringRef indent, StringRef x, StringRef cell,
                            StringRef prefix, StringRef step,
                            StringRef time);
  /// The constant energies of [free_energy] at each state and their
  /// derivatives (D161).
  void collectFreeEnergyConstants();
  /// The correction for the dispersion of a topology at the volume of the
  /// file, in kJ/mol; with `decoupled`, without the pairs that the
  /// selection of [free_energy] decouples.
  double getTopologyDispersion(bool decoupled) const;
  /// What a shift of the topology's Lennard-Jones to 0 at the cutoff takes
  /// from the pairs within it, estimated at a uniform density, in kJ/mol at
  /// the volume of the file (D210).
  double getTopologyShift(bool decoupled) const;
  /// The expressions of the pair terms, whose tails beyond the cutoff
  /// System::pairTails counts (D209).
  std::vector<Expression> tailExpressions;
  /// The integral of the tail of a pair term and its value at the cutoff,
  /// for the values of the names that its expression reads: a build asks
  /// for the same one for every pair of classes of particles that differ
  /// in what the expression does not read, and again for each derivative
  /// of the correction (D254: 114 ms of an update of JAC and
  /// 164 ms of Factor IX were these quadratures, for a exp(-r/l), which
  /// reads neither the types nor the charges). Not finite where the tail
  /// diverges. The memo is the build's: it holds what one build asked for,
  /// at most one entry for each pair of classes and each point of a
  /// derivative, and ends with it.
  struct TailIntegral {
    bool finite = false;
    double integral = 0.0, edge = 0.0;
  };
  mutable std::vector<std::map<std::vector<uint64_t>, TailIntegral>>
      tailIntegrals;
  /// Collects the tails of the observed pair terms and their derivatives
  /// in the observed constants, for the columns of `observe`.
  void collectObservedTails();
  /// Parses the expressions of the pair terms and refuses a tail that the
  /// quadrature cannot integrate.
  llvm::Error collectPairTails();
  /// The tail of the pair term `index` at the volume of the file, in
  /// kJ/mol, with the values `changes` in place of those of the run (a
  /// component `lambda_<name>` of λ, or a constant of the term), and its
  /// share of the trace of the virial; an error if the integral fails.
  struct PairTail {
    /// The tail, its share of the trace of the virial, and the estimate of
    /// what a shift to 0 at the cutoff takes from the pairs within it
    /// (D210), in kJ/mol at the volume of the file.
    double energy = 0.0, virial = 0.0, shift = 0.0;
    /// The estimate is X (1 - V / (N (4π/3) r_c³)): `shift` is X,
    /// proportional to 1 / V as the tail, and `shiftFixed` the rest, which
    /// does not depend on the volume (#224).
    double shiftFixed = 0.0;
  };
  llvm::Expected<PairTail>
  getPairTail(unsigned index, const llvm::StringMap<double> &changes) const;
  /// 1 − V / (N (4π/3) r_c³), which takes the particle itself out of the
  /// neighbors within r_c in the estimate of the shift.
  double getShiftFactor() const;
  /// Its part that does not depend on the volume once it multiplies a
  /// quantity proportional to 1 / V: −V / (N (4π/3) r_c³) (#224).
  double getShiftFixedFactor() const { return getShiftFactor() - 1.0; }
  /// The sum over the pair terms, at the values `changes`, of the tail and
  /// the estimate of the shift: what the correction adds to the quantities
  /// of the shifted potential (D210).
  /// With `fixed`, the part of the estimates that does not depend on the
  /// volume instead (#224).
  double getPairTails(const llvm::StringMap<double> &changes,
                      bool fixed = false) const;
  /// The derivative of the tail of the pair term `index` in the value
  /// `name`, with that of the estimate of the shift if `shift`: exactly 0
  /// if its expression does not read it.
  /// With `fixed`, that of the part of the estimate that does not depend
  /// on the volume instead (#224).
  double getPairTailDerivative(unsigned index, StringRef name, double value,
                               bool shift = true, bool fixed = false) const;
  /// The values of the components of λ at the state `k`.
  llvm::StringMap<double> getLambdaValues(size_t k) const {
    llvm::StringMap<double> values;
    for (const auto &[name, list] : control.freeEnergy.lambdas)
      values["lambda_" + name] = control.freeEnergy.get(name, k);
    return values;
  }
  /// The self term of particle mesh Ewald and the background of a net
  /// charge at the volume of the file, in kJ/mol, with the charges of the
  /// selection of [free_energy] times 1 − `lambda`.
  std::pair<double, double> getPMEConstants(double lambda) const;
  /// Emits the number of the step about to be taken, from the counter on
  /// the host that each step advances, and returns the operands that a
  /// program of a step takes for it.
  std::string emitNoiseValue(StringRef indent) {
    if (!needsStepNumber())
      return "";
    std::string name = "%noise" + std::to_string(numNoiseSteps++);
    os << indent << name << "_last = memref.load %noise_memory[%c0]"
       << " : memref<1xi64>\n"
       << indent << name << " = arith.addi " << name << "_last, %noise_one"
       << " : i64\n"
       << indent << "memref.store " << name << ", %noise_memory[%c0]"
       << " : memref<1xi64>\n";
    return ", " + name + ", " + idName;
  }
  unsigned numNoiseSteps = 0;
  /// Whether the thermostat rescales the velocities at the end of a period,
  /// rather than acting in every step.
  bool rescalesVelocities() const {
    return control.thermostat && !control.isLangevin();
  }
  /// Emits the restraints at the positions `x`, with the fields of the
  /// prefix `prefix`: `fResult`, the forces `f` with theirs added, and, if
  /// `u` is given, `uResult` and `wResult`, the energy `u` and the virial
  /// `w` with theirs added. The virial of a restraint, Σ d ⊗ F with
  /// d = x − x_ref, has its diagonal only; its trace, −2 U, is whole.
  void emitRestraints(StringRef indent, StringRef x, StringRef prefix,
                      StringRef f, StringRef fResult, StringRef u = "",
                      StringRef uResult = "", StringRef w = "",
                      StringRef wResult = "");
  /// Emits the learned potentials of [[energy.model]] at the positions `x`
  /// in the cell `cell` (D270): `fResult` is the forces `f`
  /// with theirs added and, if `u` is given, `uResult` the energy `u` with
  /// theirs added; returns the name of the virial `w` with theirs added, or
  /// `w` if it is empty. The host evaluates each model on the stream of the
  /// kernels and adds its forces to the buffer of `f` where it is: the call
  /// takes `f` in `outs` and yields `fResult`, so that nothing that reads
  /// the forces can move before it. The energy and the virial are numbers
  /// that the call yields, in a step that asks for them (`u` or `w`).
  std::string emitModelTerms(StringRef indent, StringRef x, StringRef cell,
                             StringRef prefix, StringRef f, StringRef fResult,
                             StringRef u, StringRef uResult, StringRef w,
                             StringRef tag);

  /// Copies the module of each model in stages into the program, its
  /// potential renamed `@staged_<name>` (D294); `@energy`
  /// calls them.
  llvm::Error emitStagedModels();
  std::vector<std::string> stagedPotentials, stagedResults;

  /// Whether the topology has virtual sites.
  bool hasSites() const { return !getSiteSets().empty(); }

  /// Whether the barostat scales the cell within the drift of the last
  /// step of a period (D92).
  bool usesTrotter() const {
    return control.barostat &&
           (control.barostatWork == BarostatWork::Trotter ||
            control.barostatWork == BarostatWork::TrotterFirstOrder);
  }
  /// Whether the barostat of Trotter type counts the energy of a scaling
  /// from the virial after it as well (D92); with a scaling every step the
  /// next scaling takes that virial anyway.
  bool countsAfterScaling() const {
    return usesTrotter() &&
           (control.barostatWork == BarostatWork::Trotter ||
            control.barostatPeriod == 1);
  }
  /// Whether every step scales the cell (a period of one step): the
  /// pressure is then that of the state that the step before left, which
  /// the run keeps in `%trotter_memory` (D92).
  bool scalesEveryStep() const {
    return usesTrotter() && control.barostatPeriod == 1;
  }
  /// The steps that close a period of coupling, after its plain steps: the
  /// last, or the last two with the barostat of Trotter type (D92).
  int64_t getClosingSteps() const {
    return usesTrotter() && !scalesEveryStep() ? 2 : 1;
  }
  /// The tuple sets of the virtual sites, those of Amber first.
  std::vector<const Program::TupleSet *> getSiteSets() const {
    std::vector<const Program::TupleSet *> sets;
    for (StringRef name : {"sites_amber", "sites_linear"})
      for (const Program::TupleSet &set : program.tupleSets)
        if (set.name == name)
          sets.push_back(&set);
    return sets;
  }
  /// Emits the positions `result`: the positions `x` with the virtual sites
  /// placed from the members of the tuples of `relations`, the prefix of
  /// the names of the relations.
  /// Emits the friction and the noise of Langevin dynamics (D135) on the
  /// velocities `velocities` into `result`, in a program of a step that has
  /// `%m`, `%noise_step`, and `%noise_ids`.
  void emitLangevin(StringRef indent, StringRef velocities,
                    StringRef result);
  /// Emits `result` = c `velocities` + (spread / √m) R for each particle of
  /// mass m, with R three normal numbers of stream 2 of the particle and
  /// the step (D135); a particle of mass 0 keeps its velocity.
  void emitNoise(StringRef indent, StringRef velocities, StringRef result,
                 double c, double spread);
  /// Emits `result`, the velocity of a step of Brownian dynamics (D163b),
  /// F / (m γ) + √(2 k_B T / (m γ Δt)) R, from the forces `forces`; the
  /// positions drift by it over the step. A particle of mass 0 does not
  /// drift.
  void emitBrownianVelocity(StringRef indent, StringRef forces,
                            StringRef result);
  void emitPlaceSites(StringRef indent, StringRef x, StringRef result,
                      StringRef relations);
  /// Emits the forces `result`: the forces `f` at the positions `x` with
  /// the force on each virtual site moved to the atoms it is built from.
  /// If `virial` is given, returns the name of that virial with the change
  /// that the move makes to it, `virialResult` if there is a change.
  std::string emitSpreadSites(StringRef indent, StringRef x, StringRef f,
                              StringRef result, StringRef relations,
                              StringRef virial = "",
                              StringRef virialResult = "");

  /// Whether the topology has waters that SETTLE constrains.
  bool hasSettles() const {
    return llvm::any_of(program.tupleSets, [](const Program::TupleSet &set) {
      return set.name == "settles";
    });
  }
  /// The tuple sets of the groups of SHAKE, of one to three hydrogens.
  std::vector<const Program::TupleSet *> getShakeSets() const {
    std::vector<const Program::TupleSet *> sets;
    for (const Program::TupleSet &set : program.tupleSets)
      if (StringRef(set.name).starts_with("shake"))
        sets.push_back(&set);
    return sets;
  }
  bool hasConstraints() const {
    return hasSettles() || !getShakeSets().empty();
  }
  /// Whether the steps that the log reads measure the kinetic energy of
  /// the half steps around their end from the velocities
  /// (D203): with constraints, which the forces alone do
  /// not give it for. Under Langevin dynamics, by g-BAOAB, the half steps
  /// are the drifts of its two A around the end of the step
  /// (D265). Brownian dynamics has no momenta.
  bool measuresHalfSteps() const {
    return hasConstraints() && !control.isBrownian();
  }
  /// Emits `%khs`, in a program of a step, the kinetic energy of each axis
  /// at the half steps around its end, from `velocities`, those it drifted
  /// with after the constraints, and those of a drift of the next step.
  /// With `solvent`, also `%kws`, the kinetic energies of the rigid waters:
  /// that of the velocities of the time of the positions, `%v2`, and
  /// K_half.
  void emitHalfStepKinetic(StringRef velocities, bool leapfrog,
                           bool solvent = false, bool geodesic = false);
  /// Whether the steps of energy report the kinetic energies of the rigid
  /// waters, from which the log gives the temperatures of the solvent and
  /// of the solute (D203).
  bool reportsSolvent() const { return measuresHalfSteps() && hasSettles(); }
  /// Whether the steps that close a period of coupling measure the half
  /// steps for the thermostat, which controls the optimal temperature
  /// (D263).
  bool measuresCoupling() const {
    return measuresHalfSteps() && rescalesVelocities();
  }
  /// Emits the call that gives the host the kinetic energies of the rigid
  /// waters `sums` (emitHalfStepKinetic) before the row of the log.
  void emitWriteSolvent(StringRef indent, StringRef sums);
  /// Emits `result`, the positions `x` with the bonds of the groups of
  /// `set` brought back to their lengths along the bonds of `old`, by the
  /// iterations of SHAKE [Ryckaert1977].
  /// With `failed`, the kernel also yields whether a length it gives is
  /// off its target, `failed`, an i1 (`emitLengthTest`,
  /// D278); so do those of emitBondConstraints and
  /// emitSettlePositions.
  void emitShakePositions(StringRef indent, StringRef old, StringRef x,
                          const Program::TupleSet &set, StringRef result,
                          StringRef failed = "");
  /// A bond of a group of constraints: its members, by their places in the
  /// tuple, and the argument of the kernel that holds its length.
  struct ConstrainedBond {
    unsigned first, second;
    std::string length;
  };
  /// Emits `change`, what bringing the bonds `bonds` of the groups of the
  /// tuple set `setName` (of `arity` members, with the fields `tuple`) back
  /// to their lengths along the bonds of `old` adds to the positions `x`,
  /// by the iterations of Newton of M-SHAKE [Krautler2001].
  void emitBondConstraints(StringRef indent, StringRef setName,
                           unsigned arity, llvm::ArrayRef<ConstrainedBond> bonds,
                           llvm::ArrayRef<std::string> tuple, StringRef old,
                           StringRef x, StringRef change,
                           StringRef failed = "");
  /// Emits, in a kernel of the constraints, whether one of the bonds
  /// `bonds`, given by their vectors and lengths, is off its target by the
  /// tolerance of `emitConstraintFailures`, or is not a number; returns the
  /// name of the i1 (D278).
  std::string emitLengthTest(
      SiteKernel &k,
      llvm::ArrayRef<std::pair<std::string, std::string>> bonds);
  /// Whether the step tests the constraints in their own kernels (the
  /// default) or in loops of their own (`MDIR_CONSTRAINT_TEST=separate`,
  /// for comparison, D278).
  bool testsConstraintsApart() const {
    const char *value = std::getenv("MDIR_CONSTRAINT_TEST");
    return value && StringRef(value) == "separate";
  }
  /// Emits `result`, the positions `x` plus `change`.
  void emitAddChange(StringRef indent, StringRef x, StringRef change,
                     StringRef result);
  /// Emits `result`, the velocities `v` at the positions `x` without their
  /// parts along the bonds of the groups of `set` (RATTLE [Andersen1983]),
  /// and returns the virial `virial` with that of the constraints added, as
  /// `emitSettleVelocities` does.
  /// Returns the virial `virial` with that of the forces of the constraints
  /// of `set` over the first half of the step added, from `change`, what
  /// the constraints added to the positions `old` moved by the drift.
  std::string emitConstraintVirial(StringRef indent,
                                   const Program::TupleSet &set,
                                   StringRef old, StringRef change,
                                   StringRef virial, StringRef virialResult);
  std::string emitShakeVelocities(StringRef indent, StringRef x, StringRef v,
                                  const Program::TupleSet &set,
                                  StringRef result, StringRef virial,
                                  StringRef virialResult);
  /// Returns the virial `virial` with that of the forces of the constraints
  /// at the positions `x`, the velocities `v` (with none along the bonds),
  /// and the forces `f` added: the forces G along the bonds that keep them
  /// at their lengths, e·(a_q − a_p) = −|v_q − v_p|²/d with a = (F + G)/m,
  /// and Σ (x_j − x_0) ⊗ G_j, for the rigid waters and each set of SHAKE.
  /// Under Langevin dynamics the changes that the constraints make within a
  /// step hold part of the friction and the noise, and these forces do not
  /// (D135).
  std::string emitConstraintForceVirial(StringRef indent, StringRef x,
                                        StringRef v, StringRef f,
                                        StringRef virial);
  /// Emits `result`, the positions `x` with the rigid waters brought back
  /// to their shape from `old`, the positions before the drift, and
  /// `change`, what that adds to the positions [Miyamoto1992].
  void emitSettlePositions(StringRef indent, StringRef old, StringRef x,
                           StringRef change, StringRef result,
                           StringRef failed = "");
  /// Emits `result`, the velocities `v` at the positions `x` without their
  /// parts along the bonds of the rigid waters. If `virial` is given,
  /// returns the name of that virial with that of the constraints added,
  /// `virialResult`.
  std::string emitSettleVelocities(StringRef indent, StringRef x, StringRef v,
                                   StringRef result, StringRef virial = "",
                                   StringRef virialResult = "");

  /// What a coupling leaves: the positions, the velocities, and the forces
  /// of the positions.
  /// The scaling of a period of coupling made within the drift of its last
/// step (D92): its factor, the inverse, and the logarithm; the trace of
/// the virial of the groups before it, with those of the constant terms;
/// the volume after it, and its cell; the kinetic energy of the velocities
/// that it scaled; and the trace of the virial of the groups after it
/// (emitMolecularTrace), where the count takes it.
struct TrotterScaling {
  std::string mu, muinv, logMu, workBefore, newVolume, cell, scale,
      kineticHalf, groupsAfter;
};

struct Coupled {
    std::string positions, velocities, forces;
  };
  /// Emits the coupling at the end of the step `step` of the positions
  /// `positions`, the velocities `velocities`, and the forces `forces`,
  /// whose potential energy is `energy` and the trace of whose virial is
  /// `trace`: the removal of the motion of the center of mass, the
  /// thermostat, and the barostat, which with the exact work evaluates the
  /// scaled positions and returns their forces (D77).
  /// The groups that the constraints keep rigid: the waters of SETTLE and
  /// the sets of SHAKE.
  std::vector<const Program::TupleSet *> getRigidGroups() const;
  /// `trace` with twice the kinetic energy of the motion within each rigid
  /// group added: the trace of the virial of the groups, which scale with
  /// their centers of mass, for a step that does not scale (the
  /// first-order count).
  std::string emitGroupTrace(StringRef indent, StringRef trace,
                             StringRef positions, StringRef velocities,
                             StringRef cell, StringRef masses,
                             StringRef relations, StringRef tag);
  /// `diagonal`, the diagonal of the virial of the forces `forces` at the
  /// positions `positions` without that of the constraints, less
  /// Σ (x_j − X) ⊙ F_j over the particles of each rigid group about its
  /// center of mass X: the diagonal of the virial of the groups,
  /// Σ X ⊙ F_group, whose element a is −dU/d ln μ_a when they scale with
  /// their centers along axis a (D116, D119). The forces of the
  /// constraints, internal to the groups, drop out of it.
  std::string emitMolecularDiagonal(StringRef indent, StringRef diagonal,
                                 StringRef positions, StringRef forces,
                                 StringRef cell, StringRef masses,
                                 StringRef relations, StringRef tag);
  /// The kinetic energy of the centers of the groups that the barostat
  /// scales (the rigid groups, every other particle alone), of each axis,
  /// at the velocities `velocities` of the time of the positions
  /// `positions` with their forces `forces`: K_g, without the motion of the
  /// center of mass if the run removes it; the term (Δt²/8) Σ F_g² / M_g;
  /// and their sum, K_half of the groups [Jung2020], exact with or without
  /// constraints (D264).
  struct GroupKinetic {
    std::string kinetic, forceTerm, half;
  };
  GroupKinetic emitGroupKinetic(StringRef indent, StringRef positions,
                                StringRef velocities, StringRef forces,
                                StringRef tag);
  /// `positions` scaled by `mu`, a scale for each axis, each rigid group
  /// with its center of mass, keeping its shape (`oneMinusMu` is 1 − mu).
  std::string emitGroupScaling(StringRef indent, StringRef positions,
                               StringRef mu, StringRef oneMinusMu,
                               StringRef cell, StringRef masses,
                               StringRef relations, StringRef tag);
  /// `groups` is the diagonal of the virial of the groups, which the
  /// pressure of the barostat takes with their kinetic energy at the half
  /// steps (D264); `optimal`, if given, the kinetic
  /// energy of the optimal temperature, K + (2/3)(K_half − K), with that
  /// of the center of mass, which the factor of the thermostat takes
  /// (D263).
  Coupled emitCoupling(StringRef indent, StringRef positions,
                       StringRef velocities, StringRef forces,
                       StringRef energy, StringRef tag, StringRef step,
                       StringRef trace, StringRef groups, StringRef optimal,
                       const TrotterScaling *trotter = nullptr);
  /// Emits and returns the name of K_half - K of each axis at the end of a
  /// step, a vector<3xf64>: from `measured`, the K_half that the step
  /// returned (measuresHalfSteps), less the kinetic energy of `full`, the
  /// velocities of the time of the positions; without constraints, from the
  /// forces, (dt^2 / 8) sum F_a^2 / m, which is that difference exactly
  /// [Jung2018]; otherwise none, an empty name.
  std::string emitKineticExcess(StringRef indent, StringRef tag,
                                StringRef measured, StringRef forces,
                                StringRef full);
  /// The pressure of the virial trace `trace` and the kinetic energy
  /// `kinetic`, and the strain that the barostat takes from it: the scale
  /// `%mu<tag>` of the positions, its inverse `%muinv<tag>`, its logarithm
  /// `%bs3<tag>`, the volume before `%bv<tag>`, and the new cell, edges
  /// `%bn<tag>_k`, which it stores where the loops take the cell from.
  void emitStrain(StringRef indent, StringRef kinetic, StringRef trace,
                  StringRef tag, StringRef step);
  /// Before the step of a period of coupling that scales the cell within
  /// its drift (the Trotter type of D92): the strain from the pressure of
  /// the velocities `velocities` of the step before, whose virial has the
  /// trace `trace`; `groups` is the trace of the virial of its groups
  /// (emitMolecularTrace). `givenKinetic`, if given, is the kinetic energy
  /// without the center of mass, computed before.
  /// The pressure is that of the groups (D264):
  /// their virial `groups` and the kinetic energy of their centers at the
  /// half steps, from `positions`, `velocities`, and `forces`, or
  /// `givenKinetic`, computed before.
  TrotterScaling emitTrotterStrain(StringRef indent, StringRef positions,
                                   StringRef velocities, StringRef forces,
                                   StringRef groups, StringRef tag,
                                   StringRef step,
                                   StringRef givenKinetic = "");
  TrotterScaling finishTrotterStrain(StringRef indent, StringRef groups,
                                     StringRef tag);
  /// The kinetic energy of `velocities` without that of the center of mass
  /// if the run removes it.
  std::string emitKineticWithoutCenter(StringRef indent, StringRef velocities,
                                       StringRef masses, StringRef tag);
  /// Stores the trace of the virial `trace`, that of the rigid groups
  /// `groups`, and the kinetic energy `kinetic` in `%trotter_memory`.
  void emitStoreTrotterState(StringRef indent, StringRef trace,
                             StringRef groups, StringRef kinetic);
  /// The stores of emitStoreTrotterState so far, which number their values.
  /// It counts per build, so that the text of a program depends only on
  /// its inputs, not on what the process built before (#152).
  unsigned trotterStores = 0;

  /// The arguments that pass the fields of the parameters on: their
  /// declarations, their values, and their types, each after a comma.
  std::string getFieldParameters() const override;
  /// The fields of the parameters, as arguments that follow others.
  /// `prefix` comes before the name of each.
  std::string getFieldValues(StringRef prefix = "%p_") const override;
  std::string getFieldTypes() const override;

  /// Emits the order of the particles at the positions `positions`, and
  /// the fields in that order. `from` and `to` are the ends of the names
  /// of the fields before and after.
  void emitReorder(StringRef indent, StringRef from, StringRef stateTo,
                   StringRef otherTo, bool withForces,
                   StringRef velocities, StringRef forcesTo = "");
  /// Emits the members of the tuples of the topology at the places that
  /// the numbers `ids` give the particles, named `prefix` and the name of
  /// the set.
  void emitRenumber(StringRef indent, const llvm::Twine &ids,
                    const llvm::Twine &prefix);

  bool isLeapfrog() const {
    return control.integrator == Integrator::Leapfrog;
  }
  bool isRestart() const { return !control.restartInput.empty(); }
  /// Whether the entry begins a run or continues the last segment as its
  /// argument %first_call says (D196, D211): a
  /// program of segments is one program for the first segment and the
  /// later ones, whose work at the start is a branch on that argument.
  bool branchesStart() const { return control.segments; }
  /// Emits `chunk`, code of the start, as the branch of %is_first: a loop of
  /// one iteration on the first call and none on the others, whose results
  /// `results` are the values `inside` of the chunk on the first call and the
  /// values given, `otherwise`, which it carries, on a later one. The chunk
  /// is written as it is: a value that it defines under the name of a result
  /// is its own (a name of a region hides that of the result after it), and
  /// its uses of the values given become the loop's arguments when the
  /// module is parsed (`mdir.carries`, carryLoopValues in ProgramModule.cpp,
  /// #568).
  /// With `segment`, the loop is a segment of the run (mdrt.segment): the
  /// neighbor structures that the chunk builds do not carry into the steps
  /// after it (D218).
  void emitStartBranch(std::string chunk, llvm::ArrayRef<std::string> results,
                       llvm::ArrayRef<std::string> inside,
                       llvm::ArrayRef<std::string> otherwise,
                       bool segment = false);
  bool emittedFirstTrips = false;
  unsigned startBranches = 0;
  /// The loop over the parts of a program of segments
  /// (D215): one activation of the entry runs every part of
  /// a simulation. Each iteration begins at the end of a part (or at the
  /// start), where it hands the host the state `carried` (positions,
  /// velocities, and forces first) where it is, with the number of their
  /// components that are not numbers, and takes from the host the step that
  /// the next part begins after and its counts; then it runs `chunk`, the
  /// steps of the part, and yields `yielded`. In the chunk the values
  /// carried and %start are renamed.
  void emitPartLoop(std::string chunk, llvm::ArrayRef<std::string> carried,
                    llvm::ArrayRef<std::string> types,
                    llvm::ArrayRef<std::string> yielded,
                    llvm::ArrayRef<std::string> boundary);
  /// The counts of the loops of a part, from %part_counts.
  void emitPartCounts();
  static int64_t getPartCountsSize() { return 9; }
  /// Whether a run from a checkpoint evaluates the forces of its first step
  /// (D172).
  bool recomputes() const { return isRestart() && control.restartRecomputes; }

  const Control &control;
  const System &system;
  Program &program;
  llvm::raw_string_ostream os;

  /// The cutoffs of the pair lists of the models of [[energy.model]], in
  /// nm, which the plugin reads from their files, and whether the program
  /// builds their pair lists (D277): not where the plugin
  /// searches its own (`MDIR_MLIP_PAIRS=search`).
  std::vector<double> modelCutoffs;
  bool modelEdges = false;

  /// The names of the masses, of the fields of the parameters, and of the
  /// numbers of the particles, where the code is that is being emitted.
  /// Inside a segment they are those of the order of the segment.
  std::string massName = "%m";
  std::string fieldPrefix = "%p_";
  std::string idName = "%id";
  /// The factor of the reference positions of the restraints, which
  /// follows the cell (scalesReference).
  std::string scaleName = "%rest_scale";
  /// The name of the cell, which changes with a barostat.
  std::string cellName = "%cell";

  std::vector<Expression> expressions;
  std::vector<Parameter> parameters;

  /// For each pair term that looks its parameters up: the table of each
  /// parameter, by name. Empty for a term that gathers them.
  std::vector<llvm::StringMap<unsigned>> termTables;

  /// The values of the parameters of each term for each pair of types, in
  /// the units of the control file: the constants of the term, and the
  /// parameters as the mixing rules and the overrides give them.
  llvm::Error collectPairValues(
      unsigned term,
      std::vector<std::vector<llvm::StringMap<double>>> &values);
  llvm::Error computeDispersion();

  /// The type of the field `field` in the program.
  static StringRef getFieldType(const Program::Field &field) {
    return field.isInteger ? "!ids" : "!real";
  }

  /// The loops of the schedule, from the outside in: their number of
  /// iterations. The last loop is the one over steps.
  struct Level {
    StringRef name;
    int64_t count;
  };
  std::vector<Level> levels;
  /// The number of steps between two energies.
  /// Whether the frames are written at the ends of the intervals between
  /// energies, which have their period (Control::getEnergyLoopPeriod).
  bool framesAtEnergies = false;
  /// What the text of the loop over frames holds that its assembly cannot
  /// take (checkFrameLoop), or empty: the error of the build.
  std::string frameLoopError;
};


} // namespace builder
} // namespace driver
} // namespace mdir

/// The constants of the reaction field at the cutoff r_c, in nm (D140): k,
/// with which the field adds f q q k r² to a pair, and c, with which the
/// potential f q q (1/r + k r² − c) is 0 at r_c [Barker1973, Tironi1995].
/// With the relative permittivity ε beyond the cutoff,
/// k = (ε − 1) / ((2ε + 1) r_c³); a conductor, ε = 0 in the control file,
/// has the limit 1 / (2 r_c³).
inline std::pair<double, double> getReactionField(const Control &control) {
  double rc = control.cutoffDistance * units::length;
  double epsilon = control.reactionFieldDielectric;
  double k = epsilon == 0.0
                 ? 1.0 / (2.0 * rc * rc * rc)
                 : (epsilon - 1.0) / ((2.0 * epsilon + 1.0) * rc * rc * rc);
  return {k, 1.0 / rc + k * rc * rc};
}

inline llvm::Error makeError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

//===----------------------------------------------------------------------===//
// Runs from a topology
//===----------------------------------------------------------------------===//

/// The Coulomb constant in kJ nm mol⁻¹ e⁻², from CODATA 2018.
constexpr double coulombInternal = 138.935457644;

/// g(x) = exp(−x²) (1 + x² + x⁴/2), the share of the dispersion −c_i c_j /
/// r⁶ at x = β r that the direct terms of particle mesh Ewald keep
/// [Essmann1995].
inline double getDispersionScreen(double x) {
  double x2 = x * x;
  return std::exp(-x2) * (1.0 + x2 + 0.5 * x2 * x2);
}

/// The diagonal of the virial `virial`, a vector of three, whose elements
/// enter the pressure of each axis.
inline void emitDiagonal(llvm::raw_ostream &os, StringRef result,
                         StringRef virial, StringRef indent) {
  for (StringRef part : {"0", "4", "8"})
    os << indent << result << "_" << part << " = vector.extract " << virial
       << "[" << part << "] : f64 from vector<9xf64>\n";
  os << indent << result << " = vector.from_elements " << result << "_0, "
     << result << "_4, " << result << "_8 : vector<3xf64>\n";
}

/// `result`, the scale of each axis of the reference positions of the
/// restraints: the edges of the cell over those of the file
/// (`%rest_edges`).
inline void emitReferenceScale(llvm::raw_ostream &os, StringRef indent,
                               StringRef result, const std::string &e0,
                               const std::string &e1, const std::string &e2) {
  os << indent << result << "_e = vector.from_elements " << e0 << ", " << e1
     << ", " << e2 << " : vector<3xf64>\n"
     << indent << result << " = arith.divf " << result
     << "_e, %rest_edges : vector<3xf64>\n";
}

/// The sum of the elements of the vector of three `vector`.
inline void emitSum3(llvm::raw_ostream &os, StringRef result,
                     StringRef vector, StringRef indent) {
  os << indent << result << " = vector.reduction <add>, " << vector
     << " : vector<3xf64> into f64\n";
}

namespace mdir {
namespace driver {
namespace builder {
/// Emits the arithmetic of a kernel of virtual sites on vectors of three
/// numbers, with names of its own.
class SiteKernel {
public:
  SiteKernel(llvm::raw_ostream &os, StringRef indent,
             StringRef prefix = "%vs")
      : os(os), indent(indent.str()), prefix(prefix.str()) {}

  std::string vector(StringRef op, StringRef a, StringRef b) {
    std::string name = fresh();
    os << indent << name << " = arith." << op << " " << a << ", " << b
       << " : vector<3xf64>\n";
    return name;
  }
  std::string real(StringRef op, StringRef a, StringRef b) {
    std::string name = fresh();
    os << indent << name << " = arith." << op << " " << a << ", " << b
       << " : f64\n";
    return name;
  }
  std::string negate(StringRef a) {
    std::string name = fresh();
    os << indent << name << " = arith.negf " << a << " : vector<3xf64>\n";
    return name;
  }
  std::string dot(StringRef a, StringRef b) {
    std::string product = vector("mulf", a, b), name = fresh();
    os << indent << name << " = vector.reduction <add>, " << product
       << " : vector<3xf64> into f64\n";
    return name;
  }
  std::string norm(StringRef a) {
    std::string square = dot(a, a), name = fresh();
    os << indent << name << " = math.sqrt " << square << " : f64\n";
    return name;
  }
  std::string splat(StringRef a) {
    std::string name = fresh();
    os << indent << name << " = vector.broadcast " << a
       << " : f64 to vector<3xf64>\n";
    return name;
  }
  std::string scale(StringRef factor, StringRef a) {
    return vector("mulf", splat(factor), a);
  }
  std::string component(StringRef a, int index) {
    std::string name = fresh();
    os << indent << name << " = vector.extract " << a << "[" << index
       << "] : f64 from vector<3xf64>\n";
    return name;
  }
  std::string zero() {
    std::string name = fresh();
    os << indent << name
       << " = arith.constant dense<0.0> : vector<3xf64>\n";
    return name;
  }
  std::string constant(double value) {
    std::string name = fresh();
    os << indent << name << " = arith.constant " << formatReal(value)
       << " : f64\n";
    return name;
  }
  std::string root(StringRef a) {
    std::string name = fresh();
    os << indent << name << " = math.sqrt " << a << " : f64\n";
    return name;
  }
  std::string compose(StringRef x, StringRef y, StringRef z) {
    std::string name = fresh();
    os << indent << name << " = vector.from_elements " << x << ", " << y
       << ", " << z << " : vector<3xf64>\n";
    return name;
  }
  std::string cross(StringRef a, StringRef b) {
    std::string a0 = component(a, 0), a1 = component(a, 1),
                a2 = component(a, 2), b0 = component(b, 0),
                b1 = component(b, 1), b2 = component(b, 2);
    auto term = [&](StringRef p, StringRef q, StringRef r, StringRef t) {
      return real("subf", real("mulf", p, q), real("mulf", r, t));
    };
    return compose(term(a1, b2, a2, b1), term(a2, b0, a0, b2),
                   term(a0, b1, a1, b0));
  }
  std::string unit(StringRef a) { return vector("divf", a, splat(norm(a))); }
  /// `op a`, of an f64, such as `math.absf`.
  std::string unary(StringRef op, StringRef a) {
    std::string name = fresh();
    os << indent << name << " = " << op << " " << a << " : f64\n";
    return name;
  }
  /// `arith.cmpf predicate, a, b` of two f64.
  std::string compare(StringRef predicate, StringRef a, StringRef b) {
    std::string name = fresh();
    os << indent << name << " = arith.cmpf " << predicate << ", " << a
       << ", " << b << " : f64\n";
    return name;
  }
  /// `arith.op a, b` of two i1.
  std::string boolean(StringRef op, StringRef a, StringRef b) {
    std::string name = fresh();
    os << indent << name << " = arith." << op << " " << a << ", " << b
       << " : i1\n";
    return name;
  }
  /// `a x + b y + c z`.
  std::string combine(StringRef a, StringRef x, StringRef b, StringRef y,
                      StringRef c, StringRef z) {
    return vector("addf", vector("addf", scale(a, x), scale(b, y)),
                  scale(c, z));
  }

private:
  std::string fresh() { return prefix + std::to_string(next++); }

  llvm::raw_ostream &os;
  std::string indent;
  std::string prefix;
  unsigned next = 0;
};

/// The frame of the extra point of Amber: the unit vectors from the owner
/// to the two hydrogens, `u` and `v`, their lengths, and the unit vector
/// along their sum, with the length of the sum.
struct AmberFrame {
  std::string u, v, lengthU, lengthV, sum, lengthSum;

  AmberFrame(SiteKernel &k, StringRef toJ, StringRef toK) {
    lengthU = k.norm(toJ);
    lengthV = k.norm(toK);
    u = k.vector("divf", toJ, k.splat(lengthU));
    v = k.vector("divf", toK, k.splat(lengthV));
    sum = k.vector("addf", u, v);
    lengthSum = k.norm(sum);
  }

  /// The forces on the owner and on the two hydrogens from the force
  /// `force` on the extra point at the distance `d`: the transpose of the
  /// derivative of the position. Perpendicular to the bisector, then to
  /// each bond: the extra point moves with the directions of the bonds,
  /// not with their lengths.
  std::array<std::string, 3> spread(SiteKernel &k, StringRef force,
                                    StringRef d) {
    std::string bisector = k.vector("divf", sum, k.splat(lengthSum));
    std::string across =
        k.vector("subf", force, k.scale(k.dot(force, bisector), bisector));
    auto toHydrogen = [&](StringRef unit, StringRef length) {
      std::string factor =
          k.real("divf", d, k.real("mulf", length, lengthSum));
      std::string perpendicular =
          k.vector("subf", across, k.scale(k.dot(across, unit), unit));
      return k.scale(factor, perpendicular);
    };
    std::string first = toHydrogen(u, lengthU),
                second = toHydrogen(v, lengthV);
    std::string owner =
        k.vector("subf", k.vector("subf", force, first), second);
    return {owner, first, second};
  }
};
/// The place of a site from its owner, p(d_j, d_k), in a kernel whose
/// arguments `%vs_j` and `%vs_k` are the displacements of the other two
/// atoms from the owner and `%vs_a` and `%vs_b` the parameters of the site:
/// a d_j + b d_k, or for Amber's extra point a along the bisector of the
/// two bonds.
inline std::string emitPlacement(SiteKernel &k, bool amber) {
  if (amber) {
    AmberFrame frame(k, "%vs_j", "%vs_k");
    return k.scale(k.real("divf", "%vs_a", frame.lengthSum), frame.sum);
  }
  return k.vector("addf", k.scale("%vs_a", "%vs_j"),
                  k.scale("%vs_b", "%vs_k"));
}
} // namespace builder
} // namespace driver
} // namespace mdir

/// The kernel of the kinetic energy of one particle.
inline void emitKineticEnergy(llvm::raw_ostream &os, StringRef result,
                              StringRef velocities, StringRef masses,
                              StringRef indent) {
  os << indent << result << " = md.sum_particles gather(" << velocities
     << ", " << masses << " : !vec, !real) {\n"
     << indent << "^bb0(%v_i: vector<3xf64>, %m_i: f64):\n"
     << indent << "  %half = arith.constant 5.0e-01 : f64\n"
     << indent << "  %sq = arith.mulf %v_i, %v_i : vector<3xf64>\n"
     << indent
     << "  %v2 = vector.reduction <add>, %sq : vector<3xf64> into f64\n"
     << indent << "  %mv2 = arith.mulf %m_i, %v2 : f64\n"
     << indent << "  %ke = arith.mulf %half, %mv2 : f64\n"
     << indent << "  md.yield %ke : f64\n"
     << indent << "} : f64\n";
}

/// Whether `line` names the value `name` (and not a longer name that
/// begins with it).
inline bool namesValue(StringRef line, StringRef name) {
  for (size_t at = line.find(name); at != StringRef::npos;
       at = line.find(name, at + 1)) {
    size_t end = at + name.size();
    if (end == line.size() ||
        !(llvm::isAlnum(line[end]) || StringRef("_$.-").contains(line[end])))
      return true;
  }
  return false;
}

/// What the assembly of the loop over frames (emitEntry) assumes of the text
/// that it makes the body of the loop from, checked (#302): the
/// assembly matches text, and a line that it does not match still parses,
/// compiles, and gives the right numbers of the frames of a test. Returns
/// the error that names the first such line and says what to do, or an
/// empty string.
///
/// - The arguments of the entry that a frame replaces. The positions and
///   the cell of a frame are values of the loop; the positions and the cell
///   that the entry is given, and the memory of the barostat that is filled
///   from them, are those of the program, the same at every frame.
/// - An evaluation. The loop evaluates what the start of a run evaluates; a
///   program whose start evaluates nothing would wait for frames and give
///   nothing.
///
/// A call in the generic form or a `return` in the loop, and a name of the
/// loop taken twice, need no check here: the parser refuses them. Memory of
/// the stack needs none either (#309): the rows that the evaluation takes
/// stand in the loop, as they stand in the loops over steps of a run, and
/// the lowerings keep the stack of the entry, which never returns, at one
/// depth whatever the text holds: for the CPU `hoist-static-allocas` takes
/// every allocation once, where the function begins, and refuses one whose
/// size is not a constant (D117); for a device the stack of each iteration
/// of a loop of the host is given back where the iteration ends.
inline std::string checkFrameLoop(StringRef body) {
  static const char *const replaced[] = {
      "%positions", "%lx", "%ly", "%lz", "%tilt_bx", "%tilt_cx", "%tilt_cy",
      "%box_memory"};
  bool evaluates = false;
  for (StringRef rest = body; !rest.empty();) {
    auto [line, tail] = rest.split('\n');
    rest = tail;
    auto fail = [&](const llvm::Twine &what) {
      return ("the program of frames: the line '" + line.trim() +
              "' of the loop over frames " + what +
              " (Builder::emitEntry; docs/python-frames.md, the frames "
              "inside one activation)")
          .str();
    };
    for (StringRef name : replaced)
      if (namesValue(line, name))
        return fail("takes '" + name +
                    "', which is of the program and not of the frame: in "
                    "the loop the positions of the frame are '%x0' and its "
                    "cell is '%cell', made from '%frame_cell'");
    evaluates |= line.contains("md.evaluate ");
  }
  if (!evaluates)
    return "the program of frames: the start of this program evaluates "
           "nothing, so its loop over frames would give nothing "
           "(Builder::emitEntry)";
  return std::string();
}

#endif // MDIR_LIB_DRIVER_BUILDERIMPL_H
