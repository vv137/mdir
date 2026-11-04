// The builder of the program of a run, shared by the files of its parts
// (docs/driver.md, Section 2.12): the class Builder and the helpers
// that more than one part takes. Private to lib/Driver.

#ifndef MDIR_LIB_DRIVER_BUILDERIMPL_H
#define MDIR_LIB_DRIVER_BUILDERIMPL_H

#include "mdir/Driver/Builder.h"
#include "TermIR.h"
#include "DynamicsIR.h"
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
#include <memory>

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

/// Per-builder custom-pair data; memoization is logically const and never
/// changes the physical inputs or the generated program.
struct PairTermPlan {
  std::vector<Expression> expressions;
  std::vector<llvm::StringMap<unsigned>> tables;
  std::vector<Expression> tailExpressions;
  struct TailIntegral {
    bool finite = false;
    double integral = 0.0, edge = 0.0;
  };
  mutable std::vector<std::map<std::vector<uint64_t>, TailIntegral>>
      tailIntegrals;
};
struct PairTermTail {
  double energy = 0.0, virial = 0.0, shift = 0.0, shiftFixed = 0.0;
};

class Builder;

/// The borrowed read-only inputs of a term adapter (D361).
/// No accessor returns the builder or its scheduling state.
class TermInputs {
public:
  virtual ~TermInputs() = default;
  virtual const System &getSystem() const = 0;
  virtual const Control &getControl() const = 0;
  virtual Element getStateElement() const = 0;
  virtual Element getForceElement() const = 0;
  virtual bool hasSites() const = 0;
  virtual bool hasConstraints() const = 0;
  virtual const PairTermPlan &getPairPlan() const = 0;
  virtual llvm::ArrayRef<Parameter> getTypeParameters() const = 0;
  virtual llvm::ArrayRef<Program::Table> getParameterTables() const = 0;
  virtual llvm::ArrayRef<Program::Field> getParameterFields() const = 0;
  virtual bool includesPair(unsigned index) const = 0;
  virtual double getShiftFixedFactor() const = 0;
  virtual llvm::ArrayRef<BuildKept::TupleGradient> getTupleGradients() const = 0;
  virtual bool hasGradientArguments() const = 0;
  virtual bool hasGradientCharges() const = 0;
  virtual bool hasTunableGradient() const = 0;
  virtual bool includesTuple(unsigned index) const = 0;
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

inline Program::TupleSet &addTupleSet(std::vector<Program::TupleSet> &sets,
                                      StringRef name, unsigned arity) {
  Program::TupleSet set;
  set.name = name.str();
  set.arity = arity;
  sets.push_back(std::move(set));
  return sets.back();
}
inline size_t addTupleField(Program::TupleSet &set, StringRef name) {
  Program::Field field;
  field.name = name.str();
  set.fields.push_back(std::move(field));
  return set.fields.size() - 1;
}

/// Parameter collection adds only the program's term data collections.
class TermParameters : public TermInputs {
public:
  virtual std::vector<Program::Field> &getFields() = 0;
  virtual std::vector<Program::Table> &getTables() = 0;
  virtual std::vector<Program::TupleSet> &getTupleSets() = 0;
  virtual std::vector<Program::GradientField> &getGradientFields() = 0;
  virtual std::vector<BuildKept::TupleGradient> &getMutableTupleGradients() = 0;
  virtual PairTermPlan &getMutablePairPlan() = 0;
  virtual std::vector<Parameter> &getMutableTypeParameters() = 0;
  virtual std::vector<Program::GradientSlot> &getGradientSlots() = 0;
  virtual std::vector<unsigned> &getGradientColumns() = 0;
  virtual std::vector<double> &getObservableVolumeConstants() = 0;
  virtual std::vector<double> &getObservableFixedConstants() = 0;
  virtual void setBareDispersion(double energy, double virial) = 0;
  virtual void addDispersion(double energy, double fixed, double virial) = 0;
  virtual void addGradientShift(double scaled, double fixed) = 0;
};

/// Actual values at a bare potential boundary; semantic fields/tables are borrowed.
struct TermNativePotential {
  mlir::Value positions, cell, neighborhood, time;
  llvm::function_ref<mlir::Value(StringRef)> field, table;
};

/// Typed topology state; relation and tuple parameters use semantic metadata keys.
struct TermNativeTopology {
  TermNativePotential physical;
  llvm::function_ref<mlir::Value(StringRef)> relation;
  llvm::function_ref<mlir::Value(StringRef, StringRef)> tupleField;
  llvm::function_ref<void(llvm::StringMap<mlir::Value> &)> bindLambdas;
  llvm::function_ref<void(StringRef, llvm::StringMap<mlir::Value> &)> bindObserved;
  llvm::function_ref<mlir::Value(unsigned)> gradientSeed, gradientConstant;
  Truncation pairTruncation;
  bool lambdaArguments, shiftsAtCutoff;
};

/// Borrowed physical values and semantic particle-field bindings at one evaluation.
/// This view is consumed synchronously and never retained by an adapter.
struct TermNativeEvaluation {
  mlir::Value positions, cell, forces, energy, virial, scale;
  llvm::function_ref<mlir::Value(StringRef)> field;
};
struct TermNativeResults {
  mlir::Value forces, energy, virial;
  llvm::SmallVector<mlir::Value> storedTerms;
};

/// Specialized preparation borrows no mutable program collections.
class TermPreparation {
public:
  virtual ~TermPreparation() = default;
  virtual llvm::Error validate(const TermInputs &) const = 0;
  virtual llvm::Error prepare(const TermInputs &, mlir::OpBuilder &) const = 0;
};
class TermHostEvaluation {
public:
  virtual ~TermHostEvaluation() = default;
  virtual bool applies(const TermInputs &) const = 0;
  virtual TermNativeResults build(const TermInputs &, mlir::OpBuilder &,
                                  const TermNativeEvaluation &) const = 0;

  virtual void buildDeclarations(const TermInputs &, mlir::OpBuilder &) const = 0;

};
/// Narrow roles for operations that are not shared by all term adapters.
class TermPostEvaluation {
public:
  virtual ~TermPostEvaluation() = default;
  virtual bool hasForceCorrection(const TermInputs &) const = 0;
  virtual TermNativeResults build(const TermInputs &, mlir::OpBuilder &,
                                  const TermNativeEvaluation &) const = 0;

  virtual unsigned getStoredTermCount(const TermInputs &) const = 0;

};
struct TermNativeOutput {
  const PotentialStateIR &physical;
  mlir::Value step;
  llvm::function_ref<mlir::Value(StringRef)> relation;
  llvm::function_ref<mlir::Value(StringRef, StringRef)> tupleField;
};
class TermCoordinateOutput {
public:
  virtual ~TermCoordinateOutput() = default;
  virtual bool hasCoordinateOutput(const TermInputs &) const = 0;
  virtual void buildAuxiliaryPotentials(const TermInputs &, mlir::OpBuilder &) const = 0;

  virtual void buildOutput(const TermInputs &, mlir::OpBuilder &, const TermNativeOutput &) const = 0;
};
class TermScalarParameters {
public:
  virtual ~TermScalarParameters() = default;
  virtual llvm::Error collectBareConstants(TermParameters &) const = 0;
  virtual void collectObservedConstants(TermParameters &) const = 0;
  virtual void appendInitialConstants(
      const TermInputs &,
      llvm::function_ref<void(std::string, double)>) const = 0;
};
class TermNonbonded;
struct TermNonbondedData {
  decltype(Program::coulombConstantEnergy) &coulombConstantEnergy;
  decltype(Program::coulombConstantVirial) &coulombConstantVirial;
  decltype(Program::coulombSelfEnergy) &coulombSelfEnergy;
  decltype(Program::dispersionEnergy) &dispersionEnergy;
  decltype(Program::dispersionFixedEnergy) &dispersionFixedEnergy;
  decltype(Program::dispersionVirial) &dispersionVirial;
  decltype(Program::fields) &fields;
  decltype(Program::gradientChargeFixed) &gradientChargeFixed;
  decltype(Program::gradientChargeVolume) &gradientChargeVolume;
  decltype(Program::gradientFields) &gradientFields;
  decltype(Program::gradientOutcomes) &gradientOutcomes;
  decltype(Program::gradientShiftEnergy) &gradientShiftEnergy;
  decltype(Program::gradientShiftFixedEnergy) &gradientShiftFixedEnergy;
  decltype(Program::lambdaFixedDerivatives) &lambdaFixedDerivatives;
  decltype(Program::lambdaVolumeDerivatives) &lambdaVolumeDerivatives;
  decltype(Program::ljpme) &ljpme;
  decltype(Program::ljpmeBeta) &ljpmeBeta;
  decltype(Program::ljpmeGrid) &ljpmeGrid;
  decltype(Program::ljpmeSelfEnergy) &ljpmeSelfEnergy;
  decltype(Program::observableFixedConstants) &observableFixedConstants;
  decltype(Program::observableVolumeConstants) &observableVolumeConstants;
  decltype(Program::pme) &pme;
  decltype(Program::pmeAliasing) &pmeAliasing;
  decltype(Program::pmeBackgroundFactor) &pmeBackgroundFactor;
  decltype(Program::pmeBandwidth) &pmeBandwidth;
  decltype(Program::pmeBeta) &pmeBeta;
  decltype(Program::pmeFarSeries) &pmeFarSeries;
  decltype(Program::pmeGrid) &pmeGrid;
  decltype(Program::pmeOrder) &pmeOrder;
  decltype(Program::pmeP3M) &pmeP3M;
  decltype(Program::pmeProlate) &pmeProlate;
  decltype(Program::pmeSelfFactor) &pmeSelfFactor;
  decltype(Program::pmeWindowBandwidth) &pmeWindowBandwidth;
  decltype(Program::reactionField) &reactionField;
  decltype(Program::stateFixedEnergies) &stateFixedEnergies;
  decltype(Program::stateVolumeEnergies) &stateVolumeEnergies;
  decltype(Program::tables) &tables;
  decltype(Program::tunableGradient) &tunableGradient;
  bool &gradientCharges;
  const DependentTerms *dependent;
  bool lambdaArguments, shiftsAtCutoff;
};
double getTopologyShiftFactor(const TermInputs &);
struct TermRoles {
  const TermPreparation *preparation = nullptr;
  const TermHostEvaluation *hostEvaluation = nullptr;
  const TermPostEvaluation *postEvaluation = nullptr;
  const TermCoordinateOutput *coordinateOutput = nullptr;
  const TermScalarParameters *scalarParameters = nullptr;
  const TermNonbonded *nonbonded = nullptr;
};

/// Typed scalar operands: sigma, epsilon, soft-core distance, physical distance.
mlir::Value buildLennardJonesIR(mlir::OpBuilder &, mlir::ValueRange,
                                double cutoff, double from, Truncation);

/// One independently evaluated/logged potential, in stable column order.
struct TermDefinition {
  unsigned rank;
  size_t ordinal;
  std::string name;
  unsigned terms;
  int tupleTerm = -1, pairTerm = -1, externalTerm = -1;
};

/// Hooks for the term kinds already moved out of the central builder
/// (D359). The explicit list fixes their emission order.
class TermBuilder {
public:
  struct PotentialSelection {
    unsigned terms = 0;
    int tupleTerm = -1, pairTerm = -1, externalTerm = -1;
    bool alchemicalOnly = false;
  };
  struct ObservedSelection {
    unsigned kind = 0;
    int index = -1;
    PotentialSelection potential;
  };
  virtual ~TermBuilder() = default;
  const TermRoles &getRoles() const { return roles; }
  int getTupleRank() const { return tupleRank; }
  StringRef getPairExclusion() const { return pairExclusion; }
  virtual unsigned getPotentialMask(const TermInputs &,
                                    const DependentTerms *dependent,
                                    bool alchemicalOnly) const {
    return dependent ? 0 : (alchemicalOnly ? alchemicalMask : potentialMask);
  }
  virtual int parameterRank() const { return 0; }
  virtual int potentialRank() const { return 0; }
  virtual int topologyRank() const { return 0; }
  virtual void collectTupleSets(TermParameters &) const {}
  virtual void refillArrays(TermParameters &) const {}
  virtual llvm::Expected<std::optional<Program::GradientOutcome>>
  collectTunableOutcome(TermParameters &, unsigned) const {
    return std::optional<Program::GradientOutcome>();
  }
  virtual llvm::Error collectBareParameters(TermParameters &) const {
    return llvm::Error::success();
  }
  virtual void collectTopologyTables(TermParameters &) const {}
  virtual llvm::Error collectDispersionConstants(TermParameters &) const {
    return llvm::Error::success();
  }
  virtual void collectGradientShiftConstants(TermParameters &) const {}
  virtual bool rejectsTunables(const TermInputs &) const { return false; }
  virtual bool needsPairNeighborhood(const TermInputs &, const PotentialSelection &) const {
    return false;
  }
  virtual mlir::Value buildBareContributions(const TermInputs &, mlir::OpBuilder &,
      const TermNativePotential &, mlir::Value total) const { return total; }
  virtual void collectParameters(TermParameters &) const {}
  virtual llvm::Error buildTopologyPotential(const TermInputs &, mlir::OpBuilder &,
      const TermNativeTopology &, const PotentialSelection &,
      llvm::function_ref<void(mlir::Value)>) const { return llvm::Error::success(); }
  virtual void appendTermDefinitions(const TermInputs &,
                                     std::vector<TermDefinition> &) const {}
  virtual ObservedSelection findObservedTerm(const TermInputs &,
                                             StringRef) const {
    return {};
  }
  virtual std::optional<double> getObservedValue(const TermInputs &, unsigned,
                                                int, StringRef) const {
    return std::nullopt;
  }
protected:
  unsigned potentialMask = 0, alchemicalMask = 0;
  int tupleRank = 0;
  StringRef pairExclusion;
  TermRoles roles;

};

/// A narrow context for primary/excluded/reciprocal preparation and emission.
class TermNonbonded {
public:
  virtual ~TermNonbonded() = default;
  virtual llvm::Error collect(TermParameters &, TermNonbondedData,
                              llvm::ArrayRef<const TermBuilder *>) const = 0;
  virtual void collectGradientFields(TermParameters &,
                                     TermNonbondedData) const = 0;
  virtual llvm::Error
  collectGradients(TermParameters &, TermNonbondedData,
                   llvm::ArrayRef<const TermBuilder *>) const = 0;
};

using TermBuilderFactory = std::unique_ptr<TermBuilder> (*)();
llvm::ArrayRef<TermBuilderFactory> registerTermBuilders();
std::unique_ptr<TermBuilder> createExternalTermBuilder();
std::unique_ptr<TermBuilder> createPullTermBuilder();
std::unique_ptr<TermBuilder> createRestraintTermBuilder();
std::unique_ptr<TermBuilder> createBornTermBuilder();
std::unique_ptr<TermBuilder> createTupleTermBuilder();
std::unique_ptr<TermBuilder> createPairTermBuilder();
std::unique_ptr<TermBuilder> createTripletTermBuilder();
std::unique_ptr<TermBuilder> createModelTermBuilder();
std::unique_ptr<TermBuilder> createNonbondedTermBuilder();
std::unique_ptr<TermBuilder> createBondTermBuilder();
std::unique_ptr<TermBuilder> createCMapTermBuilder();
std::unique_ptr<TermBuilder> createPairs14TermBuilder();
std::unique_ptr<TermBuilder> createImproperTermBuilder();
std::unique_ptr<TermBuilder> createDihedralTermBuilder();
std::unique_ptr<TermBuilder> createUreyBradleyTermBuilder();
std::unique_ptr<TermBuilder> createAngleTermBuilder();
llvm::Expected<PairTermTail> getCustomPairTail(const TermInputs &, unsigned,
                                             const llvm::StringMap<double> &);
double getCustomPairTails(const TermInputs &, const llvm::StringMap<double> &, bool);
double getCustomPairTailDerivative(const TermInputs &, unsigned, StringRef,
                                   double, bool, bool);
std::vector<const TermBuilder *> termBuildersByParameterRank(
    llvm::ArrayRef<const TermBuilder *>);
std::vector<const TermBuilder *> termBuildersByPotentialRank(
    llvm::ArrayRef<const TermBuilder *>);
std::vector<const TermBuilder *>
    termBuildersByTopologyRank(llvm::ArrayRef<const TermBuilder *>);

class Builder : private TermParameters {
public:
  Builder(const Control &control, const System &system, Program &program)
      : control(control), system(system), program(program) {
    if (control.dependentTerms && system.topology)
      dependent = getDependentTerms(control, system);
    for (auto factory : mdir::driver::builder::registerTermBuilders()) {
      ownedTermBuilders.push_back(factory());
      termBuilders.push_back(ownedTermBuilders.back().get());
    }
  }

  llvm::Error build();
  llvm::Error finishIR() { return serializeNativeProgram(program, termIR); }

  /// The arrays of another build of the same model at other values of its
  /// tunables, which this build takes in place of making them from the
  /// topology (D256); none for a build that makes its own.
  const BuildKept *kept = nullptr;
  /// Where a build that makes only its arrays leaves them, and stops.
  BuildKept *capture = nullptr;

private:
  const System &getSystem() const override { return system; }
  const Control &getControl() const override { return control; }
  Element getStateElement() const override { return program.state; }
  Element getForceElement() const override { return program.force; }
  llvm::ArrayRef<const TermBuilder *> getTermBuilders() const { return termBuilders; }
  std::vector<const TermBuilder *> termBuildersByParameterRank() const {
    return mdir::driver::builder::termBuildersByParameterRank(getTermBuilders());
  }
  std::vector<const TermBuilder *> termBuildersByPotentialRank() const {
    return mdir::driver::builder::termBuildersByPotentialRank(getTermBuilders());
  }
  bool hasHostEvaluation() const {
    return llvm::any_of(getTermBuilders(), [&](const TermBuilder *builder) {
      const auto *role = builder->getRoles().hostEvaluation;
      return role && role->applies(*this);
    });
  }
  const PairTermPlan &getPairPlan() const override { return pairPlan; }
  PairTermPlan &getMutablePairPlan() override { return pairPlan; }
  llvm::ArrayRef<Parameter> getTypeParameters() const override { return parameters; }
  std::vector<Parameter> &getMutableTypeParameters() override { return parameters; }
  llvm::ArrayRef<Program::Table> getParameterTables() const override {
    return program.tables;
  }
  bool includesPair(unsigned index) const override {
    return !dependent || dependent->hasPair(index);
  }
  std::vector<Program::GradientSlot> &getGradientSlots() override {
    return program.gradientSlots;
  }
  std::vector<unsigned> &getGradientColumns() override {
    return gradientColumns;
  }
  std::vector<double> &getObservableVolumeConstants() override {
    return program.observableVolumeConstants;
  }
  std::vector<double> &getObservableFixedConstants() override {
    return program.observableFixedConstants;
  }
  void setBareDispersion(double energy, double virial) override {
    program.dispersionEnergy = energy;
    program.dispersionVirial = virial;
  }
  void addDispersion(double energy, double fixed, double virial) override {
    program.dispersionEnergy += energy;
    program.dispersionFixedEnergy += fixed;
    program.dispersionVirial += virial;
  }
  void addGradientShift(double scaled, double fixed) override {
    program.gradientShiftEnergy += scaled;
    program.gradientShiftFixedEnergy += fixed;
  }
  std::vector<Program::Field> &getFields() override { return program.fields; }
  std::vector<Program::Table> &getTables() override { return program.tables; }
  std::vector<Program::TupleSet> &getTupleSets() override {
    return program.tupleSets;
  }

  llvm::ArrayRef<BuildKept::TupleGradient> getTupleGradients() const override {
    return tupleGradients;
  }
  llvm::ArrayRef<Program::Field> getParameterFields() const override { return program.fields; }
  bool hasGradientArguments() const override { return gradientArguments; }
  bool hasGradientCharges() const override { return gradientCharges; }
  bool hasTunableGradient() const override { return program.tunableGradient; }
  bool includesTuple(unsigned index) const override {
    return !dependent || dependent->hasTuple(index);
  }
  std::vector<Program::GradientField> &getGradientFields() override {
    return program.gradientFields;
  }
  std::vector<BuildKept::TupleGradient> &getMutableTupleGradients() override {
    return tupleGradients;
  }

  /// The stage of `collectTopology` that makes the arrays of the program
  /// from the topology: every field, table, and tuple set, each one number
  /// or one row for each particle, type, or tuple.
  void collectTopologyArrays();
  std::vector<TermDefinition> getTermDefinitions() const;
  TermNonbondedData getNonbondedData() {
    return {program.coulombConstantEnergy,
            program.coulombConstantVirial,
            program.coulombSelfEnergy,
            program.dispersionEnergy,
            program.dispersionFixedEnergy,
            program.dispersionVirial,
            program.fields,
            program.gradientChargeFixed,
            program.gradientChargeVolume,
            program.gradientFields,
            program.gradientOutcomes,
            program.gradientShiftEnergy,
            program.gradientShiftFixedEnergy,
            program.lambdaFixedDerivatives,
            program.lambdaVolumeDerivatives,
            program.ljpme,
            program.ljpmeBeta,
            program.ljpmeGrid,
            program.ljpmeSelfEnergy,
            program.observableFixedConstants,
            program.observableVolumeConstants,
            program.pme,
            program.pmeAliasing,
            program.pmeBackgroundFactor,
            program.pmeBandwidth,
            program.pmeBeta,
            program.pmeFarSeries,
            program.pmeGrid,
            program.pmeOrder,
            program.pmeP3M,
            program.pmeProlate,
            program.pmeSelfFactor,
            program.pmeWindowBandwidth,
            program.reactionField,
            program.stateFixedEnergies,
            program.stateVolumeEnergies,
            program.tables,
            program.tunableGradient,
            gradientCharges,
            dependent ? &*dependent : nullptr,
            lambdaArguments,
            shiftsAtCutoff};
  }
  /// In its place, for a build that was given the arrays of another: takes
  /// them, fills those that hold the values of the tunables from the model,
  /// and checks that none is left without its numbers.
  llvm::Error adoptArrays();
  /// With Control::dependentTerms, the terms that the program keeps: every
  /// potential has only these, and the host adds to the energy only what
  /// belongs to them (D243).
  std::optional<DependentTerms> dependent;
  llvm::Error collectParameters();
  llvm::Error emitPotential();
  /// For a run from a topology: the fields, the tables, and the tuple sets
  /// of the program, and the potential of the topology.
  llvm::Error collectTopology();

  /// Emits the potential `name` of the terms `terms` of the topology; of
  /// the terms given by expressions over tuples, pairs, or positions, only
  /// the one of index `tupleTerm`, `pairTerm`, or `externalTerm` if it is
  /// not -1.
  llvm::Error emitTopologyPotential(StringRef name, unsigned terms,
                             int tupleTerm = -1, int pairTerm = -1,
                             int externalTerm = -1);
  /// Whether the force on each virtual site is spread by the derivative of
  /// its placement (`MDIR_SITE_SPREAD=derived`) or by the transpose written
  /// out in `emitSpreadSites` (`written`), D298.
  bool spreadsSitesDerived() const;
  /// For each set of virtual sites, with the derived spreading, the
  /// potential `@spread_<set>`: Σ_t f_t · p(t), the force on each site
  /// against the place of the site from its owner, whose forces are the
  /// transpose of the derivative of the placement.
  void emitSitePotentials();
  /// The number of arguments of a potential, before those of its own.
  unsigned getNumPotentialArguments() const;

  llvm::Error emitPrograms();

  /// Collects the values of Program::startValues, which the entry takes
  /// as arguments (D227).
  void collectStartValues();
  llvm::Error emitEntry();
  /// Emits `@descend`, one step of steepest descent that moves no particle
  /// farther than `%h`, and the loops of a minimization in the entry.
  void emitDescend();
  /// Sets `levels`, the loops of the schedule of a run of dynamics.
  void setSchedule();

  /// Whether the minimization of `mdir run` can end at a row of the
  /// energies before its last step: at the first whose largest force is
  /// below a tolerance (D219), or whose step has shrunk below
  /// the floor (D274). A Python simulation checks both
  /// between its parts, on the host, and its program is that without them.
  bool stopsMinimization() const {
    return control.minimize && !control.segments;
  }
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
  struct DiagonalOutputs {
    std::string vector, xx, yy, zz;
  };

  bool hasForceCorrections() const {
    return llvm::any_of(getTermBuilders(), [&](const TermBuilder *term) {
      const auto *role = term->getRoles().postEvaluation;
      return role && role->hasForceCorrection(*this);
    });
  }
  /// Whether the reference positions of the restraints follow the cell,
  /// which a barostat changes: they are those of the file times
  /// `scaleName`, the edge of the cell over that of the file.
  bool scalesReference() const { return hasForceCorrections() && changesCell(); }
  /// Whether the programs of the steps take the number of the step: for the
  /// noise of Langevin dynamics, for the time of terms that take it, and
  /// for the stop of a run whose constraints failed
  /// (D278).
  bool needsStepNumber() const {
    return control.isLangevin() || control.isBrownian() || control.usesTime ||
           (hasConstraints() && !control.minimize);
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
  /// The potential `@tunable` (D230) takes the tunable
  /// constants of the pair terms as its last arguments, `%tg_c<column>`, in
  /// place of the values of the table `tunable_constants`, so that the
  /// differentiation of parameters gives the derivatives in them (D161).
  bool gradientArguments = false;
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
  /// The potentials `@observe<k>`, each a term of `[output] observe`: the
  /// kind of the term, its index among the terms of its kind, and the
  /// columns of the term, -1 for its energy or the place of the constant
  /// among `observedConstants` of the potential, with the column of each.
  struct ObservedTerm {
    unsigned kind = 0;
    int index = -1;
    TermBuilder::PotentialSelection potential;
    std::string name;
    std::vector<std::string> constants;
    std::vector<std::pair<int, size_t>> columns;
  };
  std::vector<ObservedTerm> getObservedTerms() const;
  /// The value of the constant `name` of the observed term `term`.
  double getObservedValue(const ObservedTerm &term, StringRef name) const {
    for (const TermBuilder *builder : getTermBuilders())
      if (auto value =
              builder->getObservedValue(*this, term.kind, term.index, name))
        return *value;
    llvm_unreachable("Control checks the observed constants");
  }
  /// Emits `@observe<k>` for each term of `[output] observe`.
  llvm::Error emitObservedPotentials();
  /// Its part that does not depend on the volume once it multiplies a
  /// quantity proportional to 1 / V: −V / (N (4π/3) r_c³) (#224).
  double getShiftFixedFactor() const override {
    return getTopologyShiftFactor(*this) - 1.0;
  }

  /// Whether the thermostat rescales the velocities at the end of a period,
  /// rather than acting in every step.
  bool rescalesVelocities() const {
    return control.thermostat && !control.isLangevin();
  }
  /// Whether the topology has virtual sites.
  bool hasSites() const override { return !getSiteSets().empty(); }

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
  bool hasConstraints() const override {
    return hasSettles() || !getShakeSets().empty();
  }
  /// Whether the steps that the log reads measure the kinetic energy of
  /// the half steps around their end from the velocities
  /// (D203): with constraints, which the forces alone do
  /// not give it for. Under Langevin dynamics, by g-BAOAB, the half steps
  /// are the drifts of its two A around the end of the step
  /// (D265). Brownian dynamics has no momenta.
  bool measuresHalfSteps() const {
    return mdir::driver::builder::measuresHalfSteps(hasConstraints(), control.isBrownian());
  }
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
  /// A bond of a group of constraints: its members, by their places in the
  /// tuple, and the argument of the kernel that holds its length.
  using ConstrainedBond = ConstraintBond;
  /// Whether the step tests the constraints in their own kernels (the
  /// default) or in loops of their own (`MDIR_CONSTRAINT_TEST=separate`,
  /// for comparison, D278).
  bool testsConstraintsApart() const {
    const char *value = std::getenv("MDIR_CONSTRAINT_TEST");
    return value && StringRef(value) == "separate";
  }
  /// Emits the coupling at the end of the step `step` of the positions
  /// `positions`, the velocities `velocities`, and the forces `forces`,
  /// whose potential energy is `energy` and the trace of whose virial is
  /// `trace`: the removal of the motion of the center of mass, the
  /// thermostat, and the barostat, which with the exact work evaluates the
  /// scaled positions and returns their forces (D77).
  /// The groups that the constraints keep rigid: the waters of SETTLE and
  /// the sets of SHAKE.
  std::vector<const Program::TupleSet *> getRigidGroups() const;

  bool isLeapfrog() const {
    return control.integrator == Integrator::Leapfrog;
  }
  bool isRestart() const { return !control.restartInput.empty(); }
  /// Whether the entry begins a run or continues the last segment as its
  /// argument %first_call says (D196, D211): a
  /// program of segments is one program for the first segment and the
  /// later ones, whose work at the start is a branch on that argument.
  bool branchesStart() const { return control.segments; }
  static int64_t getPartCountsSize() { return 9; }
  /// Whether a run from a checkpoint evaluates the forces of its first step
  /// (D172).
  bool recomputes() const { return isRestart() && control.restartRecomputes; }

  TermIRPlan termIR;
  std::vector<std::unique_ptr<TermBuilder>> ownedTermBuilders;
  std::vector<const TermBuilder *> termBuilders;

  const Control &control;
  const System &system;
  Program &program;

  PairTermPlan pairPlan;
  std::vector<Parameter> parameters;

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

#endif // MDIR_LIB_DRIVER_BUILDERIMPL_H
