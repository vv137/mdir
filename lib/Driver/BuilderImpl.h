// The builder of the program of a run, shared by the files of its parts
// (docs/driver.md, Section 2.12): the class Builder and the helpers
// that more than one part takes. Private to lib/Driver.

#ifndef MDIR_LIB_DRIVER_BUILDERIMPL_H
#define MDIR_LIB_DRIVER_BUILDERIMPL_H

#include "mdir/Driver/Builder.h"
#include "TermIR.h"
#include "ConstraintIR.h"
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

/// Emission adds the output stream and explicitly named shared operations.
class TermEmission : public TermParameters {
public:
  virtual llvm::raw_ostream &getOutput() = 0;
  virtual void emitIRFragment(StringRef indent, llvm::ArrayRef<StringRef> results,
      llvm::ArrayRef<StringRef> resultTypes, llvm::ArrayRef<StringRef> operands,
      llvm::ArrayRef<StringRef> operandTypes, TermIRRecipe recipe) = 0;
  virtual Truncation getPairTruncation() const = 0;
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

/// Borrowed names at one force-correction site, including its reference scale.
struct TermCorrection {
  StringRef scale, indent, x, prefix, f, fResult, u, uResult, w, wResult;
};

/// Inputs and output names of a hosted force/energy/virial evaluation.
struct TermEvaluation {
  StringRef indent, x, cell, prefix, f, fResult, u, uResult, w, tag;
};

/// Specialized preparation borrows no mutable program collections.
class TermPreparation {
public:
  virtual ~TermPreparation() = default;
  virtual llvm::Error validate(const TermInputs &) const = 0;
  virtual llvm::Error prepare(const TermInputs &, mlir::MLIRContext &,
                              TermIRWriter) const = 0;
};
class TermHostEvaluation {
public:
  virtual ~TermHostEvaluation() = default;
  virtual bool applies(const TermInputs &) const = 0;
  virtual void emitDeclarations(const TermInputs &, TermIRWriter) const = 0;
  virtual std::string emit(const TermInputs &, TermIRWriter,
                           const TermEvaluation &) const = 0;
};
/// Narrow roles for operations that are not shared by all term adapters.
class TermPostEvaluation {
public:
  virtual ~TermPostEvaluation() = default;
  virtual bool hasForceCorrection(const TermInputs &) const = 0;
  virtual void emitForceCorrection(TermEmission &,
                                   const TermCorrection &) const = 0;
  virtual unsigned getStoredTermCount(const TermInputs &) const = 0;
  virtual void emitStoredTerms(TermEmission &, StringRef, int &) const = 0;
};
class TermCoordinateOutput {
public:
  virtual ~TermCoordinateOutput() = default;
  virtual bool hasCoordinateOutput(const TermInputs &) const = 0;
  virtual void emitAuxiliaryPotentials(TermEmission &) const = 0;
  virtual void emitOutput(TermEmission &, const TermOutput &) const = 0;
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

/// Adds either a conventional named contribution or an explicit SSA result.
struct TermPotentialAddition {
  llvm::function_ref<void(StringRef)> named;
  llvm::function_ref<void(StringRef, StringRef)> result;
  void operator()(StringRef name) const { named(name); }
  void append(StringRef value, StringRef sum) const { result(value, sum); }
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
  virtual void emitBareContributions(TermEmission &, std::string &) const {}
  virtual void collectParameters(TermParameters &) const {}
  virtual void emitTopologyPotential(
      TermEmission &, const PotentialSelection &,
      const TermPotentialAddition &) const {}
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
  virtual void emit(TermEmission &, TermNonbondedData,
                    const TermBuilder::PotentialSelection &,
                    const TermPotentialAddition &) const = 0;
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

class Builder : private TermEmission {
public:
  Builder(const Control &control, const System &system, Program &program)
      : control(control), system(system), program(program),
        os(program.module) {
    if (control.dependentTerms && system.topology)
      dependent = getDependentTerms(control, system);
    for (auto factory : mdir::driver::builder::registerTermBuilders()) {
      ownedTermBuilders.push_back(factory());
      termBuilders.push_back(ownedTermBuilders.back().get());
    }
  }

  llvm::Error build();
  llvm::Error finishIR() { return finishTermIRProgram(program, termIR); }

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
  llvm::raw_ostream &getOutput() override { return os; }
  void emitIRFragment(StringRef indent, llvm::ArrayRef<StringRef> results,
      llvm::ArrayRef<StringRef> resultTypes, llvm::ArrayRef<StringRef> operands,
      llvm::ArrayRef<StringRef> operandTypes, TermIRRecipe recipe) override {
    termIR.emit(os, indent, results, resultTypes, operands, operandTypes,
                 std::move(recipe));
  }
  llvm::ArrayRef<BuildKept::TupleGradient> getTupleGradients() const override {
    return tupleGradients;
  }
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
  void emitTopologyPotential(StringRef name, unsigned terms,
                             int tupleTerm = -1, int pairTerm = -1,
                             int externalTerm = -1) override;
  /// Emits the coordinates of the term `term` over centers, at the
  /// positions `x` in the cell `cell`, with
  /// the members of its tuples named `relations` and the name of the set:
  /// `r`, `dx`, `dy`, `dz` in Å for two centers, `theta` in radians
  /// otherwise, mapped to their names, which begin with `prefix`.
  llvm::StringMap<std::string>
  emitCentroidCoordinates(const TupleTerm &term, StringRef indent,
                          StringRef x, StringRef cell, StringRef relations,
                          StringRef prefix) override;
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
  unsigned getNumPotentialArguments() const override;

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
  Truncation getPairTruncation() const override {
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
  /// The tail of the pair term `index` at the volume of the file, in
  /// kJ/mol, with the values `changes` in place of those of the run (a
  /// component `lambda_<name>` of λ, or a constant of the term), and its
  /// share of the trace of the virial; an error if the integral fails.

  /// Its part that does not depend on the volume once it multiplies a
  /// quantity proportional to 1 / V: −V / (N (4π/3) r_c³) (#224).
  double getShiftFixedFactor() const override {
    return getTopologyShiftFactor(*this) - 1.0;
  }

  /// Emits the number of the step about to be taken, from the counter on
  /// the host that each step advances, and returns the operands that a
  /// program of a step takes for it.
  std::string emitNoiseValue(StringRef indent);
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
  void emitForceCorrections(StringRef indent, StringRef x, StringRef prefix,
                      StringRef f, StringRef fResult, StringRef u = "",
                      StringRef uResult = "", StringRef w = "",
                      StringRef wResult = "") {
    for (const TermBuilder *term : getTermBuilders())
      if (const auto *role = term->getRoles().postEvaluation)
        role->emitForceCorrection(*this, {scaleName, indent, x, prefix, f,
                                          fResult, u, uResult, w, wResult});
  }
  /// Invokes the hosted-evaluation role at its existing correction site.
  std::string emitHostEvaluation(StringRef indent, StringRef x, StringRef cell,
      StringRef prefix, StringRef f, StringRef fResult, StringRef u,
      StringRef uResult, StringRef w, StringRef tag);

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
  /// off its target, `failed`, an i1 (D278); so do those of emitBondConstraints and
  /// emitSettlePositions.
  void emitShakePositions(StringRef indent, StringRef old, StringRef x,
                          const Program::TupleSet &set, StringRef result,
                          StringRef failed = "");
  /// A bond of a group of constraints: its members, by their places in the
  /// tuple, and the argument of the kernel that holds its length.
  using ConstrainedBond = ConstraintBond;
  /// Emits `change`, what bringing the bonds `bonds` of the groups of the
  /// tuple set `setName` (of `arity` members, with the fields `tuple`) back
  /// to their lengths along the bonds of `old` adds to the positions `x`,
  /// by the iterations of Newton of M-SHAKE [Krautler2001].
  void emitBondConstraints(StringRef indent, StringRef setName,
                           unsigned arity, llvm::ArrayRef<ConstrainedBond> bonds,
                           llvm::ArrayRef<std::string> tuple, StringRef old,
                           StringRef x, StringRef change,
                           StringRef failed = "");
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

  TermIRPlan termIR;
  std::vector<std::unique_ptr<TermBuilder>> ownedTermBuilders;
  std::vector<const TermBuilder *> termBuilders;

  const Control &control;
  const System &system;
  Program &program;
  llvm::raw_string_ostream os;

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

  PairTermPlan pairPlan;
  std::vector<Parameter> parameters;

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
