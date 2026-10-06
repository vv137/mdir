// Builds the program of a run.

#include "mdir/Driver/Builder.h"

#include "mdir/Driver/Expression.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cmath>

using namespace mdir::driver;
using llvm::StringRef;

static StringRef getName(Element element) {
  return element == Element::F32 ? "f32" : "f64";
}

namespace {

/// The Coulomb constant in the units of the control file, kcal Å mol⁻¹ e⁻²,
/// from CODATA 2018 (docs/conventions.md). An expression names it
/// `coulomb`.
constexpr double coulombConstant = 332.06371329919205;

/// The parameter of a pair from those of its two particles.
static double mix(Mixing mixing, double a, double b) {
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

class Builder {
public:
  Builder(const Control &control, const System &system, Program &program)
      : control(control), system(system), program(program),
        os(program.module) {}

  llvm::Error build();

private:
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
  /// Emits the potential `name` of the terms `terms` of the topology; of
  /// the terms given by expressions over tuples, pairs, or positions, only
  /// the one of index `tupleTerm`, `pairTerm`, or `externalTerm` if it is
  /// not -1.
  void emitTopologyPotential(StringRef name, unsigned terms,
                             int tupleTerm = -1, int pairTerm = -1,
                             int externalTerm = -1);
  /// Emits `%u_external_<name>`, the term `index` of the absolute positions
  /// (D148), and passes its name to `add`.
  void emitExternalTerm(size_t index, const ExternalTerm &term,
                        llvm::function_ref<void(StringRef)> add);
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
                          StringRef prefix);
  /// The coordinates of the terms over centers that `[output] pull`
  /// writes (D145, D149): the index of the term over tuples and the
  /// quantity of each column.
  std::vector<std::pair<size_t, std::string>> getPullColumns() const;
  /// Emits for each term over centers its energy as a function of its
  /// coordinates, `@pullterm<k>`, whose derivatives give its forces.
  void emitPullPotentials();
  /// The number of arguments of a potential, before those of its own.
  unsigned getNumPotentialArguments() const;
  /// Emits the evaluation of the columns at the positions `x` in the cell
  /// `cell`, the fields of `prefix`, the step `step`, and the time `time`,
  /// and their call to the writer.
  void emitPullOutput(StringRef indent, StringRef x, StringRef cell,
                      StringRef prefix, StringRef step, StringRef time);
  /// Emits generalized Born (D144), `%u_born` and `%u_surface` as `terms`
  /// asks, and passes their names to `add`.
  void emitBorn(unsigned terms, llvm::function_ref<void(StringRef)> add);
  /// κ of the salt of generalized Born in nm⁻¹, scaled by 0.73, or 0.
  double getDebyeKappa() const;
  /// β, the grid, the influence function, and the constant terms of
  /// particle mesh Ewald (docs/pme-m1.md).
  llvm::Error collectPME();
  /// The grid of particle mesh Ewald: given, or from the largest spacing in
  /// Å; and the table `name` of the factors of its influence function along
  /// each edge.
  llvm::Error collectMesh(StringRef name, const int64_t (&given)[3],
                          double spacing, int64_t order, bool optimal,
                          int64_t (&grid)[3]);
  /// β, the grid, the coefficients, and the self term of particle mesh
  /// Ewald for the dispersion (D162).
  llvm::Error collectLJPME();
  void emitPrograms();
  void emitEntry();
  /// Emits `@descend`, one step of steepest descent that moves no particle
  /// farther than `%h`, and the loops of a minimization in the entry.
  void emitDescend();
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
  void emitMinimization();
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
  /// Whether the cell is triclinic (docs/triclinic-m2.md).
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
  /// noise of Langevin dynamics, and for the time of terms that take it.
  bool needsStepNumber() const {
    return control.isLangevin() || control.isBrownian() || control.usesTime;
  }
  /// Terms whose expressions take the time `t` (D145): the potentials take
  /// it in ps as their last argument. A program of a step computes it from
  /// the number of the step, `%time`; the entry has `%time0`, that of the
  /// first step.
  std::string getTimeParameter() const {
    return control.usesTime ? ", %time: f64" : "";
  }
  std::string getTimeType() const { return control.usesTime ? ", f64" : ""; }
  std::string getTimeValue(StringRef name) const {
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
  void emitLambdaConstants(StringRef indent) {
    if (lambdaArguments)
      return;
    for (const auto &[name, values] : control.freeEnergy.lambdas)
      os << indent << "%lambda_" << name << " = arith.constant "
         << formatReal(getLambda(name)) << " : f64\n";
  }
  /// Binds the parameters `lambda_<name>` of the expressions.
  void bindLambdas(llvm::StringMap<std::string> &values) const {
    for (const auto &[name, v] : control.freeEnergy.lambdas)
      values["lambda_" + name] = "%lambda_" + name;
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
                    llvm::StringMap<std::string> &values) const {
    if (term != observedTerm)
      return;
    for (const std::string &name : observedConstants)
      values[name] = "%ob_" + name;
  }
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
    if (term.kind == ExternalTerms)
      for (const auto &[key, value] :
           system.topology->externalTerms[term.index].constants)
        if (key == name)
          return value;
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
  bool usesLambda(StringRef text) const {
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
  /// not give it for. Under Langevin dynamics the half steps hold the noise
  /// of a step not yet taken, and Brownian dynamics has no momenta, so
  /// neither measures them.
  bool measuresHalfSteps() const {
    return hasConstraints() && !control.isLangevin() && !control.isBrownian();
  }
  /// Emits `%khs`, in a program of a step, the kinetic energy of each axis
  /// at the half steps around its end, from `velocities`, those it drifted
  /// with after the constraints, and those of a drift of the next step.
  /// With `solvent`, also `%kws`, the kinetic energies of the rigid waters:
  /// that of the velocities of the time of the positions, `%v2`, and
  /// K_half.
  void emitHalfStepKinetic(StringRef velocities, bool leapfrog,
                           bool solvent = false);
  /// Whether the steps of energy report the kinetic energies of the rigid
  /// waters, from which the log gives the temperatures of the solvent and
  /// of the solute (D203).
  bool reportsSolvent() const { return measuresHalfSteps() && hasSettles(); }
  /// Emits the call that gives the host the kinetic energies of the rigid
  /// waters `sums` (emitHalfStepKinetic) before the row of the log.
  void emitWriteSolvent(StringRef indent, StringRef sums);
  /// Emits `result`, the positions `x` with the bonds of the groups of
  /// `set` brought back to their lengths along the bonds of `old`, by the
  /// iterations of SHAKE [Ryckaert1977].
  void emitShakePositions(StringRef indent, StringRef old, StringRef x,
                          const Program::TupleSet &set, StringRef result);
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
                           StringRef x, StringRef change);
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
                           StringRef change, StringRef result);
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
  /// `positions` scaled by `mu`, a scale for each axis, each rigid group
  /// with its center of mass, keeping its shape (`oneMinusMu` is 1 − mu).
  std::string emitGroupScaling(StringRef indent, StringRef positions,
                               StringRef mu, StringRef oneMinusMu,
                               StringRef cell, StringRef masses,
                               StringRef relations, StringRef tag);
  Coupled emitCoupling(StringRef indent, StringRef positions,
                       StringRef velocities, StringRef forces,
                       StringRef energy, StringRef tag, StringRef step,
                       StringRef trace,
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
  TrotterScaling emitTrotterStrain(StringRef indent, StringRef velocities,
                                   StringRef trace, StringRef groups,
                                   StringRef tag, StringRef step,
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

  /// The arguments that pass the fields of the parameters on: their
  /// declarations, their values, and their types, each after a comma.
  std::string getFieldParameters() const;
  /// The fields of the parameters, as arguments that follow others.
  /// `prefix` comes before the name of each.
  std::string getFieldValues(StringRef prefix = "%p_") const;
  std::string getFieldTypes() const;

  /// Emits the order of the particles at the positions `positions`, and
  /// the fields in that order. `from` and `to` are the ends of the names
  /// of the fields before and after.
  void emitReorder(StringRef indent, StringRef from, StringRef stateTo,
                   StringRef otherTo, bool withForces,
                   StringRef velocities);
  /// Emits the members of the tuples of the topology at the places that
  /// the numbers `ids` give the particles, named `prefix` and the name of
  /// the set.
  void emitRenumber(StringRef indent, const llvm::Twine &ids,
                    const llvm::Twine &prefix);

  bool isLeapfrog() const {
    return control.integrator == Integrator::Leapfrog;
  }
  bool isRestart() const {
    return !control.restartInput.empty() || control.continuesSegment;
  }
  /// Whether a run from a checkpoint evaluates the forces of its first step
  /// (D172).
  bool recomputes() const { return isRestart() && control.restartRecomputes; }

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
};

} // namespace

/// The constants of the reaction field at the cutoff r_c, in nm (D140): k,
/// with which the field adds f q q k r² to a pair, and c, with which the
/// potential f q q (1/r + k r² − c) is 0 at r_c [Barker1973, Tironi1995].
/// With the relative permittivity ε beyond the cutoff,
/// k = (ε − 1) / ((2ε + 1) r_c³); a conductor, ε = 0 in the control file,
/// has the limit 1 / (2 r_c³).
static std::pair<double, double> getReactionField(const Control &control) {
  double rc = control.cutoffDistance * units::length;
  double epsilon = control.reactionFieldDielectric;
  double k = epsilon == 0.0
                 ? 1.0 / (2.0 * rc * rc * rc)
                 : (epsilon - 1.0) / ((2.0 * epsilon + 1.0) * rc * rc * rc);
  return {k, 1.0 / rc + k * rc * rc};
}

/// The attribute of an md.sum_relation that truncates its energy at the
/// cutoff as `control` says; the power force switch is for the
/// Lennard-Jones of a topology only, and the reader rejects it elsewhere.
static std::string getTruncation(const Control &control) {
  double from = control.switchDistance * units::length;
  switch (control.truncation) {
  case Truncation::None:
  case Truncation::PowerForceSwitch:
  case Truncation::SquaredDistanceSwitch:
    return "";
  case Truncation::Shift:
    return " truncation(shift)";
  case Truncation::Switch:
    return " truncation(switch, from = " + formatReal(from) + ")";
  case Truncation::ForceSwitch:
    return " truncation(force_switch, from = " + formatReal(from) + ")";
  }
  llvm_unreachable("unknown truncation");
}

std::string Builder::getFieldParameters() const {
  std::string text;
  for (const Program::Field &field : program.fields)
    text += ", %p_" + field.name + ": " + getFieldType(field).str();
  for (const Program::Table &table : program.tables)
    text += ", %t_" + table.name + ": " + table.getType().str();
  for (const Program::TupleSet &set : program.tupleSets) {
    text += ", %r_" + set.name + ": !rel_" + set.name;
    for (const Program::Field &field : set.fields)
      text += ", %f_" + set.name + "_" + field.name + ": !of_" + set.name;
  }
  return text;
}

std::string Builder::getFieldValues(StringRef prefix) const {
  std::string text;
  for (const Program::Field &field : program.fields)
    text += (", " + prefix + field.name).str();
  // Tables do not change with the order of the particles; they keep their
  // names everywhere.
  for (const Program::Table &table : program.tables)
    text += ", %t_" + table.name;
  // Neither do the tuples of a topology: the program does not put the
  // particles in a new order when it has them.
  // The members of the tuples follow the order of the particles: their
  // names are those of the fields with `%r` for `%p`.
  std::string relations = ("%r" + prefix.drop_front(2)).str();
  for (const Program::TupleSet &set : program.tupleSets) {
    text += ", " + relations + set.name;
    for (const Program::Field &field : set.fields)
      text += ", %f_" + set.name + "_" + field.name;
  }
  return text;
}

void Builder::emitReorder(StringRef indent, StringRef from,
                          StringRef stateTo, StringRef otherTo,
                          bool withForces, StringRef velocities) {
  StringRef order = "!mdrt.permutation<@atoms>";
  // A particle goes where the anchor of its constraint group is (D110):
  // the positions that the order takes are those of the anchors, from the
  // pairs of the anchors in the order the particles are in now.
  std::string ordered = ("%x" + from).str();
  if (llvm::any_of(program.tupleSets, [](const Program::TupleSet &set) {
        return set.name == "anchors";
      })) {
    ordered = ("%xanchor" + stateTo).str();
    os << indent << "%ranchor" << stateTo << " = md_exec.renumber %ro_anchors, %id"
       << from << " : !rel_anchors, !ids\n"
       << indent << ordered << " = md.gather_tuples %ranchor" << stateTo << ", %x"
       << from << ", " << cellName << " coordinates(displacement(1, 0)) gather(%x" << from
       << " : !vec) {\n"
       << indent << "^bb0(%xanchor_d: vector<3xf64>, %xanchor_0: vector<3xf64>, "
       << "%xanchor_1: vector<3xf64>):\n"
       << indent << "  %xanchor_none = arith.constant dense<0.0> : vector<3xf64>\n"
       << indent << "  md.yield %xanchor_1, %xanchor_none : vector<3xf64>, vector<3xf64>\n"
       << indent << "} : !rel_anchors, !vec -> !vec\n";
  }
  os << indent << "%order" << stateTo << " = md_exec.spatial_order "
     << ordered << ", " << cellName << ", %id" << from << " width("
     << formatReal(program.orderWidth) << ")\n"
     << indent << "    : !vec, !ids -> " << order << "\n";
  auto permute = [&](const llvm::Twine &result, const llvm::Twine &field,
                     StringRef type) {
    os << indent << result << " = md_exec.permute " << field << ", %order"
       << stateTo << "\n"
       << indent << "    : " << type << ", " << order << " -> " << type
       << "\n";
  };
  permute("%x" + stateTo, "%x" + from, "!vec");
  permute(velocities, "%v" + from, "!vec");
  if (withForces)
    permute("%f" + stateTo, "%f" + from, "!vec");
  permute("%m" + otherTo, "%m" + from, "!real");
  for (const Program::Field &field : program.fields)
    permute("%p" + otherTo + "_" + field.name,
            "%p" + from + "_" + field.name, getFieldType(field));
  permute("%id" + otherTo, "%id" + from, "!ids");
  emitRenumber(indent, "%id" + otherTo, "%r" + otherTo + "_");
}

void Builder::emitRenumber(StringRef indent, const llvm::Twine &ids,
                           const llvm::Twine &prefix) {
  // From the members in the order of the files.
  for (const Program::TupleSet &set : program.tupleSets)
    os << indent << prefix << set.name << " = md_exec.renumber %ro_"
       << set.name << ", " << ids << " : !rel_" << set.name << ", !ids\n";
}

std::string Builder::getFieldTypes() const {
  std::string text;
  for (const Program::Field &field : program.fields)
    text += ", " + getFieldType(field).str();
  for (const Program::Table &table : program.tables)
    text += ", " + table.getType().str();
  for (const Program::TupleSet &set : program.tupleSets) {
    text += ", !rel_" + set.name;
    for (size_t i = 0, e = set.fields.size(); i != e; ++i)
      text += ", !of_" + set.name;
  }
  return text;
}

/// The overrides of the pair term `term`: those that name it, and those
/// that name no term where there is one term.
static std::vector<const PairOverride *>
getOverrides(const Control &control, const PairTerm &term) {
  std::vector<const PairOverride *> found;
  for (const PairOverride &entry : control.overrides)
    if (entry.term == term.name ||
        (entry.term.empty() && control.pairs.size() == 1))
      found.push_back(&entry);
  return found;
}

/// The index of the type named `name`, or -1.
static int findType(const Control &control, StringRef name) {
  for (auto [index, type] : llvm::enumerate(control.types))
    if (type.name == name)
      return static_cast<int>(index);
  return -1;
}

static llvm::Error makeError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

llvm::Error Builder::collectParameters() {
  for (const PairOverride &entry : control.overrides) {
    if (entry.term.empty() && control.pairs.size() != 1)
      return makeError("[[energy.nbfix]] needs 'pair' to name the pair term "
                       "when there is more than one");
    if (!entry.term.empty() &&
        llvm::none_of(control.pairs, [&](const PairTerm &term) {
          return term.name == entry.term;
        }))
      return makeError("[[energy.nbfix]] names the pair term '" + entry.term +
                       "', which does not exist");
    for (const std::string &name : {entry.first, entry.second})
      if (findType(control, name) < 0)
        return makeError("[[energy.nbfix]] names the type '" + name +
                         "', which does not exist");
  }

  for (const PairTerm &term : control.pairs) {
    auto expression = Expression::parse(term.expression, control.functions);
    if (!expression)
      return expression.takeError();
    std::vector<const PairOverride *> overrides = getOverrides(control, term);
    llvm::StringMap<unsigned> tables;

    for (const std::string &name : expression->getNames()) {
      if (name == "r" || name == "coulomb")
        continue;
      if (llvm::any_of(term.constants,
                       [&](auto &constant) { return constant.first == name; }))
        continue;

      // A parameter of the types. Every type must have it.
      std::vector<double> values;
      for (const ParticleType &type : control.types) {
        auto found = llvm::find_if(type.parameters, [&](auto &entry) {
          return entry.first == name;
        });
        if (found == type.parameters.end())
          return makeError("the expression of '" + term.name + "' uses '" +
                           name + "', which is neither 'r', nor 'coulomb', "
                           "nor a number of the term, nor a parameter of the "
                           "type '" + type.name + "'");
        values.push_back(found->second);
      }
      bool isUniform = llvm::all_of(
          values, [&](double value) { return value == values.front(); });
      if (!isUniform && !term.mixing.count(name))
        return makeError("'" + name + "' differs between the types, but the "
                         "term '" + term.name + "' has no rule of 'mixing' "
                         "for it");

      // A term with overrides looks every parameter up in a table of the
      // pairs of types (D57).
      if (!overrides.empty()) {
        Program::Table table;
        table.name = term.name + "_" + name;
        table.count = control.types.size();
        table.values.resize(table.count * table.count);
        for (unsigned a = 0; a != table.count; ++a)
          for (unsigned b = 0; b != table.count; ++b)
            table.values[a * table.count + b] =
                isUniform ? values.front()
                          : mix(term.mixing.lookup(name), values[a], values[b]);
        for (const PairOverride *entry : overrides)
          for (auto &[parameter, value] : entry->parameters)
            if (parameter == name) {
              unsigned a = findType(control, entry->first);
              unsigned b = findType(control, entry->second);
              table.values[a * table.count + b] = value;
              table.values[b * table.count + a] = value;
            }
        tables[name] = program.tables.size();
        program.tables.push_back(std::move(table));
        continue;
      }

      if (llvm::any_of(parameters,
                       [&](Parameter &known) { return known.name == name; }))
        continue;
      Parameter parameter;
      parameter.name = name;
      parameter.value = values.front();
      parameter.isUniform = isUniform;
      if (!parameter.isUniform) {
        parameter.field = program.fields.size();
        Program::Field field;
        field.name = name;
        for (unsigned type : system.types)
          field.values.push_back(values[type]);
        program.fields.push_back(std::move(field));
      }
      parameters.push_back(std::move(parameter));
    }

    // Every parameter that an override sets is one that the term uses.
    for (const PairOverride *entry : overrides)
      for (auto &[parameter, value] : entry->parameters)
        if (!tables.count(parameter))
          return makeError("[[energy.nbfix]] sets '" + parameter +
                           "', which is not a parameter of the types that "
                           "the term '" + term.name + "' uses");

    // The types of the particles, which the lookups take.
    if (!tables.empty() &&
        llvm::none_of(program.fields,
                      [](const Program::Field &field) {
                        return field.isInteger;
                      })) {
      Program::Field field;
      field.name = "type";
      field.isInteger = true;
      for (unsigned type : system.types)
        field.values.push_back(type);
      program.fields.push_back(std::move(field));
    }
    termTables.push_back(std::move(tables));
    expressions.push_back(std::move(*expression));
  }

  // The parameters of the types that the terms over triplets take for a
  // particle by its place, `sigma1` for the center (D160): a field where
  // they differ between the types, a number where they do not.
  for (const TripletTerm &term : control.triplets) {
    Expression expression =
        llvm::cantFail(Expression::parse(term.expression, control.functions));
    for (const std::string &name : expression.getNames()) {
      std::string stem = getTripletParameter(term, name);
      if (stem.empty() ||
          llvm::any_of(parameters,
                       [&](Parameter &known) { return known.name == stem; }))
        continue;
      std::vector<double> values;
      for (const ParticleType &type : control.types)
        values.push_back(llvm::find_if(type.parameters, [&](auto &entry) {
                           return entry.first == stem;
                         })->second);
      Parameter parameter;
      parameter.name = stem;
      parameter.value = values.front();
      parameter.isUniform = llvm::all_of(
          values, [&](double value) { return value == values.front(); });
      if (!parameter.isUniform) {
        parameter.field = program.fields.size();
        Program::Field field;
        field.name = stem;
        for (unsigned type : system.types)
          field.values.push_back(values[type]);
        program.fields.push_back(std::move(field));
      }
      parameters.push_back(std::move(parameter));
    }
  }
  return computeDispersion();
}

std::string Builder::getTripletParameter(const TripletTerm &term,
                                         StringRef name) const {
  if (name.size() < 2 || !llvm::is_contained("123", name.back()) ||
      llvm::any_of(term.constants,
                   [&](const auto &c) { return c.first == name; }))
    return "";
  StringRef stem = name.drop_back();
  if (control.types.empty() ||
      !llvm::all_of(control.types, [&](const ParticleType &type) {
        return llvm::any_of(type.parameters,
                            [&](const auto &p) { return p.first == stem; });
      }))
    return "";
  return stem.str();
}

llvm::Error Builder::collectPairValues(
    unsigned index, std::vector<std::vector<llvm::StringMap<double>>> &values) {
  const PairTerm &term = control.pairs[index];
  unsigned count = control.types.size();
  values.assign(count, std::vector<llvm::StringMap<double>>(count));
  for (unsigned a = 0; a != count; ++a)
    for (unsigned b = 0; b != count; ++b) {
      llvm::StringMap<double> &pair = values[a][b];
      pair["coulomb"] = coulombConstant;
      for (auto &[name, value] : term.constants)
        pair[name] = value;
      for (const Parameter &parameter : parameters) {
        auto valueOf = [&](unsigned type) {
          for (auto &[name, value] : control.types[type].parameters)
            if (name == parameter.name)
              return value;
          return 0.0;
        };
        pair[parameter.name] =
            parameter.isUniform
                ? parameter.value
                : mix(term.mixing.lookup(parameter.name), valueOf(a),
                      valueOf(b));
      }
      for (auto &[name, table] : termTables[index])
        pair[name] = program.tables[table].values[a * count + b];
    }
  return llvm::Error::success();
}

llvm::Error Builder::computeDispersion() {
  bool corrects = llvm::any_of(control.pairs, [](const PairTerm &term) {
    return term.dispersion != DispersionCorrection::None;
  });
  if (!corrects)
    return llvm::Error::success();
  if (control.truncation != Truncation::None)
    return makeError("'dispersion_correction' needs a plain cutoff: "
                     "'switch_distance' equal to 'cutoff', and no "
                     "'lennard_jones_modifier'");

  // Beyond the cutoff the term is taken to be its dispersion, −C6 / r⁶,
  // and the density to be uniform [AllenTildesley2017, GromacsManual2025]:
  //
  //   E = −(2π / 3V) N² ⟨C6⟩ / rc³        W = 6 E
  //
  // with ⟨C6⟩ the mean over the pairs of distinct particles, as GROMACS
  // takes it, so that the pressure changes by 2E / V.
  // The repulsion beyond the cutoff is left out, as the engines leave it
  // out. C6 is the limit of −r⁶ u(r); a term that does not decay as 1/r⁶ is
  // an error. The expression is in the units of the control file.
  unsigned count = control.types.size();
  std::vector<double> numbers(count, 0.0);
  for (unsigned type : system.types)
    numbers[type] += 1.0;
  double rc = control.cutoffDistance;
  double volume = system.box[0] * system.box[1] * system.box[2] /
                  (units::length * units::length * units::length);

  double energy = 0.0;
  double n = static_cast<double>(system.getNumParticles());
  double scale = n > 1.0 ? n / (n - 1.0) : 0.0;
  for (unsigned index = 0, e = control.pairs.size(); index != e; ++index) {
    const PairTerm &term = control.pairs[index];
    if (term.dispersion == DispersionCorrection::None)
      continue;
    std::vector<std::vector<llvm::StringMap<double>>> values;
    if (llvm::Error error = collectPairValues(index, values))
      return error;
    const Expression &expression = expressions[index];
    for (unsigned a = 0; a != count; ++a)
      for (unsigned b = 0; b != count; ++b) {
        llvm::StringMap<double> pair = values[a][b];
        auto c6At = [&](double r) {
          pair["r"] = r;
          return -std::pow(r, 6) * expression.evaluate(pair);
        };
        double near = c6At(1.0e3 * rc), far = c6At(1.0e4 * rc);
        if (!std::isfinite(near) || !std::isfinite(far) ||
            std::fabs(near - far) > 1.0e-9 * std::max(std::fabs(far), 1.0e-300))
          return makeError("'dispersion_corr' of the term '" + term.name +
                           "' needs a term that decays as 1/r^6 beyond the "
                           "cutoff");
        double pairs = numbers[a] * (numbers[b] - (a == b ? 1.0 : 0.0));
        energy += -2.0 * M_PI / (3.0 * volume) * scale * pairs * far /
                  (rc * rc * rc);
      }
  }
  program.dispersionEnergy = energy * units::energy;
  program.dispersionVirial = 6.0 * energy * units::energy;
  return llvm::Error::success();
}

//===----------------------------------------------------------------------===//
// Runs from a topology
//===----------------------------------------------------------------------===//

/// The Coulomb constant in kJ nm mol⁻¹ e⁻², from CODATA 2018.
constexpr double coulombInternal = 138.935457644;

llvm::Error Builder::collectTopology() {
  const Topology &topology = *system.topology;
  size_t count = topology.getNumParticles();

  // The fields of the particles.
  Program::Field charges;
  charges.name = "q";
  charges.values = topology.charges;
  program.fields.push_back(std::move(charges));
  Program::Field types;
  types.name = "type";
  types.isInteger = true;
  for (unsigned type : topology.types)
    types.values.push_back(type);
  program.fields.push_back(std::move(types));
  // Generalized Born (D144): the radius less its offset, 0.009 nm, the
  // radius that screens the others, that times the scale of the particle,
  // and the radius itself, in nm.
  if (control.implicitSolvent != Control::ImplicitSolvent::None) {
    Program::Field offset, scaled, radius;
    offset.name = "gb_offset";
    scaled.name = "gb_scaled";
    radius.name = "gb_radius";
    for (size_t i = 0; i != count; ++i) {
      double rho = topology.bornRadii[i];
      offset.values.push_back(rho - 0.009);
      scaled.values.push_back(topology.bornScreens[i] * (rho - 0.009));
      radius.values.push_back(rho);
    }
    program.fields.push_back(std::move(offset));
    program.fields.push_back(std::move(scaled));
    program.fields.push_back(std::move(radius));
  }
  (void)count;
  // The restraints: the constant of each particle, 0 for one that is not
  // restrained, and its reference position, in nm.
  if (hasRestraints()) {
    Program::Field constants;
    constants.name = "rest_k";
    constants.values = system.restraintConstants;
    program.fields.push_back(std::move(constants));
    // Under a barostat each reference is its center c, which scales with
    // the cell, plus its offset o from it, which stays (D124). The center of
    // the references of `reference_scaling = "CENTER"` is their mean, so
    // that their shape stays: scaled about the origin, the restraints would
    // hold a solute as much smaller as the cell has become. That of `"ALL"`
    // is the reference itself, with no offset. Without a barostat the
    // offsets are the references.
    size_t count = system.getNumParticles(), centered = 0;
    double mean[3] = {0.0, 0.0, 0.0};
    for (size_t i = 0; i != count; ++i)
      if (system.restraintConstants[i] > 0.0 &&
          system.restraintScaling[i] == ReferenceScaling::Center) {
        ++centered;
        for (int c = 0; c != 3; ++c)
          mean[c] += system.referencePositions[3 * i + c];
      }
    for (int c = 0; c != 3; ++c)
      mean[c] = centered ? mean[c] / centered : 0.0;
    std::vector<double> centers(3 * count, 0.0);
    if (scalesReference())
      for (size_t i = 0; i != count; ++i)
        for (int c = 0; c != 3; ++c)
          centers[3 * i + c] =
              system.restraintScaling[i] == ReferenceScaling::All
                  ? system.referencePositions[3 * i + c]
                  : mean[c];
    for (int c = 0; c != 3; ++c) {
      Program::Field reference;
      reference.name = std::string("rest_") + "xyz"[c];
      for (size_t i = 0; i != count; ++i)
        reference.values.push_back(system.referencePositions[3 * i + c] -
                                   centers[3 * i + c]);
      program.fields.push_back(std::move(reference));
    }
    for (int c = 0; scalesReference() && c != 3; ++c) {
      Program::Field center;
      center.name = std::string("rest_c") + "xyz"[c];
      for (size_t i = 0; i != count; ++i)
        center.values.push_back(centers[3 * i + c]);
      program.fields.push_back(std::move(center));
    }
  }

  // The terms of the absolute positions (D148): a flag for each particle,
  // 1 for those of the term, and each parameter given as a list.
  for (auto [k, term] : llvm::enumerate(topology.externalTerms)) {
    std::string flag = "ext" + std::to_string(k);
    Program::Field on;
    on.name = flag;
    on.values.assign(count, 0.0);
    for (unsigned particle : term.particles)
      on.values[particle] = 1.0;
    program.fields.push_back(std::move(on));
    for (const auto &[name, values] : term.parameters) {
      Program::Field field;
      field.name = flag + "_" + name;
      field.values.assign(count, 0.0);
      for (auto [place, particle] : llvm::enumerate(term.particles))
        field.values[particle] = values[place];
      program.fields.push_back(std::move(field));
    }
  }

  // The interaction groups of the pair terms, a flag for each particle
  // (D137).
  for (auto [k, groups] : llvm::enumerate(system.pairGroups))
    for (auto [g, flags] : llvm::enumerate(groups)) {
      Program::Field field;
      field.name = "pg" + std::to_string(k) + (g == 0 ? "a" : "b");
      for (bool flag : flags)
        field.values.push_back(flag ? 1.0 : 0.0);
      program.fields.push_back(std::move(field));
    }

  // The parameters of each particle that the pair terms take, `w1` and
  // `w2` of `w`, a field gathered for both particles (D165).
  for (const auto &[name, values] : system.particleParameters) {
    bool used = llvm::any_of(control.pairs, [&](const PairTerm &term) {
      Expression expression = llvm::cantFail(
          Expression::parse(term.expression, control.functions));
      return llvm::is_contained(expression.getNames(), name + "1") ||
             llvm::is_contained(expression.getNames(), name + "2");
    });
    if (!used)
      continue;
    Program::Field field;
    field.name = "pp_" + name;
    field.values = values;
    program.fields.push_back(std::move(field));
  }

  // [free_energy] (D161): 1 for each particle that it decouples.
  if (decouples()) {
    Program::Field flags;
    flags.name = "alch";
    for (bool flag : system.alchemical)
      flags.values.push_back(flag ? 1.0 : 0.0);
    program.fields.push_back(std::move(flags));
  }

  // Lennard-Jones for each pair of types.
  unsigned numTypes = topology.getNumTypes();
  program.tables.push_back({"lj_sigma", numTypes, topology.sigma});
  program.tables.push_back({"lj_epsilon", numTypes, topology.epsilon});

  auto addSet = [&](StringRef name, unsigned arity) -> Program::TupleSet & {
    Program::TupleSet set;
    set.name = name.str();
    set.arity = arity;
    program.tupleSets.push_back(std::move(set));
    return program.tupleSets.back();
  };
  auto addField = [&](Program::TupleSet &set, StringRef name) {
    Program::Field field;
    field.name = name.str();
    set.fields.push_back(std::move(field));
    return set.fields.size() - 1;
  };

  if (!topology.bonds.empty()) {
    Program::TupleSet &set = addSet("bonds", 2);
    size_t k = addField(set, "k"), r0 = addField(set, "r0");
    for (const Topology::Bond &bond : topology.bonds) {
      set.members.push_back(bond.i);
      set.members.push_back(bond.j);
      set.fields[k].values.push_back(bond.k);
      set.fields[r0].values.push_back(bond.r0);
    }
  }
  if (!topology.angles.empty()) {
    Program::TupleSet &set = addSet("angles", 3);
    size_t k = addField(set, "k"), t0 = addField(set, "theta0");
    for (const Topology::Angle &angle : topology.angles) {
      for (unsigned member : {angle.i, angle.j, angle.k})
        set.members.push_back(member);
      set.fields[k].values.push_back(angle.force);
      set.fields[t0].values.push_back(angle.theta0);
    }
  }
  if (!topology.ureyBradleys.empty()) {
    Program::TupleSet &set = addSet("urey_bradley", 2);
    size_t k = addField(set, "k"), r0 = addField(set, "r0");
    for (const Topology::UreyBradley &term : topology.ureyBradleys) {
      set.members.push_back(term.i);
      set.members.push_back(term.k);
      set.fields[k].values.push_back(term.force);
      set.fields[r0].values.push_back(term.r0);
    }
  }
  if (!topology.dihedrals.empty()) {
    Program::TupleSet &set = addSet("dihedrals", 4);
    size_t k = addField(set, "k"), n = addField(set, "n"),
           phase = addField(set, "phase");
    for (const Topology::Dihedral &dihedral : topology.dihedrals) {
      for (unsigned member :
           {dihedral.i, dihedral.j, dihedral.k, dihedral.l})
        set.members.push_back(member);
      set.fields[k].values.push_back(dihedral.force);
      set.fields[n].values.push_back(dihedral.n);
      set.fields[phase].values.push_back(dihedral.phase);
    }
  }
  if (!topology.harmonicImpropers.empty()) {
    Program::TupleSet &set = addSet("impropers", 4);
    size_t k = addField(set, "k"), xi0 = addField(set, "xi0");
    for (const Topology::HarmonicImproper &term : topology.harmonicImpropers) {
      for (unsigned member : {term.i, term.j, term.k, term.l})
        set.members.push_back(member);
      set.fields[k].values.push_back(term.force);
      set.fields[xi0].values.push_back(term.xi0);
    }
  }
  // The terms given by expressions (D136), with their parameters as fields
  // of their tuples, in the units of the control file.
  for (const TupleTerm &term : topology.tupleTerms) {
    // A term over the centers of groups (D139), one set of pairs: for
    // each group the pairs of a member and the reference of the group, and
    // for each group after the first the pair of its reference and that of
    // the first. A field for each group and each such link, `w<k>`, holds
    // the weight of the member in its pairs, 1 in the pair of the link,
    // and 0 elsewhere, so that one loop gives all the sums.
    if (term.isCentroid()) {
      size_t groups = term.centers.size();
      Program::TupleSet &set = addSet("cg_" + term.name, 2);
      set.oriented = true;
      set.reversible = false;
      for (size_t k = 0; k != 2 * groups - 1; ++k)
        addField(set, "w" + std::to_string(k));
      auto addPair = [&](unsigned i, unsigned j, size_t slot, double weight) {
        set.members.push_back(i);
        set.members.push_back(j);
        for (size_t k = 0; k != 2 * groups - 1; ++k)
          set.fields[k].values.push_back(k == slot ? weight : 0.0);
      };
      for (auto [g, center] : llvm::enumerate(term.centers)) {
        for (auto [i, weight] :
             llvm::zip_equal(center.members, center.weights))
          if (i != center.reference)
            addPair(i, center.reference, g, weight);
        if (g > 0)
          addPair(center.reference, term.centers.front().reference,
                  groups + g - 1, 1.0);
      }
      continue;
    }
    Program::TupleSet &set = addSet("custom_" + term.name, term.arity);
    for (unsigned member : term.particles)
      set.members.push_back(member);
    for (const auto &[name, values] : term.parameters) {
      size_t field = addField(set, name);
      set.fields[field].values = values;
    }
  }
  if (!topology.pairs.empty()) {
    // The factors enter the parameters: ε s_LJ, and f q_i q_j s_C.
    Program::TupleSet &set = addSet("pairs14", 2);
    size_t sigma = addField(set, "sigma"), epsilon = addField(set, "epsilon"),
           qq = addField(set, "qq");
    for (const Topology::Pair &pair : topology.pairs) {
      set.members.push_back(pair.i);
      set.members.push_back(pair.j);
      set.fields[sigma].values.push_back(pair.sigma);
      set.fields[epsilon].values.push_back(pair.epsilon * pair.scaleLJ);
      set.fields[qq].values.push_back(coulombInternal *
                                      topology.charges[pair.i] *
                                      topology.charges[pair.j] *
                                      pair.scaleCoulomb);
    }
  }
  // The virtual sites: the site, then the atoms that it is built from.
  for (auto [kind, name] :
       {std::pair{Topology::VirtualSite::AmberWater, "sites_amber"},
        std::pair{Topology::VirtualSite::Linear, "sites_linear"}}) {
    if (llvm::none_of(topology.virtualSites,
                      [&](const Topology::VirtualSite &site) {
                        return site.kind == kind;
                      }))
      continue;
    Program::TupleSet &set = addSet(name, 4);
    set.reversible = false;
    size_t a = addField(set, "a"), b = addField(set, "b");
    for (const Topology::VirtualSite &site : topology.virtualSites) {
      if (site.kind != kind)
        continue;
      for (unsigned member : {site.site, site.i, site.j, site.k})
        set.members.push_back(member);
      set.fields[a].values.push_back(site.a);
      set.fields[b].values.push_back(site.b);
    }
  }
  if (!topology.settles.empty()) {
    // The oxygen and the two hydrogens of each rigid water.
    Program::TupleSet &set = addSet("settles", 3);
    set.reversible = false;
    size_t oh = addField(set, "doh"), hh = addField(set, "dhh");
    for (const Topology::Settle &settle : topology.settles) {
      for (unsigned k = 0; k != 3; ++k)
        set.members.push_back(settle.oxygen + k);
      set.fields[oh].values.push_back(settle.distanceOH);
      set.fields[hh].values.push_back(settle.distanceHH);
    }
  }
  // The groups of SHAKE, the heavy atom first, in a set for each number of
  // hydrogens.
  for (unsigned count = 1; count <= 3; ++count) {
    if (llvm::none_of(topology.shakes, [&](const Topology::Shake &shake) {
          return shake.hydrogens.size() == count;
        }))
      continue;
    Program::TupleSet &set = addSet("shake" + std::to_string(count),
                                    count + 1);
    set.reversible = false;
    std::vector<size_t> lengths;
    for (unsigned k = 1; k <= count; ++k)
      lengths.push_back(addField(set, "d" + std::to_string(k)));
    for (const Topology::Shake &shake : topology.shakes) {
      if (shake.hydrogens.size() != count)
        continue;
      set.members.push_back(shake.center);
      for (unsigned k = 0; k != count; ++k) {
        set.members.push_back(shake.hydrogens[k]);
        set.fields[lengths[k]].values.push_back(shake.lengths[k]);
      }
    }
  }
  // The anchor of each particle: the oxygen of its rigid water, the heavy
  // atom of its SHAKE group, that of the first atom that places a virtual
  // site, or itself. The order of the particles puts a particle where its
  // anchor is (emitReorder), so that the members of a group stay together,
  // in one warp of a device for most (D110). A virtual site ordered by its
  // own position fell into another cell than its atoms where they were near
  // a face of one: an extra point of OPC in 25 waters of 1000.
  if (!topology.settles.empty() || !topology.shakes.empty() ||
      !topology.virtualSites.empty()) {
    std::vector<int32_t> anchors(count);
    for (size_t i = 0; i != count; ++i)
      anchors[i] = i;
    for (const Topology::Settle &settle : topology.settles)
      for (unsigned k = 0; k != 3; ++k)
        anchors[settle.oxygen + k] = settle.oxygen;
    for (const Topology::Shake &shake : topology.shakes)
      for (int32_t hydrogen : shake.hydrogens)
        anchors[hydrogen] = shake.center;
    for (const Topology::VirtualSite &site : topology.virtualSites)
      anchors[site.site] = anchors[site.i];
    Program::TupleSet &set = addSet("anchors", 2);
    set.reversible = false;
    set.oriented = true;
    for (size_t i = 0; i != count; ++i) {
      set.members.push_back(i);
      set.members.push_back(anchors[i]);
    }
  }
  if (!topology.cmaps.empty()) {
    // The map of each term, and the bicubic patches of every map: a row
    // of 16 coefficients for each cell.
    Program::TupleSet &set = addSet("cmap", 5);
    set.reversible = false;
    size_t map = addField(set, "map");
    for (const Topology::CMap &cmap : topology.cmaps) {
      for (unsigned member : {cmap.i, cmap.j, cmap.k, cmap.l, cmap.m})
        set.members.push_back(member);
      set.fields[map].values.push_back(cmap.map);
    }
    Program::Table table;
    table.name = "cmap";
    table.values = getCMapCoefficients(topology);
    table.columns = 16;
    table.count = table.values.size() / 16;
    program.tables.push_back(std::move(table));
  }
  if (!topology.exclusions.empty()) {
    Program::TupleSet &set = addSet("excluded", 2);
    for (auto [i, j] : topology.exclusions) {
      set.members.push_back(i);
      set.members.push_back(j);
    }
  }
  // The pairs within the selection of [free_energy] that are not excluded,
  // whose Coulomb particle mesh Ewald adds back (D161).
  if (control.pme && !system.alchemicalPairs.empty()) {
    Program::TupleSet &set = addSet("alchemical_pairs", 2);
    for (auto [i, j] : system.alchemicalPairs) {
      set.members.push_back(i);
      set.members.push_back(j);
    }
  }

  if (control.pme)
    if (llvm::Error error = collectPME())
      return error;
  if (control.ljpme)
    if (llvm::Error error = collectLJPME())
      return error;
  // The reaction field (D140): its self term, −c f Σ q² / 2, which with
  // the terms of the excluded pairs makes the field act on every pair of
  // charges within the cutoff, those of one molecule as well: the
  // convention of GROMACS [GromacsManual2025], whose Coulomb (SR) MDIR's
  // matches; OpenMM's leaves both out.
  if (control.reactionField) {
    double squares = 0.0;
    for (double q : topology.charges)
      squares += q * q;
    double self =
        -0.5 * getReactionField(control).second * coulombInternal * squares;
    program.reactionField = true;
    program.coulombConstantEnergy = self;
    program.coulombSelfEnergy = self;
  }

  // The correction for the dispersion (Section 7.2 of design-m1.md): N²
  // times the mean of C6 over the pairs of distinct particles that are not
  // excluded, as GROMACS takes it.
  if (control.topologyDispersion != DispersionCorrection::None) {
    // With [free_energy] the pairs that the selection decouples count
    // times 1 − λ: the tail of their soft-core Lennard-Jones is that of
    // the plain one times 1 − λ (D161).
    double energy = getTopologyDispersion(false);
    if (decouples()) {
      double lambda = getLambda("vdw");
      energy = (1.0 - lambda) * energy + lambda * getTopologyDispersion(true);
    }
    program.dispersionEnergy = energy;
    program.dispersionVirial = 6.0 * energy;
  }
  if (control.hasFreeEnergy)
    collectFreeEnergyConstants();
  return llvm::Error::success();
}

double Builder::getTopologyDispersion(bool decoupled) const {
  const Topology &topology = *system.topology;
  unsigned numTypes = topology.getNumTypes();
  std::vector<double> numbers(numTypes, 0.0);
  for (unsigned type : topology.types)
    numbers[type] += 1.0;
  auto c6 = [&](unsigned a, unsigned b) {
    double sigma = topology.sigma[a * numTypes + b];
    return 4.0 * topology.epsilon[a * numTypes + b] * std::pow(sigma, 6);
  };
  double sum = 0.0;
  for (unsigned a = 0; a != numTypes; ++a)
    for (unsigned b = 0; b != numTypes; ++b)
      sum += numbers[a] * (numbers[b] - (a == b ? 1.0 : 0.0)) * c6(a, b);
  for (auto [i, j] : topology.exclusions)
    sum -= 2.0 * c6(topology.types[i], topology.types[j]);
  // Without the pairs of a particle of the selection and one of the rest,
  // over the same number of pairs.
  if (decoupled) {
    std::vector<double> inside(numTypes, 0.0), outside(numTypes, 0.0);
    for (size_t i = 0, e = topology.types.size(); i != e; ++i)
      (system.alchemical[i] ? inside : outside)[topology.types[i]] += 1.0;
    for (unsigned a = 0; a != numTypes; ++a)
      for (unsigned b = 0; b != numTypes; ++b)
        sum -= 2.0 * inside[a] * outside[b] * c6(a, b);
  }
  double n = static_cast<double>(topology.getNumParticles());
  double pairs = n * (n - 1.0) - 2.0 * topology.exclusions.size();
  double mean = pairs > 0.0 ? sum / pairs : 0.0;
  double rc = control.cutoffDistance * units::length;
  double volume = system.box[0] * system.box[1] * system.box[2];
  return -2.0 * M_PI / (3.0 * volume) * n * n * mean / (rc * rc * rc);
}

std::pair<double, double> Builder::getPMEConstants(double lambda) const {
  const Topology &topology = *system.topology;
  double squares = 0.0, net = 0.0;
  for (size_t i = 0, e = topology.charges.size(); i != e; ++i) {
    double q = topology.charges[i];
    if (decouples() && system.alchemical[i])
      q *= 1.0 - lambda;
    squares += q * q;
    net += q;
  }
  double beta = program.pmeBeta;
  double self = -coulombInternal * beta / std::sqrt(M_PI) * squares;
  double volume = system.box[0] * system.box[1] * system.box[2];
  double background =
      -coulombInternal * M_PI * net * net / (2.0 * volume * beta * beta);
  return {self, background};
}

void Builder::collectFreeEnergyConstants() {
  // At each state, the constants that depend on λ: the self term of
  // particle mesh Ewald, which does not depend on the volume, and the
  // background of a net charge and the correction for the dispersion,
  // proportional to 1 / V, at the volume of the file. Their derivatives
  // follow at the state of the run: in λ of the Coulomb, 2 s s' times the
  // part of the selection in the self term, and in the background as its
  // net charge Q(λ) = Q_0 + s Q_1 says; in λ of the Lennard-Jones, the
  // difference of the two corrections (D161).
  const Control::FreeEnergy &energy = control.freeEnergy;
  size_t states = energy.getNumStates();
  bool dispersion = control.topologyDispersion != DispersionCorrection::None;
  double coupled = 0.0, decoupled = 0.0;
  if (dispersion && decouples()) {
    coupled = getTopologyDispersion(false);
    decoupled = getTopologyDispersion(true);
  } else if (dispersion) {
    coupled = decoupled = getTopologyDispersion(false);
  }
  for (size_t k = 0; k != states; ++k) {
    double fixed = 0.0, scaled = 0.0;
    if (program.pme && decouples()) {
      auto [self, background] = getPMEConstants(energy.get("coulomb", k));
      fixed += self;
      scaled += background;
    }
    double lambda = decouples() ? energy.get("vdw", k) : 0.0;
    scaled += (1.0 - lambda) * coupled + lambda * decoupled;
    program.stateFixedEnergies.push_back(fixed);
    program.stateVolumeEnergies.push_back(scaled);
  }
  for (const auto &[name, values] : energy.lambdas) {
    double fixed = 0.0, scaled = 0.0;
    if (decouples() && name == "coulomb" && program.pme) {
      // The derivatives of quadratics in λ, from three points.
      double lambda = getLambda("coulomb");
      auto [self0, background0] = getPMEConstants(0.0);
      auto [self1, background1] = getPMEConstants(1.0);
      auto [selfh, backgroundh] = getPMEConstants(0.5);
      auto slope = [&](double at0, double ath, double at1) {
        // f(λ) = a + b λ + c λ², f'(λ) = b + 2 c λ.
        double c = 2.0 * (at0 + at1 - 2.0 * ath);
        double b = at1 - at0 - c;
        return b + 2.0 * c * lambda;
      };
      fixed = slope(self0, selfh, self1);
      scaled = slope(background0, backgroundh, background1);
    }
    if (decouples() && name == "vdw")
      scaled = decoupled - coupled;
    program.lambdaFixedDerivatives.push_back(fixed);
    program.lambdaVolumeDerivatives.push_back(scaled);
  }
}

/// The smallest even number of points, at least `least`, whose prime
/// factors are 2, 3, 5, and 7, which the FFT handles fast. On an even grid
/// the influence function of sander agrees with MDIR's (docs/pme-m1.md,
/// Section 1.1).
static int64_t getSmoothSize(int64_t least) {
  for (int64_t size = least;; ++size) {
    if (size % 2 != 0)
      continue;
    int64_t rest = size;
    for (int64_t factor : {2, 3, 5, 7})
      while (rest % factor == 0)
        rest /= factor;
    if (rest == 1)
      return size;
  }
}

/// |b(m)|² of the Euler exponential spline of order `order` for each index
/// of a grid of `count` points: 1 / |Σ_{k=0}^{n−2} M_n(k + 1)
/// exp(2π i m k / K)|² [Essmann1995].
static std::vector<double> getSplineModuli(int64_t count, int64_t order) {
  // M_n at 1 .. n − 1, from the recursion M_n(x) = (x M_{n−1}(x) + (n − x)
  // M_{n−1}(x − 1)) / (n − 1), with M_2(x) = 1 − |x − 1|.
  std::vector<double> m(order + 1, 0.0), next(order + 1, 0.0);
  m[1] = 1.0;
  for (int64_t n = 3; n <= order; ++n) {
    std::fill(next.begin(), next.end(), 0.0);
    for (int64_t x = 1; x < n; ++x)
      next[x] = (x * m[x] + (n - x) * m[x - 1]) / (n - 1);
    m.swap(next);
  }
  std::vector<double> moduli(count);
  for (int64_t k = 0; k != count; ++k) {
    double re = 0.0, im = 0.0;
    for (int64_t j = 0; j <= order - 2; ++j) {
      double angle = 2.0 * M_PI * k * j / count;
      re += m[j + 1] * std::cos(angle);
      im += m[j + 1] * std::sin(angle);
    }
    moduli[k] = 1.0 / (re * re + im * im);
  }
  return moduli;
}

/// The square of the factor λ(m) of each index of a grid of `count` points
/// for B-splines of order `order`: the ratio of the sums over the aliases
/// of m, S_n(x) / S_2n(x) with S_p(x) = Σ_j (x / (x + π j))^p and
/// x = π m / K, for m from −K/2 to K/2 (docs/pme-m1.md, Section 1.1). It
/// scales the influence function so that the energy that the grid gives
/// comes closer to the Ewald sum.
static std::vector<double> getAliasFactors(int64_t count, int64_t order) {
  std::vector<double> factors(count, 1.0);
  for (int64_t k = 0; k != count; ++k) {
    int64_t m = k <= count / 2 ? k : k - count;
    if (m == 0)
      continue;
    double x = M_PI * static_cast<double>(m) / static_cast<double>(count);
    double single = 1.0, twice = 1.0;
    for (int64_t j = 1; j <= 50; ++j)
      for (double sign : {1.0, -1.0}) {
        double s = x / (x + sign * M_PI * j);
        single += std::pow(s, order);
        twice += std::pow(s, 2 * order);
      }
    double lambda = single / twice;
    factors[k] = lambda * lambda;
  }
  return factors;
}

llvm::Error Builder::collectPME() {
  const Topology &topology = *system.topology;
  double rc = control.cutoffDistance * units::length;

  // β: given, or such that erfc(β rc) is the tolerance, by bisection.
  double beta = control.pmeAlpha / units::length;
  if (beta == 0.0) {
    double low = 0.0, high = 1.0;
    while (std::erfc(high * rc) > control.pmeAlphaTolerance)
      high *= 2.0;
    for (int i = 0; i != 100; ++i) {
      double middle = 0.5 * (low + high);
      if (std::erfc(middle * rc) > control.pmeAlphaTolerance)
        low = middle;
      else
        high = middle;
    }
    beta = 0.5 * (low + high);
  }

  int64_t grid[3];
  if (llvm::Error error =
          collectMesh("pme_moduli", control.pmeGrid, control.pmeMaxSpacing,
                      control.pmeOrder, control.pmeOptimal, grid))
    return error;

  // The grid holds charges in fixed point at the scale 2^40, up to about
  // 8e6 e at a point (D70); a charge beyond 100 e is not of a molecule.
  for (auto [index, q] : llvm::enumerate(topology.charges))
    if (!(std::fabs(q) < 100.0))
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "the charge of the atom %zu is %g e; particle mesh Ewald holds "
          "charges of less than 100 e",
          index + 1, q);

  // The self term, and the background that neutralizes a net charge,
  // whose virial is its energy on the diagonal; with the charges of the
  // selection of [free_energy] scaled as at the state of the run (D161).
  program.pmeBeta = beta;
  auto [self, background] = getPMEConstants(getLambda("coulomb"));
  program.pme = true;
  program.coulombConstantEnergy = self + background;
  program.coulombSelfEnergy = self;
  program.coulombConstantVirial = 3.0 * background;
  program.pmeBeta = beta;
  for (int k = 0; k != 3; ++k)
    program.pmeGrid[k] = grid[k];
  return llvm::Error::success();
}

llvm::Error Builder::collectMesh(StringRef name, const int64_t (&given)[3],
                                 double spacing, int64_t order, bool optimal,
                                 int64_t (&grid)[3]) {
  // The grid: given, or no wider than the largest spacing in the cell of
  // the file of coordinates. The grid stays as the barostat changes the
  // cell, finer as the cell shrinks and coarser as it grows.
  for (int k = 0; k != 3; ++k) {
    grid[k] = given[k];
    double edge = system.inputBox[k] > 0.0 ? system.inputBox[k] : system.box[k];
    // A triclinic cell is spaced along its vectors: |a|, |b|, |c|.
    if (isTriclinic())
      edge = k == 0 ? system.box[0]
             : k == 1 ? std::hypot(system.tilt[0], system.box[1])
                      : std::sqrt(system.tilt[1] * system.tilt[1] +
                                  system.tilt[2] * system.tilt[2] +
                                  system.box[2] * system.box[2]);
    if (grid[k] == 0)
      grid[k] = getSmoothSize(static_cast<int64_t>(std::ceil(
          edge / (spacing * units::length) - 1e-9)));
    if (grid[k] < 2 * order)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "the grid of particle mesh Ewald has %lld points along an edge, "
          "fewer than twice the order %lld",
          static_cast<long long>(grid[k]), static_cast<long long>(order));
  }

  // The factors of the influence function along each edge, which do not
  // depend on the cell: |b(k)|² of the B-splines, times the factor of the
  // aliasing if it is taken. The program computes the rest from the cell.
  int64_t longest = std::max({grid[0], grid[1], grid[2]});
  Program::Table table;
  table.name = name.str();
  table.count = 3;
  table.columns = longest;
  table.values.assign(3 * longest, 0.0);
  for (int k = 0; k != 3; ++k) {
    std::vector<double> moduli = getSplineModuli(grid[k], order);
    if (optimal) {
      std::vector<double> factors = getAliasFactors(grid[k], order);
      for (int64_t i = 0; i != grid[k]; ++i)
        moduli[i] *= factors[i];
    }
    std::copy(moduli.begin(), moduli.end(),
              table.values.begin() + k * longest);
  }
  program.tables.push_back(std::move(table));
  return llvm::Error::success();
}

/// g(x) = exp(−x²) (1 + x² + x⁴/2), the share of the dispersion −c_i c_j /
/// r⁶ at x = β r that the direct terms of particle mesh Ewald keep
/// [Essmann1995].
static double getDispersionScreen(double x) {
  double x2 = x * x;
  return std::exp(-x2) * (1.0 + x2 + 0.5 * x2 * x2);
}

llvm::Error Builder::collectLJPME() {
  const Topology &topology = *system.topology;
  double rc = control.cutoffDistance * units::length;

  // β: given, or such that g(β rc) is the tolerance, by bisection; g
  // falls from 1 at 0.
  double beta = control.ljpmeAlpha / units::length;
  if (beta == 0.0) {
    double low = 0.0, high = 1.0;
    while (getDispersionScreen(high * rc) > control.ljpmeTolerance)
      high *= 2.0;
    for (int i = 0; i != 100; ++i) {
      double middle = 0.5 * (low + high);
      if (getDispersionScreen(middle * rc) > control.ljpmeTolerance)
        low = middle;
      else
        high = middle;
    }
    beta = 0.5 * (low + high);
  }

  int64_t grid[3];
  if (llvm::Error error =
          collectMesh("ljpme_moduli", control.ljpmeGrid,
                      control.ljpmeMaxSpacing, control.ljpmeOrder,
                      /*optimal=*/false, grid))
    return error;

  // The coefficient of each type, c = 2 √ε σ³ of the type with itself, so
  // that c_a c_b is the C6 of the pair by the geometric rule of both
  // parameters; the grid takes those of the particles, the direct terms
  // and the excluded pairs those of the pairs of types. For the pairs of
  // the Lorentz–Berthelot rule or set apart (NBFIX) the direct terms hold
  // the Lennard-Jones of the pair itself within the cutoff and take out
  // what the grid adds there [Wennberg2013].
  unsigned numTypes = topology.getNumTypes();
  std::vector<double> coefficients(numTypes);
  for (unsigned a = 0; a != numTypes; ++a) {
    double sigma = topology.sigma[a * numTypes + a];
    double epsilon = topology.epsilon[a * numTypes + a];
    coefficients[a] = 2.0 * std::sqrt(std::max(epsilon, 0.0)) *
                      sigma * sigma * sigma;
  }
  Program::Field field;
  field.name = "ljpme_c";
  double squares = 0.0;
  for (unsigned type : topology.types) {
    field.values.push_back(coefficients[type]);
    squares += coefficients[type] * coefficients[type];
  }
  program.fields.push_back(std::move(field));
  std::vector<double> products(numTypes * numTypes);
  for (unsigned a = 0; a != numTypes; ++a)
    for (unsigned b = 0; b != numTypes; ++b)
      products[a * numTypes + b] = coefficients[a] * coefficients[b];
  program.tables.push_back({"ljpme_c6", numTypes, std::move(products)});

  // The self term: the grid sums −c_i² (1 − g(β r)) / r⁶ over each
  // particle with itself, −c_i² β⁶ / 6 at r = 0, half of it each.
  double beta3 = beta * beta * beta;
  program.ljpme = true;
  program.ljpmeSelfEnergy = beta3 * beta3 / 12.0 * squares;
  program.ljpmeBeta = beta;
  for (int k = 0; k != 3; ++k)
    program.ljpmeGrid[k] = grid[k];
  return llvm::Error::success();
}

llvm::StringMap<std::string> Builder::emitCentroidCoordinates(
    const TupleTerm &term, StringRef indent, StringRef x, StringRef cell,
    StringRef relations, StringRef prefix) {
  unsigned counter = 0;
  auto next = [&]() { return (prefix + llvm::Twine(counter++)).str(); };
  auto op = [&](StringRef name, std::initializer_list<std::string> operands) {
    std::string result = next();
    os << indent << result << " = " << name << " "
       << llvm::join(operands, ", ") << " : f64\n";
    return result;
  };
  auto constant = [&](double value) {
    std::string result = next();
    os << indent << result << " = arith.constant " << formatReal(value)
       << " : f64\n";
    return result;
  };
  // One component of the displacements of the pairs of the set of the
  // term, weighted by the field `w<slot>`, summed. The sums over the one
  // set become one loop (md-exec-fuse-loops).
  std::string set = "cg_" + term.name;
  auto sum = [&](size_t slot, int component) {
    std::string result = next();
    std::string field = "%f_" + set + "_w" + std::to_string(slot);
    os << indent << result << " = md.sum_tuples " << relations << set << ", "
       << x << ", " << cell << " coordinates(displacement(0, 1))\n"
       << indent << "    tuple(" << field << " : !of_" << set << ") {\n"
       << indent << "^bb0(%gd: vector<3xf64>, %gw: f64):\n"
       << indent << "  %gdc = vector.extract %gd[" << component
       << "] : f64 from vector<3xf64>\n"
       << indent << "  %ge = arith.mulf %gw, %gdc : f64\n"
       << indent << "  md.yield %ge : f64\n"
       << indent << "} : !rel_" << set << ", !vec -> f64\n";
    return result;
  };

  // The center of each group from the reference of the first, in Å: the
  // displacement of its reference from that one, and the weighted
  // displacements of its members from its reference. Both are minimum
  // images; their sum is the center while each group lies within half the
  // least width of the cell around its reference (System.cpp).
  using Vector = std::array<std::string, 3>;
  std::string angstrom = constant(1.0 / units::length);
  std::vector<Vector> centers;
  size_t groups = term.centers.size();
  for (size_t g = 0; g != groups; ++g) {
    Vector position;
    for (int c = 0; c != 3; ++c) {
      std::string value = sum(g, c);
      if (g > 0)
        value = op("arith.addf", {sum(groups + g - 1, c), value});
      position[c] = op("arith.mulf", {value, angstrom});
    }
    centers.push_back(position);
  }
  auto subtract = [&](const Vector &u, const Vector &v) {
    Vector w;
    for (int c = 0; c != 3; ++c)
      w[c] = op("arith.subf", {u[c], v[c]});
    return w;
  };
  auto dot = [&](const Vector &u, const Vector &v) {
    std::string total = op("arith.mulf", {u[0], v[0]});
    for (int c = 1; c != 3; ++c)
      total = op("arith.addf", {total, op("arith.mulf", {u[c], v[c]})});
    return total;
  };
  auto cross = [&](const Vector &u, const Vector &v) {
    Vector w;
    for (int c = 0; c != 3; ++c) {
      int i = (c + 1) % 3, j = (c + 2) % 3;
      w[c] = op("arith.subf", {op("arith.mulf", {u[i], v[j]}),
                               op("arith.mulf", {u[j], v[i]})});
    }
    return w;
  };
  auto scale = [&](const std::string &factor, const Vector &u) {
    Vector w;
    for (int c = 0; c != 3; ++c)
      w[c] = op("arith.mulf", {factor, u[c]});
    return w;
  };
  auto norm = [&](const Vector &u) { return op("math.sqrt", {dot(u, u)}); };

  // The coordinate, as the terms over particles take it: the distance and
  // the vector from the first center to the second; the angle at the
  // second center, as atan2(|u × v|, u · v); the dihedral with trans at π.
  llvm::StringMap<std::string> values;
  if (term.arity == 2) {
    Vector d = subtract(centers[1], centers[0]);
    values["dx"] = d[0];
    values["dy"] = d[1];
    values["dz"] = d[2];
    values["r"] = norm(d);
  } else if (term.arity == 3) {
    Vector u = subtract(centers[0], centers[1]);
    Vector v = subtract(centers[2], centers[1]);
    values["theta"] = op("math.atan2", {norm(cross(u, v)), dot(u, v)});
  } else {
    Vector b0 = subtract(centers[0], centers[1]);
    Vector b1 = subtract(centers[2], centers[1]);
    Vector b2 = subtract(centers[3], centers[2]);
    Vector axis = scale(op("arith.divf", {constant(1.0), norm(b1)}), b1);
    Vector v = subtract(b0, scale(dot(b0, axis), axis));
    Vector w = subtract(b2, scale(dot(b2, axis), axis));
    values["theta"] =
        op("math.atan2", {dot(cross(axis, v), w), dot(v, w)});
  }
  return values;
}

void Builder::emitCentroidTerm(size_t index, const TupleTerm &term) {
  // Numbers outside the kernels, named after the term.
  std::string prefix = "%cb" + std::to_string(index) + "_";
  llvm::StringMap<std::string> values =
      emitCentroidCoordinates(term, "  ", "%x", "%cell", "%r_", prefix);
  for (auto [k, parameter] : llvm::enumerate(term.parameters)) {
    std::string name = prefix + "p" + std::to_string(k);
    os << "  " << name << " = arith.constant "
       << formatReal(parameter.second.front()) << " : f64\n";
    values[parameter.first] = name;
  }
  if (control.usesTime)
    values["t"] = "%time";
  bindLambdas(values);
  bindObserved(term.name, values);
  Expression expression = llvm::cantFail(
      Expression::parse(term.expression, control.functions));
  std::string energy = expression.emit(os, values, prefix + "e", "  ", term.name);
  os << "  " << prefix << "kj = arith.constant " << formatReal(units::energy)
     << " : f64\n"
     << "  %u_centroid_" << term.name << " = arith.mulf " << energy << ", "
     << prefix << "kj : f64\n";
}

std::vector<std::pair<size_t, std::string>> Builder::getPullColumns() const {
  std::vector<std::pair<size_t, std::string>> columns;
  if (control.pullFile.empty() || !system.topology)
    return columns;
  for (auto [index, term] : llvm::enumerate(system.topology->tupleTerms)) {
    if (!term.isCentroid())
      continue;
    if (term.arity == 2)
      for (const char *quantity : {"r", "dx", "dy", "dz"})
        columns.push_back({index, quantity});
    else
      columns.push_back({index, "theta"});
  }
  return columns;
}

void Builder::emitPullPotentials() {
  // The energy of each term as a function of its coordinates, which follow
  // the other arguments, so that the derivatives with respect to them give
  // the forces along the coordinates.
  std::vector<std::pair<size_t, std::string>> columns = getPullColumns();
  if (columns.empty())
    return;
  for (size_t index = 0, e = system.topology->tupleTerms.size(); index != e;
       ++index) {
    const TupleTerm &term = system.topology->tupleTerms[index];
    if (!term.isCentroid())
      continue;
    std::vector<std::string> quantities;
    for (const auto &[k, quantity] : columns)
      if (k == index)
        quantities.push_back(quantity);
    os << "md.potential @pullterm" << index << "(%x: !vec, %cell: !md.cell"
       << getFieldParameters() << getTimeParameter();
    for (size_t q = 0; q != quantities.size(); ++q)
      os << ", %pq" << q << ": f64";
    os << ") -> f64 {\n";
    emitLambdaConstants("  ");
    llvm::StringMap<std::string> values;
    for (auto [q, quantity] : llvm::enumerate(quantities))
      values[quantity] = "%pq" + std::to_string(q);
    for (auto [p, parameter] : llvm::enumerate(term.parameters)) {
      std::string name = "%pp" + std::to_string(p);
      os << "  " << name << " = arith.constant "
         << formatReal(parameter.second.front()) << " : f64\n";
      values[parameter.first] = name;
    }
    if (control.usesTime)
      values["t"] = "%time";
    bindLambdas(values);
    Expression expression = llvm::cantFail(
        Expression::parse(term.expression, control.functions));
    std::string energy = expression.emit(os, values, "%pe", "  ", term.name);
    os << "  md.return " << energy << " : f64\n}\n\n";
  }
}

unsigned Builder::getNumPotentialArguments() const {
  unsigned count = 2 + program.fields.size() + program.tables.size();
  for (const Program::TupleSet &set : program.tupleSets)
    count += 1 + set.fields.size();
  return count + (control.usesTime ? 1 : 0);
}

void Builder::emitPullOutput(StringRef indent, StringRef x, StringRef cell,
                             StringRef prefix, StringRef step,
                             StringRef time) {
  std::vector<std::pair<size_t, std::string>> columns = getPullColumns();
  if (columns.empty())
    return;
  // The coordinates of each term, its centers computed once. The members
  // of the tuples follow the order of the particles, as in
  // getFieldValues.
  std::string type = "memref<" + std::to_string(columns.size()) + "xf64>";
  std::string values = ("%pull_values" + step.drop_front()).str();
  std::string relations = ("%r" + prefix.drop_front(2)).str();
  os << indent << values << " = memref.alloca() : " << type << "\n";
  std::vector<std::string> names(columns.size());
  for (size_t index = 0, e = system.topology->tupleTerms.size(); index != e;
       ++index) {
    const TupleTerm &term = system.topology->tupleTerms[index];
    if (!term.isCentroid())
      continue;
    llvm::StringMap<std::string> coordinates = emitCentroidCoordinates(
        term, indent, x, cell, relations,
        values + "_c" + std::to_string(index) + "_");
    for (size_t column = 0; column != columns.size(); ++column)
      if (columns[column].first == index)
        names[column] = coordinates.lookup(columns[column].second);
  }
  for (size_t column = 0, e = columns.size(); column != e; ++column) {
    std::string place = values + "_i" + std::to_string(column);
    os << indent << place << " = arith.constant " << column << " : index\n"
       << indent << "memref.store " << names[column] << ", " << values << "["
       << place << "] : " << type << "\n";
  }
  // The energy of each term, in kcal/mol, and its force: of a term over
  // two centers, the force on the second, F = −(∂E/∂r d/r + ∂E/∂d) with d
  // the vector from the first, and its component along d, in kcal/mol/Å;
  // of an angle or a dihedral, −∂E/∂θ, in kcal/mol/rad.
  std::string terms = ("%pull_terms" + step.drop_front()).str();
  unsigned numTerms = 0;
  for (const TupleTerm &term : system.topology->tupleTerms)
    numTerms += term.isCentroid() ? (term.arity == 2 ? 5 : 2) : 0;
  std::string termType = "memref<" + std::to_string(numTerms) + "xf64>";
  os << indent << terms << " = memref.alloca() : " << termType << "\n";
  unsigned slot = 0;
  unsigned base = getNumPotentialArguments();
  for (size_t index = 0, e = system.topology->tupleTerms.size(); index != e;
       ++index) {
    if (!system.topology->tupleTerms[index].isCentroid())
      continue;
    std::vector<std::string> coordinates;
    for (size_t column = 0; column != columns.size(); ++column)
      if (columns[column].first == index)
        coordinates.push_back(names[column]);
    std::string name = terms + "_" + std::to_string(index);
    os << indent << name << "_e";
    for (size_t q = 0; q != coordinates.size(); ++q)
      os << ", " << name << "_d" << q;
    os << " = md.evaluate @pullterm" << index << "(" << x << ", " << cell
       << getFieldValues(prefix) << getTimeValue(time);
    for (const std::string &coordinate : coordinates)
      os << ", " << coordinate;
    os << ")\n" << indent << "    request [energy";
    for (size_t q = 0; q != coordinates.size(); ++q)
      os << ", derivative(" << base + q << ")";
    os << "]\n" << indent << "    : (!vec, !md.cell" << getFieldTypes()
       << getTimeType();
    for (size_t q = 0; q != coordinates.size(); ++q)
      os << ", f64";
    os << ") -> (f64";
    for (size_t q = 0; q != coordinates.size(); ++q)
      os << ", f64";
    os << ")\n";
    auto store = [&](const std::string &value) {
      std::string place = name + "_s" + std::to_string(slot);
      os << indent << place << " = arith.constant " << slot++ << " : index\n"
         << indent << "memref.store " << value << ", " << terms << "["
         << place << "] : " << termType << "\n";
    };
    store(name + "_e");
    unsigned counter = 0;
    auto op = [&](StringRef operation,
                  std::initializer_list<std::string> operands) {
      std::string result = name + "_v" + std::to_string(counter++);
      os << indent << result << " = " << operation << " "
         << llvm::join(operands, ", ") << " : f64\n";
      return result;
    };
    auto derivative = [&](size_t q) { return name + "_d" + std::to_string(q); };
    if (coordinates.size() == 1) {
      store(op("arith.negf", {derivative(0)}));
      continue;
    }
    std::string radial = op("arith.divf", {derivative(0), coordinates[0]});
    std::array<std::string, 3> force;
    for (int c = 0; c != 3; ++c)
      force[c] = op("arith.negf",
                    {op("arith.addf",
                        {op("arith.mulf", {radial, coordinates[1 + c]}),
                         derivative(1 + c)})});
    std::string along = op("arith.mulf", {force[0], coordinates[1]});
    for (int c = 1; c != 3; ++c)
      along = op("arith.addf",
                 {along, op("arith.mulf", {force[c], coordinates[1 + c]})});
    store(op("arith.divf", {along, coordinates[0]}));
    for (int c = 0; c != 3; ++c)
      store(force[c]);
  }
  os << indent << values << "_cast = memref.cast " << values << " : " << type
     << " to memref<?xf64>\n"
     << indent << terms << "_cast = memref.cast " << terms << " : "
     << termType << " to memref<?xf64>\n"
     << indent << "func.call @mdrtWritePull(" << step << ", " << values
     << "_cast, " << terms << "_cast) : (i64, memref<?xf64>, memref<?xf64>) "
        "-> ()\n";
}

void Builder::emitFreeEnergyOutput(StringRef indent, StringRef x,
                                   StringRef cell, StringRef prefix,
                                   StringRef step, StringRef time) {
  if (control.freeEnergyFile.empty())
    return;
  // `@alchemical` at the state of the run with its derivatives in each
  // component of λ, and its energy at every other state; the host adds the
  // constant terms (D161). The values: the derivatives, then the energies
  // of the states in order. With one state its energy differs from itself
  // only, so the states are not evaluated (D190).
  const Control::FreeEnergy &energy = control.freeEnergy;
  size_t components = energy.lambdas.size(), states = energy.getNumStates();
  if (states == 1)
    states = 0;
  std::string name = ("%fe" + step.drop_front()).str();
  std::string type =
      "memref<" + std::to_string(components + states) + "xf64>";
  os << indent << name << " = memref.alloca() : " << type << "\n";
  unsigned base = getNumPotentialArguments();
  auto lambdas = [&](size_t k) {
    std::string text;
    for (size_t c = 0; c != components; ++c) {
      std::string value = name + "_l" + std::to_string(k) + "_" +
                          std::to_string(c);
      os << indent << value << " = arith.constant "
         << formatReal(energy.lambdas[c].second[k]) << " : f64\n";
      text += ", " + value;
    }
    return text;
  };
  std::string types = "(!vec, !md.cell" + getFieldTypes() + getTimeType();
  for (size_t c = 0; c != components; ++c)
    types += ", f64";
  types += ")";
  auto store = [&](const std::string &value, size_t slot) {
    std::string place = name + "_s" + std::to_string(slot);
    os << indent << place << " = arith.constant " << slot << " : index\n"
       << indent << "memref.store " << value << ", " << name << "["
       << place << "] : " << type << "\n";
  };
  // The derivatives at the state of the run, then the energy at every
  // state in a loop, whose kernels are compiled once, with the components
  // of each state from a table.
  size_t state = static_cast<size_t>(energy.state);
  std::string arguments = lambdas(state);
  os << indent;
  for (size_t c = 0; c != components; ++c)
    os << (c ? ", " : "") << name << "_d" << c;
  os << " = md.evaluate @alchemical(" << x << ", " << cell
     << getFieldValues(prefix) << getTimeValue(time) << arguments << ")\n"
     << indent << "    request [";
  for (size_t c = 0; c != components; ++c)
    os << (c ? ", " : "") << "derivative(" << base + c << ")";
  os << "]\n" << indent << "    : " << types << " -> (";
  for (size_t c = 0; c != components; ++c)
    os << (c ? ", " : "") << "f64";
  os << ")\n";
  for (size_t c = 0; c != components; ++c)
    store(name + "_d" + std::to_string(c), c);
  if (states == 0) {
    os << indent << name << "_cast = memref.cast " << name << " : " << type
       << " to memref<?xf64>\n"
       << indent << "func.call @mdrtWriteFreeEnergy(" << step << ", " << name
       << "_cast) : (i64, memref<?xf64>) -> ()\n";
    return;
  }
  std::string table = name + "_table";
  std::string tableType = "memref<" + std::to_string(states) + "x" +
                          std::to_string(components) + "xf64>";
  os << indent << table << " = memref.alloca() : " << tableType << "\n";
  for (size_t k = 0; k != states; ++k)
    for (size_t c = 0; c != components; ++c) {
      std::string at = name + "_t" + std::to_string(k) + "_" +
                       std::to_string(c);
      os << indent << at << "v = arith.constant "
         << formatReal(energy.lambdas[c].second[k]) << " : f64\n"
         << indent << at << "k = arith.constant " << k << " : index\n"
         << indent << at << "c = arith.constant " << c << " : index\n"
         << indent << "memref.store " << at << "v, " << table << "[" << at
         << "k, " << at << "c] : " << tableType << "\n";
    }
  std::string inner = (indent + "  ").str();
  os << indent << name << "_count = arith.constant " << states
     << " : index\n"
     << indent << name << "_first = arith.constant " << components
     << " : index\n"
     << indent << "scf.for " << name << "_k = %c0 to " << name
     << "_count step %c1 {\n";
  std::string loaded;
  for (size_t c = 0; c != components; ++c) {
    std::string value = name + "_lk" + std::to_string(c);
    os << inner << name << "_ci" << c << " = arith.constant " << c
       << " : index\n"
       << inner << value << " = memref.load " << table << "[" << name
       << "_k, " << name << "_ci" << c << "] : " << tableType << "\n";
    loaded += ", " + value;
  }
  os << inner << name << "_e = md.evaluate @alchemical(" << x << ", " << cell
     << getFieldValues(prefix) << getTimeValue(time) << loaded << ")\n"
     << inner << "    request [energy] : " << types << " -> f64\n"
     << inner << name << "_slot = arith.addi " << name << "_first, " << name
     << "_k : index\n"
     << inner << "memref.store " << name << "_e, " << name << "[" << name
     << "_slot] : " << type << "\n"
     << indent << "}\n";
  os << indent << name << "_cast = memref.cast " << name << " : " << type
     << " to memref<?xf64>\n"
     << indent << "func.call @mdrtWriteFreeEnergy(" << step << ", " << name
     << "_cast) : (i64, memref<?xf64>) -> ()\n";
}

std::vector<Builder::ObservedTerm> Builder::getObservedTerms() const {
  // The terms in the order of their first column; Control has checked the
  // names and the constants.
  std::vector<ObservedTerm> terms;
  if (control.observablesFile.empty() || !system.topology)
    return terms;
  for (auto [column, observable] : llvm::enumerate(control.observables)) {
    auto it = llvm::find_if(terms, [&](const ObservedTerm &term) {
      return term.name == observable.term;
    });
    if (it == terms.end()) {
      ObservedTerm term;
      term.name = observable.term;
      for (auto [k, pair] : llvm::enumerate(control.pairs))
        if (pair.name == term.name)
          term.kind = PairTerms, term.index = static_cast<int>(k);
      for (auto [k, tuple] : llvm::enumerate(system.topology->tupleTerms))
        if (tuple.name == term.name)
          term.kind = TupleTerms, term.index = static_cast<int>(k);
      for (auto [k, external] :
           llvm::enumerate(system.topology->externalTerms))
        if (external.name == term.name)
          term.kind = ExternalTerms, term.index = static_cast<int>(k);
      terms.push_back(std::move(term));
      it = std::prev(terms.end());
    }
    int place = -1;
    if (!observable.constant.empty()) {
      place = static_cast<int>(it->constants.size());
      it->constants.push_back(observable.constant);
    }
    it->columns.push_back({place, column});
  }
  return terms;
}

void Builder::emitObservedPotentials() {
  // Each term alone, with its observed constants as arguments, whose
  // derivatives the differentiation of parameters gives as it gives
  // dH/dλ (D161): exactly 0 where the energy provably does not depend on
  // them, an error where a dependence has no rule.
  for (auto [k, term] : llvm::enumerate(getObservedTerms())) {
    observedTerm = term.name;
    observedConstants = term.constants;
    emitTopologyPotential("observe" + std::to_string(k), term.kind,
                          term.kind == TupleTerms ? term.index : -1,
                          term.kind == PairTerms ? term.index : -1,
                          term.kind == ExternalTerms ? term.index : -1);
    observedTerm.clear();
    observedConstants.clear();
  }
}

void Builder::emitObservablesOutput(StringRef indent, StringRef x,
                                    StringRef cell, StringRef prefix,
                                    StringRef step, StringRef time) {
  std::vector<ObservedTerm> terms = getObservedTerms();
  if (terms.empty())
    return;
  // The values in the order of `observe`, in kJ/mol and kJ/mol per unit of
  // the constant; the writer converts them.
  std::string name = ("%ob" + step.drop_front()).str();
  std::string type =
      "memref<" + std::to_string(control.observables.size()) + "xf64>";
  os << indent << name << " = memref.alloca() : " << type << "\n";
  unsigned base = getNumPotentialArguments();
  for (auto [k, term] : llvm::enumerate(terms)) {
    std::string at = name + "_" + std::to_string(k);
    std::string arguments, types = "(!vec, !md.cell" + getFieldTypes() +
                                   getTimeType();
    // The constants at their values, which the term holds for each of its
    // tuples, particles, or pairs alike.
    for (auto [c, constant] : llvm::enumerate(term.constants)) {
      std::string value = at + "_v" + std::to_string(c);
      os << indent << value << " = arith.constant "
         << formatReal(getObservedValue(term, constant)) << " : f64\n";
      arguments += ", " + value;
      types += ", f64";
    }
    types += ")";
    os << indent;
    for (auto [i, column] : llvm::enumerate(term.columns))
      os << (i ? ", " : "") << at << "_r" << i;
    os << " = md.evaluate @observe" << k << "(" << x << ", " << cell
       << getFieldValues(prefix) << getTimeValue(time) << arguments << ")\n"
       << indent << "    request [";
    for (auto [i, column] : llvm::enumerate(term.columns)) {
      os << (i ? ", " : "");
      if (column.first < 0)
        os << "energy";
      else
        os << "derivative(" << base + column.first << ")";
    }
    os << "]\n" << indent << "    : " << types << " -> (";
    for (size_t i = 0; i != term.columns.size(); ++i)
      os << (i ? ", " : "") << "f64";
    os << ")\n";
    for (auto [i, column] : llvm::enumerate(term.columns)) {
      std::string place = at + "_s" + std::to_string(i);
      os << indent << place << " = arith.constant " << column.second
         << " : index\n"
         << indent << "memref.store " << at << "_r" << i << ", " << name
         << "[" << place << "] : " << type << "\n";
    }
  }
  os << indent << name << "_cast = memref.cast " << name << " : " << type
     << " to memref<?xf64>\n"
     << indent << "func.call @mdrtWriteObservables(" << step << ", " << name
     << "_cast) : (i64, memref<?xf64>) -> ()\n";
}

double Builder::getDebyeKappa() const {
  // κ² = 2 N_A e² c / (ε0 ε_r k_B T) for a 1:1 salt of c mol/L, with the
  // constants of CODATA 2018 in SI units, c in mol/m³; κ in nm⁻¹.
  if (control.saltConcentration <= 0.0)
    return 0.0;
  const double avogadro = 6.02214076e23, charge = 1.602176634e-19,
               permittivity = 8.8541878128e-12, boltzmann = 1.380649e-23;
  double concentration = control.saltConcentration * 1000.0;
  double kappa2 = 2.0 * avogadro * charge * charge * concentration /
                  (permittivity * control.solventDielectric * boltzmann *
                   control.temperature);
  return 0.73 * std::sqrt(kappa2) * 1e-9;
}

void Builder::emitBorn(unsigned terms,
                       llvm::function_ref<void(StringRef)> add) {
  // Every pair within the cutoff, those that the topology excludes as
  // well.
  double cutoff = control.cutoffDistance * units::length;
  os << "  %ngb = md.neighborhood %x, %cell cutoff(" << formatReal(cutoff)
     << ") : !vec -> !pairs\n";
  // With a cutoff R of the radii (D152), the descreening reaches R and the
  // largest screened radius beyond it, which System checked is within the
  // cutoff.
  double radiusCutoff = control.bornRadiusCutoff * units::length;
  std::string reach = "%ngb";
  if (radiusCutoff > 0.0) {
    double largest = 0.0;
    for (size_t i = 0, e = system.topology->bornRadii.size(); i != e; ++i)
      largest = std::max(largest, system.topology->bornScreens[i] *
                                      (system.topology->bornRadii[i] - 0.009));
    reach = "%ngb_radii";
    os << "  %ngb_radii = md.neighborhood %x, %cell cutoff("
       << formatReal(std::min(radiusCutoff + largest, cutoff))
       << ") : !vec -> !pairs\n";
  }
  // The integral of the descreening of i by the sphere of j of the radius
  // s_j [Hawkins1996]: with L = max(ρ̃_i, |r − s_j|) and U = r + s_j,
  //   ½ (1/L − 1/U + r/4 (1/U² − 1/L²) + ln(L/U) / (2r) + s_j²/(4r) (1/L² − 1/U²)),
  // and 1/ρ̃_i − 1/L more where i lies within the sphere of j, 0 where the
  // sphere does not reach it, L ≥ U. A cutoff R of the radii takes the
  // integral over the shells within R alone, U = min(r + s_j, R), so that
  // the radii do not jump where a sphere crosses it: the smooth cutoff
  // `rgbmax` of Amber's pmemd (gb_ene.F90), written here from the integral
  // rather than its series (D152).
  os << "  %gb_integral = md.gather_relation " << reach << ", %x, %cell gather("
        "%p_gb_offset, %p_gb_scaled : !real, !real)\n"
     << "      exchange(none) {\n"
     << "  ^bb0(%r: f64, %d: vector<3xf64>, %ri: f64, %rj: f64, %si: f64, "
        "%sj: f64):\n"
     << "    %zero = arith.constant 0.0 : f64\n"
     << "    %one = arith.constant 1.0 : f64\n"
     << "    %two = arith.constant 2.0 : f64\n"
     << "    %quarter = arith.constant 0.25 : f64\n"
     << "    %half = arith.constant 0.5 : f64\n"
     << "    %upper" << (radiusCutoff > 0.0 ? "_sphere" : "")
     << " = arith.addf %r, %sj : f64\n";
  if (radiusCutoff > 0.0)
    os << "    %rgb = arith.constant " << formatReal(radiusCutoff)
       << " : f64\n"
       << "    %upper = arith.minimumf %upper_sphere, %rgb : f64\n";
  os << "    %gap = arith.subf %r, %sj : f64\n"
     << "    %agap = math.absf %gap : f64\n"
     << "    %lower = arith.maximumf %ri, %agap : f64\n"
     << "    %l = arith.divf %one, %lower : f64\n"
     << "    %u = arith.divf %one, %upper : f64\n"
     << "    %l2 = arith.mulf %l, %l : f64\n"
     << "    %u2 = arith.mulf %u, %u : f64\n"
     << "    %t0 = arith.subf %l, %u : f64\n"
     << "    %du = arith.subf %u2, %l2 : f64\n"
     << "    %qr = arith.mulf %quarter, %r : f64\n"
     << "    %t1 = arith.mulf %qr, %du : f64\n"
     << "    %ratio = arith.divf %u, %l : f64\n"
     << "    %log = math.log %ratio : f64\n"
     << "    %hr = arith.divf %half, %r : f64\n"
     << "    %t2 = arith.mulf %hr, %log : f64\n"
     << "    %sj2 = arith.mulf %sj, %sj : f64\n"
     << "    %qs = arith.mulf %quarter, %sj2 : f64\n"
     << "    %qsr = arith.divf %qs, %r : f64\n"
     << "    %dl = arith.subf %l2, %u2 : f64\n"
     << "    %t3 = arith.mulf %qsr, %dl : f64\n"
     << "    %a0 = arith.addf %t0, %t1 : f64\n"
     << "    %a1 = arith.addf %a0, %t2 : f64\n"
     << "    %a2 = arith.addf %a1, %t3 : f64\n"
     << "    %inner = arith.subf %sj, %r : f64\n"
     << "    %within = arith.cmpf olt, %ri, %inner : f64\n"
     << "    %iri = arith.divf %one, %ri : f64\n"
     << "    %e0 = arith.subf %iri, %l : f64\n"
     << "    %e1 = arith.mulf %two, %e0 : f64\n"
     << "    %a3 = arith.addf %a2, %e1 : f64\n"
     << "    %a4 = arith.select %within, %a3, %a2 : f64\n"
     << "    %reach = arith.cmpf olt, %lower, %upper : f64\n"
     << "    %k = arith.select %reach, %a4, %zero : f64\n"
     << "    md.yield %k : f64\n"
     << "  } : !pairs, !vec -> !real\n";
  // The Born radii of Hawkins, Cramer, and Truhlar [Hawkins1996],
  // B = 1 / (1/ρ̃ − I/2) (the factor ½ of the integral).
  if (control.implicitSolvent == Control::ImplicitSolvent::HCT)
    os << "  %gb_born = md.map_particles gather(%gb_integral, %p_gb_offset "
          ": !real, !real) {\n"
       << "  ^bb0(%i: f64, %ri: f64):\n"
       << "    %half = arith.constant 0.5 : f64\n"
       << "    %one = arith.constant 1.0 : f64\n"
       << "    %hi = arith.mulf %half, %i : f64\n"
       << "    %iri = arith.divf %one, %ri : f64\n"
       << "    %den = arith.subf %iri, %hi : f64\n"
       << "    %bi = arith.divf %one, %den : f64\n"
       << "    md.yield %bi : f64\n"
       << "  } : !real\n";
  // Those of Onufriev, Bashford, and Case [Onufriev2004]: with ψ = I ρ̃ / 2,
  // B = 1 / (1/ρ̃ − tanh(α ψ − β ψ² + γ ψ³) / ρ).
  bool first = control.implicitSolvent == Control::ImplicitSolvent::OBC1;
  double alpha = first ? 0.8 : 1.0, beta = first ? 0.0 : 0.8,
         gamma = first ? 2.909125 : 4.85;
  if (control.implicitSolvent != Control::ImplicitSolvent::HCT)
    os << "  %gb_born = md.map_particles gather(%gb_integral, %p_gb_offset, "
          "%p_gb_radius : !real, !real, !real) {\n"
       << "  ^bb0(%i: f64, %ri: f64, %r0: f64):\n"
       << "    %half = arith.constant 0.5 : f64\n"
       << "    %one = arith.constant 1.0 : f64\n"
       << "    %alpha = arith.constant " << formatReal(alpha) << " : f64\n"
       << "    %beta = arith.constant " << formatReal(beta) << " : f64\n"
       << "    %gamma = arith.constant " << formatReal(gamma) << " : f64\n"
       << "    %hi = arith.mulf %half, %i : f64\n"
       << "    %psi = arith.mulf %hi, %ri : f64\n"
       << "    %psi2 = arith.mulf %psi, %psi : f64\n"
       << "    %psi3 = arith.mulf %psi2, %psi : f64\n"
       << "    %a = arith.mulf %alpha, %psi : f64\n"
       << "    %b = arith.mulf %beta, %psi2 : f64\n"
       << "    %c = arith.mulf %gamma, %psi3 : f64\n"
       << "    %s0 = arith.subf %a, %b : f64\n"
       << "    %s1 = arith.addf %s0, %c : f64\n"
       << "    %t = math.tanh %s1 : f64\n"
       << "    %tr = arith.divf %t, %r0 : f64\n"
       << "    %iri = arith.divf %one, %ri : f64\n"
       << "    %den = arith.subf %iri, %tr : f64\n"
       << "    %bi = arith.divf %one, %den : f64\n"
       << "    md.yield %bi : f64\n"
       << "  } : !real\n";
  if (terms & GeneralizedBorn) {
    // −τ f q_i q_j / f_GB over the pairs and −τ f q_i² / (2 B_i) for each
    // particle, f_GB = sqrt(r² + B_i B_j exp(−r² / (4 B_i B_j))) and
    // τ = 1/ε_solute − 1/ε_solvent. A salt screens the solvent with the
    // Debye length 1/κ, τ = 1/ε_solute − exp(−κ f_GB)/ε_solvent, with κ
    // scaled by 0.73 for the layer around the solute that excludes the ions
    // [Srinivasan1999] (D152); f_GB is B_i for a particle.
    double tau = 1.0 / control.soluteDielectric -
                 1.0 / control.solventDielectric;
    double kappa = getDebyeKappa();
    // Emits `%ct`, f τ with the factor `factor` of f, for the distance
    // `distance`.
    auto emitScreening = [&](double factor, StringRef distance) {
      if (kappa == 0.0) {
        os << "    %ct = arith.constant "
           << formatReal(-factor * tau * coulombInternal) << " : f64\n";
        return;
      }
      os << "    %cf = arith.constant " << formatReal(-factor * coulombInternal)
         << " : f64\n"
         << "    %ein = arith.constant "
         << formatReal(1.0 / control.soluteDielectric) << " : f64\n"
         << "    %eout = arith.constant "
         << formatReal(1.0 / control.solventDielectric) << " : f64\n"
         << "    %nkappa = arith.constant " << formatReal(-kappa) << " : f64\n"
         << "    %kd = arith.mulf %nkappa, " << distance << " : f64\n"
         << "    %ekd = math.exp %kd : f64\n"
         << "    %screened = arith.mulf %ekd, %eout : f64\n"
         << "    %tau = arith.subf %ein, %screened : f64\n"
         << "    %ct = arith.mulf %cf, %tau : f64\n";
    };
    os << "  %u_gb_pairs = md.sum_relation %ngb, %x, %cell gather(%p_q, "
          "%gb_born : !real, !real)\n"
       << "      exchange(symmetric) {\n"
       << "  ^bb0(%r: f64, %d: vector<3xf64>, %qi: f64, %qj: f64, %bi: f64, "
          "%bj: f64):\n"
       << "    %quarter = arith.constant 0.25 : f64\n"
       << "    %bb = arith.mulf %bi, %bj : f64\n"
       << "    %r2 = arith.mulf %r, %r : f64\n"
       << "    %bb4 = arith.divf %quarter, %bb : f64\n"
       << "    %x0 = arith.mulf %r2, %bb4 : f64\n"
       << "    %nx = arith.negf %x0 : f64\n"
       << "    %ex = math.exp %nx : f64\n"
       << "    %be = arith.mulf %bb, %ex : f64\n"
       << "    %s = arith.addf %r2, %be : f64\n"
       << "    %fgb = math.sqrt %s : f64\n";
    emitScreening(1.0, "%fgb");
    os << "    %qq = arith.mulf %qi, %qj : f64\n"
       << "    %cq = arith.mulf %ct, %qq : f64\n"
       << "    %e = arith.divf %cq, %fgb : f64\n"
       << "    md.yield %e : f64\n"
       << "  } : !pairs, !vec -> f64\n"
       << "  %u_gb_self = md.sum_particles gather(%p_q, %gb_born : !real, "
          "!real) {\n"
       << "  ^bb0(%qi: f64, %bi: f64):\n";
    emitScreening(0.5, "%bi");
    os << "    %qq = arith.mulf %qi, %qi : f64\n"
       << "    %cq = arith.mulf %ct, %qq : f64\n"
       << "    %e = arith.divf %cq, %bi : f64\n"
       << "    md.yield %e : f64\n"
       << "  } : f64\n"
       << "  %u_born = arith.addf %u_gb_pairs, %u_gb_self : f64\n";
    add("born");
  }
  if ((terms & Surface) && control.surfaceAreaEnergy > 0.0) {
    // The nonpolar term of Schaefer et al., 4π γ (ρ + 0.14 nm)² (ρ/B)⁶,
    // with γ from kcal/(mol Å²) in kJ/(mol nm²).
    double tension = control.surfaceAreaEnergy * units::energy /
                     (units::length * units::length);
    os << "  %u_surface = md.sum_particles gather(%p_gb_radius, %gb_born : "
          "!real, !real) {\n"
       << "  ^bb0(%r0: f64, %bi: f64):\n"
       << "    %c = arith.constant " << formatReal(4.0 * M_PI * tension)
       << " : f64\n"
       << "    %probe = arith.constant 0.14 : f64\n"
       << "    %six = arith.constant 6 : i32\n"
       << "    %rp = arith.addf %r0, %probe : f64\n"
       << "    %rp2 = arith.mulf %rp, %rp : f64\n"
       << "    %ratio = arith.divf %r0, %bi : f64\n"
       << "    %r6 = math.fpowi %ratio, %six : f64, i32\n"
       << "    %a = arith.mulf %c, %rp2 : f64\n"
       << "    %e = arith.mulf %a, %r6 : f64\n"
       << "    md.yield %e : f64\n"
       << "  } : f64\n";
    add("surface");
  }
}

void Builder::emitExternalTerm(size_t index, const ExternalTerm &term,
                                llvm::function_ref<void(StringRef)> add) {
  // Σ_i k(x_i) over the particles of the term, the custom external force of
  // OpenMM [Eastman2017] (D148), the flag `ext<k>` 1 for
  // them and 0 for the others, which the kernel selects on: the positions
  // in Å, the energy in kcal/mol. The positions are never wrapped (D74), so
  // that the term is continuous along a trajectory.
  Expression expression = llvm::cantFail(
      Expression::parse(term.expression, control.functions));
  std::vector<std::string> names = expression.getNames();
  bool charges = llvm::is_contained(names, "q");
  std::string flag = "ext" + std::to_string(index);
  std::string prefix = "%xe" + std::to_string(index) + "_";
  // In the frame of the cell (D154), the positions scaled by the edges of
  // the cell of the input over those of the cell.
  bool frame = term.scaling == ExternalTerm::Scaling::Cell;
  if (frame) {
    os << "  " << prefix << "edges = md_exec.cell_edges %cell : vector<3xf64>\n"
       << "  " << prefix << "input = arith.constant dense<[";
    for (int k = 0; k != 3; ++k)
      os << (k ? ", " : "")
         << formatReal(system.inputBox[k] > 0.0 ? system.inputBox[k]
                                                 : system.box[k]);
    os << "]> : vector<3xf64>\n"
       << "  " << prefix << "frame = arith.divf " << prefix << "input, "
       << prefix << "edges : vector<3xf64>\n";
  }
  os << "  %u_external_" << term.name << " = md.sum_particles gather(%x, %p_"
     << flag;
  if (charges)
    os << ", %p_q";
  for (const auto &parameter : term.parameters)
    os << ", %p_" << flag << "_" << parameter.first;
  os << " : !vec, !real";
  if (charges)
    os << ", !real";
  for (size_t k = 0; k != term.parameters.size(); ++k)
    os << ", !real";
  os << ") {\n  ^bb0(" << prefix << "pos: vector<3xf64>, " << prefix
     << "on: f64";
  if (charges)
    os << ", " << prefix << "q: f64";
  for (size_t k = 0; k != term.parameters.size(); ++k)
    os << ", " << prefix << "p" << k << ": f64";
  os << "):\n";
  llvm::StringMap<std::string> values;
  os << "    " << prefix << "a = arith.constant "
     << formatReal(1.0 / units::length) << " : f64\n";
  std::string position = prefix + "pos";
  if (frame) {
    os << "    " << prefix << "posf = arith.mulf " << prefix << "pos, "
       << prefix << "frame : vector<3xf64>\n";
    position = prefix + "posf";
  }
  for (int c = 0; c != 3; ++c) {
    std::string component = prefix + "xyz"[c];
    os << "    " << component << "n = vector.extract " << position << "["
       << c << "] : f64 from vector<3xf64>\n"
       << "    " << component << " = arith.mulf " << component << "n, "
       << prefix << "a : f64\n";
    values[std::string(1, "xyz"[c])] = component;
  }
  if (charges)
    values["q"] = prefix + "q";
  for (auto [k, parameter] : llvm::enumerate(term.parameters))
    values[parameter.first] = prefix + "p" + std::to_string(k);
  for (auto [k, constant] : llvm::enumerate(term.constants)) {
    std::string name = prefix + "c" + std::to_string(k);
    os << "    " << name << " = arith.constant "
       << formatReal(constant.second) << " : f64\n";
    values[constant.first] = name;
  }
  if (control.usesTime)
    values["t"] = "%time";
  bindLambdas(values);
  bindObserved(term.name, values);
  std::string energy = expression.emit(os, values, prefix + "e", "    ", term.name);
  os << "    " << prefix << "kj = arith.constant " << formatReal(units::energy)
     << " : f64\n"
     << "    " << prefix << "u = arith.mulf " << energy << ", " << prefix
     << "kj : f64\n"
     << "    " << prefix << "zero = arith.constant 0.0 : f64\n"
     << "    " << prefix << "in = arith.cmpf one, " << prefix << "on, "
     << prefix << "zero : f64\n"
     << "    " << prefix << "k = arith.select " << prefix << "in, " << prefix
     << "u, " << prefix << "zero : f64\n"
     << "    md.yield " << prefix << "k : f64\n"
     << "  } : f64\n";
  add("external_" + term.name);
}

void Builder::emitTopologyPotential(StringRef name, unsigned terms,
                                    int tupleTerm, int pairTerm,
                                    int externalTerm) {
  double cutoff = control.cutoffDistance * units::length;
  auto has = [&](StringRef set) {
    return llvm::any_of(program.tupleSets, [&](const Program::TupleSet &s) {
      return s.name == set;
    });
  };
  // 4ε (σ¹² Φ₁₂(r) − σ⁶ Φ₆(r)), with Φₙ = r⁻ⁿ truncated as `modifier`
  // says, with rc the cutoff and rs the switching distance:
  //
  //   none             r⁻ⁿ
  //   potential shift  r⁻ⁿ − rc⁻ⁿ
  //   force switch     r⁻ⁿ − Cₙ − [r > rs] (Aₙ/3 (r − rs)³ + Bₙ/4 (r − rs)⁴),
  //                    a cubic polynomial added to the force, so that it and
  //                    its derivative vanish at rc [GromacsManual2025]
  //   power force      r⁻ⁿ − (rs rc)^(−n/2)               for r <= rs
  //   switch           kₙ (r^(−n/2) − rc^(−n/2))²          otherwise,
  //                    kₙ = rc^(n/2) / (rc^(n/2) − rs^(n/2)): the force of
  //                    each power times a switch linear in r^(n/2)
  //                    [Steinbach1994]
  //
  // σⁿ Φₙ is written with s = σ/r and g = σ, so that no power of r alone
  // leaves the range of f32.
  double from = control.switchDistance * units::length;
  // `distance` is r, or the soft-core distance of [free_energy] (D161).
  auto emitLennardJones = [&](StringRef sigma, StringRef epsilon,
                              StringRef result, Truncation modifier,
                              StringRef distance = "%r") {
    os << "    %c4 = arith.constant 4.0 : f64\n"
       << "    %e4 = arith.mulf %c4, " << epsilon << " : f64\n"
       << "    %sr = arith.divf " << sigma << ", " << distance << " : f64\n";
    if (modifier == Truncation::None ||
        modifier == Truncation::SquaredDistanceSwitch) {
      os << "    %i6 = arith.constant 6 : i32\n"
         << "    %s6 = math.fpowi %sr, %i6 : f64, i32\n"
         << "    %s12 = arith.mulf %s6, %s6 : f64\n"
         << "    %t = arith.subf %s12, %s6 : f64\n"
         << "    " << (modifier == Truncation::SquaredDistanceSwitch
                            ? StringRef("%lj_raw") : result)
         << " = arith.mulf %e4, %t : f64\n";
      if (modifier == Truncation::SquaredDistanceSwitch) {
        // The cubic switching potential in r² [Brooks1983]. Write it
        // as (1-y)²(1+2y), y=(r²-rs²)/(rc²-rs²), to avoid cancellation
        // near rc. Clamping y gives the constant branches and a zero
        // derivative at both ends. AD differentiates the full product.
        os << "    %sw_rs2 = arith.constant " << formatReal(from * from)
           << " : f64\n"
           << "    %sw_width_inv = arith.constant "
           << formatReal(1.0 / (cutoff * cutoff - from * from)) << " : f64\n"
           << "    %sw_zero = arith.constant 0.0 : f64\n"
           << "    %sw_one = arith.constant 1.0 : f64\n"
           << "    %sw_two = arith.constant 2.0 : f64\n"
           << "    %sw_r2 = arith.mulf %r, %r : f64\n"
           << "    %sw_dr2 = arith.subf %sw_r2, %sw_rs2 : f64\n"
           << "    %sw_y_raw = arith.mulf %sw_dr2, %sw_width_inv : f64\n"
           << "    %sw_below = arith.cmpf olt, %sw_y_raw, %sw_zero : f64\n"
           << "    %sw_y_lo = arith.select %sw_below, %sw_zero, %sw_y_raw : f64\n"
           << "    %sw_above = arith.cmpf ogt, %sw_y_lo, %sw_one : f64\n"
           << "    %sw_y = arith.select %sw_above, %sw_one, %sw_y_lo : f64\n"
           << "    %sw_complement = arith.subf %sw_one, %sw_y : f64\n"
           << "    %sw_square = arith.mulf %sw_complement, %sw_complement : f64\n"
           << "    %sw_twice = arith.mulf %sw_two, %sw_y : f64\n"
           << "    %sw_factor = arith.addf %sw_one, %sw_twice : f64\n"
           << "    %sw_s = arith.mulf %sw_square, %sw_factor : f64\n"
           << "    " << result << " = arith.mulf %lj_raw, %sw_s : f64\n";
      }
      return;
    }
    os << "    %i3 = arith.constant 3 : i32\n"
       << "    %s3 = math.fpowi %sr, %i3 : f64, i32\n"
       << "    %s6 = arith.mulf %s3, %s3 : f64\n"
       << "    %s12 = arith.mulf %s6, %s6 : f64\n"
       << "    %g3 = math.fpowi " << sigma << ", %i3 : f64, i32\n"
       << "    %g6 = arith.mulf %g3, %g3 : f64\n"
       << "    %g12 = arith.mulf %g6, %g6 : f64\n";
    auto constant = [&](StringRef name, double value) {
      os << "    %" << name << " = arith.constant " << formatReal(value)
         << " : f64\n";
    };
    // a = σ¹² Φ₁₂ and b = σ⁶ Φ₆, from what each power loses, g¹² c₁₂ and
    // g⁶ c₆.
    auto emitSubtract = [&](StringRef c12, StringRef c6) {
      os << "    %gc12 = arith.mulf %g12, " << c12 << " : f64\n"
         << "    %a = arith.subf %s12, %gc12 : f64\n"
         << "    %gc6 = arith.mulf %g6, " << c6 << " : f64\n"
         << "    %b = arith.subf %s6, %gc6 : f64\n";
    };
    std::string a = "%a", b = "%b";
    if (modifier == Truncation::Shift) {
      constant("rc12", std::pow(cutoff, -12.0));
      constant("rc6", std::pow(cutoff, -6.0));
      emitSubtract("%rc12", "%rc6");
    } else if (modifier == Truncation::ForceSwitch) {
      double width = cutoff - from;
      os << "    %rs = arith.constant " << formatReal(from) << " : f64\n"
         << "    %zero = arith.constant 0.0 : f64\n"
         << "    %dr = arith.subf " << distance << ", %rs : f64\n"
         << "    %beyond = arith.cmpf ogt, " << distance << ", %rs : f64\n"
         << "    %u = arith.select %beyond, %dr, %zero : f64\n"
         << "    %u2 = arith.mulf %u, %u : f64\n"
         << "    %u3 = arith.mulf %u2, %u : f64\n";
      for (int n : {12, 6}) {
        double an = -n * ((n + 4) * cutoff - (n + 1) * from) /
                    (std::pow(cutoff, n + 2) * width * width);
        double bn = n * ((n + 3) * cutoff - (n + 1) * from) /
                    (std::pow(cutoff, n + 2) * width * width * width);
        double cn = std::pow(cutoff, -n) - an / 3.0 * std::pow(width, 3) -
                    bn / 4.0 * std::pow(width, 4);
        std::string k = std::to_string(n);
        constant("a" + k, an / 3.0);
        constant("b" + k, bn / 4.0);
        constant("c" + k, cn);
        // Cₙ + u³ (Aₙ/3 + (Bₙ/4) u)
        os << "    %pb" << k << " = arith.mulf %b" << k << ", %u : f64\n"
           << "    %pa" << k << " = arith.addf %a" << k << ", %pb" << k
           << " : f64\n"
           << "    %pu" << k << " = arith.mulf %u3, %pa" << k << " : f64\n"
           << "    %p" << k << " = arith.addf %c" << k << ", %pu" << k
           << " : f64\n";
      }
      emitSubtract("%p12", "%p6");
    } else {
      // Below rs the shifted powers; above, the switched squares.
      constant("in12", std::pow(from * cutoff, -6.0));
      constant("in6", std::pow(from * cutoff, -3.0));
      emitSubtract("%in12", "%in6");
      constant("k12", std::pow(cutoff, 6.0) /
                          (std::pow(cutoff, 6.0) - std::pow(from, 6.0)));
      constant("k6", std::pow(cutoff, 3.0) /
                         (std::pow(cutoff, 3.0) - std::pow(from, 3.0)));
      constant("rc6", std::pow(cutoff, -6.0));
      constant("rc3", std::pow(cutoff, -3.0));
      os << "    %gr6 = arith.mulf %g6, %rc6 : f64\n"
         << "    %h6 = arith.subf %s6, %gr6 : f64\n"
         << "    %h6s = arith.mulf %h6, %h6 : f64\n"
         << "    %a_out = arith.mulf %k12, %h6s : f64\n"
         << "    %gr3 = arith.mulf %g3, %rc3 : f64\n"
         << "    %h3 = arith.subf %s3, %gr3 : f64\n"
         << "    %h3s = arith.mulf %h3, %h3 : f64\n"
         << "    %b_out = arith.mulf %k6, %h3s : f64\n"
         << "    %rs = arith.constant " << formatReal(from) << " : f64\n"
         << "    %inside = arith.cmpf ole, " << distance << ", %rs : f64\n"
         << "    %a_sel = arith.select %inside, %a, %a_out : f64\n"
         << "    %b_sel = arith.select %inside, %b, %b_out : f64\n";
      a = "%a_sel";
      b = "%b_sel";
    }
    os << "    %t = arith.subf " << a << ", " << b << " : f64\n"
       << "    " << result << " = arith.mulf %e4, %t : f64\n";
  };

  os << "md.potential @" << name << "(%x: !vec, %cell: !md.cell"
     << getFieldParameters() << getTimeParameter() << getLambdaParameters()
     << getObservedParameters() << ") -> f64 {\n";
  emitLambdaConstants("  ");
  // [free_energy] (D161): `@alchemical` has only what depends on λ.
  bool alchemicalOnly = terms & Alchemical;
  if (alchemicalOnly) {
    unsigned kept = LennardJones | Coulomb | CoulombExcluded |
                    CoulombWithin | CoulombReciprocal | TupleTerms |
                    PairTerms | ExternalTerms;
    if (!decouples())
      kept &= ~(LennardJones | Coulomb | CoulombExcluded | CoulombWithin |
                CoulombReciprocal);
    terms &= kept;
  }
  // Which λ the kernels of the pairs take: as arguments, or as constants
  // of the state of the run that are not 0.
  bool scalesCoulomb =
      decouples() && (lambdaArguments || getLambda("coulomb") != 0.0);
  bool scalesVdw = decouples() && (lambdaArguments || getLambda("vdw") != 0.0);
  std::string total;
  auto add = [&](StringRef term) {
    std::string value = ("%u_" + term).str();
    if (total.empty()) {
      total = value;
      return;
    }
    std::string sum = ("%total_" + term).str();
    os << "  " << sum << " = arith.addf " << total << ", " << value
       << " : f64\n";
    total = sum;
  };

  // Lennard-Jones and Coulomb, both cut at the cutoff with no shift, over
  // the pairs that are not excluded.
  bool lj = terms & LennardJones, coulomb = terms & Coulomb;
  bool pairTerms = (terms & PairTerms) && !control.pairs.empty();
  if (lj || coulomb || pairTerms) {
    os << "  %n = md.neighborhood %x, %cell cutoff(" << formatReal(cutoff)
       << ")";
    if (has("excluded"))
      os << " exclude(%r_excluded : !rel_excluded)";
    os << " : !vec -> !pairs\n";
  }
  // The pair terms given by expressions (D137), over the pairs of the
  // topology that are not excluded, truncated at the cutoff as the
  // Lennard-Jones is, in the units of the control file: r, sigma in Å, q
  // in e, epsilon and the energy in kcal/mol. The reader has tested that
  // the expression is symmetric in the two particles, which the kernel
  // then asserts; sigma and
  // epsilon those of the pair, with the pairs of types set apart, and
  // sigma1, epsilon1 those of the type of each particle with itself.
  for (auto [k, term] : llvm::enumerate(control.pairs)) {
    if (!pairTerms || (pairTerm >= 0 && static_cast<int>(k) != pairTerm))
      continue;
    if (alchemicalOnly && !usesLambda(term.expression))
      continue;
    bool grouped = k < system.pairGroups.size() && !system.pairGroups[k].empty();
    std::string g = std::to_string(k), set = "pair_" + term.name;
    Expression expression = llvm::cantFail(Expression::parse(term.expression, control.functions));
    const std::vector<std::string> &used = expression.getNames();
    auto uses = [&](StringRef name) { return llvm::is_contained(used, name); };
    // The parameters of each particle that the term takes (D165).
    std::vector<std::string> stems;
    for (const auto &[name, values] : system.particleParameters)
      if (uses(name + "1") || uses(name + "2"))
        stems.push_back(name);
    os << "  %u_" << set << " = md.sum_relation %n, %x, %cell gather(%p_type, "
       << "%p_q" << (grouped ? ", %p_pg" + g + "a, %p_pg" + g + "b" : "");
    for (const std::string &stem : stems)
      os << ", %p_pp_" << stem;
    os << " : !ids, !real" << (grouped ? ", !real, !real" : "");
    for (size_t n = 0; n != stems.size(); ++n)
      os << ", !real";
    os << ")\n"
       << "      exchange(symmetric, asserted)" << getTruncation(control)
       << " {\n"
       << "  ^bb0(%r: f64, %d: vector<3xf64>, %type_i: i32, %type_j: i32, "
       << "%q_i: f64, %q_j: f64"
       << (grouped ? ", %ga_i: f64, %ga_j: f64, %gb_i: f64, %gb_j: f64" : "");
    for (const std::string &stem : stems)
      os << ", %pp_" << stem << "_i: f64, %pp_" << stem << "_j: f64";
    os << "):\n"
       << "    %pt_angstrom = arith.constant " << formatReal(1.0 / units::length)
       << " : f64\n"
       << "    %pt_kcal = arith.constant " << formatReal(1.0 / units::energy)
       << " : f64\n"
       << "    %pt_r = arith.mulf %r, %pt_angstrom : f64\n";
    llvm::StringMap<std::string> values;
    values["r"] = "%pt_r";
    if (control.usesTime)
      values["t"] = "%time";
    bindLambdas(values);
    values["q1"] = "%q_i";
    values["q2"] = "%q_j";
    for (const std::string &stem : stems) {
      values[stem + "1"] = "%pp_" + stem + "_i";
      values[stem + "2"] = "%pp_" + stem + "_j";
    }
    auto lookup = [&](StringRef variable, StringRef table, StringRef a,
                      StringRef b, StringRef factor) {
      if (!uses(variable))
        return;
      std::string raw = ("%pt_" + variable + "_raw").str();
      std::string value = ("%pt_" + variable).str();
      os << "    " << raw << " = md.lookup %t_" << table << "[" << a << ", "
         << b << "] : !table, i32, i32 -> f64\n"
         << "    " << value << " = arith.mulf " << raw << ", " << factor
         << " : f64\n";
      values[variable] = value;
    };
    lookup("sigma", "lj_sigma", "%type_i", "%type_j", "%pt_angstrom");
    lookup("epsilon", "lj_epsilon", "%type_i", "%type_j", "%pt_kcal");
    lookup("sigma1", "lj_sigma", "%type_i", "%type_i", "%pt_angstrom");
    lookup("sigma2", "lj_sigma", "%type_j", "%type_j", "%pt_angstrom");
    lookup("epsilon1", "lj_epsilon", "%type_i", "%type_i", "%pt_kcal");
    lookup("epsilon2", "lj_epsilon", "%type_j", "%type_j", "%pt_kcal");
    if (uses("coulomb")) {
      os << "    %pt_coulomb = arith.constant " << formatReal(coulombConstant)
         << " : f64\n";
      values["coulomb"] = "%pt_coulomb";
    }
    for (auto [c, constant] : llvm::enumerate(term.constants)) {
      std::string value = "%pt_c" + std::to_string(c);
      os << "    " << value << " = arith.constant "
         << formatReal(constant.second) << " : f64\n";
      values[constant.first] = value;
    }
    bindObserved(term.name, values);
    std::string energy = expression.emit(os, values, "%pte", "    ", term.name);
    os << "    %pt_kj = arith.constant " << formatReal(units::energy)
       << " : f64\n"
       << "    %pt_e = arith.mulf " << energy << ", %pt_kj : f64\n";
    std::string value = "%pt_e";
    if (grouped) {
      // A pair of a particle of each group, counted once even if both are
      // in both.
      os << "    %pt_ab = arith.mulf %ga_i, %gb_j : f64\n"
         << "    %pt_ba = arith.mulf %ga_j, %gb_i : f64\n"
         << "    %pt_either = arith.addf %pt_ab, %pt_ba : f64\n"
         << "    %pt_one = arith.constant 1.0 : f64\n"
         << "    %pt_mask = arith.minimumf %pt_either, %pt_one : f64\n"
         << "    %pt_masked = arith.mulf %pt_e, %pt_mask : f64\n";
      value = "%pt_masked";
    }
    os << "    md.yield " << value << " : f64\n"
       << "  } : !pairs, !vec -> f64\n";
    add(set);
  }
  if ((terms & (GeneralizedBorn | Surface)) &&
      control.implicitSolvent != Control::ImplicitSolvent::None)
    emitBorn(terms, add);
  if (terms & ExternalTerms)
    for (auto [index, term] : llvm::enumerate(system.topology->externalTerms))
      if ((externalTerm < 0 || static_cast<int>(index) == externalTerm) &&
          !(alchemicalOnly && !usesLambda(term.expression)))
        emitExternalTerm(index, term, add);
  if (lj || coulomb) {
    // [free_energy] (D161): a pair with one particle in the selection,
    // `%al_x` = 1, is decoupled. Its Lennard-Jones is that of the
    // soft-core distance r_A, r_A^6 = alpha sigma^6 lambda^p + r^6, times
    // 1 - lambda [Beutler1994], and its Coulomb is times 1 - lambda: the
    // charge of the particle in the selection scaled. The pairs within the
    // selection stay.
    bool flags = (lj && scalesVdw) || (coulomb && scalesCoulomb) ||
                 (alchemicalOnly && decouples());
    os << "  %u_nonbonded = md.sum_relation %n, %x, %cell gather(%p_type, "
          "%p_q"
       << (flags ? ", %p_alch : !ids, !real, !real)\n"
                 : " : !ids, !real)\n")
       << "      exchange(symmetric) {\n"
       << "  ^bb0(%r: f64, %d: vector<3xf64>, %type_i: i32, %type_j: i32, "
          "%q_i: f64, %q_j: f64"
       << (flags ? ", %al_i: f64, %al_j: f64" : "") << "):\n";
    if (flags)
      os << "    %al_ij = arith.mulf %al_i, %al_j : f64\n"
         << "    %al_sum = arith.addf %al_i, %al_j : f64\n"
         << "    %al_twice = arith.addf %al_ij, %al_ij : f64\n"
         << "    %cross = arith.subf %al_sum, %al_twice : f64\n"
         << "    %al_one = arith.constant 1.0 : f64\n";
    std::string value;
    if (lj) {
      os << "    %sigma = md.lookup %t_lj_sigma[%type_i, %type_j] : !table, "
            "i32, i32 -> f64\n"
         << "    %epsilon = md.lookup %t_lj_epsilon[%type_i, %type_j] : "
            "!table, i32, i32 -> f64\n";
      if (scalesVdw) {
        // The distance r_A of a decoupled pair, r of the others. A pair
        // without a σ, which has no Lennard-Jones, takes 0.3 nm in r_A,
        // so that r_A stays above 0 where its particles overlap.
        double alpha = control.freeEnergy.softCoreAlpha;
        std::string distance = "%r";
        if (alpha > 0.0) {
          os << "    %sc_alpha = arith.constant " << formatReal(alpha)
             << " : f64\n"
             << "    %sc_zero = arith.constant 0.0 : f64\n"
             << "    %sc_floor = arith.constant 3.0e-01 : f64\n"
             << "    %sc_has = arith.cmpf ogt, %sigma, %sc_zero : f64\n"
             << "    %sc_sigma = arith.select %sc_has, %sigma, %sc_floor : f64\n"
             << "    %sc_i3 = arith.constant 3 : i32\n"
             << "    %sc_s3 = math.fpowi %sc_sigma, %sc_i3 : f64, i32\n"
             << "    %sc_s6 = arith.mulf %sc_s3, %sc_s3 : f64\n"
             << "    %sc_r3 = math.fpowi %r, %sc_i3 : f64, i32\n"
             << "    %sc_r6 = arith.mulf %sc_r3, %sc_r3 : f64\n";
          std::string power = "%lambda_vdw";
          if (control.freeEnergy.softCorePower == 2) {
            os << "    %sc_lp = arith.mulf %lambda_vdw, %lambda_vdw : f64\n";
            power = "%sc_lp";
          }
          os << "    %sc_as = arith.mulf %sc_alpha, %sc_s6 : f64\n"
             << "    %sc_asl = arith.mulf %sc_as, " << power << " : f64\n"
             << "    %sc_x = arith.addf %sc_r6, %sc_asl : f64\n"
             ;
          if (control.truncation != Truncation::None &&
              control.truncation != Truncation::Shift) {
            os << "    %sc_sixth = arith.constant "
               << formatReal(1.0 / 6.0) << " : f64\n"
               << "    %sc_ra = math.powf %sc_x, %sc_sixth : f64\n"
               << "    %sc_half = arith.constant 5.0e-01 : f64\n"
               << "    %sc_is = arith.cmpf ogt, %cross, %sc_half : f64\n"
               << "    %sc_r = arith.select %sc_is, %sc_ra, %r : f64\n";
            distance = "%sc_r";
          }
        }
        bool plain = control.truncation == Truncation::None ||
                     control.truncation == Truncation::Shift;
        if (alpha > 0.0 && plain) {
          // Without a switch, (σ/r_A)⁶ = σ⁶ / x needs no root: one
          // expression for every pair, x = r⁶ for those not decoupled.
          os << "    %sc_xs = arith.mulf %sc_asl, %cross : f64\n"
             << "    %sc_xx = arith.addf %sc_r6, %sc_xs : f64\n"
             << "    %sc_g3 = math.fpowi %sigma, %sc_i3 : f64, i32\n"
             << "    %sc_g6 = arith.mulf %sc_g3, %sc_g3 : f64\n"
             << "    %sc_q6 = arith.divf %sc_g6, %sc_xx : f64\n"
             << "    %sc_q12 = arith.mulf %sc_q6, %sc_q6 : f64\n";
          std::string a = "%sc_q12", b = "%sc_q6";
          if (control.truncation == Truncation::Shift) {
            double cut6 = std::pow(cutoff, -6.0);
            os << "    %sc_rc6 = arith.constant " << formatReal(cut6)
               << " : f64\n"
               << "    %sc_rc12 = arith.constant " << formatReal(cut6 * cut6)
               << " : f64\n"
               << "    %sc_g12 = arith.mulf %sc_g6, %sc_g6 : f64\n"
               << "    %sc_c12 = arith.mulf %sc_g12, %sc_rc12 : f64\n"
               << "    %sc_c6 = arith.mulf %sc_g6, %sc_rc6 : f64\n"
               << "    %sc_a = arith.subf %sc_q12, %sc_c12 : f64\n"
               << "    %sc_b = arith.subf %sc_q6, %sc_c6 : f64\n";
            a = "%sc_a";
            b = "%sc_b";
          }
          os << "    %sc_c4 = arith.constant 4.0 : f64\n"
             << "    %sc_e4 = arith.mulf %sc_c4, %epsilon : f64\n"
             << "    %sc_t = arith.subf " << a << ", " << b << " : f64\n"
             << "    %lj_full = arith.mulf %sc_e4, %sc_t : f64\n";
        } else {
          emitLennardJones("%sigma", "%epsilon", "%lj_full",
                           control.truncation, distance);
        }
        os << "    %lj_cl = arith.mulf %cross, %lambda_vdw : f64\n"
           << "    %lj_scale = arith.subf %al_one, %lj_cl : f64\n"
           << "    %lj = arith.mulf %lj_scale, %lj_full : f64\n";
      } else {
        emitLennardJones("%sigma", "%epsilon", "%lj", control.truncation);
      }
      value = "%lj";
      if (program.ljpme) {
        // The dispersion that the grid sums within the cutoff,
        // −c_i c_j (1 − g(β r)) / r⁶, taken out again, so that the pair
        // has its own Lennard-Jones there [Essmann1995, Wennberg2013]
        // (D162); shifted with it to 0 at the cutoff if it is.
        double beta = program.ljpmeBeta;
        os << "    %lp_c6 = md.lookup %t_ljpme_c6[%type_i, %type_j] : "
              "!table, i32, i32 -> f64\n"
           << "    %lp_beta = arith.constant " << formatReal(beta)
           << " : f64\n"
           << "    %lp_x = arith.mulf %lp_beta, %r : f64\n"
           << "    %lp_x2 = arith.mulf %lp_x, %lp_x : f64\n"
           << "    %lp_nx2 = arith.negf %lp_x2 : f64\n"
           << "    %lp_e = math.exp %lp_nx2 : f64\n"
           << "    %lp_x4 = arith.mulf %lp_x2, %lp_x2 : f64\n"
           << "    %lp_half = arith.constant 0.5 : f64\n"
           << "    %lp_h4 = arith.mulf %lp_half, %lp_x4 : f64\n"
           << "    %lp_one = arith.constant 1.0 : f64\n"
           << "    %lp_p1 = arith.addf %lp_one, %lp_x2 : f64\n"
           << "    %lp_p = arith.addf %lp_p1, %lp_h4 : f64\n"
           << "    %lp_g = arith.mulf %lp_e, %lp_p : f64\n"
           << "    %lp_kept = arith.subf %lp_one, %lp_g : f64\n"
           << "    %lp_r2 = arith.mulf %r, %r : f64\n"
           << "    %lp_r4 = arith.mulf %lp_r2, %lp_r2 : f64\n"
           << "    %lp_r6 = arith.mulf %lp_r4, %lp_r2 : f64\n"
           << "    %lp_cr = arith.divf %lp_c6, %lp_r6 : f64\n"
           << "    %lp_d = arith.mulf %lp_cr, %lp_kept : f64\n";
        std::string correction = "%lp_d";
        if (control.truncation == Truncation::Shift) {
          double shift = (1.0 - getDispersionScreen(beta * cutoff)) /
                         std::pow(cutoff, 6.0);
          os << "    %lp_shift = arith.constant " << formatReal(shift)
             << " : f64\n"
             << "    %lp_cs = arith.mulf %lp_c6, %lp_shift : f64\n"
             << "    %lp_ds = arith.subf %lp_d, %lp_cs : f64\n";
          correction = "%lp_ds";
        }
        os << "    %ljp = arith.addf %lj, " << correction << " : f64\n";
        value = "%ljp";
      }
    }
    if (coulomb) {
      os << "    %f = arith.constant " << formatReal(coulombInternal)
         << " : f64\n";
      if (scalesCoulomb)
        os << "    %qq0 = arith.mulf %q_i, %q_j : f64\n"
           << "    %qq_cl = arith.mulf %cross, %lambda_coulomb : f64\n"
           << "    %qq_scale = arith.subf %al_one, %qq_cl : f64\n"
           << "    %qq = arith.mulf %qq0, %qq_scale : f64\n";
      else
        os << "    %qq = arith.mulf %q_i, %q_j : f64\n";
      os << "    %fqq = arith.mulf %f, %qq : f64\n";
      // A decoupled pair may overlap once its Lennard-Jones is off; its
      // Coulomb, then 0, takes a distance of at least 1e-4 nm, which no
      // pair that interacts comes near, so that it is not 0 times ∞.
      std::string rq = "%r";
      if (scalesCoulomb) {
        os << "    %rq_least = arith.constant 1.0e-04 : f64\n"
           << "    %rq = arith.maximumf %r, %rq_least : f64\n";
        rq = "%rq";
      }
      if (program.pme) {
        // The direct sum of particle mesh Ewald, f q q erfc(β r) / r,
        // shifted to 0 at the cutoff if the control file asks.
        double beta = program.pmeBeta;
        os << "    %beta = arith.constant " << formatReal(beta) << " : f64\n"
           << "    %br = arith.mulf %beta, " << rq << " : f64\n"
           << "    %erfc = math.erfc %br : f64\n"
           << "    %screened = arith.divf %erfc, " << rq << " : f64\n";
        std::string kernel = "%screened";
        if (control.pmeShift) {
          os << "    %shift = arith.constant "
             << formatReal(std::erfc(beta * cutoff) / cutoff) << " : f64\n"
             << "    %shifted = arith.subf %screened, %shift : f64\n";
          kernel = "%shifted";
        }
        os << "    %coulomb = arith.mulf %fqq, " << kernel << " : f64\n";
      } else if (program.reactionField) {
        // f q q (1/r + k r² − c), 0 at the cutoff (D140).
        auto [k, c] = getReactionField(control);
        os << "    %one = arith.constant 1.0 : f64\n"
           << "    %krf = arith.constant " << formatReal(k) << " : f64\n"
           << "    %crf = arith.constant " << formatReal(c) << " : f64\n"
           << "    %inverse = arith.divf %one, " << rq << " : f64\n"
           << "    %rr = arith.mulf %r, %r : f64\n"
           << "    %field = arith.mulf %krf, %rr : f64\n"
           << "    %near = arith.addf %inverse, %field : f64\n"
           << "    %rf = arith.subf %near, %crf : f64\n"
           << "    %coulomb = arith.mulf %fqq, %rf : f64\n";
      } else {
        os << "    %coulomb = arith.divf %fqq, " << rq << " : f64\n";
      }
      if (value.empty()) {
        value = "%coulomb";
      } else {
        os << "    %k = arith.addf " << value << ", %coulomb : f64\n";
        value = "%k";
      }
    }
    // In `@alchemical` the pairs that λ leaves alone add nothing.
    if (alchemicalOnly) {
      os << "    %masked = arith.mulf " << value << ", %cross : f64\n";
      value = "%masked";
    }
    os << "    md.yield " << value << " : f64\n"
       << "  } : !pairs, !vec -> f64\n";
    add("nonbonded");
  }

  if (program.pme && (terms & CoulombExcluded) && has("excluded")) {
    // The excluded pairs take their share of the reciprocal sum out again:
    // −f q_i q_j erf(β r) / r, with the charges that the reciprocal sum
    // takes, those of the selection of [free_energy] times 1 − λ (D161).
    bool flags = scalesCoulomb || (alchemicalOnly && decouples());
    os << "  %u_excluded = md.sum_tuples %r_excluded, %x, %cell coordinates("
          "distance(0, 1))\n"
       << "      gather(%p_q" << (flags ? ", %p_alch : !real, !real" : " : !real")
       << ") {\n"
       << "  ^bb0(%r: f64, %q_i: f64, %q_j: f64"
       << (flags ? ", %al_i: f64, %al_j: f64" : "") << "):\n"
       << "    %f = arith.constant " << formatReal(-coulombInternal)
       << " : f64\n"
       << "    %beta = arith.constant " << formatReal(program.pmeBeta)
       << " : f64\n"
       << "    %qq0 = arith.mulf %q_i, %q_j : f64\n";
    std::string qq = "%qq0";
    if (flags) {
      os << "    %al_one = arith.constant 1.0 : f64\n"
         << "    %al_li = arith.mulf %al_i, %lambda_coulomb : f64\n"
         << "    %al_si = arith.subf %al_one, %al_li : f64\n"
         << "    %al_lj = arith.mulf %al_j, %lambda_coulomb : f64\n"
         << "    %al_sj = arith.subf %al_one, %al_lj : f64\n"
         << "    %al_s = arith.mulf %al_si, %al_sj : f64\n"
         << "    %qq = arith.mulf %qq0, %al_s : f64\n";
      qq = "%qq";
    }
    os << "    %fqq = arith.mulf %f, " << qq << " : f64\n"
       << "    %br = arith.mulf %beta, %r : f64\n"
       << "    %erf = math.erf %br : f64\n"
       << "    %shielded = arith.divf %erf, %r : f64\n"
       << "    %e = arith.mulf %fqq, %shielded : f64\n";
    std::string e = "%e";
    if (alchemicalOnly) {
      // Only the pairs within the selection depend on λ.
      os << "    %al_both = arith.mulf %al_i, %al_j : f64\n"
         << "    %masked = arith.mulf %e, %al_both : f64\n";
      e = "%masked";
    }
    os << "    md.yield " << e << " : f64\n"
       << "  } : !rel_excluded, !vec -> f64\n";
    add("excluded");
  }
  if (program.pme && (terms & CoulombWithin) && scalesCoulomb &&
      has("alchemical_pairs")) {
    // The pairs within the selection that are not excluded keep their
    // Coulomb, which the reciprocal sum of the scaled charges takes times
    // (1 − λ)²: (1 − (1 − λ)²) f q_i q_j erf(β r) / r adds the rest (D161).
    os << "  %u_within = md.sum_tuples %r_alchemical_pairs, %x, %cell "
          "coordinates(distance(0, 1))\n"
       << "      gather(%p_q : !real) {\n"
       << "  ^bb0(%r: f64, %q_i: f64, %q_j: f64):\n"
       << "    %f = arith.constant " << formatReal(coulombInternal)
       << " : f64\n"
       << "    %beta = arith.constant " << formatReal(program.pmeBeta)
       << " : f64\n"
       << "    %one = arith.constant 1.0 : f64\n"
       << "    %s = arith.subf %one, %lambda_coulomb : f64\n"
       << "    %s2 = arith.mulf %s, %s : f64\n"
       << "    %w = arith.subf %one, %s2 : f64\n"
       << "    %qq = arith.mulf %q_i, %q_j : f64\n"
       << "    %fqq = arith.mulf %f, %qq : f64\n"
       << "    %fw = arith.mulf %fqq, %w : f64\n"
       << "    %br = arith.mulf %beta, %r : f64\n"
       << "    %erf = math.erf %br : f64\n"
       << "    %shielded = arith.divf %erf, %r : f64\n"
       << "    %e = arith.mulf %fw, %shielded : f64\n"
       << "    md.yield %e : f64\n"
       << "  } : !rel_alchemical_pairs, !vec -> f64\n";
    add("within");
  }
  if (program.reactionField && (terms & CoulombExcluded) &&
      !alchemicalOnly && has("excluded")) {
    // The field acts on the excluded pairs within the cutoff as well:
    // f q_i q_j (k r² − c), as GROMACS has it (D140).
    auto [k, c] = getReactionField(control);
    os << "  %u_excluded = md.sum_tuples %r_excluded, %x, %cell coordinates("
          "distance(0, 1))\n"
       << "      gather(%p_q : !real) {\n"
       << "  ^bb0(%r: f64, %q_i: f64, %q_j: f64):\n"
       << "    %f = arith.constant " << formatReal(coulombInternal)
       << " : f64\n"
       << "    %krf = arith.constant " << formatReal(k) << " : f64\n"
       << "    %crf = arith.constant " << formatReal(c) << " : f64\n"
       << "    %rc = arith.constant " << formatReal(cutoff) << " : f64\n"
       << "    %zero = arith.constant 0.0 : f64\n"
       << "    %qq = arith.mulf %q_i, %q_j : f64\n"
       << "    %fqq = arith.mulf %f, %qq : f64\n"
       << "    %rr = arith.mulf %r, %r : f64\n"
       << "    %field = arith.mulf %krf, %rr : f64\n"
       << "    %rf = arith.subf %field, %crf : f64\n"
       << "    %all = arith.mulf %fqq, %rf : f64\n"
       << "    %inside = arith.cmpf olt, %r, %rc : f64\n"
       << "    %e = arith.select %inside, %all, %zero : f64\n"
       << "    md.yield %e : f64\n"
       << "  } : !rel_excluded, !vec -> f64\n";
    add("excluded");
  }
  if (program.ljpme && (terms & LennardJonesExcluded) && has("excluded")) {
    // The excluded pairs take their share of the grid of the dispersion
    // out again: c_i c_j P(3, β² r²) / r⁶ with P(3, y) = 1 − exp(−y) (1 +
    // y + y²/2) = 1 − g(β r) (D162). Below y = 1, P is taken from its
    // series exp(−y) y³ Σ_k y^k / (k + 3)!, which has no cancellation;
    // the pairs of a molecule are close, where 1 − g is small.
    os << "  %u_lj_excluded = md.sum_tuples %r_excluded, %x, %cell "
          "coordinates(distance(0, 1))\n"
       << "      gather(%p_type : !ids) {\n"
       << "  ^bb0(%r: f64, %type_i: i32, %type_j: i32):\n"
       << "    %c6 = md.lookup %t_ljpme_c6[%type_i, %type_j] : !table, i32, "
          "i32 -> f64\n"
       << "    %beta = arith.constant " << formatReal(program.ljpmeBeta)
       << " : f64\n"
       << "    %bx = arith.mulf %beta, %r : f64\n"
       << "    %y = arith.mulf %bx, %bx : f64\n"
       << "    %ny = arith.negf %y : f64\n"
       << "    %ey = math.exp %ny : f64\n"
       << "    %one = arith.constant 1.0 : f64\n"
       << "    %half = arith.constant 0.5 : f64\n";
    // The series, by Horner's rule, to k = 10: the rest is below 1e-10 of
    // it.
    double factorial = 6.0;
    std::vector<double> coefficients;
    for (int k = 0; k <= 10; ++k) {
      coefficients.push_back(1.0 / factorial);
      factorial *= k + 4;
    }
    os << "    %s10 = arith.constant " << formatReal(coefficients[10])
       << " : f64\n";
    for (int k = 9; k >= 0; --k)
      os << "    %a" << k << " = arith.constant "
         << formatReal(coefficients[k]) << " : f64\n"
         << "    %sy" << k << " = arith.mulf %s" << k + 1 << ", %y : f64\n"
         << "    %s" << k << " = arith.addf %sy" << k << ", %a" << k
         << " : f64\n";
    os << "    %y2 = arith.mulf %y, %y : f64\n"
       << "    %y3 = arith.mulf %y2, %y : f64\n"
       << "    %ey3 = arith.mulf %ey, %y3 : f64\n"
       << "    %series = arith.mulf %ey3, %s0 : f64\n"
       << "    %hy2 = arith.mulf %half, %y2 : f64\n"
       << "    %p1 = arith.addf %one, %y : f64\n"
       << "    %p = arith.addf %p1, %hy2 : f64\n"
       << "    %g = arith.mulf %ey, %p : f64\n"
       << "    %closed = arith.subf %one, %g : f64\n"
       << "    %small = arith.cmpf olt, %y, %one : f64\n"
       << "    %kept = arith.select %small, %series, %closed : f64\n"
       << "    %r2 = arith.mulf %r, %r : f64\n"
       << "    %r4 = arith.mulf %r2, %r2 : f64\n"
       << "    %r6 = arith.mulf %r4, %r2 : f64\n"
       << "    %cr = arith.divf %c6, %r6 : f64\n"
       << "    %e = arith.mulf %cr, %kept : f64\n"
       << "    md.yield %e : f64\n"
       << "  } : !rel_excluded, !vec -> f64\n";
    add("lj_excluded");
  }
  if (program.ljpme && (terms & LennardJonesReciprocal)) {
    // The grid of the dispersion (D162), of the coefficients c_i.
    os << "  %u_lj_reciprocal, %f_lj_reciprocal, %w_lj_reciprocal = "
          "md.reciprocal %x, %p_ljpme_c, %cell, %t_ljpme_moduli\n"
       << "      grid([" << program.ljpmeGrid[0] << ", "
       << program.ljpmeGrid[1] << ", " << program.ljpmeGrid[2]
       << "]) order(" << control.ljpmeOrder << ") beta("
       << formatReal(program.ljpmeBeta) << ") coulomb(1.0) dispersion\n"
       << "      : !vec, !real, !grid -> f64, !vec, vector<9xf64>\n";
    add("lj_reciprocal");
  }
  if (program.pme && (terms & CoulombReciprocal)) {
    // The charges of the selection of [free_energy] times 1 − λ (D161).
    std::string charges = "%p_q";
    if (scalesCoulomb) {
      os << "  %q_scaled = md.map_particles gather(%p_q, %p_alch : !real, "
            "!real) {\n"
         << "  ^bb0(%q: f64, %al: f64):\n"
         << "    %one = arith.constant 1.0 : f64\n"
         << "    %al_l = arith.mulf %al, %lambda_coulomb : f64\n"
         << "    %al_s = arith.subf %one, %al_l : f64\n"
         << "    %qs = arith.mulf %q, %al_s : f64\n"
         << "    md.yield %qs : f64\n"
         << "  } : !real\n";
      charges = "%q_scaled";
    }
    os << "  %u_reciprocal, %f_reciprocal, %w_reciprocal = md.reciprocal %x, "
       << charges << ", %cell, %t_pme_moduli\n"
       << "      grid([" << program.pmeGrid[0] << ", " << program.pmeGrid[1]
       << ", " << program.pmeGrid[2] << "]) order(" << control.pmeOrder
       << ") beta(" << formatReal(program.pmeBeta) << ") coulomb("
       << formatReal(coulombInternal) << ")\n"
       << "      : !vec, !real, !grid -> f64, !vec, vector<9xf64>\n";
    add("reciprocal");
  }
  if ((terms & Bonds) && has("bonds")) {
    os << "  %u_bonds = md.sum_tuples %r_bonds, %x, %cell coordinates("
          "distance(0, 1))\n"
       << "      tuple(%f_bonds_k, %f_bonds_r0 : !of_bonds, !of_bonds) {\n"
       << "  ^bb0(%r: f64, %k: f64, %r0: f64):\n"
       << "    %half = arith.constant 0.5 : f64\n"
       << "    %dr = arith.subf %r, %r0 : f64\n"
       << "    %dr2 = arith.mulf %dr, %dr : f64\n"
       << "    %hk = arith.mulf %half, %k : f64\n"
       << "    %e = arith.mulf %hk, %dr2 : f64\n"
       << "    md.yield %e : f64\n"
       << "  } : !rel_bonds, !vec -> f64\n";
    add("bonds");
  }
  if ((terms & Angles) && has("angles")) {
    os << "  %u_angles = md.sum_tuples %r_angles, %x, %cell coordinates("
          "angle(0, 1, 2))\n"
       << "      tuple(%f_angles_k, %f_angles_theta0 : !of_angles, "
          "!of_angles) {\n"
       << "  ^bb0(%theta: f64, %k: f64, %theta0: f64):\n"
       << "    %half = arith.constant 0.5 : f64\n"
       << "    %dt = arith.subf %theta, %theta0 : f64\n"
       << "    %dt2 = arith.mulf %dt, %dt : f64\n"
       << "    %hk = arith.mulf %half, %k : f64\n"
       << "    %e = arith.mulf %hk, %dt2 : f64\n"
       << "    md.yield %e : f64\n"
       << "  } : !rel_angles, !vec -> f64\n";
    add("angles");
  }
  if ((terms & UreyBradleys) && has("urey_bradley")) {
    os << "  %u_urey_bradley = md.sum_tuples %r_urey_bradley, %x, %cell "
          "coordinates(distance(0, 1))\n"
       << "      tuple(%f_urey_bradley_k, %f_urey_bradley_r0 : "
          "!of_urey_bradley, !of_urey_bradley) {\n"
       << "  ^bb0(%r: f64, %k: f64, %r0: f64):\n"
       << "    %half = arith.constant 0.5 : f64\n"
       << "    %dr = arith.subf %r, %r0 : f64\n"
       << "    %dr2 = arith.mulf %dr, %dr : f64\n"
       << "    %hk = arith.mulf %half, %k : f64\n"
       << "    %e = arith.mulf %hk, %dr2 : f64\n"
       << "    md.yield %e : f64\n"
       << "  } : !rel_urey_bradley, !vec -> f64\n";
    add("urey_bradley");
  }
  if ((terms & Dihedrals) && has("dihedrals")) {
    os << "  %u_dihedrals = md.sum_tuples %r_dihedrals, %x, %cell "
          "coordinates(dihedral(0, 1, 2, 3))\n"
       << "      tuple(%f_dihedrals_k, %f_dihedrals_n, %f_dihedrals_phase : "
          "!of_dihedrals, !of_dihedrals, !of_dihedrals) {\n"
       << "  ^bb0(%phi: f64, %k: f64, %period: f64, %phase: f64):\n"
       << "    %one = arith.constant 1.0 : f64\n"
       << "    %nphi = arith.mulf %period, %phi : f64\n"
       << "    %a = arith.subf %nphi, %phase : f64\n"
       << "    %cos = math.cos %a : f64\n"
       << "    %s = arith.addf %one, %cos : f64\n"
       << "    %e = arith.mulf %k, %s : f64\n"
       << "    md.yield %e : f64\n"
       << "  } : !rel_dihedrals, !vec -> f64\n";
    add("dihedrals");
  }
  if ((terms & HarmonicImpropers) && has("impropers")) {
    // ½ k (ξ − ξ0)², with ξ − ξ0 brought into [−π, π).
    os << "  %u_impropers = md.sum_tuples %r_impropers, %x, %cell "
          "coordinates(dihedral(0, 1, 2, 3))\n"
       << "      tuple(%f_impropers_k, %f_impropers_xi0 : !of_impropers, "
          "!of_impropers) {\n"
       << "  ^bb0(%xi: f64, %k: f64, %xi0: f64):\n"
       << "    %half = arith.constant 0.5 : f64\n"
       << "    %pi = arith.constant " << formatReal(M_PI) << " : f64\n"
       << "    %turn = arith.constant " << formatReal(2.0 * M_PI) << " : f64\n"
       << "    %per_turn = arith.constant " << formatReal(0.5 / M_PI)
       << " : f64\n"
       << "    %d = arith.subf %xi, %xi0 : f64\n"
       << "    %dp = arith.addf %d, %pi : f64\n"
       << "    %turns = arith.mulf %dp, %per_turn : f64\n"
       << "    %whole = math.floor %turns : f64\n"
       << "    %back = arith.mulf %whole, %turn : f64\n"
       << "    %w = arith.subf %d, %back : f64\n"
       << "    %w2 = arith.mulf %w, %w : f64\n"
       << "    %hk = arith.mulf %half, %k : f64\n"
       << "    %e = arith.mulf %hk, %w2 : f64\n"
       << "    md.yield %e : f64\n"
       << "  } : !rel_impropers, !vec -> f64\n";
    add("impropers");
  }
  // The terms given by expressions (D136), in the units of the control
  // file: the distance in Å, the angles in radians, the energy in kcal/mol.
  for (auto [index, term] : llvm::enumerate(system.topology->tupleTerms)) {
    if (!(terms & TupleTerms) ||
        (tupleTerm >= 0 && static_cast<int>(index) != tupleTerm))
      continue;
    if (alchemicalOnly && !usesLambda(term.expression))
      continue;
    if (term.isCentroid()) {
      emitCentroidTerm(index, term);
      add("centroid_" + term.name);
      continue;
    }
    static const char *const coordinates[] = {
        "", "", "distance(0, 1)", "angle(0, 1, 2)", "dihedral(0, 1, 2, 3)"};
    std::string set = "custom_" + term.name;
    // A compound term (D165) takes the coordinates that its expression
    // names, of any members of its tuples.
    std::string list = term.isCompound() ? "" : coordinates[term.arity];
    for (auto [k, coordinate] : llvm::enumerate(term.coordinates)) {
      list += (k ? ", " : "") + coordinate.kind + "(";
      for (auto [p, place] : llvm::enumerate(coordinate.places))
        list += (p ? ", " : "") + std::to_string(place);
      list += ")";
    }
    os << "  %u_" << set << " = md.sum_tuples %r_" << set
       << ", %x, %cell coordinates(" << list << ")";
    if (!term.parameters.empty()) {
      os << "\n      tuple(";
      for (auto [k, parameter] : llvm::enumerate(term.parameters))
        os << (k ? ", " : "") << "%f_" << set << "_" << parameter.first;
      os << " : ";
      for (size_t k = 0; k != term.parameters.size(); ++k)
        os << (k ? ", " : "") << "!of_" << set;
      os << ")";
    }
    llvm::StringMap<std::string> values;
    if (term.isCompound()) {
      os << " {\n  ^bb0(";
      for (size_t k = 0; k != term.coordinates.size(); ++k)
        os << (k ? ", " : "") << "%c" << k << ": f64";
    } else {
      os << " {\n  ^bb0(%c: f64";
    }
    for (const auto &parameter : term.parameters)
      os << ", %cp_" << parameter.first << ": f64";
    os << "):\n";
    if (term.isCompound() || term.arity == 2)
      os << "    %c_scale = arith.constant " << formatReal(1.0 / units::length)
         << " : f64\n";
    if (term.isCompound()) {
      // Distances in Å, angles in radians.
      for (auto [k, coordinate] : llvm::enumerate(term.coordinates)) {
        std::string variable = "%c" + std::to_string(k);
        if (coordinate.kind == "distance") {
          os << "    " << variable << "_a = arith.mulf " << variable
             << ", %c_scale : f64\n";
          variable += "_a";
        }
        values[coordinate.name] = variable;
      }
    } else {
      std::string variable = "%c";
      if (term.arity == 2) {
        os << "    %c_a = arith.mulf %c, %c_scale : f64\n";
        variable = "%c_a";
      }
      values[term.getVariable()] = variable;
    }
    if (control.usesTime)
      values["t"] = "%time";
    bindLambdas(values);
    for (const auto &parameter : term.parameters)
      values[parameter.first] = "%cp_" + parameter.first;
    bindObserved(term.name, values);
    Expression expression = llvm::cantFail(Expression::parse(term.expression, control.functions));
    std::string energy = expression.emit(os, values, "%ce", "    ", term.name);
    os << "    %c_kj = arith.constant " << formatReal(units::energy)
       << " : f64\n"
       << "    %c_e = arith.mulf " << energy << ", %c_kj : f64\n"
       << "    md.yield %c_e : f64\n"
       << "  } : !rel_" << set << ", !vec -> f64\n";
    add(set);
  }
  bool lj14 = terms & LennardJones14, coulomb14 = terms & Coulomb14;
  if ((lj14 || coulomb14) && has("pairs14")) {
    os << "  %u_pairs14 = md.sum_tuples %r_pairs14, %x, %cell coordinates("
          "distance(0, 1))\n"
       << "      tuple(%f_pairs14_sigma, %f_pairs14_epsilon, %f_pairs14_qq : "
          "!of_pairs14, !of_pairs14, !of_pairs14) {\n"
       << "  ^bb0(%r: f64, %sigma: f64, %epsilon: f64, %qq: f64):\n";
    std::string value;
    if (lj14) {
      // The pairs three bonds apart as they are, except under the power
      // force or squared-distance switch, which also switches these pairs.
      emitLennardJones("%sigma", "%epsilon", "%lj",
                       (control.truncation == Truncation::PowerForceSwitch ||
                        control.truncation == Truncation::SquaredDistanceSwitch)
                           ? control.truncation
                           : Truncation::None);
      value = "%lj";
    }
    if (coulomb14) {
      os << "    %coulomb = arith.divf %qq, %r : f64\n";
      if (value.empty()) {
        value = "%coulomb";
      } else {
        os << "    %k = arith.addf " << value << ", %coulomb : f64\n";
        value = "%k";
      }
    }
    os << "    md.yield " << value << " : f64\n"
       << "  } : !rel_pairs14, !vec -> f64\n";
    add("pairs14");
  }
  if ((terms & CMaps) && has("cmap")) {
    // The cell of φ and ψ, each from −180° in steps of 360° / n, and the
    // places t and u in it; φ = 180° is the first cell again. The patch
    // is E = Σ c_ij t^i u^j [MacKerell2004].
    unsigned n = system.topology->cmapResolution;
    os << "  %u_cmap = md.sum_tuples %r_cmap, %x, %cell coordinates("
          "dihedral(0, 1, 2, 3), dihedral(1, 2, 3, 4))\n"
       << "      tuple(%f_cmap_map : !of_cmap) {\n"
       << "  ^bb0(%phi: f64, %psi: f64, %map: f64):\n"
       << "    %scale = arith.constant " << formatReal(n / (2.0 * M_PI))
       << " : f64\n"
       << "    %shift = arith.constant " << formatReal(n / 2.0) << " : f64\n"
       << "    %cmap_n = arith.constant " << n << " : i32\n";
    for (StringRef angle : {"phi", "psi"}) {
      std::string a = angle.str();
      os << "    %" << a << "_s = arith.mulf %" << a << ", %scale : f64\n"
         << "    %" << a << "_x = arith.addf %" << a << "_s, %shift : f64\n"
         << "    %" << a << "_f = math.floor %" << a << "_x : f64\n"
         << "    %" << a << "_t = arith.subf %" << a << "_x, %" << a
         << "_f : f64\n"
         << "    %" << a << "_i = arith.fptosi %" << a
         << "_f : f64 to i32\n"
         << "    %" << a << "_j = arith.addi %" << a
         << "_i, %cmap_n : i32\n"
         << "    %" << a << "_k = arith.remsi %" << a
         << "_j, %cmap_n : i32\n";
    }
    os << "    %m = arith.fptosi %map : f64 to i32\n"
       << "    %mn = arith.muli %m, %cmap_n : i32\n"
       << "    %row = arith.addi %mn, %phi_k : i32\n"
       << "    %rown = arith.muli %row, %cmap_n : i32\n"
       << "    %cellid = arith.addi %rown, %psi_k : i32\n";
    for (int k = 0; k != 16; ++k)
      os << "    %k" << k << " = arith.constant " << k << " : i32\n"
         << "    %c" << k << " = md.lookup %t_cmap[%cellid, %k" << k
         << "] : !grid, i32, i32 -> f64\n";
    // Horner in u for each power of t, then in t.
    for (int i = 0; i != 4; ++i) {
      std::string last = "%c" + std::to_string(4 * i + 3);
      for (int j = 2; j >= 0; --j) {
        std::string name = "%r" + std::to_string(i) + std::to_string(j);
        os << "    " << name << "m = arith.mulf " << last
           << ", %psi_t : f64\n"
           << "    " << name << " = arith.addf " << name << "m, %c"
           << 4 * i + j << " : f64\n";
        last = name;
      }
    }
    std::string last = "%r30";
    for (int i = 2; i >= 0; --i) {
      std::string name = "%e" + std::to_string(i);
      os << "    " << name << "m = arith.mulf " << last << ", %phi_t : f64\n"
         << "    " << name << " = arith.addf " << name << "m, %r" << i
         << "0 : f64\n";
      last = name;
    }
    os << "    md.yield %e0 : f64\n"
       << "  } : !rel_cmap, !vec -> f64\n";
    add("cmap");
  }
  if (total.empty()) {
    os << "  %zero = arith.constant 0.0 : f64\n";
    total = "%zero";
  }
  os << "  md.return " << total << " : f64\n}\n\n";
}

llvm::Error Builder::emitPotential() {
  double cutoff = control.cutoffDistance * units::length;

  if (control.truncation == Truncation::PowerForceSwitch)
    return makeError("the power force switch is for the Lennard-Jones of a "
                     "topology");
  std::string truncation = getTruncation(control);

  os << "md.potential @energy(%x: !vec, %cell: !md.cell"
     << getFieldParameters() << getTimeParameter() << ") -> f64 {\n";
  os << "  %n = md.neighborhood %x, %cell cutoff(" << formatReal(cutoff)
     << ") : !vec -> !pairs\n";

  std::string total;
  for (auto [index, term] : llvm::enumerate(control.pairs)) {
    const Expression &expression = expressions[index];

    // The fields that the kernel reads, and the values of the two
    // particles of a pair.
    const llvm::StringMap<unsigned> &tables = termTables[index];
    std::string gathered, types, arguments;
    if (!tables.empty()) {
      gathered = "%p_type";
      types = "!ids";
      arguments = ", %type_i: i32, %type_j: i32";
    }
    for (const Parameter &parameter : parameters) {
      if (!tables.empty() || parameter.isUniform ||
          !llvm::is_contained(expression.getNames(), parameter.name))
        continue;
      gathered += (gathered.empty() ? "" : ", ") + ("%p_" + parameter.name);
      types += std::string(types.empty() ? "" : ", ") + "!real";
      arguments += ", %" + parameter.name + "_i: f64, %" + parameter.name +
                   "_j: f64";
    }

    std::string result = "%u" + std::to_string(index);
    os << "  " << result << " = md.sum_relation %n, %x, %cell";
    if (!gathered.empty())
      os << " gather(" << gathered << " : " << types << ")";
    os << "\n      exchange(symmetric)" << truncation << " {\n";
    os << "  ^bb0(%r: f64, %d: vector<3xf64>" << arguments << "):\n";

    // The expression is evaluated in the units of the control file.
    llvm::StringMap<std::string> values;
    os << "    %to_length = arith.constant "
       << formatReal(1.0 / units::length) << " : f64\n";
    os << "    %r_in = arith.mulf %r, %to_length : f64\n";
    values["r"] = "%r_in";

    for (auto &[name, value] : term.constants) {
      os << "    %c_" << name << " = arith.constant " << formatReal(value)
         << " : f64\n";
      values[name] = "%c_" + name;
    }
    if (llvm::is_contained(expression.getNames(), "coulomb")) {
      os << "    %coulomb = arith.constant " << formatReal(coulombConstant)
         << " : f64\n";
      values["coulomb"] = "%coulomb";
    }
    for (auto &[name, table] : tables) {
      os << "    %" << name.str() << " = md.lookup %t_"
         << program.tables[table].name
         << "[%type_i, %type_j] : !table, i32, i32 -> f64\n";
      values[name] = "%" + name.str();
    }
    for (const Parameter &parameter : parameters) {
      if (!tables.empty() ||
          !llvm::is_contained(expression.getNames(), parameter.name))
        continue;
      const std::string &name = parameter.name;
      if (parameter.isUniform) {
        os << "    %" << name << " = arith.constant "
           << formatReal(parameter.value) << " : f64\n";
      } else if (term.mixing.lookup(name) == Mixing::Arithmetic) {
        os << "    %half_" << name << " = arith.constant 5.0e-01 : f64\n";
        os << "    %sum_" << name << " = arith.addf %" << name << "_i, %"
           << name << "_j : f64\n";
        os << "    %" << name << " = arith.mulf %half_" << name << ", %sum_"
           << name << " : f64\n";
      } else if (term.mixing.lookup(name) == Mixing::Product) {
        os << "    %" << name << " = arith.mulf %" << name << "_i, %" << name
           << "_j : f64\n";
      } else {
        os << "    %product_" << name << " = arith.mulf %" << name
           << "_i, %" << name << "_j : f64\n";
        os << "    %" << name << " = math.sqrt %product_" << name
           << " : f64\n";
      }
      values[name] = "%" + name;
    }

    std::string value = expression.emit(os, values, "%e", "    ", term.name);
    os << "    %to_energy = arith.constant " << formatReal(units::energy)
       << " : f64\n";
    os << "    %u = arith.mulf " << value << ", %to_energy : f64\n";
    os << "    md.yield %u : f64\n";
    os << "  } : !pairs, !vec -> f64\n";

    if (total.empty()) {
      total = result;
    } else {
      std::string sum = "%total" + std::to_string(index);
      os << "  " << sum << " = arith.addf " << total << ", " << result
         << " : f64\n";
      total = sum;
    }
  }
  for (auto [index, term] : llvm::enumerate(control.triplets)) {
    std::string result = emitTripletTerm(index, term);
    std::string sum = "%total_t" + std::to_string(index);
    os << "  " << sum << " = arith.addf " << total << ", " << result
       << " : f64\n";
    total = sum;
  }
  os << "  md.return " << total << " : f64\n}\n\n";
  return llvm::Error::success();
}

std::string Builder::emitTripletTerm(size_t index, const TripletTerm &term) {
  // The triplets of the neighborhood within the cutoff of the term, each
  // center with each unordered pair of its legs once (D160); the center is
  // particle 1 of the expression and place 1 of the IR, so the ends 2 and
  // 3 are places 0 and 2.
  std::string k = std::to_string(index);
  std::string relation = "%triplets" + k, result = "%u_triplet" + k;
  os << "  " << relation << " = md.triplets %n cutoff("
     << formatReal(term.cutoff * units::length)
     << ") : !pairs -> !md.relation<@atoms, 3, reversal>\n";

  Expression expression =
      llvm::cantFail(Expression::parse(term.expression, control.functions));
  // cos(theta) is the coordinate cosine(0, 1, 2), whose derivative has no
  // 1/sin(theta), which a collinear triplet makes infinite
  // (design-m1.md, Section 4.1); theta alone is the angle.
  expression.replaceCall("cos", "theta", "cos(theta)");
  const std::vector<std::string> &names = expression.getNames();
  struct Variable {
    const char *name, *coordinate;
    bool isLength;
  };
  static const Variable variables[] = {
      {"r12", "distance(0, 1)", true},
      {"r13", "distance(2, 1)", true},
      {"r23", "distance(0, 2)", true},
      {"theta", "angle(0, 1, 2)", false},
      {"cos(theta)", "cosine(0, 1, 2)", false}};
  // A loop over tuples takes at least one coordinate: r12, if the
  // expression uses none.
  std::vector<const Variable *> used;
  for (const Variable &variable : variables)
    if (llvm::is_contained(names, variable.name))
      used.push_back(&variable);
  if (used.empty())
    used.push_back(&variables[0]);

  // The parameters of the particles that differ between the types, one
  // value for each place.
  std::vector<std::string> gathered;
  for (const std::string &name : names) {
    std::string stem = getTripletParameter(term, name);
    if (stem.empty() || llvm::is_contained(gathered, stem))
      continue;
    const Parameter &parameter = *llvm::find_if(
        parameters, [&](const Parameter &p) { return p.name == stem; });
    if (!parameter.isUniform)
      gathered.push_back(stem);
  }

  std::string p = "%t" + k + "_";
  os << "  " << result << " = md.sum_tuples " << relation
     << ", %x, %cell coordinates(";
  llvm::interleaveComma(used, os,
                        [&](const Variable *v) { os << v->coordinate; });
  os << ")";
  if (!gathered.empty()) {
    os << "\n      gather(";
    llvm::interleaveComma(gathered, os,
                          [&](const std::string &s) { os << "%p_" << s; });
    os << " : ";
    llvm::interleaveComma(gathered, os,
                          [&](const std::string &) { os << "!real"; });
    os << ")";
  }
  os << " {\n  ^bb0(";
  llvm::StringMap<std::string> values;
  for (auto [i, v] : llvm::enumerate(used))
    os << (i ? ", " : "") << p << "v" << i << ": f64";
  for (const std::string &stem : gathered)
    for (int place = 0; place != 3; ++place)
      os << ", " << p << stem << "_" << place << ": f64";
  os << "):\n";

  // The expression is evaluated in the units of the control file.
  os << "    " << p << "to_length = arith.constant "
     << formatReal(1.0 / units::length) << " : f64\n";
  for (auto [i, v] : llvm::enumerate(used)) {
    std::string argument = p + "v" + std::to_string(i);
    if (v->isLength) {
      os << "    " << argument << "_in = arith.mulf " << argument << ", " << p
         << "to_length : f64\n";
      argument += "_in";
    }
    values[v->name] = argument;
  }
  for (auto &[name, value] : term.constants) {
    os << "    " << p << "c_" << name << " = arith.constant "
       << formatReal(value) << " : f64\n";
    values[name] = p + "c_" + name;
  }
  // Particle 1 is the center, place 1; 2 and 3 are places 0 and 2.
  static const int places[] = {1, 0, 2};
  for (const std::string &name : names) {
    std::string stem = getTripletParameter(term, name);
    if (stem.empty())
      continue;
    if (llvm::is_contained(gathered, stem)) {
      values[name] = p + stem + "_" +
                     std::to_string(places[name.back() - '1']);
      continue;
    }
    const Parameter &parameter = *llvm::find_if(
        parameters, [&](const Parameter &q) { return q.name == stem; });
    os << "    " << p << "u_" << name << " = arith.constant "
       << formatReal(parameter.value) << " : f64\n";
    values[name] = p + "u_" + name;
  }
  if (control.usesTime)
    values["t"] = "%time";
  std::string value = expression.emit(os, values, p + "e", "    ", term.name);
  os << "    " << p << "to_energy = arith.constant "
     << formatReal(units::energy) << " : f64\n";
  os << "    " << p << "u = arith.mulf " << value << ", " << p
     << "to_energy : f64\n";
  os << "    md.yield " << p << "u : f64\n";
  os << "  } : !md.relation<@atoms, 3, reversal>, !vec -> f64\n";
  return result;
}

/// The diagonal of the virial `virial`, a vector of three, whose elements
/// enter the pressure of each axis.
static void emitDiagonal(llvm::raw_ostream &os, StringRef result,
                         StringRef virial, StringRef indent) {
  for (StringRef part : {"0", "4", "8"})
    os << indent << result << "_" << part << " = vector.extract " << virial
       << "[" << part << "] : f64 from vector<9xf64>\n";
  os << indent << result << " = vector.from_elements " << result << "_0, "
     << result << "_4, " << result << "_8 : vector<3xf64>\n";
}

/// The kinetic energy of the velocities `velocities` along each axis,
/// Σ m v ⊙ v / 2, a vector of three.
static void emitKineticVector(llvm::raw_ostream &os, StringRef result,
                              StringRef velocities, StringRef masses,
                              StringRef indent) {
  os << indent << result << " = md.sum_particles gather(" << velocities
     << ", " << masses << " : !vec, !real) {\n"
     << indent << "^bb0(%kv_v: vector<3xf64>, %kv_m: f64):\n"
     << indent << "  %kv_c = arith.constant 5.0e-01 : f64\n"
     << indent << "  %kv_hm = arith.mulf %kv_c, %kv_m : f64\n"
     << indent << "  %kv_hmb = vector.broadcast %kv_hm : f64 to vector<3xf64>\n"
     << indent << "  %kv_sq = arith.mulf %kv_v, %kv_v : vector<3xf64>\n"
     << indent << "  %kv_ke = arith.mulf %kv_hmb, %kv_sq : vector<3xf64>\n"
     << indent << "  md.yield %kv_ke : vector<3xf64>\n"
     << indent << "} : vector<3xf64>\n";
}

/// `result`, the scale of each axis of the reference positions of the
/// restraints: the edges of the cell over those of the file
/// (`%rest_edges`).
static void emitReferenceScale(llvm::raw_ostream &os, StringRef indent,
                               StringRef result, const std::string &e0,
                               const std::string &e1, const std::string &e2) {
  os << indent << result << "_e = vector.from_elements " << e0 << ", " << e1
     << ", " << e2 << " : vector<3xf64>\n"
     << indent << result << " = arith.divf " << result
     << "_e, %rest_edges : vector<3xf64>\n";
}

/// The sum of the elements of the vector of three `vector`.
static void emitSum3(llvm::raw_ostream &os, StringRef result,
                     StringRef vector, StringRef indent) {
  os << indent << result << " = vector.reduction <add>, " << vector
     << " : vector<3xf64> into f64\n";
}

/// The trace of the virial `virial`, which enters the pressure.
static void emitTrace(llvm::raw_ostream &os, StringRef result,
                      StringRef virial, StringRef indent) {
  for (StringRef part : {"0", "4", "8"})
    os << indent << result << "_" << part << " = vector.extract " << virial
       << "[" << part << "] : f64 from vector<9xf64>\n";
  os << indent << result << "_04 = arith.addf " << result << "_0, " << result
     << "_4 : f64\n"
     << indent << result << " = arith.addf " << result << "_04, " << result
     << "_8 : f64\n";
}

void Builder::emitPrograms() {
  if (control.minimize) {
    emitDescend();
    return;
  }
  std::string evaluate = "md.evaluate @energy(%x1, %cell" + getFieldValues() +
                         getTimeValue("%time") + ")";
  std::string signature =
      "(!vec, !md.cell" + getFieldTypes() + getTimeType() + ")";
  // Virtual sites are placed after the positions move, and the forces on
  // them are moved to their atoms before the velocities do. Rigid waters
  // are constrained before the sites are placed.
  bool sites = hasSites();
  bool settles = hasSettles();
  std::vector<const Program::TupleSet *> shakeSets = getShakeSets();
  bool constraints = settles || !shakeSets.empty();

  // With restraints the names of an evaluation have `p`, and the
  // restraints give those that the step goes on with.
  std::string held = hasRestraints() ? "p" : "";
  // The velocities that leave none along a constrained bond (the second
  // half of RATTLE), with the virial of the impulses if `virial` is given.
  // Returns the velocities and the virial.
  auto project = [&](StringRef x, StringRef input, StringRef base,
                     std::string virial) {
    std::string current = input.str();
    unsigned steps = (settles ? 1 : 0) + shakeSets.size(), step = 0;
    auto next = [&]() {
      ++step;
      return step == steps ? base.str()
                           : (base + "c" + std::to_string(step)).str();
    };
    if (settles) {
      std::string result = next();
      std::string w = step == steps ? "%w1c" : "%w1c1";
      std::string sum = emitSettleVelocities("  ", x, current, result,
                                             virial, w);
      if (!virial.empty())
        virial = sum;
      current = result;
    }
    for (const Program::TupleSet *set : shakeSets) {
      std::string result = next();
      std::string w = "%w1s_" + set->name;
      std::string sum = emitShakeVelocities("  ", x, current, *set, result,
                                            virial, w);
      if (!virial.empty())
        virial = sum;
      current = result;
    }
    return std::make_pair(current, virial);
  };

  // Velocity Verlet (Swope et al., J. Chem. Phys. 76, 637 (1982)) stores
  // the velocities of the time of the positions; leapfrog, those half a
  // step behind (design-m1.md, Section 23). Both carry the forces of the
  // positions, so that a step evaluates once, at its end. The step of
  // energy returns the energy and the virial of the new positions, and
  // with leapfrog also the velocities of their time, which the energies
  // and the barostat take.
  //
  // With the barostat of Trotter type (D92), the last step of a period of
  // coupling, `step_trotter`, scales the cell in the middle of its drift
  // by the factor `%mu`: it drifts half, scales the positions by μ (each
  // rigid group with its center of mass) and the velocities by 1/μ, and
  // drifts the other half ([Bernetti2020], SI Sec. V.C, eq. S12a-d; its
  // eq. S13a, which writes the four as one, leaves out the scaling of
  // q(t) and has Δt for Δt/2). Leapfrog drifts with the velocities of the
  // middle of the step as velocity Verlet does, so the scaling is the same.
  // It returns as the step of energy does, and the kinetic energy of the
  // velocities that it scaled.
  bool leapfrog = isLeapfrog();
  bool trotter = usesTrotter();
  // The steps of the barostat of Trotter type need the virial, and the
  // energy only where the log takes it: `step_virial` gives the pressure of
  // the strain, `step_trotter` scales, `step_trotter_energy` scales at the
  // end of an interval between energies.
  // With constraints, the steps whose rows the log writes measure the half
  // steps (`measures`); the barostat that closes a period without scaling
  // within the drift then takes `step_coupling`, a step of energy that
  // does not.
  struct Kind {
    const char *name;
    bool energy, virial, scales, measures;
  };
  llvm::SmallVector<Kind> kinds = {{"step", false, false, false, false},
                                   {"step_energy", true, true, false, true}};
  if (trotter) {
    if (!scalesEveryStep())
      kinds.push_back({"step_virial", false, true, false, false});
    kinds.push_back(
        {"step_trotter", false, countsAfterScaling(), true, false});
    kinds.push_back({"step_trotter_energy", true, true, true, true});
  } else if (control.barostat && measuresHalfSteps()) {
    kinds.push_back({"step_coupling", true, true, false, false});
  }
  for (const Kind &kind : kinds) {
    bool withEnergy = kind.energy, withVirial = kind.virial;
    bool scales = kind.scales;
    bool returnsCurrent = leapfrog && (withVirial || scales);
    // The steps of the rows of the log return the kinetic energy of each
    // axis at the half steps around their end, measured
    // (D203).
    bool measured = measuresHalfSteps() && kind.measures;
    // The steps around a scaling of Trotter type also return the trace of
    // the virial of the groups at their new positions, which the count of
    // the work takes (D116).
    bool molecular = trotter && withVirial &&
                     (scales || StringRef(kind.name) == "step_virial");
    // Brownian dynamics (D163b) neither conserves a volume of phase space
    // nor reverses; it couples to the bath.
    bool brownian = control.isBrownian();
    os << "dyn.program @" << kind.name
       << "(%x: !vec, %v: !vec, %f: !vec, %m: !real,\n"
       << "    %cell: !md.cell, %dt: f64"
       << (scales ? ", %mu: vector<3xf64>" : "")
       << getScaleParameter() << getFieldParameters() << getNoiseParameter()
       << ")\n"
       << "    -> (!vec, !vec, !vec" << (withEnergy ? ", f64" : "")
       << (withVirial ? ", vector<9xf64>" : "")
       << (molecular ? ", vector<3xf64>" : "")
       << (returnsCurrent ? ", !vec" : "")
       << (scales ? ", vector<3xf64>" : "")
       << (measured ? ", vector<3xf64>" : "")
       << (measured && withEnergy && reportsSolvent() ? ", vector<3xf64>"
                                                      : "")
       << ")\n"
       << "    attributes {"
       << (leapfrog ? "velocity_offset = -0.5,\n                " : "")
       << (brownian ? "provides = [\"thermostatting\"]} {\n"
                    : "provides = [\"symplectic\", \"time_reversible\"]} {\n")
       << "  %c = arith.constant 5.0e-01 : f64\n"
       << "  %half = arith.mulf %c, %dt : f64\n";
    // The time at the end of the step, at which its forces are evaluated
    // (D145).
    if (control.usesTime)
      os << "  %time_steps = arith.sitofp %noise_step : i64 to f64\n"
         << "  %time = arith.mulf %time_steps, %dt : f64\n";
    if (brownian) {
      // The velocity of the step, which the drift takes over Δt; the
      // velocities stored are those of the displacement, Δx / Δt.
      emitBrownianVelocity("  ", "%f", "%v1");
    } else if (!leapfrog) {
      os << "  %v1 = dyn.kick %v, %f, %m, %half : !vec\n";
    } else if (withVirial && constraints) {
      // The velocities of the time of the positions, as velocity Verlet
      // has them, and its first half kick. The drift then takes the
      // positions where the kick of a whole step does, up to the
      // constraints, which bring both to the same place; the constraints
      // of the step then give the virial of the first half (D76).
      os << "  %v0u = dyn.kick %v, %f, %m, %half : !vec\n";
      std::string current = project("%x", "%v0u", "%v0", "").first;
      os << "  %v1 = dyn.kick " << current << ", %f, %m, %half : !vec\n";
    } else {
      os << "  %v1 = dyn.kick %v, %f, %m, %dt : !vec\n";
    }
    std::string drifting = "%v1";
    std::string drifted = sites || constraints ? "%x1d" : "%x1";
    if (scales) {
      // The scale of each axis, μ, the same for the three with isotropic
      // coupling (D119).
      os << "  %xh = dyn.drift %x, %v1, %half : !vec\n"
         << "  %mu_unit = arith.constant dense<1.0> : vector<3xf64>\n"
         << "  %mu_m1 = arith.subf %mu_unit, %mu : vector<3xf64>\n"
         << "  %muinv = arith.divf %mu_unit, %mu : vector<3xf64>\n";
      // Langevin dynamics acts before the scaling, which takes the kinetic
      // energy of the velocities that it scales.
      std::string middle = "%v1";
      if (control.isLangevin()) {
        emitLangevin("  ", "%v1", "%v1o");
        middle = "%v1o";
      }
      std::string scaled = emitGroupScaling("  ", "%xh", "%mu", "%mu_m1",
                                            "%cell", "%m", "%r_", "_t");
      // The kinetic energy of each axis of the velocities that the
      // scaling takes.
      emitKineticVector(os, "%khalf", middle, "%m", "  ");
      os << "  %v1t = md.map_particles gather(" << middle << " : !vec) {\n"
         << "  ^bb0(%v_i: vector<3xf64>):\n"
         << "    %v_s = arith.mulf %muinv, %v_i : vector<3xf64>\n"
         << "    md.yield %v_s : vector<3xf64>\n"
         << "  } : !vec\n"
         << "  " << drifted << " = dyn.drift " << scaled
         << ", %v1t, %half : !vec\n";
      drifting = "%v1t";
    } else if (control.isLangevin()) {
      // The middle scheme: half a drift, the friction and the noise, and
      // the other half ([Zhang2019]; BAOAB of [Leimkuhler2013]).
      os << "  %xh = dyn.drift %x, %v1, %half : !vec\n";
      emitLangevin("  ", "%v1", "%v1o");
      os << "  " << drifted << " = dyn.drift %xh, %v1o, %half : !vec\n";
      drifting = "%v1o";
    } else {
      os << "  " << drifted << " = dyn.drift %x, %v1, %dt : !vec\n";
    }
    // The constraints back to their lengths, and the velocities that take
    // the atoms there over the step (the first half of RATTLE).
    std::string velocities = drifting;
    std::vector<std::pair<const Program::TupleSet *, std::string>> changes;
    if (constraints) {
      std::string current = "%x1d";
      std::string constrained = sites ? "%x1s" : "%x1";
      unsigned steps = (settles ? 1 : 0) + shakeSets.size(), step = 0;
      auto next = [&]() {
        ++step;
        return step == steps ? constrained
                             : "%x1c" + std::to_string(step);
      };
      if (settles) {
        std::string result = next();
        emitSettlePositions("  ", "%x", current, "%dx1", result);
        changes.push_back({program.tupleSets.data(), "%dx1"});
        for (const Program::TupleSet &set : program.tupleSets)
          if (set.name == "settles")
            changes.back().first = &set;
        current = result;
      }
      for (const Program::TupleSet *set : shakeSets) {
        std::string result = next();
        emitShakePositions("  ", "%x", current, *set, result);
        changes.push_back({set, result + "_change"});
        current = result;
      }
      os << "  %one = arith.constant 1.0 : f64\n"
         << "  %rate = arith.divf %one, %dt : f64\n"
         << "  %v1c = md.map_particles gather(" << drifting << ", " << current
         << ", %x1d : !vec, !vec, !vec) {\n"
         << "  ^bb0(%vs_v: vector<3xf64>, %vs_c: vector<3xf64>, "
            "%vs_u: vector<3xf64>):\n"
         << "    %vs_d = arith.subf %vs_c, %vs_u : vector<3xf64>\n"
         << "    %vs_r = vector.broadcast %rate : f64 to vector<3xf64>\n"
         << "    %vs_dv = arith.mulf %vs_r, %vs_d : vector<3xf64>\n"
         << "    %vs_sum = arith.addf %vs_v, %vs_dv : vector<3xf64>\n"
         << "    md.yield %vs_sum : vector<3xf64>\n"
         << "  } : !vec\n";
      velocities = "%v1c";
    }
    if (sites)
      emitPlaceSites("  ", constraints ? "%x1s" : "%x1d", "%x1", "%r_");
    std::string raw = sites ? "e" : "";
    if (withEnergy)
      os << "  %u1" << held << ", %f1" << held << raw << ", %w1" << held
         << raw << " = " << evaluate
         << "\n      request [energy, forces, virial]\n"
         << "      : " << signature << " -> (f64, !vec, vector<9xf64>)\n";
    else if (withVirial)
      os << "  %f1" << held << raw << ", %w1" << held << raw << " = "
         << evaluate << "\n      request [forces, virial]\n"
         << "      : " << signature << " -> (!vec, vector<9xf64>)\n";
    else
      os << "  %f1" << held << raw << " = " << evaluate << " request [forces]\n"
         << "      : " << signature << " -> !vec\n";
    std::string virial = "%w1" + held;
    if (sites)
      virial = emitSpreadSites("  ", "%x1", "%f1" + held + "e", "%f1" + held,
                               "%r_", withVirial ? "%w1" + held + "e" : "",
                               "%w1" + held);
    if (hasRestraints()) {
      emitRestraints("  ", "%x1", "%p_", "%f1p", "%f1",
                     withEnergy ? "%u1p" : "", withEnergy ? "%u1" : "",
                     withVirial ? virial : "", withVirial ? "%w1" : "");
      virial = "%w1";
    }
    std::string groups;
    if (molecular) {
      emitDiagonal(os, "%gd1", virial, "  ");
      groups = emitMolecularDiagonal("  ", "%gd1", "%x1", "%f1", "%cell",
                                     "%m", "%r_", "1");
    }
    if (leapfrog && !withVirial && !scales) {
      os << "  dyn.return %x1, " << velocities
         << ", %f1 : !vec, !vec, !vec\n}\n\n";
      continue;
    }
    if (brownian) {
      // No kick: the velocities are those of the displacement, with the
      // constraints. The virial is that of the forces alone.
      os << "  dyn.return %x1, " << velocities << ", %f1"
         << (withEnergy ? ", %u1" : "") << (withVirial ? ", " + virial : "")
         << " : !vec, !vec, !vec" << (withEnergy ? ", f64" : "")
         << (withVirial ? ", vector<9xf64>" : "") << "\n}\n\n";
      continue;
    }
    // The second half kick, to the velocities of the time of the new
    // positions: those of the next step with velocity Verlet, and those of
    // the energies with leapfrog.
    os << "  %v2" << (constraints ? "u" : "") << " = dyn.kick " << velocities
       << ", %f1, %m, %half : !vec\n";
    // The virial of the constraints: the mean of those of the two halves of
    // the step, or, under Langevin dynamics, whose friction and noise act
    // within the drift, that of their forces at its end (D135).
    bool instantaneous = control.isLangevin();
    if (constraints && withVirial && !instantaneous) {
      unsigned index = 0;
      for (auto [set, change] : changes)
        virial = emitConstraintVirial("  ", *set, "%x", change, virial,
                                      "%w1x" + std::to_string(index++));
    }
    if (constraints) {
      std::string halves =
          project("%x1", "%v2u", "%v2",
                  withVirial && !instantaneous ? virial : "")
              .second;
      if (withVirial)
        virial = instantaneous
                     ? emitConstraintForceVirial("  ", "%x1", "%v2", "%f1",
                                                 virial)
                     : halves;
    }
    std::string stored = leapfrog ? velocities : "%v2";
    // The steps of energy also report the rigid waters, the solvent.
    bool solvent = measured && withEnergy && reportsSolvent();
    if (measured)
      emitHalfStepKinetic(velocities, leapfrog, solvent);
    if (withVirial || scales)
      os << "  dyn.return %x1, " << stored << ", %f1"
         << (withEnergy ? ", %u1" : "")
         << (withVirial ? ", " + virial : "")
         << (molecular ? ", " + groups : "")
         << (returnsCurrent ? ", %v2" : "") << (scales ? ", %khalf" : "")
         << (measured ? ", %khs" : "") << (solvent ? ", %kws" : "") << "\n"
         << "      : !vec, !vec, !vec" << (withEnergy ? ", f64" : "")
         << (withVirial ? ", vector<9xf64>" : "")
         << (molecular ? ", vector<3xf64>" : "")
         << (returnsCurrent ? ", !vec" : "")
         << (scales ? ", vector<3xf64>" : "")
         << (measured ? ", vector<3xf64>" : "")
         << (solvent ? ", vector<3xf64>" : "") << "\n";
    else
      os << "  dyn.return %x1, %v2, %f1 : !vec, !vec, !vec\n";
    os << "}\n\n";
  }
}

void Builder::emitWriteSolvent(StringRef indent, StringRef sums) {
  std::string base = sums.str();
  os << indent << base << "_k = vector.extract " << sums
     << "[0] : f64 from vector<3xf64>\n"
     << indent << base << "_h = vector.extract " << sums
     << "[1] : f64 from vector<3xf64>\n"
     << indent << "func.call @mdrtWriteSolvent(" << base << "_k, " << base
     << "_h) : (f64, f64) -> ()\n";
}

void Builder::emitHalfStepKinetic(StringRef velocities, bool leapfrog,
                                  bool solvent) {
  // The kinetic energy of each axis at the half steps around the end of
  // the step, K_half = [K(t - dt/2) + K(t + dt/2)] / 2 [Jung2019], from the
  // velocities after their constraints: `velocities`, those that this step
  // drifted with, and those that the next step will drift with, which are
  // taken here as that step takes them, by a kick, a drift, and the
  // constraints of the positions. The forces alone do not give them with
  // constraints (D203). Nothing here feeds the state
  // that the step returns.
  bool settles = hasSettles();
  std::vector<const Program::TupleSet *> shakeSets = getShakeSets();
  if (leapfrog)
    os << "  %v3k = dyn.kick " << velocities << ", %f1, %m, %dt : !vec\n";
  else
    os << "  %v3k = dyn.kick %v2, %f1, %m, %half : !vec\n";
  os << "  %x3d = dyn.drift %x1, %v3k, %dt : !vec\n";
  std::string current = "%x3d";
  unsigned steps = (settles ? 1 : 0) + shakeSets.size(), step = 0;
  auto next = [&]() {
    ++step;
    return step == steps ? std::string("%x3c")
                         : "%x3c" + std::to_string(step);
  };
  if (settles) {
    std::string result = next();
    emitSettlePositions("  ", "%x1", current, "%dx3", result);
    current = result;
  }
  for (const Program::TupleSet *set : shakeSets) {
    std::string result = next();
    emitShakePositions("  ", "%x1", current, *set, result);
    current = result;
  }
  os << "  %h3one = arith.constant 1.0 : f64\n"
     << "  %h3rate = arith.divf %h3one, %dt : f64\n"
     << "  %v3c = md.map_particles gather(%v3k, " << current
     << ", %x3d : !vec, !vec, !vec) {\n"
     << "  ^bb0(%vs_v: vector<3xf64>, %vs_c: vector<3xf64>, "
        "%vs_u: vector<3xf64>):\n"
     << "    %vs_d = arith.subf %vs_c, %vs_u : vector<3xf64>\n"
     << "    %vs_r = vector.broadcast %h3rate : f64 to vector<3xf64>\n"
     << "    %vs_dv = arith.mulf %vs_r, %vs_d : vector<3xf64>\n"
     << "    %vs_sum = arith.addf %vs_v, %vs_dv : vector<3xf64>\n"
     << "    md.yield %vs_sum : vector<3xf64>\n"
     << "  } : !vec\n"
     << "  %khs = md.sum_particles gather(" << velocities
     << ", %v3c, %m : !vec, !vec, !real) {\n"
     << "  ^bb0(%kh_a: vector<3xf64>, %kh_b: vector<3xf64>, %kh_m: f64):\n"
     << "    %kh_c = arith.constant 2.5e-01 : f64\n"
     << "    %kh_qm = arith.mulf %kh_c, %kh_m : f64\n"
     << "    %kh_qmb = vector.broadcast %kh_qm : f64 to vector<3xf64>\n"
     << "    %kh_aa = arith.mulf %kh_a, %kh_a : vector<3xf64>\n"
     << "    %kh_bb = arith.mulf %kh_b, %kh_b : vector<3xf64>\n"
     << "    %kh_s = arith.addf %kh_aa, %kh_bb : vector<3xf64>\n"
     << "    %kh_ke = arith.mulf %kh_qmb, %kh_s : vector<3xf64>\n"
     << "    md.yield %kh_ke : vector<3xf64>\n"
     << "  } : vector<3xf64>\n";
  if (!solvent)
    return;
  // The kinetic energies of the rigid waters, K and K_half, as the first
  // two elements of a vector.
  os << "  %kws = md.sum_tuples %r_settles, %x1, %cell\n"
     << "    coordinates(displacement(1, 0))\n"
     << "    gather(%v2, " << velocities << ", %v3c, %m"
     << " : !vec, !vec, !vec, !real) {\n"
     << "  ^bb0(%ws_r: vector<3xf64>, %ws_f0: vector<3xf64>, "
        "%ws_f1: vector<3xf64>, %ws_f2: vector<3xf64>, "
        "%ws_a0: vector<3xf64>, %ws_a1: vector<3xf64>, "
        "%ws_a2: vector<3xf64>, %ws_b0: vector<3xf64>, "
        "%ws_b1: vector<3xf64>, %ws_b2: vector<3xf64>, %ws_m0: f64, "
        "%ws_m1: f64, %ws_m2: f64):\n";
  std::string full, half;
  for (int j = 0; j != 3; ++j) {
    std::string n = std::to_string(j);
    os << "    %ws_ff" << n << " = arith.mulf %ws_f" << n << ", %ws_f" << n
       << " : vector<3xf64>\n"
       << "    %ws_fs" << n << " = vector.reduction <add>, %ws_ff" << n
       << " : vector<3xf64> into f64\n"
       << "    %ws_fm" << n << " = arith.mulf %ws_m" << n << ", %ws_fs" << n
       << " : f64\n"
       << "    %ws_aa" << n << " = arith.mulf %ws_a" << n << ", %ws_a" << n
       << " : vector<3xf64>\n"
       << "    %ws_bb" << n << " = arith.mulf %ws_b" << n << ", %ws_b" << n
       << " : vector<3xf64>\n"
       << "    %ws_hh" << n << " = arith.addf %ws_aa" << n << ", %ws_bb" << n
       << " : vector<3xf64>\n"
       << "    %ws_hs" << n << " = vector.reduction <add>, %ws_hh" << n
       << " : vector<3xf64> into f64\n"
       << "    %ws_hm" << n << " = arith.mulf %ws_m" << n << ", %ws_hs" << n
       << " : f64\n";
    if (j == 0) {
      full = "%ws_fm0";
      half = "%ws_hm0";
    } else {
      os << "    %ws_ft" << n << " = arith.addf " << full << ", %ws_fm" << n
         << " : f64\n"
         << "    %ws_ht" << n << " = arith.addf " << half << ", %ws_hm" << n
         << " : f64\n";
      full = "%ws_ft" + n;
      half = "%ws_ht" + n;
    }
  }
  os << "    %ws_c2 = arith.constant 5.0e-01 : f64\n"
     << "    %ws_c4 = arith.constant 2.5e-01 : f64\n"
     << "    %ws_k = arith.mulf %ws_c2, " << full << " : f64\n"
     << "    %ws_kh = arith.mulf %ws_c4, " << half << " : f64\n"
     << "    %ws_z = arith.constant 0.0 : f64\n"
     << "    %ws_v = vector.from_elements %ws_k, %ws_kh, %ws_z : vector<3xf64>\n"
     << "    md.yield %ws_v : vector<3xf64>\n"
     << "  } : !rel_settles, !vec -> vector<3xf64>\n";
}

void Builder::emitLangevin(StringRef indent, StringRef velocities,
                           StringRef result) {
  // v' = c v + sqrt((1 - c^2) k_B T / m) R, with c = exp(-gamma dt) and R
  // three standard normal numbers.
  double c = std::exp(-control.friction * control.timestep);
  double kT = units::boltzmann * control.temperature;
  emitNoise(indent, velocities, result, c, std::sqrt((1.0 - c * c) * kT));
}

void Builder::emitBrownianVelocity(StringRef indent, StringRef forces,
                                   StringRef result) {
  // The step of Ermak and McCammon, J. Chem. Phys. 69, 1352 (1978), without
  // hydrodynamic interactions: Δx = Δt F / (m γ) + √(2 k_B T Δt /
  // (m γ)) R, written as a velocity that the positions drift by over Δt. A
  // particle of mass 0 (a virtual site, placed after the drift) does not
  // drift; the step does not read the velocities of the step before.
  double gamma = control.friction;
  double kT = units::boltzmann * control.temperature;
  std::string in = (indent + "  ").str();
  std::string drift = (result + "_drift").str();
  os << indent << drift << " = md.map_particles gather(" << forces
     << ", %m : !vec, !real) {\n"
     << indent << "^bb0(%bf: vector<3xf64>, %bm: f64):\n"
     << in << "%bzero = arith.constant 0.0 : f64\n"
     << in << "%bmassive = arith.cmpf ogt, %bm, %bzero : f64\n"
     << in << "%bone = arith.constant 1.0 : f64\n"
     << in << "%bsafe = arith.select %bmassive, %bm, %bone : f64\n"
     << in << "%bgamma = arith.constant " << formatReal(gamma) << " : f64\n"
     << in << "%bmg = arith.mulf %bsafe, %bgamma : f64\n"
     << in << "%bmgb = vector.broadcast %bmg : f64 to vector<3xf64>\n"
     << in << "%bv = arith.divf %bf, %bmgb : vector<3xf64>\n"
     << in << "%bstill = arith.constant dense<0.0> : vector<3xf64>\n"
     << in << "%bout = arith.select %bmassive, %bv, %bstill : vector<3xf64>\n"
     << in << "md.yield %bout : vector<3xf64>\n"
     << indent << "} : !vec\n";
  emitNoise(indent, drift, result, 1.0,
            std::sqrt(2.0 * kT / (gamma * control.timestep)));
}

void Builder::emitNoise(StringRef indent, StringRef velocities,
                        StringRef result, double c, double spread) {
  // The normal numbers R come from one block of Philox 4x32-10
  // [Salmon2011] under the key of A13: the seed; the counter of the step,
  // the number of the particle, and stream 2 in the high byte of the last
  // word. The four words give four uniform numbers, (w + 1/2) 2^-32, and
  // the method of Box and Muller two pairs of normal numbers, of which the
  // first three are taken. A particle of mass 0 keeps its velocity.
  auto i32 = [](uint32_t value) {
    return std::to_string(static_cast<int32_t>(value));
  };
  std::string in = (indent + "  ").str();
  os << indent << result << " = md.map_particles gather(" << velocities
     << ", %m, %noise_ids : !vec, !real, !ids) {\n"
     << indent << "^bb0(%lv: vector<3xf64>, %lm: f64, %lid: i32):\n"
     << in << "%l32 = arith.constant 32 : i64\n"
     << in << "%lshift = arith.shrui %noise_step, %l32 : i64\n"
     << in << "%lc0_0 = arith.trunci %noise_step : i64 to i32\n"
     << in << "%lc1_0 = arith.trunci %lshift : i64 to i32\n"
     << in << "%lc3_0 = arith.constant " << i32(2u << 24) << " : i32\n"
     << in << "%lma = arith.constant " << i32(0xD2511F53u) << " : i32\n"
     << in << "%lmb = arith.constant " << i32(0xCD9E8D57u) << " : i32\n";
  uint32_t k0 = static_cast<uint32_t>(control.seed);
  uint32_t k1 = static_cast<uint32_t>(control.seed >> 32);
  std::string c0 = "%lc0_0", c1 = "%lc1_0", c2 = "%lid", c3 = "%lc3_0";
  for (int round = 0; round != 10; ++round) {
    if (round != 0) {
      k0 += 0x9E3779B9u;
      k1 += 0xBB67AE85u;
    }
    std::string r = std::to_string(round);
    os << in << "%llo0_" << r << ", %lhi0_" << r
       << " = arith.mului_extended %lma, " << c0 << " : i32\n"
       << in << "%llo1_" << r << ", %lhi1_" << r
       << " = arith.mului_extended %lmb, " << c2 << " : i32\n"
       << in << "%lk0_" << r << " = arith.constant " << i32(k0)
       << " : i32\n"
       << in << "%lk1_" << r << " = arith.constant " << i32(k1)
       << " : i32\n"
       << in << "%la_" << r << " = arith.xori %lhi1_" << r << ", " << c1
       << " : i32\n"
       << in << "%lb_" << r << " = arith.xori %la_" << r << ", %lk0_" << r
       << " : i32\n"
       << in << "%lc_" << r << " = arith.xori %lhi0_" << r << ", " << c3
       << " : i32\n"
       << in << "%ld_" << r << " = arith.xori %lc_" << r << ", %lk1_" << r
       << " : i32\n";
    c0 = "%lb_" + r;
    c1 = "%llo1_" + r;
    c2 = "%ld_" + r;
    c3 = "%llo0_" + r;
  }
  os << in << "%lhalf = arith.constant 5.0e-01 : f64\n"
     << in << "%lscale = arith.constant 0x3DF0000000000000 : f64\n"
     << in << "%lminus2 = arith.constant -2.0 : f64\n"
     << in << "%ltwopi = arith.constant 6.283185307179586 : f64\n";
  std::string words[] = {c0, c1, c2, c3};
  for (int w = 0; w != 4; ++w) {
    std::string n = std::to_string(w);
    os << in << "%lf" << n << " = arith.uitofp " << words[w]
       << " : i32 to f64\n"
       << in << "%lg" << n << " = arith.addf %lf" << n << ", %lhalf : f64\n"
       << in << "%lu" << n << " = arith.mulf %lg" << n << ", %lscale : f64\n";
  }
  for (int pair = 0; pair != 2; ++pair) {
    std::string a = std::to_string(2 * pair), b = std::to_string(2 * pair + 1),
                p = std::to_string(pair);
    os << in << "%llog" << p << " = math.log %lu" << a << " : f64\n"
       << in << "%lsq" << p << " = arith.mulf %lminus2, %llog" << p
       << " : f64\n"
       << in << "%lr" << p << " = math.sqrt %lsq" << p << " : f64\n"
       << in << "%lt" << p << " = arith.mulf %ltwopi, %lu" << b << " : f64\n"
       << in << "%lcos" << p << " = math.cos %lt" << p << " : f64\n"
       << in << "%lncos" << p << " = arith.mulf %lr" << p << ", %lcos" << p
       << " : f64\n";
    if (pair == 0)
      os << in << "%lsin0 = math.sin %lt0 : f64\n"
         << in << "%lnsin0 = arith.mulf %lr0, %lsin0 : f64\n";
  }
  os << in << "%lnoise = vector.from_elements %lncos0, %lnsin0, %lncos1"
     << " : vector<3xf64>\n"
     << in << "%lzero = arith.constant 0.0 : f64\n"
     << in << "%lmassive = arith.cmpf ogt, %lm, %lzero : f64\n"
     << in << "%lone = arith.constant 1.0 : f64\n"
     << in << "%lsafe = arith.select %lmassive, %lm, %lone : f64\n"
     << in << "%lspread = arith.constant " << formatReal(spread) << " : f64\n"
     << in << "%lroot = math.sqrt %lsafe : f64\n"
     << in << "%lsigma = arith.divf %lspread, %lroot : f64\n"
     << in << "%lsigmab = vector.broadcast %lsigma : f64 to vector<3xf64>\n"
     << in << "%lkick = arith.mulf %lsigmab, %lnoise : vector<3xf64>\n"
     << in << "%lc = arith.constant dense<" << formatReal(c)
     << "> : vector<3xf64>\n"
     << in << "%ldamped = arith.mulf %lc, %lv : vector<3xf64>\n"
     << in << "%lnew = arith.addf %ldamped, %lkick : vector<3xf64>\n"
     << in << "%lout = arith.select %lmassive, %lnew, %lv : vector<3xf64>\n"
     << in << "md.yield %lout : vector<3xf64>\n"
     << indent << "} : !vec\n";
}

namespace {
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
} // namespace

void Builder::emitPlaceSites(StringRef indent, StringRef x, StringRef result,
                             StringRef relations) {
  std::string current = x.str();
  std::string inner = (indent + "  ").str();
  std::vector<const Program::TupleSet *> sets = getSiteSets();
  for (const Program::TupleSet *each : sets) {
    const Program::TupleSet &set = *each;
    bool amber = set.name == "sites_amber";
    bool last = each == sets.back();
    // The change of the position of each site: to where it is built, from
    // the owner, less where it is.
    std::string delta = (result + "_" + set.name).str();
    os << indent << delta << " = md.gather_tuples " << relations << set.name
       << ", " << current << ", %cell\n"
       << indent << "    coordinates(displacement(2, 1), displacement(3, 1), "
                    "displacement(0, 1))\n"
       << indent << "    tuple(%f_" << set.name << "_a, %f_" << set.name
       << "_b : !of_" << set.name << ", !of_" << set.name << ") {\n"
       << indent << "^bb0(%vs_j: vector<3xf64>, %vs_k: vector<3xf64>, "
                    "%vs_s: vector<3xf64>, %vs_a: f64, %vs_b: f64):\n";
    SiteKernel k(os, inner);
    std::string placed;
    if (amber) {
      AmberFrame frame(k, "%vs_j", "%vs_k");
      placed = k.scale(k.real("divf", "%vs_a", frame.lengthSum), frame.sum);
    } else {
      placed = k.vector("addf", k.scale("%vs_a", "%vs_j"),
                        k.scale("%vs_b", "%vs_k"));
    }
    std::string change = k.vector("subf", placed, "%vs_s");
    std::string zero = k.zero();
    os << inner << "md.yield " << change << ", " << zero << ", " << zero
       << ", " << zero
       << " : vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>\n"
       << indent << "} : !rel_" << set.name << ", !vec -> !vec\n";
    std::string next =
        last ? result.str() : (result + "_" + set.name + "_x").str();
    os << indent << next << " = md.map_particles gather(" << current << ", "
       << delta << " : !vec, !vec) {\n"
       << indent << "^bb0(%vs_x: vector<3xf64>, %vs_d: vector<3xf64>):\n"
       << inner << "%vs_sum = arith.addf %vs_x, %vs_d : vector<3xf64>\n"
       << inner << "md.yield %vs_sum : vector<3xf64>\n"
       << indent << "} : !vec\n";
    current = next;
  }
}

std::string Builder::emitSpreadSites(StringRef indent, StringRef x,
                                     StringRef f, StringRef result,
                                     StringRef relations, StringRef virial,
                                     StringRef virialResult) {
  std::string current = f.str(), currentVirial = virial.str();
  std::string inner = (indent + "  ").str();
  std::vector<const Program::TupleSet *> sets = getSiteSets();
  for (const Program::TupleSet *each : sets) {
    const Program::TupleSet &set = *each;
    bool amber = set.name == "sites_amber";
    bool last = each == sets.back();
    std::string delta = (result + "_" + set.name).str();
    os << indent << delta << " = md.gather_tuples " << relations << set.name
       << ", " << x << ", %cell\n"
       << indent << "    coordinates(displacement(2, 1), displacement(3, 1))\n"
       << indent << "    gather(" << current << " : !vec)\n"
       << indent << "    tuple(%f_" << set.name << "_a, %f_" << set.name
       << "_b : !of_" << set.name << ", !of_" << set.name << ") {\n"
       << indent << "^bb0(%vs_j: vector<3xf64>, %vs_k: vector<3xf64>, "
                    "%vs_f0: vector<3xf64>, %vs_f1: vector<3xf64>, "
                    "%vs_f2: vector<3xf64>, %vs_f3: vector<3xf64>, "
                    "%vs_a: f64, %vs_b: f64):\n";
    SiteKernel k(os, inner);
    std::array<std::string, 3> spread;
    if (amber) {
      AmberFrame frame(k, "%vs_j", "%vs_k");
      spread = frame.spread(k, "%vs_f0", "%vs_a");
    } else {
      std::string first = k.scale("%vs_a", "%vs_f0"),
                  second = k.scale("%vs_b", "%vs_f0");
      spread = {k.vector("subf", k.vector("subf", "%vs_f0", first), second),
                first, second};
    }
    std::string removed = k.negate("%vs_f0");
    os << inner << "md.yield " << removed << ", " << spread[0] << ", "
       << spread[1] << ", " << spread[2]
       << " : vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>\n"
       << indent << "} : !rel_" << set.name << ", !vec -> !vec\n";

    // The virial changes where the site is not a linear combination of the
    // atoms: W = Σ d ⊗ F with the forces on the atoms instead of that on
    // the site, Σ_k (x_k − x_s) ⊗ F_k.
    if (amber && !currentVirial.empty()) {
      std::string change = (virialResult + "_" + set.name).str();
      os << indent << change << " = md.sum_tuples " << relations << set.name
         << ", " << x << ", %cell\n"
         << indent << "    coordinates(displacement(2, 1), displacement(3, "
                      "1), displacement(1, 0), displacement(2, 0), "
                      "displacement(3, 0))\n"
         << indent << "    gather(" << current << " : !vec)\n"
         << indent << "    tuple(%f_" << set.name << "_a, %f_" << set.name
         << "_b : !of_" << set.name << ", !of_" << set.name << ") {\n"
         << indent << "^bb0(%vs_j: vector<3xf64>, %vs_k: vector<3xf64>, "
                      "%vs_e1: vector<3xf64>, %vs_e2: vector<3xf64>, "
                      "%vs_e3: vector<3xf64>, "
                      "%vs_f0: vector<3xf64>, %vs_f1: vector<3xf64>, "
                      "%vs_f2: vector<3xf64>, %vs_f3: vector<3xf64>, "
                      "%vs_a: f64, %vs_b: f64):\n";
      SiteKernel w(os, inner);
      AmberFrame frame(w, "%vs_j", "%vs_k");
      std::array<std::string, 3> forces = frame.spread(w, "%vs_f0", "%vs_a");
      std::array<std::string, 3> arms = {"%vs_e1", "%vs_e2", "%vs_e3"};
      std::string elements;
      for (int a = 0; a != 3; ++a) {
        std::string row;
        for (int m = 0; m != 3; ++m) {
          std::string term = w.scale(w.component(arms[m], a), forces[m]);
          row = row.empty() ? term : w.vector("addf", row, term);
        }
        for (int b = 0; b != 3; ++b)
          elements += (elements.empty() ? "" : ", ") + w.component(row, b);
      }
      os << inner << "%vs_w = vector.from_elements " << elements
         << " : vector<9xf64>\n"
         << inner << "md.yield %vs_w : vector<9xf64>\n"
         << indent << "} : !rel_" << set.name << ", !vec -> vector<9xf64>\n";
      std::string next = virialResult.str();
      os << indent << next << " = arith.addf " << currentVirial << ", "
         << change << " : vector<9xf64>\n";
      currentVirial = next;
    }

    std::string next =
        last ? result.str() : (result + "_" + set.name + "_f").str();
    os << indent << next << " = md.map_particles gather(" << current << ", "
       << delta << " : !vec, !vec) {\n"
       << indent << "^bb0(%vs_x: vector<3xf64>, %vs_d: vector<3xf64>):\n"
       << inner << "%vs_sum = arith.addf %vs_x, %vs_d : vector<3xf64>\n"
       << inner << "md.yield %vs_sum : vector<3xf64>\n"
       << indent << "} : !vec\n";
    current = next;
  }
  return currentVirial;
}

void Builder::emitSettlePositions(StringRef indent, StringRef old,
                                  StringRef x, StringRef change,
                                  StringRef result) {
  // Below double precision, the three bonds of the water by M-SHAKE: the
  // positions that SETTLE turns are of the size of the water, and their
  // rounding in f32 over the change of a step is an error of the
  // velocities, which heated JAC by 7 K in 2 ns; M-SHAKE carries the
  // change itself, a small number [Jung2026] (D112).
  if (control.precision != Precision::Double) {
    emitBondConstraints(indent, "settles", 3,
                        {{0, 1, "%vs_doh"}, {0, 2, "%vs_doh"},
                         {1, 2, "%vs_dhh"}},
                        {"%f_settles_doh", "%f_settles_dhh"}, old, x, change);
    emitAddChange(indent, x, change, result);
    return;
  }
  std::string inner = (indent + "  ").str();
  // In the frame of the old plane of the water, with the origin at the new
  // center of mass, the new triangle is the rigid one turned by three
  // angles (Miyamoto and Kollman 1992, Appendix A). The positions are
  // taken relative to the new oxygen, in the minimum image; the old ones
  // are moved by the same periods of the cell.
  os << indent << change << " = md.gather_tuples %r_settles, " << x
     << ", %cell\n"
     << indent << "    coordinates(displacement(1, 0), displacement(2, 0))\n"
     << indent << "    gather(" << old << ", " << x
     << ", %m : !vec, !vec, !real)\n"
     << indent << "    tuple(%f_settles_doh, %f_settles_dhh : !of_settles, "
                  "!of_settles) {\n"
     << indent << "^bb0(%vs_b1: vector<3xf64>, %vs_c1: vector<3xf64>, "
                  "%vs_o0: vector<3xf64>, %vs_o1: vector<3xf64>, "
                  "%vs_o2: vector<3xf64>, %vs_n0: vector<3xf64>, "
                  "%vs_n1: vector<3xf64>, %vs_n2: vector<3xf64>, "
                  "%vs_m0: f64, %vs_m1: f64, %vs_m2: f64, %vs_doh: f64, "
                  "%vs_dhh: f64):\n";
  SiteKernel k(os, inner);
  // The old bonds, in the periods of the new ones.
  auto oldBond = [&](StringRef bond, StringRef o, StringRef n) {
    std::string raw = k.vector("subf", n, "%vs_n0");
    std::string shift = k.vector("subf", bond, raw);
    return k.vector("addf", k.vector("subf", o, "%vs_o0"), shift);
  };
  std::string b0 = oldBond("%vs_b1", "%vs_o1", "%vs_n1");
  std::string c0 = oldBond("%vs_c1", "%vs_o2", "%vs_n2");
  // The rigid triangle: the oxygen at ra from the center of mass along the
  // bisector, the hydrogens at rb behind it and rc to each side.
  std::string two = k.constant(2.0), half = k.constant(0.5),
              one = k.constant(1.0);
  std::string total =
      k.real("addf", "%vs_m0", k.real("mulf", two, "%vs_m1"));
  std::string rc = k.real("mulf", half, "%vs_dhh");
  std::string height = k.root(k.real(
      "subf", k.real("mulf", "%vs_doh", "%vs_doh"), k.real("mulf", rc, rc)));
  std::string ra = k.real(
      "divf", k.real("mulf", k.real("mulf", two, "%vs_m1"), height), total);
  std::string rb = k.real("subf", height, ra);
  // The new positions about their center of mass.
  std::string center = k.scale(k.real("divf", "%vs_m1", total),
                               k.vector("addf", "%vs_b1", "%vs_c1"));
  std::string a1 = k.negate(center);
  std::string b1 = k.vector("subf", "%vs_b1", center);
  std::string c1 = k.vector("subf", "%vs_c1", center);
  std::string z = k.unit(k.cross(b0, c0));
  std::string xAxis = k.unit(k.cross(a1, z));
  std::string y = k.cross(z, xAxis);
  std::string xb0 = k.dot(b0, xAxis), yb0 = k.dot(b0, y);
  std::string xc0 = k.dot(c0, xAxis), yc0 = k.dot(c0, y);
  std::string za1 = k.dot(a1, z);
  std::string xb1 = k.dot(b1, xAxis), yb1 = k.dot(b1, y), zb1 = k.dot(b1, z);
  std::string xc1 = k.dot(c1, xAxis), yc1 = k.dot(c1, y), zc1 = k.dot(c1, z);
  auto complement = [&](StringRef sine) {
    return k.root(k.real("subf", one, k.real("mulf", sine, sine)));
  };
  std::string sinPhi = k.real("divf", za1, ra);
  std::string cosPhi = complement(sinPhi);
  std::string sinPsi =
      k.real("divf", k.real("subf", zb1, zc1),
             k.real("mulf", k.real("mulf", two, rc), cosPhi));
  std::string cosPsi = complement(sinPsi);
  std::string ya2 = k.real("mulf", ra, cosPhi);
  std::string xb2 = k.real("mulf", k.real("subf", k.constant(0.0), rc),
                           cosPsi);
  std::string t1 = k.real("mulf", k.real("subf", k.constant(0.0), rb),
                          cosPhi);
  std::string t2 = k.real("mulf", k.real("mulf", rc, sinPsi), sinPhi);
  std::string yb2 = k.real("subf", t1, t2), yc2 = k.real("addf", t1, t2);
  auto sum3 = [&](StringRef a, StringRef b, StringRef c) {
    return k.real("addf", k.real("addf", a, b), c);
  };
  std::string alpha = sum3(k.real("mulf", xb2, k.real("subf", xb0, xc0)),
                           k.real("mulf", yb0, yb2),
                           k.real("mulf", yc0, yc2));
  std::string beta = sum3(k.real("mulf", xb2, k.real("subf", yc0, yb0)),
                          k.real("mulf", xb0, yb2),
                          k.real("mulf", xc0, yc2));
  std::string gamma = k.real(
      "addf",
      k.real("subf", k.real("mulf", xb0, yb1), k.real("mulf", xb1, yb0)),
      k.real("subf", k.real("mulf", xc0, yc1), k.real("mulf", xc1, yc0)));
  std::string both = k.real("addf", k.real("mulf", alpha, alpha),
                            k.real("mulf", beta, beta));
  std::string sinTheta = k.real(
      "divf",
      k.real("subf", k.real("mulf", alpha, gamma),
             k.real("mulf", beta,
                    k.root(k.real("subf", both,
                                  k.real("mulf", gamma, gamma))))),
      both);
  std::string cosTheta = complement(sinTheta);
  std::string zero = k.constant(0.0);
  std::string xa3 = k.real("subf", zero, k.real("mulf", ya2, sinTheta));
  std::string ya3 = k.real("mulf", ya2, cosTheta);
  std::string xb3 = k.real("subf", k.real("mulf", xb2, cosTheta),
                           k.real("mulf", yb2, sinTheta));
  std::string yb3 = k.real("addf", k.real("mulf", xb2, sinTheta),
                           k.real("mulf", yb2, cosTheta));
  std::string xc3 = k.real("subf", k.real("subf", zero,
                                          k.real("mulf", xb2, cosTheta)),
                           k.real("mulf", yc2, sinTheta));
  std::string yc3 = k.real("addf", k.real("subf", zero,
                                          k.real("mulf", xb2, sinTheta)),
                           k.real("mulf", yc2, cosTheta));
  // Back to the cell, relative to the new oxygen.
  auto back = [&](StringRef px, StringRef py, StringRef pz) {
    return k.vector("addf", k.combine(px, xAxis, py, y, pz, z), center);
  };
  std::string a3 = back(xa3, ya3, za1);
  std::string b3 = back(xb3, yb3, zb1);
  std::string c3 = back(xc3, yc3, zc1);
  std::string db = k.vector("subf", b3, "%vs_b1");
  std::string dc = k.vector("subf", c3, "%vs_c1");
  os << inner << "md.yield " << a3 << ", " << db << ", " << dc
     << " : vector<3xf64>, vector<3xf64>, vector<3xf64>\n"
     << indent << "} : !rel_settles, !vec -> !vec\n";
  os << indent << result << " = md.map_particles gather(" << x << ", "
     << change << " : !vec, !vec) {\n"
     << indent << "^bb0(%vs_x: vector<3xf64>, %vs_d: vector<3xf64>):\n"
     << inner << "%vs_sum = arith.addf %vs_x, %vs_d : vector<3xf64>\n"
     << inner << "md.yield %vs_sum : vector<3xf64>\n"
     << indent << "} : !vec\n";
}

std::string Builder::emitConstraintVirial(StringRef indent,
                                          const Program::TupleSet &set,
                                          StringRef old, StringRef change,
                                          StringRef virial,
                                          StringRef virialResult) {
  // The forces of the constraints over the first half of the step: the
  // drift took x + dt v + dt² F / 2m to x + Δ, so G = 2 m Δ / dt², along
  // the bonds before the drift. Their virial Σ (x_j − x_0) ⊗ G_j, taken at
  // those positions, with the weight ½: with that of the second half, the
  // virial of the constraints is the mean of the two, as the kinetic
  // energy of the pressure is (D45).
  unsigned count = set.arity - 1;
  std::string inner = (indent + "  ").str();
  std::string coordinates, arguments;
  for (unsigned k = 1; k <= count; ++k) {
    coordinates += (k == 1 ? "" : ", ") +
                   ("displacement(" + std::to_string(k) + ", 0)");
    arguments += "%vs_r" + std::to_string(k) + ": vector<3xf64>, ";
  }
  for (unsigned k = 0; k <= count; ++k)
    arguments += "%vs_c" + std::to_string(k) + ": vector<3xf64>, ";
  for (unsigned k = 0; k <= count; ++k)
    arguments += "%vs_m" + std::to_string(k) + ": f64" +
                 (k == count ? "" : ", ");
  std::string sum = (virialResult + "_first").str();
  os << indent << sum << " = md.sum_tuples %r_" << set.name << ", " << old
     << ", %cell\n"
     << indent << "    coordinates(" << coordinates << ")\n"
     << indent << "    gather(" << change << ", %m : !vec, !real) {\n"
     << indent << "^bb0(" << arguments << "):\n";
  SiteKernel w(os, inner);
  std::string rate = w.real("divf", w.constant(1.0),
                            w.real("mulf", "%dt", "%dt"));
  std::vector<std::string> forces;
  for (unsigned j = 1; j <= count; ++j)
    forces.push_back(w.scale(
        w.real("mulf", "%vs_m" + std::to_string(j), rate),
        "%vs_c" + std::to_string(j)));
  std::string elements;
  for (int a = 0; a != 3; ++a) {
    std::string row;
    for (unsigned j = 1; j <= count; ++j) {
      std::string term = w.scale(w.component("%vs_r" + std::to_string(j), a),
                                 forces[j - 1]);
      row = row.empty() ? term : w.vector("addf", row, term);
    }
    for (int b = 0; b != 3; ++b)
      elements += (elements.empty() ? "" : ", ") + w.component(row, b);
  }
  os << inner << "%vs_w = vector.from_elements " << elements
     << " : vector<9xf64>\n"
     << inner << "md.yield %vs_w : vector<9xf64>\n"
     << indent << "} : !rel_" << set.name << ", !vec -> vector<9xf64>\n";
  os << indent << virialResult << " = arith.addf " << virial << ", " << sum
     << " : vector<9xf64>\n";
  return virialResult.str();
}

/// Emits the solution of `A x = b` for a system of at most 3 × 3, by
/// Gaussian elimination without pivoting: the matrices of the constraints
/// of a group are dominated by their diagonals.
/// The iterations of Newton of the positions of SHAKE: from the error of
/// a step of 4 fs with repartitioned masses, about 1e-2 of a bond, four
/// reach the rounding of f64; two more are a margin.
static const int shakeIterations = 6;

static std::vector<std::string>
emitSmallSolve(SiteKernel &k, std::vector<std::vector<std::string>> A,
               std::vector<std::string> b) {
  unsigned count = b.size();
  for (unsigned c = 0; c != count; ++c)
    for (unsigned r = c + 1; r != count; ++r) {
      std::string factor = k.real("divf", A[r][c], A[c][c]);
      for (unsigned d = c; d != count; ++d)
        A[r][d] = k.real("subf", A[r][d], k.real("mulf", factor, A[c][d]));
      b[r] = k.real("subf", b[r], k.real("mulf", factor, b[c]));
    }
  std::vector<std::string> x(count);
  for (unsigned c = count; c-- != 0;) {
    std::string sum = b[c];
    for (unsigned d = c + 1; d != count; ++d)
      sum = k.real("subf", sum, k.real("mulf", A[c][d], x[d]));
    x[c] = k.real("divf", sum, A[c][c]);
  }
  return x;
}

void Builder::emitShakePositions(StringRef indent, StringRef old,
                                 StringRef x, const Program::TupleSet &set,
                                 StringRef result) {
  unsigned count = set.arity - 1;
  std::string change = (result + "_change").str();
  // The bonds of the heavy atom 0 to its hydrogens.
  std::vector<ConstrainedBond> bonds;
  std::vector<std::string> tuple;
  for (unsigned k = 1; k <= count; ++k) {
    bonds.push_back({0, k, "%vs_d" + std::to_string(k)});
    tuple.push_back("%f_" + set.name + "_d" + std::to_string(k));
  }
  emitBondConstraints(indent, set.name, set.arity, bonds, tuple, old, x,
                      change);
  emitAddChange(indent, x, change, result);
}

void Builder::emitAddChange(StringRef indent, StringRef x, StringRef change,
                            StringRef result) {
  std::string inner = (indent + "  ").str();
  os << indent << result << " = md.map_particles gather(" << x << ", "
     << change << " : !vec, !vec) {\n"
     << indent << "^bb0(%vs_x: vector<3xf64>, %vs_d: vector<3xf64>):\n"
     << inner << "%vs_sum = arith.addf %vs_x, %vs_d : vector<3xf64>\n"
     << inner << "md.yield %vs_sum : vector<3xf64>\n"
     << indent << "} : !vec\n";
}

void Builder::emitBondConstraints(StringRef indent, StringRef setName,
                                  unsigned arity,
                                  llvm::ArrayRef<ConstrainedBond> bonds,
                                  llvm::ArrayRef<std::string> tuple, StringRef old,
                                  StringRef x, StringRef change) {
  unsigned count = bonds.size();
  std::string inner = (indent + "  ").str();
  std::string coordinates, tupleTypes, arguments;
  for (auto [k, bond] : llvm::enumerate(bonds)) {
    coordinates += (k == 0 ? "" : ", ") +
                   ("displacement(" + std::to_string(bond.second) + ", " +
                    std::to_string(bond.first) + ")");
    arguments += "%vs_r" + std::to_string(k) + ": vector<3xf64>, ";
  }
  for (StringRef field : {"o", "n"})
    for (unsigned j = 0; j != arity; ++j)
      arguments += ("%vs_" + field + std::to_string(j) + ": vector<3xf64>, ")
                       .str();
  for (unsigned j = 0; j != arity; ++j)
    arguments += "%vs_m" + std::to_string(j) + ": f64, ";
  std::string tupleList;
  for (auto [k, name] : llvm::enumerate(tuple)) {
    tupleList += (k == 0 ? "" : ", ") + name;
    tupleTypes += (k == 0 ? "" : ", ") + ("!of_" + setName).str();
  }
  // The arguments of the tuple fields: the lengths the bonds name.
  std::vector<std::string> fieldArguments;
  for (const ConstrainedBond &bond : bonds)
    if (!llvm::is_contained(fieldArguments, bond.length))
      fieldArguments.push_back(bond.length);
  for (auto [k, name] : llvm::enumerate(fieldArguments))
    arguments += name + ": f64" +
                 (k + 1 == fieldArguments.size() ? "" : ", ");
  os << indent << change << " = md.gather_tuples %r_" << setName << ", " << x
     << ", %cell\n"
     << indent << "    coordinates(" << coordinates << ")\n"
     << indent << "    gather(" << old << ", " << x
     << ", %m : !vec, !vec, !real)\n"
     << indent << "    tuple(" << tupleList << " : " << tupleTypes << ") {\n"
     << indent << "^bb0(" << arguments << "):\n";
  SiteKernel k(os, inner);
  // The old bonds, in the periods of the new ones.
  std::vector<std::string> olds, inverse;
  std::string one = k.constant(1.0), two = k.constant(2.0);
  for (unsigned j = 0; j != arity; ++j)
    inverse.push_back(k.real("divf", one, "%vs_m" + std::to_string(j)));
  for (auto [i, bond] : llvm::enumerate(bonds)) {
    std::string a = std::to_string(bond.first), b = std::to_string(bond.second);
    std::string raw = k.vector("subf", "%vs_n" + b, "%vs_n" + a);
    std::string shift = k.vector("subf", "%vs_r" + std::to_string(i), raw);
    olds.push_back(k.vector(
        "addf", k.vector("subf", "%vs_o" + b, "%vs_o" + a), shift));
  }
  std::string zero = k.zero();
  // For one bond the fixed-old-direction projection is a quadratic.
  // With r the predicted bond, s the old bond, and r' = r + t s,
  // t = (d²-r²)/(r·s + sqrt((r·s)² + s²(d²-r²))). Rationalizing the
  // near root avoids subtracting nearly equal numbers. Keep the same
  // mass-weighted impulses, periodic image, and small-change storage as
  // M-SHAKE. A checked fast path falls back to the existing Newton solve.
  bool analytic = control.analyticBonds && count == 1;
  if (analytic) {
    const auto &bond = bonds.front();
    std::string r = "%vs_r0", s = olds.front();
    std::string d2 = k.real("mulf", bond.length, bond.length);
    std::string error = k.real("subf", d2, k.dot(r, r));
    std::string rs = k.dot(r, s);
    std::string discriminant = k.real(
        "addf", k.real("mulf", rs, rs), k.real("mulf", k.dot(s, s), error));
    std::string t = k.real("divf", error,
                           k.real("addf", rs, k.root(discriminant)));
    std::string lambda = k.real(
        "divf", t, k.real("addf", inverse[bond.first], inverse[bond.second]));
    std::string push = k.scale(lambda, s);
    std::string da = k.negate(k.scale(inverse[bond.first], push));
    std::string db = k.scale(inverse[bond.second], push);
    std::string corrected = k.vector("subf", k.vector("addf", r, db), da);
    std::string residual = k.real("subf", k.dot(corrected, corrected), d2);
    std::string tolerance = k.real(
        "mulf", d2, k.constant(control.precision == Precision::Double
                                   ? 1.0e-12 : 1.0e-6));
    std::string scalarZero = k.constant(0.0);
    os << inner << "%vs_residual = math.absf " << residual << " : f64\n"
       << inner << "%vs_accurate = arith.cmpf ole, %vs_residual, "
       << tolerance << " : f64\n"
       << inner << "%vs_forward = arith.cmpf ogt, " << rs << ", "
       << scalarZero << " : f64\n"
       << inner << "%vs_analytic = arith.andi %vs_accurate, %vs_forward : i1\n"
       << inner << "%vs_checked0, %vs_checked1 = scf.if %vs_analytic -> "
                   "(vector<3xf64>, vector<3xf64>) {\n"
       << inner << "  scf.yield " << da << ", " << db
       << " : vector<3xf64>, vector<3xf64>\n"
       << inner << "} else {\n";
    inner += "  ";
  }
  // The members move along the old bonds s_l: for each bond l = (a, b),
  // b by λ_l s_l / m_b and a by −λ_l s_l / m_a. Each iteration of Newton
  // solves the constraints linearized at the current bonds r_k exactly,
  //   Σ_l 2 (r_k · s_l) w_kl λ_l = d_k² − r_k²,
  // with w_kl what bond l moves bond k by per unit of λ_l s_l: the inverse
  // mass of each member that the two share, with its sign; the error
  // squares with each iteration whatever the masses (M-SHAKE
  // [Krautler2001]). The atoms carry what they moved, small numbers, so
  // that the change keeps its digits in f32: its rounding becomes an
  // error of the velocities a step divides by [Jung2026].
  os << inner << "%vs_c0 = arith.constant 0 : index\n"
     << inner << "%vs_c1 = arith.constant 1 : index\n"
     << inner << "%vs_iterations = arith.constant " << shakeIterations
     << " : index\n";
  std::string results, inits, types;
  for (unsigned j = 0; j != arity; ++j) {
    std::string n = std::to_string(j);
    results += (j == 0 ? "" : ", ") + ("%vs_a" + n);
    inits += (j == 0 ? "" : ", ") + ("%vs_s" + n + " = " + zero);
    types += (j == 0 ? "" : ", ") + std::string("vector<3xf64>");
  }
  os << inner << results << " = scf.for %vs_sweep = %vs_c0 to "
                            "%vs_iterations step %vs_c1\n"
     << inner << "    iter_args(" << inits << ") -> (" << types << ") {\n";
  SiteKernel l(os, inner + "  ", "%vsl");
  std::vector<std::string> moved;
  for (unsigned j = 0; j != arity; ++j)
    moved.push_back("%vs_s" + std::to_string(j));
  std::vector<std::string> current, error;
  for (auto [i, bond] : llvm::enumerate(bonds)) {
    current.push_back(l.vector(
        "subf",
        l.vector("addf", "%vs_r" + std::to_string(i), moved[bond.second]),
        moved[bond.first]));
    error.push_back(l.real("subf", l.real("mulf", bond.length, bond.length),
                           l.dot(current.back(), current.back())));
  }
  // w_kl: what bond l moves bond k by, per unit of λ_l s_l: the inverse
  // masses of the members the two share, with their signs.
  auto weightOf = [&](const ConstrainedBond &k,
                      const ConstrainedBond &bond) -> std::string {
    std::string sum;
    auto add = [&](bool negative, unsigned j) {
      if (sum.empty())
        sum = negative ? l.real("subf", l.constant(0.0), inverse[j])
                       : inverse[j];
      else
        sum = l.real(negative ? "subf" : "addf", sum, inverse[j]);
    };
    if (k.second == bond.second)
      add(false, k.second);
    if (k.second == bond.first)
      add(true, k.second);
    if (k.first == bond.second)
      add(true, k.first);
    if (k.first == bond.first)
      add(false, k.first);
    return sum.empty() ? l.constant(0.0) : sum;
  };
  std::vector<std::vector<std::string>> A(count,
                                          std::vector<std::string>(count));
  for (unsigned i = 0; i != count; ++i)
    for (unsigned j = 0; j != count; ++j)
      A[i][j] = l.real("mulf", l.real("mulf", two, l.dot(current[i],
                                                          olds[j])),
                       weightOf(bonds[i], bonds[j]));
  std::vector<std::string> lambda = emitSmallSolve(l, A, error);
  for (auto [j, bond] : llvm::enumerate(bonds)) {
    std::string push = l.scale(lambda[j], olds[j]);
    moved[bond.second] = l.vector("addf", moved[bond.second],
                                  l.scale(inverse[bond.second], push));
    moved[bond.first] = l.vector("subf", moved[bond.first],
                                 l.scale(inverse[bond.first], push));
  }
  std::string yielded;
  for (unsigned j = 0; j != arity; ++j)
    yielded += (j == 0 ? "" : ", ") + moved[j];
  os << inner << "  scf.yield " << yielded << " : " << types << "\n"
     << inner << "}\n";
  if (analytic) {
    os << inner << "scf.yield " << results << " : " << types << "\n";
    inner.resize(inner.size() - 2);
    os << inner << "}\n";
    results = "%vs_checked0, %vs_checked1";
  }
  os << inner << "md.yield " << results << " : " << types << "\n"
     << indent << "} : !rel_" << setName << ", !vec -> !vec\n";
}

std::string Builder::emitShakeVelocities(StringRef indent, StringRef x,
                                         StringRef v,
                                         const Program::TupleSet &set,
                                         StringRef result, StringRef virial,
                                         StringRef virialResult) {
  unsigned count = set.arity - 1;
  std::string inner = (indent + "  ").str();
  std::string coordinates, tuple, tupleTypes, arguments;
  for (unsigned k = 1; k <= count; ++k) {
    coordinates += (k == 1 ? "" : ", ") + ("displacement(" +
                                           std::to_string(k) + ", 0)");
    tuple += (k == 1 ? "" : ", ") + ("%f_" + set.name + "_d" +
                                     std::to_string(k));
    tupleTypes += (k == 1 ? "" : ", ") + ("!of_" + set.name);
    arguments += "%vs_r" + std::to_string(k) + ": vector<3xf64>, ";
  }
  for (unsigned k = 0; k <= count; ++k)
    arguments += "%vs_v" + std::to_string(k) + ": vector<3xf64>, ";
  for (unsigned k = 0; k <= count; ++k)
    arguments += "%vs_m" + std::to_string(k) + ": f64, ";
  for (unsigned k = 1; k <= count; ++k)
    arguments += "%vs_d" + std::to_string(k) + ": f64" +
                 (k == count ? "" : ", ");
  auto header = [&](StringRef name, StringRef op, StringRef field) {
    os << indent << name << " = md." << op << " %r_" << set.name << ", " << x
       << ", %cell\n"
       << indent << "    coordinates(" << coordinates << ")\n"
       << indent << "    gather(" << field << ", %m : !vec, !real)\n"
       << indent << "    tuple(" << tuple << " : " << tupleTypes << ") {\n"
       << indent << "^bb0(" << arguments << "):\n";
  };
  // The impulses τ along the bonds that leave no velocity along them:
  // A τ = −b, A_ij = δ_ij / m_j + e_i · e_j / m_0, b_i = e_i · (v_i − v_0),
  // by Gaussian elimination. Returns the changes of the velocities.
  auto solve = [&](SiteKernel &k) {
    std::vector<std::string> units, rhs;
    std::string one = k.constant(1.0), zero = k.constant(0.0);
    std::string inverseCenter = k.real("divf", one, "%vs_m0");
    std::vector<std::string> inverse;
    for (unsigned j = 1; j <= count; ++j) {
      std::string n = std::to_string(j);
      units.push_back(k.unit("%vs_r" + n));
      inverse.push_back(k.real("divf", one, "%vs_m" + n));
      rhs.push_back(k.real(
          "subf", zero,
          k.dot(units.back(), k.vector("subf", "%vs_v" + n, "%vs_v0"))));
    }
    std::vector<std::vector<std::string>> A(count,
                                            std::vector<std::string>(count));
    for (unsigned i = 0; i != count; ++i)
      for (unsigned j = 0; j != count; ++j) {
        std::string coupling =
            k.real("mulf", k.dot(units[i], units[j]), inverseCenter);
        A[i][j] = i == j ? k.real("addf", coupling, inverse[i]) : coupling;
      }
    std::vector<std::string> impulse = emitSmallSolve(k, A, rhs);
    std::vector<std::string> changes(count + 1);
    std::string center = k.zero();
    for (unsigned j = 0; j != count; ++j) {
      std::string push = k.scale(impulse[j], units[j]);
      changes[j + 1] = k.scale(inverse[j], push);
      center = k.vector("subf", center, k.scale(inverseCenter, push));
    }
    changes[0] = center;
    return changes;
  };

  std::string change = (result + "_change").str();
  header(change, "gather_tuples", v);
  SiteKernel k(os, inner);
  std::vector<std::string> changes = solve(k);
  std::string yielded, types;
  for (unsigned j = 0; j <= count; ++j) {
    yielded += (j == 0 ? "" : ", ") + changes[j];
    types += (j == 0 ? "" : ", ") + std::string("vector<3xf64>");
  }
  os << inner << "md.yield " << yielded << " : " << types << "\n"
     << indent << "} : !rel_" << set.name << ", !vec -> !vec\n";

  std::string virialName = virial.str();
  if (!virial.empty()) {
    // The forces of the constraints over the second half of the step,
    // G = 2 m Δv / dt, and their virial Σ (x_j − x_0) ⊗ G_j, with the
    // weight ½: the virial of the constraints is the mean of those of the
    // two halves (emitConstraintVirial). The changes are read, not solved
    // again, as for SETTLE (#97); the clusters are disjoint.
    std::string sum = (virialResult + "_sum").str();
    header(sum, "sum_tuples", change);
    SiteKernel w(os, inner);
    std::vector<std::string> dv(count + 1);
    for (unsigned j = 0; j <= count; ++j)
      dv[j] = "%vs_v" + std::to_string(j);
    std::string two = w.constant(1.0);
    std::string elements;
    std::vector<std::string> forces;
    for (unsigned j = 1; j <= count; ++j)
      forces.push_back(w.scale(
          w.real("divf", w.real("mulf", two, "%vs_m" + std::to_string(j)),
                 "%dt"),
          dv[j]));
    for (int a = 0; a != 3; ++a) {
      std::string row;
      for (unsigned j = 1; j <= count; ++j) {
        std::string term = w.scale(
            w.component("%vs_r" + std::to_string(j), a), forces[j - 1]);
        row = row.empty() ? term : w.vector("addf", row, term);
      }
      for (int b = 0; b != 3; ++b)
        elements += (elements.empty() ? "" : ", ") + w.component(row, b);
    }
    os << inner << "%vs_w = vector.from_elements " << elements
       << " : vector<9xf64>\n"
       << inner << "md.yield %vs_w : vector<9xf64>\n"
       << indent << "} : !rel_" << set.name << ", !vec -> vector<9xf64>\n";
    os << indent << virialResult << " = arith.addf " << virial << ", " << sum
       << " : vector<9xf64>\n";
    virialName = virialResult.str();
  }
  os << indent << result << " = md.map_particles gather(" << v << ", "
     << change << " : !vec, !vec) {\n"
     << indent << "^bb0(%vs_x: vector<3xf64>, %vs_d: vector<3xf64>):\n"
     << inner << "%vs_sum = arith.addf %vs_x, %vs_d : vector<3xf64>\n"
     << inner << "md.yield %vs_sum : vector<3xf64>\n"
     << indent << "} : !vec\n";
  return virialName;
}

std::string Builder::emitConstraintForceVirial(StringRef indent,
                                               StringRef x, StringRef v,
                                               StringRef f,
                                               StringRef virial) {
  std::string inner = (indent + "  ").str();
  std::string current = virial.str();
  unsigned count = 0;
  // A group of `members` particles and its bonds (p, q), with the
  // displacements of the members from the first.
  auto emitGroup = [&](StringRef setName, unsigned members,
                       llvm::ArrayRef<std::pair<unsigned, unsigned>> bonds,
                       StringRef tupleFields, StringRef tupleTypes,
                       unsigned numTupleFields) {
    std::string coordinates, arguments;
    for (unsigned j = 1; j != members; ++j) {
      coordinates += (j == 1 ? "" : ", ") +
                     ("displacement(" + std::to_string(j) + ", 0)");
      arguments += "%cf_r" + std::to_string(j) + ": vector<3xf64>, ";
    }
    for (StringRef field : {"v", "f"})
      for (unsigned j = 0; j != members; ++j)
        arguments += ("%cf_" + field + std::to_string(j) + ": vector<3xf64>, ").str();
    for (unsigned j = 0; j != members; ++j)
      arguments += "%cf_m" + std::to_string(j) + ": f64, ";
    for (unsigned j = 0; j != numTupleFields; ++j)
      arguments += "%cf_t" + std::to_string(j) + ": f64, ";
    arguments.resize(arguments.size() - 2);
    std::string sum = "%cfw" + std::to_string(count++);
    os << indent << sum << " = md.sum_tuples %r_" << setName << ", " << x
       << ", %cell\n"
       << indent << "    coordinates(" << coordinates << ")\n"
       << indent << "    gather(" << v << ", " << f << ", %m : !vec, !vec, "
       << "!real)\n";
    if (numTupleFields)
      os << indent << "    tuple(" << tupleFields << " : " << tupleTypes
         << ")";
    os << " {\n" << indent << "^bb0(" << arguments << "):\n";
    SiteKernel k(os, inner, "%cfk");
    std::string zero = k.constant(0.0), one = k.constant(1.0);
    // The position of member j relative to the first, and its velocity,
    // acceleration without the constraints, and inverse mass.
    auto position = [&](unsigned j) { return "%cf_r" + std::to_string(j); };
    std::vector<std::string> u, inverse;
    for (unsigned j = 0; j != members; ++j) {
      std::string n = std::to_string(j);
      inverse.push_back(k.real("divf", one, "%cf_m" + n));
      u.push_back(k.scale(inverse.back(), "%cf_f" + n));
    }
    std::vector<std::string> units, rhs;
    for (auto [p, q] : bonds) {
      std::string d = p == 0 ? position(q)
                             : k.vector("subf", position(q), position(p));
      std::string length = k.norm(d);
      units.push_back(k.unit(d));
      std::string relative = k.vector("subf", "%cf_v" + std::to_string(q),
                                      "%cf_v" + std::to_string(p));
      std::string centripetal =
          k.real("divf", k.dot(relative, relative), length);
      std::string along =
          k.dot(units.back(), k.vector("subf", u[q], u[p]));
      rhs.push_back(k.real("subf", zero, k.real("addf", along, centripetal)));
    }
    // A_bc: what a unit force along bond c, +e_c on q_c and -e_c on p_c,
    // does to the acceleration along bond b.
    size_t n = bonds.size();
    std::vector<std::vector<std::string>> A(n, std::vector<std::string>(n));
    for (size_t b = 0; b != n; ++b)
      for (size_t c = 0; c != n; ++c) {
        auto [pb, qb] = bonds[b];
        auto [pc, qc] = bonds[c];
        auto on = [&](unsigned x) {
          return (x == qc ? 1 : 0) - (x == pc ? 1 : 0);
        };
        std::string coefficient = zero;
        if (on(qb) != 0)
          coefficient = k.real(on(qb) > 0 ? "addf" : "subf", coefficient,
                               inverse[qb]);
        if (on(pb) != 0)
          coefficient = k.real(on(pb) > 0 ? "subf" : "addf", coefficient,
                               inverse[pb]);
        A[b][c] = k.real("mulf", k.dot(units[b], units[c]), coefficient);
      }
    std::vector<std::string> tension = emitSmallSolve(k, A, rhs);
    // The force on each member but the first, and the virial.
    std::string elements;
    std::vector<std::string> forces(members, "");
    for (unsigned j = 1; j != members; ++j) {
      std::string force = k.zero();
      for (size_t c = 0; c != n; ++c) {
        auto [pc, qc] = bonds[c];
        if (j != pc && j != qc)
          continue;
        std::string push = k.scale(tension[c], units[c]);
        force = k.vector(j == qc ? "addf" : "subf", force, push);
      }
      forces[j] = force;
    }
    for (int a = 0; a != 3; ++a) {
      std::string row;
      for (unsigned j = 1; j != members; ++j) {
        std::string term = k.scale(k.component(position(j), a), forces[j]);
        row = row.empty() ? term : k.vector("addf", row, term);
      }
      for (int b = 0; b != 3; ++b)
        elements += (elements.empty() ? "" : ", ") + k.component(row, b);
    }
    os << inner << "%cf_w = vector.from_elements " << elements
       << " : vector<9xf64>\n"
       << inner << "md.yield %cf_w : vector<9xf64>\n"
       << indent << "} : !rel_" << setName << ", !vec -> vector<9xf64>\n";
    std::string next = sum + "_total";
    os << indent << next << " = arith.addf " << current << ", " << sum
       << " : vector<9xf64>\n";
    current = next;
  };
  for (const Program::TupleSet &set : program.tupleSets) {
    if (set.name == "settles") {
      static const std::pair<unsigned, unsigned> bonds[] = {
          {0, 1}, {0, 2}, {1, 2}};
      emitGroup(set.name, 3, bonds, "", "", 0);
    }
  }
  for (const Program::TupleSet *set : getShakeSets()) {
    std::vector<std::pair<unsigned, unsigned>> bonds;
    for (unsigned j = 1; j != set->arity; ++j)
      bonds.push_back({0, j});
    emitGroup(set->name, set->arity, bonds, "", "", 0);
  }
  return current;
}

std::string Builder::emitSettleVelocities(StringRef indent, StringRef x,
                                          StringRef v, StringRef result,
                                          StringRef virial,
                                          StringRef virialResult) {
  std::string inner = (indent + "  ").str();
  // The impulses along the three bonds that leave no velocity along them:
  // A τ = −b, with b the velocities along the bonds and A what a unit
  // impulse along one bond does to the velocity along another.
  auto emitKernel = [&](SiteKernel &k) {
    std::string e0 = k.unit("%vs_d10"), e1 = k.unit("%vs_d20"),
                e2 = k.unit("%vs_d21");
    std::string one = k.constant(1.0);
    std::string io = k.real("divf", one, "%vs_m0");
    std::string ih = k.real("divf", one, "%vs_m1");
    std::string zero = k.constant(0.0);
    auto minus = [&](StringRef a) { return k.real("subf", zero, a); };
    std::string r0 =
        minus(k.dot(e0, k.vector("subf", "%vs_v1", "%vs_v0")));
    std::string r1 =
        minus(k.dot(e1, k.vector("subf", "%vs_v2", "%vs_v0")));
    std::string r2 =
        minus(k.dot(e2, k.vector("subf", "%vs_v2", "%vs_v1")));
    std::string a00 = k.real("addf", io, ih), a11 = a00;
    std::string a22 = k.real("addf", ih, ih);
    std::string a01 = k.real("mulf", k.dot(e0, e1), io);
    std::string a02 = minus(k.real("mulf", k.dot(e0, e2), ih));
    std::string a12 = k.real("mulf", k.dot(e1, e2), ih);
    auto det3 = [&](StringRef m00, StringRef m01, StringRef m02,
                    StringRef m10, StringRef m11, StringRef m12,
                    StringRef m20, StringRef m21, StringRef m22) {
      auto minor = [&](StringRef p, StringRef q, StringRef r, StringRef s) {
        return k.real("subf", k.real("mulf", p, q), k.real("mulf", r, s));
      };
      return k.real(
          "addf",
          k.real("subf", k.real("mulf", m00, minor(m11, m22, m12, m21)),
                 k.real("mulf", m01, minor(m10, m22, m12, m20))),
          k.real("mulf", m02, minor(m10, m21, m11, m20)));
    };
    std::string det = det3(a00, a01, a02, a01, a11, a12, a02, a12, a22);
    std::string t0 = k.real(
        "divf", det3(r0, a01, a02, r1, a11, a12, r2, a12, a22), det);
    std::string t1 = k.real(
        "divf", det3(a00, r0, a02, a01, r1, a12, a02, r2, a22), det);
    std::string t2 = k.real(
        "divf", det3(a00, a01, r0, a01, a11, r1, a02, a12, r2), det);
    // Impulse c along e_c from p to q: +τ e / m_q on q, −τ e / m_p on p.
    std::string i0 = k.scale(t0, e0), i1 = k.scale(t1, e1),
                i2 = k.scale(t2, e2);
    std::string dv0 = k.negate(k.scale(io, k.vector("addf", i0, i1)));
    std::string dv1 = k.scale(ih, k.vector("subf", i0, i2));
    std::string dv2 = k.scale(ih, k.vector("addf", i1, i2));
    return std::array<std::string, 3>{dv0, dv1, dv2};
  };
  auto header = [&](StringRef op, StringRef field) {
    os << indent << " = md." << op << " %r_settles, " << x << ", %cell\n"
       << indent << "    coordinates(displacement(1, 0), displacement(2, 0), "
                    "displacement(2, 1))\n"
       << indent << "    gather(" << field << ", %m : !vec, !real)\n"
       << indent << "    tuple(%f_settles_doh, %f_settles_dhh : !of_settles, "
                    "!of_settles) {\n"
       << indent << "^bb0(%vs_d10: vector<3xf64>, %vs_d20: vector<3xf64>, "
                    "%vs_d21: vector<3xf64>, %vs_v0: vector<3xf64>, "
                    "%vs_v1: vector<3xf64>, %vs_v2: vector<3xf64>, "
                    "%vs_m0: f64, %vs_m1: f64, %vs_m2: f64, %vs_doh: f64, "
                    "%vs_dhh: f64):\n";
  };
  std::string change = (result + "_settle").str();
  os << indent << change;
  header("gather_tuples", v);
  SiteKernel k(os, inner);
  std::array<std::string, 3> dv = emitKernel(k);
  os << inner << "md.yield " << dv[0] << ", " << dv[1] << ", " << dv[2]
     << " : vector<3xf64>, vector<3xf64>, vector<3xf64>\n"
     << indent << "} : !rel_settles, !vec -> !vec\n";

  std::string virialName = virial.str();
  if (!virial.empty()) {
    // The forces of the constraints over the second half of the step,
    // G = 2 m Δv / dt, and their virial Σ (x_i − x_O) ⊗ G_i, with the
    // weight ½ (emitConstraintVirial). The changes Δv are those that the
    // step applies, read rather than solved again: a solve of its own
    // could round otherwise once fused with the step's (#97), and the
    // dynamics of a step would depend on whether it writes energies. The
    // groups are disjoint, so a particle's change is its group's.
    std::string sum = (virialResult + "_settle").str();
    os << indent << sum;
    header("sum_tuples", change);
    SiteKernel w(os, inner);
    std::string factor = w.real("divf", "%vs_m1", "%dt");
    std::array<std::string, 2> arms = {"%vs_d10", "%vs_d20"};
    std::array<std::string, 2> forces = {w.scale(factor, "%vs_v1"),
                                         w.scale(factor, "%vs_v2")};
    std::string elements;
    for (int a = 0; a != 3; ++a) {
      std::string row;
      for (int m = 0; m != 2; ++m) {
        std::string term = w.scale(w.component(arms[m], a), forces[m]);
        row = row.empty() ? term : w.vector("addf", row, term);
      }
      for (int b = 0; b != 3; ++b)
        elements += (elements.empty() ? "" : ", ") + w.component(row, b);
    }
    os << inner << "%vs_w = vector.from_elements " << elements
       << " : vector<9xf64>\n"
       << inner << "md.yield %vs_w : vector<9xf64>\n"
       << indent << "} : !rel_settles, !vec -> vector<9xf64>\n";
    os << indent << virialResult << " = arith.addf " << virial << ", "
       << sum << " : vector<9xf64>\n";
    virialName = virialResult.str();
  }
  os << indent << result << " = md.map_particles gather(" << v << ", "
     << change << " : !vec, !vec) {\n"
     << indent << "^bb0(%vs_x: vector<3xf64>, %vs_d: vector<3xf64>):\n"
     << inner << "%vs_sum = arith.addf %vs_x, %vs_d : vector<3xf64>\n"
     << inner << "md.yield %vs_sum : vector<3xf64>\n"
     << indent << "} : !vec\n";
  return virialName;
}

/// The kernel of the kinetic energy of one particle.
static void emitKineticEnergy(llvm::raw_ostream &os, StringRef result,
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

void Builder::emitLevel(unsigned level, StringRef indent) {
  // The state that the loops carry: positions, velocities, and forces.
  std::string state = "!vec, !vec, !vec";
  auto getValues = [&](const llvm::Twine &suffix) {
    std::string x = ("%x" + suffix).str(), v = ("%v" + suffix).str(),
                f = ("%f" + suffix).str();
    return x + ", " + v + ", " + f;
  };
  auto getInits = [&](const llvm::Twine &inside, const llvm::Twine &outside) {
    std::string text;
    for (StringRef name : {"x", "v", "f"}) {
      text += (text.empty() ? "" : ", ") +
              ("%" + name + inside + " = %" + name + outside).str();
    }
    return text;
  };
  // A segment begins with the particles in the order of their positions.
  auto isReordered = [&](unsigned index) {
    return program.reorders && levels[index].name == "segment";
  };

  std::string inner = (indent + "  ").str();
  std::string here = std::to_string(level);
  std::string outside = level == nestBegin
                            ? nestOutside
                            : isReordered(level - 1)
                                  ? "s"
                                  : "a" + std::to_string(level - 1);
  const Level &current = levels[level];
  bool isStepLoop = level + 1 == getNestEnd();
  bool reorders = isReordered(level);

  // With a new order in every iteration, the loop carries the fields that
  // do not change otherwise as well.
  std::string moreResults, moreInits, moreTypes, moreYielded;
  if (reorders) {
    moreResults = ", %me" + here;
    moreInits = ", %ma" + here + " = " + massName;
    moreTypes = ", !real";
    moreYielded = ", %ms";
    for (const Program::Field &field : program.fields) {
      moreResults += ", %pe" + here + "_" + field.name;
      moreInits +=
          ", %pa" + here + "_" + field.name + " = " + fieldPrefix + field.name;
      moreTypes += ", " + getFieldType(field).str();
      moreYielded += ", %ps_" + field.name;
    }
    moreResults += ", %ide" + here;
    moreInits += ", %ida" + here + " = " + idName;
    moreTypes += ", !ids";
    moreYielded += ", %ids";
  }

  os << indent << getValues("e" + here) << moreResults << " = scf.for %i"
     << here << " = %c0 to %n" << here << " step %c1\n"
     << indent << "    iter_args(" << getInits("a" + here, outside)
     << moreInits << ")\n"
     << indent << "    -> (" << state << moreTypes << ") {\n";

  // With a barostat the cell changes: each iteration takes it from where
  // the barostat keeps it.
  std::string outerCell = cellName, outerScale = scaleName;
  if (changesCell()) {
    cellName = "%cell" + here;
    for (int k = 0; k != getCellSize(); ++k)
      os << inner << "%edge" << here << "_" << k << " = memref.load "
         << "%box_memory[%c_edge" << k << "] : " << getBoxMemoryType()
         << "\n";
    emitCellOf(inner, cellName, "%edge" + here);
    if (scalesReference()) {
      scaleName = "%rest_scale" + here;
      emitReferenceScale(os, inner, scaleName, "%edge" + here + "_0",
                         "%edge" + here + "_1", "%edge" + here + "_2");
    }
  }

  std::string outerMass = massName, outerPrefix = fieldPrefix,
              outerId = idName;
  if (reorders) {
    emitReorder(inner, "a" + here, "s", "s", /*withForces=*/true, "%vs");
    massName = "%ms";
    fieldPrefix = "%ps_";
    idName = "%ids";
  }

  if (isStepLoop) {
    std::string noise1 = emitNoiseValue(inner);
    os << inner << getValues("b" + here) << " = dyn.step @step(";
    os << "%xa" << here << ", %va" << here << ", %fa" << here << ", "
         << massName << ", " << cellName << ", %dt" << getScaleValue() << getFieldValues(fieldPrefix) << noise1
         << ")\n"
         << inner << "    : (!vec, !vec, !vec, !real, !md.cell, f64" << getScaleType()
         << getFieldTypes() << getNoiseType() << ") -> (!vec, !vec, !vec)\n";
    os << inner << "scf.yield " << getValues("b" + here) << " : " << state
       << "\n";
  } else {
    emitLevel(level + 1, inner);
    std::string below = std::to_string(level + 1);
    std::string last = "e" + below;
    // The barostat may have changed the cell in the loop below; the steps
    // after it take the cell from where the barostat keeps it.
    if (changesCell()) {
      cellName = "%cellr" + here;
      for (int k = 0; k != getCellSize(); ++k)
        os << inner << "%edger" << here << "_" << k << " = memref.load "
           << "%box_memory[%c_edge" << k << "] : " << getBoxMemoryType()
           << "\n";
      emitCellOf(inner, cellName, "%edger" + here);
      if (scalesReference()) {
        scaleName = "%rest_scaler" + here;
        emitReferenceScale(os, inner, scaleName, "%edger" + here + "_0",
                           "%edger" + here + "_1", "%edger" + here + "_2");
      }
    }

    // The iteration of the loop of `upto` that is under way, counted over
    // the whole run.
    auto emitIteration = [&](unsigned upto) {
      std::string counted;
      for (unsigned i = nestBegin; i <= upto; ++i) {
        std::string index = "%i" + std::to_string(i);
        if (counted.empty()) {
          counted = index;
          continue;
        }
        std::string scaled = "%s" + here + "_" + std::to_string(i);
        os << inner << scaled << "m = arith.muli " << counted << ", %n" << i
           << " : index\n";
        os << inner << scaled << " = arith.addi " << scaled << "m, " << index
           << " : index\n";
        counted = scaled;
      }
      return counted;
    };
    // The number of the step that has just been taken.
    auto emitStep = [&]() {
      if (current.name == "couple" && level != nestBegin &&
          levels[level - 1].name == "energy") {
        // The interval between energies holds one period more than its
        // loop over periods: the steps before the interval, and those of
        // the periods of this interval so far.
        std::string outer = std::to_string(level - 1);
        std::string counted = emitIteration(level - 1);
        os << inner << "%before" << here << " = arith.muli " << counted
           << ", %per" << outer << " : index\n";
        os << inner << "%done" << here << " = arith.addi %i" << here
           << ", %c1 : index\n";
        os << inner << "%within" << here << " = arith.muli %done" << here
           << ", %per" << here << " : index\n";
        os << inner << "%steps" << here << " = arith.addi %before" << here
           << ", %within" << here << " : index\n";
      } else {
        std::string counted = emitIteration(level);
        os << inner << "%done" << here << " = arith.addi " << counted
           << ", %c1 : index\n";
        os << inner << "%steps" << here << " = arith.muli %done" << here
           << ", %per" << here << " : index\n";
      }
      os << inner << "%since" << here << " = arith.index_cast %steps"
         << here << " : index to i64\n";
      os << inner << "%step" << here << " = arith.addi %since" << here
         << ", " << nestStart << " : i64\n";
    };
    // The velocities as the step after the coupling takes them, and the
    // state that the loop yields.
    // With leapfrog and a barostat, the coupling takes the velocities of
    // the time of the positions, `current`, and the stored ones are half a
    // kick behind those it leaves: the steps are then those of velocity
    // Verlet, and the energy that the coupling gives is that which the log
    // counts (D76).
    auto getCoupled = [&](StringRef x, StringRef v, StringRef f,
                          StringRef energy, StringRef trace,
                          StringRef current = "",
                          const TrotterScaling *trotter = nullptr) {
      Coupled coupled =
          emitCoupling(inner, x, current.empty() ? v : current, f, energy,
                       here, "%step" + here, trace, trotter);
      std::string velocities = coupled.velocities;
      if (!current.empty()) {
        std::string behind = "%vh" + here;
        os << inner << behind << " = dyn.kick " << velocities << ", "
           << coupled.forces << ", " << massName
           << ", %half_back : !vec\n";
        velocities = behind;
      }
      return coupled.positions + ", " + velocities + ", " + coupled.forces;
    };
    bool couplesBelow = levels[level + 1].name == "couple";
    // The last two steps of a period with the barostat of Trotter type
    // (D92): the step of energy whose pressure gives the strain, and the
    // step that scales the cell within its drift, in the new cell; their
    // results are named with `name`, as those of a step of energy, with
    // the kinetic energy of the scaled velocities `%kh<name>`. Returns the
    // scaling.
    auto emitTrotterSteps = [&](StringRef name, bool withEnergy) {
      std::string a = "j" + here, n = name.str();
      bool leapfrog = isLeapfrog();
      std::string kinetic, groups, trace, x, v;
      if (scalesEveryStep()) {
        // The pressure of the state that the step before left (D92): the
        // diagonals of the virial and of the virial of the groups, and the
        // kinetic energy of each axis.
        std::string m = "%tm" + here;
        std::string loaded[9];
        for (int slot = 0; slot != 9; ++slot) {
          loaded[slot] = m + "_" + std::to_string(slot);
          os << inner << m << "_i" << slot << " = arith.constant " << slot
             << " : index\n"
             << inner << loaded[slot] << " = memref.load %trotter_memory["
             << m << "_i" << slot << "] : memref<9xf64>\n";
        }
        std::string vectors[3];
        for (int v = 0; v != 3; ++v) {
          vectors[v] = m + "_v" + std::to_string(v);
          os << inner << vectors[v] << " = vector.from_elements "
             << loaded[3 * v] << ", " << loaded[3 * v + 1] << ", "
             << loaded[3 * v + 2] << " : vector<3xf64>\n";
        }
        trace = vectors[0];
        groups = vectors[1];
        kinetic = vectors[2];
        x = "%x" + last;
        v = "%v" + last;
      } else {
        // The step whose pressure gives the strain.
        std::string noise2 = emitNoiseValue(inner);
        os << inner << getValues(a) << ", %w" << a << ", %gw" << a
           << (leapfrog ? ", %vc" + a : "") << " = dyn.step @step_virial(%x"
           << last << ", %v" << last << ", %f" << last << ", " << massName
           << ", " << cellName << ", %dt" << getScaleValue()
           << getFieldValues(fieldPrefix) << noise2 << ")\n"
           << inner << "    : (!vec, !vec, !vec, !real, !md.cell, f64"
           << getScaleType() << getFieldTypes() << getNoiseType()
           << ") -> (!vec, !vec, !vec, vector<9xf64>, vector<3xf64>"
           << (leapfrog ? ", !vec" : "") << ")\n";
        trace = "%dg" + a;
        emitDiagonal(os, trace, "%w" + a, inner);
        groups = "%gw" + a;
        x = "%x" + a;
        v = "%v" + a;
      }
      emitStep();
      TrotterScaling scaling =
          scalesEveryStep()
              ? emitTrotterStrain(inner, "", trace, groups, here,
                                  "%step" + here, kinetic)
              : emitTrotterStrain(inner, leapfrog ? "%vc" + a : "%v" + a,
                                  trace, groups, here, "%step" + here);
      std::string outerScale = scaleName;
      if (!scaling.scale.empty())
        scaleName = scaling.scale;
      bool virial = withEnergy || countsAfterScaling();
      std::string noise3 = emitNoiseValue(inner);
      os << inner << "%x" << n << ", %v" << n << ", %f" << n
         << (withEnergy ? ", %u" + n : "") << (virial ? ", %w" + n : "")
         << (virial ? ", %gw" + n : "")
         << (leapfrog ? ", %vc" + n : "") << ", %kh" << n
         << (withEnergy && measuresHalfSteps() ? ", %khs" + n : "")
         << (withEnergy && reportsSolvent() ? ", %kws" + n : "")
         << " = dyn.step @step_trotter" << (withEnergy ? "_energy" : "")
         << "(" << x << ", " << v << ", %f"
         << (scalesEveryStep() ? last : a) << ", " << massName << ", "
         << scaling.cell << ", %dt, " << scaling.mu << getScaleValue()
         << getFieldValues(fieldPrefix) << noise3 << ")\n"
         << inner
         << "    : (!vec, !vec, !vec, !real, !md.cell, f64, vector<3xf64>"
         << getScaleType() << getFieldTypes() << getNoiseType() << ") -> (!vec, !vec, !vec"
         << (withEnergy ? ", f64" : "")
         << (virial ? ", vector<9xf64>, vector<3xf64>" : "")
         << (leapfrog ? ", !vec" : "") << ", vector<3xf64>"
         << (withEnergy && measuresHalfSteps() ? ", vector<3xf64>" : "")
         << (withEnergy && reportsSolvent() ? ", vector<3xf64>" : "")
         << ")\n";
      scaleName = outerScale;
      scaling.kineticHalf = "%kh" + n;
      if (virial)
        scaling.groupsAfter = "%gw" + n;
      return scaling;
    };

    if (current.name == "couple") {
      // The last step of the period, and the coupling after it. The
      // barostat needs the virial of the step.
      std::string trace;
      if (usesTrotter()) {
        std::string k = "k" + here;
        TrotterScaling scaling = emitTrotterSteps(k, /*withEnergy=*/false);
        if (countsAfterScaling()) {
          trace = "%dg" + k;
          emitDiagonal(os, trace, "%w" + k, inner);
        }
        std::string coupled =
            getCoupled("%x" + k, "%v" + k, "%f" + k, "", trace,
                       isLeapfrog() ? "%vc" + k : "", &scaling);
        os << inner << "scf.yield " << coupled << " : " << state << "\n";
        os << indent << "}\n";
        cellName = outerCell;
        scaleName = outerScale;
        return;
      }
      if (control.barostat) {
        // With leapfrog the pressure takes the velocities of the time of
        // the positions, which the step of energy returns.
        std::string current = isLeapfrog() ? "%vck" + here : "";
        std::string noise4 = emitNoiseValue(inner);
        os << inner << getValues("k" + here) << ", %uk" << here << ", %wk"
           << here << (isLeapfrog() ? ", " + current : "")
           << " = dyn.step @step_"
           << (measuresHalfSteps() ? "coupling" : "energy") << "(%x" << last
           << ", %v" << last << ", %f" << last << ", " << massName << ", "
           << cellName
           << ", %dt" << getScaleValue() << getFieldValues(fieldPrefix) << noise4 << ")\n"
           << inner << "    : (!vec, !vec, !vec, !real, !md.cell, f64" << getScaleType()
           << getFieldTypes() << getNoiseType()
           << ") -> (!vec, !vec, !vec, f64, vector<9xf64>"
           << (isLeapfrog() ? ", !vec" : "") << ")\n";
        trace = "%dgk" + here;
        emitDiagonal(os, trace, "%wk" + here, inner);
        emitStep();
        std::string coupled =
            getCoupled("%xk" + here, "%vk" + here, "%fk" + here,
                       "%uk" + here, trace, current);
        os << inner << "scf.yield " << coupled << " : " << state << "\n";
        os << indent << "}\n";
        cellName = outerCell;
        scaleName = outerScale;
        return;
      }
      std::string noise5 = emitNoiseValue(inner);
      os << inner << getValues("k" + here) << " = dyn.step @step(";
      os << "%x" << last << ", %v" << last << ", %f" << last << ", "
           << massName << ", " << cellName << ", %dt" << getScaleValue() << getFieldValues(fieldPrefix) << noise5
           << ")\n"
           << inner << "    : (!vec, !vec, !vec, !real, !md.cell, f64" << getScaleType()
           << getFieldTypes() << getNoiseType() << ") -> (!vec, !vec, !vec)\n";
      emitStep();
      std::string coupled =
          getCoupled("%xk" + here, "%vk" + here, "%fk" + here, "", "");
      os << inner << "scf.yield " << coupled << " : " << state << "\n";
      os << indent << "}\n";
      cellName = outerCell;
      scaleName = outerScale;
      return;
    }

    if (current.name == "energy" && couplesBelow) {
      // The last period of the interval: its steps but the last, which
      // the step of energy takes.
      std::string steps = std::to_string(getNestEnd() - 1);
      os << inner << getValues("q" + here) << " = scf.for %j" << here
         << " = %c0 to %n" << steps << " step %c1\n"
         << inner << "    iter_args(" << getInits("p" + here, last) << ")\n"
         << inner << "    -> (" << state << ") {\n";
      std::string noise6 = emitNoiseValue((inner + "  "));
      os << inner << "  " << getValues("r" + here) << " = dyn.step @step(";
      os << "%xp" << here << ", %vp" << here << ", %fp" << here << ", "
           << massName << ", " << cellName << ", %dt" << getScaleValue() << getFieldValues(fieldPrefix) << noise6
           << ")\n"
           << inner << "      : (!vec, !vec, !vec, !real, !md.cell, f64" << getScaleType()
           << getFieldTypes() << getNoiseType() << ") -> (!vec, !vec, !vec)\n";
      os << inner << "  scf.yield " << getValues("r" + here) << " : "
         << state << "\n"
         << inner << "}\n";
      last = "q" + here;
    }

    if (current.name == "energy") {
      // The last step of the interval, and the energies after it.
      // With leapfrog the step of energy also returns the velocities of
      // the time of the positions, which the kinetic energy takes.
      std::string virialName = "%w", energyName = "%u", now = "%vn";
      // With the barostat of Trotter type the last period ends with the
      // step that scales the cell (D92).
      bool scalesHere = usesTrotter() && couplesBelow;
      TrotterScaling scaling;
      if (scalesHere) {
        scaling = emitTrotterSteps("l", /*withEnergy=*/true);
        virialName = "%wl";
        energyName = "%ul";
        now = "%vcl";
      } else {
        std::string noise7 = emitNoiseValue(inner);
        os << inner << "%xl, %vl, %fl, %u, %w"
           << (isLeapfrog() ? ", %vn" : "")
           << (measuresHalfSteps() ? ", %khsl" : "")
           << (reportsSolvent() ? ", %kwsl" : "")
           << " = dyn.step @step_energy(%x"
           << last << ", %v" << last << ", %f" << last << ", " << massName
           << ", " << cellName << ", %dt" << getScaleValue()
           << getFieldValues(fieldPrefix) << noise7 << ")\n"
           << inner << "    : (!vec, !vec, !vec, !real, !md.cell, f64"
           << getScaleType() << getFieldTypes() << getNoiseType()
           << ") -> (!vec, !vec, !vec, f64, vector<9xf64>"
           << (isLeapfrog() ? ", !vec" : "")
           << (measuresHalfSteps() ? ", vector<3xf64>" : "")
           << (reportsSolvent() ? ", vector<3xf64>" : "") << ")\n";
      }
      std::string full = isLeapfrog() ? now : "%vl";
      emitKineticEnergy(os, "%k", full, massName, inner);
      // K_half - K, measured from the velocities of the half steps with
      // constraints and from the forces without (D203).
      std::string excess = emitKineticExcess(
          inner, "l" + here, measuresHalfSteps() ? "%khsl" : "", "%fl", full);
      if (excess.empty())
        os << inner << "%g = arith.constant 0.0 : f64\n";
      else
        emitSum3(os, "%g", excess, inner);
      if (reportsSolvent())
        emitWriteSolvent(inner, "%kwsl");
      emitTrace(os, "%tr", virialName, inner);
      if (couplesBelow && control.barostat)
        emitDiagonal(os, "%dg", virialName, inner);
      if (!scalesHere)
        emitStep();
      os << inner << "func.call @mdrtWriteEnergies(%step" << here << ", "
         << energyName << ", %k, %g, %tr) : (i64, f64, f64, f64, f64) -> ()\n";
      if (!getPullColumns().empty()) {
        // The coordinates of the terms over centers at the positions after
        // the step, at its time (D145).
        std::string time = "%pull_time" + here;
        if (control.usesTime)
          os << inner << time << "_steps = arith.sitofp %step" << here
             << " : i64 to f64\n"
             << inner << time << " = arith.mulf " << time << "_steps, %dt"
             << " : f64\n";
        emitPullOutput(inner, "%xl", cellName, fieldPrefix, "%step" + here,
                       time);
      }
      if (!control.freeEnergyFile.empty()) {
        // dH/dλ and the energies of the other states at the positions
        // after the step (D161).
        std::string time = "%fe_time" + here;
        if (control.usesTime)
          os << inner << time << "_steps = arith.sitofp %step" << here
             << " : i64 to f64\n"
             << inner << time << " = arith.mulf " << time << "_steps, %dt"
             << " : f64\n";
        emitFreeEnergyOutput(inner, "%xl", cellName, fieldPrefix,
                             "%step" + here, time);
      }
      if (!control.observablesFile.empty()) {
        // The observed terms at the positions after the step (D189).
        std::string time = "%ob_time" + here;
        if (control.usesTime)
          os << inner << time << "_steps = arith.sitofp %step" << here
             << " : i64 to f64\n"
             << inner << time << " = arith.mulf " << time << "_steps, %dt"
             << " : f64\n";
        emitObservablesOutput(inner, "%xl", cellName, fieldPrefix,
                              "%step" + here, time);
      }
      // Without a periodic cell (D142), whether the particles have spread
      // so far that images interact, at every row of the log as well as at
      // the frames and checkpoints.
      if (!control.periodic && !framesAtEnergies)
        os << inner << "mdrt.host_call @mdrtCheckSpread(%step" << here
           << ", %xl, " << idName << ") : (i64, !vec, !ids)\n";
      std::string yielded =
          couplesBelow
              ? getCoupled("%xl", "%vl", "%fl", energyName,
                           control.barostat ? "%dg" : "",
                           isLeapfrog() && control.barostat ? now : "",
                           scalesHere ? &scaling : nullptr)
              : getValues("l");
      // The frame of the positions that the interval leaves, as a loop
      // over frames writes them.
      if (framesAtEnergies)
        os << inner << "mdrt.host_call @mdrtWriteFrame(%step" << here << ", "
           << StringRef(yielded).split(',').first << ", " << idName
           << ") : (i64, !vec, !ids)\n";
      os << inner << "scf.yield " << yielded << " : " << state << "\n";
    } else {
      if (current.name == "frame") {
        emitStep();
        os << inner << "mdrt.host_call @mdrtWriteFrame(%step" << here
           << ", %x" << last << ", " << idName << ") : (i64, !vec, !ids)\n";
      }
      if (current.name == "segment") {
        // The state as the next step needs it.
        emitStep();
        os << inner << "mdrt.host_call @mdrtWriteCheckpoint(%step" << here
           << ", " << getValues(last) << ", " << idName << ")\n"
           << inner << "    : (i64, " << state << ", !ids)\n";
      }
      os << inner << "scf.yield " << getValues(last) << moreYielded << " : "
         << state << moreTypes << "\n";
    }
  }

  os << indent << "}";
  if (current.name == "segment")
    os << " {mdrt.segment}";
  os << "\n";

  massName = outerMass;
  fieldPrefix = outerPrefix;
  idName = outerId;
  cellName = outerCell;
  scaleName = outerScale;
  // What the loop has left is in the order of its last iteration.
  if (reorders) {
    massName = "%me" + here;
    fieldPrefix = "%pe" + here + "_";
    idName = "%ide" + here;
    emitRenumber(indent, idName, "%re" + here + "_");
  }
}

std::vector<const Program::TupleSet *> Builder::getRigidGroups() const {
  std::vector<const Program::TupleSet *> groups = getShakeSets();
  for (const Program::TupleSet &set : program.tupleSets)
    if (set.name == "settles")
      groups.insert(groups.begin(), &set);
  return groups;
}

std::string Builder::emitGroupTrace(StringRef indent, StringRef trace,
                                    StringRef positions, StringRef velocities,
                                    StringRef cell, StringRef masses,
                                    StringRef relations, StringRef tag) {
  std::string t = tag.str();
  std::string inner = (indent + "  ").str();
  std::string groupTrace = trace.str();
  for (const Program::TupleSet *set : getRigidGroups()) {
    unsigned count = set->arity - 1;
    std::string arguments = "%vs_r: vector<3xf64>, ";
    for (unsigned k = 0; k <= count; ++k)
      arguments += "%vs_v" + std::to_string(k) + ": vector<3xf64>, ";
    for (unsigned k = 0; k <= count; ++k)
      arguments += "%vs_m" + std::to_string(k) + ": f64" +
                   (k == count ? "" : ", ");
    std::string internal = "%bki" + t + "_" + set->name;
    os << indent << internal << " = md.sum_tuples " << relations
       << set->name << ", " << positions << ", " << cell << "\n"
       << indent << "    coordinates(displacement(1, 0))\n"
       << indent << "    gather(" << velocities << ", " << masses
       << " : !vec, !real) {\n"
       << indent << "^bb0(" << arguments << "):\n";
    SiteKernel k(os, inner);
    std::string total = "%vs_m0";
    std::string moment = k.scale("%vs_m0", "%vs_v0");
    for (unsigned j = 1; j <= count; ++j) {
      std::string n = std::to_string(j);
      total = k.real("addf", total, "%vs_m" + n);
      moment = k.vector("addf", moment, k.scale("%vs_m" + n, "%vs_v" + n));
    }
    std::string center =
        k.scale(k.real("divf", k.constant(1.0), total), moment);
    std::string twice;
    for (unsigned j = 0; j <= count; ++j) {
      std::string n = std::to_string(j);
      std::string relative = k.vector("subf", "%vs_v" + n, center);
      std::string term =
          k.real("mulf", "%vs_m" + n, k.dot(relative, relative));
      twice = twice.empty() ? term : k.real("addf", twice, term);
    }
    os << inner << "md.yield " << twice << " : f64\n"
       << indent << "} : !rel_" << set->name << ", !vec -> f64\n";
    std::string next = "%bgt" + t + "_" + set->name;
    os << indent << next << " = arith.addf " << groupTrace << ", "
       << internal << " : f64\n";
    groupTrace = next;
  }
  return groupTrace;
}

std::string Builder::emitMolecularDiagonal(StringRef indent,
                                           StringRef diagonal,
                                        StringRef positions, StringRef forces,
                                        StringRef cell, StringRef masses,
                                        StringRef relations, StringRef tag) {
  std::string t = tag.str();
  std::string inner = (indent + "  ").str();
  std::string molecular = diagonal.str();
  for (const Program::TupleSet *set : getRigidGroups()) {
    unsigned count = set->arity - 1;
    std::string coordinates, arguments;
    for (unsigned k = 1; k <= count; ++k) {
      coordinates += (k == 1 ? "" : ", ") +
                     ("displacement(" + std::to_string(k) + ", 0)");
      arguments += "%vs_r" + std::to_string(k) + ": vector<3xf64>, ";
    }
    for (unsigned k = 0; k <= count; ++k)
      arguments += "%vs_f" + std::to_string(k) + ": vector<3xf64>, ";
    for (unsigned k = 0; k <= count; ++k)
      arguments += "%vs_m" + std::to_string(k) + ": f64" +
                   (k == count ? "" : ", ");
    // With the places r_k about the first particle and c the center of
    // mass about it, Σ (r_k − c) ⊙ F_k = Σ r_k ⊙ F_k − c ⊙ Σ F_k.
    std::string internal = "%bgi" + t + "_" + set->name;
    os << indent << internal << " = md.sum_tuples " << relations
       << set->name << ", " << positions << ", " << cell << "\n"
       << indent << "    coordinates(" << coordinates << ")\n"
       << indent << "    gather(" << forces << ", " << masses
       << " : !vec, !real) {\n"
       << indent << "^bb0(" << arguments << "):\n";
    SiteKernel k(os, inner);
    std::string total = "%vs_m0", moment = k.zero(), sum = "%vs_f0",
                places = k.zero();
    for (unsigned j = 1; j <= count; ++j) {
      std::string n = std::to_string(j);
      total = k.real("addf", total, "%vs_m" + n);
      moment = k.vector("addf", moment, k.scale("%vs_m" + n, "%vs_r" + n));
      sum = k.vector("addf", sum, "%vs_f" + n);
      places = k.vector("addf", places,
                        k.vector("mulf", "%vs_r" + n, "%vs_f" + n));
    }
    std::string center =
        k.scale(k.real("divf", k.constant(1.0), total), moment);
    std::string about =
        k.vector("subf", places, k.vector("mulf", center, sum));
    os << inner << "md.yield " << about << " : vector<3xf64>\n"
       << indent << "} : !rel_" << set->name << ", !vec -> vector<3xf64>\n";
    std::string next = "%bgm" + t + "_" + set->name;
    os << indent << next << " = arith.subf " << molecular << ", " << internal
       << " : vector<3xf64>\n";
    molecular = next;
  }
  return molecular;
}

std::string Builder::emitGroupScaling(StringRef indent, StringRef positions,
                                      StringRef mu, StringRef oneMinusMu,
                                      StringRef cell, StringRef masses,
                                      StringRef relations, StringRef tag) {
  std::string t = tag.str();
  std::string inner = (indent + "  ").str();
  std::string newPositions = "%xc" + t;
  os << indent << newPositions << " = md.map_particles gather(" << positions
     << " : !vec) {\n"
     << indent << "^bb0(%x_i: vector<3xf64>):\n"
     << indent << "  %x_scaled = arith.mulf " << mu
     << ", %x_i : vector<3xf64>\n"
     << indent << "  md.yield %x_scaled : vector<3xf64>\n"
     << indent << "} : !vec\n";
  // A group that the constraints keep rigid moves with its center of
  // mass, keeping its shape: each of its particles moves back by
  // (μ − 1) times its place about the center. Were the bonds stretched,
  // the constraints of the next step would take them back with a change
  // of the velocities, and heat the system.
  for (const Program::TupleSet *set : getRigidGroups()) {
    unsigned count = set->arity - 1;
    std::string coordinates, arguments;
    for (unsigned k = 1; k <= count; ++k) {
      coordinates += (k == 1 ? "" : ", ") +
                     ("displacement(" + std::to_string(k) + ", 0)");
      arguments += "%vs_r" + std::to_string(k) + ": vector<3xf64>, ";
    }
    for (unsigned k = 0; k <= count; ++k)
      arguments += "%vs_m" + std::to_string(k) + ": f64" +
                   (k == count ? "" : ", ");
    std::string change = "%xg" + t + "_" + set->name;
    os << indent << change << " = md.gather_tuples " << relations
       << set->name << ", " << positions << ", " << cell << "\n"
       << indent << "    coordinates(" << coordinates << ")\n"
       << indent << "    gather(" << masses << " : !real) {\n"
       << indent << "^bb0(" << arguments << "):\n";
    SiteKernel k(os, inner);
    std::string total = "%vs_m0", moment = k.zero();
    for (unsigned j = 1; j <= count; ++j) {
      std::string n = std::to_string(j);
      total = k.real("addf", total, "%vs_m" + n);
      moment = k.vector("addf", moment, k.scale("%vs_m" + n, "%vs_r" + n));
    }
    std::string one = k.constant(1.0);
    std::string center = k.scale(k.real("divf", one, total), moment);
    std::string yielded, types;
    for (unsigned j = 0; j <= count; ++j) {
      std::string place =
          j == 0 ? k.negate(center)
                 : k.vector("subf", "%vs_r" + std::to_string(j), center);
      yielded += (j == 0 ? "" : ", ") + k.vector("mulf", oneMinusMu, place);
      types += (j == 0 ? "" : ", ") + std::string("vector<3xf64>");
    }
    os << inner << "md.yield " << yielded << " : " << types << "\n"
       << indent << "} : !rel_" << set->name << ", !vec -> !vec\n";
    std::string moved = change + "_x";
    os << indent << moved << " = md.map_particles gather(" << newPositions
       << ", " << change << " : !vec, !vec) {\n"
       << indent << "^bb0(%x_i: vector<3xf64>, %d_i: vector<3xf64>):\n"
       << indent << "  %x_moved = arith.addf %x_i, %d_i : vector<3xf64>\n"
       << indent << "  md.yield %x_moved : vector<3xf64>\n"
       << indent << "} : !vec\n";
    newPositions = moved;
  }
  return newPositions;
}

void Builder::emitStrain(StringRef indent, StringRef kinetic,
                         StringRef diagonal, StringRef tag, StringRef step) {
  std::string t = tag.str();
  for (int k = 0; k != getCellSize(); ++k)
    os << indent << "%be" << t << "_" << k << " = memref.load %box_memory"
       << "[%c_edge" << k << "] : " << getBoxMemoryType() << "\n";
  os << indent << "%bxy" << t << " = arith.mulf %be" << t << "_0, %be" << t
     << "_1 : f64\n"
     << indent << "%bv" << t << " = arith.mulf %bxy" << t << ", %be" << t
     << "_2 : f64\n";
  // The pressure of each axis, in bar: (2 K_a + W_aa + C / (3 V)) / V, with
  // C / V the trace of the virials of the constant terms.
  os << indent << "%bk2" << t << " = arith.addf " << kinetic << ", "
     << kinetic << " : vector<3xf64>\n"
     << indent << "%bw0" << t << " = arith.addf %bk2" << t << ", "
     << diagonal << " : vector<3xf64>\n"
     << indent << "%bwc" << t << " = arith.divf %baro_constant, %bv" << t
     << " : f64\n"
     << indent << "%bwc3" << t << " = arith.mulf %bwc" << t
     << ", %c_third : f64\n"
     << indent << "%bwcb" << t << " = vector.broadcast %bwc3" << t
     << " : f64 to vector<3xf64>\n"
     << indent << "%bw" << t << " = arith.addf %bw0" << t << ", %bwcb" << t
     << " : vector<3xf64>\n"
     << indent << "%bpc" << t << " = arith.divf %c_bar, %bv" << t
     << " : f64\n"
     << indent << "%bpcb" << t << " = vector.broadcast %bpc" << t
     << " : f64 to vector<3xf64>\n"
     << indent << "%bp3" << t << " = arith.mulf %bw" << t << ", %bpcb" << t
     << " : vector<3xf64>\n";
  if (control.anisotropic) {
    // Anisotropic (D163c): each axis by the strain of its own pressure and
    // noise, eq. (9b) of [Bernetti2020] on each.
    std::string strains[3];
    for (int k = 0; k != 3; ++k) {
      std::string a = std::to_string(k);
      strains[k] = "%strain" + a + "_" + t;
      os << indent << "%bpa" << a << "_" << t << " = vector.extract %bp3"
         << t << "[" << k << "] : f64 from vector<3xf64>\n"
         << indent << "%baxis" << a << "_" << t << " = arith.constant " << k
         << " : i64\n"
         << indent << strains[k]
         << " = func.call @mdrtBarostatStrainAxis(%seed, " << step
         << ", %baxis" << a << "_" << t << ", %bpa" << a << "_" << t
         << ", %baro_target, %bv" << t << ", %baro_kt, %baro_beta_" << a
         << ", %baro_rate)\n"
         << indent
         << "    : (i64, i64, i64, f64, f64, f64, f64, f64, f64) -> f64\n";
    }
    os << indent << "%bs3" << t << " = vector.from_elements " << strains[0]
       << ", " << strains[1] << ", " << strains[2] << " : vector<3xf64>\n";
  } else if (!control.semiIsotropic) {
    // Isotropic: the mean pressure gives ε, the change of ln V, by a step
    // of λ = √V (eq. S7 of [Bernetti2020]); each axis scales by ε / 3.
    emitSum3(os, "%bps" + t, "%bp3" + t, indent);
    os << indent << "%bp" << t << " = arith.mulf %bps" << t
       << ", %c_third : f64\n"
       << indent << "%strain" << t
       << " = func.call @mdrtBarostatStrain(%seed, " << step << ", %bp" << t
       << ", %baro_target, %bv" << t
       << ", %baro_kt, %baro_beta, %baro_rate)\n"
       << indent << "    : (i64, i64, f64, f64, f64, f64, f64, f64) -> f64\n"
       << indent << "%bsa" << t << " = arith.mulf %strain" << t
       << ", %c_third : f64\n"
       << indent << "%bs3" << t << " = vector.broadcast %bsa" << t
       << " : f64 to vector<3xf64>\n";
  } else {
    // Semi-isotropic: the mean pressure of x and y gives the change of
    // ln A, that of z the change of ln L, by eqs. (9a) and (9b) of
    // [Bernetti2020] with their own noises (D119); x and y scale by half
    // the first, z by the second.
    os << indent << "%bpx" << t << " = vector.extract %bp3" << t
       << "[0] : f64 from vector<3xf64>\n"
       << indent << "%bpy" << t << " = vector.extract %bp3" << t
       << "[1] : f64 from vector<3xf64>\n"
       << indent << "%bpz" << t << " = vector.extract %bp3" << t
       << "[2] : f64 from vector<3xf64>\n"
       << indent << "%bpxys" << t << " = arith.addf %bpx" << t << ", %bpy"
       << t << " : f64\n"
       << indent << "%bpxy" << t << " = arith.mulf %bpxys" << t
       << ", %couple_half : f64\n"
       << indent << "%strainxy" << t
       << " = func.call @mdrtBarostatStrainArea(%seed, " << step << ", %bpxy"
       << t << ", %baro_target, %bv" << t << ", %be" << t
       << "_2, %baro_kt, %baro_beta, %baro_rate, %baro_tension)\n"
       << indent
       << "    : (i64, i64, f64, f64, f64, f64, f64, f64, f64, f64) -> f64\n"
       << indent << "%strainz" << t
       << " = func.call @mdrtBarostatStrainHeight(%seed, " << step
       << ", %bpz" << t << ", %baro_target, %bv" << t
       << ", %baro_kt, %baro_beta_z, %baro_rate)\n"
       << indent << "    : (i64, i64, f64, f64, f64, f64, f64, f64) -> f64\n"
       << indent << "%bsxy" << t << " = arith.mulf %strainxy" << t
       << ", %couple_half : f64\n"
       << indent << "%bs3" << t << " = vector.from_elements %bsxy" << t
       << ", %bsxy" << t << ", %strainz" << t << " : vector<3xf64>\n";
  }
  os << indent << "%mu" << t << " = math.exp %bs3" << t
     << " : vector<3xf64>\n"
     << indent << "%muinv" << t << " = arith.divf %c_unit3, %mu" << t
     << " : vector<3xf64>\n";
  // The new cell, which the next iteration takes from memory: H diag(μ),
  // the tilts b_x and c_x of the column of x scaled by μ_x and c_y by μ_y
  // (I3 of docs/triclinic-m2.md).
  static const int column[] = {0, 1, 2, 0, 0, 1};
  for (int k = 0; k != getCellSize(); ++k) {
    if (k < 3)
      os << indent << "%bmu" << t << "_" << k << " = vector.extract %mu" << t
         << "[" << k << "] : f64 from vector<3xf64>\n";
    os << indent << "%bn" << t << "_" << k << " = arith.mulf %be" << t
       << "_" << k << ", %bmu" << t << "_" << column[k] << " : f64\n"
       << indent << "memref.store %bn" << t << "_" << k
       << ", %box_memory[%c_edge" << k << "] : " << getBoxMemoryType()
       << "\n";
  }
  os << indent << "func.call @mdrtSetBox(%bn" << t << "_0, %bn" << t
     << "_1, %bn" << t << "_2) : (f64, f64, f64) -> ()\n";
  if (isTriclinic())
    os << indent << "func.call @mdrtSetTilt(%bn" << t << "_3, %bn" << t
       << "_4, %bn" << t << "_5) : (f64, f64, f64) -> ()\n";
}

Builder::TrotterScaling Builder::emitTrotterStrain(
    StringRef indent, StringRef velocities, StringRef diagonal,
    StringRef groups, StringRef tag, StringRef step,
    StringRef givenKinetic) {
  std::string t = tag.str();
  // The kinetic energy of the pressure, axis by axis: that of the
  // velocities of the step, without the center of mass.
  std::string kinetic = givenKinetic.str();
  if (kinetic.empty())
    kinetic = emitKineticWithoutCenter(indent, velocities, massName, t);
  emitStrain(indent, kinetic, diagonal, t, step);
  return finishTrotterStrain(indent, groups, t);
}

std::string Builder::emitKineticWithoutCenter(StringRef indent,
                                              StringRef velocities,
                                              StringRef masses,
                                              StringRef tag) {
  std::string t = tag.str();
  std::string kinetic = "%tk" + t;
  emitKineticVector(os, kinetic, velocities, masses, indent);
  if (control.comPeriod > 0) {
    os << indent << "%tpc" << t << " = md.sum_particles gather("
       << velocities << ", " << masses << " : !vec, !real) {\n"
       << indent << "^bb0(%v_i: vector<3xf64>, %m_i: f64):\n"
       << indent << "  %mb = vector.broadcast %m_i : f64 to vector<3xf64>\n"
       << indent << "  %p = arith.mulf %mb, %v_i : vector<3xf64>\n"
       << indent << "  md.yield %p : vector<3xf64>\n"
       << indent << "} : vector<3xf64>\n"
       << indent << "%tvcm" << t << " = arith.divf %tpc" << t
       << ", %total_mass : vector<3xf64>\n"
       << indent << "%tpvs" << t << " = arith.mulf %tpc" << t << ", %tvcm"
       << t << " : vector<3xf64>\n"
       << indent << "%tkcm" << t << " = arith.mulf %c_half3, %tpvs" << t
       << " : vector<3xf64>\n"
       << indent << "%tkt" << t << " = arith.subf " << kinetic << ", %tkcm"
       << t << " : vector<3xf64>\n";
    kinetic = "%tkt" + t;
  }
  return kinetic;
}

void Builder::emitStoreTrotterState(StringRef indent, StringRef diagonal,
                                    StringRef groups, StringRef kinetic) {
  static unsigned stores = 0;
  std::string t = std::to_string(stores++);
  StringRef vectors[] = {diagonal, groups, kinetic};
  std::string values[9];
  for (int v = 0; v != 3; ++v)
    for (int k = 0; k != 3; ++k) {
      int slot = 3 * v + k;
      values[slot] = "%tms" + t + "_" + std::to_string(slot);
      os << indent << values[slot] << " = vector.extract " << vectors[v]
         << "[" << k << "] : f64 from vector<3xf64>\n"
         << indent << "%tmi" << t << "_" << slot << " = arith.constant "
         << slot << " : index\n"
         << indent << "memref.store " << values[slot]
         << ", %trotter_memory[%tmi" << t << "_" << slot
         << "] : memref<9xf64>\n";
    }
  // The checkpoints keep it, so a run that continues takes the same.
  os << indent << "func.call @mdrtSetBarostatState(";
  for (int slot = 0; slot != 9; ++slot)
    os << (slot ? ", " : "") << values[slot];
  os << ") : (f64, f64, f64, f64, f64, f64, f64, f64, f64) -> ()\n";
}

Builder::TrotterScaling Builder::finishTrotterStrain(StringRef indent,
                                                     StringRef groups,
                                                     StringRef tag) {
  std::string t = tag.str();
  TrotterScaling scaling;
  scaling.mu = "%mu" + t;
  scaling.muinv = "%muinv" + t;
  scaling.logMu = "%bs3" + t;
  scaling.workBefore = "%tw" + t;
  os << indent << scaling.workBefore << " = arith.addf " << groups
     << ", %bwcb" << t << " : vector<3xf64>\n";
  scaling.newVolume = "%tvn" + t;
  os << indent << "%tvn0" << t << " = arith.mulf %bn" << t << "_0, %bn" << t
     << "_1 : f64\n"
     << indent << scaling.newVolume << " = arith.mulf %tvn0" << t << ", %bn"
     << t << "_2 : f64\n";
  scaling.cell = "%tcell" + t;
  emitCellOf(indent, scaling.cell, "%bn" + t);
  if (scalesReference()) {
    scaling.scale = "%tscale" + t;
    emitReferenceScale(os, indent, scaling.scale, "%bn" + t + "_0",
                       "%bn" + t + "_1", "%bn" + t + "_2");
  }
  return scaling;
}

std::string Builder::emitKineticExcess(StringRef indent, StringRef tag,
                                       StringRef measured, StringRef forces,
                                       StringRef full) {
  std::string t = tag.str();
  std::string result = "%kex" + t;
  if (!measured.empty()) {
    emitKineticVector(os, "%kexf" + t, full, massName, indent);
    os << indent << result << " = arith.subf " << measured << ", %kexf" << t
       << " : vector<3xf64>\n";
    return result;
  }
  // Under Langevin dynamics with constraints neither is measured; Brownian
  // dynamics has no momenta.
  if (hasConstraints() || control.isBrownian())
    return "";
  // Without constraints, K(v - dt F / 2m) and K(v + dt F / 2m) average to
  // K(v) + (dt^2 / 8) sum F^2 / m [Jung2018].
  os << indent << "%kexg" << t << " = md.sum_particles gather(" << forces
     << ", " << massName << " : !vec, !real) {\n"
     << indent << "^bb0(%kx_f: vector<3xf64>, %kx_m: f64):\n"
     << indent << "  %kx_sq = arith.mulf %kx_f, %kx_f : vector<3xf64>\n"
     << indent << "  %kx_zero = arith.constant 0.0 : f64\n"
     << indent << "  %kx_one = arith.constant 1.0 : f64\n"
     << indent << "  %kx_massless = arith.cmpf oeq, %kx_m, %kx_zero : f64\n"
     << indent << "  %kx_safe = arith.select %kx_massless, %kx_one, %kx_m : f64\n"
     << indent << "  %kx_inv = arith.divf %kx_one, %kx_safe : f64\n"
     << indent << "  %kx_w = arith.select %kx_massless, %kx_zero, %kx_inv : f64\n"
     << indent << "  %kx_wb = vector.broadcast %kx_w : f64 to vector<3xf64>\n"
     << indent << "  %kx_g = arith.mulf %kx_sq, %kx_wb : vector<3xf64>\n"
     << indent << "  md.yield %kx_g : vector<3xf64>\n"
     << indent << "} : vector<3xf64>\n"
     << indent << "%kexd" << t << " = arith.mulf %dt, %dt : f64\n"
     << indent << "%kexc" << t << " = arith.constant 1.25e-01 : f64\n"
     << indent << "%kexs" << t << " = arith.mulf %kexd" << t << ", %kexc"
     << t << " : f64\n"
     << indent << "%kexb" << t << " = vector.broadcast %kexs" << t
     << " : f64 to vector<3xf64>\n"
     << indent << result << " = arith.mulf %kexg" << t << ", %kexb" << t
     << " : vector<3xf64>\n";
  return result;
}

Builder::Coupled
Builder::emitCoupling(StringRef indent, StringRef positions,
                      StringRef velocities, StringRef forces,
                      StringRef energy, StringRef tag, StringRef step,
                      StringRef diagonal, const TrotterScaling *trotter) {
  std::string newForces = forces.str();
  bool removesMotion = control.comPeriod > 0;
  std::string t = tag.str();
  if (removesMotion) {
    // The velocity of the center of mass.
    os << indent << "%pc" << t << " = md.sum_particles gather(" << velocities
       << ", " << massName << " : !vec, !real) {\n"
       << indent << "^bb0(%v_i: vector<3xf64>, %m_i: f64):\n"
       << indent << "  %mb = vector.broadcast %m_i : f64 to vector<3xf64>\n"
       << indent << "  %p = arith.mulf %mb, %v_i : vector<3xf64>\n"
       << indent << "  md.yield %p : vector<3xf64>\n"
       << indent << "} : vector<3xf64>\n"
       << indent << "%vcm" << t << " = arith.divf %pc" << t
       << ", %total_mass : vector<3xf64>\n";
  }
  // The kinetic energy of the center of mass, P·V / 2, which the removal
  // takes away.
  if (removesMotion)
    os << indent << "%pvs" << t << " = arith.mulf %pc" << t << ", %vcm" << t
       << " : vector<3xf64>\n"
       << indent << "%pv" << t << " = vector.reduction <add>, %pvs" << t
       << " : vector<3xf64> into f64\n"
       << indent << "%kcm" << t << " = arith.mulf %couple_half, %pv" << t
       << " : f64\n";
  else
    os << indent << "%kcm" << t << " = arith.constant 0.0 : f64\n";
  // The energy that the coupling takes: that of the center of mass, and
  // what the thermostat takes from the rest.
  std::string bath = "%kcm" + t;
  std::string kineticAxes;
  if (control.barostat) {
    // The kinetic energy of each axis, which the pressure of each axis
    // takes, and of the center of mass; their sums for the thermostat.
    os << indent << "// The kinetic energy of each axis.\n";
    emitKineticVector(os, "%kv" + t, velocities, massName, indent);
    emitSum3(os, "%kc" + t, "%kv" + t, indent);
    std::string vector = "%kv" + t;
    if (removesMotion) {
      os << indent << "%pvv" << t << " = arith.mulf %pc" << t << ", %vcm" << t
         << " : vector<3xf64>\n"
         << indent << "%kcmv" << t << " = arith.mulf %c_half3, %pvv" << t
         << " : vector<3xf64>\n"
         << indent << "%ktv" << t << " = arith.subf %kv" << t << ", %kcmv"
         << t << " : vector<3xf64>\n";
      vector = "%ktv" + t;
    }
    kineticAxes = vector;
  }
  if (rescalesVelocities()) {
    std::string kinetic = "%kc" + t;
    if (!control.barostat)
      emitKineticEnergy(os, kinetic, velocities, massName, indent);
    if (removesMotion) {
      os << indent << "%kt" << t << " = arith.subf %kc" << t << ", %kcm" << t
         << " : f64\n";
      kinetic = "%kt" + t;
    }
    if (control.isNoseHoover()) {
      // A Nose-Hoover chain (D163a) moves on the host over the period and
      // counts the change of its energy into the bath itself.
      os << indent << "%alpha" << t << " = func.call @mdrtNoseHooverFactor("
         << step << ", " << kinetic << ") : (i64, f64) -> f64\n"
         << indent << "%alpha2" << t << " = arith.mulf %alpha" << t
         << ", %alpha" << t << " : f64\n";
    } else {
      os << indent << "%alpha" << t << " = func.call @mdrtBussiFactor(%seed, "
         << step << ", " << kinetic
         << ", %target_kinetic, %freedom, %decay)\n"
         << indent << "    : (i64, i64, f64, f64, f64, f64) -> f64\n"
         << indent << "%alpha2" << t << " = arith.mulf %alpha" << t
         << ", %alpha" << t << " : f64\n"
         << indent << "%kn" << t << " = arith.mulf %alpha2" << t << ", "
         << kinetic << " : f64\n"
         << indent << "%heat" << t << " = arith.subf %kc" << t << ", %kn"
         << t << " : f64\n";
      bath = "%heat" + t;
    }
  }
  // Stochastic cell rescaling (Bernetti and Bussi 2020): the pressure of
  // the step, with the virials of the correction for the dispersion and of
  // the background of a net charge at this volume, gives the change ε of
  // the logarithm of the volume. The positions and the cell scale by
  // μ = exp(ε/3), the velocities by 1/μ (design-m1.md, Section 11.4).
  std::string newPositions = positions.str();
  if (control.barostat) {
    // The kinetic energy of the pressure: that of the velocities of the
    // step, without the center of mass.
    std::string kinetic = kineticAxes;
    if (!trotter)
      emitStrain(indent, kinetic, diagonal, t, step);
    // The energy that the scaling gives the system, summed over the axes:
    // exactly in the velocities, (1/μ_a² − 1) K_a, and in the positions
    // either exactly, from the energy of the scaled positions (below), or
    // to first order, −(μ_a − 1) W_aa, with W the virial of the groups and
    // of the constant terms.
    bool exact = control.barostatWork == BarostatWork::Exact;
    // The kinetic energy of each axis after the thermostat.
    std::string after = kinetic;
    if (rescalesVelocities()) {
      os << indent << "%alpha2b" << t << " = vector.broadcast %alpha2" << t
         << " : f64 to vector<3xf64>\n"
         << indent << "%knv" << t << " = arith.mulf %alpha2b" << t << ", "
         << kinetic << " : vector<3xf64>\n";
      after = "%knv" + t;
    }
    std::string relations =
        ("%r" + StringRef(fieldPrefix).drop_front(2)).str();
    if (trotter) {
      // The scaling was made within the drift of the step (Trotter type,
      // [Bernetti2020], SI Sec. V.C): the velocities of its middle by 1/μ_a
      // along axis a, which changes their kinetic energy by
      // Σ (1/μ_a² − 1) K_a, and the positions by μ_a, which changes the
      // potential energy by −Σ ln μ_a W_aa to first order, W the virial of
      // the groups, Σ X ⊙ F_group, with those of the constant terms. W is
      // taken as the mean of those of the evaluations before and after the
      // scaling, which makes the count exact to second order in the strain
      // (D92). The virial of the step with twice the internal kinetic
      // energy would not do: the constraints of the step that scales
      // straddle the scaling, and the count was biased by the square of the
      // strain (D116).
      // To first order, from the virial before the scaling only.
      if (!countsAfterScaling()) {
        os << indent << "%bwl" << t << " = arith.mulf "
           << trotter->workBefore << ", " << trotter->logMu
           << " : vector<3xf64>\n";
        emitSum3(os, "%bwork" + t, "%bwl" + t, indent);
      }
      std::string groups = countsAfterScaling() ? trotter->groupsAfter : "";
      // With a scaling every step, the state that the next one takes its
      // pressure from: the kinetic energy after the thermostat.
      if (scalesEveryStep())
        emitStoreTrotterState(indent, diagonal, groups, after);
      if (countsAfterScaling()) {
        os << indent << "%bwca" << t << " = arith.divf %baro_constant, "
           << trotter->newVolume << " : f64\n"
           << indent << "%bwca3" << t << " = arith.mulf %bwca" << t
           << ", %c_third : f64\n"
           << indent << "%bwcab" << t << " = vector.broadcast %bwca3" << t
           << " : f64 to vector<3xf64>\n"
           << indent << "%bwb" << t << " = arith.addf " << groups << ", %bwcab"
           << t << " : vector<3xf64>\n"
           << indent << "%bws" << t << " = arith.addf " << trotter->workBefore
           << ", %bwb" << t << " : vector<3xf64>\n"
           << indent << "%bwm" << t << " = arith.mulf %bws" << t
           << ", %c_half3 : vector<3xf64>\n"
           << indent << "%bwl" << t << " = arith.mulf %bwm" << t << ", "
           << trotter->logMu << " : vector<3xf64>\n";
        emitSum3(os, "%bwork" + t, "%bwl" + t, indent);
      }
      os << indent << "%bmi2" << t << " = arith.mulf " << trotter->muinv
         << ", " << trotter->muinv << " : vector<3xf64>\n"
         << indent << "%bmi21" << t << " = arith.subf %bmi2" << t
         << ", %c_unit3 : vector<3xf64>\n"
         << indent << "%bdkv" << t << " = arith.mulf %bmi21" << t << ", "
         << trotter->kineticHalf << " : vector<3xf64>\n";
      emitSum3(os, "%bdk" + t, "%bdkv" + t, indent);
      os
         << indent << "%bdku" << t << " = arith.subf %bdk" << t << ", %bwork"
         << t << " : f64\n"
         << indent << "%btake" << t << " = arith.subf " << bath << ", %bdku"
         << t << " : f64\n";
      bath = "%btake" + t;
    }
    if (!trotter)
      os << indent << "%bm1" << t << " = arith.subf %c_unit3, %mu" << t
         << " : vector<3xf64>\n";
    if (!exact && !trotter) {
      // To first order, isotropic only (Control): the trace of the virial
      // of the step with twice the internal kinetic energy, a third to
      // each axis.
      emitSum3(os, "%dtr" + t, diagonal, indent);
      std::string groupTrace =
          emitGroupTrace(indent, "%dtr" + t, positions, velocities, cellName,
                         massName, relations, t);
      os << indent << "%bwt" << t << " = arith.addf " << groupTrace
         << ", %bwc" << t << " : f64\n"
         << indent << "%bwt3" << t << " = arith.mulf %bwt" << t
         << ", %c_third : f64\n"
         << indent << "%bwtb" << t << " = vector.broadcast %bwt3" << t
         << " : f64 to vector<3xf64>\n"
         << indent << "%bwl" << t << " = arith.mulf %bm1" << t << ", %bwtb"
         << t << " : vector<3xf64>\n";
      emitSum3(os, "%bwork" + t, "%bwl" + t, indent);
    }
    if (!trotter) {
      os << indent << "%bmi2" << t << " = arith.mulf %muinv" << t
         << ", %muinv" << t << " : vector<3xf64>\n"
         << indent << "%bmi21" << t << " = arith.subf %bmi2" << t
         << ", %c_unit3 : vector<3xf64>\n"
         << indent << "%bdkv" << t << " = arith.mulf %bmi21" << t << ", "
         << after << " : vector<3xf64>\n";
      emitSum3(os, "%bdk" + t, "%bdkv" + t, indent);
    }
    if (!exact && !trotter) {
      os << indent << "%bgain" << t << " = arith.addf %bwork" << t << ", %bdk"
         << t << " : f64\n"
         << indent << "%btake" << t << " = arith.subf " << bath << ", %bgain"
         << t << " : f64\n";
      bath = "%btake" + t;
    }
    if (!trotter)
      newPositions = emitGroupScaling(indent, positions, "%mu" + t,
                                      "%bm1" + t, cellName, massName,
                                      relations, t);
    if (exact && !trotter) {
      // The scaled positions in the new cell, with the virtual sites placed
      // on their atoms, and their energy and forces, which the next step
      // takes: the change of the potential energy is counted exactly, and
      // no step begins with the forces of other positions (D77).
      std::string cell = "%bcell" + t;
      emitCellOf(indent, cell, "%bn" + t);
      std::string outerCell = cellName, outerScale = scaleName;
      cellName = cell;
      if (scalesReference()) {
        scaleName = "%bscale" + t;
        emitReferenceScale(os, indent, scaleName, "%bn" + t + "_0",
                           "%bn" + t + "_1", "%bn" + t + "_2");
      }
      if (hasSites()) {
        std::string placed = "%xcs" + t;
        emitPlaceSites(indent, newPositions, placed, relations);
        newPositions = placed;
      }
      std::string held = hasRestraints() ? "p" : "";
      std::string raw = hasSites() ? "e" : "";
      // The work takes the energy, and the next step the forces; the
      // virial of the scaled positions is not needed.
      std::string u = "%bu" + t, f = "%bf" + t;
      // The time of the step that the scaling follows (D145).
      std::string time = "%btime" + t;
      if (control.usesTime)
        os << indent << time << "_steps_i = memref.load %noise_memory[%c0]"
           << " : memref<1xi64>\n"
           << indent << time << "_steps = arith.sitofp " << time
           << "_steps_i : i64 to f64\n"
           << indent << time << " = arith.mulf " << time << "_steps, %dt"
           << " : f64\n";
      os << indent << u << held << ", " << f << held << raw
         << " = md.evaluate @energy(" << newPositions << ", " << cell
         << getFieldValues(fieldPrefix) << getTimeValue(time) << ")\n"
         << indent << "    request [energy, forces]\n"
         << indent << "    : (!vec, !md.cell" << getFieldTypes()
         << getTimeType() << ") -> (f64, !vec)\n";
      if (hasSites())
        emitSpreadSites(indent, newPositions, f + held + "e", f + held,
                        relations);
      if (hasRestraints())
        emitRestraints(indent, newPositions, fieldPrefix, f + "p", f,
                       u + "p", u);
      cellName = outerCell;
      scaleName = outerScale;
      newForces = f;
      // The constant terms of the energy, the correction for the
      // dispersion and the background of a net charge, are proportional to
      // 1 / V and not in `md.evaluate`: their change, C (1/V' - 1/V), is
      // part of the work too, or the conserved energy would carry
      // E_c(V) - E_c(V_0).
      os << indent << "%bnxy" << t << " = arith.mulf %bn" << t << "_0, %bn"
         << t << "_1 : f64\n"
         << indent << "%bnv" << t << " = arith.mulf %bnxy" << t << ", %bn" << t
         << "_2 : f64\n"
         << indent << "%bnvi" << t << " = arith.divf %c_unit, %bnv" << t
         << " : f64\n"
         << indent << "%bovi" << t << " = arith.divf %c_unit, %bv" << t
         << " : f64\n"
         << indent << "%bdvi" << t << " = arith.subf %bnvi" << t << ", %bovi"
         << t << " : f64\n"
         << indent << "%bdc" << t << " = arith.mulf %baro_energy_constant, %bdvi"
         << t << " : f64\n"
         << indent << "%bdue" << t << " = arith.subf " << u << ", " << energy
         << " : f64\n"
         << indent << "%bdu" << t << " = arith.addf %bdue" << t << ", %bdc" << t
         << " : f64\n"
         << indent << "%bgain" << t << " = arith.addf %bdu" << t << ", %bdk"
         << t << " : f64\n"
         << indent << "%btake" << t << " = arith.subf " << bath << ", %bgain"
         << t << " : f64\n";
      bath = "%btake" + t;
    }
  }
  os << indent << "func.call @mdrtAddBath(" << bath << ") : (f64) -> ()\n";
  os << indent << "%vc" << t << " = md.map_particles gather(" << velocities
     << " : !vec) {\n"
     << indent << "^bb0(%v_i: vector<3xf64>):\n";
  std::string value = "%v_i";
  if (removesMotion) {
    os << indent << "  %d = arith.subf %v_i, %vcm" << t
       << " : vector<3xf64>\n";
    value = "%d";
  }
  if (rescalesVelocities()) {
    os << indent << "  %ab = vector.broadcast %alpha" << t
       << " : f64 to vector<3xf64>\n"
       << indent << "  %s = arith.mulf %ab, " << value
       << " : vector<3xf64>\n";
    value = "%s";
  }
  if (control.barostat && !trotter) {
    os << indent << "  %v_scaled = arith.mulf %muinv" << t << ", " << value
       << " : vector<3xf64>\n";
    value = "%v_scaled";
  }
  os << indent << "  md.yield " << value << " : vector<3xf64>\n"
     << indent << "} : !vec\n";
  return {newPositions, "%vc" + t, newForces};
}

void Builder::emitRestraints(StringRef indent, StringRef x,
                             StringRef prefix, StringRef f,
                             StringRef fResult, StringRef u,
                             StringRef uResult, StringRef w,
                             StringRef wResult) {
  std::string fields = (prefix + "rest_k, " + prefix + "rest_x, " + prefix +
                        "rest_y, " + prefix + "rest_z")
                           .str();
  std::string types = "!real, !real, !real, !real";
  std::string arguments = "%k_i: f64, %rx_i: f64, %ry_i: f64, %rz_i: f64";
  if (scalesReference()) {
    fields += (", " + prefix + "rest_cx, " + prefix + "rest_cy, " + prefix +
               "rest_cz")
                  .str();
    types += ", !real, !real, !real";
    arguments += ", %rcx_i: f64, %rcy_i: f64, %rcz_i: f64";
  }
  // d = x − x_ref, and k d. Under a barostat the fields hold the offsets o
  // of the references from their centers c, and the center follows the cell
  // while the offset stays, x_ref = s ⊙ c + o (D124).
  auto emitOffset = [&](StringRef inner) {
    std::string reference = "%ref_i";
    os << inner << reference << " = vector.from_elements %rx_i, %ry_i, %rz_i "
                   ": vector<3xf64>\n";
    if (scalesReference()) {
      os << inner << "%rc_i = vector.from_elements %rcx_i, %rcy_i, %rcz_i "
                     ": vector<3xf64>\n"
         << inner << "%rcs_i = arith.mulf " << scaleName
         << ", %rc_i : vector<3xf64>\n"
         << inner << "%refs_i = arith.addf %rcs_i, %ref_i : vector<3xf64>\n";
      reference = "%refs_i";
    }
    os << inner << "%d_i = arith.subf %x_i, " << reference
       << " : vector<3xf64>\n"
       << inner << "%kb_i = vector.broadcast %k_i : f64 to vector<3xf64>\n"
       << inner << "%kd_i = arith.mulf %kb_i, %d_i : vector<3xf64>\n";
  };
  std::string inner = (indent + "  ").str();
  os << indent << fResult << " = md.map_particles gather(" << x << ", " << f
     << ", " << fields << " : !vec, !vec, " << types << ") {\n"
     << indent << "^bb0(%x_i: vector<3xf64>, %f_i: vector<3xf64>, "
     << arguments << "):\n";
  emitOffset(inner);
  os << inner << "%two_i = arith.constant dense<2.0> : vector<3xf64>\n"
     << inner << "%pull_i = arith.mulf %two_i, %kd_i : vector<3xf64>\n"
     << inner << "%fr_i = arith.subf %f_i, %pull_i : vector<3xf64>\n"
     << inner << "md.yield %fr_i : vector<3xf64>\n"
     << indent << "} : !vec\n";
  if (u.empty() && w.empty())
    return;
  // Σ k d ⊙ d, whose sum is the energy and whose elements times −2 are the
  // diagonal of the virial. Under a barostat the virial is Σ F ⊙ (x − s ⊙ c)
  // = −2 Σ k d ⊙ (d + o), the derivative of the energy when the positions
  // and the centers of the references scale and their offsets do not.
  std::string sum = ((u.empty() ? wResult : uResult) + "_parts").str();
  os << indent << sum << " = md.sum_particles gather(" << x << ", " << fields
     << " : !vec, " << types << ") {\n"
     << indent << "^bb0(%x_i: vector<3xf64>, " << arguments << "):\n";
  emitOffset(inner);
  os << inner << "%kdd_i = arith.mulf %kd_i, %d_i : vector<3xf64>\n"
     << inner << "md.yield %kdd_i : vector<3xf64>\n"
     << indent << "} : vector<3xf64>\n";
  std::string virialSum = sum;
  if (!w.empty() && scalesReference()) {
    virialSum = (wResult + "_vparts").str();
    os << indent << virialSum << " = md.sum_particles gather(" << x << ", "
       << fields << " : !vec, " << types << ") {\n"
       << indent << "^bb0(%x_i: vector<3xf64>, " << arguments << "):\n";
    emitOffset(inner);
    os << inner << "%dpo_i = arith.addf %d_i, %ref_i : vector<3xf64>\n"
       << inner << "%kdo_i = arith.mulf %kd_i, %dpo_i : vector<3xf64>\n"
       << inner << "md.yield %kdo_i : vector<3xf64>\n"
       << indent << "} : vector<3xf64>\n";
  }
  if (!u.empty())
    os << indent << uResult << "_r = vector.reduction <add>, " << sum
       << " : vector<3xf64> into f64\n"
       << indent << uResult << " = arith.addf " << u << ", " << uResult
       << "_r : f64\n";
  if (w.empty())
    return;
  std::string zero = (wResult + "_zero").str();
  os << indent << zero << " = arith.constant 0.0 : f64\n"
     << indent << wResult << "_m2 = arith.constant -2.0 : f64\n";
  std::string diagonal[3];
  for (int c = 0; c != 3; ++c) {
    diagonal[c] = (wResult + "_d" + std::to_string(c)).str();
    os << indent << diagonal[c] << "_s = vector.extract " << virialSum
       << "[" << c << "] : f64 from vector<3xf64>\n"
       << indent << diagonal[c] << " = arith.mulf " << diagonal[c] << "_s, "
       << wResult << "_m2 : f64\n";
  }
  os << indent << wResult << "_r = vector.from_elements " << diagonal[0]
     << ", " << zero << ", " << zero << ", " << zero << ", " << diagonal[1]
     << ", " << zero << ", " << zero << ", " << zero << ", " << diagonal[2]
     << " : vector<9xf64>\n"
     << indent << wResult << " = arith.addf " << w << ", " << wResult
     << "_r : vector<9xf64>\n";
}

void Builder::emitConstrainedDescent(StringRef indent, StringRef x,
                                     StringRef f, StringRef result) {
  bool settles = hasSettles();
  std::vector<const Program::TupleSet *> shakeSets = getShakeSets();
  unsigned steps = (settles ? 1 : 0) + shakeSets.size(), step = 0;
  std::string current = (result + "_all").str();
  if (steps == 0)
    current = result.str();
  os << indent << current << " = md.map_particles gather(" << f
     << ", %m : !vec, !real) {\n"
     << indent << "^bb0(%f_i: vector<3xf64>, %m_i: f64):\n"
     << indent << "  %zero_s = arith.constant 0.0 : f64\n"
     << indent << "  %unit_s = arith.constant 1.0 : f64\n"
     << indent << "  %zero_v = arith.constant dense<0.0> : vector<3xf64>\n"
     << indent << "  %massive = arith.cmpf ogt, %m_i, %zero_s : f64\n"
     << indent << "  %m_safe = arith.select %massive, %m_i, %unit_s : f64\n"
     << indent << "  %mb = vector.broadcast %m_safe : f64 to vector<3xf64>\n"
     << indent << "  %ratio = arith.divf %f_i, %mb : vector<3xf64>\n"
     << indent << "  %g_i = arith.select %massive, %ratio, %zero_v "
        ": vector<3xf64>\n"
     << indent << "  md.yield %g_i : vector<3xf64>\n"
     << indent << "} : !vec\n";
  // As RATTLE takes the velocities off the constraints.
  auto next = [&]() {
    ++step;
    return step == steps ? result.str()
                         : (result + "_c" + std::to_string(step)).str();
  };
  if (settles) {
    std::string projected = next();
    emitSettleVelocities(indent, x, current, projected);
    current = projected;
  }
  for (const Program::TupleSet *set : shakeSets) {
    std::string projected = next();
    emitShakeVelocities(indent, x, current, *set, projected, "", "");
    current = projected;
  }
}

std::string Builder::emitStartConstraintTrace(StringRef x, StringRef f,
                                              StringRef v, StringRef trace,
                                              bool axes) {
  // With `axes`, the diagonal instead of the trace, each axis with twice
  // its own internal kinetic energy: the identity holds for the trace, and
  // for each axis only in the mean, which serves the first scaling of a
  // run (D119).
  std::string name = axes ? "%dgc0" : "%trc0";
  std::string type = axes ? "vector<3xf64>" : "f64";
  std::string descent = axes ? "%gc0d" : "%gc0";
  emitConstrainedDescent("  ", x, f, descent);
  std::vector<const Program::TupleSet *> groups = getShakeSets();
  for (const Program::TupleSet &set : program.tupleSets)
    if (set.name == "settles")
      groups.insert(groups.begin(), &set);
  std::string current = trace.str();
  for (const Program::TupleSet *set : groups) {
    unsigned count = set->arity - 1;
    std::string coordinates, arguments;
    for (unsigned k = 1; k <= count; ++k) {
      coordinates += (k == 1 ? "" : ", ") +
                     ("displacement(" + std::to_string(k) + ", 0)");
      arguments += "%vs_r" + std::to_string(k) + ": vector<3xf64>, ";
    }
    for (StringRef field : {"g", "f", "v"})
      for (unsigned k = 0; k <= count; ++k)
        arguments += ("%vs_" + field + std::to_string(k) + ": vector<3xf64>, ")
                         .str();
    for (unsigned k = 0; k <= count; ++k)
      arguments += "%vs_m" + std::to_string(k) + ": f64" +
                   (k == count ? "" : ", ");
    std::string sum = name + "_" + set->name;
    os << "  " << sum << " = md.sum_tuples %r_" << set->name << ", " << x
       << ", %cell\n"
       << "    coordinates(" << coordinates << ")\n"
       << "    gather(" << descent << ", " << f << ", " << v
       << ", %m : !vec, !vec, !vec, !real) {\n"
       << "  ^bb0(" << arguments << "):\n";
    SiteKernel k(os, "    ");
    // Σ (x_k − x_0) · G⁰_k over the particles but the first, whose arm is
    // 0.
    std::string arms;
    for (unsigned j = 1; j <= count; ++j) {
      std::string n = std::to_string(j);
      std::string force = k.vector(
          "subf", k.scale("%vs_m" + n, "%vs_g" + n), "%vs_f" + n);
      std::string term = axes ? k.vector("mulf", "%vs_r" + n, force)
                              : k.dot("%vs_r" + n, force);
      arms = arms.empty() ? term
                          : axes ? k.vector("addf", arms, term)
                                 : k.real("addf", arms, term);
    }
    // 2 K_int = Σ m |v − V|².
    std::string total = "%vs_m0", moment = k.scale("%vs_m0", "%vs_v0");
    for (unsigned j = 1; j <= count; ++j) {
      std::string n = std::to_string(j);
      total = k.real("addf", total, "%vs_m" + n);
      moment = k.vector("addf", moment, k.scale("%vs_m" + n, "%vs_v" + n));
    }
    std::string center =
        k.scale(k.real("divf", k.constant(1.0), total), moment);
    std::string twice;
    for (unsigned j = 0; j <= count; ++j) {
      std::string n = std::to_string(j);
      std::string relative = k.vector("subf", "%vs_v" + n, center);
      std::string term =
          axes ? k.scale("%vs_m" + n, k.vector("mulf", relative, relative))
               : k.real("mulf", "%vs_m" + n, k.dot(relative, relative));
      twice = twice.empty() ? term
                            : axes ? k.vector("addf", twice, term)
                                   : k.real("addf", twice, term);
    }
    std::string virial = axes ? k.vector("subf", arms, twice)
                              : k.real("subf", arms, twice);
    os << "    md.yield " << virial << " : " << type << "\n"
       << "  } : !rel_" << set->name << ", !vec -> " << type << "\n";
    std::string next = name + "_" + set->name + "_sum";
    os << "  " << next << " = arith.addf " << current << ", " << sum
       << " : " << type << "\n";
    current = next;
  }
  return current;
}

void Builder::emitDescend() {
  // The step moves each particle along g, its force over its mass without
  // the parts along the constraints, by `%h` times g over the 16-norm of
  // g. That norm is no less than the largest |g|, so that no particle
  // moves farther than `%h`, and it is taken of g over its root mean
  // square, which keeps the powers finite. The constraints weigh the
  // particles by their masses, so that the step stays downhill once they
  // take the groups back to their shapes. Virtual sites have no mass and
  // do not move.
  bool sites = hasSites();
  bool settles = hasSettles();
  std::vector<const Program::TupleSet *> shakeSets = getShakeSets();
  bool constraints = settles || !shakeSets.empty();
  std::string evaluate = "md.evaluate @energy(%x1, %cell" + getFieldValues() +
                         getTimeValue("%zero") + ")";
  std::string signature =
      "(!vec, !md.cell" + getFieldTypes() + getTimeType() + ")";
  os << "dyn.program @descend(%x: !vec, %f: !vec, %m: !real, "
        "%cell: !md.cell,\n    %h: f64"
     << getFieldParameters() << ")\n    -> (!vec, !vec, f64) {\n"
     << "  %zero = arith.constant 0.0 : f64\n"
     << "  %tiny = arith.constant 1.0e-300 : f64\n"
     << "  %count = arith.constant "
     << formatReal(static_cast<double>(system.getNumParticles()))
     << " : f64\n";
  emitConstrainedDescent("  ", "%x", "%f", "%g");
  os << "  %square = md.sum_particles gather(%g : !vec) {\n"
     << "  ^bb0(%f_i: vector<3xf64>):\n"
     << "    %sq = arith.mulf %f_i, %f_i : vector<3xf64>\n"
     << "    %s = vector.reduction <add>, %sq : vector<3xf64> into f64\n"
     << "    md.yield %s : f64\n"
     << "  } : f64\n"
     << "  %mean = arith.divf %square, %count : f64\n"
     << "  %mean_safe = arith.maximumf %mean, %tiny : f64\n"
     << "  %rms = math.sqrt %mean_safe : f64\n"
     << "  %power = md.sum_particles gather(%g : !vec) {\n"
     << "  ^bb0(%f_i: vector<3xf64>):\n"
     << "    %sq = arith.mulf %f_i, %f_i : vector<3xf64>\n"
     << "    %s = vector.reduction <add>, %sq : vector<3xf64> into f64\n"
     << "    %r2 = arith.divf %s, %mean_safe : f64\n"
     << "    %r4 = arith.mulf %r2, %r2 : f64\n"
     << "    %r8 = arith.mulf %r4, %r4 : f64\n"
     << "    %r16 = arith.mulf %r8, %r8 : f64\n"
     << "    md.yield %r16 : f64\n"
     << "  } : f64\n"
     << "  %root2 = math.sqrt %power : f64\n"
     << "  %root4 = math.sqrt %root2 : f64\n"
     << "  %root8 = math.sqrt %root4 : f64\n"
     << "  %root16 = math.sqrt %root8 : f64\n"
     << "  %norm = arith.mulf %rms, %root16 : f64\n"
     << "  %scale = arith.divf %h, %norm : f64\n"
     << "  %x1" << (sites || constraints ? "d" : "")
     << " = md.map_particles gather(%x, %g : !vec, !vec) {\n"
     << "  ^bb0(%x_i: vector<3xf64>, %f_i: vector<3xf64>):\n"
     << "    %sb = vector.broadcast %scale : f64 to vector<3xf64>\n"
     << "    %d = arith.mulf %sb, %f_i : vector<3xf64>\n"
     << "    %moved = arith.addf %x_i, %d : vector<3xf64>\n"
     << "    md.yield %moved : vector<3xf64>\n"
     << "  } : !vec\n";
  // The rigid groups back to their shapes, and the sites where their
  // atoms put them.
  if (constraints) {
    std::string current = "%x1d";
    std::string constrained = sites ? "%x1s" : "%x1";
    unsigned steps = (settles ? 1 : 0) + shakeSets.size(), step = 0;
    auto next = [&]() {
      ++step;
      return step == steps ? constrained : "%x1c" + std::to_string(step);
    };
    if (settles) {
      std::string result = next();
      emitSettlePositions("  ", "%x", current, "%dx1", result);
      current = result;
    }
    for (const Program::TupleSet *set : shakeSets) {
      std::string result = next();
      emitShakePositions("  ", "%x", current, *set, result);
      current = result;
    }
  }
  if (sites)
    emitPlaceSites("  ", constraints ? "%x1s" : "%x1d", "%x1", "%r_");
  std::string raw = sites ? "e" : "";
  std::string held = hasRestraints() ? "p" : "";
  os << "  %u1" << held << ", %f1" << held << raw << " = " << evaluate
     << "\n      request [energy, forces]\n"
     << "      : " << signature << " -> (f64, !vec)\n";
  if (sites)
    emitSpreadSites("  ", "%x1", "%f1" + held + "e", "%f1" + held, "%r_");
  if (hasRestraints()) {
    os << "  %w1z = arith.constant dense<0.0> : vector<9xf64>\n";
    emitRestraints("  ", "%x1", "%p_", "%f1p", "%f1", "%u1p", "%u1", "%w1z",
                   "%w1r");
  }
  os << "  dyn.return %x1, %f1, %u1 : !vec, !vec, f64\n}\n\n";
}

void Builder::emitMinimization() {
  // The positions of the file on the surface of the constraints first. A
  // step is taken from positions on it and is constrained again, so a
  // start off it would put into every trial the change that takes the
  // groups to their shapes, which does not shrink with the step: where
  // the file has close contacts (a bilayer from packmol, bonds to
  // hydrogens 0.02 Å from their lengths) that change alone can raise the
  // energy, and no step is ever taken. A segment that continues a
  // minimization begins where the last one ended, on that surface
  // (D202).
  std::string x0 = "%x0";
  bool settles = hasSettles();
  std::vector<const Program::TupleSet *> shakeSets = getShakeSets();
  if ((settles || !shakeSets.empty()) && !control.continuesSegment) {
    std::string current = "%x0";
    unsigned steps = (settles ? 1 : 0) + shakeSets.size(), step = 0;
    std::string constrained = hasSites() ? "%x0ks" : "%x0k";
    auto next = [&]() {
      ++step;
      return step == steps ? constrained : "%x0k" + std::to_string(step);
    };
    if (settles) {
      std::string result = next();
      emitSettlePositions("  ", "%x0", current, "%dx0k", result);
      current = result;
    }
    for (const Program::TupleSet *set : shakeSets) {
      std::string result = next();
      emitShakePositions("  ", "%x0", current, *set, result);
      current = result;
    }
    if (hasSites())
      emitPlaceSites("  ", "%x0ks", "%x0k", "%r_");
    x0 = "%x0k";
  }
  // The energy and the forces at the start, and the terms.
  std::string raw = hasSites() ? "e" : "";
  std::string held = hasRestraints() ? "p" : "";
  os << "  %u0" << held << ", %f0" << held << raw
     << " = md.evaluate @energy(" << x0 << ", %cell" << getFieldValues()
     << getTimeValue("%time0") << ")\n"
     << "      request [energy, forces]\n"
     << "      : (!vec, !md.cell" << getFieldTypes() << getTimeType()
     << ") -> (f64, !vec)\n";
  if (hasSites())
    emitSpreadSites("  ", x0, "%f0" + held + "e", "%f0" + held, "%r_");
  if (hasRestraints()) {
    os << "  %w0z = arith.constant dense<0.0> : vector<9xf64>\n";
    emitRestraints("  ", x0, "%p_", "%f0p", "%f0", "%u0p", "%u0", "%w0z",
                   "%w0r");
  }
  // The terms at the start, which a continued segment does not log.
  if (!isRestart())
    emitTerms(x0);
  // The length of the first step: that of the control file, or the one
  // that the last segment left.
  std::string h0 = control.segments ? "%first_size" : "%h0";
  if (!control.segments)
    os << "  %h0 = arith.constant "
       << formatReal(control.minimizeStep * units::length) << " : f64\n";
  os << "  %h_most = arith.constant " << formatReal(1.0 * units::length)
     << " : f64\n"
     << "  %grow = arith.constant 1.2 : f64\n"
     << "  %shrink = arith.constant 0.2 : f64\n"
     << "  %one = arith.constant 1.0 : f64\n"
     << "  %zero = arith.constant 0.0 : f64\n"
     ;
  // The log has the forces without their parts along the constraints,
  // those that the minimization lowers.
  auto emitReported = [&](StringRef indent, StringRef x, StringRef f,
                          StringRef tag) {
    std::string direction = ("%gr" + tag).str();
    emitConstrainedDescent(indent, x, f, direction);
    std::string reported = ("%fr" + tag).str();
    os << indent << reported << " = md.map_particles gather(" << direction
       << ", %m : !vec, !real) {\n"
       << indent << "^bb0(%g_i: vector<3xf64>, %m_i: f64):\n"
       << indent << "  %mb = vector.broadcast %m_i : f64 to vector<3xf64>\n"
       << indent << "  %f_i = arith.mulf %mb, %g_i : vector<3xf64>\n"
       << indent << "  md.yield %f_i : vector<3xf64>\n"
       << indent << "} : !vec\n";
    return reported;
  };
  std::string reported = emitReported("  ", x0, "%f0", "0");
  os << "  mdrt.host_call @mdrtWriteMinimization(%start, %u0, " << h0 << ", "
     << reported << ", %id)\n"
     << "      : (i64, f64, f64, !vec, !ids)\n";

  // A step is taken if it lowers the energy, and the next one is longer;
  // otherwise the next one is shorter, from where the step began. The
  // loops: over the intervals between frames (one if there are none),
  // over the intervals between energies in each, and over the steps.
  // A segment (D202) is one interval of the steps that
  // the entry takes, with a row at its end.
  std::string state = "!vec, !vec, f64, f64";
  if (control.segments) {
    os << "  %n0 = arith.constant 1 : index\n"
       << "  %n1 = arith.constant 1 : index\n"
       << "  %n2 = arith.index_cast %count_outer : i64 to index\n"
       << "  %per0 = arith.addi %n2, %c0 : index\n";
  } else {
    int64_t period = control.energyPeriod;
    int64_t framePeriod =
        control.framePeriod > 0 ? control.framePeriod : control.numSteps;
    os << "  %n0 = arith.constant " << control.numSteps / framePeriod
       << " : index\n"
       << "  %n1 = arith.constant " << framePeriod / period << " : index\n"
       << "  %n2 = arith.constant " << period << " : index\n"
       << "  %per0 = arith.constant " << framePeriod << " : index\n";
  }
  os << "  %xe0, %fe0, %ue0, %he0 = scf.for %i0 = %c0 to %n0 step %c1\n"
     << "      iter_args(%xa0 = " << x0
     << ", %fa0 = %f0, %ua0 = %u0, %ha0 = " << h0 << ")\n"
     << "      -> (" << state << ") {\n"
     << "    %xe1, %fe1, %ue1, %he1 = scf.for %i1 = %c0 to %n1 step %c1\n"
     << "        iter_args(%xa1 = %xa0, %fa1 = %fa0, %ua1 = %ua0, "
        "%ha1 = %ha0)\n"
     << "        -> (" << state << ") {\n"
     << "      %xe2, %fe2, %ue2, %he2 = scf.for %i2 = %c0 to %n2 step %c1\n"
     << "          iter_args(%xa2 = %xa1, %fa2 = %fa1, %ua2 = %ua1, "
        "%ha2 = %ha1)\n"
     << "          -> (" << state << ") {\n"
     << "        %xt, %ft, %ut = dyn.step @descend(%xa2, %fa2, %m, %cell, "
        "%ha2" << getFieldValues() << ")\n"
     << "            : (!vec, !vec, !real, !md.cell, f64" << getFieldTypes()
     << ") -> (!vec, !vec, f64)\n"
     << "        %lower = arith.cmpf olt, %ut, %ua2 : f64\n"
     << "        %longer = arith.mulf %ha2, %grow : f64\n"
     << "        %capped = arith.minimumf %longer, %h_most : f64\n"
     << "        %shorter = arith.mulf %ha2, %shrink : f64\n"
     << "        %ub = arith.select %lower, %ut, %ua2 : f64\n"
     << "        %hb = arith.select %lower, %capped, %shorter : f64\n"
     << "        %keep = arith.select %lower, %one, %zero : f64\n";
  // The fields are chosen particle by particle, so that each has storage
  // of its own.
  for (StringRef field : {"x", "f"})
    os << "        %" << field << "b = md.map_particles gather(%" << field
       << "t, %" << field << "a2 : !vec, !vec) {\n"
       << "        ^bb0(%new_i: vector<3xf64>, %old_i: vector<3xf64>):\n"
       << "          %half = arith.constant 5.0e-01 : f64\n"
       << "          %taken = arith.cmpf ogt, %keep, %half : f64\n"
       << "          %chosen = arith.select %taken, %new_i, %old_i "
          ": vector<3xf64>\n"
       << "          md.yield %chosen : vector<3xf64>\n"
       << "        } : !vec\n";
  os << "        scf.yield %xb, %fb, %ub, %hb : " << state << "\n"
     << "      }\n"
     << "      %before = arith.muli %i0, %per0 : index\n"
     << "      %done = arith.addi %i1, %c1 : index\n"
     << "      %within = arith.muli %done, %n2 : index\n"
     << "      %steps = arith.addi %before, %within : index\n"
     << "      %since = arith.index_cast %steps : index to i64\n"
     << "      %step = arith.addi %since, %start : i64\n";
  reported = emitReported("      ", "%xe2", "%fe2", "1");
  os << "      mdrt.host_call @mdrtWriteMinimization(%step, %ue2, %he2, "
     << reported << ", %id)\n"
     << "          : (i64, f64, f64, !vec, !ids)\n"
     << "      scf.yield %xe2, %fe2, %ue2, %he2 : " << state << "\n"
     << "    }\n";
  if (control.framePeriod > 0)
    os << "    %frames = arith.addi %i0, %c1 : index\n"
       << "    %frame_steps = arith.muli %frames, %per0 : index\n"
       << "    %frame_since = arith.index_cast %frame_steps : index to i64\n"
       << "    %frame_step = arith.addi %frame_since, %start : i64\n"
       << "    mdrt.host_call @mdrtWriteFrame(%frame_step, %xe1, %id) "
          ": (i64, !vec, !ids)\n";
  os << "    scf.yield %xe1, %fe1, %ue1, %he1 : " << state << "\n"
     << "  }\n";
  // The checkpoint holds the positions, and velocities of 0.
  if (control.checkpointPeriod > 0)
    os << "  %c_total = arith.constant " << control.numSteps << " : i64\n"
       << "  %end = arith.addi %start, %c_total : i64\n"
       << "  mdrt.host_call @mdrtWriteCheckpoint(%end, %xe0, %v0, %id)\n"
       << "      : (i64, !vec, !vec, !ids)\n";
  os << "  mdrt.host_call @mdrtFinish(%xe0, %v0, %id) : (!vec, !vec, !ids)\n";
  // The forces at the positions that the segment ends at.
  if (control.segments)
    os << "  mdrt.host_call @mdrtFinishForces(%fe0, %id) : (!vec, !ids)\n";
  os << "  return\n}\n";
}

void Builder::emitTerms(StringRef x) {
  if (!system.topology)
    return;
  // The restraints, if any, last: their energy at the start is that of the
  // evaluation before the terms.
  // The terms given by expressions follow those of the topology, those
  // over tuples (D136) and then those over pairs (D137).
  int custom = static_cast<int>(system.topology->tupleTerms.size());
  int pairs = static_cast<int>(control.pairs.size());
  int born = static_cast<int>(system.bornTermNames.size());
  int external = static_cast<int>(system.topology->externalTerms.size());
  int size =
      14 + custom + pairs + born + external + (hasRestraints() ? 1 : 0);
  std::string type = "memref<" + std::to_string(size) + "xf64>";
  os << "  %terms = memref.alloca() : " << type << "\n";
  int index = 0;
  for (StringRef name :
       {"term_lj", "term_coulomb", "term_bonds", "term_angles",
        "term_dihedrals", "term_lj14", "term_coulomb14", "term_cmap",
        "term_excluded", "term_reciprocal", "term_urey_bradley",
        "term_impropers", "term_lj_excluded", "term_lj_reciprocal"}) {
    os << "  %" << name << " = md.evaluate @" << name << "(" << x << ", %cell"
       << getFieldValues() << getTimeValue("%time0") << ") request [energy]\n"
       << "      : (!vec, !md.cell" << getFieldTypes() << getTimeType()
       << ") -> f64\n"
       << "  %i_" << name << " = arith.constant " << index++
       << " : index\n"
       << "  memref.store %" << name << ", %terms[%i_" << name
       << "] : " << type << "\n";
  }
  for (int k = 0; k != custom + pairs + born + external; ++k) {
    std::string name =
        k < custom ? "term_custom" + std::to_string(k)
        : k < custom + pairs ? "term_pair" + std::to_string(k - custom)
        : k == custom + pairs && born > 0 ? "term_born"
        : k < custom + pairs + born
            ? "term_surface"
            : "term_external" + std::to_string(k - custom - pairs - born);
    os << "  %" << name << " = md.evaluate @" << name << "(" << x << ", %cell"
       << getFieldValues() << getTimeValue("%time0") << ") request [energy]\n"
       << "      : (!vec, !md.cell" << getFieldTypes() << getTimeType()
       << ") -> f64\n"
       << "  %i_" << name << " = arith.constant " << index++ << " : index\n"
       << "  memref.store %" << name << ", %terms[%i_" << name << "] : "
       << type << "\n";
  }
  if (hasRestraints())
    os << "  %i_restraints = arith.constant " << index << " : index\n"
       << "  memref.store %u0_r, %terms[%i_restraints] : " << type << "\n";
  os << "  %terms_cast = memref.cast %terms : " << type << " to "
        "memref<?xf64>\n"
     << "  call @mdrtWriteTerms(%terms_cast) : (memref<?xf64>) -> ()\n";
}

void Builder::emitEntry() {
  StringRef state = getName(program.state);
  StringRef force = getName(program.force);
  StringRef mass = getName(program.mass);
  StringRef parameter = getName(program.parameter);

  os << "func.func private @mdrtWriteEnergies(i64, f64, f64, f64, f64)\n"
     << "    attributes {llvm.emit_c_interface}\n"
     << "func.func private @mdrtWritePull(i64, memref<?xf64>, memref<?xf64>)\n"
     << "    attributes {llvm.emit_c_interface}\n"
     << "func.func private @mdrtWriteFreeEnergy(i64, memref<?xf64>)\n"
     << "    attributes {llvm.emit_c_interface}\n"
     << "func.func private @mdrtWriteObservables(i64, memref<?xf64>)\n"
     << "    attributes {llvm.emit_c_interface}\n"
     << "func.func private @mdrtWriteTerms(memref<?xf64>)\n"
     << "    attributes {llvm.emit_c_interface}\n"
     << "func.func private @mdrtWriteVirial(f64, f64, f64)\n"
     << "    attributes {llvm.emit_c_interface}\n"
     << "func.func private @mdrtWriteFrame(i64, memref<?x3x" << state
     << ">, memref<?xi32>)\n    attributes {llvm.emit_c_interface}\n"
     << "func.func private @mdrtCheckSpread(i64, memref<?x3x" << state
     << ">, memref<?xi32>)\n    attributes {llvm.emit_c_interface}\n"
     << "func.func private @mdrtFinish(memref<?x3x" << state
     << ">, memref<?x3x" << state << ">, memref<?xi32>)\n"
     << "    attributes {llvm.emit_c_interface}\n";
  if (control.segments)
    os << "func.func private @mdrtFinishForces(memref<?x3x" << force
       << ">, memref<?xi32>)\n    attributes {llvm.emit_c_interface}\n";
  if (control.minimize)
    os << "func.func private @mdrtWriteMinimization(i64, f64, f64, memref<?x3x"
       << force << ">, memref<?xi32>)\n"
       << "    attributes {llvm.emit_c_interface}\n";
  if (reportsSolvent())
    os << "func.func private @mdrtWriteSolvent(f64, f64)\n";
  if (rescalesVelocities() && control.isNoseHoover())
    os << "func.func private @mdrtNoseHooverFactor(i64, f64) -> f64\n";
  else if (rescalesVelocities())
    os << "func.func private @mdrtBussiFactor(i64, i64, f64, f64, f64, f64) "
          "-> f64\n";
  if (control.getCouplingPeriod() > 0)
    os << "func.func private @mdrtAddBath(f64)\n"
       << "    attributes {llvm.emit_c_interface}\n";
  if (control.barostat)
    os << "func.func private @mdrtBarostatStrain(i64, i64, f64, f64, f64, f64, "
          "f64, f64) -> f64\n"
       << "func.func private @mdrtBarostatStrainArea(i64, i64, f64, f64, f64, "
          "f64, f64, f64, f64, f64) -> f64\n"
       << "func.func private @mdrtBarostatStrainHeight(i64, i64, f64, f64, "
          "f64, f64, f64, f64) -> f64\n"
       << "func.func private @mdrtBarostatStrainAxis(i64, i64, i64, f64, f64, "
          "f64, f64, f64, f64) -> f64\n"
       << "func.func private @mdrtSetBox(f64, f64, f64)\n"
       << "    attributes {llvm.emit_c_interface}\n"
       << "func.func private @mdrtSetTilt(f64, f64, f64)\n"
       << "    attributes {llvm.emit_c_interface}\n";
  if (scalesEveryStep())
    os << "func.func private @mdrtSetBarostatState(f64, f64, f64, f64, f64, "
          "f64, f64, f64, f64)\n"
       << "    attributes {llvm.emit_c_interface}\n";
  if (control.checkpointPeriod > 0) {
    os << "func.func private @mdrtWriteCheckpoint(i64, memref<?x3x" << state
       << ">, memref<?x3x" << state << ">";
    if (program.writesForces)
      os << ", memref<?x3x" << force << ">";
    os << ", memref<?xi32>)\n    attributes {llvm.emit_c_interface}\n";
  }
  os << "\n";

  os << "func.func @" << program.entry << "(\n"
     << "    %positions: memref<?x3x" << state << ">, %velocities: memref<?x3x"
     << state << ">,\n";
  if (program.takesForces)
    os << "    %forces: memref<?x3x" << force << ">,\n";
  os << "    %masses: memref<?x" << mass << ">";
  for (const Program::Field &field : program.fields)
    os << ", %b_" << field.name << ": memref<?x"
       << (field.isInteger ? StringRef("i32") : parameter) << ">";
  for (const Program::Table &table : program.tables)
    os << ", %bt_" << table.name << ": memref<?x?xf64>";
  for (const Program::TupleSet &set : program.tupleSets) {
    os << ",\n    %br_" << set.name << ": memref<?x" << set.arity << "xi32>";
    for (const Program::Field &field : set.fields)
      os << ", %bf_" << set.name << "_" << field.name << ": memref<?xf64>";
  }
  os << ",\n    %identities: memref<?xi32>,\n"
     << "    %lx: f64, %ly: f64, %lz: f64, %dt: f64, %start: i64";
  // The counts of the loops of a program of segments (D196):
  // the iterations of the outer loop, the plain steps in an iteration of the
  // loop over the periods of coupling, and plain steps after the loops; then
  // whether the segment ends with a plain step of energy, and whether it
  // ends with a period that closes with a step of energy, with its plain
  // steps.
  if (control.segments)
    os << ",\n    %count_outer: i64, %count_inner: i64, %count_tail: i64,"
       << "\n    %count_energy_plain: i64, %count_energy_close: i64,"
       << " %count_energy_inner: i64";
  // A minimization in segments takes its steps in %count_outer and the
  // length of its first step, which the last segment left
  // (D202).
  if (control.segments && control.minimize)
    os << ", %first_size: f64";
  os << ") {\n";

  os << "  %c0 = arith.constant 0 : index\n"
     << "  %c1 = arith.constant 1 : index\n";
  // The time of the first step, which the evaluations at the start take
  // (D145); a minimization takes 0.
  if (control.usesTime) {
    if (control.minimize)
      os << "  %time0 = arith.constant 0.0 : f64\n";
    else
      os << "  %time0_steps = arith.sitofp %start : i64 to f64\n"
         << "  %time0 = arith.mulf %time0_steps, %dt : f64\n";
  }
  // The number of the last step taken, which keys the random numbers of
  // Langevin dynamics (D135).
  if (needsStepNumber())
    os << "  %noise_memory = memref.alloca() : memref<1xi64>\n"
       << "  memref.store %start, %noise_memory[%c0] : memref<1xi64>\n"
       << "  %noise_one = arith.constant 1 : i64\n";
  // A minimization has loops of its own (emitMinimization).
  if (control.segments && !control.minimize) {
    // The counts that the entry takes. The first nest is over the periods
    // of coupling, with plain steps inside, or over steps; the second, if
    // any, is one period that closes with a step of energy.
    bool couples = levels[0].name == "couple";
    os << "  %n0 = arith.index_cast %count_outer : i64 to index\n";
    if (couples)
      os << "  %n1 = arith.index_cast %count_inner : i64 to index\n"
         << "  %n2 = arith.index_cast %count_energy_close : i64 to index\n"
         << "  %n3 = arith.constant 0 : index\n"
         << "  %n4 = arith.index_cast %count_energy_inner : i64 to index\n";
    os << "  %n_tail = arith.index_cast %count_tail : i64 to index\n"
       << "  %n_plain = arith.index_cast %count_energy_plain : i64 to index\n";
  } else {
    for (auto [index, level] : llvm::enumerate(levels))
      os << "  %n" << index << " = arith.constant " << level.count
         << " : index\n";
  }
  // The number of steps in one iteration of each loop. An iteration of
  // the loop over energy intervals takes one step after its loop over
  // steps.
  // An iteration of the loop over the periods of coupling takes one as
  // well (two with the barostat of Trotter type), and one over energy intervals a whole period after its loop over
  // periods.
  int64_t steps = 1;
  if (control.minimize) {
    // None: emitMinimization counts its steps.
  } else if (control.segments && levels[0].name == "couple") {
    // An iteration over a period takes its plain steps and those that close
    // it: one, or two with the barostat of Trotter type. The period of the
    // second nest is the whole of its one interval of energy.
    os << "  %per1 = arith.constant 1 : index\n"
       << "  %closing = arith.constant " << getClosingSteps() << " : index\n"
       << "  %per0 = arith.addi %n1, %closing : index\n"
       << "  %per4 = arith.constant 1 : index\n"
       << "  %per3 = arith.addi %n4, %closing : index\n"
       << "  %per2 = arith.addi %n4, %closing : index\n";
  } else if (control.segments) {
    os << "  %per0 = arith.constant 1 : index\n";
  }
  for (unsigned i = control.segments ? 0 : levels.size(); i-- != 0;) {
    os << "  %per" << i << " = arith.constant " << steps << " : index\n";
    steps *= levels[i].count;
    if (i == 0)
      continue;
    // With the barostat of Trotter type, two steps (setSchedule).
    if (levels[i - 1].name == "couple")
      steps += usesTrotter() && !scalesEveryStep() ? 2 : 1;
    else if (levels[i - 1].name == "energy")
      steps += levels[i].name == "couple" ? control.getCouplingPeriod() : 1;
  }

  // The fields as the buffers hold them, and in the order of the positions
  // if the run keeps that order.
  bool givenForces = isRestart() && program.takesForces;
  std::string velocities = isLeapfrog() && !isRestart() ? "%vg" : "%v0";
  std::string given = program.reorders ? "_in" : "";
  if (changesCell()) {
    // Where the barostat keeps the cell, on the host.
    os << "  %box_memory = memref.alloca() : " << getBoxMemoryType() << "\n";
    // With a scaling every step, the diagonals of the virial and of that of
    // the rigid groups, and the kinetic energy of each axis without the
    // center of mass, of the state that the last step left (D92, D119).
    if (scalesEveryStep())
      os << "  %trotter_memory = memref.alloca() : memref<9xf64>\n";
    // A triclinic cell keeps its tilts after its diagonal (I3 of
    // docs/triclinic-m2.md: the barostat scales them with their columns).
    static const char *const initial[] = {"%lx", "%ly", "%lz",
                                          "%tilt_bx", "%tilt_cx", "%tilt_cy"};
    if (isTriclinic())
      os << "  %tilt_bx = arith.constant " << formatReal(system.tilt[0])
         << " : f64\n"
         << "  %tilt_cx = arith.constant " << formatReal(system.tilt[1])
         << " : f64\n"
         << "  %tilt_cy = arith.constant " << formatReal(system.tilt[2])
         << " : f64\n";
    for (int k = 0; k != getCellSize(); ++k)
      os << "  %c_edge" << k << " = arith.constant " << k << " : index\n"
         << "  memref.store " << initial[k] << ", %box_memory[%c_edge" << k
         << "] : " << getBoxMemoryType() << "\n";
  }
  if (scalesReference()) {
    // The edges of the cell of the file, which the reference positions of
    // the restraints are for; they scale with the cell, axis by axis.
    std::string edges[3];
    for (int k = 0; k != 3; ++k) {
      double edge = system.inputBox[k] > 0.0 ? system.inputBox[k]
                                             : system.box[k];
      edges[k] = "%rest_edge" + std::to_string(k);
      os << "  " << edges[k] << " = arith.constant " << formatReal(edge)
         << " : f64\n";
    }
    os << "  %rest_edges = vector.from_elements " << edges[0] << ", "
       << edges[1] << ", " << edges[2] << " : vector<3xf64>\n";
    emitReferenceScale(os, "  ", "%rest_scale", "%lx", "%ly", "%lz");
  }
  if (isTriclinic()) {
    if (!changesCell())
      os << "  %tilt_bx = arith.constant " << formatReal(system.tilt[0])
         << " : f64\n"
         << "  %tilt_cx = arith.constant " << formatReal(system.tilt[1])
         << " : f64\n"
         << "  %tilt_cy = arith.constant " << formatReal(system.tilt[2])
         << " : f64\n";
    os << "  %cell = md.triclinic_cell %lx, %ly, %lz, %tilt_bx, %tilt_cx, "
          "%tilt_cy\n";
  } else
    os << "  %cell = md.orthorhombic_cell %lx, %ly, %lz\n";
  os << "  %x" << (program.reorders ? "_in" : "0") << (hasSites() ? "u" : "")
     << " = mdrt.from_buffer %positions : memref<?x3x" << state
     << "> to !vec\n"
     << "  " << (program.reorders ? "%v_in" : velocities)
     << " = mdrt.from_buffer %velocities : memref<?x3x" << state
     << "> to !vec\n"
     << "  %m" << given << " = mdrt.from_buffer %masses : memref<?x" << mass
     << "> to !real\n";
  for (const Program::Field &field : program.fields)
    os << "  %p" << given << "_" << field.name << " = mdrt.from_buffer %b_"
       << field.name << " : memref<?x"
       << (field.isInteger ? StringRef("i32") : parameter) << "> to "
       << getFieldType(field) << "\n";
  for (const Program::Table &table : program.tables)
    os << "  %t_" << table.name << " = mdrt.from_buffer %bt_" << table.name
       << " : memref<?x?xf64> to " << table.getType() << "\n";
  for (const Program::TupleSet &set : program.tupleSets) {
    // The members in the order of the files, which a new order of the
    // particles renumbers.
    os << "  %r" << (program.reorders ? "o" : "") << "_" << set.name
       << " = mdrt.from_buffer %br_" << set.name << " : memref<?x"
       << set.arity << "xi32> to !rel_" << set.name << "\n";
    for (const Program::Field &field : set.fields)
      os << "  %f_" << set.name << "_" << field.name
         << " = mdrt.from_buffer %bf_" << set.name << "_" << field.name
         << " : memref<?xf64> to !of_" << set.name << "\n";
  }
  os << "  %id" << given
     << " = mdrt.from_buffer %identities : memref<?xi32> to !ids\n";
  // A run that continues an earlier one has the forces of the step before.
  if (givenForces)
    os << "  %f" << (program.reorders ? "_in" : "0")
       << " = mdrt.from_buffer %forces : memref<?x3x" << force
       << "> to !vec\n";
  // The virtual sites where their atoms put them, whatever the file says.
  if (hasSites()) {
    StringRef positions = program.reorders ? "%x_in" : "%x0";
    emitPlaceSites("  ", (positions + "u").str(), positions,
                   program.reorders ? "%ro_" : "%r_");
  }
  if (program.reorders)
    emitReorder("  ", "_in", "0", "", givenForces, velocities);
  if (control.minimize) {
    emitMinimization();
    return;
  }

  if (control.getCouplingPeriod() > 0) {
    // What the coupling of the velocities takes: the total mass; and for
    // the thermostat the key of the random numbers, the degrees of freedom,
    // the mean kinetic energy at the temperature of the bath, and how much
    // of the kinetic energy is kept over one period.
    os << "  %couple_half = arith.constant 5.0e-01 : f64\n"
       << "  %total_mass = md.sum_particles gather(%m" << given
       << " : !real) {\n"
       << "  ^bb0(%m_i: f64):\n"
       << "    %mb = vector.broadcast %m_i : f64 to vector<3xf64>\n"
       << "    md.yield %mb : vector<3xf64>\n"
       << "  } : vector<3xf64>\n";
    if (control.thermostat) {
      // The degrees of freedom of the log. The momentum stays 0 with or
      // without its removal, from velocities that begin with none.
      double freedom = system.getDegreesOfFreedom();
      double target =
          0.5 * freedom * units::boltzmann * control.temperature;
      double decay = std::exp(
          -static_cast<double>(control.getCouplingPeriod()) *
          control.timestep / control.tauT);
      if (control.barostat) {
        // The barostat, in bar: the target pressure, the compressibility,
        // Δt_p / τ_p, k_B T of the bath, and the virial of the corrections
        // times the volume, which is proportional to 1 / V.
        double bar = 1.01325;
        double volume = system.box[0] * system.box[1] * system.box[2];
        double constant = (program.dispersionVirial +
                           program.coulombConstantVirial) *
                          volume;
        // The energies of the same terms times the volume (the self term of
        // particle mesh Ewald does not depend on it), for the exact work.
        double energyConstant =
            (program.dispersionEnergy + program.coulombConstantEnergy -
             program.coulombSelfEnergy) *
            volume;
        for (int k = 0; k != 3; ++k)
          os << "  %baro_beta_" << k << " = arith.constant "
             << formatReal(control.compressibilities[k] / bar) << " : f64\n";
        os << "  %baro_target = arith.constant "
           << formatReal(control.pressure * bar) << " : f64\n"
           << "  %baro_beta = arith.constant "
           << formatReal(control.compressibility / bar) << " : f64\n"
           << "  %baro_rate = arith.constant "
           << formatReal(static_cast<double>(control.barostatPeriod) *
                         control.timestep / control.tauP)
           << " : f64\n"
           << "  %baro_kt = arith.constant "
           << formatReal(units::boltzmann * control.temperature) << " : f64\n"
           << "  %baro_constant = arith.constant " << formatReal(constant)
           << " : f64\n"
           << "  %baro_energy_constant = arith.constant "
           << formatReal(energyConstant) << " : f64\n"
           << "  %c_bar = arith.constant 16.6053906717 : f64\n"
           << "  %c_three = arith.constant 3.0 : f64\n"
           << "  %c_third = arith.constant "
           << formatReal(1.0 / 3.0) << " : f64\n"
           << "  %c_unit = arith.constant 1.0 : f64\n"
           << "  %c_unit3 = arith.constant dense<1.0> : vector<3xf64>\n"
           << "  %c_half3 = arith.constant dense<5.0e-01> : vector<3xf64>\n"
           << "  %baro_beta_z = arith.constant "
           << formatReal(control.compressibilityZ / bar) << " : f64\n"
           << "  %baro_tension = arith.constant "
           << formatReal(control.surfaceTension * control.surfaces *
                         units::dynePerCmToBarNm)
           << " : f64\n";
      }
      os << "  %seed = arith.constant " << static_cast<int64_t>(control.seed)
         << " : i64\n"
         << "  %freedom = arith.constant " << formatReal(freedom) << " : f64\n"
         << "  %target_kinetic = arith.constant " << formatReal(target)
         << " : f64\n"
         << "  %decay = arith.constant " << formatReal(decay) << " : f64\n";
    }
  }

  if (isLeapfrog() && control.barostat)
    os << "  %c_half_back = arith.constant -5.0e-01 : f64\n"
       << "  %half_back = arith.mulf %c_half_back, %dt : f64\n";

  // The energy, the forces, and the virial of the state at the start, as
  // %u0, %f0, and %w0; returns the name of the virial.
  auto emitStartForces = [&]() -> std::string {
    StringRef raw = hasSites() ? "e" : "";
    std::string held = hasRestraints() ? "p" : "";
    os << "  %u0" << held << ", %f0" << held << raw << ", %w0" << held << raw
       << " = md.evaluate @energy(%x0, %cell" << getFieldValues()
       << getTimeValue("%time0") << ")\n"
       << "      request [energy, forces, virial]\n"
       << "      : (!vec, !md.cell" << getFieldTypes() << getTimeType()
       << ") -> (f64, !vec, vector<9xf64>)\n";
    std::string virial = "%w0" + held;
    if (hasSites())
      virial = emitSpreadSites("  ", "%x0", "%f0" + held + "e", "%f0" + held,
                               "%r_", "%w0" + held + "e", "%w0" + held);
    if (hasRestraints()) {
      emitRestraints("  ", "%x0", "%p_", "%f0p", "%f0", "%u0p", "%u0",
                     virial, "%w0");
      virial = "%w0";
    }
    return virial;
  };

  // A run that begins from the checkpoint of a run of other physics
  // evaluates the forces of its first step rather than take those of the
  // checkpoint (D172).
  if (recomputes())
    emitStartForces();

  if (!isRestart()) {
    // The energies at the start.
    std::string virial = emitStartForces();
    emitTerms();
    // The diagonal of the virial, which semi-isotropic coupling takes axis
    // by axis, for comparison with other programs (D119).
    if (system.topology) {
      emitDiagonal(os, "%dgw0", virial, "  ");
      os << "  call @mdrtWriteVirial(%dgw0_0, %dgw0_4, %dgw0_8)"
            " : (f64, f64, f64) -> ()\n";
    }
    emitKineticEnergy(os, "%k0", velocities, "%m", "  ");
    // K_half - K from the forces without constraints; with them there is
    // no step before the start to measure, and the row takes K
    // (D203).
    std::string startExcess =
        emitKineticExcess("  ", "s0", "", "%f0", velocities);
    if (startExcess.empty())
      os << "  %g0 = arith.constant 0.0 : f64\n";
    else
      emitSum3(os, "%g0", startExcess, "  ");
    emitTrace(os, "%tr0", virial, "  ");
    std::string trace = "%tr0";
    if (hasConstraints())
      trace = emitStartConstraintTrace("%x0", "%f0", velocities, trace);
    os << "  call @mdrtWriteEnergies(%start, %u0, %k0, %g0, " << trace
       << ")\n"
       << "      : (i64, f64, f64, f64, f64) -> ()\n";
    emitPullOutput("  ", "%x0", "%cell", "%p_", "%start", "%time0");
    emitFreeEnergyOutput("  ", "%x0", "%cell", "%p_", "%start", "%time0");
    emitObservablesOutput("  ", "%x0", "%cell", "%p_", "%start", "%time0");
    // The state that the first scaling takes its pressure from (D92), by
    // axes.
    if (scalesEveryStep()) {
      emitDiagonal(os, "%dg0", virial, "  ");
      std::string diagonal = "%dg0";
      if (hasConstraints())
        diagonal = emitStartConstraintTrace("%x0", "%f0", velocities,
                                            diagonal, /*axes=*/true);
      emitStoreTrotterState(
          "  ", diagonal,
          emitMolecularDiagonal("  ", "%dg0", "%x0", "%f0", "%cell", "%m",
                                "%r_", "s0"),
          emitKineticWithoutCenter("  ", velocities, "%m", "s0"));
    }

    if (isLeapfrog()) {
      // v(-dt/2) = v(0) - (dt/2) F(0) / m.
      os << "  %back = arith.constant -5.0e-01 : f64\n"
         << "  %behind = arith.mulf %back, %dt : f64\n"
         << "  %v0 = dyn.kick %vg, %f0, %m, %behind : !vec\n";
    }
  }

  if (isRestart() && scalesEveryStep() &&
      system.barostatState.size() == 9) {
    // The state that the first scaling takes its pressure from, as the
    // checkpoint keeps it (D92).
    std::string vectors[3];
    for (int v = 0; v != 3; ++v) {
      std::string names[3];
      for (int k = 0; k != 3; ++k) {
        names[k] = "%bstate" + std::to_string(3 * v + k);
        os << "  " << names[k] << " = arith.constant "
           << formatReal(system.barostatState[3 * v + k]) << " : f64\n";
      }
      vectors[v] = "%bstatev" + std::to_string(v);
      os << "  " << vectors[v] << " = vector.from_elements " << names[0]
         << ", " << names[1] << ", " << names[2] << " : vector<3xf64>\n";
    }
    emitStoreTrotterState("  ", vectors[0], vectors[1], vectors[2]);
  } else if (isRestart() && scalesEveryStep()) {
    // A checkpoint of a run that did not scale every step holds no virial:
    // the state that the first scaling takes its pressure from is evaluated
    // once (D92). With leapfrog the stored velocities are half a kick
    // behind those of the time of the positions.
    StringRef raw = hasSites() ? "e" : "";
    std::string held = hasRestraints() ? "p" : "";
    os << "  %ubs" << held << ", %fbs" << held << raw << ", %wbs" << held << raw
       << " = md.evaluate @energy(%x0, %cell" << getFieldValues()
       << getTimeValue("%time0") << ")\n"
       << "      request [energy, forces, virial]\n"
       << "      : (!vec, !md.cell" << getFieldTypes() << getTimeType()
       << ") -> (f64, !vec, vector<9xf64>)\n";
    std::string virial = "%wbs" + held;
    if (hasSites())
      virial = emitSpreadSites("  ", "%x0", "%fbs" + held + "e", "%fbs" + held,
                               "%r_", "%wbs" + held + "e", "%wbs" + held);
    if (hasRestraints()) {
      emitRestraints("  ", "%x0", "%p_", "%fbsp", "%fbs", "%ubsp", "%ubs",
                     virial, "%wbs");
      virial = "%wbs";
    }
    std::string current = velocities;
    if (isLeapfrog()) {
      os << "  %ahead = arith.constant 5.0e-01 : f64\n"
         << "  %ahead_dt = arith.mulf %ahead, %dt : f64\n"
         << "  %vs_now = dyn.kick " << velocities
         << ", %fbs, %m, %ahead_dt : !vec\n";
      current = "%vs_now";
    }
    emitDiagonal(os, "%dgs", virial, "  ");
    std::string diagonal = "%dgs";
    if (hasConstraints())
      diagonal = emitStartConstraintTrace("%x0", "%fbs", current, diagonal,
                                          /*axes=*/true);
    emitStoreTrotterState(
        "  ", diagonal,
        emitMolecularDiagonal("  ", "%dgs", "%x0", "%fbs", "%cell", "%m",
                              "%r_", "s0"),
        emitKineticWithoutCenter("  ", current, "%m", "s0"));
  }

  std::string last = "e0";
  if (control.segments) {
    nestEnd = levels[0].name == "couple" ? 2 : 1;
    emitLevel(0, "  ");
    // The plain steps of the period in which the segment ends, after the
    // loops (D196); the next segment closes the period. The
    // barostat may have changed the cell in the loops.
    std::string outerCell = cellName, outerScale = scaleName;
    if (changesCell()) {
      cellName = "%cell_tail";
      for (int k = 0; k != getCellSize(); ++k)
        os << "  %edge_tail_" << k << " = memref.load %box_memory[%c_edge"
           << k << "] : " << getBoxMemoryType() << "\n";
      emitCellOf("  ", cellName, "%edge_tail");
      if (scalesReference()) {
        scaleName = "%rest_scale_tail";
        emitReferenceScale(os, "  ", scaleName, "%edge_tail_0",
                           "%edge_tail_1", "%edge_tail_2");
      }
    }
    os << "  %xt, %vt, %ft = scf.for %i_tail = %c0 to %n_tail step %c1\n"
       << "      iter_args(%xta = %xe0, %vta = %ve0, %fta = %fe0)\n"
       << "      -> (!vec, !vec, !vec) {\n";
    std::string noise = emitNoiseValue("    ");
    os << "    %xtb, %vtb, %ftb = dyn.step @step(%xta, %vta, %fta, "
       << massName << ", " << cellName << ", %dt" << getScaleValue()
       << getFieldValues(fieldPrefix) << noise << ")\n"
       << "        : (!vec, !vec, !vec, !real, !md.cell, f64"
       << getScaleType() << getFieldTypes() << getNoiseType()
       << ") -> (!vec, !vec, !vec)\n"
       << "    scf.yield %xtb, %vtb, %ftb : !vec, !vec, !vec\n"
       << "  }\n";
    last = emitSegmentEnergyStep("%xt", "%vt", "%ft");
    cellName = outerCell;
    scaleName = outerScale;
    // A period that closes with a step of energy, as an interval of the log
    // of `mdir run` ends, after the steps of the segment before it.
    if (levels[0].name == "couple") {
      os << "  %steps_a = arith.muli %n0, %per0 : index\n"
         << "  %steps_t = arith.addi %steps_a, %n_tail : index\n"
         << "  %steps_p = arith.addi %steps_t, %n_plain : index\n"
         << "  %steps_e = arith.index_cast %steps_p : index to i64\n"
         << "  %start_e = arith.addi %start, %steps_e : i64\n";
      nestBegin = 2;
      nestEnd = 5;
      nestOutside = last;
      nestStart = "%start_e";
      emitLevel(2, "  ");
      last = "e2";
    }
  } else {
    emitLevel(0, "  ");
  }
  os << "  mdrt.host_call @mdrtFinish(%x" << last << ", %v" << last << ", "
     << idName << ") : (!vec, !vec, !ids)\n";
  // The forces that the next segment begins with.
  if (control.segments)
    os << "  mdrt.host_call @mdrtFinishForces(%f" << last << ", " << idName
       << ") : (!vec, !ids)\n";
  os << "  return\n}\n";
}

std::string Builder::emitSegmentEnergyStep(StringRef x, StringRef v,
                                           StringRef f) {
  // At most one plain step, of energy, as `mdir run` takes at a row of its
  // log without coupling: its energies go to mdrtWriteEnergies.
  os << "  %steps_pa = arith.muli %n0, %per0 : index\n"
     << "  %steps_pt = arith.addi %steps_pa, %n_tail : index\n"
     << "  %steps_pe = arith.addi %steps_pt, %c1 : index\n"
     << "  %steps_pi = arith.index_cast %steps_pe : index to i64\n"
     << "  %step_plain = arith.addi %start, %steps_pi : i64\n";
  os << "  %xp, %vp, %fp = scf.for %i_plain = %c0 to %n_plain step %c1\n"
     << "      iter_args(%xpa = " << x << ", %vpa = " << v << ", %fpa = "
     << f << ")\n"
     << "      -> (!vec, !vec, !vec) {\n";
  std::string noise = emitNoiseValue("    ");
  os << "    %xpl, %vpl, %fpl, %upl, %wpl" << (isLeapfrog() ? ", %vpn" : "")
     << (measuresHalfSteps() ? ", %khspl" : "")
     << (reportsSolvent() ? ", %kwspl" : "")
     << " = dyn.step @step_energy(%xpa, %vpa, %fpa, " << massName << ", "
     << cellName << ", %dt" << getScaleValue() << getFieldValues(fieldPrefix)
     << noise << ")\n"
     << "        : (!vec, !vec, !vec, !real, !md.cell, f64" << getScaleType()
     << getFieldTypes() << getNoiseType()
     << ") -> (!vec, !vec, !vec, f64, vector<9xf64>"
     << (isLeapfrog() ? ", !vec" : "")
     << (measuresHalfSteps() ? ", vector<3xf64>" : "")
     << (reportsSolvent() ? ", vector<3xf64>" : "") << ")\n";
  if (reportsSolvent())
    emitWriteSolvent("    ", "%kwspl");
  std::string plainFull = isLeapfrog() ? "%vpn" : "%vpl";
  emitKineticEnergy(os, "%kpl", plainFull, massName, "    ");
  std::string plainExcess = emitKineticExcess(
      "    ", "pl", measuresHalfSteps() ? "%khspl" : "", "%fpl", plainFull);
  if (plainExcess.empty())
    os << "    %gpl = arith.constant 0.0 : f64\n";
  else
    emitSum3(os, "%gpl", plainExcess, "    ");
  emitTrace(os, "%trpl", "%wpl", "    ");
  os << "    func.call @mdrtWriteEnergies(%step_plain, %upl, %kpl, %gpl, "
        "%trpl) : (i64, f64, f64, f64, f64) -> ()\n";
  if (!control.periodic)
    os << "    mdrt.host_call @mdrtCheckSpread(%step_plain, %xpl, " << idName
       << ") : (i64, !vec, !ids)\n";
  os << "    scf.yield %xpl, %vpl, %fpl : !vec, !vec, !vec\n"
     << "  }\n";
  return "p";
}

/// The largest number of particles within `reach` of a particle, found
/// with cells of the width of the reach.
static int64_t countMostNeighbors(const System &system, double reach) {
  size_t count = system.getNumParticles();
  int cells[3];
  for (int k = 0; k != 3; ++k)
    cells[k] = std::max(1, static_cast<int>(system.box[k] / reach));
  auto wrap = [&](double x, int k) {
    double length = system.box[k];
    return x - length * std::floor(x / length);
  };
  auto cellOf = [&](size_t i, int k) {
    int c = static_cast<int>(wrap(system.positions[3 * i + k], k) /
                             system.box[k] * cells[k]);
    return std::min(c, cells[k] - 1);
  };
  std::vector<std::vector<size_t>> members(cells[0] * cells[1] * cells[2]);
  for (size_t i = 0; i != count; ++i)
    members[(cellOf(i, 0) * cells[1] + cellOf(i, 1)) * cells[2] +
            cellOf(i, 2)]
        .push_back(i);

  double reach2 = reach * reach;
  int64_t most = 0;
  for (size_t i = 0; i != count; ++i) {
    int64_t found = 0;
    int home[3] = {cellOf(i, 0), cellOf(i, 1), cellOf(i, 2)};
    // Each cell once, where the cells on the two sides coincide.
    std::vector<int> seen;
    for (int dx = -1; dx <= 1; ++dx)
      for (int dy = -1; dy <= 1; ++dy)
        for (int dz = -1; dz <= 1; ++dz) {
          int c[3] = {home[0] + dx, home[1] + dy, home[2] + dz};
          for (int k = 0; k != 3; ++k)
            c[k] = (c[k] % cells[k] + cells[k]) % cells[k];
          int cell = (c[0] * cells[1] + c[1]) * cells[2] + c[2];
          if (llvm::is_contained(seen, cell))
            continue;
          seen.push_back(cell);
          for (size_t j : members[cell]) {
            if (j == i)
              continue;
            double r2 = 0.0;
            for (int k = 0; k != 3; ++k) {
              double d = system.positions[3 * i + k] -
                         system.positions[3 * j + k];
              d -= system.box[k] * std::round(d / system.box[k]);
              r2 += d * d;
            }
            found += r2 < reach2;
          }
        }
    most = std::max(most, found);
  }
  return most;
}

void Builder::setSchedule() {
  // A period of zero means no output of that kind, and no loop for it.
  int64_t steps = control.numSteps;
  if (control.checkpointPeriod > 0) {
    levels.push_back({"segment", steps / control.checkpointPeriod});
    steps = control.checkpointPeriod;
  }
  // Frames more frequent than energies are written at the end of each
  // interval between energies, which then has their period.
  int64_t energyPeriod = control.getEnergyLoopPeriod();
  framesAtEnergies = energyPeriod != control.energyPeriod;
  if (control.framePeriod > 0 && !framesAtEnergies) {
    levels.push_back({"frame", steps / control.framePeriod});
    steps = control.framePeriod;
  }
  // The velocities are coupled at the end of the last step of a period of
  // coupling, which is taken after the loop over steps, as the last step
  // of an interval between energies is. The loop over the periods of an
  // interval between energies leaves the last period to the interval,
  // which ends it with its step of energy.
  // With the barostat of Trotter type the last two steps of a period are
  // taken after its loop: the step whose pressure gives the strain, and
  // the step that scales the cell within its drift (D92).
  int64_t coupling = control.getCouplingPeriod();
  int64_t taken = usesTrotter() && !scalesEveryStep() ? 2 : 1;
  if (energyPeriod > 0) {
    levels.push_back({"energy", steps / energyPeriod});
    steps = energyPeriod;
    if (coupling > 0) {
      levels.push_back({"couple", steps / coupling - 1});
      levels.push_back({"step", coupling - taken});
    } else {
      levels.push_back({"step", steps - 1});
    }
  } else if (coupling > 0) {
    levels.push_back({"couple", steps / coupling});
    levels.push_back({"step", coupling - taken});
  } else {
    levels.push_back({"step", steps});
  }
}

llvm::Error Builder::build() {
  // The neighbor structures of a triclinic cell hold every pair within
  // their reach while the reach is at most half of the least of a_x, b_y,
  // c_z, the bound of the minimum image in one pass (docs/triclinic-m2.md).
  if (isTriclinic()) {
    double least = std::min({system.box[0], system.box[1], system.box[2]});
    double reach = control.pairlistDistance * units::length;
    if (reach > 0.5 * least)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "'pairlist_distance', %g Å, exceeds half of the least of a_x, b_y, "
          "c_z of the triclinic cell, %g Å",
          control.pairlistDistance, 0.5 * least / units::length);
  }
  // A pair is taken once, in the minimum image, which holds every image
  // within the cutoff only while the cell is wider than twice the cutoff:
  // for a triclinic cell, its diagonal (I2 of docs/triclinic-m2.md).
  static const char axes[] = "xyz";
  for (int k = 0; k != 3; ++k)
    if (system.box[k] < 2.0 * control.cutoffDistance * units::length)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "the cell is %.4f Å along %c, less than twice the cutoff, %.4f Å",
          system.box[k] / units::length, axes[k],
          control.cutoffDistance);
  program.entry = "mdir_run";
  switch (control.precision) {
  case Precision::Single:
    program.state = program.mass = Element::F32;
    program.force = program.parameter = Element::F32;
    break;
  case Precision::Mixed:
    program.state = program.mass = Element::F64;
    program.force = program.parameter = Element::F32;
    // In the deterministic mode the forces that a step carries to the next
    // are stored as the state is (#102). Buffers of forces in f32 at the
    // host calls would make the forces that a loop of steps carries f32
    // where the loop's result reaches such a call (a run whose every step
    // writes energies) and f64 where it does not (md-exec-assign-precision
    // gives fields stored together the type of a buffer that holds one),
    // and the trajectory would depend on the schedule of the energies.
    if (control.deterministic)
      program.force = Element::F64;
    break;
  case Precision::Double:
    program.state = program.mass = Element::F64;
    program.force = program.parameter = Element::F64;
    break;
  }
  // Velocity Verlet begins a step with the forces of the step before.
  program.writesForces = !control.minimize;
  program.takesForces = isRestart() && !recomputes() && !control.minimize;

  program.skin =
      (control.pairlistDistance - control.cutoffDistance) * units::length;
  if (control.prunedDistance != 0.0)
    program.pruneSkin =
        (control.prunedDistance - control.cutoffDistance) * units::length;
  // Cells of half the reach of a neighbor structure: particles that are
  // neighbors are then a few cells apart in memory.
  program.reorders = control.reorder && !control.minimize;
  program.orderWidth = 0.5 * control.pairlistDistance * units::length;
  program.neighborWidth = control.neighborWidth;
  if (program.neighborWidth == 0) {
    // Half as many again as the particle with the most neighbors has at
    // the start, and a few more. A uniform density would underestimate a
    // box with room in it, such as a solvated molecule that tleap makes.
    double reach = control.pairlistDistance * units::length;
    int64_t most = countMostNeighbors(system, reach);
    int64_t width = static_cast<int64_t>(std::ceil(1.5 * most)) + 16;
    program.neighborWidth = (width + 7) / 8 * 8;
  }

  // A program of segments has no outputs of its own: its loops are those
  // of the periods of coupling and of the steps (D196), or of the steps of
  // a minimization (D202).
  if (control.segments && (control.energyPeriod > 0 ||
                           control.framePeriod > 0 ||
                           control.checkpointPeriod > 0))
    return makeError("a program of segments takes no energies, frames, or "
                     "checkpoints");

  // The loops of the schedule; a minimization has its own
  // (emitMinimization).
  if (!control.minimize)
    setSchedule();
  if (control.segments && !control.minimize) {
    bool couples = levels[0].name == "couple";
    program.segmentPeriod = couples ? control.getCouplingPeriod() : 0;
    program.closingSteps = getClosingSteps();
    // The second nest: one interval of energy of one period, which its
    // step of energy closes, as `mdir run` ends an interval of its log.
    if (couples) {
      levels.push_back({"energy", 1});
      levels.push_back({"couple", 0});
      levels.push_back({"step", 0});
    }
  }

  if (system.topology) {
    if (llvm::Error error = collectTopology())
      return error;
  } else if (llvm::Error error = collectParameters()) {
    return error;
  }
  // The polynomials of the tabulated functions, a row of 4ⁿ numbers for
  // each cell of a function of n arguments (D138, D165), or the values of a
  // discrete one.
  for (const TabulatedFunction &function : control.functions) {
    Program::Table table;
    table.name = function.getTableName();
    if (llvm::any_of(program.tables, [&](const Program::Table &other) {
          return other.name == table.name;
        }))
      return makeError("the table of the function '" + function.name +
                       "' takes the name of another; rename the function");
    table.values = function.getCoefficients();
    table.columns = function.getColumns();
    table.count = table.values.size() / table.columns;
    program.tables.push_back(std::move(table));
  }

  os << "!vec   = !md.field<@atoms, 3 x f64>\n"
     << "!real  = !md.field<@atoms, f64>\n"
     << "!ids   = !md.field<@atoms, i32>\n"
     << "!table = !md.table<2, f64, symmetric>\n"
     << "!grid = !md.table<2, f64>\n"
     << "!pairs = !md.relation<@atoms, 2, unordered>\n";
  for (const Program::TupleSet &set : program.tupleSets) {
    os << "!rel_" << set.name << " = !md.relation<@atoms, " << set.arity
       << ", " << set.getOrientation() << ", @" << set.name << ">\n"
       << "!of_" << set.name << " = !md.field<@" << set.name << ", f64>\n";
  }
  os << "\nmd.particle_set @atoms\n";
  for (const Program::TupleSet &set : program.tupleSets)
    os << "md.tuple_set @" << set.name << " on(@atoms) arity(" << set.arity
       << ") orientation(" << set.getOrientation() << ")"
       << (set.arity > 1 && set.isDisjoint() ? " disjoint" : "") << "\n";
  // The groups of the constraints share no atom with one another either
  // (D83): the bonds of a rigid water are gone before SHAKE groups the
  // bonds of hydrogen, and a hydrogen is in one group only (findShakes).
  // What the groups promise is checked here all the same, and without the
  // promise the loops keep their order.
  std::vector<const Program::TupleSet *> groups;
  for (const Program::TupleSet &set : program.tupleSets)
    if (set.name == "settles")
      groups.push_back(&set);
  for (const Program::TupleSet *set : getShakeSets())
    groups.push_back(set);
  std::vector<int32_t> members;
  for (const Program::TupleSet *set : groups)
    members.insert(members.end(), set->members.begin(), set->members.end());
  std::sort(members.begin(), members.end());
  if (groups.size() >= 2 &&
      std::adjacent_find(members.begin(), members.end()) == members.end()) {
    os << "md.disjoint_union @constraints on(@atoms) of [";
    llvm::interleaveComma(groups, os, [&](const Program::TupleSet *set) {
      os << "@" << set->name;
    });
    os << "]\n";
  }
  os << "\n";
  if (system.topology) {
    emitTopologyPotential("energy", AllTerms);
    // Each term on its own, for the log at the start.
    if (!isRestart())
      for (auto [name, term] :
           {std::pair<StringRef, unsigned>{"term_lj", LennardJones},
            {"term_coulomb", Coulomb},
            {"term_bonds", Bonds},
            {"term_angles", Angles},
            {"term_dihedrals", Dihedrals},
            {"term_lj14", LennardJones14},
            {"term_coulomb14", Coulomb14},
            {"term_cmap", CMaps},
            {"term_excluded", CoulombExcluded | CoulombWithin},
            {"term_reciprocal", CoulombReciprocal},
            {"term_urey_bradley", UreyBradleys},
            {"term_impropers", HarmonicImpropers},
            {"term_lj_excluded", LennardJonesExcluded},
            {"term_lj_reciprocal", LennardJonesReciprocal}})
        emitTopologyPotential(name, term);
    if (!isRestart())
      for (size_t k = 0, e = system.topology->tupleTerms.size(); k != e; ++k)
        emitTopologyPotential("term_custom" + std::to_string(k), TupleTerms,
                              static_cast<int>(k));
    if (!isRestart())
      for (size_t k = 0, e = control.pairs.size(); k != e; ++k)
        emitTopologyPotential("term_pair" + std::to_string(k), PairTerms, -1,
                              static_cast<int>(k));
    if (!isRestart() && !system.bornTermNames.empty()) {
      emitTopologyPotential("term_born", GeneralizedBorn);
      if (system.bornTermNames.size() > 1)
        emitTopologyPotential("term_surface", Surface);
    }
    if (!isRestart())
      for (size_t k = 0, e = system.topology->externalTerms.size(); k != e;
           ++k)
        emitTopologyPotential("term_external" + std::to_string(k),
                              ExternalTerms, -1, -1, static_cast<int>(k));
  }
  else if (llvm::Error error = emitPotential())
    return error;
  // What depends on λ, as a function of it, for dH/dλ and the energies of
  // the other states (D161).
  if (system.topology && !control.freeEnergyFile.empty()) {
    lambdaArguments = true;
    emitTopologyPotential("alchemical", AllTerms | Alchemical);
    lambdaArguments = false;
  }
  emitObservedPotentials();
  emitPullPotentials();
  emitPrograms();
  emitEntry();
  return llvm::Error::success();
}

llvm::Expected<Program> mdir::driver::buildProgram(const Control &control,
                                                   const System &system) {
  Program program;
  Builder builder(control, system, program);
  if (llvm::Error error = builder.build())
    return std::move(error);
  return std::move(program);
}
