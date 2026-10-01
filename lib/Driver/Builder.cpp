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
    AllTerms = 1023,
  };
  /// Emits the potential `name` of the terms `terms` of the topology.
  void emitTopologyPotential(StringRef name, unsigned terms);
  /// β, the grid, the influence function, and the constant terms of
  /// particle mesh Ewald (docs/pme-m1.md).
  llvm::Error collectPME();
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
                                       StringRef v, StringRef trace);
  /// Sets `levels`, the loops of the schedule of a run of dynamics.
  void setSchedule();
  void emitMinimization();
  /// Emits the energy of each term at the positions `%x0`, for the log.
  void emitTerms();

  /// Emits the loops of the schedule, from `level` inward, and returns the
  /// values that the loop of `level` results in.
  void emitLevel(unsigned level, StringRef indent);
  /// Whether the cell changes in the run, which a barostat does.
  bool changesCell() const { return control.barostat; }
  /// Whether particles are restrained to reference positions.
  bool hasRestraints() const { return !system.restraintConstants.empty(); }
  /// Whether the reference positions of the restraints follow the cell,
  /// which a barostat changes: they are those of the file times
  /// `scaleName`, the edge of the cell over that of the file.
  bool scalesReference() const { return hasRestraints() && changesCell(); }
  std::string getScaleValue() const {
    return scalesReference() ? ", " + scaleName : "";
  }
  std::string getScaleType() const {
    return scalesReference() ? ", f64" : "";
  }
  std::string getScaleParameter() const {
    return scalesReference() ? ", %rest_scale: f64" : "";
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
    return control.barostat && control.barostatWork == BarostatWork::Trotter;
  }
  /// Whether every step scales the cell (a period of one step): the
  /// pressure is then that of the state that the step before left, which
  /// the run keeps in `%trotter_memory` (D92).
  bool scalesEveryStep() const {
    return usesTrotter() && control.barostatPeriod == 1;
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
  /// Emits `result`, the positions `x` with the bonds of the groups of
  /// `set` brought back to their lengths along the bonds of `old`, by the
  /// iterations of SHAKE [Ryckaert1977].
  void emitShakePositions(StringRef indent, StringRef old, StringRef x,
                          const Program::TupleSet &set, StringRef result);
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
/// the virial before it, with those of the constant terms and of the rigid
/// groups; the volume after it, and its cell; and the kinetic energy of
/// the velocities that it scaled.
struct TrotterScaling {
  std::string mu, muinv, logMu, workBefore, newVolume, cell, scale,
      kineticHalf;
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
  /// their centers of mass.
  std::string emitGroupTrace(StringRef indent, StringRef trace,
                             StringRef positions, StringRef velocities,
                             StringRef cell, StringRef masses,
                             StringRef relations, StringRef tag);
  /// `positions` scaled by `mu`, each rigid group with its center of mass,
  /// keeping its shape (`oneMinusMu` is 1 − mu).
  std::string emitGroupScaling(StringRef indent, StringRef positions,
                               StringRef mu, StringRef oneMinusMu,
                               StringRef cell, StringRef masses,
                               StringRef relations, StringRef tag);
  Coupled emitCoupling(StringRef indent, StringRef positions,
                       StringRef velocities, StringRef forces,
                       StringRef energy, StringRef tag, StringRef step,
                       StringRef trace,
                       const TrotterScaling *trotter = nullptr);
  /// The pressure of the virial trace `trace` and the kinetic energy
  /// `kinetic`, and the strain that the barostat takes from it: the scale
  /// `%mu<tag>` of the positions, its inverse `%muinv<tag>`, its logarithm
  /// `%bs3<tag>`, the volume before `%bv<tag>`, and the new cell, edges
  /// `%bn<tag>_k`, which it stores where the loops take the cell from.
  void emitStrain(StringRef indent, StringRef kinetic, StringRef trace,
                  StringRef tag, StringRef step);
  /// Before the step of a period of coupling that scales the cell within
  /// its drift (the Trotter type of D92): the strain from the pressure of
  /// the positions `positions` and the velocities `velocities` of the step
  /// before, whose virial has the trace `trace`.
  /// `givenKinetic` and `givenGroups`, if given, are the kinetic energy
  /// without the center of mass and the trace of the virial of the groups
  /// (emitGroupTrace), computed before.
  TrotterScaling emitTrotterStrain(StringRef indent, StringRef positions,
                                   StringRef velocities, StringRef trace,
                                   StringRef tag, StringRef step,
                                   StringRef givenKinetic = "",
                                   StringRef givenGroups = "");
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
  bool isRestart() const { return !control.restartInput.empty(); }

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
  os << indent << "%order" << stateTo << " = md_exec.spatial_order %x"
     << from << ", " << cellName << ", %id" << from << " width("
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
    auto expression = Expression::parse(term.expression);
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
  return computeDispersion();
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
  (void)count;
  // The restraints: the constant of each particle, 0 for one that is not
  // restrained, and its reference position, in nm.
  if (hasRestraints()) {
    Program::Field constants;
    constants.name = "rest_k";
    constants.values = system.restraintConstants;
    program.fields.push_back(std::move(constants));
    for (int c = 0; c != 3; ++c) {
      Program::Field reference;
      reference.name = std::string("rest_") + "xyz"[c];
      for (size_t i = 0, e = system.getNumParticles(); i != e; ++i)
        reference.values.push_back(system.referencePositions[3 * i + c]);
      program.fields.push_back(std::move(reference));
    }
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

  if (control.pme)
    if (llvm::Error error = collectPME())
      return error;

  // The correction for the dispersion (Section 7.2 of design-m1.md): N²
  // times the mean of C6 over the pairs of distinct particles that are not
  // excluded, as GROMACS takes it.
  if (control.topologyDispersion == DispersionCorrection::None)
    return llvm::Error::success();
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
  double n = static_cast<double>(topology.getNumParticles());
  double pairs = n * (n - 1.0) - 2.0 * topology.exclusions.size();
  double mean = pairs > 0.0 ? sum / pairs : 0.0;
  double rc = control.cutoffDistance * units::length;
  double volume = system.box[0] * system.box[1] * system.box[2];
  double energy =
      -2.0 * M_PI / (3.0 * volume) * n * n * mean / (rc * rc * rc);
  program.dispersionEnergy = energy;
  program.dispersionVirial = 6.0 * energy;
  return llvm::Error::success();
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

  // The grid: given, or no wider than the largest spacing in the cell of
  // the file of coordinates. The grid stays as the barostat changes the
  // cell, finer as the cell shrinks and coarser as it grows.
  int64_t grid[3];
  for (int k = 0; k != 3; ++k) {
    grid[k] = control.pmeGrid[k];
    double edge = system.inputBox[k] > 0.0 ? system.inputBox[k] : system.box[k];
    if (grid[k] == 0)
      grid[k] = getSmoothSize(static_cast<int64_t>(std::ceil(
          edge / (control.pmeMaxSpacing * units::length) - 1e-9)));
    if (grid[k] < 2 * control.pmeOrder)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "the grid of particle mesh Ewald has %lld points along an edge, "
          "fewer than twice the order %lld",
          static_cast<long long>(grid[k]),
          static_cast<long long>(control.pmeOrder));
  }

  // The factors of the influence function along each edge, which do not
  // depend on the cell: |b(k)|² of the B-splines, times the factor of the
  // aliasing if it is taken. The program computes the rest from the cell.
  int64_t longest = std::max({grid[0], grid[1], grid[2]});
  Program::Table table;
  table.name = "pme_moduli";
  table.count = 3;
  table.columns = longest;
  table.values.assign(3 * longest, 0.0);
  for (int k = 0; k != 3; ++k) {
    std::vector<double> moduli = getSplineModuli(grid[k], control.pmeOrder);
    if (control.pmeOptimal) {
      std::vector<double> factors = getAliasFactors(grid[k], control.pmeOrder);
      for (int64_t i = 0; i != grid[k]; ++i)
        moduli[i] *= factors[i];
    }
    std::copy(moduli.begin(), moduli.end(),
              table.values.begin() + k * longest);
  }
  program.tables.push_back(std::move(table));

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
  // whose virial is its energy on the diagonal.
  double squares = 0.0, net = 0.0;
  for (double q : topology.charges) {
    squares += q * q;
    net += q;
  }
  double self = -coulombInternal * beta / std::sqrt(M_PI) * squares;
  double volume = system.box[0] * system.box[1] * system.box[2];
  double background =
      -coulombInternal * M_PI * net * net / (2.0 * volume * beta * beta);
  program.pme = true;
  program.pmeConstantEnergy = self + background;
  program.pmeSelfEnergy = self;
  program.pmeConstantVirial = 3.0 * background;
  program.pmeBeta = beta;
  for (int k = 0; k != 3; ++k)
    program.pmeGrid[k] = grid[k];
  return llvm::Error::success();
}

void Builder::emitTopologyPotential(StringRef name, unsigned terms) {
  double cutoff = control.cutoffDistance * units::length;
  auto has = [&](StringRef set) {
    return llvm::any_of(program.tupleSets, [&](const Program::TupleSet &s) {
      return s.name == set;
    });
  };
  auto emitLennardJones = [&](StringRef sigma, StringRef epsilon,
                              StringRef result) {
    os << "    %c4 = arith.constant 4.0 : f64\n"
       << "    %i6 = arith.constant 6 : i32\n"
       << "    %sr = arith.divf " << sigma << ", %r : f64\n"
       << "    %s6 = math.fpowi %sr, %i6 : f64, i32\n"
       << "    %s12 = arith.mulf %s6, %s6 : f64\n"
       << "    %t = arith.subf %s12, %s6 : f64\n"
       << "    %e4 = arith.mulf %c4, " << epsilon << " : f64\n"
       << "    " << result << " = arith.mulf %e4, %t : f64\n";
  };

  os << "md.potential @" << name << "(%x: !vec, %cell: !md.cell"
     << getFieldParameters() << ") -> f64 {\n";
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
  if (lj || coulomb) {
    os << "  %n = md.neighborhood %x, %cell cutoff(" << formatReal(cutoff)
       << ")";
    if (has("excluded"))
      os << " exclude(%r_excluded : !rel_excluded)";
    os << " : !vec -> !pairs\n"
       << "  %u_nonbonded = md.sum_relation %n, %x, %cell gather(%p_type, "
          "%p_q : !ids, !real)\n"
       << "      exchange(symmetric) {\n"
       << "  ^bb0(%r: f64, %d: vector<3xf64>, %type_i: i32, %type_j: i32, "
          "%q_i: f64, %q_j: f64):\n";
    std::string value;
    if (lj) {
      os << "    %sigma = md.lookup %t_lj_sigma[%type_i, %type_j] : !table, "
            "i32, i32 -> f64\n"
         << "    %epsilon = md.lookup %t_lj_epsilon[%type_i, %type_j] : "
            "!table, i32, i32 -> f64\n";
      emitLennardJones("%sigma", "%epsilon", "%lj");
      value = "%lj";
    }
    if (coulomb) {
      os << "    %f = arith.constant " << formatReal(coulombInternal)
         << " : f64\n"
         << "    %qq = arith.mulf %q_i, %q_j : f64\n"
         << "    %fqq = arith.mulf %f, %qq : f64\n";
      if (program.pme) {
        // The direct sum of particle mesh Ewald, f q q erfc(β r) / r,
        // shifted to 0 at the cutoff if the control file asks.
        double beta = program.pmeBeta;
        os << "    %beta = arith.constant " << formatReal(beta) << " : f64\n"
           << "    %br = arith.mulf %beta, %r : f64\n"
           << "    %erfc = math.erfc %br : f64\n"
           << "    %screened = arith.divf %erfc, %r : f64\n";
        std::string kernel = "%screened";
        if (control.pmeShift) {
          os << "    %shift = arith.constant "
             << formatReal(std::erfc(beta * cutoff) / cutoff) << " : f64\n"
             << "    %shifted = arith.subf %screened, %shift : f64\n";
          kernel = "%shifted";
        }
        os << "    %coulomb = arith.mulf %fqq, " << kernel << " : f64\n";
      } else {
        os << "    %coulomb = arith.divf %fqq, %r : f64\n";
      }
      if (value.empty()) {
        value = "%coulomb";
      } else {
        os << "    %k = arith.addf %lj, %coulomb : f64\n";
        value = "%k";
      }
    }
    os << "    md.yield " << value << " : f64\n"
       << "  } : !pairs, !vec -> f64\n";
    add("nonbonded");
  }

  if (program.pme && (terms & CoulombExcluded) && has("excluded")) {
    // The excluded pairs take their share of the reciprocal sum out again:
    // −f q_i q_j erf(β r) / r.
    os << "  %u_excluded = md.sum_tuples %r_excluded, %x, %cell coordinates("
          "distance(0, 1))\n"
       << "      gather(%p_q : !real) {\n"
       << "  ^bb0(%r: f64, %q_i: f64, %q_j: f64):\n"
       << "    %f = arith.constant " << formatReal(-coulombInternal)
       << " : f64\n"
       << "    %beta = arith.constant " << formatReal(program.pmeBeta)
       << " : f64\n"
       << "    %qq = arith.mulf %q_i, %q_j : f64\n"
       << "    %fqq = arith.mulf %f, %qq : f64\n"
       << "    %br = arith.mulf %beta, %r : f64\n"
       << "    %erf = math.erf %br : f64\n"
       << "    %shielded = arith.divf %erf, %r : f64\n"
       << "    %e = arith.mulf %fqq, %shielded : f64\n"
       << "    md.yield %e : f64\n"
       << "  } : !rel_excluded, !vec -> f64\n";
    add("excluded");
  }
  if (program.pme && (terms & CoulombReciprocal)) {
    os << "  %u_reciprocal, %f_reciprocal, %w_reciprocal = md.reciprocal %x, "
          "%p_q, %cell, %t_pme_moduli\n"
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
  bool lj14 = terms & LennardJones14, coulomb14 = terms & Coulomb14;
  if ((lj14 || coulomb14) && has("pairs14")) {
    os << "  %u_pairs14 = md.sum_tuples %r_pairs14, %x, %cell coordinates("
          "distance(0, 1))\n"
       << "      tuple(%f_pairs14_sigma, %f_pairs14_epsilon, %f_pairs14_qq : "
          "!of_pairs14, !of_pairs14, !of_pairs14) {\n"
       << "  ^bb0(%r: f64, %sigma: f64, %epsilon: f64, %qq: f64):\n";
    std::string value;
    if (lj14) {
      emitLennardJones("%sigma", "%epsilon", "%lj");
      value = "%lj";
    }
    if (coulomb14) {
      os << "    %coulomb = arith.divf %qq, %r : f64\n";
      if (value.empty()) {
        value = "%coulomb";
      } else {
        os << "    %k = arith.addf %lj, %coulomb : f64\n";
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
  double from = control.switchDistance * units::length;

  std::string truncation;
  switch (control.truncation) {
  case Truncation::None:
    break;
  case Truncation::Shift:
    truncation = " truncation(shift)";
    break;
  case Truncation::Switch:
    truncation = " truncation(switch, from = " + formatReal(from) + ")";
    break;
  case Truncation::ForceSwitch:
    truncation = " truncation(force_switch, from = " + formatReal(from) + ")";
    break;
  }

  os << "md.potential @energy(%x: !vec, %cell: !md.cell"
     << getFieldParameters() << ") -> f64 {\n";
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

    std::string value = expression.emit(os, values, "%e", "    ");
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
  os << "  md.return " << total << " : f64\n}\n\n";
  return llvm::Error::success();
}

void Builder::emitPrograms() {
  if (control.minimize) {
    emitDescend();
    return;
  }
  std::string evaluate = "md.evaluate @energy(%x1, %cell" + getFieldValues() +
                         ")";
  std::string signature = "(!vec, !md.cell" + getFieldTypes() + ")";
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
  struct Kind {
    const char *name;
    bool energy, virial, scales;
  };
  llvm::SmallVector<Kind> kinds = {{"step", false, false, false},
                                   {"step_energy", true, true, false}};
  if (trotter) {
    if (!scalesEveryStep())
      kinds.push_back({"step_virial", false, true, false});
    kinds.push_back({"step_trotter", false, true, true});
    kinds.push_back({"step_trotter_energy", true, true, true});
  }
  for (const Kind &kind : kinds) {
    bool withEnergy = kind.energy, withVirial = kind.virial;
    bool scales = kind.scales;
    bool returnsCurrent = leapfrog && withVirial;
    os << "dyn.program @" << kind.name
       << "(%x: !vec, %v: !vec, %f: !vec, %m: !real,\n"
       << "    %cell: !md.cell, %dt: f64" << (scales ? ", %mu: f64" : "")
       << getScaleParameter() << getFieldParameters() << ")\n"
       << "    -> (!vec, !vec, !vec" << (withEnergy ? ", f64" : "")
       << (withVirial ? ", vector<9xf64>" : "")
       << (returnsCurrent ? ", !vec" : "") << (scales ? ", f64" : "")
       << ")\n"
       << "    attributes {"
       << (leapfrog ? "velocity_offset = -0.5,\n                " : "")
       << "provides = [\"symplectic\", \"time_reversible\"]} {\n"
       << "  %c = arith.constant 5.0e-01 : f64\n"
       << "  %half = arith.mulf %c, %dt : f64\n";
    if (!leapfrog) {
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
      os << "  %xh = dyn.drift %x, %v1, %half : !vec\n"
         << "  %mu_unit = arith.constant 1.0 : f64\n"
         << "  %mu_m1 = arith.subf %mu_unit, %mu : f64\n"
         << "  %muinv = arith.divf %mu_unit, %mu : f64\n";
      std::string scaled = emitGroupScaling("  ", "%xh", "%mu", "%mu_m1",
                                            "%cell", "%m", "%r_", "_t");
      // The kinetic energy of the velocities that the scaling takes.
      os << "  %khalf = md.sum_particles gather(%v1, %m : !vec, !real) {\n"
         << "  ^bb0(%kh_v: vector<3xf64>, %kh_m: f64):\n"
         << "    %kh_c = arith.constant 5.0e-01 : f64\n"
         << "    %kh_s = arith.mulf %kh_v, %kh_v : vector<3xf64>\n"
         << "    %kh_r = vector.reduction <add>, %kh_s : vector<3xf64> into "
            "f64\n"
         << "    %kh_mr = arith.mulf %kh_m, %kh_r : f64\n"
         << "    %kh_k = arith.mulf %kh_c, %kh_mr : f64\n"
         << "    md.yield %kh_k : f64\n"
         << "  } : f64\n";
      os << "  %v1t = md.map_particles gather(%v1 : !vec) {\n"
         << "  ^bb0(%v_i: vector<3xf64>):\n"
         << "    %mib = vector.broadcast %muinv : f64 to vector<3xf64>\n"
         << "    %v_s = arith.mulf %mib, %v_i : vector<3xf64>\n"
         << "    md.yield %v_s : vector<3xf64>\n"
         << "  } : !vec\n"
         << "  " << drifted << " = dyn.drift " << scaled
         << ", %v1t, %half : !vec\n";
      drifting = "%v1t";
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
    if (leapfrog && !withVirial) {
      os << "  dyn.return %x1, " << velocities
         << ", %f1 : !vec, !vec, !vec\n}\n\n";
      continue;
    }
    // The second half kick, to the velocities of the time of the new
    // positions: those of the next step with velocity Verlet, and those of
    // the energies with leapfrog.
    os << "  %v2" << (constraints ? "u" : "") << " = dyn.kick " << velocities
       << ", %f1, %m, %half : !vec\n";
    // The virial of the constraints over the first half of the step.
    if (constraints && withVirial) {
      unsigned index = 0;
      for (auto [set, change] : changes)
        virial = emitConstraintVirial("  ", *set, "%x", change, virial,
                                      "%w1x" + std::to_string(index++));
    }
    if (constraints)
      virial = project("%x1", "%v2u", "%v2", withVirial ? virial : "")
                   .second;
    std::string stored = leapfrog ? velocities : "%v2";
    if (withVirial)
      os << "  dyn.return %x1, " << stored << ", %f1"
         << (withEnergy ? ", %u1" : "") << ", " << virial
         << (returnsCurrent ? ", %v2" : "") << (scales ? ", %khalf" : "")
         << "\n"
         << "      : !vec, !vec, !vec" << (withEnergy ? ", f64" : "")
         << ", vector<9xf64>" << (returnsCurrent ? ", !vec" : "")
         << (scales ? ", f64" : "") << "\n";
    else
      os << "  dyn.return %x1, %v2, %f1 : !vec, !vec, !vec\n";
    os << "}\n\n";
  }
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
  std::string inner = (indent + "  ").str();
  std::string change = (result + "_change").str();
  std::string coordinates, tuple, tupleTypes, arguments;
  for (unsigned k = 1; k <= count; ++k) {
    coordinates += (k == 1 ? "" : ", ") + ("displacement(" +
                                           std::to_string(k) + ", 0)");
    tuple += (k == 1 ? "" : ", ") + ("%f_" + set.name + "_d" +
                                     std::to_string(k));
    tupleTypes += (k == 1 ? "" : ", ") + ("!of_" + set.name);
    arguments += "%vs_r" + std::to_string(k) + ": vector<3xf64>, ";
  }
  for (StringRef field : {"o", "n"})
    for (unsigned k = 0; k <= count; ++k)
      arguments += ("%vs_" + field + std::to_string(k) + ": vector<3xf64>, ")
                       .str();
  for (unsigned k = 0; k <= count; ++k)
    arguments += "%vs_m" + std::to_string(k) + ": f64, ";
  for (unsigned k = 1; k <= count; ++k)
    arguments += "%vs_d" + std::to_string(k) + ": f64" +
                 (k == count ? "" : ", ");
  os << indent << change << " = md.gather_tuples %r_" << set.name << ", " << x
     << ", %cell\n"
     << indent << "    coordinates(" << coordinates << ")\n"
     << indent << "    gather(" << old << ", " << x
     << ", %m : !vec, !vec, !real)\n"
     << indent << "    tuple(" << tuple << " : " << tupleTypes << ") {\n"
     << indent << "^bb0(" << arguments << "):\n";
  SiteKernel k(os, inner);
  // The old bonds, in the periods of the new ones.
  std::vector<std::string> bonds, inverse;
  std::string one = k.constant(1.0), two = k.constant(2.0);
  std::string inverseCenter = k.real("divf", one, "%vs_m0");
  for (unsigned j = 1; j <= count; ++j) {
    std::string n = std::to_string(j);
    std::string raw = k.vector("subf", "%vs_n" + n, "%vs_n0");
    std::string shift = k.vector("subf", "%vs_r" + n, raw);
    bonds.push_back(k.vector(
        "addf", k.vector("subf", "%vs_o" + n, "%vs_o0"), shift));
    inverse.push_back(k.real("divf", one, "%vs_m" + n));
  }
  std::string zero = k.zero();
  // The atoms move along the old bonds s_j: the hydrogen j by λ_j s_j / m_j
  // and the heavy atom by −Σ λ_j s_j / m_0. Each iteration of Newton solves
  // the constraints linearized at the current bonds r_k exactly,
  //   Σ_j 2 (r_k · s_j) (δ_kj / m_j + 1 / m_0) λ_j = d_k² − r_k²,
  // so that the error squares with each iteration whatever the masses; a
  // relaxation bond by bond shrinks it only by about m_H / m_X, which the
  // repartitioned masses of hydrogen make close to 1/2.
  os << inner << "%vs_c0 = arith.constant 0 : index\n"
     << inner << "%vs_c1 = arith.constant 1 : index\n"
     << inner << "%vs_iterations = arith.constant " << shakeIterations
     << " : index\n";
  std::string results, inits, types;
  for (unsigned j = 0; j <= count; ++j) {
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
  for (unsigned j = 0; j <= count; ++j)
    moved.push_back("%vs_s" + std::to_string(j));
  std::vector<std::string> current, error;
  for (unsigned j = 1; j <= count; ++j) {
    std::string n = std::to_string(j);
    current.push_back(l.vector(
        "subf", l.vector("addf", "%vs_r" + n, moved[j]), moved[0]));
    error.push_back(l.real("subf", l.real("mulf", "%vs_d" + n, "%vs_d" + n),
                           l.dot(current.back(), current.back())));
  }
  std::vector<std::vector<std::string>> A(count,
                                          std::vector<std::string>(count));
  for (unsigned i = 0; i != count; ++i)
    for (unsigned j = 0; j != count; ++j) {
      std::string weight = i == j ? l.real("addf", inverse[j], inverseCenter)
                                  : inverseCenter;
      A[i][j] = l.real("mulf", l.real("mulf", two, l.dot(current[i],
                                                          bonds[j])),
                       weight);
    }
  std::vector<std::string> lambda = emitSmallSolve(l, A, error);
  for (unsigned j = 1; j <= count; ++j) {
    std::string push = l.scale(lambda[j - 1], bonds[j - 1]);
    moved[j] = l.vector("addf", moved[j], l.scale(inverse[j - 1], push));
    moved[0] = l.vector("subf", moved[0], l.scale(inverseCenter, push));
  }
  std::string yielded;
  for (unsigned j = 0; j <= count; ++j)
    yielded += (j == 0 ? "" : ", ") + moved[j];
  os << inner << "  scf.yield " << yielded << " : " << types << "\n"
     << inner << "}\n"
     << inner << "md.yield " << results << " : " << types << "\n"
     << indent << "} : !rel_" << set.name << ", !vec -> !vec\n";
  os << indent << result << " = md.map_particles gather(" << x << ", "
     << change << " : !vec, !vec) {\n"
     << indent << "^bb0(%vs_x: vector<3xf64>, %vs_d: vector<3xf64>):\n"
     << inner << "%vs_sum = arith.addf %vs_x, %vs_d : vector<3xf64>\n"
     << inner << "md.yield %vs_sum : vector<3xf64>\n"
     << indent << "} : !vec\n";
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
  auto header = [&](StringRef name, StringRef op) {
    os << indent << name << " = md." << op << " %r_" << set.name << ", " << x
       << ", %cell\n"
       << indent << "    coordinates(" << coordinates << ")\n"
       << indent << "    gather(" << v << ", %m : !vec, !real)\n"
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
  header(change, "gather_tuples");
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
    // two halves (emitConstraintVirial).
    std::string sum = (virialResult + "_sum").str();
    header(sum, "sum_tuples");
    SiteKernel w(os, inner);
    std::vector<std::string> dv = solve(w);
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
  auto header = [&](StringRef op, StringRef type) {
    os << indent << " = md." << op << " %r_settles, " << x << ", %cell\n"
       << indent << "    coordinates(displacement(1, 0), displacement(2, 0), "
                    "displacement(2, 1))\n"
       << indent << "    gather(" << v << ", %m : !vec, !real)\n"
       << indent << "    tuple(%f_settles_doh, %f_settles_dhh : !of_settles, "
                    "!of_settles) {\n"
       << indent << "^bb0(%vs_d10: vector<3xf64>, %vs_d20: vector<3xf64>, "
                    "%vs_d21: vector<3xf64>, %vs_v0: vector<3xf64>, "
                    "%vs_v1: vector<3xf64>, %vs_v2: vector<3xf64>, "
                    "%vs_m0: f64, %vs_m1: f64, %vs_m2: f64, %vs_doh: f64, "
                    "%vs_dhh: f64):\n";
    (void)type;
  };
  std::string change = (result + "_settle").str();
  os << indent << change;
  header("gather_tuples", "!vec");
  SiteKernel k(os, inner);
  std::array<std::string, 3> dv = emitKernel(k);
  os << inner << "md.yield " << dv[0] << ", " << dv[1] << ", " << dv[2]
     << " : vector<3xf64>, vector<3xf64>, vector<3xf64>\n"
     << indent << "} : !rel_settles, !vec -> !vec\n";

  std::string virialName = virial.str();
  if (!virial.empty()) {
    // The forces of the constraints over the second half of the step,
    // G = 2 m Δv / dt, and their virial Σ (x_i − x_O) ⊗ G_i, with the
    // weight ½ (emitConstraintVirial).
    std::string sum = (virialResult + "_settle").str();
    os << indent << sum;
    header("sum_tuples", "vector<9xf64>");
    SiteKernel w(os, inner);
    std::array<std::string, 3> change = emitKernel(w);
    std::string factor = w.real("divf", "%vs_m1", "%dt");
    std::array<std::string, 2> arms = {"%vs_d10", "%vs_d20"};
    std::array<std::string, 2> forces = {w.scale(factor, change[1]),
                                         w.scale(factor, change[2])};
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

/// The sum over the particles of F^2 / m. With the time step it gives the
/// kinetic energy at the half steps before and after a step, from which
/// the temperature and the pressure are estimated (Jung et al., J. Chem.
/// Phys. 148, 164109 (2018)).
static void emitForceSquare(llvm::raw_ostream &os, StringRef result,
                            StringRef forces, StringRef masses,
                            StringRef indent) {
  os << indent << result << " = md.sum_particles gather(" << forces << ", "
     << masses << " : !vec, !real) {\n"
     << indent << "^bb0(%f_i: vector<3xf64>, %m_i: f64):\n"
     << indent << "  %sq = arith.mulf %f_i, %f_i : vector<3xf64>\n"
     << indent
     << "  %f2 = vector.reduction <add>, %sq : vector<3xf64> into f64\n"
     << indent << "  %g = arith.divf %f2, %m_i : f64\n"
     << indent << "  %zero = arith.constant 0.0 : f64\n"
     << indent << "  %massless = arith.cmpf oeq, %m_i, %zero : f64\n"
     << indent << "  %h = arith.select %massless, %zero, %g : f64\n"
     << indent << "  md.yield %h : f64\n"
     << indent << "} : f64\n";
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
  std::string outside = level == 0
                            ? "0"
                            : isReordered(level - 1)
                                  ? "s"
                                  : "a" + std::to_string(level - 1);
  const Level &current = levels[level];
  bool isStepLoop = level + 1 == levels.size();
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
    for (int k = 0; k != 3; ++k)
      os << inner << "%edge" << here << "_" << k << " = memref.load "
         << "%box_memory[%c_edge" << k << "] : memref<3xf64>\n";
    os << inner << cellName << " = md.orthorhombic_cell %edge" << here
       << "_0, %edge" << here << "_1, %edge" << here << "_2\n";
    if (scalesReference()) {
      scaleName = "%rest_scale" + here;
      os << inner << scaleName << " = arith.divf %edge" << here
         << "_0, %rest_edge : f64\n";
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
    os << inner << getValues("b" + here) << " = dyn.step @step(";
    os << "%xa" << here << ", %va" << here << ", %fa" << here << ", "
         << massName << ", " << cellName << ", %dt" << getScaleValue() << getFieldValues(fieldPrefix)
         << ")\n"
         << inner << "    : (!vec, !vec, !vec, !real, !md.cell, f64" << getScaleType()
         << getFieldTypes() << ") -> (!vec, !vec, !vec)\n";
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
      for (int k = 0; k != 3; ++k)
        os << inner << "%edger" << here << "_" << k << " = memref.load "
           << "%box_memory[%c_edge" << k << "] : memref<3xf64>\n";
      os << inner << cellName << " = md.orthorhombic_cell %edger" << here
         << "_0, %edger" << here << "_1, %edger" << here << "_2\n";
      if (scalesReference()) {
        scaleName = "%rest_scaler" + here;
        os << inner << scaleName << " = arith.divf %edger" << here
           << "_0, %rest_edge : f64\n";
      }
    }

    // The iteration of the loop of `upto` that is under way, counted over
    // the whole run.
    auto emitIteration = [&](unsigned upto) {
      std::string counted;
      for (unsigned i = 0; i <= upto; ++i) {
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
      if (current.name == "couple" && level != 0 &&
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
         << ", %start : i64\n";
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
        // The pressure of the state that the step before left (D92).
        std::string m = "%tm" + here;
        for (int k = 0; k != 3; ++k)
          os << inner << m << "_" << k << " = memref.load %trotter_memory"
             << "[%c_edge" << k << "] : memref<3xf64>\n";
        trace = m + "_0";
        groups = m + "_1";
        kinetic = m + "_2";
        x = "%x" + last;
        v = "%v" + last;
      } else {
        // The step whose pressure gives the strain.
        os << inner << getValues(a) << ", %w" << a
           << (leapfrog ? ", %vc" + a : "") << " = dyn.step @step_virial(%x"
           << last << ", %v" << last << ", %f" << last << ", " << massName
           << ", " << cellName << ", %dt" << getScaleValue()
           << getFieldValues(fieldPrefix) << ")\n"
           << inner << "    : (!vec, !vec, !vec, !real, !md.cell, f64"
           << getScaleType() << getFieldTypes()
           << ") -> (!vec, !vec, !vec, vector<9xf64>"
           << (leapfrog ? ", !vec" : "") << ")\n";
        trace = "%tr" + a;
        emitTrace(os, trace, "%w" + a, inner);
        x = "%x" + a;
        v = "%v" + a;
      }
      emitStep();
      TrotterScaling scaling =
          scalesEveryStep()
              ? emitTrotterStrain(inner, "", "", trace, here,
                                  "%step" + here, kinetic, groups)
              : emitTrotterStrain(inner, "%x" + a,
                                  leapfrog ? "%vc" + a : "%v" + a, trace,
                                  here, "%step" + here);
      std::string outerScale = scaleName;
      if (!scaling.scale.empty())
        scaleName = scaling.scale;
      os << inner << "%x" << n << ", %v" << n << ", %f" << n
         << (withEnergy ? ", %u" + n : "") << ", %w" << n
         << (leapfrog ? ", %vc" + n : "") << ", %kh" << n
         << " = dyn.step @step_trotter" << (withEnergy ? "_energy" : "")
         << "(" << x << ", " << v << ", %f"
         << (scalesEveryStep() ? last : a) << ", " << massName << ", "
         << scaling.cell << ", %dt, " << scaling.mu << getScaleValue()
         << getFieldValues(fieldPrefix) << ")\n"
         << inner << "    : (!vec, !vec, !vec, !real, !md.cell, f64, f64"
         << getScaleType() << getFieldTypes() << ") -> (!vec, !vec, !vec"
         << (withEnergy ? ", f64" : "") << ", vector<9xf64>"
         << (leapfrog ? ", !vec" : "") << ", f64)\n";
      scaleName = outerScale;
      scaling.kineticHalf = "%kh" + n;
      return scaling;
    };

    if (current.name == "couple") {
      // The last step of the period, and the coupling after it. The
      // barostat needs the virial of the step.
      std::string trace;
      if (usesTrotter()) {
        std::string k = "k" + here;
        TrotterScaling scaling = emitTrotterSteps(k, /*withEnergy=*/false);
        trace = "%tr" + k;
        emitTrace(os, trace, "%w" + k, inner);
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
        os << inner << getValues("k" + here) << ", %uk" << here << ", %wk"
           << here << (isLeapfrog() ? ", " + current : "")
           << " = dyn.step @step_energy(%x" << last << ", %v" << last
           << ", %f" << last << ", " << massName << ", " << cellName
           << ", %dt" << getScaleValue() << getFieldValues(fieldPrefix) << ")\n"
           << inner << "    : (!vec, !vec, !vec, !real, !md.cell, f64" << getScaleType()
           << getFieldTypes()
           << ") -> (!vec, !vec, !vec, f64, vector<9xf64>"
           << (isLeapfrog() ? ", !vec" : "") << ")\n";
        trace = "%trk" + here;
        emitTrace(os, trace, "%wk" + here, inner);
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
      os << inner << getValues("k" + here) << " = dyn.step @step(";
      os << "%x" << last << ", %v" << last << ", %f" << last << ", "
           << massName << ", " << cellName << ", %dt" << getScaleValue() << getFieldValues(fieldPrefix)
           << ")\n"
           << inner << "    : (!vec, !vec, !vec, !real, !md.cell, f64" << getScaleType()
           << getFieldTypes() << ") -> (!vec, !vec, !vec)\n";
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
      std::string steps = std::to_string(levels.size() - 1);
      os << inner << getValues("q" + here) << " = scf.for %j" << here
         << " = %c0 to %n" << steps << " step %c1\n"
         << inner << "    iter_args(" << getInits("p" + here, last) << ")\n"
         << inner << "    -> (" << state << ") {\n";
      os << inner << "  " << getValues("r" + here) << " = dyn.step @step(";
      os << "%xp" << here << ", %vp" << here << ", %fp" << here << ", "
           << massName << ", " << cellName << ", %dt" << getScaleValue() << getFieldValues(fieldPrefix)
           << ")\n"
           << inner << "      : (!vec, !vec, !vec, !real, !md.cell, f64" << getScaleType()
           << getFieldTypes() << ") -> (!vec, !vec, !vec)\n";
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
        os << inner << "%xl, %vl, %fl, %u, %w"
           << (isLeapfrog() ? ", %vn" : "") << " = dyn.step @step_energy(%x"
           << last << ", %v" << last << ", %f" << last << ", " << massName
           << ", " << cellName << ", %dt" << getScaleValue()
           << getFieldValues(fieldPrefix) << ")\n"
           << inner << "    : (!vec, !vec, !vec, !real, !md.cell, f64"
           << getScaleType() << getFieldTypes()
           << ") -> (!vec, !vec, !vec, f64, vector<9xf64>"
           << (isLeapfrog() ? ", !vec" : "") << ")\n";
      }
      emitKineticEnergy(os, "%k", isLeapfrog() ? now : "%vl", massName,
                        inner);
      // With constraints the forces do not give the kinetic energies of
      // the half steps (Section 9 of design-m1.md); the log takes that of
      // the step.
      if (hasConstraints())
        os << inner << "%g = arith.constant 0.0 : f64\n";
      else
        emitForceSquare(os, "%g", "%fl", massName, inner);
      emitTrace(os, "%tr", virialName, inner);
      if (!scalesHere)
        emitStep();
      os << inner << "func.call @mdrtWriteEnergies(%step" << here << ", "
         << energyName << ", %k, %g, %tr) : (i64, f64, f64, f64, f64) -> ()\n";
      std::string yielded =
          couplesBelow
              ? getCoupled("%xl", "%vl", "%fl", energyName, "%tr",
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
     << indent << "  %mub = vector.broadcast " << mu
     << " : f64 to vector<3xf64>\n"
     << indent << "  %x_scaled = arith.mulf %mub, %x_i : vector<3xf64>\n"
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
      yielded += (j == 0 ? "" : ", ") + k.scale(oneMinusMu, place);
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
                         StringRef trace, StringRef tag, StringRef step) {
  std::string t = tag.str();
  for (int k = 0; k != 3; ++k)
    os << indent << "%be" << t << "_" << k << " = memref.load %box_memory"
       << "[%c_edge" << k << "] : memref<3xf64>\n";
  os << indent << "%bxy" << t << " = arith.mulf %be" << t << "_0, %be" << t
     << "_1 : f64\n"
     << indent << "%bv" << t << " = arith.mulf %bxy" << t << ", %be" << t
     << "_2 : f64\n";
  os << indent << "%bk2" << t << " = arith.addf " << kinetic << ", "
     << kinetic << " : f64\n"
     << indent << "%bw0" << t << " = arith.addf %bk2" << t << ", " << trace
     << " : f64\n"
     << indent << "%bwc" << t << " = arith.divf %baro_constant, %bv" << t
     << " : f64\n"
     << indent << "%bw" << t << " = arith.addf %bw0" << t << ", %bwc" << t
     << " : f64\n"
     << indent << "%b3v" << t << " = arith.mulf %c_three, %bv" << t
     << " : f64\n"
     << indent << "%bpi" << t << " = arith.divf %bw" << t << ", %b3v" << t
     << " : f64\n"
     << indent << "%bp" << t << " = arith.mulf %bpi" << t
     << ", %c_bar : f64\n"
     << indent << "%strain" << t
     << " = func.call @mdrtBarostatStrain(%seed, " << step << ", %bp" << t
     << ", %baro_target, %bv" << t
     << ", %baro_kt, %baro_beta, %baro_rate)\n"
     << indent << "    : (i64, i64, f64, f64, f64, f64, f64, f64) -> f64\n"
     << indent << "%bs3" << t << " = arith.mulf %strain" << t
     << ", %c_third : f64\n"
     << indent << "%mu" << t << " = math.exp %bs3" << t << " : f64\n"
     << indent << "%muinv" << t << " = arith.divf %c_unit, %mu" << t
     << " : f64\n";
  // The new cell, which the next iteration takes from memory.
  for (int k = 0; k != 3; ++k)
    os << indent << "%bn" << t << "_" << k << " = arith.mulf %be" << t
       << "_" << k << ", %mu" << t << " : f64\n"
       << indent << "memref.store %bn" << t << "_" << k
       << ", %box_memory[%c_edge" << k << "] : memref<3xf64>\n";
  os << indent << "func.call @mdrtSetBox(%bn" << t << "_0, %bn" << t
     << "_1, %bn" << t << "_2) : (f64, f64, f64) -> ()\n";
}

Builder::TrotterScaling Builder::emitTrotterStrain(
    StringRef indent, StringRef positions, StringRef velocities,
    StringRef trace, StringRef tag, StringRef step, StringRef givenKinetic,
    StringRef givenGroups) {
  std::string t = tag.str();
  // The kinetic energy of the pressure: that of the velocities of the
  // step, without the center of mass.
  std::string kinetic = givenKinetic.str();
  if (kinetic.empty())
    kinetic = emitKineticWithoutCenter(indent, velocities, massName, t);
  emitStrain(indent, kinetic, trace, t, step);
  std::string relations =
      ("%r" + StringRef(fieldPrefix).drop_front(2)).str();
  std::string groups = givenGroups.str();
  if (groups.empty())
    groups = emitGroupTrace(indent, trace, positions, velocities, cellName,
                            massName, relations, "a" + t);
  return finishTrotterStrain(indent, groups, t);
}

std::string Builder::emitKineticWithoutCenter(StringRef indent,
                                              StringRef velocities,
                                              StringRef masses,
                                              StringRef tag) {
  std::string t = tag.str();
  std::string kinetic = "%tk" + t;
  emitKineticEnergy(os, kinetic, velocities, masses, indent);
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
       << indent << "%tpv" << t << " = vector.reduction <add>, %tpvs" << t
       << " : vector<3xf64> into f64\n"
       << indent << "%tkcm" << t << " = arith.mulf %couple_half, %tpv" << t
       << " : f64\n"
       << indent << "%tkt" << t << " = arith.subf " << kinetic << ", %tkcm"
       << t << " : f64\n";
    kinetic = "%tkt" + t;
  }
  return kinetic;
}

void Builder::emitStoreTrotterState(StringRef indent, StringRef trace,
                                    StringRef groups, StringRef kinetic) {
  StringRef values[] = {trace, groups, kinetic};
  for (int k = 0; k != 3; ++k)
    os << indent << "memref.store " << values[k]
       << ", %trotter_memory[%c_edge" << k << "] : memref<3xf64>\n";
  // The checkpoints keep it, so a run that continues takes the same.
  os << indent << "func.call @mdrtSetBarostatState(" << trace << ", "
     << groups << ", " << kinetic << ") : (f64, f64, f64) -> ()\n";
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
     << ", %bwc" << t << " : f64\n";
  scaling.newVolume = "%tvn" + t;
  os << indent << "%tvn0" << t << " = arith.mulf %bn" << t << "_0, %bn" << t
     << "_1 : f64\n"
     << indent << scaling.newVolume << " = arith.mulf %tvn0" << t << ", %bn"
     << t << "_2 : f64\n";
  scaling.cell = "%tcell" + t;
  os << indent << scaling.cell << " = md.orthorhombic_cell %bn" << t
     << "_0, %bn" << t << "_1, %bn" << t << "_2\n";
  if (scalesReference()) {
    scaling.scale = "%tscale" + t;
    os << indent << scaling.scale << " = arith.divf %bn" << t
       << "_0, %rest_edge : f64\n";
  }
  return scaling;
}

Builder::Coupled
Builder::emitCoupling(StringRef indent, StringRef positions,
                      StringRef velocities, StringRef forces,
                      StringRef energy, StringRef tag, StringRef step,
                      StringRef trace, const TrotterScaling *trotter) {
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
  // The energy that the coupling takes: that of the center of mass, and
  // what the thermostat takes from the rest.
  std::string bath = "%kcm" + t;
  if (control.thermostat) {
    std::string kinetic = "%kc" + t;
    emitKineticEnergy(os, kinetic, velocities, massName, indent);
    if (removesMotion) {
      os << indent << "%kt" << t << " = arith.subf %kc" << t << ", %kcm" << t
         << " : f64\n";
      kinetic = "%kt" + t;
    }
    os << indent << "%alpha" << t << " = func.call @mdrtBussiFactor(%seed, "
       << step << ", " << kinetic
       << ", %target_kinetic, %freedom, %decay)\n"
       << indent << "    : (i64, i64, f64, f64, f64, f64) -> f64\n"
       << indent << "%alpha2" << t << " = arith.mulf %alpha" << t
       << ", %alpha" << t << " : f64\n"
       << indent << "%kn" << t << " = arith.mulf %alpha2" << t << ", "
       << kinetic << " : f64\n"
       << indent << "%heat" << t << " = arith.subf %kc" << t << ", %kn" << t
       << " : f64\n";
    bath = "%heat" + t;
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
    std::string kinetic = removesMotion ? "%kt" + t : "%kc" + t;
    if (!trotter)
      emitStrain(indent, kinetic, trace, t, step);
    // The energy that the scaling gives the system: exactly in the
    // velocities, (1/μ² − 1) K, and in the positions either exactly, from
    // the energy of the scaled positions (below), or to first order,
    // −(μ − 1) tr W, with W the sum of d (x) K over the pairs and the
    // virials of the constant terms.
    bool exact = control.barostatWork == BarostatWork::Exact;
    std::string after = control.thermostat ? "%kn" + t : kinetic;
    std::string relations =
        ("%r" + StringRef(fieldPrefix).drop_front(2)).str();
    if (trotter) {
      // The scaling was made within the drift of the step (Trotter type,
      // [Bernetti2020], SI Sec. V.C): the velocities of its middle by 1/μ,
      // which changes their kinetic energy by (1/μ² − 1) K, and the
      // positions by μ, which changes the potential energy by −ln μ tr W
      // to first order: tr W, with the virials of the constant terms and
      // of the motion within the rigid groups, is taken as the mean of
      // those before and after the scaling, which makes the count exact
      // to second order in the strain (D92).
      std::string after =
          emitGroupTrace(indent, trace, positions, velocities, cellName,
                         massName, relations, t);
      // With a scaling every step, the state that the next one takes its
      // pressure from: the kinetic energy after the thermostat.
      if (scalesEveryStep())
        emitStoreTrotterState(indent, trace, after,
                              control.thermostat ? "%kn" + t : kinetic);
      os << indent << "%bwca" << t << " = arith.divf %baro_constant, "
         << trotter->newVolume << " : f64\n"
         << indent << "%bwb" << t << " = arith.addf " << after << ", %bwca"
         << t << " : f64\n"
         << indent << "%bws" << t << " = arith.addf " << trotter->workBefore
         << ", %bwb" << t << " : f64\n"
         << indent << "%bwm" << t << " = arith.mulf %bws" << t
         << ", %couple_half : f64\n"
         << indent << "%bwork" << t << " = arith.mulf %bwm" << t << ", "
         << trotter->logMu << " : f64\n"
         << indent << "%bmi2" << t << " = arith.mulf " << trotter->muinv
         << ", " << trotter->muinv << " : f64\n"
         << indent << "%bmi21" << t << " = arith.subf %bmi2" << t
         << ", %c_unit : f64\n"
         << indent << "%bdk" << t << " = arith.mulf %bmi21" << t << ", "
         << trotter->kineticHalf << " : f64\n"
         << indent << "%bdku" << t << " = arith.subf %bdk" << t << ", %bwork"
         << t << " : f64\n"
         << indent << "%btake" << t << " = arith.subf " << bath << ", %bdku"
         << t << " : f64\n";
      bath = "%btake" + t;
    }
    std::string groupTrace =
        exact || trotter
            ? trace.str()
            : emitGroupTrace(indent, trace, positions, velocities,
                             cellName, massName, relations, t);
    if (!trotter)
      os << indent << "%bm1" << t << " = arith.subf %c_unit, %mu" << t
         << " : f64\n";
    if (!exact && !trotter)
      os << indent << "%bwt" << t << " = arith.addf " << groupTrace
         << ", %bwc" << t << " : f64\n"
         << indent << "%bwork" << t << " = arith.mulf %bm1" << t << ", %bwt"
         << t << " : f64\n";
    if (!trotter)
      os << indent << "%bmi2" << t << " = arith.mulf %muinv" << t
         << ", %muinv" << t << " : f64\n"
         << indent << "%bmi21" << t << " = arith.subf %bmi2" << t
         << ", %c_unit : f64\n"
         << indent << "%bdk" << t << " = arith.mulf %bmi21" << t << ", "
         << after << " : f64\n";
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
      os << indent << cell << " = md.orthorhombic_cell %bn" << t << "_0, %bn"
         << t << "_1, %bn" << t << "_2\n";
      std::string outerCell = cellName, outerScale = scaleName;
      cellName = cell;
      if (scalesReference()) {
        scaleName = "%bscale" + t;
        os << indent << scaleName << " = arith.divf %bn" << t
           << "_0, %rest_edge : f64\n";
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
      os << indent << u << held << ", " << f << held << raw
         << " = md.evaluate @energy(" << newPositions << ", " << cell
         << getFieldValues(fieldPrefix) << ")\n"
         << indent << "    request [energy, forces]\n"
         << indent << "    : (!vec, !md.cell" << getFieldTypes()
         << ") -> (f64, !vec)\n";
      if (hasSites())
        emitSpreadSites(indent, newPositions, f + held + "e", f + held,
                        relations);
      if (hasRestraints())
        emitRestraints(indent, newPositions, fieldPrefix, f + "p", f,
                       u + "p", u);
      cellName = outerCell;
      scaleName = outerScale;
      newForces = f;
      os << indent << "%bdu" << t << " = arith.subf " << u << ", " << energy
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
  if (control.thermostat) {
    os << indent << "  %ab = vector.broadcast %alpha" << t
       << " : f64 to vector<3xf64>\n"
       << indent << "  %s = arith.mulf %ab, " << value
       << " : vector<3xf64>\n";
    value = "%s";
  }
  if (control.barostat && !trotter) {
    os << indent << "  %mib = vector.broadcast %muinv" << t
       << " : f64 to vector<3xf64>\n"
       << indent << "  %v_scaled = arith.mulf %mib, " << value
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
  std::string arguments = "%k_i: f64, %rx_i: f64, %ry_i: f64, %rz_i: f64";
  // d = x − x_ref, and k d; the reference follows the cell.
  auto emitOffset = [&](StringRef inner) {
    std::string reference = "%ref_i";
    os << inner << reference << " = vector.from_elements %rx_i, %ry_i, %rz_i "
                   ": vector<3xf64>\n";
    if (scalesReference()) {
      os << inner << "%sb_i = vector.broadcast " << scaleName
         << " : f64 to vector<3xf64>\n"
         << inner << "%refs_i = arith.mulf %sb_i, %ref_i : vector<3xf64>\n";
      reference = "%refs_i";
    }
    os << inner << "%d_i = arith.subf %x_i, " << reference
       << " : vector<3xf64>\n"
       << inner << "%kb_i = vector.broadcast %k_i : f64 to vector<3xf64>\n"
       << inner << "%kd_i = arith.mulf %kb_i, %d_i : vector<3xf64>\n";
  };
  std::string inner = (indent + "  ").str();
  os << indent << fResult << " = md.map_particles gather(" << x << ", " << f
     << ", " << fields << " : !vec, !vec, !real, !real, !real, !real) {\n"
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
  // diagonal of the virial.
  std::string sum = ((u.empty() ? wResult : uResult) + "_parts").str();
  os << indent << sum << " = md.sum_particles gather(" << x << ", " << fields
     << " : !vec, !real, !real, !real, !real) {\n"
     << indent << "^bb0(%x_i: vector<3xf64>, " << arguments << "):\n";
  emitOffset(inner);
  os << inner << "%kdd_i = arith.mulf %kd_i, %d_i : vector<3xf64>\n"
     << inner << "md.yield %kdd_i : vector<3xf64>\n"
     << indent << "} : vector<3xf64>\n";
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
    os << indent << diagonal[c] << "_s = vector.extract " << sum << "[" << c
       << "] : f64 from vector<3xf64>\n"
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
                                              StringRef v, StringRef trace) {
  emitConstrainedDescent("  ", x, f, "%gc0");
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
    std::string sum = "%trc0_" + set->name;
    os << "  " << sum << " = md.sum_tuples %r_" << set->name << ", " << x
       << ", %cell\n"
       << "    coordinates(" << coordinates << ")\n"
       << "    gather(%gc0, " << f << ", " << v
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
      std::string term = k.dot("%vs_r" + n, force);
      arms = arms.empty() ? term : k.real("addf", arms, term);
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
      std::string term = k.real("mulf", "%vs_m" + n, k.dot(relative, relative));
      twice = twice.empty() ? term : k.real("addf", twice, term);
    }
    std::string virial = k.real("subf", arms, twice);
    os << "    md.yield " << virial << " : f64\n"
       << "  } : !rel_" << set->name << ", !vec -> f64\n";
    std::string next = "%trc0_" + set->name + "_sum";
    os << "  " << next << " = arith.addf " << current << ", " << sum
       << " : f64\n";
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
                         ")";
  std::string signature = "(!vec, !md.cell" + getFieldTypes() + ")";
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
  // The energy and the forces at the start, and the terms.
  std::string raw = hasSites() ? "e" : "";
  std::string held = hasRestraints() ? "p" : "";
  os << "  %u0" << held << ", %f0" << held << raw
     << " = md.evaluate @energy(%x0, %cell" << getFieldValues() << ")\n"
     << "      request [energy, forces]\n"
     << "      : (!vec, !md.cell" << getFieldTypes() << ") -> (f64, !vec)\n";
  if (hasSites())
    emitSpreadSites("  ", "%x0", "%f0" + held + "e", "%f0" + held, "%r_");
  if (hasRestraints()) {
    os << "  %w0z = arith.constant dense<0.0> : vector<9xf64>\n";
    emitRestraints("  ", "%x0", "%p_", "%f0p", "%f0", "%u0p", "%u0", "%w0z",
                   "%w0r");
  }
  emitTerms();
  os << "  %h0 = arith.constant "
     << formatReal(control.minimizeStep * units::length) << " : f64\n"
     << "  %h_most = arith.constant " << formatReal(1.0 * units::length)
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
  std::string reported = emitReported("  ", "%x0", "%f0", "0");
  os << "  mdrt.host_call @mdrtWriteMinimization(%start, %u0, %h0, "
     << reported << ", %id)\n"
     << "      : (i64, f64, f64, !vec, !ids)\n";

  // A step is taken if it lowers the energy, and the next one is longer;
  // otherwise the next one is shorter, from where the step began. The
  // loops: over the intervals between frames (one if there are none),
  // over the intervals between energies in each, and over the steps.
  int64_t period = control.energyPeriod;
  int64_t framePeriod =
      control.framePeriod > 0 ? control.framePeriod : control.numSteps;
  std::string state = "!vec, !vec, f64, f64";
  os << "  %n0 = arith.constant " << control.numSteps / framePeriod
     << " : index\n"
     << "  %n1 = arith.constant " << framePeriod / period << " : index\n"
     << "  %n2 = arith.constant " << period << " : index\n"
     << "  %per0 = arith.constant " << framePeriod << " : index\n"
     << "  %xe0, %fe0, %ue0, %he0 = scf.for %i0 = %c0 to %n0 step %c1\n"
     << "      iter_args(%xa0 = %x0, %fa0 = %f0, %ua0 = %u0, %ha0 = %h0)\n"
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
  os << "  mdrt.host_call @mdrtFinish(%xe0, %v0, %id) : (!vec, !vec, !ids)\n"
     << "  return\n}\n";
}

void Builder::emitTerms() {
  if (!system.topology)
    return;
  // The restraints, if any, last: their energy at the start is that of the
  // evaluation before the terms.
  int size = hasRestraints() ? 11 : 10;
  std::string type = "memref<" + std::to_string(size) + "xf64>";
  os << "  %terms = memref.alloca() : " << type << "\n";
  int index = 0;
  for (StringRef name :
       {"term_lj", "term_coulomb", "term_bonds", "term_angles",
        "term_dihedrals", "term_lj14", "term_coulomb14", "term_cmap",
        "term_excluded", "term_reciprocal"}) {
    os << "  %" << name << " = md.evaluate @" << name << "(%x0, %cell"
       << getFieldValues() << ") request [energy]\n"
       << "      : (!vec, !md.cell" << getFieldTypes() << ") -> f64\n"
       << "  %i_" << name << " = arith.constant " << index++
       << " : index\n"
       << "  memref.store %" << name << ", %terms[%i_" << name
       << "] : " << type << "\n";
  }
  if (hasRestraints())
    os << "  %i_restraints = arith.constant 10 : index\n"
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
     << "func.func private @mdrtWriteTerms(memref<?xf64>)\n"
     << "    attributes {llvm.emit_c_interface}\n"
     << "func.func private @mdrtWriteFrame(i64, memref<?x3x" << state
     << ">, memref<?xi32>)\n    attributes {llvm.emit_c_interface}\n"
     << "func.func private @mdrtFinish(memref<?x3x" << state
     << ">, memref<?x3x" << state << ">, memref<?xi32>)\n"
     << "    attributes {llvm.emit_c_interface}\n";
  if (control.minimize)
    os << "func.func private @mdrtWriteMinimization(i64, f64, f64, memref<?x3x"
       << force << ">, memref<?xi32>)\n"
       << "    attributes {llvm.emit_c_interface}\n";
  if (control.thermostat)
    os << "func.func private @mdrtBussiFactor(i64, i64, f64, f64, f64, f64) "
          "-> f64\n";
  if (control.getCouplingPeriod() > 0)
    os << "func.func private @mdrtAddBath(f64)\n"
       << "    attributes {llvm.emit_c_interface}\n";
  if (control.barostat)
    os << "func.func private @mdrtBarostatStrain(i64, i64, f64, f64, f64, f64, "
          "f64, f64) -> f64\n"
       << "func.func private @mdrtSetBox(f64, f64, f64)\n"
       << "    attributes {llvm.emit_c_interface}\n";
  if (scalesEveryStep())
    os << "func.func private @mdrtSetBarostatState(f64, f64, f64)\n"
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
     << "    %lx: f64, %ly: f64, %lz: f64, %dt: f64, %start: i64) {\n";

  os << "  %c0 = arith.constant 0 : index\n"
     << "  %c1 = arith.constant 1 : index\n";
  for (auto [index, level] : llvm::enumerate(levels))
    os << "  %n" << index << " = arith.constant " << level.count
       << " : index\n";
  // The number of steps in one iteration of each loop. An iteration of
  // the loop over energy intervals takes one step after its loop over
  // steps.
  // An iteration of the loop over the periods of coupling takes one as
  // well (two with the barostat of Trotter type), and one over energy intervals a whole period after its loop over
  // periods.
  int64_t steps = 1;
  for (unsigned i = levels.size(); i-- != 0;) {
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
    os << "  %box_memory = memref.alloca() : memref<3xf64>\n";
    // With a scaling every step, the trace of the virial, that of the
    // rigid groups, and the kinetic energy without the center of mass of
    // the state that the last step left (D92).
    if (scalesEveryStep())
      os << "  %trotter_memory = memref.alloca() : memref<3xf64>\n";
    for (int k = 0; k != 3; ++k)
      os << "  %c_edge" << k << " = arith.constant " << k << " : index\n"
         << "  memref.store %l" << "xyz"[k] << ", %box_memory[%c_edge" << k
         << "] : memref<3xf64>\n";
  }
  if (scalesReference()) {
    // The edge of the cell of the file, which the reference positions of
    // the restraints are for.
    double edge = system.inputBox[0] > 0.0 ? system.inputBox[0]
                                           : system.box[0];
    os << "  %rest_edge = arith.constant " << formatReal(edge) << " : f64\n"
       << "  %rest_scale = arith.divf %lx, %rest_edge : f64\n";
  }
  os << "  %cell = md.orthorhombic_cell %lx, %ly, %lz\n"
     << "  %x" << (program.reorders ? "_in" : "0") << (hasSites() ? "u" : "")
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
                           program.pmeConstantVirial) *
                          volume;
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
           << "  %c_bar = arith.constant 16.6053906717 : f64\n"
           << "  %c_three = arith.constant 3.0 : f64\n"
           << "  %c_third = arith.constant "
           << formatReal(1.0 / 3.0) << " : f64\n"
           << "  %c_unit = arith.constant 1.0 : f64\n";
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

  if (!isRestart()) {
    // The energies at the start.
    StringRef raw = hasSites() ? "e" : "";
    std::string held = hasRestraints() ? "p" : "";
    os << "  %u0" << held << ", %f0" << held << raw << ", %w0" << held << raw
       << " = md.evaluate @energy(%x0, %cell" << getFieldValues() << ")\n"
       << "      request [energy, forces, virial]\n"
       << "      : (!vec, !md.cell" << getFieldTypes()
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
    emitTerms();
    emitKineticEnergy(os, "%k0", velocities, "%m", "  ");
    if (hasConstraints())
      os << "  %g0 = arith.constant 0.0 : f64\n";
    else
      emitForceSquare(os, "%g0", "%f0", "%m", "  ");
    emitTrace(os, "%tr0", virial, "  ");
    std::string trace = "%tr0";
    if (hasConstraints())
      trace = emitStartConstraintTrace("%x0", "%f0", velocities, trace);
    os << "  call @mdrtWriteEnergies(%start, %u0, %k0, %g0, " << trace
       << ")\n"
       << "      : (i64, f64, f64, f64, f64) -> ()\n";
    // The state that the first scaling takes its pressure from (D92).
    if (scalesEveryStep())
      emitStoreTrotterState(
          "  ", trace,
          emitGroupTrace("  ", trace, "%x0", velocities, "%cell", "%m",
                         "%r_", "s0"),
          emitKineticWithoutCenter("  ", velocities, "%m", "s0"));

    if (isLeapfrog()) {
      // v(-dt/2) = v(0) - (dt/2) F(0) / m.
      os << "  %back = arith.constant -5.0e-01 : f64\n"
         << "  %behind = arith.mulf %back, %dt : f64\n"
         << "  %v0 = dyn.kick %vg, %f0, %m, %behind : !vec\n";
    }
  }

  if (isRestart() && scalesEveryStep() && !system.barostatState.empty()) {
    // The state that the first scaling takes its pressure from, as the
    // checkpoint keeps it (D92).
    std::string names[3];
    for (int k = 0; k != 3; ++k) {
      names[k] = "%bstate" + std::to_string(k);
      os << "  " << names[k] << " = arith.constant "
         << formatReal(system.barostatState[k]) << " : f64\n";
    }
    emitStoreTrotterState("  ", names[0], names[1], names[2]);
  } else if (isRestart() && scalesEveryStep()) {
    // A checkpoint of a run that did not scale every step holds no virial:
    // the state that the first scaling takes its pressure from is evaluated
    // once (D92). With leapfrog the stored velocities are half a kick
    // behind those of the time of the positions.
    StringRef raw = hasSites() ? "e" : "";
    std::string held = hasRestraints() ? "p" : "";
    os << "  %ubs" << held << ", %fbs" << held << raw << ", %wbs" << held << raw
       << " = md.evaluate @energy(%x0, %cell" << getFieldValues() << ")\n"
       << "      request [energy, forces, virial]\n"
       << "      : (!vec, !md.cell" << getFieldTypes()
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
    emitTrace(os, "%trs", virial, "  ");
    std::string trace = "%trs";
    if (hasConstraints())
      trace = emitStartConstraintTrace("%x0", "%fbs", current, trace);
    emitStoreTrotterState(
        "  ", trace,
        emitGroupTrace("  ", trace, "%x0", current, "%cell", "%m", "%r_",
                       "s0"),
        emitKineticWithoutCenter("  ", current, "%m", "s0"));
  }

  emitLevel(0, "  ");
  os << "  mdrt.host_call @mdrtFinish(%xe0, %ve0, " << idName
     << ") : (!vec, !vec, !ids)\n"
     << "  return\n}\n";
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
  // A pair is taken once, in the minimum image, which holds every image
  // within the cutoff only while the cell is wider than twice the cutoff.
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
    break;
  case Precision::Double:
    program.state = program.mass = Element::F64;
    program.force = program.parameter = Element::F64;
    break;
  }
  // Velocity Verlet begins a step with the forces of the step before.
  program.writesForces = !control.minimize;
  program.takesForces = isRestart() && !control.minimize;

  program.skin =
      (control.pairlistDistance - control.cutoffDistance) * units::length;
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

  // The loops of the schedule; a minimization has its own
  // (emitMinimization).
  if (!control.minimize)
    setSchedule();

  if (system.topology) {
    if (llvm::Error error = collectTopology())
      return error;
  } else if (llvm::Error error = collectParameters()) {
    return error;
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
            {"term_excluded", CoulombExcluded},
            {"term_reciprocal", CoulombReciprocal}})
        emitTopologyPotential(name, term);
  }
  else if (llvm::Error error = emitPotential())
    return error;
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
