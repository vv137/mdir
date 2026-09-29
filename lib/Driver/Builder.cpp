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
    AllTerms = 255,
  };
  /// Emits the potential `name` of the terms `terms` of the topology.
  void emitTopologyPotential(StringRef name, unsigned terms);
  void emitPrograms();
  void emitEntry();

  /// Emits the loops of the schedule, from `level` inward, and returns the
  /// values that the loop of `level` results in.
  void emitLevel(unsigned level, StringRef indent);
  /// Whether the topology has virtual sites.
  bool hasSites() const { return !getSiteSets().empty(); }
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

  /// Emits the coupling of the velocities `velocities` at the end of the
  /// step `step`: the removal of the motion of the center of mass and the
  /// thermostat. Returns the name of the velocities after it.
  std::string emitCoupling(StringRef indent, StringRef velocities,
                           StringRef tag, StringRef step);

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
  int64_t stepsPerEnergy = 0;
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
     << from << ", %cell, %id" << from << " width("
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
    return makeError("'dispersion_corr' needs a plain cutoff: 'switchdist' "
                     "equal to 'cutoffdist', and no 'vdw_shift' or "
                     "'vdw_force_switch'");

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
         << "    %fqq = arith.mulf %f, %qq : f64\n"
         << "    %coulomb = arith.divf %fqq, %r : f64\n";
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
  std::string evaluate = "md.evaluate @energy(%x1, %cell" + getFieldValues() +
                         ")";
  std::string signature = "(!vec, !md.cell" + getFieldTypes() + ")";
  // Virtual sites are placed after the positions move, and the forces on
  // them are moved to their atoms before the velocities do. Rigid waters
  // are constrained before the sites are placed.
  bool sites = hasSites();
  bool settles = hasSettles();

  if (isLeapfrog()) {
    // The stored velocities are half a step behind the positions.
    os << "dyn.program @step(%x: !vec, %v: !vec, %m: !real, %cell: !md.cell, "
          "%dt: f64"
       << getFieldParameters() << ")\n    -> (!vec, !vec)\n"
       << "    attributes {velocity_offset = -0.5,\n"
       << "                provides = [\"symplectic\", "
          "\"time_reversible\"]} {\n"
       << "  %f" << (sites ? "e" : "") << " = md.evaluate @energy(%x, %cell"
       << getFieldValues() << ") request [forces]\n      : " << signature
       << " -> !vec\n";
    if (sites)
      emitSpreadSites("  ", "%x", "%fe", "%f", "%r_");
    os << "  %v1 = dyn.kick %v, %f, %m, %dt : !vec\n"
       << "  %x1" << (sites ? "d" : "") << " = dyn.drift %x, %v1, %dt : !vec\n";
    if (sites)
      emitPlaceSites("  ", "%x1d", "%x1", "%r_");
    os << "  dyn.return %x1, %v1 : !vec, !vec\n}\n\n";
    return;
  }

  // Velocity Verlet (Swope et al., J. Chem. Phys. 76, 637 (1982)), with and
  // without the energy of the new positions.
  for (bool withEnergy : {false, true}) {
    os << "dyn.program @" << (withEnergy ? "step_energy" : "step")
       << "(%x: !vec, %v: !vec, %f: !vec, %m: !real,\n"
       << "    %cell: !md.cell, %dt: f64" << getFieldParameters() << ")\n"
       << "    -> (!vec, !vec, !vec"
       << (withEnergy ? ", f64, vector<9xf64>" : "") << ")\n"
       << "    attributes {provides = [\"symplectic\", "
          "\"time_reversible\"]} {\n"
       << "  %c = arith.constant 5.0e-01 : f64\n"
       << "  %half = arith.mulf %c, %dt : f64\n"
       << "  %v1 = dyn.kick %v, %f, %m, %half : !vec\n"
       << "  %x1" << (sites || settles ? "d" : "")
       << " = dyn.drift %x, %v1, %dt : !vec\n";
    // The rigid waters back in shape, and the velocities that take them
    // there over the step (the first half of RATTLE).
    std::string velocities = "%v1";
    if (settles) {
      emitSettlePositions("  ", "%x", "%x1d", "%dx1", sites ? "%x1s" : "%x1");
      os << "  %one = arith.constant 1.0 : f64\n"
         << "  %rate = arith.divf %one, %dt : f64\n"
         << "  %v1c = md.map_particles gather(%v1, %dx1 : !vec, !vec) {\n"
         << "  ^bb0(%vs_v: vector<3xf64>, %vs_d: vector<3xf64>):\n"
         << "    %vs_r = vector.broadcast %rate : f64 to vector<3xf64>\n"
         << "    %vs_dv = arith.mulf %vs_r, %vs_d : vector<3xf64>\n"
         << "    %vs_sum = arith.addf %vs_v, %vs_dv : vector<3xf64>\n"
         << "    md.yield %vs_sum : vector<3xf64>\n"
         << "  } : !vec\n";
      velocities = "%v1c";
    }
    if (sites)
      emitPlaceSites("  ", settles ? "%x1s" : "%x1d", "%x1", "%r_");
    StringRef raw = sites ? "e" : "";
    if (withEnergy)
      os << "  %u1, %f1" << raw << ", %w1" << raw << " = " << evaluate
         << "\n      request [energy, forces, virial]\n"
         << "      : " << signature << " -> (f64, !vec, vector<9xf64>)\n";
    else
      os << "  %f1" << raw << " = " << evaluate << " request [forces]\n"
         << "      : " << signature << " -> !vec\n";
    std::string virial = "%w1";
    if (sites)
      virial = emitSpreadSites("  ", "%x1", "%f1e", "%f1", "%r_",
                               withEnergy ? "%w1e" : "", "%w1");
    os << "  %v2" << (settles ? "u" : "") << " = dyn.kick " << velocities
       << ", %f1, %m, %half : !vec\n";
    // No velocity along a rigid bond (the second half of RATTLE).
    if (settles)
      virial = emitSettleVelocities("  ", "%x1", "%v2u", "%v2",
                                    withEnergy ? virial : "", "%w1c");
    if (withEnergy)
      os << "  dyn.return %x1, %v2, %f1, %u1, " << virial << "\n"
         << "      : !vec, !vec, !vec, f64, vector<9xf64>\n";
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
  SiteKernel(llvm::raw_ostream &os, StringRef indent)
      : os(os), indent(indent.str()) {}

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
  std::string fresh() { return "%vs" + std::to_string(next++); }

  llvm::raw_ostream &os;
  std::string indent;
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
    // G = 2 m Δv / dt, and their virial Σ (x_i − x_O) ⊗ G_i.
    std::string sum = (virialResult + "_settle").str();
    os << indent << sum;
    header("sum_tuples", "vector<9xf64>");
    SiteKernel w(os, inner);
    std::array<std::string, 3> change = emitKernel(w);
    std::string factor = w.real(
        "divf", w.real("mulf", w.constant(2.0), "%vs_m1"), "%dt");
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
  // The state that the loops carry: positions and velocities, and with
  // velocity Verlet the forces.
  std::string state = isLeapfrog() ? "!vec, !vec" : "!vec, !vec, !vec";
  auto getValues = [&](const llvm::Twine &suffix) {
    std::string x = ("%x" + suffix).str(), v = ("%v" + suffix).str(),
                f = ("%f" + suffix).str();
    return isLeapfrog() ? x + ", " + v : x + ", " + v + ", " + f;
  };
  auto getInits = [&](const llvm::Twine &inside, const llvm::Twine &outside) {
    std::string text;
    for (StringRef name : {"x", "v", "f"}) {
      if (name == "f" && isLeapfrog())
        continue;
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

  std::string outerMass = massName, outerPrefix = fieldPrefix,
              outerId = idName;
  if (reorders) {
    emitReorder(inner, "a" + here, "s", "s", /*withForces=*/!isLeapfrog(),
                "%vs");
    massName = "%ms";
    fieldPrefix = "%ps_";
    idName = "%ids";
  }

  if (isStepLoop) {
    os << inner << getValues("b" + here) << " = dyn.step @step(";
    if (isLeapfrog())
      os << "%xa" << here << ", %va" << here << ", " << massName
         << ", %cell, %dt" << getFieldValues(fieldPrefix) << ")\n"
         << inner << "    : (!vec, !vec, !real, !md.cell, f64"
         << getFieldTypes() << ") -> (!vec, !vec)\n";
    else
      os << "%xa" << here << ", %va" << here << ", %fa" << here << ", "
         << massName << ", %cell, %dt" << getFieldValues(fieldPrefix)
         << ")\n"
         << inner << "    : (!vec, !vec, !vec, !real, !md.cell, f64"
         << getFieldTypes() << ") -> (!vec, !vec, !vec)\n";
    os << inner << "scf.yield " << getValues("b" + here) << " : " << state
       << "\n";
  } else {
    emitLevel(level + 1, inner);
    std::string below = std::to_string(level + 1);
    std::string last = "e" + below;

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
    auto getCoupled = [&](StringRef x, StringRef v, StringRef f) {
      std::string coupled = emitCoupling(inner, v, here, "%step" + here);
      return isLeapfrog() ? (x + ", " + coupled).str()
                          : (x + ", " + coupled + ", " + f).str();
    };
    bool couplesBelow = levels[level + 1].name == "couple";

    if (current.name == "couple") {
      // The last step of the period, and the coupling after it.
      os << inner << getValues("k" + here) << " = dyn.step @step(";
      if (isLeapfrog())
        os << "%x" << last << ", %v" << last << ", " << massName
           << ", %cell, %dt" << getFieldValues(fieldPrefix) << ")\n"
           << inner << "    : (!vec, !vec, !real, !md.cell, f64"
           << getFieldTypes() << ") -> (!vec, !vec)\n";
      else
        os << "%x" << last << ", %v" << last << ", %f" << last << ", "
           << massName << ", %cell, %dt" << getFieldValues(fieldPrefix)
           << ")\n"
           << inner << "    : (!vec, !vec, !vec, !real, !md.cell, f64"
           << getFieldTypes() << ") -> (!vec, !vec, !vec)\n";
      emitStep();
      std::string coupled =
          getCoupled("%xk" + here, "%vk" + here, "%fk" + here);
      os << inner << "scf.yield " << coupled << " : " << state << "\n";
      os << indent << "}\n";
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
      if (isLeapfrog())
        os << "%xp" << here << ", %vp" << here << ", " << massName
           << ", %cell, %dt" << getFieldValues(fieldPrefix) << ")\n"
           << inner << "      : (!vec, !vec, !real, !md.cell, f64"
           << getFieldTypes() << ") -> (!vec, !vec)\n";
      else
        os << "%xp" << here << ", %vp" << here << ", %fp" << here << ", "
           << massName << ", %cell, %dt" << getFieldValues(fieldPrefix)
           << ")\n"
           << inner << "      : (!vec, !vec, !vec, !real, !md.cell, f64"
           << getFieldTypes() << ") -> (!vec, !vec, !vec)\n";
      os << inner << "  scf.yield " << getValues("r" + here) << " : "
         << state << "\n"
         << inner << "}\n";
      last = "q" + here;
    }

    if (current.name == "energy") {
      // The last step of the interval, and the energies after it.
      std::string virialName = "%w";
      if (isLeapfrog()) {
        os << inner << "%xl, %vl = dyn.step @step(%x" << last << ", %v"
           << last << ", " << massName << ", %cell, %dt"
           << getFieldValues(fieldPrefix) << ")\n"
           << inner << "    : (!vec, !vec, !real, !md.cell, f64"
           << getFieldTypes() << ") -> (!vec, !vec)\n";
        StringRef raw = hasSites() ? "e" : "";
        os << inner << "%u, %fl" << raw << ", %w" << raw
           << " = md.evaluate @energy(%xl, %cell"
           << getFieldValues(fieldPrefix) << ")\n"
           << inner << "    request [energy, forces, virial]\n"
           << inner << "    : (!vec, !md.cell" << getFieldTypes()
           << ") -> (f64, !vec, vector<9xf64>)\n";
        if (hasSites())
          virialName = emitSpreadSites(
              inner, "%xl", "%fle", "%fl",
              ("%r" + StringRef(fieldPrefix).drop_front(2)).str(), "%we",
              "%w");
        // The stored velocities are half a step behind. Those of the time
        // of the positions are half a kick ahead of them.
        os << inner << "%vn = dyn.kick %vl, %fl, " << massName
           << ", %half_dt : !vec\n";
      } else {
        os << inner << "%xl, %vl, %fl, %u, %w = dyn.step @step_energy(%x"
           << last
           << ", %v" << last << ", %f" << last << ", " << massName
           << ", %cell, %dt" << getFieldValues(fieldPrefix) << ")\n"
           << inner << "    : (!vec, !vec, !vec, !real, !md.cell, f64"
           << getFieldTypes()
           << ") -> (!vec, !vec, !vec, f64, vector<9xf64>)\n";
      }
      emitKineticEnergy(os, "%k", isLeapfrog() ? "%vn" : "%vl", massName,
                        inner);
      // With constraints the forces do not give the kinetic energies of
      // the half steps (Section 9 of design-m1.md); the log takes that of
      // the step.
      if (hasSettles())
        os << inner << "%g = arith.constant 0.0 : f64\n";
      else
        emitForceSquare(os, "%g", "%fl", massName, inner);
      emitTrace(os, "%tr", virialName, inner);
      emitStep();
      os << inner << "func.call @mdrtWriteEnergies(%step" << here
         << ", %u, %k, %g, %tr) : (i64, f64, f64, f64, f64) -> ()\n";
      std::string yielded =
          couplesBelow ? getCoupled("%xl", "%vl", "%fl") : getValues("l");
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
  // What the loop has left is in the order of its last iteration.
  if (reorders) {
    massName = "%me" + here;
    fieldPrefix = "%pe" + here + "_";
    idName = "%ide" + here;
    emitRenumber(indent, idName, "%re" + here + "_");
  }
}

std::string Builder::emitCoupling(StringRef indent, StringRef velocities,
                                  StringRef tag, StringRef step) {
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
  if (control.thermostat) {
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
  os << indent << "  md.yield " << value << " : vector<3xf64>\n"
     << indent << "} : !vec\n";
  return "%vc" + t;
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
  if (control.thermostat)
    os << "func.func private @mdrtBussiFactor(i64, i64, f64, f64, f64, f64) "
          "-> f64\n";
  if (control.getCouplingPeriod() > 0)
    os << "func.func private @mdrtAddBath(f64)\n"
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
  // well, and one over energy intervals a whole period after its loop over
  // periods.
  int64_t steps = 1;
  for (unsigned i = levels.size(); i-- != 0;) {
    os << "  %per" << i << " = arith.constant " << steps << " : index\n";
    steps *= levels[i].count;
    if (i == 0)
      continue;
    if (levels[i - 1].name == "couple")
      steps += 1;
    else if (levels[i - 1].name == "energy")
      steps += levels[i].name == "couple" ? control.getCouplingPeriod() : 1;
  }

  // The fields as the buffers hold them, and in the order of the positions
  // if the run keeps that order.
  bool givenForces = isRestart() && program.takesForces;
  std::string velocities = isLeapfrog() && !isRestart() ? "%vg" : "%v0";
  std::string given = program.reorders ? "_in" : "";
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
      os << "  %seed = arith.constant " << static_cast<int64_t>(control.seed)
         << " : i64\n"
         << "  %freedom = arith.constant " << formatReal(freedom) << " : f64\n"
         << "  %target_kinetic = arith.constant " << formatReal(target)
         << " : f64\n"
         << "  %decay = arith.constant " << formatReal(decay) << " : f64\n";
    }
  }

  if (isLeapfrog())
    os << "  %c_half = arith.constant 5.0e-01 : f64\n"
       << "  %half_dt = arith.mulf %c_half, %dt : f64\n";

  if (!isRestart()) {
    // The energies at the start.
    StringRef raw = hasSites() ? "e" : "";
    os << "  %u0, %f0" << raw << ", %w0" << raw
       << " = md.evaluate @energy(%x0, %cell" << getFieldValues() << ")\n"
       << "      request [energy, forces, virial]\n"
       << "      : (!vec, !md.cell" << getFieldTypes()
       << ") -> (f64, !vec, vector<9xf64>)\n";
    std::string virial = "%w0";
    if (hasSites())
      virial = emitSpreadSites("  ", "%x0", "%f0e", "%f0", "%r_", "%w0e",
                               "%w0");
    if (system.topology) {
      os << "  %terms = memref.alloca() : memref<8xf64>\n";
      int index = 0;
      for (StringRef name :
           {"term_lj", "term_coulomb", "term_bonds", "term_angles",
            "term_dihedrals", "term_lj14", "term_coulomb14", "term_cmap"}) {
        os << "  %" << name << " = md.evaluate @" << name << "(%x0, %cell"
           << getFieldValues() << ") request [energy]\n"
           << "      : (!vec, !md.cell" << getFieldTypes() << ") -> f64\n"
           << "  %i_" << name << " = arith.constant " << index++
           << " : index\n"
           << "  memref.store %" << name << ", %terms[%i_" << name
           << "] : memref<8xf64>\n";
      }
      os << "  %terms_cast = memref.cast %terms : memref<8xf64> to "
            "memref<?xf64>\n"
         << "  call @mdrtWriteTerms(%terms_cast) : (memref<?xf64>) -> ()\n";
    }
    emitKineticEnergy(os, "%k0", velocities, "%m", "  ");
    if (hasSettles())
      os << "  %g0 = arith.constant 0.0 : f64\n";
    else
      emitForceSquare(os, "%g0", "%f0", "%m", "  ");
    emitTrace(os, "%tr0", virial, "  ");
    os << "  call @mdrtWriteEnergies(%start, %u0, %k0, %g0, %tr0)\n"
       << "      : (i64, f64, f64, f64, f64) -> ()\n";

    if (isLeapfrog()) {
      // v(-dt/2) = v(0) - (dt/2) F(0) / m.
      os << "  %back = arith.constant -5.0e-01 : f64\n"
         << "  %behind = arith.mulf %back, %dt : f64\n"
         << "  %v0 = dyn.kick %vg, %f0, %m, %behind : !vec\n";
    }
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

llvm::Error Builder::build() {
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
  program.writesForces = !isLeapfrog();
  program.takesForces = isRestart() && !isLeapfrog();

  program.skin =
      (control.pairlistDistance - control.cutoffDistance) * units::length;
  // Cells of half the reach of a neighbor structure: particles that are
  // neighbors are then a few cells apart in memory.
  program.reorders = control.reorder;
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

  // The loops of the schedule. A period of zero means no output of that
  // kind, and no loop for it.
  int64_t steps = control.numSteps;
  if (control.checkpointPeriod > 0) {
    levels.push_back({"segment", steps / control.checkpointPeriod});
    steps = control.checkpointPeriod;
  }
  if (control.framePeriod > 0) {
    levels.push_back({"frame", steps / control.framePeriod});
    steps = control.framePeriod;
  }
  // The velocities are coupled at the end of the last step of a period of
  // coupling, which is taken after the loop over steps, as the last step
  // of an interval between energies is. The loop over the periods of an
  // interval between energies leaves the last period to the interval,
  // which ends it with its step of energy.
  int64_t coupling = control.getCouplingPeriod();
  if (control.energyPeriod > 0) {
    levels.push_back({"energy", steps / control.energyPeriod});
    steps = control.energyPeriod;
    stepsPerEnergy = steps;
    if (coupling > 0) {
      levels.push_back({"couple", steps / coupling - 1});
      levels.push_back({"step", coupling - 1});
    } else {
      levels.push_back({"step", steps - 1});
    }
  } else if (coupling > 0) {
    levels.push_back({"couple", steps / coupling});
    levels.push_back({"step", coupling - 1});
  } else {
    levels.push_back({"step", steps});
  }

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
       << ") orientation(" << set.getOrientation() << ")\n";
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
            {"term_cmap", CMaps}})
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
