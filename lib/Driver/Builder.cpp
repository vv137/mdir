// Builds the program of a run.

#include "mdir/Driver/Builder.h"

#include "mdir/Driver/Expression.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

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
  void emitPrograms();
  void emitEntry();

  /// Emits the loops of the schedule, from `level` inward, and returns the
  /// values that the loop of `level` results in.
  void emitLevel(unsigned level, StringRef indent);

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
    text += ", %t_" + table.name + ": !table";
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
}

std::string Builder::getFieldTypes() const {
  std::string text;
  for (const Program::Field &field : program.fields)
    text += ", " + getFieldType(field).str();
  for (size_t i = 0, e = program.tables.size(); i != e; ++i)
    text += ", !table";
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
  //   E = −(2π / 3V) Σ_ab N_a (N_b − δ_ab) C6_ab / rc³        W = 6 E
  //
  // for each pair of types a and b, so that the pressure changes by 2E / V.
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
        energy += -2.0 * M_PI / (3.0 * volume) * pairs * far /
                  (rc * rc * rc);
      }
  }
  program.dispersionEnergy = energy * units::energy;
  program.dispersionVirial = 6.0 * energy * units::energy;
  return llvm::Error::success();
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

  if (isLeapfrog()) {
    // The stored velocities are half a step behind the positions.
    os << "dyn.program @step(%x: !vec, %v: !vec, %m: !real, %cell: !md.cell, "
          "%dt: f64"
       << getFieldParameters() << ")\n    -> (!vec, !vec)\n"
       << "    attributes {velocity_offset = -0.5,\n"
       << "                provides = [\"symplectic\", "
          "\"time_reversible\"]} {\n"
       << "  %f = md.evaluate @energy(%x, %cell" << getFieldValues()
       << ") request [forces]\n      : " << signature << " -> !vec\n"
       << "  %v1 = dyn.kick %v, %f, %m, %dt : !vec\n"
       << "  %x1 = dyn.drift %x, %v1, %dt : !vec\n"
       << "  dyn.return %x1, %v1 : !vec, !vec\n}\n\n";
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
       << "  %x1 = dyn.drift %x, %v1, %dt : !vec\n";
    if (withEnergy)
      os << "  %u1, %f1, %w1 = " << evaluate
         << "\n      request [energy, forces, virial]\n"
         << "      : " << signature << " -> (f64, !vec, vector<9xf64>)\n";
    else
      os << "  %f1 = " << evaluate << " request [forces]\n"
         << "      : " << signature << " -> !vec\n";
    os << "  %v2 = dyn.kick %v1, %f1, %m, %half : !vec\n";
    if (withEnergy)
      os << "  dyn.return %x1, %v2, %f1, %u1, %w1\n"
         << "      : !vec, !vec, !vec, f64, vector<9xf64>\n";
    else
      os << "  dyn.return %x1, %v2, %f1 : !vec, !vec, !vec\n";
    os << "}\n\n";
  }
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
     << indent << "  md.yield %g : f64\n"
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

    // The number of the step that has just been taken.
    auto emitStep = [&]() {
      std::string counted;
      for (unsigned i = 0; i <= level; ++i) {
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
      os << inner << "%done" << here << " = arith.addi " << counted
         << ", %c1 : index\n";
      os << inner << "%steps" << here << " = arith.muli %done" << here
         << ", %per" << here << " : index\n";
      os << inner << "%since" << here << " = arith.index_cast %steps"
         << here << " : index to i64\n";
      os << inner << "%step" << here << " = arith.addi %since" << here
         << ", %start : i64\n";
    };

    if (current.name == "energy") {
      // The last step of the interval, and the energies after it.
      if (isLeapfrog()) {
        os << inner << "%xl, %vl = dyn.step @step(%x" << last << ", %v"
           << last << ", " << massName << ", %cell, %dt"
           << getFieldValues(fieldPrefix) << ")\n"
           << inner << "    : (!vec, !vec, !real, !md.cell, f64"
           << getFieldTypes() << ") -> (!vec, !vec)\n";
        os << inner << "%u, %fl, %w = md.evaluate @energy(%xl, %cell"
           << getFieldValues(fieldPrefix) << ")\n"
           << inner << "    request [energy, forces, virial]\n"
           << inner << "    : (!vec, !md.cell" << getFieldTypes()
           << ") -> (f64, !vec, vector<9xf64>)\n";
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
      emitForceSquare(os, "%g", "%fl", massName, inner);
      emitTrace(os, "%tr", "%w", inner);
      emitStep();
      os << inner << "func.call @mdrtWriteEnergies(%step" << here
         << ", %u, %k, %g, %tr) : (i64, f64, f64, f64, f64) -> ()\n";
      os << inner << "scf.yield " << getValues("l") << " : " << state << "\n";
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
  }
}

void Builder::emitEntry() {
  StringRef state = getName(program.state);
  StringRef force = getName(program.force);
  StringRef mass = getName(program.mass);
  StringRef parameter = getName(program.parameter);

  os << "func.func private @mdrtWriteEnergies(i64, f64, f64, f64, f64)\n"
     << "    attributes {llvm.emit_c_interface}\n"
     << "func.func private @mdrtWriteFrame(i64, memref<?x3x" << state
     << ">, memref<?xi32>)\n    attributes {llvm.emit_c_interface}\n"
     << "func.func private @mdrtFinish(memref<?x3x" << state
     << ">, memref<?x3x" << state << ">, memref<?xi32>)\n"
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
  int64_t steps = 1;
  for (unsigned i = levels.size(); i-- != 0;) {
    os << "  %per" << i << " = arith.constant " << steps << " : index\n";
    steps *= levels[i].count;
    if (i != 0 && levels[i - 1].name == "energy")
      steps += 1;
  }

  // The fields as the buffers hold them, and in the order of the positions
  // if the run keeps that order.
  bool givenForces = isRestart() && program.takesForces;
  std::string velocities = isLeapfrog() && !isRestart() ? "%vg" : "%v0";
  std::string given = program.reorders ? "_in" : "";
  os << "  %cell = md.orthorhombic_cell %lx, %ly, %lz\n"
     << "  %x" << (program.reorders ? "_in" : "0")
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
       << " : memref<?x?xf64> to !table\n";
  os << "  %id" << given
     << " = mdrt.from_buffer %identities : memref<?xi32> to !ids\n";
  // A run that continues an earlier one has the forces of the step before.
  if (givenForces)
    os << "  %f" << (program.reorders ? "_in" : "0")
       << " = mdrt.from_buffer %forces : memref<?x3x" << force
       << "> to !vec\n";
  if (program.reorders)
    emitReorder("  ", "_in", "0", "", givenForces, velocities);

  if (isLeapfrog())
    os << "  %c_half = arith.constant 5.0e-01 : f64\n"
       << "  %half_dt = arith.mulf %c_half, %dt : f64\n";

  if (!isRestart()) {
    // The energies at the start.
    os << "  %u0, %f0, %w0 = md.evaluate @energy(%x0, %cell"
       << getFieldValues() << ")\n"
       << "      request [energy, forces, virial]\n"
       << "      : (!vec, !md.cell" << getFieldTypes()
       << ") -> (f64, !vec, vector<9xf64>)\n";
    emitKineticEnergy(os, "%k0", velocities, "%m", "  ");
    emitForceSquare(os, "%g0", "%f0", "%m", "  ");
    emitTrace(os, "%tr0", "%w0", "  ");
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
    // Half as many again as a uniform density gives, and a few more.
    double volume = system.box[0] * system.box[1] * system.box[2];
    double reach = control.pairlistDistance * units::length;
    double expected = static_cast<double>(system.getNumParticles()) / volume *
                      4.0 / 3.0 * M_PI * reach * reach * reach;
    int64_t width = static_cast<int64_t>(std::ceil(1.5 * expected)) + 16;
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
  if (control.energyPeriod > 0) {
    levels.push_back({"energy", steps / control.energyPeriod});
    steps = control.energyPeriod;
    stepsPerEnergy = steps;
    // The last step of the interval is taken after the loop over steps.
    levels.push_back({"step", steps - 1});
  } else {
    levels.push_back({"step", steps});
  }

  if (llvm::Error error = collectParameters())
    return error;

  os << "!vec   = !md.field<@atoms, 3 x f64>\n"
     << "!real  = !md.field<@atoms, f64>\n"
     << "!ids   = !md.field<@atoms, i32>\n"
     << "!table = !md.table<2, f64, symmetric>\n"
     << "!pairs = !md.relation<@atoms, 2, unordered>\n\n"
     << "md.particle_set @atoms\n\n";
  if (llvm::Error error = emitPotential())
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
