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
    text += ", %p_" + field.name + ": !real";
  return text;
}

std::string Builder::getFieldValues(StringRef prefix) const {
  std::string text;
  for (const Program::Field &field : program.fields)
    text += (", " + prefix + field.name).str();
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
            "%p" + from + "_" + field.name, "!real");
  permute("%id" + otherTo, "%id" + from, "!ids");
}

std::string Builder::getFieldTypes() const {
  std::string text;
  for (size_t i = 0, e = program.fields.size(); i != e; ++i)
    text += ", !real";
  return text;
}

llvm::Error Builder::collectParameters() {
  for (const PairTerm &term : control.pairs) {
    auto expression = Expression::parse(term.expression);
    if (!expression)
      return expression.takeError();

    for (const std::string &name : expression->getNames()) {
      if (name == "r")
        continue;
      if (llvm::any_of(term.constants,
                       [&](auto &constant) { return constant.first == name; }))
        continue;
      if (llvm::any_of(parameters,
                       [&](Parameter &known) { return known.name == name; }))
        continue;

      // A parameter of the types. Every type must have it.
      Parameter parameter;
      parameter.name = name;
      std::vector<double> values;
      for (const ParticleType &type : control.types) {
        auto found = llvm::find_if(type.parameters, [&](auto &entry) {
          return entry.first == name;
        });
        if (found == type.parameters.end())
          return llvm::createStringError(
              llvm::inconvertibleErrorCode(),
              "the expression of '%s' uses '%s', which is neither 'r', nor "
              "a number of the term, nor a parameter of the type '%s'",
              term.name.c_str(), name.c_str(), type.name.c_str());
        values.push_back(found->second);
      }
      parameter.value = values.front();
      parameter.isUniform = llvm::all_of(
          values, [&](double value) { return value == values.front(); });

      if (!parameter.isUniform) {
        if (!term.mixing.count(name))
          return llvm::createStringError(
              llvm::inconvertibleErrorCode(),
              "'%s' differs between the types, but the term '%s' has no "
              "rule of 'mixing' for it",
              name.c_str(), term.name.c_str());
        parameter.field = program.fields.size();
        Program::Field field;
        field.name = name;
        for (unsigned type : system.types)
          field.values.push_back(values[type]);
        program.fields.push_back(std::move(field));
      }
      parameters.push_back(std::move(parameter));
    }
    expressions.push_back(std::move(*expression));
  }
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
    std::string gathered, types, arguments;
    for (const Parameter &parameter : parameters) {
      if (parameter.isUniform ||
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
    for (const Parameter &parameter : parameters) {
      if (!llvm::is_contained(expression.getNames(), parameter.name))
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
      moreTypes += ", !real";
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
    os << ", %b_" << field.name << ": memref<?x" << parameter << ">";
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
       << field.name << " : memref<?x" << parameter << "> to !real\n";
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
