// The outputs of the pulls, the free energy, the observables, the
// tunable gradients, and the solvent (docs/driver.md, Section 2.12).

#include "BuilderImpl.h"

using namespace mdir::driver;
using namespace mdir::driver::builder;
namespace prolate = mdir::prolate;
namespace influence = mdir::influence;

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
      // That of a tunable is a value of the entry, which an update changes
      // without changing the text (D232).
      if (control.isObservedTunable(term.name, constant))
        value = "%obv_" + term.name + "_" + constant;
      else
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

void Builder::emitTunableGradientOutput(StringRef indent, StringRef x,
                                        StringRef cell, StringRef prefix,
                                        StringRef time) {
  if (!program.tunableGradient)
    return;
  // A loop of one iteration if the host asks (the start value
  // `tunable_gradient` is not 0) and none otherwise, as the branch of the
  // start is: a run takes none of it.
  std::string inner = (indent + "  ").str();
  size_t count = program.gradientSlots.size();
  std::string type = "memref<" + std::to_string(count + 1) + "xf64>";
  os << indent << "%tg_none = arith.constant 0.0 : f64\n"
     << indent << "%tg_asked = arith.cmpf one, %tunable_gradient, %tg_none"
                  " : f64\n"
     << indent << "%tg_trips = arith.select %tg_asked, %c1, %c0 : index\n"
     << indent << "scf.for %tg_i = %c0 to %tg_trips step %c1 {\n"
     << inner << "%tg_out = memref.alloca() : " << type << "\n";
  size_t fields = program.gradientFields.size();
  {
    unsigned base = getNumPotentialArguments();
    size_t constants = control.tunableConstants.size();
    std::string arguments,
        types = "(!vec, !md.cell" + getFieldTypes() + getTimeType();
    for (size_t c = 0; c != constants; ++c) {
      arguments += ", %tg_c" + std::to_string(c);
      types += ", f64";
    }
    // The fields of zeros through which the derivatives in the parameters
    // of tuples leave as fields of the particles: one field of the program
    // serves all.
    size_t seeds = getNumGradientSeeds();
    for (size_t c = 0; c != seeds; ++c) {
      arguments += (", " + prefix + "tg_zero").str();
      types += ", !real";
    }
    // The argument of `@tunable` that each field is the derivative in: the
    // fields of zeros, then the charges.
    unsigned chargeArgument = 2;
    for (const Program::Field &field : program.fields) {
      if (field.name == "q")
        break;
      ++chargeArgument;
    }
    auto getArgument = [&](size_t c) {
      return c < seeds ? base + constants + c : chargeArgument;
    };
    types += ")";
    os << inner << "%tg_e";
    for (size_t i = 0; i != count; ++i)
      os << ", %tg_r" << i;
    for (size_t c = 0; c != fields; ++c)
      os << ", %tg_f" << c;
    os << " = md.evaluate @tunable(" << x << ", " << cell
       << getFieldValues(prefix) << getTimeValue(time) << arguments << ")\n"
       << inner << "    request [energy";
    for (size_t i = 0; i != count; ++i)
      os << ", derivative(" << base + gradientColumns[i] << ")";
    for (size_t c = 0; c != fields; ++c)
      os << ", derivative(" << getArgument(c) << ")";
    program.gradientArguments.clear();
    for (size_t i = 0; i != count; ++i)
      program.gradientArguments.push_back(
          {base + gradientColumns[i], program.gradientSlots[i].tunable});
    for (size_t c = 0; c != fields; ++c)
      program.gradientArguments.push_back(
          {static_cast<unsigned>(getArgument(c)),
           program.gradientFields[c].tunable});
    os << "]\n" << inner << "    : " << types << " -> (f64";
    for (size_t i = 0; i != count; ++i)
      os << ", f64";
    for (size_t c = 0; c != fields; ++c)
      os << ", !real";
    os << ")\n";
    // The energy of that potential less that of the potential of the run,
    // which the host adds to the energy that the run reports.
    os << inner << "%tg_u = md.evaluate @energy(" << x << ", " << cell
       << getFieldValues(prefix) << getTimeValue(time) << ")\n"
       << inner << "    request [energy]\n"
       << inner << "    : (!vec, !md.cell" << getFieldTypes() << getTimeType()
       << ") -> f64\n"
       << inner << "%tg_shift = arith.subf %tg_e, %tg_u : f64\n"
       << inner << "%tg_last = arith.constant " << count << " : index\n"
       << inner << "memref.store %tg_shift, %tg_out[%tg_last] : " << type
       << "\n";
    for (size_t c = 0; c != fields; ++c)
      os << inner << "%tg_n" << c << " = arith.constant " << c << " : i64\n"
         << inner << "mdrt.host_call @mdrtWriteTunableGradientField(%tg_n"
         << c << ", %tg_f" << c << ", " << idName << ")\n"
         << inner << "    : (i64, !real, !ids)\n";
    for (size_t i = 0; i != count; ++i)
      os << inner << "%tg_s" << i << " = arith.constant " << i << " : index\n"
         << inner << "memref.store %tg_r" << i << ", %tg_out[%tg_s" << i
         << "] : " << type << "\n";
  }
  os << inner << "%tg_cast = memref.cast %tg_out : " << type
     << " to memref<?xf64>\n"
     << inner << "func.call @mdrtWriteTunableGradient(%tg_cast)"
                 " : (memref<?xf64>) -> ()\n"
     << indent << "}\n";
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
