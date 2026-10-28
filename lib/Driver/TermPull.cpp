// Pull-coordinate potentials and output, registered under D359 (#683).
#include "BuilderImpl.h"

using namespace mdir::driver;
using namespace mdir::driver::builder;

namespace mdir::driver::builder {
class PullTermBuilder final : public TermBuilder, public TermCoordinateOutput {
public:
  PullTermBuilder() { roles.coordinateOutput = this; }
  bool rejectsTunables(const TermInputs &context) const override {
    return hasCoordinateOutput(context);
  }
  bool hasCoordinateOutput(const TermInputs &context) const override {
    return !getColumns(context).empty();
  }
  void emitAuxiliaryPotentials(TermEmission &context) const override;
  void emitOutput(TermEmission &context, const TermOutput &point) const override {
    if (getColumns(context).empty())
      return;
    std::string time = point.time.str();
    if (point.deriveTime) {
      // Preserve the coordinates at the end of the step and its time (D145).
      time = ("%pull_time" + point.loopSuffix).str();
      if (context.getControl().usesTime)
        context.getOutput()
            << point.indent << time << "_steps = arith.sitofp " << point.step
            << " : i64 to f64\n"
            << point.indent << time << " = arith.mulf " << time << "_steps, %dt"
            << " : f64\n";
    }
    emitPullOutput(context, point.indent, point.x, point.cell, point.prefix,
                   point.step, time);
  }

private:
  static std::vector<std::pair<size_t, std::string>> getColumns(const TermInputs &);
  static void emitPullOutput(TermEmission &, StringRef indent, StringRef x,
                            StringRef cell, StringRef prefix, StringRef step,
                            StringRef time);
};
} // namespace mdir::driver::builder

std::unique_ptr<TermBuilder> mdir::driver::builder::createPullTermBuilder() {
  return std::make_unique<PullTermBuilder>();
}

std::vector<std::pair<size_t, std::string>>
PullTermBuilder::getColumns(const TermInputs &context) {
  const Control &control = context.getControl();
  const System &system = context.getSystem();
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

void PullTermBuilder::emitAuxiliaryPotentials(TermEmission &context) const {
  const Control &control = context.getControl();
  const System &system = context.getSystem();
  llvm::raw_ostream &os = context.getOutput();
  // The energy of each term as a function of its coordinates, which follow
  // the other arguments, so that the derivatives with respect to them give
  // the forces along the coordinates.
  std::vector<std::pair<size_t, std::string>> columns = getColumns(context);
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
       << context.getFieldParameters() << context.getTimeParameter();
    for (size_t q = 0; q != quantities.size(); ++q)
      os << ", %pq" << q << ": f64";
    os << ") -> f64 {\n";
    context.emitLambdaConstants("  ");
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
    context.bindLambdas(values);
    Expression expression = llvm::cantFail(
        Expression::parse(term.expression, control.functions));
    std::string energy =
        emitExpressionIR(context, expression, values, "%pe", "  ", term.name);
    os << "  md.return " << energy << " : f64\n}\n\n";
  }
}

void PullTermBuilder::emitPullOutput(TermEmission &context, StringRef indent,
                             StringRef x, StringRef cell,
                             StringRef prefix, StringRef step,
                             StringRef time) {
  const System &system = context.getSystem();
  llvm::raw_ostream &os = context.getOutput();
  std::vector<std::pair<size_t, std::string>> columns = getColumns(context);
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
    llvm::StringMap<std::string> coordinates = context.emitCentroidCoordinates(
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
  unsigned base = context.getNumPotentialArguments();
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
       << context.getFieldValues(prefix) << context.getTimeValue(time);
    for (const std::string &coordinate : coordinates)
      os << ", " << coordinate;
    os << ")\n" << indent << "    request [energy";
    for (size_t q = 0; q != coordinates.size(); ++q)
      os << ", derivative(" << base + q << ")";
    os << "]\n" << indent << "    : (!vec, !md.cell" << context.getFieldTypes()
       << context.getTimeType();
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

