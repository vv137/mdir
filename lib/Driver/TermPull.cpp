// Pull-coordinate potentials and output, registered under D359 (#683).
#include "BuilderImpl.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mdir::driver;
using namespace mdir::driver::builder;

struct PullPotentialIRConfig {
  explicit PullPotentialIRConfig(ExpressionIR expression)
      : expression(std::move(expression)) {}
  ExpressionIR expression;
  std::string symbol, termName;
  std::vector<std::string> quantities;
  std::vector<std::pair<std::string, double>> lambdas, parameters;
  std::vector<std::pair<std::string, size_t>> argumentBindings, tableBindings;
};

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
  auto columns = getColumns(context);
  if (columns.empty())
    return;
  // The shared energy potential supplies the exact common argument types.
  // A recipe borrows those types only when it constructs the auxiliary IR.
  std::vector<std::string> commonNames = {"%x", "%cell"};
  for (const auto &field : context.getFields())
    commonNames.push_back("%p_" + field.name);
  for (const auto &table : context.getTables())
    commonNames.push_back("%t_" + table.name);
  for (const auto &set : context.getTupleSets()) {
    commonNames.push_back("%r_" + set.name);
    for (const auto &field : set.fields)
      commonNames.push_back("%f_" + set.name + "_" + field.name);
  }
  if (control.usesTime)
    commonNames.push_back("%time");
  auto argument = [&](StringRef name) -> size_t {
    auto found = llvm::find(commonNames, name);
    assert(found != commonNames.end() && "missing pull potential argument");
    return found - commonNames.begin();
  };
  for (auto [index, term] : llvm::enumerate(system.topology->tupleTerms)) {
    if (!term.isCentroid())
      continue;
    Expression expression =
        llvm::cantFail(Expression::parse(term.expression, control.functions));
    PullPotentialIRConfig config(expression.prepareIR());
    config.symbol = "pullterm" + std::to_string(index);
    config.termName = term.name;
    llvm::StringMap<std::string> variables;
    for (const auto &[k, quantity] : columns)
      if (k == index) {
        variables[quantity] = "";
        config.quantities.push_back(quantity);
      }
    for (const auto &parameter : term.parameters) {
      config.parameters.emplace_back(parameter.first, parameter.second.front());
      variables[parameter.first] = "";
    }
    if (control.usesTime)
      variables["t"] = "%time";
    context.bindLambdas(variables);
    for (const auto &[name, values] : control.freeEnergy.lambdas)
      config.lambdas.emplace_back(
          "lambda_" + name,
          control.freeEnergy.get(name, control.freeEnergy.state));
    llvm::SmallVector<StringRef> supplied;
    for (const auto &entry : variables)
      supplied.push_back(entry.getKey());
    auto inputs = config.expression.getInputs(supplied);
    for (const auto &name : inputs.names) {
      auto found = variables.find(name);
      assert(found != variables.end() && "missing pull expression binding");
      if (!found->second.empty() && found->second == "%time")
        config.argumentBindings.emplace_back(name, argument(found->second));
    }
    for (const auto &name : inputs.tables)
      config.tableBindings.emplace_back(name, argument("%t_" + name));
    context.emitIRFragment(
        "", {}, {}, {}, {},
        [config](mlir::OpBuilder &builder, mlir::ValueRange) -> TermIRValues {
          using namespace mlir;
          auto module =
              cast<ModuleOp>(builder.getInsertionBlock()->getParentOp());
          auto energy = module.lookupSymbol<mdir::md::PotentialOp>("energy");
          assert(energy && "missing shared energy potential");
          llvm::SmallVector<Type> types(energy.getFunctionType().getInputs());
          size_t coordinateBase = types.size();
          types.append(config.quantities.size(), builder.getF64Type());
          auto functionType =
              builder.getFunctionType(types, {builder.getF64Type()});
          auto loc = builder.getUnknownLoc();
          auto potential = mdir::md::PotentialOp::create(
              builder, loc, config.symbol, functionType, StringAttr{},
              ArrayAttr{}, ArrayAttr{});
          {
            OpBuilder::InsertionGuard guard(builder);
            llvm::SmallVector<Location> locations(types.size(), loc);
            Block *block =
                builder.createBlock(&potential.getBody(), {}, types, locations);
            auto args = block->getArguments();
            auto constant = [&](double value) -> Value {
              return arith::ConstantOp::create(builder, loc,
                                               builder.getF64FloatAttr(value));
            };
            llvm::StringMap<Value> bindings, tables, lambdas;
            // Constants retain their original order even when unused.
            for (const auto &[name, value] : config.lambdas)
              lambdas[name] = constant(value);
            for (auto [q, name] : llvm::enumerate(config.quantities))
              bindings[name] = args[coordinateBase + q];
            for (const auto &[name, value] : config.parameters)
              bindings[name] = constant(value);
            for (const auto &[name, slot] : config.argumentBindings)
              bindings[name] = args[slot];
            // Apply lambda bindings after parameters, as the textual emitter
            // did.
            for (const auto &entry : lambdas)
              bindings[entry.getKey()] = entry.getValue();
            for (const auto &[name, slot] : config.tableBindings)
              tables[name] = args[slot];
            Value result = config.expression.build(builder, bindings, tables,
                                                   config.termName);
            mdir::md::ReturnOp::create(builder, loc, ValueRange{result});
          }
          return {};
        });
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

