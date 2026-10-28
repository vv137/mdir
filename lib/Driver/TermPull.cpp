// Pull-coordinate potentials and output, registered under D359 (#683).
#include "BuilderImpl.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
        context.emitIRFragment(
            point.indent, {time}, {"f64"}, {point.step, "%dt"}, {"i64", "f64"},
            [](mlir::OpBuilder &builder,
               mlir::ValueRange values) -> TermIRValues {
              auto loc = builder.getUnknownLoc();
              mlir::Value steps = mlir::arith::SIToFPOp::create(
                  builder, loc, builder.getF64Type(), values[0]);
              return {
                  mlir::arith::MulFOp::create(builder, loc, steps, values[1])};
            });
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
  auto columns = getColumns(context);
  if (columns.empty())
    return;
  std::string type = "memref<" + std::to_string(columns.size()) + "xf64>";
  std::string values = ("%pull_values" + step.drop_front()).str();
  std::string relations = ("%r" + prefix.drop_front(2)).str();
  context.emitIRFragment(
      indent, {values}, {type}, {}, {},
      [count = columns.size()](mlir::OpBuilder &builder,
                               mlir::ValueRange) -> TermIRValues {
        auto type =
            mlir::MemRefType::get({int64_t(count)}, builder.getF64Type());
        return {mlir::memref::AllocaOp::create(builder, builder.getUnknownLoc(),
                                               type)};
      });
  std::vector<std::string> names(columns.size());
  for (auto [index, term] : llvm::enumerate(system.topology->tupleTerms)) {
    if (!term.isCentroid())
      continue;
    auto coordinates = context.emitCentroidCoordinates(
        term, indent, x, cell, relations,
        values + "_c" + std::to_string(index) + "_");
    for (auto [column, entry] : llvm::enumerate(columns))
      if (entry.first == index)
        names[column] = coordinates.lookup(entry.second);
  }
  for (auto [column, name] : llvm::enumerate(names)) {
    context.emitIRFragment(
        indent, {}, {}, {values, name}, {type, "f64"},
        [column](mlir::OpBuilder &builder,
                 mlir::ValueRange operands) -> TermIRValues {
          auto loc = builder.getUnknownLoc();
          mlir::Value place =
              mlir::arith::ConstantIndexOp::create(builder, loc, column);
          mlir::memref::StoreOp::create(builder, loc, operands[1], operands[0],
                                        mlir::ValueRange{place});
          return {};
        });
  }
  // For two centers, F = -(dE/dr d/r + dE/dd); report its radial
  // component and Cartesian components. For an angle, report -dE/dtheta.
  std::string terms = ("%pull_terms" + step.drop_front()).str();
  unsigned numTerms = 0;
  for (const auto &term : system.topology->tupleTerms)
    numTerms += term.isCentroid() ? (term.arity == 2 ? 5 : 2) : 0;
  std::string termType = "memref<" + std::to_string(numTerms) + "xf64>";
  context.emitIRFragment(
      indent, {terms}, {termType}, {}, {},
      [numTerms](mlir::OpBuilder &builder, mlir::ValueRange) -> TermIRValues {
        auto type =
            mlir::MemRefType::get({int64_t(numTerms)}, builder.getF64Type());
        return {mlir::memref::AllocaOp::create(builder, builder.getUnknownLoc(),
                                               type)};
      });
  std::vector<std::string> commonNames = {x.str(), cell.str()};
  std::vector<std::string> commonTypes = {"!vec", "!md.cell"};
  auto common = [&](std::string name, std::string type) {
    commonNames.push_back(std::move(name));
    commonTypes.push_back(std::move(type));
  };
  for (const auto &field : context.getFields())
    common((prefix + field.name).str(), field.isInteger ? "!ids" : "!real");
  for (const auto &table : context.getTables())
    common("%t_" + table.name, table.getType().str());
  for (const auto &set : context.getTupleSets()) {
    common(relations + set.name, "!rel_" + set.name);
    for (const auto &field : set.fields)
      common("%f_" + set.name + "_" + field.name, "!of_" + set.name);
  }
  if (context.getControl().usesTime)
    common(time.str(), "f64");
  unsigned slot = 0, base = context.getNumPotentialArguments();
  assert(commonNames.size() == base &&
         "pull potential argument count mismatch");
  for (auto [index, term] : llvm::enumerate(system.topology->tupleTerms)) {
    if (!term.isCentroid())
      continue;
    std::vector<std::string> coordinates;
    for (auto [column, entry] : llvm::enumerate(columns))
      if (entry.first == index)
        coordinates.push_back(names[column]);
    std::string name = terms + "_" + std::to_string(index);
    std::vector<std::string> inputs = commonNames, inputTypes = commonTypes,
                             outputs = {name + "_e"};
    for (auto [q, coordinate] : llvm::enumerate(coordinates)) {
      inputs.push_back(coordinate);
      inputTypes.push_back("f64");
      outputs.push_back(name + "_d" + std::to_string(q));
    }
    llvm::SmallVector<StringRef> in, inTypes, out,
        outTypes(outputs.size(), "f64");
    for (const auto &s : inputs)
      in.push_back(s);
    for (const auto &s : inputTypes)
      inTypes.push_back(s);
    for (const auto &s : outputs)
      out.push_back(s);
    context.emitIRFragment(
        indent, out, outTypes, in, inTypes,
        [index, base, count = coordinates.size()](
            mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
          llvm::SmallVector<int32_t> kinds = {
              int32_t(mdir::md::Request::Energy)};
          llvm::SmallVector<int64_t> arguments = {-1};
          for (size_t q = 0; q != count; ++q) {
            kinds.push_back(int32_t(mdir::md::Request::Derivative));
            arguments.push_back(base + q);
          }
          llvm::SmallVector<mlir::Type> types(count + 1, builder.getF64Type());
          auto op = mdir::md::EvaluateOp::create(
              builder, builder.getUnknownLoc(), types,
              mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                           "pullterm" + std::to_string(index)),
              values, builder.getDenseI32ArrayAttr(kinds),
              builder.getDenseI64ArrayAttr(arguments));
          return TermIRValues(op.getResults());
        });
    inputs = {terms};
    inputTypes = {termType};
    for (const auto &result : outputs) {
      inputs.push_back(result);
      inputTypes.push_back("f64");
    }
    for (const auto &coordinate : coordinates) {
      inputs.push_back(coordinate);
      inputTypes.push_back("f64");
    }
    in.clear();
    inTypes.clear();
    for (const auto &s : inputs)
      in.push_back(s);
    for (const auto &s : inputTypes)
      inTypes.push_back(s);
    context.emitIRFragment(
        indent, {}, {}, in, inTypes,
        [slot, count = coordinates.size()](
            mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
          using namespace mlir;
          auto loc = builder.getUnknownLoc();
          unsigned next = slot;
          auto store = [&](Value value) {
            Value place = arith::ConstantIndexOp::create(builder, loc, next++);
            memref::StoreOp::create(builder, loc, value, values[0],
                                    ValueRange{place});
          };
          auto derivative = [&](size_t q) -> Value { return values[2 + q]; };
          auto coordinate = [&](size_t q) -> Value {
            return values[2 + count + q];
          };
          auto mul = [&](Value a, Value b) -> Value {
            return arith::MulFOp::create(builder, loc, a, b);
          };
          auto add = [&](Value a, Value b) -> Value {
            return arith::AddFOp::create(builder, loc, a, b);
          };
          auto div = [&](Value a, Value b) -> Value {
            return arith::DivFOp::create(builder, loc, a, b);
          };
          auto neg = [&](Value a) -> Value {
            return arith::NegFOp::create(builder, loc, a);
          };
          store(values[1]);
          if (count == 1) {
            store(neg(derivative(0)));
            return {};
          }
          Value radial = div(derivative(0), coordinate(0));
          std::array<Value, 3> force;
          for (int c = 0; c != 3; ++c) {
            Value product = mul(radial, coordinate(1 + c));
            Value sum = add(product, derivative(1 + c));
            force[c] = neg(sum);
          }
          Value along = mul(force[0], coordinate(1));
          for (int c = 1; c != 3; ++c) {
            Value product = mul(force[c], coordinate(1 + c));
            along = add(along, product);
          }
          store(div(along, coordinate(0)));
          for (Value value : force)
            store(value);
          return {};
        });
    slot += coordinates.size() == 1 ? 2 : 5;
  }
  context.emitIRFragment(
      indent, {}, {}, {step, values, terms}, {"i64", type, termType},
      [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        auto loc = builder.getUnknownLoc();
        auto dynamic = mlir::MemRefType::get({mlir::ShapedType::kDynamic},
                                             builder.getF64Type());
        mlir::Value coordinates =
            mlir::memref::CastOp::create(builder, loc, dynamic, values[1]);
        mlir::Value terms =
            mlir::memref::CastOp::create(builder, loc, dynamic, values[2]);
        mlir::func::CallOp::create(
            builder, loc, "mdrtWritePull", mlir::TypeRange{},
            mlir::ValueRange{values[0], coordinates, terms});
        return {};
      });
}
