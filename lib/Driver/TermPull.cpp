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

namespace {
struct PullOutputTermIR {
  size_t index, groups;
  unsigned arity;
  mlir::Value relation;
  llvm::SmallVector<mlir::Value> weights;
};
struct PullOutputSlots {
  size_t index, groups;
  unsigned arity, relation;
  llvm::SmallVector<unsigned> weights;
};
void buildPullColumns(mlir::OpBuilder &builder, mlir::ValueRange values,
                      unsigned slot, size_t count) {
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
  return;
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
return;
}
void buildPullOutput(mlir::OpBuilder &builder, const PotentialStateIR &state,
    mlir::Value step, llvm::ArrayRef<PullOutputTermIR> inputs) {
  using namespace mlir;
  auto loc = builder.getUnknownLoc();
  size_t columns = 0, termCount = 0;
  for (const auto &term : inputs) {
    columns += term.arity == 2 ? 4 : 1;
    termCount += term.arity == 2 ? 5 : 2;
  }
  Value memory = memref::AllocaOp::create(builder, loc,
      MemRefType::get({int64_t(columns)}, builder.getF64Type()));
  std::vector<SmallVector<Value>> coordinates;
  for (const auto &term : inputs) {
    SmallVector<Value> values{term.relation, state.positions, state.cell};
    llvm::append_range(values, term.weights);
    coordinates.push_back(buildCentroidCoordinates(builder, values, term.groups, term.arity));
  }
  unsigned column = 0;
  for (const auto &values : coordinates)
    for (Value value : values) {
      Value place = arith::ConstantIndexOp::create(builder, loc, column++);
      memref::StoreOp::create(builder, loc, value, memory, ValueRange{place});
    }
  Value terms = memref::AllocaOp::create(builder, loc,
      MemRefType::get({int64_t(termCount)}, builder.getF64Type()));
  auto common = potentialArguments(state);
  unsigned slot = 0;
  for (auto [index, term] : llvm::enumerate(inputs)) {
    const auto &points = coordinates[index];
    SmallVector<Value> values = common;
    llvm::append_range(values, points);
    SmallVector<int32_t> kinds{int32_t(mdir::md::Request::Energy)};
    SmallVector<int64_t> arguments{-1};
    for (size_t q = 0; q != points.size(); ++q) {
      kinds.push_back(int32_t(mdir::md::Request::Derivative));
      arguments.push_back(common.size() + q);
    }
    SmallVector<Type> types(points.size() + 1, builder.getF64Type());
    auto result = mdir::md::EvaluateOp::create(builder, loc, types,
        FlatSymbolRefAttr::get(builder.getContext(), "pullterm" + std::to_string(term.index)),
        values, builder.getDenseI32ArrayAttr(kinds), builder.getDenseI64ArrayAttr(arguments));
    SmallVector<Value> output{terms};
    llvm::append_range(output, result.getResults());
    llvm::append_range(output, points);
    buildPullColumns(builder, output, slot, points.size());
    slot += points.size() == 1 ? 2 : 5;
  }
  auto dynamic = MemRefType::get({ShapedType::kDynamic}, builder.getF64Type());
  Value positions = memref::CastOp::create(builder, loc, dynamic, memory);
  Value energies = memref::CastOp::create(builder, loc, dynamic, terms);
  func::CallOp::create(builder, loc, "mdrtWritePull", TypeRange{},
      ValueRange{step, positions, energies});
}
} // namespace

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
  void buildOutput(const TermInputs &, mlir::OpBuilder &, const TermNativeOutput &) const override;
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
  const size_t timeSlot = control.usesTime ? context.getNumPotentialArguments() - 1 : 0;
  auto tableArgument = [&](StringRef name) -> size_t {
    auto tables = context.getParameterTables();
    auto found = llvm::find_if(tables, [&](const auto &table) { return table.name == name; });
    assert(found != tables.end() && "missing pull potential table");
    return 2 + context.getParameterFields().size() + std::distance(tables.begin(), found);
  };
  for (auto [index, term] : llvm::enumerate(system.topology->tupleTerms)) {
    if (!term.isCentroid())
      continue;
    Expression expression =
        llvm::cantFail(Expression::parse(term.expression, control.functions));
    PullPotentialIRConfig config(expression.prepareIR());
    config.symbol = "pullterm" + std::to_string(index);
    config.termName = term.name;
    llvm::StringMap<bool> variables;
    for (const auto &[k, quantity] : columns)
      if (k == index) {
        variables[quantity] = false;
        config.quantities.push_back(quantity);
      }
    for (const auto &parameter : term.parameters) {
      config.parameters.emplace_back(parameter.first, parameter.second.front());
      variables[parameter.first] = false;
    }
    if (control.usesTime)
      variables["t"] = true;
    for (const auto &[name, values] : control.freeEnergy.lambdas) {
      variables["lambda_" + name] = false;
      config.lambdas.emplace_back(
          "lambda_" + name,
          control.freeEnergy.get(name, control.freeEnergy.state));
    }
    llvm::SmallVector<StringRef> supplied;
    for (const auto &entry : variables)
      supplied.push_back(entry.getKey());
    auto inputs = config.expression.getInputs(supplied);
    for (const auto &name : inputs.names) {
      auto found = variables.find(name);
      assert(found != variables.end() && "missing pull expression binding");
      if (found->second)
        config.argumentBindings.emplace_back(name, timeSlot);
    }
    for (const auto &name : inputs.tables)
      config.tableBindings.emplace_back(name, tableArgument(name));
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

void PullTermBuilder::buildOutput(const TermInputs &context, mlir::OpBuilder &builder,
                                  const TermNativeOutput &point) const {
  if (!hasCoordinateOutput(context)) return;
  std::vector<PullOutputTermIR> inputs;
  for (auto [index, term] : llvm::enumerate(context.getTopology().tupleTerms)) {
    if (!term.isCentroid()) continue;
    std::string set = "cg_" + term.name;
    PullOutputTermIR input{index, term.centers.size(), term.arity, point.relation(set), {}};
    for (size_t field = 0; field + 1 < 2 * term.centers.size(); ++field)
      input.weights.push_back(point.tupleField(set, "w" + std::to_string(field)));
    inputs.push_back(std::move(input));
  }
  buildPullOutput(builder, point.physical, point.step, inputs);
}

void PullTermBuilder::emitPullOutput(TermEmission &context, StringRef indent,
    StringRef x, StringRef cell, StringRef prefix, StringRef step, StringRef time) {
  if (getColumns(context).empty()) return;
  std::string relations = ("%r" + prefix.drop_front(2)).str();
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

  unsigned base = context.getNumPotentialArguments();
  assert(commonNames.size() == base && "pull potential argument count mismatch");
  std::vector<PullOutputSlots> layout;
  for (auto [index, term] : llvm::enumerate(context.getTopology().tupleTerms)) {
    if (!term.isCentroid()) continue;
    std::string name = "cg_" + term.name;
    unsigned offset = context.getFields().size() + context.getTables().size();
    const Program::TupleSet *selected = nullptr;
    for (const auto &set : context.getTupleSets()) {
      if (set.name == name) { selected = &set; break; }
      offset += 1 + set.fields.size();
    }
    assert(selected && "pull relation must exist");
    PullOutputSlots slots{index, term.centers.size(), term.arity, offset, {}};
    for (size_t field = 0; field + 1 < 2 * term.centers.size(); ++field) {
      std::string parameter = "w" + std::to_string(field);
      auto found = llvm::find_if(selected->fields,
          [&](const auto &candidate) { return candidate.name == parameter; });
      assert(found != selected->fields.end() && "pull weight must exist");
      slots.weights.push_back(offset + 1 + std::distance(selected->fields.begin(), found));
    }
    layout.push_back(std::move(slots));
  }
  common(step.str(), "i64");
  llvm::SmallVector<StringRef> names, types;
  for (const auto &name : commonNames) names.push_back(name);
  for (const auto &type : commonTypes) types.push_back(type);
  context.emitIRFragment(indent, {}, {}, names, types,
      [layout = std::move(layout), base, timed = context.getControl().usesTime](
          mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        PotentialStateIR state{values[0], values[1], values.slice(2, base - 2 - unsigned(timed)),
            timed ? values[base - 1] : mlir::Value{}};
        std::vector<PullOutputTermIR> inputs;
        for (const auto &slots : layout) {
          PullOutputTermIR term{slots.index, slots.groups, slots.arity, state.fields[slots.relation], {}};
          for (unsigned slot : slots.weights) term.weights.push_back(state.fields[slot]);
          inputs.push_back(std::move(term));
        }
        buildPullOutput(builder, state, values[base], inputs);
        return {};
      });
}
