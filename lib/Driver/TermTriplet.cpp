// Custom triplet terms and particle-place parameters (D359, D361, #683).
#include "BuilderImpl.h"
#include "mdir/Dialect/MD/MDCoordinates.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"

using namespace mdir::driver;
using namespace mdir::driver::builder;

struct TripletCoordinateIR {
  std::string name;
  mdir::md::Coordinate coordinate;
};
struct TripletParameterIR {
  std::string name;
  double value = 0.0;
  std::optional<size_t> argument;
};
struct TripletIRConfig {
  explicit TripletIRConfig(ExpressionIR expression)
      : expression(std::move(expression)) {}
  ExpressionIR expression;
  std::string termName;
  double cutoff = 0.0;
  size_t gatheredCount = 0;
  std::vector<TripletCoordinateIR> coordinates;
  std::vector<std::pair<std::string, double>> constants;
  std::vector<TripletParameterIR> parameters;
  std::vector<std::pair<std::string, size_t>> externalBindings, functionTables;
};

namespace mdir::driver::builder {
class TripletTermBuilder final : public TermBuilder {
public:
  int potentialRank() const override { return 1; }
  llvm::Error collectBareParameters(TermParameters &context) const override;
  mlir::Value buildBareContributions(const TermInputs &, mlir::OpBuilder &,
      const TermNativePotential &, mlir::Value total) const override;

private:
  static std::string getTripletParameter(const Control &, const TripletTerm &,
                                         StringRef);
  static mlir::Value buildTripletTerm(const TermInputs &, mlir::OpBuilder &,
      const TermNativePotential &, const TripletTerm &);
};
} // namespace mdir::driver::builder

std::unique_ptr<TermBuilder> mdir::driver::builder::createTripletTermBuilder() {
  return std::make_unique<TripletTermBuilder>();
}

llvm::Error
TripletTermBuilder::collectBareParameters(TermParameters &context) const {
  const Control &control = context.getControl();
  const System &system = context.getSystem();
  auto &parameters = context.getMutableTypeParameters();
  auto &fields = context.getFields();
  // The parameters of the types that the terms over triplets take for a
  // particle by its place, `sigma1` for the center (D160): a field where
  // they differ between the types, a number where they do not.
  for (const TripletTerm &term : control.triplets) {
    Expression expression =
        llvm::cantFail(Expression::parse(term.expression, control.functions));
    for (const std::string &name : expression.getNames()) {
      std::string stem = getTripletParameter(control, term, name);
      if (stem.empty() || llvm::any_of(parameters, [&](Parameter &known) {
            return known.name == stem;
          }))
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
        parameter.field = fields.size();
        Program::Field field;
        field.name = stem;
        for (unsigned type : system.types)
          field.values.push_back(values[type]);
        fields.push_back(std::move(field));
      }
      parameters.push_back(std::move(parameter));
    }
  }
  return llvm::Error::success();
}

std::string TripletTermBuilder::getTripletParameter(const Control &control,
                                                    const TripletTerm &term,
                                                    StringRef name) {
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

mlir::Value TripletTermBuilder::buildBareContributions(
    const TermInputs &context, mlir::OpBuilder &builder,
    const TermNativePotential &state, mlir::Value total) const {
  for (const auto &term : context.getControl().triplets) {
    auto result = buildTripletTerm(context, builder, state, term);
    total = mlir::arith::AddFOp::create(builder, builder.getUnknownLoc(), total, result);
  }
  return total;
}

mlir::Value TripletTermBuilder::buildTripletTerm(
    const TermInputs &context, mlir::OpBuilder &builder,
    const TermNativePotential &state, const TripletTerm &term) {
  const Control &control = context.getControl();
  auto parameters = context.getTypeParameters();
  Expression expression =
      llvm::cantFail(Expression::parse(term.expression, control.functions));
  // The cosine coordinate avoids the angle derivative's 1/sin(theta) at
  // collinear triplets (docs/tuples.md, Section 3.1).
  expression.replaceCall("cos", "theta", "cos(theta)");
  const auto &names = expression.getNames();
  TripletIRConfig config(expression.prepareIR());
  config.termName = term.name;
  config.cutoff = term.cutoff * units::length;
  const TripletCoordinateIR variables[] = {
      {"r12", {mdir::md::CoordinateKind::Distance, {0, 1}}},
      {"r13", {mdir::md::CoordinateKind::Distance, {2, 1}}},
      {"r23", {mdir::md::CoordinateKind::Distance, {0, 2}}},
      {"theta", {mdir::md::CoordinateKind::Angle, {0, 1, 2}}},
      {"cos(theta)", {mdir::md::CoordinateKind::Cosine, {0, 1, 2}}}};
  for (const auto &variable : variables)
    if (llvm::is_contained(names, variable.name))
      config.coordinates.push_back(variable);
  if (config.coordinates.empty())
    config.coordinates.push_back(variables[0]);
  std::vector<std::string> gathered;
  auto parameterFor = [&](StringRef stem) -> const Parameter & {
    auto found = llvm::find_if(
        parameters, [&](const Parameter &p) { return p.name == stem; });
    assert(found != parameters.end() && "missing triplet type parameter");
    return *found;
  };
  for (const auto &name : names) {
    std::string stem = getTripletParameter(control, term, name);
    if (stem.empty() || llvm::is_contained(gathered, stem))
      continue;
    if (!parameterFor(stem).isUniform)
      gathered.push_back(stem);
  }
  config.gatheredCount = gathered.size();
  for (const auto &constant : term.constants)
    config.constants.push_back(constant);
  // Particle 1 is the center at tuple place 1; ends 2 and 3 occupy 0 and 2.
  static const int places[] = {1, 0, 2};
  for (const auto &name : names) {
    std::string stem = getTripletParameter(control, term, name);
    if (stem.empty())
      continue;
    TripletParameterIR parameter{name, parameterFor(stem).value, std::nullopt};
    auto found = llvm::find(gathered, stem);
    if (found != gathered.end())
      parameter.argument = config.coordinates.size() +
                           3 * (found - gathered.begin()) +
                           places[name.back() - '1'];
    config.parameters.push_back(std::move(parameter));
  }
  TermIRValues values{state.neighborhood, state.positions, state.cell};
  auto operand = [&](mlir::Value value) {
    size_t slot = values.size();
    values.push_back(value);
    return slot;
  };
  for (const auto &stem : gathered) operand(state.field(stem));
  llvm::StringMap<mlir::Value> bindings;
  for (const auto &coordinate : config.coordinates)
    bindings[coordinate.name] = {};
  for (const auto &[name, value] : config.constants)
    bindings[name] = {};
  for (const auto &parameter : config.parameters)
    bindings[parameter.name] = {};
  if (control.usesTime)
    bindings["t"] = state.time;
  llvm::SmallVector<StringRef> supplied;
  for (const auto &entry : bindings)
    supplied.push_back(entry.getKey());
  auto inputs = config.expression.getInputs(supplied);
  for (const auto &name : inputs.names) {
    auto found = bindings.find(name);
    assert(found != bindings.end() && "missing triplet expression binding");
    if (found->second)
      config.externalBindings.emplace_back(name, operand(found->second));
  }
  for (const auto &name : inputs.tables)
    config.functionTables.emplace_back(name, operand(state.table(name)));
        using namespace mlir;
        auto loc = builder.getUnknownLoc();
        auto pairType = cast<mdir::md::RelationType>(values[0].getType());
        auto relationType = mdir::md::RelationType::get(
            builder.getContext(), pairType.getParticleSet(), 3,
            mdir::md::Orientation::Reversal, FlatSymbolRefAttr{});
        auto relation = mdir::md::TripletsOp::create(
            builder, loc, relationType, values[0],
            builder.getF64FloatAttr(config.cutoff));
        llvm::SmallVector<mdir::md::Coordinate> coordinates;
        for (const auto &coordinate : config.coordinates)
          coordinates.push_back(coordinate.coordinate);
        DenseI32ArrayAttr kinds;
        DenseI64ArrayAttr members;
        mdir::md::getCoordinateAttrs(builder, coordinates, kinds, members);
        auto sum = mdir::md::SumTuplesOp::create(
            builder, loc, builder.getF64Type(), relation.getResult(), values[1],
            values[2], mlir::ValueRange(values).slice(3, config.gatheredCount), ValueRange{},
            kinds, members);
        {
          OpBuilder::InsertionGuard guard(builder);
          size_t count = config.coordinates.size() + 3 * config.gatheredCount;
          llvm::SmallVector<Type> blockTypes(count, builder.getF64Type());
          llvm::SmallVector<Location> locations(count, loc);
          Block *block =
              builder.createBlock(&sum.getKernel(), {}, blockTypes, locations);
          auto args = block->getArguments();
          auto constant = [&](double value) -> Value {
            return arith::ConstantOp::create(builder, loc,
                                             builder.getF64FloatAttr(value));
          };
          Value toLength = constant(1.0 / units::length);
          llvm::StringMap<Value> bindings, tables;
          for (auto [k, coordinate] : llvm::enumerate(config.coordinates)) {
            Value value = args[k];
            if (coordinate.coordinate.kind ==
                mdir::md::CoordinateKind::Distance)
              value = arith::MulFOp::create(builder, loc, value, toLength);
            bindings[coordinate.name] = value;
          }
          for (const auto &[name, value] : config.constants)
            bindings[name] = constant(value);
          for (const auto &parameter : config.parameters)
            bindings[parameter.name] = parameter.argument
                                           ? args[*parameter.argument]
                                           : constant(parameter.value);
          for (const auto &[name, slot] : config.externalBindings)
            bindings[name] = values[slot];
          for (const auto &[name, slot] : config.functionTables)
            tables[name] = values[slot];
          Value expression = config.expression.build(builder, bindings, tables,
                                                     config.termName);
          Value toEnergy = constant(units::energy);
          Value energy =
              arith::MulFOp::create(builder, loc, expression, toEnergy);
          mdir::md::YieldOp::create(builder, loc, ValueRange{energy});
        }
        return sum.getResult();
}
