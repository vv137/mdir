// Custom tuple and centroid terms (D359, D361, #683).
#include "PotentialIR.h"
#include "BuilderImpl.h"
#include "mdir/Dialect/MD/MDCoordinates.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"

using namespace mdir::driver;
using namespace mdir::driver::builder;
using TupleGradient = BuildKept::TupleGradient;

struct TupleLoopIRConfig {
  explicit TupleLoopIRConfig(ExpressionIR expression)
      : expression(std::move(expression)) {}
  ExpressionIR expression;
  std::string termName;
  unsigned arity = 0;
  bool scaleDistances = false;
  size_t gatheredCount = 0;
  std::vector<mdir::md::Coordinate> coordinates;
  std::vector<std::string> coordinateNames, parameters;
  std::vector<TupleGradient> gradients;
  std::vector<std::pair<std::string, size_t>> externalBindings, functionTables;
};

struct CentroidTermIRConfig {
  explicit CentroidTermIRConfig(ExpressionIR expression)
      : expression(std::move(expression)) {}
  ExpressionIR expression;
  std::string termName;
  std::vector<std::pair<std::string, double>> parameters;
  std::vector<std::pair<std::string, size_t>> externalBindings, functionTables;
};

static mlir::Value buildCentroidContribution(mlir::OpBuilder &builder, mlir::ValueRange values, const CentroidTermIRConfig &config) {
using namespace mlir;
        auto loc = builder.getUnknownLoc();
        auto constant = [&](double value) -> Value {
          return arith::ConstantOp::create(builder, loc,
                                           builder.getF64FloatAttr(value));
        };
        llvm::StringMap<Value> bindings, tables;
        for (const auto &[name, value] : config.parameters)
          bindings[name] = constant(value);
        for (const auto &[name, slot] : config.externalBindings)
          bindings[name] = values[slot];
        for (const auto &[name, slot] : config.functionTables)
          tables[name] = values[slot];
        Value expression =
            config.expression.build(builder, bindings, tables, config.termName);
        Value kj = constant(units::energy);
        return arith::MulFOp::create(builder, loc, expression, kj);

}

static mlir::Value buildTupleContribution(mlir::OpBuilder &builder, mlir::ValueRange values, const TupleLoopIRConfig &config) {
using namespace mlir;
          auto loc = builder.getUnknownLoc();
          DenseI32ArrayAttr kinds;
          DenseI64ArrayAttr members;
          mdir::md::getCoordinateAttrs(builder, config.coordinates, kinds,
                                       members);
          auto sum = mdir::md::SumTuplesOp::create(
              builder, loc, builder.getF64Type(), values[0], values[1],
              values[2], values.slice(3, config.gatheredCount),
              values.slice(3 + config.gatheredCount,
                           config.parameters.size() + config.gradients.size()),
              kinds, members);
          size_t coordinateCount = config.coordinates.size();
          size_t parameterStart =
              coordinateCount + config.gatheredCount * config.arity;
          size_t count = parameterStart + config.parameters.size() +
                         config.gradients.size();
          llvm::SmallVector<Type> blockTypes(count, builder.getF64Type());
          llvm::SmallVector<Location> locations(count, loc);
          OpBuilder::InsertionGuard insertionGuard(builder);
          Block *block =
              builder.createBlock(&sum.getKernel(), {}, blockTypes, locations);
          auto args = block->getArguments();
          auto constant = [&](double value) -> Value {
            return arith::ConstantOp::create(builder, loc,
                                             builder.getF64FloatAttr(value));
          };
          llvm::StringMap<Value> bindings, tables;
          Value scale;
          if (config.scaleDistances)
            scale = constant(1.0 / units::length);
          for (size_t k = 0; k != coordinateCount; ++k) {
            Value value = args[k];
            if (config.coordinates[k].kind ==
                mdir::md::CoordinateKind::Distance)
              value = arith::MulFOp::create(builder, loc, value, scale);
            bindings[config.coordinateNames[k]] = value;
          }
          for (size_t k = 0; k != config.parameters.size(); ++k)
            bindings[config.parameters[k]] = args[parameterStart + k];
          size_t gatherStart = coordinateCount;
          for (auto [k, gradient] : llvm::enumerate(config.gradients)) {
            auto parameter = llvm::find(config.parameters, gradient.parameter);
            assert(parameter != config.parameters.end() &&
                   "unknown tuple gradient parameter");
            Value value =
                args[parameterStart + (parameter - config.parameters.begin())];
            Value selector =
                args[parameterStart + config.parameters.size() + k];
            Value zero = constant(0.0);
            for (unsigned c = 0; c != gradient.count; ++c) {
              Value number = constant(static_cast<double>(c));
              Value equal = arith::CmpFOp::create(
                  builder, loc, arith::CmpFPredicate::OEQ, selector, number);
              Value derivative =
                  args[gatherStart + c * config.arity + gradient.place];
              Value selected = arith::SelectOp::create(builder, loc, equal,
                                                       derivative, zero);
              value = arith::AddFOp::create(builder, loc, value, selected);
            }
            bindings[gradient.parameter] = value;
            gatherStart += gradient.count * config.arity;
          }
          for (const auto &[name, slot] : config.externalBindings)
            bindings[name] = values[slot];
          for (const auto &[name, slot] : config.functionTables)
            tables[name] = values[slot];
          Value expression = config.expression.build(builder, bindings, tables,
                                                     config.termName);
          Value kj = constant(units::energy);
          Value energy = arith::MulFOp::create(builder, loc, expression, kj);
          mdir::md::YieldOp::create(builder, loc, ValueRange{energy});
          return sum.getResult();

}

namespace mdir::driver::builder {
class TupleTermBuilder final : public TermBuilder {
public:
  TupleTermBuilder() {
    potentialMask = TermInputs::TupleTerms;
    alchemicalMask = TermInputs::TupleTerms;
    nativePotentialMask = potentialMask;
  }
  unsigned getPotentialMask(const TermInputs &, const DependentTerms *dependent,
                            bool) const override {
    return !dependent || !dependent->tuples.empty() ? potentialMask : 0;
  }
  int potentialRank() const override { return -3; }
  int topologyRank() const override { return 3; }
  void collectTupleSets(TermParameters &context) const override;
  void refillArrays(TermParameters &context) const override {
    const Topology &topology = context.getTopology();
    for (Program::TupleSet &set : context.getTupleSets()) {
      for (const TupleTerm &term : topology.tupleTerms)
        if (!term.isCentroid() && set.name == "custom_" + term.name)
          for (Program::Field &field : set.fields)
            if (field.values.empty())
              fillTupleParameter(field, term);
    }
  }
  void buildTopologyPotential(const TermInputs &, mlir::OpBuilder &, const TermNativeTopology &,
      const PotentialSelection &, llvm::function_ref<void(mlir::Value)>) const override;
  void emitTopologyPotential(TermEmission &context,
      const PotentialSelection &selection,
      const TermPotentialAddition & add) const override;

  void appendTermDefinitions(
      const TermInputs &context,
      std::vector<TermDefinition> &definitions) const override {
    for (size_t k = 0, e = context.getTopology().tupleTerms.size(); k != e; ++k)
      definitions.push_back({14, k, "term_custom" + std::to_string(k),
                             TermInputs::TupleTerms, static_cast<int>(k)});
  }
  ObservedSelection findObservedTerm(const TermInputs &context,
                                     StringRef name) const override {
    for (auto [k, term] : llvm::enumerate(context.getTopology().tupleTerms))
      if (term.name == name)
        return {TermInputs::TupleTerms,
                static_cast<int>(k),
                {TermInputs::TupleTerms, static_cast<int>(k), -1, -1, false}};
    return {};
  }
  std::optional<double> getObservedValue(const TermInputs &context, unsigned kind,
                                        int index, StringRef name) const override {
    if (kind == TermInputs::TupleTerms)
      for (const auto &[key, values] : context.getTopology().tupleTerms[index].parameters)
        if (key == name)
          return values.front();
    return std::nullopt;
  }
  llvm::Expected<std::optional<Program::GradientOutcome>>
  collectTunableOutcome(TermParameters &context, unsigned k) const override {
    const Control &control = context.getControl();
    const auto &declaration = control.tunableDeclarations[k];
    if (declaration.kind != Control::TunableKind::TupleParameter)
      return std::optional<Program::GradientOutcome>();
    // A parameter that the expression of its term reads has its fields
    // (collectTopology); one that it does not read enters no value.
    const TupleTerm &term =
        context.getTopology().tupleTerms[declaration.termIndex];
    if (term.isCentroid())
      return makeError(
          "System.tunable_gradient: the derivative of the energy in the "
          "tunable '" + declaration.name + "', a parameter of the term '" +
          term.name + "' over centers of groups, is not implemented yet");
    if (llvm::any_of(context.getTupleGradients(), [&](const TupleGradient &gradient) {
          return gradient.tunable == k;
        }))
      return std::optional(Program::GradientOutcome::Rule);

    return std::optional(Program::GradientOutcome::Zero);
  }

private:
  static void fillTupleParameter(Program::Field &, const TupleTerm &);
  static void emitCentroidTerm(TermEmission &, size_t, const TupleTerm &);
};
} // namespace mdir::driver::builder

std::unique_ptr<TermBuilder> mdir::driver::builder::createTupleTermBuilder() {
  return std::make_unique<TupleTermBuilder>();
}

void TupleTermBuilder::fillTupleParameter(Program::Field &field,
                                 const TupleTerm &term) {
  for (const auto &[name, values] : term.parameters)
    if (name == field.name)
      field.values = values;
}


void TupleTermBuilder::collectTupleSets(TermParameters &context) const {
  const Topology &topology = context.getTopology();
  const Control &control = context.getControl();
  auto addSet = [&](StringRef name, unsigned arity) -> Program::TupleSet & {
    Program::TupleSet set;
    set.name = name.str();
    set.arity = arity;
    context.getTupleSets().push_back(std::move(set));
    return context.getTupleSets().back();
  };
  auto addField = [&](Program::TupleSet &set, StringRef name) {
    Program::Field field;
    field.name = name.str();
    set.fields.push_back(std::move(field));
    return set.fields.size() - 1;
  };
  // The terms given by expressions (D136), with their parameters as fields
  // of their tuples, in the units of the control file.
  for (const TupleTerm &term : topology.tupleTerms) {
    // A term over the centers of groups (D139), one set of pairs: for
    // each group the pairs of a member and the reference of the group, and
    // for each group after the first the pair of its reference and that of
    // the first. A field for each group and each such link, `w<k>`, holds
    // the weight of the member in its pairs, 1 in the pair of the link,
    // and 0 elsewhere, so that one loop gives all the sums.
    if (term.isCentroid()) {
      size_t groups = term.centers.size();
      Program::TupleSet &set = addSet("cg_" + term.name, 2);
      set.oriented = true;
      set.reversible = false;
      for (size_t k = 0; k != 2 * groups - 1; ++k)
        addField(set, "w" + std::to_string(k));
      auto addPair = [&](unsigned i, unsigned j, size_t slot, double weight) {
        set.members.push_back(i);
        set.members.push_back(j);
        for (size_t k = 0; k != 2 * groups - 1; ++k)
          set.fields[k].values.push_back(k == slot ? weight : 0.0);
      };
      for (auto [g, center] : llvm::enumerate(term.centers)) {
        for (auto [i, weight] :
             llvm::zip_equal(center.members, center.weights))
          if (i != center.reference)
            addPair(i, center.reference, g, weight);
        if (g > 0)
          addPair(center.reference, term.centers.front().reference,
                  groups + g - 1, 1.0);
      }
      continue;
    }
    Program::TupleSet &set = addSet("custom_" + term.name, term.arity);
    for (unsigned member : term.particles)
      set.members.push_back(member);
    for (const auto &[name, values] : term.parameters)
      fillTupleParameter(set.fields[addField(set, name)], term);
    // The tunable parameters of the term whose derivatives the program
    // carries (D230): for each, the place whose members are
    // shared by the fewest tuples, and for each tuple its number among the
    // tuples of its member there.
    if (context.hasTunableGradient())
      for (auto [k, declaration] :
           llvm::enumerate(control.tunableDeclarations)) {
        if (declaration.kind != Control::TunableKind::TupleParameter ||
            &topology.tupleTerms[declaration.termIndex] != &term)
          continue;
        Expression expression = llvm::cantFail(
            Expression::parse(term.expression, control.functions));
        if (!llvm::is_contained(expression.getNames(), declaration.parameter))
          continue;
        size_t tuples = term.particles.size() / term.arity;
        unsigned place = 0, count = 0;
        std::vector<double> colors;
        for (unsigned candidate = 0; candidate != term.arity; ++candidate) {
          std::map<unsigned, unsigned> seen;
          std::vector<double> numbers;
          unsigned most = 0;
          for (size_t t = 0; t != tuples; ++t) {
            unsigned number = seen[term.particles[t * term.arity + candidate]]++;
            numbers.push_back(number);
            most = std::max(most, number + 1);
          }
          if (candidate == 0 || most < count) {
            place = candidate;
            count = most;
            colors = std::move(numbers);
          }
        }
        TupleGradient gradient;
        gradient.tunable = static_cast<unsigned>(k);
        gradient.term = declaration.termIndex;
        gradient.parameter = declaration.parameter;
        gradient.place = place;
        gradient.count = count;
        gradient.first = static_cast<unsigned>(context.getGradientFields().size());
        for (unsigned c = 0; c != count; ++c) {
          Program::GradientField field;
          field.tunable = gradient.tunable;
          for (size_t t = 0; t != tuples; ++t)
            if (colors[t] == c)
              field.sites.push_back(
                  {static_cast<uint32_t>(t),
                   term.particles[t * term.arity + place]});
          context.getGradientFields().push_back(std::move(field));
        }
        size_t field = addField(set, "tgc_" + declaration.parameter);
        set.fields[field].values = std::move(colors);
        context.getMutableTupleGradients().push_back(std::move(gradient));
      }
  }

}

void TupleTermBuilder::emitCentroidTerm(TermEmission &context, size_t index, const TupleTerm &term) {
  const Control &control = context.getControl();
  std::string prefix = "%cb" + std::to_string(index) + "_";
  llvm::StringMap<std::string> variables =
      context.emitCentroidCoordinates(term, "  ", "%x", "%cell", "%r_", prefix);
  Expression expression =
      llvm::cantFail(Expression::parse(term.expression, control.functions));
  CentroidTermIRConfig config(expression.prepareIR());
  config.termName = term.name;
  for (const auto &parameter : term.parameters) {
    config.parameters.emplace_back(parameter.first, parameter.second.front());
    variables[parameter.first] = "";
  }
  if (control.usesTime)
    variables["t"] = "%time";
  context.bindLambdas(variables);
  context.bindObserved(term.name, variables);
  llvm::SmallVector<StringRef> supplied;
  for (const auto &entry : variables)
    supplied.push_back(entry.getKey());
  auto inputs = config.expression.getInputs(supplied);
  std::vector<std::string> names;
  llvm::SmallVector<StringRef> types;
  auto operand = [&](std::string name, StringRef type) {
    size_t slot = names.size();
    names.push_back(std::move(name));
    types.push_back(type);
    return slot;
  };
  for (const auto &name : inputs.names) {
    auto found = variables.find(name);
    assert(found != variables.end() && "missing centroid expression binding");
    if (!found->second.empty())
      config.externalBindings.emplace_back(name, operand(found->second, "f64"));
  }
  for (const auto &name : inputs.tables)
    config.functionTables.emplace_back(name, operand("%t_" + name, "!grid"));
  llvm::SmallVector<StringRef> operands;
  for (const auto &name : names)
    operands.push_back(name);
  std::string result = "%u_centroid_" + term.name;
  context.emitIRFragment(
      "  ", {result}, {"f64"}, operands, types,
      [config](mlir::OpBuilder &builder,
               mlir::ValueRange values) -> TermIRValues {
        return {buildCentroidContribution(builder, values, config)};
      });
}


void TupleTermBuilder::emitTopologyPotential(TermEmission &context,
    const PotentialSelection &selection,
    const TermPotentialAddition & add) const {
  const System &system = context.getSystem();
  const Control &control = context.getControl();
  unsigned terms = selection.terms;
  int tupleTerm = selection.tupleTerm;
  bool alchemicalOnly = selection.alchemicalOnly;
  // The terms given by expressions (D136), in the units of the control
  // file: the distance in Å, the angles in radians, the energy in kcal/mol.
  for (auto [index, term] : llvm::enumerate(system.topology->tupleTerms)) {
    if (!(terms & TermInputs::TupleTerms) ||
        (tupleTerm >= 0 && static_cast<int>(index) != tupleTerm))
      continue;
    if (!context.includesTuple(static_cast<unsigned>(index)))
      continue;
    if (alchemicalOnly && !context.usesLambda(term.expression))
      continue;
    if (term.isCentroid()) {
      emitCentroidTerm(context, index, term);
      add("centroid_" + term.name);
      continue;
    }
    std::string set = "custom_" + term.name;
    Expression expression =
        llvm::cantFail(Expression::parse(term.expression, control.functions));
    TupleLoopIRConfig config(expression.prepareIR());
    config.termName = term.name;
    config.arity = term.arity;
    config.scaleDistances = term.isCompound() || term.arity == 2;
    if (term.isCompound()) {
      for (const auto &coordinate : term.coordinates) {
        auto kind = mdir::md::symbolizeCoordinateKind(coordinate.kind);
        assert(kind && "unknown validated tuple coordinate");
        mdir::md::Coordinate item{*kind, {}};
        item.members.append(coordinate.places.begin(), coordinate.places.end());
        config.coordinates.push_back(std::move(item));
        config.coordinateNames.push_back(coordinate.name);
      }
    } else {
      mdir::md::Coordinate coordinate;
      coordinate.kind = term.arity == 2   ? mdir::md::CoordinateKind::Distance
                        : term.arity == 3 ? mdir::md::CoordinateKind::Angle
                                          : mdir::md::CoordinateKind::Dihedral;
      for (unsigned k = 0; k != term.arity; ++k)
        coordinate.members.push_back(k);
      config.coordinates.push_back(std::move(coordinate));
      config.coordinateNames.push_back(term.getVariable().str());
    }
    if (context.hasGradientArguments())
      for (const TupleGradient &gradient : context.getTupleGradients())
        if (gradient.term == index)
          config.gradients.push_back(gradient);
    std::string relationType = "!rel_" + set, fieldType = "!of_" + set;
    std::vector<std::string> names = {"%r_" + set, "%x", "%cell"};
    llvm::SmallVector<StringRef> types = {relationType, "!vec", "!md.cell"};
    auto operand = [&](std::string name, StringRef type) {
      size_t slot = names.size();
      names.push_back(std::move(name));
      types.push_back(type);
      return slot;
    };
    for (const auto &gradient : config.gradients)
      for (unsigned c = 0; c != gradient.count; ++c)
        operand("%tg_d" + std::to_string(gradient.first + c), "!real");
    config.gatheredCount = names.size() - 3;
    for (const auto &parameter : term.parameters) {
      config.parameters.push_back(parameter.first);
      operand("%f_" + set + "_" + parameter.first, fieldType);
    }
    for (const auto &gradient : config.gradients)
      operand("%f_" + set + "_tgc_" + gradient.parameter, fieldType);
    llvm::StringMap<std::string> variables;
    for (const auto &name : config.coordinateNames)
      variables[name] = "";
    if (control.usesTime)
      variables["t"] = "%time";
    context.bindLambdas(variables);
    for (const auto &name : config.parameters)
      variables[name] = "";
    for (const auto &gradient : config.gradients)
      variables[gradient.parameter] = "";
    context.bindObserved(term.name, variables);
    llvm::SmallVector<StringRef> supplied;
    for (const auto &entry : variables)
      supplied.push_back(entry.getKey());
    auto inputs = config.expression.getInputs(supplied);
    for (const auto &name : inputs.names) {
      auto found = variables.find(name);
      assert(found != variables.end() && "missing tuple expression binding");
      if (!found->second.empty())
        config.externalBindings.emplace_back(name,
                                             operand(found->second, "f64"));
    }
    for (const auto &name : inputs.tables)
      config.functionTables.emplace_back(name, operand("%t_" + name, "!grid"));
    llvm::SmallVector<StringRef> operands;
    for (const auto &name : names)
      operands.push_back(name);
    std::string result = "%u_" + set;
    context.emitIRFragment(
        "  ", {result}, {"f64"}, operands, types,
        [config](mlir::OpBuilder &builder,
                 mlir::ValueRange values) -> TermIRValues {
          return {buildTupleContribution(builder, values, config)};
        });
    add(set);
  }

}

void TupleTermBuilder::buildTopologyPotential(const TermInputs &context,
    mlir::OpBuilder &builder, const TermNativeTopology &state,
    const PotentialSelection &selection, llvm::function_ref<void(mlir::Value)> add) const {
  const Control &control = context.getControl();
  for (auto [index, term] : llvm::enumerate(context.getTopology().tupleTerms)) {
    if (!(selection.terms & TermInputs::TupleTerms) ||
        (selection.tupleTerm >= 0 && static_cast<int>(index) != selection.tupleTerm) ||
        !context.includesTuple(index) ||
        (selection.alchemicalOnly && !context.usesLambda(term.expression))) continue;
    llvm::StringMap<mlir::Value> variables;
    if (term.isCentroid()) {
      std::string set = "cg_" + term.name;
      TermIRValues geometry{state.relation(set), state.physical.positions, state.physical.cell};
      for (size_t slot = 0; slot + 1 < 2 * term.centers.size(); ++slot)
        geometry.push_back(state.tupleField(set, "w" + std::to_string(slot)));
      auto coords = buildCentroidCoordinates(builder, geometry, term.centers.size(), term.arity);
      if (term.arity == 2) {
        llvm::SmallVector<StringRef> keys{"r", "dx", "dy", "dz"};
        for (auto [slot, key] : llvm::enumerate(keys)) variables[key] = coords[slot];
      } else variables["theta"] = coords[0];
      auto expression = llvm::cantFail(Expression::parse(term.expression, control.functions));
      CentroidTermIRConfig config(expression.prepareIR());
      config.termName = term.name;
      for (const auto &[name, values] : term.parameters) {
        config.parameters.emplace_back(name, values.front());
        variables[name] = {};
      }
      if (control.usesTime) variables["t"] = state.physical.time;
      state.bindLambdas(variables);
      state.bindObserved(term.name, variables);
      llvm::SmallVector<StringRef> supplied;
      for (const auto &entry : variables) supplied.push_back(entry.getKey());
      auto inputs = config.expression.getInputs(supplied);
      TermIRValues values;
      for (const auto &name : inputs.names)
        if (auto value = variables.lookup(name)) {
          config.externalBindings.emplace_back(name, values.size()); values.push_back(value);
        }
      for (const auto &name : inputs.tables) {
        config.functionTables.emplace_back(name, values.size()); values.push_back(state.physical.table(name));
      }
      add(buildCentroidContribution(builder, values, config));
      continue;
    }
    std::string set = "custom_" + term.name;
    Expression expression =
        llvm::cantFail(Expression::parse(term.expression, control.functions));
    TupleLoopIRConfig config(expression.prepareIR());
    config.termName = term.name;
    config.arity = term.arity;
    config.scaleDistances = term.isCompound() || term.arity == 2;
    if (term.isCompound()) {
      for (const auto &coordinate : term.coordinates) {
        auto kind = mdir::md::symbolizeCoordinateKind(coordinate.kind);
        assert(kind && "unknown validated tuple coordinate");
        mdir::md::Coordinate item{*kind, {}};
        item.members.append(coordinate.places.begin(), coordinate.places.end());
        config.coordinates.push_back(std::move(item));
        config.coordinateNames.push_back(coordinate.name);
      }
    } else {
      mdir::md::Coordinate coordinate;
      coordinate.kind = term.arity == 2   ? mdir::md::CoordinateKind::Distance
                        : term.arity == 3 ? mdir::md::CoordinateKind::Angle
                                          : mdir::md::CoordinateKind::Dihedral;
      for (unsigned k = 0; k != term.arity; ++k)
        coordinate.members.push_back(k);
      config.coordinates.push_back(std::move(coordinate));
      config.coordinateNames.push_back(term.getVariable().str());
    }
    if (context.hasGradientArguments())
      for (const TupleGradient &gradient : context.getTupleGradients())
        if (gradient.term == index)
          config.gradients.push_back(gradient);

    TermIRValues values{state.relation(set), state.physical.positions, state.physical.cell};
    for (const auto &gradient : config.gradients)
      for (unsigned c = 0; c != gradient.count; ++c)
        values.push_back(state.gradientSeed(gradient.first + c));
    config.gatheredCount = values.size() - 3;
    for (const auto &[name, data] : term.parameters) {
      config.parameters.push_back(name);
      values.push_back(state.tupleField(set, name));
    }
    for (const auto &gradient : config.gradients)
      values.push_back(state.tupleField(set, "tgc_" + gradient.parameter));
    for (const auto &name : config.coordinateNames) variables[name] = {};
    if (control.usesTime) variables["t"] = state.physical.time;
    state.bindLambdas(variables);
    for (const auto &name : config.parameters) variables[name] = {};
    for (const auto &gradient : config.gradients) variables[gradient.parameter] = {};
    state.bindObserved(term.name, variables);
    llvm::SmallVector<StringRef> supplied;
    for (const auto &entry : variables) supplied.push_back(entry.getKey());
    auto inputs = config.expression.getInputs(supplied);
    for (const auto &name : inputs.names)
      if (auto value = variables.lookup(name)) {
        config.externalBindings.emplace_back(name, values.size()); values.push_back(value);
      }
    for (const auto &name : inputs.tables) {
      config.functionTables.emplace_back(name, values.size()); values.push_back(state.physical.table(name));
    }
    add(buildTupleContribution(builder, values, config));
  }
}
