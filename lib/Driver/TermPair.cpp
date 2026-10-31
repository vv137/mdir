// Custom pair terms and their bounded scalar tail services (D359, D361, #683).
#include "BuilderImpl.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include <array>

using namespace mdir::driver;
using namespace mdir::driver::builder;

/// The overrides of the pair term `term`: those that name it, and those
/// that name no term where there is one term.
static std::vector<const PairOverride *> getOverrides(const Control &control,
                                                      const PairTerm &term) {
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

struct TopologyPairConstantIR {
  std::string name;
  double value;
  int64_t slot = -1;
  bool argument = false;
};

struct TopologyPairIRConfig {
  explicit TopologyPairIRConfig(ExpressionIR expression)
      : expression(std::move(expression)) {}
  ExpressionIR expression;
  std::string termName;
  std::vector<std::string> stems;
  std::array<bool, 6> lookups{};
  bool grouped = false, coulomb = false, switched = false;
  double switchFrom = 0.0;
  mdir::md::Truncation modifier = mdir::md::Truncation::None;
  size_t gatheredCount = 0, sigmaTable = 0, epsilonTable = 0,
         constantsTable = 0;
  std::vector<TopologyPairConstantIR> constants;
  std::vector<std::pair<std::string, size_t>> externalBindings, functionTables;
};

struct BarePairIRConfig {
  explicit BarePairIRConfig(ExpressionIR expression)
      : expression(std::move(expression)) {}
  ExpressionIR expression;
  std::string termName;
  bool coulomb = false, tableParameters = false, switched = false;
  double switchFrom = 0.0;
  mdir::md::Truncation modifier = mdir::md::Truncation::None;
  size_t gatheredCount = 0;
  std::optional<size_t> previousTotal;
  std::vector<std::pair<std::string, double>> constants;
  std::vector<std::pair<Parameter, Mixing>> parameters;
  std::vector<std::pair<std::string, size_t>> tables, functions;
};

static mlir::Value buildTopologyPairContribution(mlir::OpBuilder &builder, mlir::ValueRange values, const TopologyPairIRConfig &config) {
            using namespace mlir;
            auto loc = builder.getUnknownLoc();
            auto *ctx = builder.getContext();
            auto sum = mdir::md::SumRelationOp::create(
                builder, loc, builder.getF64Type(), values[0], values[1],
                values[2], values.slice(3, config.gatheredCount),
                mdir::md::ExchangeAttr::get(ctx, mdir::md::Exchange::Symmetric),
                mdir::md::ExchangeBasisAttr::get(
                    ctx, mdir::md::ExchangeBasis::Asserted),
                mdir::md::TruncationAttr::get(ctx, config.modifier),
                config.switched ? builder.getF64FloatAttr(config.switchFrom)
                                : FloatAttr{});
            llvm::SmallVector<Type> blockTypes = {
                builder.getF64Type(),
                VectorType::get({3}, builder.getF64Type()),
                builder.getI32Type(),
                builder.getI32Type(),
                builder.getF64Type(),
                builder.getF64Type()};
            if (config.grouped)
              blockTypes.append(4, builder.getF64Type());
            blockTypes.append(2 * config.stems.size(), builder.getF64Type());
            llvm::SmallVector<Location> locations(blockTypes.size(), loc);
            OpBuilder::InsertionGuard insertionGuard(builder);
            Block *block = builder.createBlock(&sum.getKernel(), {}, blockTypes,
                                               locations);
            auto args = block->getArguments();
            auto constant = [&](double value) -> Value {
              return arith::ConstantOp::create(builder, loc,
                                               builder.getF64FloatAttr(value));
            };
            auto integer = [&](int64_t value) -> Value {
              return arith::ConstantOp::create(
                  builder, loc, builder.getI32IntegerAttr(value));
            };
            auto mul = [&](Value a, Value b) -> Value {
              return arith::MulFOp::create(builder, loc, a, b);
            };
            auto lookup = [&](Value table, Value a, Value b) -> Value {
              return mdir::md::LookupOp::create(
                  builder, loc, builder.getF64Type(), table, ValueRange{a, b});
            };
            Value angstrom = constant(1.0 / units::length);
            Value kcal = constant(1.0 / units::energy);
            Value r = mul(args[0], angstrom);
            llvm::StringMap<Value> bindings, tables;
            bindings["r"] = r;
            bindings["q1"] = args[4];
            bindings["q2"] = args[5];
            size_t firstStem = 6 + (config.grouped ? 4 : 0);
            for (size_t c = 0; c != config.stems.size(); ++c) {
              bindings[config.stems[c] + "1"] = args[firstStem + 2 * c];
              bindings[config.stems[c] + "2"] = args[firstStem + 2 * c + 1];
            }
            const std::array<StringRef, 6> lookupNames = {
                "sigma", "epsilon", "sigma1", "sigma2", "epsilon1", "epsilon2"};
            for (size_t c = 0; c != lookupNames.size(); ++c) {
              if (!config.lookups[c])
                continue;
              bool sigma = c == 0 || c == 2 || c == 3;
              Value a = c == 3 || c == 5 ? args[3] : args[2];
              Value b = c < 2 ? args[3] : a;
              Value raw = lookup(
                  values[sigma ? config.sigmaTable : config.epsilonTable], a,
                  b);
              bindings[lookupNames[c]] = mul(raw, sigma ? angstrom : kcal);
            }
            if (config.coulomb)
              bindings["coulomb"] = constant(coulombConstant);
            for (const auto &entry : config.constants) {
              if (entry.argument)
                continue;
              Value value;
              if (entry.slot >= 0) {
                Value row = integer(0);
                Value column = integer(entry.slot);
                value = lookup(values[config.constantsTable], row, column);
              } else {
                value = constant(entry.value);
              }
              bindings[entry.name] = value;
            }
            for (const auto &[name, index] : config.externalBindings)
              bindings[name] = values[index];
            for (const auto &[name, index] : config.functionTables)
              tables[name] = values[index];
            Value expression = config.expression.build(builder, bindings,
                                                       tables, config.termName);
            Value kj = constant(units::energy);
            Value energy = mul(expression, kj);
            if (config.grouped) {
              Value ab = mul(args[6], args[9]);
              Value ba = mul(args[7], args[8]);
              Value either = arith::AddFOp::create(builder, loc, ab, ba);
              Value one = constant(1.0);
              Value mask = arith::MinimumFOp::create(builder, loc, either, one);
              energy = mul(energy, mask);
            }
            mdir::md::YieldOp::create(builder, loc, ValueRange{energy});
            return sum.getResult();

}

static TermIRValues buildBarePairContribution(mlir::OpBuilder &builder, mlir::ValueRange values, const BarePairIRConfig &config) {
            using namespace mlir;
            auto loc = builder.getUnknownLoc();
            auto *ctx = builder.getContext();
            auto sum = mdir::md::SumRelationOp::create(
                builder, loc, builder.getF64Type(), values[0], values[1],
                values[2], values.slice(3, config.gatheredCount),
                mdir::md::ExchangeAttr::get(ctx, mdir::md::Exchange::Symmetric),
                mdir::md::ExchangeBasisAttr::get(
                    ctx, mdir::md::ExchangeBasis::Proof),
                mdir::md::TruncationAttr::get(ctx, config.modifier),
                config.switched ? builder.getF64FloatAttr(config.switchFrom)
                                : FloatAttr{});
            {
              OpBuilder::InsertionGuard guard(builder);
              llvm::SmallVector<Type> blockTypes = {
                  builder.getF64Type(),
                  VectorType::get({3}, builder.getF64Type())};
              if (config.tableParameters)
                blockTypes.append(2, builder.getI32Type());
              else
                for (const auto &entry : config.parameters)
                  if (!entry.first.isUniform)
                    blockTypes.append(2, builder.getF64Type());
              llvm::SmallVector<Location> locations(blockTypes.size(), loc);
              Block *block = builder.createBlock(&sum.getKernel(), {},
                                                 blockTypes, locations);
              auto args = block->getArguments();
              auto constant = [&](double value) -> Value {
                return arith::ConstantOp::create(
                    builder, loc, builder.getF64FloatAttr(value));
              };
              auto mul = [&](Value a, Value b) -> Value {
                return arith::MulFOp::create(builder, loc, a, b);
              };
              Value toLength = constant(1.0 / units::length);
              Value r = mul(args[0], toLength);
              llvm::StringMap<Value> bindings, tables;
              bindings["r"] = r;
              for (const auto &[name, value] : config.constants)
                bindings[name] = constant(value);
              if (config.coulomb)
                bindings["coulomb"] = constant(coulombConstant);
              for (const auto &[name, slot] : config.tables)
                bindings[name] = mdir::md::LookupOp::create(
                    builder, loc, builder.getF64Type(), values[slot],
                    ValueRange{args[2], args[3]});
              size_t argument = 2;
              for (const auto &[parameter, mixing] : config.parameters) {
                Value value;
                if (parameter.isUniform)
                  value = constant(parameter.value);
                else {
                  Value a = args[argument++], b = args[argument++];
                  if (mixing == Mixing::Arithmetic) {
                    Value half = constant(0.5);
                    Value sum = arith::AddFOp::create(builder, loc, a, b);
                    value = mul(half, sum);
                  } else {
                    value = mul(a, b);
                    if (mixing != Mixing::Product)
                      value = math::SqrtOp::create(builder, loc, value);
                  }
                }
                bindings[parameter.name] = value;
              }
              for (const auto &[name, slot] : config.functions)
                tables[name] = values[slot];
              Value expression = config.expression.build(
                  builder, bindings, tables, config.termName);
              Value toEnergy = constant(units::energy);
              Value energy = mul(expression, toEnergy);
              mdir::md::YieldOp::create(builder, loc, ValueRange{energy});
            }
            TermIRValues result = {sum.getResult()};
            if (config.previousTotal)
              result.push_back(arith::AddFOp::create(
                  builder, loc, values[*config.previousTotal],
                  sum.getResult()));
            return result;
}

namespace mdir::driver::builder {
class PairTermBuilder final : public TermBuilder, public TermScalarParameters {
public:
  PairTermBuilder() {
    roles.scalarParameters = this;
    potentialMask = alchemicalMask = TermInputs::PairTerms;
    nativePotentialMask = potentialMask;
  }
  unsigned getPotentialMask(const TermInputs &, const DependentTerms *dependent,
                            bool) const override {
    return !dependent || !dependent->pairs.empty() ? potentialMask : 0;
  }
  int parameterRank() const override { return 1; }
  int potentialRank() const override { return -2; }
  llvm::Error collectBareParameters(TermParameters &context) const override {
    const Control &control = context.getControl();
    const System &system = context.getSystem();
    auto &parameters = context.getMutableTypeParameters();
    auto &termTables = context.getMutablePairPlan().tables;
    auto &expressions = context.getMutablePairPlan().expressions;
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
      auto expression = Expression::parse(term.expression, control.functions);
      if (!expression)
        return expression.takeError();
      std::vector<const PairOverride *> overrides = getOverrides(control, term);
      llvm::StringMap<unsigned> tables;

      for (const std::string &name : expression->getNames()) {
        if (name == "r" || name == "coulomb")
          continue;
        if (llvm::any_of(term.constants, [&](auto &constant) {
              return constant.first == name;
            }))
          continue;

        // A parameter of the types. Every type must have it.
        std::vector<double> values;
        for (const ParticleType &type : control.types) {
          auto found = llvm::find_if(type.parameters, [&](auto &entry) {
            return entry.first == name;
          });
          if (found == type.parameters.end())
            return makeError("the expression of '" + term.name + "' uses '" +
                             name +
                             "', which is neither 'r', nor 'coulomb', "
                             "nor a number of the term, nor a parameter of the "
                             "type '" +
                             type.name + "'");
          values.push_back(found->second);
        }
        bool isUniform = llvm::all_of(
            values, [&](double value) { return value == values.front(); });
        if (!isUniform && !term.mixing.count(name))
          return makeError("'" + name +
                           "' differs between the types, but the "
                           "term '" +
                           term.name +
                           "' has no rule of 'mixing' "
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
                  isUniform
                      ? values.front()
                      : mix(term.mixing.lookup(name), values[a], values[b]);
          for (const PairOverride *entry : overrides)
            for (auto &[parameter, value] : entry->parameters)
              if (parameter == name) {
                unsigned a = findType(control, entry->first);
                unsigned b = findType(control, entry->second);
                table.values[a * table.count + b] = value;
                table.values[b * table.count + a] = value;
              }
          tables[name] = context.getTables().size();
          context.getTables().push_back(std::move(table));
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
          parameter.field = context.getFields().size();
          Program::Field field;
          field.name = name;
          for (unsigned type : system.types)
            field.values.push_back(values[type]);
          context.getFields().push_back(std::move(field));
        }
        parameters.push_back(std::move(parameter));
      }

      // Every parameter that an override sets is one that the term uses.
      for (const PairOverride *entry : overrides)
        for (auto &[parameter, value] : entry->parameters)
          if (!tables.count(parameter))
            return makeError("[[energy.nbfix]] sets '" + parameter +
                             "', which is not a parameter of the types that "
                             "the term '" +
                             term.name + "' uses");

      // The types of the particles, which the lookups take.
      if (!tables.empty() &&
          llvm::none_of(context.getFields(), [](const Program::Field &field) {
            return field.isInteger;
          })) {
        Program::Field field;
        field.name = "type";
        field.isInteger = true;
        for (unsigned type : system.types)
          field.values.push_back(type);
        context.getFields().push_back(std::move(field));
      }
      termTables.push_back(std::move(tables));
      expressions.push_back(std::move(*expression));
    }

    return llvm::Error::success();
  }
  llvm::Error collectBareConstants(TermParameters &context) const override {
    return computeDispersion(context);
  }
  void collectParameters(TermParameters &context) const override {
    const Control &control = context.getControl();
    const System &system = context.getSystem();
    // The interaction groups of the pair terms, a flag for each particle
    // (D137).
    for (auto [k, groups] : llvm::enumerate(system.pairGroups))
      for (auto [g, flags] : llvm::enumerate(groups)) {
        Program::Field field;
        field.name = "pg" + std::to_string(k) + (g == 0 ? "a" : "b");
        for (bool flag : flags)
          field.values.push_back(flag ? 1.0 : 0.0);
        context.getFields().push_back(std::move(field));
      }

    // The parameters of each particle that the pair terms take, `w1` and
    // `w2` of `w`, a field gathered for both particles (D165).
    for (const auto &[name, values] : system.particleParameters) {
      bool used = llvm::any_of(control.pairs, [&](const PairTerm &term) {
        Expression expression = llvm::cantFail(
            Expression::parse(term.expression, control.functions));
        return llvm::is_contained(expression.getNames(), name + "1") ||
               llvm::is_contained(expression.getNames(), name + "2");
      });
      if (!used)
        continue;
      Program::Field field;
      field.name = "pp_" + name;
      field.values = values;
      context.getFields().push_back(std::move(field));
    }
  }
  void collectTopologyTables(TermParameters &context) const override {
    const Control &control = context.getControl();
    // The tunable constants of the pair terms (D213), a row that
    // the kernels read in place of constants of their text.
    if (!control.tunableConstants.empty()) {
      Program::Table table;
      table.name = "tunable_constants";
      table.count = 1;
      table.columns = control.tunableConstants.size();
      fillTunableConstants(context, table);
      context.getTables().push_back(std::move(table));
    }
  }
  void refillArrays(TermParameters &context) const override {
    for (Program::Table &table : context.getTables())
      if (table.values.empty())
        fillTunableConstants(context, table);
  }
  static void fillTunableConstants(const TermInputs &context,
                                   Program::Table &table) {
    if (table.name != "tunable_constants")
      return;
    const Control &control = context.getControl();

    // The tunable constants of the pair terms (D213), a row that the
    // kernels read in place of constants of their text.
    table.values.clear();
    for (const auto &[term, name] : control.tunableConstants)
      for (const auto &[key, value] : control.pairs[term].constants)
        if (key == name)
          table.values.push_back(value);
  }
  bool
  needsPairNeighborhood(const TermInputs &context,
                        const PotentialSelection &selection) const override {
    return (selection.terms & TermInputs::PairTerms) &&
           !context.getControl().pairs.empty();
  }
  llvm::Error buildTopologyPotential(const TermInputs &context, mlir::OpBuilder &builder,
      const TermNativeTopology &state, const PotentialSelection &selection,
      llvm::function_ref<void(mlir::Value)> add) const override {
    const Control &control = context.getControl();
    const System &system = context.getSystem();
    bool pairTerms =
        (selection.terms & TermInputs::PairTerms) && !control.pairs.empty();
    int pairTerm = selection.pairTerm;
    bool alchemicalOnly = selection.alchemicalOnly;
    Truncation truncation = state.pairTruncation;
    // The pair terms given by expressions (D137), over the pairs of the
    // topology that are not excluded, truncated at the cutoff as the
    // Lennard-Jones is, in the units of the control file: r, sigma in Å, q
    // in e, epsilon and the energy in kcal/mol. The reader has tested that
    // the expression is symmetric in the two particles, which the kernel
    // then asserts; sigma and
    // epsilon those of the pair, with the pairs of types set apart, and
    // sigma1, epsilon1 those of the type of each particle with itself.
    for (auto [k, term] : llvm::enumerate(control.pairs)) {
      if (!pairTerms || (pairTerm >= 0 && static_cast<int>(k) != pairTerm))
        continue;
      if (!context.includesPair(static_cast<unsigned>(k)))
        continue;
      if (alchemicalOnly && !context.usesLambda(term.expression))
        continue;
      bool grouped =
          k < system.pairGroups.size() && !system.pairGroups[k].empty();
      std::string g = std::to_string(k), set = "pair_" + term.name;
      Expression expression =
          llvm::cantFail(Expression::parse(term.expression, control.functions));
      const std::vector<std::string> &used = expression.getNames();
      auto uses = [&](StringRef name) {
        return llvm::is_contained(used, name);
      };
      // The parameters of each particle that the term takes (D165).
      std::vector<std::string> stems;
      for (const auto &[name, values] : system.particleParameters)
        if (uses(name + "1") || uses(name + "2"))
          stems.push_back(name);
      TopologyPairIRConfig config(expression.prepareIR());
      config.termName = term.name;
      config.grouped = grouped;
      config.stems = stems;
      config.coulomb = uses("coulomb");
      const std::array<StringRef, 6> lookupNames = {
          "sigma", "epsilon", "sigma1", "sigma2", "epsilon1", "epsilon2"};
      for (size_t c = 0; c != lookupNames.size(); ++c)
        config.lookups[c] = uses(lookupNames[c]);
      switch (truncation) {
      case Truncation::None:
      case Truncation::PowerForceSwitch:
      case Truncation::SquaredDistanceSwitch:
        break;
      case Truncation::Shift:
        config.modifier = mdir::md::Truncation::Shift;
        break;
      case Truncation::Switch:
        config.modifier = mdir::md::Truncation::Switch;
        config.switched = true;
        break;
      case Truncation::ForceSwitch:
        config.modifier = mdir::md::Truncation::ForceSwitch;
        config.switched = true;
        break;
      }
      config.switchFrom = control.switchDistance * units::length;
      TermIRValues values{state.physical.neighborhood, state.physical.positions,
          state.physical.cell, state.physical.field("type"), state.physical.field("q")};
      auto operand = [&](mlir::Value value) {
        size_t index = values.size(); values.push_back(value); return index;
      };
      if (grouped) {
        operand(state.physical.field("pg" + g + "a"));
        operand(state.physical.field("pg" + g + "b"));
      }
      for (const auto &stem : stems)
        operand(state.physical.field("pp_" + stem));
      config.gatheredCount = values.size() - 3;
      if (config.lookups[0] || config.lookups[2] || config.lookups[3])
        config.sigmaTable = operand(state.physical.table("lj_sigma"));
      if (config.lookups[1] || config.lookups[4] || config.lookups[5])
        config.epsilonTable = operand(state.physical.table("lj_epsilon"));
      // Empty entries are values produced inside the kernel. Nonempty ones
      // name external values. Apply exactly the former binding precedence.
      llvm::StringMap<mlir::Value> variables;
      variables["r"] = {};
      if (control.usesTime)
        variables["t"] = state.physical.time;
      state.bindLambdas(variables);
      variables["q1"] = {};
      variables["q2"] = {};
      for (const auto &stem : stems) {
        variables[stem + "1"] = {};
        variables[stem + "2"] = {};
      }
      for (size_t c = 0; c != lookupNames.size(); ++c)
        if (config.lookups[c])
          variables[lookupNames[c]] = {};
      if (config.coulomb)
        variables["coulomb"] = {};
      bool tableConstants = false;
      for (const auto &constant : term.constants) {
        TopologyPairConstantIR entry{constant.first, constant.second};
        auto tunable =
            llvm::find(control.tunableConstants,
                       std::pair<unsigned, std::string>(
                           static_cast<unsigned>(k), constant.first));
        if (tunable != control.tunableConstants.end()) {
          entry.slot = tunable - control.tunableConstants.begin();
          entry.argument = context.hasGradientArguments();
        }
        variables[constant.first] =
            entry.argument ? state.gradientConstant(entry.slot) : mlir::Value{};
        tableConstants |= entry.slot >= 0 && !entry.argument;
        config.constants.push_back(std::move(entry));
      }
      if (tableConstants)
        config.constantsTable = operand(state.physical.table("tunable_constants"));
      state.bindObserved(term.name, variables);
      llvm::SmallVector<StringRef> supplied;
      for (const auto &entry : variables)
        supplied.push_back(entry.getKey());
      auto inputs = config.expression.getInputs(supplied);
      for (const auto &name : inputs.names) {
        auto found = variables.find(name);
        assert(found != variables.end() && "missing topology pair binding");
        mlir::Value value = found->second;
        if (value)
          config.externalBindings.emplace_back(name, operand(value));
      }
      for (const auto &name : inputs.tables)
        config.functionTables.emplace_back(name,
                                           operand(state.physical.table(name)));
      add(buildTopologyPairContribution(builder, values, config));
    }

    return llvm::Error::success();
  }
  void emitTopologyPotential(
      TermEmission &context, const PotentialSelection &selection,
      const TermPotentialAddition & add) const override {
    const Control &control = context.getControl();
    const System &system = context.getSystem();
    bool pairTerms =
        (selection.terms & TermInputs::PairTerms) && !control.pairs.empty();
    int pairTerm = selection.pairTerm;
    bool alchemicalOnly = selection.alchemicalOnly;
    Truncation truncation = context.getPairTruncation();
    // The pair terms given by expressions (D137), over the pairs of the
    // topology that are not excluded, truncated at the cutoff as the
    // Lennard-Jones is, in the units of the control file: r, sigma in Å, q
    // in e, epsilon and the energy in kcal/mol. The reader has tested that
    // the expression is symmetric in the two particles, which the kernel
    // then asserts; sigma and
    // epsilon those of the pair, with the pairs of types set apart, and
    // sigma1, epsilon1 those of the type of each particle with itself.
    for (auto [k, term] : llvm::enumerate(control.pairs)) {
      if (!pairTerms || (pairTerm >= 0 && static_cast<int>(k) != pairTerm))
        continue;
      if (!context.includesPair(static_cast<unsigned>(k)))
        continue;
      if (alchemicalOnly && !context.usesLambda(term.expression))
        continue;
      bool grouped =
          k < system.pairGroups.size() && !system.pairGroups[k].empty();
      std::string g = std::to_string(k), set = "pair_" + term.name;
      Expression expression =
          llvm::cantFail(Expression::parse(term.expression, control.functions));
      const std::vector<std::string> &used = expression.getNames();
      auto uses = [&](StringRef name) {
        return llvm::is_contained(used, name);
      };
      // The parameters of each particle that the term takes (D165).
      std::vector<std::string> stems;
      for (const auto &[name, values] : system.particleParameters)
        if (uses(name + "1") || uses(name + "2"))
          stems.push_back(name);
      TopologyPairIRConfig config(expression.prepareIR());
      config.termName = term.name;
      config.grouped = grouped;
      config.stems = stems;
      config.coulomb = uses("coulomb");
      const std::array<StringRef, 6> lookupNames = {
          "sigma", "epsilon", "sigma1", "sigma2", "epsilon1", "epsilon2"};
      for (size_t c = 0; c != lookupNames.size(); ++c)
        config.lookups[c] = uses(lookupNames[c]);
      switch (truncation) {
      case Truncation::None:
      case Truncation::PowerForceSwitch:
      case Truncation::SquaredDistanceSwitch:
        break;
      case Truncation::Shift:
        config.modifier = mdir::md::Truncation::Shift;
        break;
      case Truncation::Switch:
        config.modifier = mdir::md::Truncation::Switch;
        config.switched = true;
        break;
      case Truncation::ForceSwitch:
        config.modifier = mdir::md::Truncation::ForceSwitch;
        config.switched = true;
        break;
      }
      config.switchFrom = control.switchDistance * units::length;
      std::vector<std::string> names = {"%n", "%x", "%cell", "%p_type", "%p_q"};
      llvm::SmallVector<StringRef> types = {"!pairs", "!vec", "!md.cell",
                                            "!ids", "!real"};
      auto operand = [&](std::string name, StringRef type) {
        size_t index = names.size();
        names.push_back(std::move(name));
        types.push_back(type);
        return index;
      };
      if (grouped) {
        operand("%p_pg" + g + "a", "!real");
        operand("%p_pg" + g + "b", "!real");
      }
      for (const auto &stem : stems)
        operand("%p_pp_" + stem, "!real");
      config.gatheredCount = names.size() - 3;
      if (config.lookups[0] || config.lookups[2] || config.lookups[3])
        config.sigmaTable = operand("%t_lj_sigma", "!table");
      if (config.lookups[1] || config.lookups[4] || config.lookups[5])
        config.epsilonTable = operand("%t_lj_epsilon", "!table");
      // Empty entries are values produced inside the kernel. Nonempty ones
      // name external values. Apply exactly the former binding precedence.
      llvm::StringMap<std::string> variables;
      variables["r"] = "";
      if (control.usesTime)
        variables["t"] = "%time";
      context.bindLambdas(variables);
      variables["q1"] = "";
      variables["q2"] = "";
      for (const auto &stem : stems) {
        variables[stem + "1"] = "";
        variables[stem + "2"] = "";
      }
      for (size_t c = 0; c != lookupNames.size(); ++c)
        if (config.lookups[c])
          variables[lookupNames[c]] = "";
      if (config.coulomb)
        variables["coulomb"] = "";
      bool tableConstants = false;
      for (const auto &constant : term.constants) {
        TopologyPairConstantIR entry{constant.first, constant.second};
        auto tunable =
            llvm::find(control.tunableConstants,
                       std::pair<unsigned, std::string>(
                           static_cast<unsigned>(k), constant.first));
        if (tunable != control.tunableConstants.end()) {
          entry.slot = tunable - control.tunableConstants.begin();
          entry.argument = context.hasGradientArguments();
        }
        variables[constant.first] =
            entry.argument ? "%tg_c" + std::to_string(entry.slot) : "";
        tableConstants |= entry.slot >= 0 && !entry.argument;
        config.constants.push_back(std::move(entry));
      }
      if (tableConstants)
        config.constantsTable = operand("%t_tunable_constants", "!grid");
      context.bindObserved(term.name, variables);
      llvm::SmallVector<StringRef> supplied;
      for (const auto &entry : variables)
        supplied.push_back(entry.getKey());
      auto inputs = config.expression.getInputs(supplied);
      for (const auto &name : inputs.names) {
        auto found = variables.find(name);
        assert(found != variables.end() && "missing topology pair binding");
        const std::string &value = found->second;
        if (!value.empty())
          config.externalBindings.emplace_back(name, operand(value, "f64"));
      }
      for (const auto &name : inputs.tables)
        config.functionTables.emplace_back(name,
                                           operand("%t_" + name, "!grid"));
      llvm::SmallVector<StringRef> operands;
      for (const auto &name : names)
        operands.push_back(name);
      std::string result = "%u_" + set;
      context.emitIRFragment(
          "  ", {result}, {"f64"}, operands, types,
          [config](mlir::OpBuilder &builder,
                   mlir::ValueRange values) -> TermIRValues {
            return {buildTopologyPairContribution(builder, values, config)};
          });
      add(set);
    }
  }
  mlir::Value buildBareContributions(const TermInputs &context, mlir::OpBuilder &builder,
      const TermNativePotential &state, mlir::Value total) const override {
    const Control &control = context.getControl();
    const auto &expressions = context.getPairPlan().expressions;
    const auto &termTables = context.getPairPlan().tables;
    const auto &parameters = context.getTypeParameters();
    for (auto [index, term] : llvm::enumerate(control.pairs)) {
      const Expression &expression = expressions[index];
      BarePairIRConfig config(expression.prepareIR());
      config.termName = term.name;
      config.tableParameters = !termTables[index].empty();
      config.coulomb = llvm::is_contained(expression.getNames(), "coulomb");
      for (const auto &entry : term.constants)
        config.constants.push_back(entry);
      TermIRValues values{state.neighborhood, state.positions, state.cell};
      auto operand = [&](mlir::Value value) {
        size_t slot = values.size();
        values.push_back(value);
        return slot;
      };
      llvm::SmallVector<StringRef> supplied = {"r"};
      for (const auto &entry : config.constants)
        supplied.push_back(entry.first);
      if (config.coulomb)
        supplied.push_back("coulomb");
      if (config.tableParameters)
        operand(state.field("type"));
      else {
        for (const Parameter &parameter : parameters) {
          if (!llvm::is_contained(expression.getNames(), parameter.name))
            continue;
          config.parameters.emplace_back(parameter,
                                         term.mixing.lookup(parameter.name));
          if (!parameter.isUniform)
            operand(state.field(parameter.name));
        }
      }
      config.gatheredCount = values.size() - 3;
      for (const auto &[name, tableIndex] : termTables[index]) {
        const auto &table = context.getParameterTables()[tableIndex];
        config.tables.emplace_back(name.str(),
                                   operand(state.table(table.name)));
      }
      for (const auto &entry : config.tables)
        supplied.push_back(entry.first);
      for (const auto &entry : config.parameters)
        supplied.push_back(entry.first.name);
      auto inputs = config.expression.getInputs(supplied);
      for (const auto &name : inputs.tables)
        config.functions.emplace_back(name, operand(state.table(name)));
      if (total)
        config.previousTotal = operand(total);
      config.switchFrom = control.switchDistance * units::length;
      switch (control.truncation) {
      case Truncation::None:
      case Truncation::PowerForceSwitch:
      case Truncation::SquaredDistanceSwitch:
        break;
      case Truncation::Shift:
        config.modifier = mdir::md::Truncation::Shift;
        break;
      case Truncation::Switch:
        config.modifier = mdir::md::Truncation::Switch;
        config.switched = true;
        break;
      case Truncation::ForceSwitch:
        config.modifier = mdir::md::Truncation::ForceSwitch;
        config.switched = true;
        break;
      }
      total = buildBarePairContribution(builder, values, config).back();
    }
    return total;
  }

  void appendTermDefinitions(
      const TermInputs &context,
      std::vector<TermDefinition> &definitions) const override {
    for (size_t k = 0, e = context.getControl().pairs.size(); k != e; ++k)
      definitions.push_back({15, k, "term_pair" + std::to_string(k),
                             TermInputs::PairTerms, -1, static_cast<int>(k)});
  }
  ObservedSelection findObservedTerm(const TermInputs &context,
                                     StringRef name) const override {
    for (auto [k, term] : llvm::enumerate(context.getControl().pairs))
      if (term.name == name)
        return {TermInputs::PairTerms,
                static_cast<int>(k),
                {TermInputs::PairTerms, -1, static_cast<int>(k), -1, false}};
    return {};
  }
  std::optional<double> getObservedValue(const TermInputs &context,
                                         unsigned kind, int index,
                                         StringRef name) const override {
    if (kind == TermInputs::PairTerms)
      for (const auto &[key, value] :
           context.getControl().pairs[index].constants)
        if (key == name)
          return value;
    return std::nullopt;
  }
  llvm::Error
  collectDispersionConstants(TermParameters &context) const override {
    const System &system = context.getSystem();
    bool shifted = context.getControl().truncation == Truncation::Shift;
    // The tails of the pair terms (D209).
    if (llvm::Error error = collectPairTails(context))
      return error;
    for (unsigned index = 0, e = system.pairTails.size(); index != e; ++index) {
      PairTermTail tail = llvm::cantFail(getCustomPairTail(context, index, {}));
      context.addDispersion(tail.energy + (shifted ? tail.shift : 0.0),
                            shifted ? tail.shiftFixed : 0.0, tail.virial);
    }

    return llvm::Error::success();
  }
  void appendInitialConstants(
      const TermInputs &context,
      llvm::function_ref<void(std::string, double)> add) const override {
    const auto &control = context.getControl();
    for (auto [k, constant] : llvm::enumerate(control.tunableConstants))
      for (const auto &[name, value] : control.pairs[constant.first].constants)
        if (name == constant.second)
          add("tg_c" + std::to_string(k), value);
  }
  void collectObservedConstants(TermParameters &context) const override {
    const Control &control = context.getControl();
    const System &system = context.getSystem();
    for (auto [column, observable] : llvm::enumerate(control.observables))
      for (auto [index, term] : llvm::enumerate(control.pairs)) {
        if (term.name != observable.term || index >= system.pairTails.size())
          continue;
        if (observable.constant.empty()) {
          PairTermTail tail =
              llvm::cantFail(getCustomPairTail(context, index, {}));
          context.getObservableVolumeConstants()[column] =
              tail.energy + tail.shift;
          context.getObservableFixedConstants()[column] = tail.shiftFixed;
          continue;
        }
        double value = 0.0;
        for (const auto &[name, v] : term.constants)
          if (name == observable.constant)
            value = v;
        context.getObservableVolumeConstants()[column] =
            getCustomPairTailDerivative(context, index, observable.constant,
                                        value, true, false);
        context.getObservableFixedConstants()[column] =
            getCustomPairTailDerivative(context, index, observable.constant,
                                        value, /*shift=*/true, /*fixed=*/true);
      }
  }
  void collectGradientShiftConstants(TermParameters &context) const override {
    const System &system = context.getSystem();
    for (unsigned index = 0, e = system.pairTails.size(); index != e; ++index) {
      PairTermTail pair = llvm::cantFail(getCustomPairTail(context, index, {}));
      context.addGradientShift(pair.shift, pair.shiftFixed);
    }
  }
  llvm::Expected<std::optional<Program::GradientOutcome>>
  collectTunableOutcome(TermParameters &context, unsigned k) const override {
    const Control &control = context.getControl();
    const System &system = context.getSystem();
    const auto &declaration = control.tunableDeclarations[k];
    if (declaration.kind != Control::TunableKind::PairConstant)
      return std::optional<Program::GradientOutcome>();
    bool dispersion = control.topologyDispersion != DispersionCorrection::None;
    // A constant that the expression of its term does not read enters
    // no value of the program.
    unsigned index = declaration.termIndex;
    const PairTerm &term = control.pairs[index];
    Expression expression =
        llvm::cantFail(Expression::parse(term.expression, control.functions));
    if (!llvm::is_contained(expression.getNames(), declaration.parameter))
      return std::optional(Program::GradientOutcome::Zero);

    auto column = llvm::find(
        control.tunableConstants,
        std::pair<unsigned, std::string>(index, declaration.parameter));
    double value = 0.0;
    for (const auto &[name, v] : term.constants)
      if (name == declaration.parameter)
        value = v;
    // The tail of the term and the estimate of its shift, which the
    // host adds to the energy, by the quadrature that computes them
    // (D209, D210).
    Program::GradientSlot slot;
    slot.tunable = static_cast<unsigned>(k);
    if (dispersion && index < context.getPairPlan().tailExpressions.size() &&
        index < system.pairTails.size())
      slot.volumeConstant = getCustomPairTailDerivative(
          context, index, declaration.parameter, value, true, false);
    if (dispersion && index < context.getPairPlan().tailExpressions.size() &&
        index < system.pairTails.size())
      slot.fixedConstant = getCustomPairTailDerivative(
          context, index, declaration.parameter, value,
          /*shift=*/true, /*fixed=*/true);
    if (!std::isfinite(slot.volumeConstant) ||
        !std::isfinite(slot.fixedConstant))
      return makeError(
          "the derivative of the tail of the pair term '" + term.name +
          "' in its constant '" + declaration.parameter +
          "' cannot be integrated, for the tunable '" + declaration.name + "'");
    context.getGradientSlots().push_back(slot);
    context.getGradientColumns().push_back(
        static_cast<unsigned>(column - control.tunableConstants.begin()));

    return std::optional(Program::GradientOutcome::Rule);
  }

private:
  static llvm::Error collectPairTails(TermParameters &);
  static llvm::Error
  collectPairValues(const TermInputs &, unsigned,
                    std::vector<std::vector<llvm::StringMap<double>>> &);
  static llvm::Error computeDispersion(TermParameters &);
};
} // namespace mdir::driver::builder

std::unique_ptr<TermBuilder> mdir::driver::builder::createPairTermBuilder() {
  return std::make_unique<PairTermBuilder>();
}

llvm::Error PairTermBuilder::collectPairValues(
    const TermInputs &context, unsigned index,
    std::vector<std::vector<llvm::StringMap<double>>> &values) {
  const Control &control = context.getControl();
  const auto &parameters = context.getTypeParameters();
  const auto &termTables = context.getPairPlan().tables;
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
        pair[parameter.name] = parameter.isUniform
                                   ? parameter.value
                                   : mix(term.mixing.lookup(parameter.name),
                                         valueOf(a), valueOf(b));
      }
      for (auto &[name, table] : termTables[index])
        pair[name] = context.getParameterTables()[table].values[a * count + b];
    }
  return llvm::Error::success();
}

llvm::Error PairTermBuilder::computeDispersion(TermParameters &context) {
  const Control &control = context.getControl();
  const System &system = context.getSystem();
  const auto &expressions = context.getPairPlan().expressions;
  bool corrects = llvm::any_of(control.pairs, [](const PairTerm &term) {
    return term.dispersion != DispersionCorrection::None;
  });
  if (!corrects)
    return llvm::Error::success();
  if (control.truncation != Truncation::None)
    return makeError("'dispersion_correction' needs a plain cutoff: "
                     "'switch_distance' equal to 'cutoff', and no "
                     "'lennard_jones_modifier'");

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
    if (llvm::Error error = collectPairValues(context, index, values))
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
        energy +=
            -2.0 * M_PI / (3.0 * volume) * scale * pairs * far / (rc * rc * rc);
      }
  }
  context.setBareDispersion(energy * units::energy,
                            6.0 * energy * units::energy);
  return llvm::Error::success();
}

/// The integral of `f` over [0, 1] by adaptive Gauss–Kronrod quadrature
/// of 7 and 15 points [Piessens1983], to an error of 1e-13 of the whole or
/// of the integral of `size`, the size of the terms that `f` sums (#155);
/// NaN if it reaches neither. A difference of two terms that nearly cancel,
/// as an NBFIX correction at the force field's own parameters, has a value
/// at the level of their rounding, which no relative error of the value
/// could reach. The rule of 15 points is exact for polynomials of degree 22
/// and that of 7 for degree 13, so a polynomial of degree 13 or less takes
/// one interval and its value to round-off.
static double integrateUnit(llvm::function_ref<double(double)> f,
                            llvm::function_ref<double(double)> size) {
  static const double xk[8] = {
      0.991455371120812639206854697526329, 0.949107912342758524526189684047851,
      0.864864423359769072789712788640926, 0.741531185599394439863864773280788,
      0.586087235467691130294144845693013, 0.405845151377397166906606412076961,
      0.207784955007898467600689403773245, 0.0};
  static const double wk[8] = {
      0.022935322010529224963732008058970, 0.063092092629978553290700663189204,
      0.104790010322250183839876322541518, 0.140653259715525918745189590510238,
      0.169004726639267902826583426598550, 0.190350578064785409913256402421014,
      0.204432940075298892414161999234649, 0.209482141084727828012999174891714};
  static const double wg[4] = {
      0.129484966168869693270611432679082, 0.279705391489276667901467771423780,
      0.381830050505118944950369775488975, 0.417959183673469387755102040816327};
  struct Interval {
    double a, b, value, error, size;
  };
  auto rule = [&](double a, double b) {
    double center = 0.5 * (a + b), half = 0.5 * (b - a);
    double fc = f(center);
    double kronrod = wk[7] * fc, gauss = wg[3] * fc;
    double sized = wk[7] * size(center);
    for (int k = 0; k != 7; ++k) {
      double left = center - half * xk[k], right = center + half * xk[k];
      double sum = f(left) + f(right);
      kronrod += wk[k] * sum;
      sized += wk[k] * (size(left) + size(right));
      if (k % 2 == 1)
        gauss += wg[k / 2] * sum;
    }
    return Interval{a, b, kronrod * half, std::fabs((kronrod - gauss) * half),
                    sized * half};
  };
  std::vector<Interval> intervals = {rule(0.0, 1.0)};
  for (int iteration = 0; iteration != 4000; ++iteration) {
    double value = 0.0, error = 0.0, sized = 0.0;
    for (const Interval &interval : intervals) {
      value += interval.value;
      error += interval.error;
      sized += interval.size;
    }
    if (!std::isfinite(value) || !std::isfinite(error))
      return std::nan("");
    if (error <= 1.0e-13 * std::fmax(std::fabs(value), sized) || error == 0.0)
      return value;
    auto worst = std::max_element(
        intervals.begin(), intervals.end(),
        [](const Interval &x, const Interval &y) { return x.error < y.error; });
    double a = worst->a, b = worst->b, middle = 0.5 * (a + b);
    if (!(middle > a && middle < b))
      return std::nan("");
    *worst = rule(a, middle);
    intervals.push_back(rule(middle, b));
  }
  return std::nan("");
}

llvm::Error PairTermBuilder::collectPairTails(TermParameters &context) {
  const Control &control = context.getControl();
  auto &tailExpressions = context.getMutablePairPlan().tailExpressions;
  // The pairs that each term counts, from the system
  // (D209); a term that the correction leaves out
  // has none.
  for (const PairTerm &term : control.pairs)
    tailExpressions.push_back(
        llvm::cantFail(Expression::parse(term.expression, control.functions)));
  for (auto [index, term] : llvm::enumerate(control.pairs)) {
    auto tail = getCustomPairTail(context, index, {});
    if (!tail) {
      llvm::consumeError(tail.takeError());
      return makeError("the correction for the dispersion integrates the "
                       "pair term '" +
                       term.name +
                       "' beyond the cutoff, "
                       "and its tail cannot be integrated; give "
                       "'dispersion_correction = \"NONE\"' in the term to "
                       "leave it out");
    }
  }
  return llvm::Error::success();
}

llvm::Expected<PairTermTail> mdir::driver::builder::getCustomPairTail(
    const TermInputs &context, unsigned index,
    const llvm::StringMap<double> &changes) {
  const Control &control = context.getControl();
  const System &system = context.getSystem();
  const auto &tailExpressions = context.getPairPlan().tailExpressions;
  auto &tailIntegrals = context.getPairPlan().tailIntegrals;
  // The tail of a pair term goes with the term
  // (D243).
  if (!context.includesPair(index))
    return PairTermTail();
  // With a uniform density beyond the cutoff [AllenTildesley2017], each
  // unordered pair adds (4π / V) I, I = ∫_rc^∞ r² u(r) dr, to the energy
  // and (4π / V)(3 I + rc³ u(rc)) to the trace of the virial, which is
  // −(4π / V) ∫_rc^∞ r³ u'(r) dr integrated by parts; for −C6 / r⁶ that is
  // 6 times the energy, as for the topology's correction. The sum takes
  // the factor N² / P of the topology's correction, P the ordered pairs
  // that are not excluded, so that both follow one convention. In
  // s = rc / r, I = rc³ ∫_0^1 u(rc / s) s⁻⁴ ds, whose integrand is a
  // polynomial in s for a sum of powers r⁻ᵏ, k ≥ 4, which the quadrature
  // integrates exactly. A shift to 0 at the cutoff takes u(r_c) from each
  // pair within it, at a uniform density (4π / 3V) r_c³ u(r_c) for each
  // pair, with the same factor, times getShiftFactor() for the particle
  // itself (D210).
  const Expression &expression = tailExpressions[index];
  double rc = control.cutoffDistance;
  double energy = 0.0, virial = 0.0, shift = 0.0;
  if (tailIntegrals.size() <= index)
    tailIntegrals.resize(index + 1);
  for (const System::TailPair &pair : system.pairTails[index]) {
    llvm::StringMap<double> values = pair.values;
    for (const auto &entry : changes)
      if (values.count(entry.getKey()))
        values[entry.getKey()] = entry.getValue();
    // The integral is a function of the values of the names that the
    // expression reads, in their order, and of nothing else of the pair:
    // one that was computed for these values is taken, the same number to
    // the bit.
    std::vector<uint64_t> key;
    for (const std::string &name : expression.getNames()) {
      auto found = values.find(name);
      uint64_t bits = 0;
      if (found != values.end())
        std::memcpy(&bits, &found->second, sizeof bits);
      key.push_back(bits);
      key.push_back(found != values.end());
    }
    auto [memo, computes] = tailIntegrals[index].try_emplace(std::move(key));
    PairTermPlan::TailIntegral &tail = memo->second;
    if (computes && hasFiniteTail(expression, values, rc)) {
      auto u = [&](double r) {
        values["r"] = r;
        return expression.evaluate(values);
      };
      auto size = [&](double r) {
        values["r"] = r;
        return expression.evaluateMagnitude(values);
      };
      tail.finite = true;
      tail.integral = integrateUnit(
          [&](double x) {
            if (x == 0.0)
              return 0.0;
            return rc * rc * rc * u(rc / x) / (x * x * x * x);
          },
          [&](double x) {
            if (x == 0.0)
              return 0.0;
            return rc * rc * rc * size(rc / x) / (x * x * x * x);
          });
      tail.edge = rc * rc * rc * u(rc);
    }
    if (!tail.finite)
      return makeError("diverges");
    double integral = tail.integral, edge = tail.edge;
    if (!std::isfinite(integral) || !std::isfinite(edge))
      return makeError("cannot be integrated");
    energy += pair.count * integral;
    virial += pair.count * (3.0 * integral + edge);
    shift += pair.count * edge / 3.0;
  }
  const Topology &topology = *system.topology;
  double n = static_cast<double>(topology.getNumParticles());
  double ordered = n * (n - 1.0) - 2.0 * topology.exclusions.size();
  double scale = ordered > 0.0 ? n * n / ordered : 0.0;
  double volume = system.box[0] * system.box[1] * system.box[2] /
                  (units::length * units::length * units::length);
  double factor = 4.0 * M_PI / volume * scale * units::energy;
  PairTermTail tail;
  tail.energy = factor * energy;
  tail.virial = factor * virial;
  tail.shift = factor * shift;
  tail.shiftFixed = factor * shift * context.getShiftFixedFactor();
  return tail;
}

double mdir::driver::builder::getCustomPairTails(
    const TermInputs &context, const llvm::StringMap<double> &changes,
    bool fixed) {
  const System &system = context.getSystem();
  // A tail that a state makes diverge is NaN in the output, not 0.
  double sum = 0.0;
  for (unsigned index = 0, e = system.pairTails.size(); index != e; ++index) {
    auto tail = getCustomPairTail(context, index, changes);
    if (!tail) {
      llvm::consumeError(tail.takeError());
      return std::nan("");
    }
    sum += fixed ? tail->shiftFixed : tail->energy + tail->shift;
  }
  return sum;
}

double mdir::driver::builder::getCustomPairTailDerivative(
    const TermInputs &context, unsigned index, StringRef name, double value,
    bool shift, bool fixed) {
  const auto &tailExpressions = context.getPairPlan().tailExpressions;
  if (!llvm::is_contained(tailExpressions[index].getNames(), name))
    return 0.0;
  // Central differences extrapolated to h → 0 (Richardson); where a side
  // cannot be evaluated (a λ at 0 of an expression in √λ), the one-sided
  // differences of second order.
  auto at = [&](double x) {
    llvm::StringMap<double> changes;
    changes[name] = x;
    auto tail = getCustomPairTail(context, index, changes);
    if (!tail) {
      llvm::consumeError(tail.takeError());
      return std::nan("");
    }
    if (fixed)
      return tail->shiftFixed;
    return tail->energy + (shift ? tail->shift : 0.0);
  };
  double h = 1.0e-3 * std::max(std::fabs(value), 1.0);
  auto central = [&](double step) {
    return (at(value + step) - at(value - step)) / (2.0 * step);
  };
  double d = (4.0 * central(0.5 * h) - central(h)) / 3.0;
  if (std::isfinite(d))
    return d;
  for (double sign : {1.0, -1.0}) {
    double step = sign * 1.0e-4 * std::max(std::fabs(value), 1.0);
    double one =
        (-3.0 * at(value) + 4.0 * at(value + step) - at(value + 2.0 * step)) /
        (2.0 * step);
    if (std::isfinite(one))
      return one;
  }
  return std::nan("");
}
