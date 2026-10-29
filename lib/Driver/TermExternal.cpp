// The external terms: parameter fields, potentials, and energy output
// (D359). Entry buffers follow the shared field list.

#include "BuilderImpl.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include <array>

using namespace mdir::driver;
using namespace mdir::driver::builder;

struct ExternalIRConfig {
  explicit ExternalIRConfig(ExpressionIR expression)
      : expression(std::move(expression)) {}
  ExpressionIR expression;
  std::string termName;
  bool frame = false, charges = false;
  std::array<double, 3> inputBox{};
  size_t gatheredCount = 0;
  std::vector<std::string> parameters;
  std::vector<std::pair<std::string, double>> constants;
  std::vector<std::pair<std::string, size_t>> externalBindings, functionTables;
};

namespace mdir {
namespace driver {
namespace builder {

class ExternalTermBuilder final : public TermBuilder {
public:
  ExternalTermBuilder() {
    potentialMask = TermInputs::ExternalTerms;
    alchemicalMask = TermInputs::ExternalTerms;
  }
  void collectParameters(TermParameters &context) const override {
    const Topology &topology = context.getTopology();
    size_t count = topology.getNumParticles();
    auto &fields = context.getFields();
    // The terms of the absolute positions (D148): a flag for each particle,
    // 1 for those of the term, and each parameter given as a list.
    for (auto [k, term] : llvm::enumerate(topology.externalTerms)) {
      std::string flag = context.indexedName("ext", k);
      Program::Field on;
      on.name = flag;
      on.values.assign(count, 0.0);
      for (unsigned particle : term.particles)
        on.values.edit()[particle] = 1.0;
      fields.push_back(std::move(on));
      for (const auto &[name, values] : term.parameters) {
        Program::Field field;
        field.name = flag + "_" + name;
        field.values.assign(count, 0.0);
        for (auto [place, particle] : llvm::enumerate(term.particles))
          field.values.edit()[particle] = values[place];
        fields.push_back(std::move(field));
      }
    }
  }

  void emitTopologyPotential(
      TermEmission &context, const PotentialSelection &selection,
      const TermPotentialAddition & add) const override {
    if (selection.terms & TermInputs::ExternalTerms)
      for (auto [index, term] :
           llvm::enumerate(context.getTopology().externalTerms))
        if ((selection.externalTerm < 0 ||
             static_cast<int>(index) == selection.externalTerm) &&
            !(selection.alchemicalOnly && !context.usesLambda(term.expression)))
          emitExternalTerm(context, index, term, add);
  }



  void appendTermDefinitions(
      const TermInputs &context,
      std::vector<TermDefinition> &definitions) const override {
    for (size_t k = 0, e = context.getTopology().externalTerms.size(); k != e;
         ++k)
      definitions.push_back({17, k, "term_external" + std::to_string(k),
                             TermInputs::ExternalTerms, -1, -1,
                             static_cast<int>(k)});
  }

  ObservedSelection findObservedTerm(const TermInputs &context,
                                     StringRef name) const override {
    for (auto [index, term] :
         llvm::enumerate(context.getTopology().externalTerms))
      if (term.name == name)
        return {TermInputs::ExternalTerms, static_cast<int>(index),
                {TermInputs::ExternalTerms, -1, -1, static_cast<int>(index), false}};
    return {};
  }

  std::optional<double> getObservedValue(const TermInputs &context, unsigned kind,
                                        int index, StringRef name) const override {
    if (kind == TermInputs::ExternalTerms)
      for (const auto &[key, value] :
           context.getTopology().externalTerms[index].constants)
        if (key == name)
          return value;
    return std::nullopt;
  }

private:
  static void emitExternalTerm(TermEmission &context, size_t index,
                              const ExternalTerm &term,
                              const TermPotentialAddition & add);
};

} // namespace builder
} // namespace driver
} // namespace mdir

std::unique_ptr<TermBuilder> mdir::driver::builder::createExternalTermBuilder() {
  return std::make_unique<ExternalTermBuilder>();
}

void ExternalTermBuilder::emitExternalTerm(
    TermEmission &context, size_t index, const ExternalTerm &term,
    const TermPotentialAddition & add) {
  const Control &control = context.getControl();
  const System &system = context.getSystem();
  Expression expression =
      llvm::cantFail(Expression::parse(term.expression, control.functions));
  ExternalIRConfig config(expression.prepareIR());
  config.termName = term.name;
  config.charges = llvm::is_contained(expression.getNames(), "q");
  config.frame = term.scaling == ExternalTerm::Scaling::Cell;
  for (int k = 0; k != 3; ++k)
    config.inputBox[k] =
        system.inputBox[k] > 0.0 ? system.inputBox[k] : system.box[k];
  std::string flag = context.indexedName("ext", index);
  std::vector<std::string> names = {"%x", "%p_" + flag};
  llvm::SmallVector<StringRef> types = {"!vec", "!real"};
  auto operand = [&](std::string name, StringRef type) {
    size_t slot = names.size();
    names.push_back(std::move(name));
    types.push_back(type);
    return slot;
  };
  if (config.charges)
    operand("%p_q", "!real");
  llvm::StringMap<std::string> variables;
  for (char component : StringRef("xyz"))
    variables[std::string(1, component)] = "";
  if (config.charges)
    variables["q"] = "";
  for (const auto &parameter : term.parameters) {
    config.parameters.push_back(parameter.first);
    operand("%p_" + flag + "_" + parameter.first, "!real");
    variables[parameter.first] = "";
  }
  config.gatheredCount = names.size();
  for (const auto &constant : term.constants) {
    config.constants.push_back(constant);
    variables[constant.first] = "";
  }
  // The cell operand follows gathered fields; it is absent for a fixed frame.
  if (config.frame)
    operand("%cell", "!md.cell");
  if (control.usesTime)
    variables["t"] = "%time";
  context.bindLambdas(variables);
  context.bindObserved(term.name, variables);
  llvm::SmallVector<StringRef> supplied;
  for (const auto &entry : variables)
    supplied.push_back(entry.getKey());
  auto inputs = config.expression.getInputs(supplied);
  for (const auto &name : inputs.names) {
    auto found = variables.find(name);
    assert(found != variables.end() && "missing external expression binding");
    if (!found->second.empty())
      config.externalBindings.emplace_back(name, operand(found->second, "f64"));
  }
  for (const auto &name : inputs.tables)
    config.functionTables.emplace_back(name, operand("%t_" + name, "!grid"));
  llvm::SmallVector<StringRef> operands;
  for (const auto &name : names)
    operands.push_back(name);
  std::string result = "%u_external_" + term.name;
  context.emitIRFragment(
      "  ", {result}, {"f64"}, operands, types,
      [config](mlir::OpBuilder &builder,
               mlir::ValueRange values) -> TermIRValues {
        using namespace mlir;
        auto loc = builder.getUnknownLoc();
        auto f64 = builder.getF64Type();
        auto vec = VectorType::get({3}, f64);
        auto constant = [&](double value) -> Value {
          return arith::ConstantOp::create(builder, loc,
                                           builder.getF64FloatAttr(value));
        };
        Value frame;
        if (config.frame) {
          Value edges = mdir::md_exec::CellEdgesOp::create(
              builder, loc, vec, values[config.gatheredCount]);
          Value input = arith::ConstantOp::create(
              builder, loc,
              DenseFPElementsAttr::get(
                  vec, llvm::ArrayRef<double>(config.inputBox)));
          frame = arith::DivFOp::create(builder, loc, input, edges);
        }
        auto sum = mdir::md::SumParticlesOp::create(
            builder, loc, f64, values.take_front(config.gatheredCount));
        {
          OpBuilder::InsertionGuard guard(builder);
          llvm::SmallVector<Type> blockTypes(config.gatheredCount, f64);
          blockTypes[0] = vec;
          llvm::SmallVector<Location> locations(blockTypes.size(), loc);
          Block *block =
              builder.createBlock(&sum.getKernel(), {}, blockTypes, locations);
          auto args = block->getArguments();
          Value toLength = constant(1.0 / units::length);
          Value position = args[0];
          if (config.frame)
            position = arith::MulFOp::create(builder, loc, position, frame);
          llvm::StringMap<Value> bindings, tables;
          for (int c = 0; c != 3; ++c) {
            Value component =
                vector::ExtractOp::create(builder, loc, position, c);
            bindings[std::string(1, "xyz"[c])] =
                arith::MulFOp::create(builder, loc, component, toLength);
          }
          size_t slot = 2;
          if (config.charges)
            bindings["q"] = args[slot++];
          for (const auto &name : config.parameters)
            bindings[name] = args[slot++];
          for (const auto &[name, value] : config.constants)
            bindings[name] = constant(value);
          for (const auto &[name, slot] : config.externalBindings)
            bindings[name] = values[slot];
          for (const auto &[name, slot] : config.functionTables)
            tables[name] = values[slot];
          Value expression = config.expression.build(builder, bindings, tables,
                                                     config.termName);
          Value toEnergy = constant(units::energy);
          Value energy =
              arith::MulFOp::create(builder, loc, expression, toEnergy);
          Value zero = constant(0.0);
          Value active = arith::CmpFOp::create(
              builder, loc, arith::CmpFPredicate::ONE, args[1], zero);
          Value selected =
              arith::SelectOp::create(builder, loc, active, energy, zero);
          mdir::md::YieldOp::create(builder, loc, ValueRange{selected});
        }
        return {sum.getResult()};
      });
  add("external_" + term.name);
}
