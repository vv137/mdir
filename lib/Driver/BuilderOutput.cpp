// Term-energy, free-energy, observable, tunable-gradient, and solvent outputs.
#include "BuilderImpl.h"
#include "OutputIR.h"

using namespace mdir::driver;
using namespace mdir::driver::builder;

namespace {
// Connections to the remaining assembly are confined to the builder wrapper.
// Native output construction receives physical state and owned scalar config.
struct OutputConnections {
  std::vector<std::string> values, types;
  size_t fields;
  bool timed;

  OutputConnections(const Program &program, StringRef positions, StringRef cell,
                    StringRef prefix, StringRef time, bool timed)
      : timed(timed) {
    add(positions, "!vec");
    add(cell, "!md.cell");
    for (const auto &field : program.fields)
      add((prefix + field.name).str(), field.isInteger ? "!ids" : "!real");
    for (const auto &table : program.tables)
      add("%t_" + table.name, table.getType());
    std::string relations = ("%r" + prefix.drop_front(2)).str();
    for (const auto &set : program.tupleSets) {
      add(relations + set.name, "!rel_" + set.name);
      for (const auto &field : set.fields)
        add("%f_" + set.name + "_" + field.name, "!of_" + set.name);
    }
    fields = values.size() - 2;
    if (timed)
      add(time, "f64");
  }
  void add(StringRef value, StringRef type) {
    values.push_back(value.str());
    types.push_back(type.str());
  }
  llvm::SmallVector<StringRef> names() const {
    return {values.begin(), values.end()};
  }
  llvm::SmallVector<StringRef> signatures() const {
    return {types.begin(), types.end()};
  }
};

OutputStateIR state(mlir::ValueRange values, size_t fields, bool timed) {
  return {values[0], values[1], values.slice(2, fields),
          timed ? values[2 + fields] : mlir::Value{}};
}
} // namespace

void Builder::emitFreeEnergyOutput(StringRef indent, StringRef x,
                                   StringRef cell, StringRef prefix,
                                   StringRef step, StringRef time) {
  if (control.freeEnergyFile.empty())
    return;
  // Derivatives at the run's state, then energies in state order. A single
  // state has no relative energy to evaluate (D161, D190).
  FreeEnergyOutputConfig config{{}, static_cast<size_t>(control.freeEnergy.state),
                                  getNumPotentialArguments()};
  for (const auto &component : control.freeEnergy.lambdas)
    config.lambdas.push_back(component.second);
  OutputConnections input(program, x, cell, prefix, time, control.usesTime);
  size_t tail = input.values.size();
  input.add(step, "i64");
  bool loop = control.freeEnergy.getNumStates() > 1;
  if (loop) {
    input.add("%c0", "index");
    input.add("%c1", "index");
  }
  emitIRFragment(
      indent, {}, {}, input.names(), input.signatures(),
      [config = std::move(config), fields = input.fields, timed = input.timed,
       tail, loop](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        buildFreeEnergyOutput(builder, state(values, fields, timed), values[tail],
                                loop ? values[tail + 1] : mlir::Value{},
                                loop ? values[tail + 2] : mlir::Value{}, config);
        return {};
      });
}

void Builder::emitObservablesOutput(StringRef indent, StringRef x,
                                    StringRef cell, StringRef prefix,
                                    StringRef step, StringRef time) {
  std::vector<ObservedTerm> terms = getObservedTerms();
  if (terms.empty())
    return;
  OutputConnections input(program, x, cell, prefix, time, control.usesTime);
  size_t tail = input.values.size();
  input.add(step, "i64");
  std::vector<ObservableOutputTermIR> config;
  unsigned tunables = 0;
  for (auto [k, term] : llvm::enumerate(terms)) {
    ObservableOutputTermIR output;
    output.potential = "observe" + std::to_string(k);
    for (const auto &constant : term.constants) {
      // Tunables are entry values; updates leave the program unchanged (D232).
      bool tunable = control.isObservedTunable(term.name, constant);
      output.constants.push_back({tunable ? 0.0 : getObservedValue(term, constant),
                                  tunable ? std::optional<unsigned>(tunables++)
                                          : std::nullopt});
      if (tunable)
        input.add("%obv_" + term.name + "_" + constant, "f64");
    }
    for (auto column : term.columns)
      output.columns.push_back({column.first, column.second});
    config.push_back(std::move(output));
  }
  emitIRFragment(
      indent, {}, {}, input.names(), input.signatures(),
      [config = std::move(config), fields = input.fields, timed = input.timed,
       tail, count = control.observables.size(), base = getNumPotentialArguments()](
          mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        buildObservablesOutput(builder, state(values, fields, timed), values[tail],
                                 count, base, config, values.drop_front(tail + 1));
        return {};
      });
}

void Builder::emitTunableGradientOutput(StringRef indent, StringRef x,
                                        StringRef cell, StringRef prefix,
                                        StringRef time) {
  if (!program.tunableGradient)
    return;
  // A one-iteration loop only when the host asks. Runtime gradients never
  // introduce work into an ordinary run.
  OutputConnections input(program, x, cell, prefix, time, control.usesTime);
  size_t tail = input.values.size();
  input.add("%tunable_gradient", "f64");
  input.add("%c0", "index");
  input.add("%c1", "index");
  size_t constants = control.tunableConstants.size(), seeds = getNumGradientSeeds();
  for (size_t c = 0; c != constants; ++c)
    input.add("%tg_c" + std::to_string(c), "f64");
  for (size_t c = 0; c != seeds; ++c)
    input.add((prefix + "tg_zero").str(), "!real");
  bool fieldResults = !program.gradientFields.empty();
  if (fieldResults)
    input.add(idName, "!ids");
  unsigned base = getNumPotentialArguments(), chargeArgument = 2;
  for (const auto &field : program.fields) {
    if (field.name == "q")
      break;
    ++chargeArgument;
  }
  TunableGradientOutputConfig config;
  program.gradientArguments.clear();
  for (auto [i, slot] : llvm::enumerate(program.gradientSlots)) {
    unsigned argument = base + gradientColumns[i];
    config.scalarArguments.push_back(argument);
    program.gradientArguments.push_back({argument, slot.tunable});
  }
  for (auto [c, field] : llvm::enumerate(program.gradientFields)) {
    unsigned argument = c < seeds ? base + constants + c : chargeArgument;
    config.fieldArguments.push_back(argument);
    program.gradientArguments.push_back({argument, field.tunable});
  }
  emitIRFragment(
      indent, {}, {}, input.names(), input.signatures(),
      [config = std::move(config), fields = input.fields, timed = input.timed,
       tail, constants, seeds, fieldResults](mlir::OpBuilder &builder,
                                            mlir::ValueRange values) -> TermIRValues {
        buildTunableGradientOutput(
            builder, state(values, fields, timed), values[tail], values[tail + 1],
            values[tail + 2], fieldResults ? values.back() : mlir::Value{},
            values.slice(tail + 3, constants), values.slice(tail + 3 + constants, seeds),
            config);
        return {};
      });
}

void Builder::emitWriteSolvent(StringRef indent, StringRef sums) {
  emitIRFragment(
      indent, {}, {}, {sums}, {"vector<3xf64>"},
      [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        buildWriteSolvent(builder, values[0]);
        return {};
      });
}

void Builder::emitTerms(StringRef x) {
  if (!system.topology)
    return;
  // Expressions retain topology/tuple/pair order. Stored corrections follow
  // the evaluated terms, as in the initial energy row.
  const auto definitions = getTermDefinitions();
  unsigned storedTermCount = 0;
  for (const TermBuilder *builder : getTermBuilders())
    if (const auto *role = builder->getRoles().postEvaluation)
      storedTermCount += role->getStoredTermCount(*this);
  size_t count = definitions.size() + storedTermCount;
  std::string type = "memref<" + std::to_string(count) + "xf64>";
  std::vector<std::string> potentials;
  for (const auto &definition : definitions)
    potentials.push_back(definition.name);
  OutputConnections input(program, x, "%cell", "%p_", "%time0", control.usesTime);
  emitIRFragment(
      "  ", {"%terms"}, {type}, input.names(), input.signatures(),
      [potentials = std::move(potentials), count, fields = input.fields,
       timed = input.timed](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        return {buildTermEnergyBuffer(builder, state(values, fields, timed), potentials, count)};
      });
  int index = definitions.size();
  for (const TermBuilder *builder : getTermBuilders())
    if (const auto *role = builder->getRoles().postEvaluation)
      role->emitStoredTerms(*this, type, index);
  emitIRFragment(
      "  ", {}, {}, {"%terms"}, {type},
      [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        buildWriteTerms(builder, values[0]);
        return {};
      });
}
