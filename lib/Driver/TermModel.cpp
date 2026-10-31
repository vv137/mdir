// Staged and hosted model terms, with private preparation per build
// (D359, D361, D362, #683).
#include "BuilderImpl.h"
#include "StagedModelIR.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mdir/Dialect/MDRT/MDRTOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"

using namespace mdir::driver;
using namespace mdir::driver::builder;

struct HostedModelIRConfig {
  bool edges = false, energy = false, virial = false;
  int stateBits = 64, forceBits = 64;
  size_t count = 0, energySlot = 0, virialSlot = 0;
  std::vector<double> cutoffs;
};

static TermIRValues buildHostedModelEvaluation(mlir::OpBuilder &builder, mlir::ValueRange values, const HostedModelIRConfig &config) {
  using namespace mlir;
  auto loc = builder.getUnknownLoc();
  auto *ctx = builder.getContext();
  auto f64 = builder.getF64Type();
  auto vec6 = VectorType::get({6}, f64), vec9 = VectorType::get({9}, f64);
  Value cell =
      mdir::md_exec::CellNumbersOp::create(builder, loc, vec6, values[0]);
  llvm::SmallVector<Value> cells;
  for (int c = 0; c != 6; ++c)
    cells.push_back(vector::ExtractOp::create(builder, loc, cell, c));
  Value bits =
      arith::ConstantIntOp::create(builder, loc, config.stateBits, 64);
  Value forceBits =
      arith::ConstantIntOp::create(builder, loc, config.forceBits, 64);
  Value forces = values[3], energy, virial;
  if (config.energy)
    energy = values[config.energySlot];
  if (config.virial)
    virial = values[config.virialSlot];
  auto atoms =
      cast<mdir::md::FieldType>(values[1].getType()).getParticleSet();
  auto pairsType = mdir::md::RelationType::get(
      ctx, atoms, 2, mdir::md::Orientation::Unordered,
      FlatSymbolRefAttr{});
  auto edgesType = mdir::md::EdgesType::get(ctx, atoms);
  bool numbers = config.energy || config.virial;
  std::string callee = std::string("mdrtModelEvaluate") +
                       (config.edges ? "Edges" : "") +
                       (numbers ? "Energy" : "");
  for (size_t index = 0; index != config.count; ++index) {
    Value edges;
    if (config.edges) {
      auto cutoff = builder.getF64FloatAttr(config.cutoffs[index]);
      Value neighborhood = mdir::md::NeighborhoodOp::create(
          builder, loc, pairsType, values[1], values[0], Value{}, cutoff);
      edges =
          mdir::md::EdgesOp::create(builder, loc, edgesType, neighborhood,
                                    values[1], values[0], cutoff);
    }
    Value model = arith::ConstantIntOp::create(builder, loc, index, 64);
    llvm::SmallVector<Value> inputs = {model, values[1], values[2]};
    if (config.edges)
      inputs.push_back(edges);
    llvm::append_range(inputs, cells);
    inputs.push_back(bits);
    inputs.push_back(forceBits);
    llvm::SmallVector<Type> results = {forces.getType()};
    if (numbers) {
      results.push_back(f64);
      results.push_back(vec9);
    }
    auto call = mdir::mdrt::HostCallOp::create(
        builder, loc, results, FlatSymbolRefAttr::get(ctx, callee),
        inputs, ValueRange{forces}, builder.getUnitAttr());
    forces = call.getResults()[0];
    if (config.energy)
      energy = arith::AddFOp::create(builder, loc, energy,
                                     call.getResults()[1]);
    if (config.virial)
      virial = arith::AddFOp::create(builder, loc, virial,
                                     call.getResults()[2]);
  }
  TermIRValues results = {forces};
  if (config.energy)
    results.push_back(energy);
  if (config.virial)
    results.push_back(virial);
  return results;
}

namespace mdir::driver::builder {
class ModelTermBuilder final : public TermPreparation,
                               public TermHostEvaluation,
                               public TermBuilder {
  struct State {
    bool edges = false;
    std::vector<double> cutoffs;
    std::vector<std::shared_ptr<StagedModelIR>> potentials;
  };
  std::unique_ptr<State> state;

public:
  ModelTermBuilder() : state(std::make_unique<State>()) {
    roles.preparation = this;
    roles.hostEvaluation = this;
  }
  int parameterRank() const override { return -3; }
  int topologyRank() const override { return 5; }
  llvm::Error validate(const TermInputs &context) const override;
  llvm::Error prepare(const TermInputs &context,
                      mlir::MLIRContext &irContext,
                      TermIRWriter emitIR) const override;
  bool applies(const TermInputs &context) const override {
    return !context.getControl().models.empty();
  }
  TermNativeResults build(const TermInputs &context, mlir::OpBuilder &builder,
                          const TermNativeEvaluation &point) const override {
    HostedModelIRConfig config;
    config.edges = state->edges;
    config.energy = bool(point.energy);
    config.virial = bool(point.virial);
    config.stateBits = context.getStateElement() == Element::F64 ? 64 : 32;
    config.forceBits = context.getForceElement() == Element::F64 ? 64 : 32;
    config.count = context.getControl().models.size();
    config.cutoffs = state->cutoffs;
    TermIRValues values{point.cell, point.positions, point.field("model_z"), point.forces};
    if (config.energy) {
      config.energySlot = values.size();
      values.push_back(point.energy);
    }
    if (config.virial) {
      config.virialSlot = values.size();
      values.push_back(point.virial);
    }
    auto result = buildHostedModelEvaluation(builder, values, config);
    return {result[0], config.energy ? result[1] : mlir::Value{},
            config.virial ? result[config.energy ? 2 : 1] : mlir::Value{}};
  }
  std::string emit(const TermInputs &context, TermIRWriter emitIR,
                   const TermEvaluation &point) const override;
  void collectParameters(TermParameters &context) const override {
    const Control &control = context.getControl();
    const Topology &topology = context.getTopology();
    size_t count = topology.getNumParticles();
    // The atomic numbers that a learned potential takes
    // (D270), in the order of the particles that the program
    // keeps, as the other fields; a model in stages takes them too
    // (D294).
    if (!control.models.empty() || !control.stagedModels.empty()) {
      Program::Field numbers;
      numbers.name = "model_z";
      numbers.isInteger = true;
      for (size_t i = 0; i != count; ++i)
        numbers.values.push_back(
            i < topology.atomicNumbers.size() ? topology.atomicNumbers[i] : 0);
      context.getFields().push_back(std::move(numbers));
    }
  }
  llvm::Error buildTopologyPotential(const TermInputs &, mlir::OpBuilder &builder,
      const TermNativeTopology &point, const PotentialSelection &selection,
      llvm::function_ref<void(mlir::Value)> add) const override {
    if (selection.terms != TermInputs::AllTerms) return llvm::Error::success();
    for (const auto &model : state->potentials) {
      auto energy = model->cloneBody(builder,
          mlir::ValueRange{point.physical.positions, point.physical.cell,
                          point.physical.field("model_z")});
      if (!energy) return energy.takeError();
      add(*energy);
    }
    return llvm::Error::success();
  }

private:
  void emitDeclarations(const TermInputs &, TermIRWriter) const override;
  void buildDeclarations(const TermInputs &, mlir::OpBuilder &) const override;
};
} // namespace mdir::driver::builder

std::unique_ptr<TermBuilder> mdir::driver::builder::createModelTermBuilder() {
  return std::make_unique<ModelTermBuilder>();
}

llvm::Error ModelTermBuilder::validate(const TermInputs &context) const {
  const Control &control = context.getControl();
  const System &system = context.getSystem();
  // A learned potential takes every particle with its element, and ML1 has
  // no constraints and no virtual sites beside it
  // (docs/metatomic-ml1.md, Section 6.9).
  if (!control.stagedModels.empty()) {
    if (!system.topology)
      return makeError("a model in stages of [[energy.model]] needs a "
                       "topology, which gives the elements of the particles");
    if (context.hasSites() || !system.topology->virtualSites.empty())
      return makeError("a model in stages takes no virtual sites yet");
  }
  if (!control.models.empty()) {
    if (!system.topology)
      return makeError("a term of [[energy.model]] needs a topology, which "
                       "gives the elements of the particles");
    if (context.hasConstraints())
      return makeError("a term of [[energy.model]] takes no constraints "
                       "yet: give a flexible model (no SETTLE or SHAKE)");
    if (context.hasSites() || !system.topology->virtualSites.empty())
      return makeError("a term of [[energy.model]] takes no virtual sites "
                       "yet");
    if (control.precision == Precision::Single)
      return makeError("a term of [[energy.model]] takes the mixed or the "
                       "double precision");
    // A frame evaluator is chosen when a Python simulation is made, after
    // the checks of the control: its forces and virial come from
    // evaluations of their own, which do not take the model.
    if (control.frames)
      return makeError("a term of [[energy.model]] is not evaluated by the "
                       "frame evaluator yet");
    if (control.tunables || control.tunableGradient)
      return makeError("a term of [[energy.model]] has no derivative in a "
                       "tunable parameter: tunables are refused");
    // The pair list is a value of the program, at the model's cutoff
    // (D277).
    llvm::Expected<int> mode = getPairsMode();
    if (!mode)
      return mode.takeError();
    state->edges = *mode != MDIR_MLIP_PAIRS_SEARCH;
    if (state->edges)
      for (const Control::ModelTerm &term : control.models) {
        llvm::Expected<double> cutoff = getModelCutoff(term);
        if (!cutoff)
          return cutoff.takeError();
        state->cutoffs.push_back(*cutoff);
      }
  }
  return llvm::Error::success();
}

llvm::Error ModelTermBuilder::prepare(const TermInputs &context,
                                      mlir::MLIRContext &irContext,
                                      TermIRWriter emitIR) const {
  state->potentials.clear();
  for (const Control::ModelTerm &term : context.getControl().stagedModels) {
    auto buffer = llvm::MemoryBuffer::getFile(term.path);
    if (!buffer)
      return makeError("cannot read the model in stages '" + term.name +
                       "' from '" + term.path +
                       "': " + buffer.getError().message());
    auto model = StagedModelIR::parse(irContext, (*buffer)->getBuffer(),
                                      term.path, term.name);
    if (!model)
      return model.takeError();
    emitIR("", {}, {}, {}, {},
           [input = *model](mlir::OpBuilder &builder, mlir::ValueRange)
               -> llvm::Expected<TermIRValues> {
             if (auto error = input->cloneDeclarations(builder))
               return std::move(error);
             return TermIRValues{};
           });
    state->potentials.push_back(std::move(*model));
  }
  return llvm::Error::success();
}

std::string ModelTermBuilder::emit(const TermInputs &context,
                                   TermIRWriter emitIR,
                                   const TermEvaluation &point) const {
  const Control &control = context.getControl();
  HostedModelIRConfig config;
  config.edges = state->edges;
  config.energy = !point.u.empty();
  config.virial = !point.w.empty();
  config.stateBits = context.getStateElement() == Element::F64 ? 64 : 32;
  config.forceBits = context.getForceElement() == Element::F64 ? 64 : 32;
  config.count = control.models.size();
  config.cutoffs = state->cutoffs;
  std::vector<std::string> names = {point.cell.str(), point.x.str(),
                                    (point.prefix + "model_z").str(),
                                    point.f.str()};
  std::vector<std::string> types = {"!md.cell", "!vec", "!ids", "!vec"};
  auto operand = [&](std::string name, std::string type) {
    size_t slot = names.size();
    names.push_back(std::move(name));
    types.push_back(std::move(type));
    return slot;
  };
  if (config.energy)
    config.energySlot = operand(point.u.str(), "f64");
  if (config.virial)
    config.virialSlot = operand(point.w.str(), "vector<9xf64>");
  std::string virial = point.w.str();
  if (config.virial && config.count)
    virial =
        "%mws_m" + point.tag.str() + "_" + std::to_string(config.count - 1);
  std::vector<std::string> results = {point.fResult.str()},
                           resultTypes = {"!vec"};
  if (config.energy) {
    results.push_back(point.uResult.str());
    resultTypes.push_back("f64");
  }
  if (config.virial) {
    results.push_back(virial);
    resultTypes.push_back("vector<9xf64>");
  }
  llvm::SmallVector<StringRef> in, inTypes, out, outTypes;
  for (const auto &s : names)
    in.push_back(s);
  for (const auto &s : types)
    inTypes.push_back(s);
  for (const auto &s : results)
    out.push_back(s);
  for (const auto &s : resultTypes)
    outTypes.push_back(s);
  emitIR(
      point.indent, out, outTypes, in, inTypes,
      [config](mlir::OpBuilder &builder,
               mlir::ValueRange values) -> TermIRValues {
        return buildHostedModelEvaluation(builder, values, config);
      });
  return virial;
}

namespace {
void buildModelDeclarations(mlir::OpBuilder &builder, bool edges, bool state64, bool force64) {
using namespace mlir;
auto loc = builder.getUnknownLoc();
Type state =
    state64 ? Type(builder.getF64Type()) : Type(builder.getF32Type());
Type force =
    force64 ? Type(builder.getF64Type()) : Type(builder.getF32Type());
auto i32 = builder.getI32Type(), i64 = builder.getI64Type();
auto dynamic = ShapedType::kDynamic;
llvm::SmallVector<Type> inputs = {i64,
                                  MemRefType::get({dynamic, 3}, state),
                                  MemRefType::get({dynamic}, i32)};
if (edges) {
  inputs.push_back(MemRefType::get({dynamic, 5}, i32));
  inputs.push_back(MemRefType::get({dynamic}, i32));
  inputs.push_back(MemRefType::get({dynamic}, i32));
  inputs.push_back(i64);
}
inputs.append(6, builder.getF64Type());
inputs.append(2, i64);
inputs.push_back(MemRefType::get({dynamic, 3}, force));
std::string name =
    std::string("mdrtModelEvaluate") + (edges ? "Edges" : "");
for (bool energy : {false, true}) {
  llvm::SmallVector<Type> arguments = inputs;
  if (energy) {
    arguments.push_back(MemRefType::get({1}, builder.getF64Type()));
    arguments.push_back(MemRefType::get({9}, builder.getF64Type()));
  }
  auto type = builder.getFunctionType(arguments, {});
  auto function = func::FuncOp::create(
      builder, loc, name + (energy ? "Energy" : ""), type);
  function.setPrivate();
  function->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  for (unsigned index : {1u, 2u})
    function.setArgAttr(index, "mdrt.read_only", builder.getUnitAttr());
  if (edges)
    for (unsigned index : {3u, 4u, 5u})
      function.setArgAttr(index, "mdrt.read_only",
                          builder.getUnitAttr());
}
}
} // namespace

void ModelTermBuilder::buildDeclarations(const TermInputs &context, mlir::OpBuilder &builder) const {
  if (context.getControl().models.empty()) return;
  buildModelDeclarations(builder, state->edges, context.getStateElement() == Element::F64,
      context.getForceElement() == Element::F64);
}

void ModelTermBuilder::emitDeclarations(const TermInputs &context,
                                        TermIRWriter emitIR) const {
  if (context.getControl().models.empty())
    return;
  bool edges = state->edges;
  bool state64 = context.getStateElement() == Element::F64;
  bool force64 = context.getForceElement() == Element::F64;
  emitIR(
      "", {}, {}, {}, {},
      [edges, state64, force64](mlir::OpBuilder &builder,
                                mlir::ValueRange) -> TermIRValues {
        buildModelDeclarations(builder, edges, state64, force64);
        return {};
      });
}
