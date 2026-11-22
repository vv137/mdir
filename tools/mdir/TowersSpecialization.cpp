#include "TowersSpecialization.h"
#include "mdir/Compiler/Compile.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mdir/Conversion/TowersForce.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Driver/RigidUnits.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/RegionUtils.h"
#include <cmath>

using namespace mlir;
using namespace mdir;
using namespace mdir::driver;

namespace {
struct PairModule {
  MLIRContext context{compiler::getRegistry(),
                      MLIRContext::Threading::DISABLED};
  OwningOpRef<ModuleOp> module;
  Program program;
};
bool lower(ModuleOp module, StringRef pipeline) {
  PassManager pm(module.getContext(), ModuleOp::getOperationName(),
                 PassManager::Nesting::Implicit);
  return succeeded(parsePassPipeline(pipeline, pm)) &&
         succeeded(pm.run(module));
}
} // namespace

llvm::Expected<mdir::tool::TowersEmitter>
mdir::tool::prepareTowersSpecialization(const Control &control,
                                        const System &system, int reach,
                                        int capacity, int device) {
  // Construct the direct term through the normal force-field front end.
  // Its traversal is replaced after precision specialization.
  Control source = control;
  // Setup sees the input atom order. Its private rebuild packs the atoms.
  source.reorder = false;
  source.neighborStructure = NeighborStructure::Matrix;
  if (auto error = resolveRigidUnits(source, system))
    return std::move(error);
  auto program = buildProgram(source, system);
  if (!program)
    return program.takeError();
  compiler::useCudaToolkit();
  auto options = compiler::getGpuOptions(source, device);
  if (!options)
    return options.takeError();
  auto state = std::make_shared<PairModule>();
  auto &context = state->context;
  compiler::shareThreadPool(context);
  context.loadAllAvailableDialects();
  std::string diagnostic;
  ScopedDiagnosticHandler handler(
      &context, [&](Diagnostic &d) { diagnostic += d.str() + "\n"; });
  auto fail = [&](StringRef why) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "towers_shape: %s: %s", why.str().c_str(),
                                   diagnostic.c_str());
  };
  auto module = parseSourceString<ModuleOp>(program->module, &context);
  if (!module ||
      !lower(
          *module,
          compiler::getPairRegionPipeline(source, *program) +
              "md-exec-assign-storage{memory=device tables=f32},canonicalize"))
    return fail("could not specialize direct pair region");
  md_exec::PairForOp term;
  module->walk([&](md_exec::PairForOp op) {
    if (op.getReduce().empty() && op.getOuts().size() == 1 &&
        op.getExchange(0) == md::Exchange::Antisymmetric) {
      term = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (!term)
    return fail("no force-only direct pair region");
  double rc = control.cutoffDistance * units::length;
  kernels::TowersLayout layout{int(system.getNumParticles()),
                               int(std::floor(system.box[0] / rc)),
                               int(std::floor(system.box[1] / rc)),
                               int(std::floor(system.box[2] * reach / rc)),
                               reach,
                               capacity,
                               false,
                               true};
  // Give the compiled kernel its real table dimensions and bare-pointer
  // arguments. The regular GPU lowering also expands md.lookup to a load.
  SmallVector<md::LookupOp> lookups;
  term.walk([&](md::LookupOp op) { lookups.push_back(op); });
  OpBuilder builder(term);
  DenseMap<Value, Value> tables;
  for (auto lookup : lookups) {
    auto type = dyn_cast<MemRefType>(lookup.getTable().getType());
    if (!type || type.getRank() != 2 || !system.topology)
      return fail("unsupported direct parameter table");
    auto n = static_cast<int64_t>(system.topology->getNumTypes());
    auto fixed =
        MemRefType::get({n, n}, type.getElementType(),
                        MemRefLayoutAttrInterface(), type.getMemorySpace());
    auto &table = tables[lookup.getTable()];
    if (!table)
      table = memref::CastOp::create(builder, term.getLoc(), fixed,
                                     lookup.getTable());
    lookup.getTableMutable().assign(table);
  }
  kernels::lowerLookups(term);
  // Immutable radial tables are inputs shared by every lane.
  SmallVector<memref::GetGlobalOp> globals;
  term.walk([&](memref::GetGlobalOp op) { globals.push_back(op); });
  for (auto global : globals)
    global->moveBefore(term);
  int exclusions = system.topology ? system.topology->exclusions.size() : 0;
  md_exec::PairForOp correction;
  if (exclusions) {
    md_exec::TupleForOp excluded;
    for (auto tuple : term->getBlock()->getOps<md_exec::TupleForOp>()) {
      auto *inc = tuple.getIncidence().getDefiningOp();
      auto name =
          inc ? inc->getAttrOfType<FlatSymbolRefAttr>("md_exec.tuple_set")
              : FlatSymbolRefAttr();
      if (name && name.getValue() == "excluded" && tuple.getReduce().empty()) {
        if (excluded)
          return fail("multiple physical corrections for the excluded set");
        excluded = tuple;
      }
    }
    correction = cast<md_exec::PairForOp>(builder.clone(*term));
    Block &body = correction.getKernel().front();
    for (auto &op : llvm::make_early_inc_range(llvm::reverse(body)))
      op.erase();
    builder.setInsertionPointToEnd(&body);
    Value force;
    if (excluded) {
      auto coords = excluded.getCoordinates();
      if (coords.size() != 1 ||
          coords.front().members != ArrayRef<int64_t>({0, 1}) ||
          excluded.getArity() != 2 || !excluded.getParameters().empty() ||
          excluded.getOuts().size() != 1)
        return fail("unsupported physical exclusion correction");
      Block &old = excluded.getKernel().front();
      IRMapping map;
      map.map(old.getArgument(0), body.getArgument(1));
      for (auto [i, field] : llvm::enumerate(excluded.getIns())) {
        auto it = llvm::find(term.getIns(), field);
        if (it == term.getIns().end())
          return fail("exclusion field is absent from the direct pair");
        auto j = it - term.getIns().begin();
        for (int a : {0, 1})
          map.map(old.getArgument(1 + 2 * i + a),
                  body.getArgument(2 + 2 * j + a));
      }
      for (auto &op : old.without_terminator())
        builder.clone(op, map);
      force = map.lookup(old.getTerminator()->getOperand(0));
    } else {
      auto vector = body.getArgument(1).getType();
      force = arith::ConstantOp::create(builder, term.getLoc(), vector,
                                        builder.getZeroAttr(vector));
    }
    md_exec::YieldOp::create(builder, term.getLoc(), ValueRange{force});
  }
  state->module = std::move(module);
  state->program = std::move(*program);
  return TowersEmitter([state, term, correction, layout, exclusions,
                        options =
                            std::move(*options)](const TowersShape &shape,
                                                 TowersBuffers *trial) mutable
                           -> llvm::Expected<std::string> {
    std::string diagnostic;
    ScopedDiagnosticHandler handler(
        &state->context, [&](Diagnostic &d) { diagnostic += d.str() + "\n"; });
    if (trial) {
      llvm::SetVector<Value> captures;
      getUsedValuesDefinedAbove(term->getRegion(0), term->getRegion(0),
                                captures);
      if (correction)
        getUsedValuesDefinedAbove(correction->getRegion(0),
                                  correction->getRegion(0), captures);
      auto buffers = bindTowersTrialInputs(state->program, term.getIns(),
                                           captures.getArrayRef());
      if (!buffers)
        return buffers.takeError();
      *trial = std::move(*buffers);
    }
    OwningOpRef<ModuleOp> image(ModuleOp::create(term.getLoc()));
    // The 4n force specialization uses a two-block compiler launch bound.
    // A higher control pin asks whether that compiled entry can reside there.
    kernels::TowersForceShape launch{
        int(shape.panel),
        int(shape.stripe),
        int(shape.threads),
        int(shape.records),
        int(std::min<int64_t>(2, shape.blocksPerSM)),
        false,
        true,
        exclusions,
        int(shape.split)};
    if (failed(kernels::emitTowersBuild(*image, layout)) ||
        failed(kernels::emitTowersForce(*image, layout, term, launch, false,
                                        correction)) ||
        !lower(*image, "canonicalize,cse,mdir-gpu-lower-to-nvvm{" + options +
                           " kernel-bare-ptr-calling-convention=true}"))
      return llvm::createStringError(
          "towers_shape: could not compile direct pair specialization: %s",
          diagnostic.c_str());
    auto binary = image->lookupSymbol<gpu::BinaryOp>("towers");
    if (!binary || binary.getObjects().empty())
      return llvm::createStringError(
          "towers_shape: missing compiled tower entry");
    return cast<gpu::ObjectAttr>(binary.getObjects()[0]).getObject().str();
  });
}
