// Identify ordinary TOWERS loops after storage assignment exposes their
// carried banks and constraint checks.
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
namespace mdir {
LogicalResult lowerTowersChunks(ModuleOp, func::FuncOp, ArrayRef<scf::ForOp>);
}

namespace {
static bool launchesForce(Operation &op) {
  if (isa<gpu::LaunchOp, gpu::LaunchFuncOp>(op))
    return true;
  if (auto call = dyn_cast<func::CallOp>(op))
    return call.getCallee().starts_with("mdrt_gpu_pme_") ||
           call.getCallee().starts_with("mdrtCudaFFT");
  return isa<scf::IfOp>(op) &&
         op.walk([](gpu::LaunchOp) { return WalkResult::interrupt(); })
             .wasInterrupted();
}

static bool isOrdinaryLoop(scf::ForOp loop) {
  bool start = false, finish = false;
  bool towers = false;
  for (Operation &op : *loop.getBody()) {
    if (auto call = dyn_cast<func::CallOp>(op)) {
      if (call.getCallee() == "mdrtWriteEnergies" || call.getCallee() == "mdrtBussiFactor" ||
          call.getCallee() == "mdrtBussiNormal")
        return false;
      if (call.getCallee() == "mdrtFlagStart") {
        if (start)
          return false;
        start = true;
      }
      if (call.getCallee() == "mdrtFlagFinish")
        finish = true;
    }
    if (!start || finish)
      continue;
    if (auto launch = dyn_cast<gpu::LaunchFuncOp>(op)) {
      if (launch.getKernelName().getValue().starts_with("towers_force_virial"))
        return false;
      if (launch.getKernelName().getValue().starts_with("towers_finish"))
        towers = true;
    }
    if (launchesForce(op)) {
      if (op.getNumResults())
        return false;
    }
  }
  // Only a loop whose yielded storage is a permutation of its input storage.
  // Coupling and observation steps carry reductions and occur outside it.
  if (!finish || !towers || loop.getNumRegionIterArgs() == 0)
    return false;
  for (Value v : loop.getBody()->getTerminator()->getOperands())
    if (!llvm::is_contained(loop.getRegionIterArgs(), v))
      return false;
  return true;
}

struct TowersStepGraph : PassWrapper<TowersStepGraph, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TowersStepGraph)
  StringRef getArgument() const final { return "mdir-towers-step-graph"; }
  StringRef getDescription() const final {
    return "Replace ordinary TOWERS step loops with captured step graphs "
           "(D439), as the GPU lowering pipeline runs it";
  }
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    bool trial = mod->hasAttr("mdir.step_graph_trial");
    if (!trial && !mod->hasAttr("mdir.towers_dt"))
      return;
    auto fn = mod.lookupSymbol<func::FuncOp>("mdir_run");
    if (!fn)
      return;
    SmallVector<scf::ForOp> chunks;
    fn.walk([&](scf::ForOp loop) { if (isOrdinaryLoop(loop)) chunks.push_back(loop); });
    if (chunks.empty()) {
      fn.emitError("step_graph: replay requires an ordinary constrained TOWERS step between schedule boundaries");
      return signalPassFailure();
    }
    if (trial) {
      // Bracket exactly the same ordinary loop before either submission path
      // replaces it. Coupling, reporting and initial compilation are outside.
      OpBuilder b(mod.getContext());
      auto timer = [&](StringRef name, TypeRange args) {
        b.setInsertionPointToEnd(mod.getBody());
        auto f = func::FuncOp::create(b, fn.getLoc(), name, b.getFunctionType(args, {}));
        f.setPrivate();
        return f;
      };
      auto begin = timer("mdrtStepTrialBegin", {});
      auto end = timer("mdrtStepTrialEnd", b.getI64Type());
      for (auto loop : chunks) {
        b.setInsertionPoint(loop);
        Value count = arith::IndexCastOp::create(b, loop.getLoc(), b.getI64Type(), loop.getUpperBound());
        func::CallOp::create(b, loop.getLoc(), begin);
        b.setInsertionPointAfter(loop);
        func::CallOp::create(b, loop.getLoc(), end, ValueRange{count});
      }
    }
    if (mod->hasAttr("mdir.towers_dt") && failed(mdir::lowerTowersChunks(mod, fn, chunks))) signalPassFailure();
  }
};
}

namespace mdir {
std::unique_ptr<Pass> createTowersStepGraph() {
  return std::make_unique<TowersStepGraph>();
}
// For the tests of the graph builder's refusals (mdir-opt).
void registerTowersStepGraph() { PassRegistration<TowersStepGraph>(); }
}
