// The batched bodies of neural stages, lowered to functions of buffers that
// the tiles of a loop call (D[ml4-batched-bodies], docs/ml4-ir.md,
// Section 8.3).
//
// A batched body is a function of `linalg` on tensors with a static number
// of rows, StableHLO legalized to `linalg` (#377, the ruling of 2026-10-17).
// MDIR lowers it with upstream passes: elementwise fusion, one-shot
// bufferization with the results as buffers that the caller passes, and
// `linalg` to parallel loops. On a device each parallel loop becomes one
// kernel with a thread for each of its iterations; the constants of the
// body (the weights) are read from the kernels' own globals, and the body's
// intermediate buffers are allocated on the device for the call, from the
// runtime's pool of blocks.

#include "mdir/Conversion/Passes.h"

#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace mdir;

namespace mdir {

#define GEN_PASS_DEF_LOWERSTAGEBODIES
#include "mdir/Conversion/Passes.h.inc"

namespace {

/// `name` with every character other than a letter or a digit an
/// underscore.
static std::string identifier(StringRef name) {
  std::string out;
  for (char c : name)
    out.push_back(llvm::isAlnum(c) ? c : '_');
  return out;
}

/// The index `value` of the type index.
static Value index(OpBuilder &builder, Location loc, int64_t value) {
  return arith::ConstantIndexOp::create(builder, loc, value);
}

/// Replaces the top-level parallel loop `loop`, of constant bounds, with a
/// kernel of a thread for each of its iterations, the iterations numbered
/// in the order of its dimensions, the last fastest.
static LogicalResult launchLoop(scf::ParallelOp loop, int64_t blockSize) {
  Location loc = loop.getLoc();
  if (!loop.getInitVals().empty())
    return loop.emitOpError()
           << "in a stage's body reduces over its dimensions, which a kernel "
              "of a thread for each iteration does not do";
  SmallVector<int64_t> lower, upper, step;
  for (auto [l, u, s] : llvm::zip(loop.getLowerBound(), loop.getUpperBound(),
                                  loop.getStep())) {
    std::optional<int64_t> lc = getConstantIntValue(l), uc = getConstantIntValue(u),
                           sc = getConstantIntValue(s);
    if (!lc || !uc || !sc || *sc <= 0)
      return loop.emitOpError()
             << "in a stage's body has bounds that are not constants";
    lower.push_back(*lc);
    upper.push_back(*uc);
    step.push_back(*sc);
  }
  int64_t total = 1;
  SmallVector<int64_t> extents;
  for (unsigned d = 0; d != lower.size(); ++d) {
    extents.push_back((upper[d] - lower[d] + step[d] - 1) / step[d]);
    total *= extents.back();
  }
  OpBuilder builder(loop);
  if (total == 0) {
    loop.erase();
    return success();
  }
  Value one = index(builder, loc, 1);
  Value grid = index(builder, loc, (total + blockSize - 1) / blockSize);
  Value block = index(builder, loc, blockSize);
  auto launch = gpu::LaunchOp::create(builder, loc, grid, one, one, block, one,
                                      one);
  OpBuilder kernel = OpBuilder::atBlockEnd(&launch.getBody().front());
  Value item = arith::AddIOp::create(
      kernel, loc,
      arith::MulIOp::create(kernel, loc, launch.getBlockIds().x,
                            index(kernel, loc, blockSize)),
      launch.getThreadIds().x);
  Value inside = arith::CmpIOp::create(kernel, loc, arith::CmpIPredicate::ult,
                                       item, index(kernel, loc, total));
  auto guard = scf::IfOp::create(kernel, loc, inside, /*withElse=*/false);
  OpBuilder then = guard.getThenBodyBuilder();
  IRMapping mapping;
  Value rest = item;
  for (unsigned d = lower.size(); d-- != 0;) {
    Value place = arith::RemUIOp::create(then, loc, rest,
                                         index(then, loc, extents[d]));
    rest = arith::DivUIOp::create(then, loc, rest, index(then, loc, extents[d]));
    Value iv = arith::AddIOp::create(
        then, loc, index(then, loc, lower[d]),
        arith::MulIOp::create(then, loc, place, index(then, loc, step[d])));
    mapping.map(loop.getInductionVars()[d], iv);
  }
  for (Operation &op : loop.getBody()->without_terminator())
    then.clone(op, mapping);
  gpu::TerminatorOp::create(kernel, loc);
  loop.erase();

  // A global (a constant of the body) is read where the kernel is, so that
  // the outlining takes it into the kernel's module.
  launch.getBody().walk([&](Operation *op) {
    for (OpOperand &operand : op->getOpOperands()) {
      auto global = operand.get().getDefiningOp<memref::GetGlobalOp>();
      if (!global || launch->isAncestor(global))
        continue;
      OpBuilder inner(&launch.getBody().front(),
                      launch.getBody().front().begin());
      operand.set(inner.clone(*global)->getResult(0));
    }
  });
  return success();
}

/// Puts the buffers of `function` on the device: an allocation becomes one of
/// the device, released before the function returns, and a copy one of the
/// device. Fails on an access of a buffer outside a kernel.
static LogicalResult moveToDevice(func::FuncOp function) {
  SmallVector<Operation *> allocations, copies;
  WalkResult walk = function.walk([&](Operation *op) {
    if (op->getParentOfType<gpu::LaunchOp>())
      return WalkResult::advance();
    if (isa<memref::AllocOp>(op))
      allocations.push_back(op);
    else if (isa<memref::CopyOp>(op))
      copies.push_back(op);
    else if (isa<memref::LoadOp, memref::StoreOp, scf::ForOp, scf::ParallelOp,
                 linalg::LinalgOp>(op))
      return op->emitOpError()
                 << "in a stage's body is left on the host, where it cannot "
                    "read the device's buffers",
             WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (walk.wasInterrupted())
    return failure();
  SmallVector<Value> device;
  for (Operation *op : allocations) {
    auto alloc = cast<memref::AllocOp>(op);
    OpBuilder builder(op);
    Value buffer = gpu::AllocOp::create(builder, op->getLoc(), alloc.getType(),
                                        /*asyncToken=*/Type(), ValueRange(),
                                        alloc.getDynamicSizes(),
                                        alloc.getSymbolOperands())
                       .getMemref();
    alloc.replaceAllUsesWith(buffer);
    op->erase();
    device.push_back(buffer);
  }
  for (Operation *op : copies) {
    auto copy = cast<memref::CopyOp>(op);
    OpBuilder builder(op);
    gpu::MemcpyOp::create(builder, op->getLoc(), TypeRange(), ValueRange(),
                          copy.getTarget(), copy.getSource());
    op->erase();
  }
  // Released at the return; the pool of the runtime gives the same blocks to
  // the next call (mgpuMemAlloc).
  function.walk([&](func::ReturnOp ret) {
    OpBuilder builder(ret);
    for (Value buffer : device)
      gpu::DeallocOp::create(builder, ret.getLoc(), TypeRange(), ValueRange(),
                             buffer);
  });
  function.walk([&](memref::DeallocOp op) { op.erase(); });
  return success();
}

/// On the host, the intermediate buffers are released at the return.
static void releaseOnHost(func::FuncOp function) {
  SmallVector<Value> buffers;
  function.walk([&](memref::AllocOp alloc) { buffers.push_back(alloc); });
  function.walk([&](func::ReturnOp ret) {
    OpBuilder builder(ret);
    for (Value buffer : buffers)
      memref::DeallocOp::create(builder, ret.getLoc(), buffer);
  });
}

class LowerStageBodies : public impl::LowerStageBodiesBase<LowerStageBodies> {
public:
  using impl::LowerStageBodiesBase<LowerStageBodies>::LowerStageBodiesBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    impl::LowerStageBodiesBase<LowerStageBodies>::getDependentDialects(
        registry);
    bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
        registry);
    linalg::registerBufferizableOpInterfaceExternalModels(registry);
    tensor::registerBufferizableOpInterfaceExternalModels(registry);
    arith::registerBufferizableOpInterfaceExternalModels(registry);
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    llvm::SetVector<StringAttr> names;
    module.walk([&](md::StageOp stage) {
      if (!stage.getRows())
        return;
      names.insert(stage.getForwardAttr().getAttr());
      if (stage.getVjpAttr())
        names.insert(stage.getVjpAttr().getAttr());
    });
    if (names.empty())
      return;

    // The bodies are lowered in a module of their own, so that the passes
    // see nothing else.
    OwningOpRef<ModuleOp> bodies = ModuleOp::create(module.getLoc());
    for (StringAttr name : names) {
      auto function = module.lookupSymbol<func::FuncOp>(name);
      if (!function || function.isExternal()) {
        module.emitError() << "'" << name.getValue()
                           << "', a batched body, is not a function in the "
                              "module";
        return signalPassFailure();
      }
      if (function->hasAttr("md.stage_body"))
        continue;
      // The rows stay known once the tensors are buffers.
      auto first = dyn_cast<RankedTensorType>(
          function.getFunctionType().getNumInputs()
              ? function.getFunctionType().getInput(0)
              : Type());
      if (!first) {
        function.emitOpError() << "is a batched body without tensors";
        return signalPassFailure();
      }
      function->setAttr("md.stage_rows",
                        IntegerAttr::get(IntegerType::get(context, 64),
                                         first.getDimSize(0)));
      // A name that the kernels made from it can carry in PTX: letters,
      // digits, and underscores.
      std::string fresh = "md_stage_" + identifier(name.getValue());
      if (failed(SymbolTable::replaceAllSymbolUses(
              function, StringAttr::get(context, fresh), module)))
        return signalPassFailure();
      function.setSymName(fresh);
      function->remove();
      bodies->push_back(function);
    }

    PassManager pm(context);
    pm.addPass(createLinalgElementwiseOpFusionPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    bufferization::OneShotBufferizePassOptions options;
    options.bufferizeFunctionBoundaries = true;
    // The caller passes buffers of the rows of a tile, of their own.
    options.functionBoundaryTypeConversion =
        bufferization::LayoutMapOption::IdentityLayoutMap;
    pm.addPass(bufferization::createOneShotBufferizePass(options));
    // The results are copied to buffers that the caller passes. (Hoisting
    // their allocations instead crashed in LLVM 23.1.2 on these bodies.)
    bufferization::BufferResultsToOutParamsPassOptions outParams;
    outParams.modifyPublicFunctions = true;
    pm.addPass(bufferization::createBufferResultsToOutParamsPass(outParams));
    pm.addPass(createCanonicalizerPass());
    if (failed(pm.run(*bodies)))
      return signalPassFailure();
    // A copy into a result's buffer becomes a loop as well, so that it runs
    // where the buffers are.
    bodies->walk([&](memref::CopyOp copy) {
      OpBuilder builder(copy);
      linalg::CopyOp::create(builder, copy.getLoc(), copy.getSource(),
                             copy.getTarget());
      copy.erase();
    });
    PassManager loops(context);
    loops.addNestedPass<func::FuncOp>(createConvertLinalgToParallelLoopsPass());
    loops.addPass(createCanonicalizerPass());
    loops.addPass(createCSEPass());
    if (failed(loops.run(*bodies)))
      return signalPassFailure();

    for (func::FuncOp function : bodies->getOps<func::FuncOp>()) {
      if (device) {
        SmallVector<scf::ParallelOp> loops;
        function.walk([&](scf::ParallelOp loop) {
          if (!loop->getParentOfType<scf::ParallelOp>())
            loops.push_back(loop);
        });
        for (scf::ParallelOp loop : loops)
          if (failed(launchLoop(loop, blockSize)))
            return signalPassFailure();
        if (failed(moveToDevice(function)))
          return signalPassFailure();
      } else {
        releaseOnHost(function);
      }
      function->setAttr("md.stage_body", UnitAttr::get(context));
    }

    // Back into the module, the globals of the bodies' constants under names
    // of their own.
    for (auto global :
         llvm::to_vector(bodies->getOps<memref::GlobalOp>())) {
      StringAttr old = global.getSymNameAttr();
      std::string fresh = "md_stage_" + identifier(old.getValue());
      if (failed(SymbolTable::replaceAllSymbolUses(
              old, StringAttr::get(context, fresh), bodies->getOperation())))
        return signalPassFailure();
      global.setSymName(fresh);
    }
    SymbolTable table(module);
    for (Operation &op : llvm::make_early_inc_range(*bodies->getBody())) {
      op.remove();
      table.insert(&op);
    }
  }
};

} // namespace
} // namespace mdir
