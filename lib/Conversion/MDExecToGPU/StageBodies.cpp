// The batched bodies of neural stages, lowered to functions of buffers that
// the tiles of a loop call (D293, docs/ml4-ir.md,
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

/// Whether `op` is elementwise: a generic of parallel loops only, with one
/// result whose indexing map is a permutation, on tensors.
static bool isElementwise(linalg::GenericOp op) {
  return op.hasPureTensorSemantics() && op.getNumResults() == 1 &&
         op.getNumReductionLoops() == 0 &&
         op.getMatchingIndexingMap(op.getDpsInitOperand(0)).isPermutation();
}

/// Whether the payload of `op` is cheap to evaluate again: a few ops of
/// `arith`, none of `math` (the exponentials and the trigonometric functions
/// of the radial basis and of the cutoff).
static bool isCheap(linalg::GenericOp op) {
  unsigned count = 0;
  for (Operation &inner : op.getBody()->without_terminator()) {
    if (inner.getDialect() && inner.getDialect()->getNamespace() == "math")
      return false;
    ++count;
  }
  return count <= 8;
}

/// Gives each consumer that elementwise fusion would take a copy of its
/// elementwise producer when the producer has several uses: the fusion
/// takes a producer of one use only, and a producer of several stays its own
/// kernel, its result a round trip through global memory. A copy is made
/// where it does not multiply the producer's work: into a consumer of the
/// same loops, or of more loops if the producer is cheap. The original stays
/// for the uses that are not fused (a contraction, a slice, the return).
/// The idea, a producer that is cheap to recompute cloned into each region
/// that uses it rather than stored, is IREE's
/// (iree-org/iree v3.12.0, `compiler/src/iree/compiler/DispatchCreation/
/// CloneProducersIntoDispatchRegions.cpp`). Returns whether anything was
/// copied.
static bool copyProducersOfSeveralUses(func::FuncOp function) {
  // A copy's payload holds at most this many ops, so that repeated rounds of
  // copying and fusing do not grow the bodies without bound.
  constexpr unsigned kLargest = 64;
  SmallVector<linalg::GenericOp> producers;
  function.walk([&](linalg::GenericOp op) {
    if (isElementwise(op) && !op->getResult(0).hasOneUse() &&
        !op->getResult(0).use_empty() &&
        op.getBody()->getOperations().size() <= kLargest)
      producers.push_back(op);
  });
  bool changed = false;
  for (linalg::GenericOp producer : producers) {
    SmallVector<OpOperand *> uses, fused;
    for (OpOperand &use : producer->getResult(0).getUses())
      uses.push_back(&use);
    for (OpOperand *use : uses) {
      auto consumer = dyn_cast<linalg::GenericOp>(use->getOwner());
      if (!consumer || !linalg::areElementwiseOpsFusable(use))
        continue;
      if (consumer.getNumLoops() != producer.getNumLoops() &&
          !isCheap(producer))
        continue;
      fused.push_back(use);
    }
    if (fused.empty())
      continue;
    // The original serves the last of them if every use is fused.
    if (fused.size() == uses.size())
      fused.pop_back();
    for (OpOperand *use : fused) {
      OpBuilder builder(producer);
      use->set(builder.clone(*producer)->getResult(0));
      changed = true;
    }
  }
  return changed;
}

/// Starts each reduction whose initial value is a `linalg.fill` of a constant
/// from that constant, read at the first iteration of its reduced loops,
/// instead of from a buffer that a kernel of its own fills: the init becomes
/// `tensor.empty`, whose contents the payload no longer reads. This is the
/// meaning of the generic unchanged: at the first reduced iteration the
/// accumulator is the fill's value, and after it the partial sum.
static void startReductionsFromConstants(func::FuncOp function) {
  SmallVector<linalg::GenericOp> reductions;
  function.walk([&](linalg::GenericOp op) {
    if (op.hasPureTensorSemantics() && op.getNumReductionLoops() != 0)
      reductions.push_back(op);
  });
  for (linalg::GenericOp op : reductions) {
    SmallVector<unsigned> reduced;
    for (auto [d, type] : llvm::enumerate(op.getIteratorTypesArray()))
      if (type == utils::IteratorType::reduction)
        reduced.push_back(d);
    for (OpOperand &init : op.getDpsInitsMutable()) {
      auto fill = init.get().getDefiningOp<linalg::FillOp>();
      if (!fill)
        continue;
      Value constant = fill.getInputs()[0];
      if (!matchPattern(constant, m_Constant()))
        continue;
      Block *body = op.getBody();
      BlockArgument accumulator = op.getMatchingBlockArgument(&init);
      OpBuilder builder = OpBuilder::atBlockBegin(body);
      Location loc = op.getLoc();
      Value first;
      for (unsigned d : reduced) {
        Value index = linalg::IndexOp::create(builder, loc, d);
        Value zero = arith::CmpIOp::create(
            builder, loc, arith::CmpIPredicate::eq, index,
            arith::ConstantIndexOp::create(builder, loc, 0));
        first = first ? arith::AndIOp::create(builder, loc, first, zero)
                      : zero;
      }
      // The constant, defined outside the payload, is captured by it.
      Value start = arith::SelectOp::create(builder, loc, first, constant,
                                            accumulator);
      accumulator.replaceAllUsesExcept(start, start.getDefiningOp());
      OpBuilder outside(op);
      auto type = cast<RankedTensorType>(init.get().getType());
      init.set(tensor::EmptyOp::create(outside, loc, type.getShape(),
                                       type.getElementType()));
    }
  }
}

/// Rewrites each `tensor.pad` of a constant with static sizes as an
/// elementwise generic that reads its source where the index falls inside
/// it and gives the constant elsewhere, so that elementwise fusion takes it
/// into its consumer: the legalization makes the adjoint of a slice a pad
/// (a 512-wide adjoint of the three slices of a 384-wide input, added up),
/// which bufferizes to a fill and a copy, two kernels each.
static LogicalResult padsAsGathers(func::FuncOp function) {
  SmallVector<tensor::PadOp> pads;
  function.walk([&](tensor::PadOp pad) { pads.push_back(pad); });
  for (tensor::PadOp pad : pads) {
    Value constant = pad.getConstantPaddingValue();
    auto source = cast<RankedTensorType>(pad.getSource().getType());
    auto result = cast<RankedTensorType>(pad.getResult().getType());
    if (!constant || !source.hasStaticShape() || !result.hasStaticShape())
      continue;
    SmallVector<int64_t> low(pad.getStaticLow());
    if (llvm::any_of(low, ShapedType::isDynamic) ||
        llvm::any_of(pad.getStaticHigh(), ShapedType::isDynamic))
      continue;
    OpBuilder builder(pad);
    Location loc = pad.getLoc();
    unsigned rank = result.getRank();
    Value empty = tensor::EmptyOp::create(builder, loc, result.getShape(),
                                          result.getElementType());
    SmallVector<AffineMap> maps = {builder.getMultiDimIdentityMap(rank)};
    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);
    // A pad of a slice (an adjoint of a slice) reads the slice's source.
    Value input = pad.getSource();
    SmallVector<int64_t> offsets(rank, 0);
    if (auto slice = input.getDefiningOp<tensor::ExtractSliceOp>())
      if (slice.getSourceType().getRank() == rank &&
          llvm::all_of(slice.getStaticStrides(),
                       [](int64_t s) { return s == 1; }) &&
          !llvm::any_of(slice.getStaticOffsets(), ShapedType::isDynamic)) {
        offsets.assign(slice.getStaticOffsets().begin(),
                       slice.getStaticOffsets().end());
        input = slice.getSource();
      }
    auto gather = linalg::GenericOp::create(
        builder, loc, result, ValueRange(), ValueRange{empty}, maps, iterators,
        [&](OpBuilder &b, Location l, ValueRange) {
          Value inside, zero = arith::ConstantIndexOp::create(b, l, 0);
          SmallVector<Value> at;
          for (unsigned d = 0; d != rank; ++d) {
            Value index = linalg::IndexOp::create(b, l, d);
            Value shifted = arith::SubIOp::create(
                b, l, index, arith::ConstantIndexOp::create(b, l, low[d]));
            int64_t size = source.getDimSize(d);
            Value in = arith::AndIOp::create(
                b, l,
                arith::CmpIOp::create(b, l, arith::CmpIPredicate::sge, shifted,
                                      zero),
                arith::CmpIOp::create(
                    b, l, arith::CmpIPredicate::slt, shifted,
                    arith::ConstantIndexOp::create(b, l, size)));
            inside = inside ? arith::AndIOp::create(b, l, inside, in) : in;
            // Clamped, so that the read stays inside the source.
            Value clamped = arith::MinSIOp::create(
                b, l, arith::MaxSIOp::create(b, l, shifted, zero),
                arith::ConstantIndexOp::create(b, l, size - 1));
            at.push_back(offsets[d] ? arith::AddIOp::create(
                                          b, l, clamped,
                                          arith::ConstantIndexOp::create(
                                              b, l, offsets[d]))
                                    : clamped);
          }
          Value value = tensor::ExtractOp::create(b, l, input, at);
          Value chosen = arith::SelectOp::create(b, l, inside, value, constant);
          linalg::YieldOp::create(b, l, chosen);
        });
    pad.replaceAllUsesWith(gather.getResult(0));
    pad.erase();
  }
  return success();
}

/// Rewrites each `linalg.fill` that is not the initial value of a named op
/// (a product) as a generic, which elementwise fusion takes into its
/// consumers like any other elementwise producer.
static void generalizeFills(func::FuncOp function) {
  SmallVector<linalg::FillOp> fills;
  function.walk([&](linalg::FillOp fill) {
    bool initOfNamed = llvm::any_of(fill->getUses(), [](OpOperand &use) {
      auto user = dyn_cast<linalg::LinalgOp>(use.getOwner());
      return user && !isa<linalg::GenericOp>(user) && user.isDpsInit(&use);
    });
    if (!initOfNamed && fill.hasPureTensorSemantics())
      fills.push_back(fill);
  });
  IRRewriter rewriter(function.getContext());
  for (linalg::FillOp fill : fills) {
    rewriter.setInsertionPoint(fill);
    (void)linalg::generalizeNamedOp(rewriter, fill);
  }
}

/// Writes each result in the buffer that the caller passes for it, where the
/// bufferization allocated a buffer of the same type and copied it there at
/// the end (`buffer-results-to-out-params`): the allocation is replaced by
/// the caller's buffer and the copy, a kernel, goes. A buffer that is copied
/// to two results, or that is not an allocation of the function, is left.
/// (The option of that pass that does this, hoisting static allocations,
/// crashed in LLVM 23.1.2 on these bodies.)
static void writeResultsInPlace(func::FuncOp function) {
  llvm::DenseMap<Value, unsigned> copiesOf;
  SmallVector<memref::CopyOp> copies;
  function.walk([&](memref::CopyOp copy) {
    copies.push_back(copy);
    ++copiesOf[copy.getSource()];
  });
  for (memref::CopyOp copy : copies) {
    // The caller's buffer of a result: an argument that is only written,
    // by this copy (the operands are not written in place).
    Block &entry = function.getBody().front();
    auto target = dyn_cast<BlockArgument>(copy.getTarget());
    if (!target || target.getOwner() != &entry || !target.hasOneUse() ||
        copy->getBlock() != &entry)
      continue;
    auto alloc = copy.getSource().getDefiningOp<memref::AllocOp>();
    if (!alloc || alloc.getType() != target.getType() ||
        copiesOf.lookup(alloc) != 1 || alloc->getBlock() != &entry)
      continue;
    // Nothing reads, writes, or releases the allocation after the copy.
    if (llvm::any_of(alloc->getUsers(), [&](Operation *user) {
          return user != copy &&
                 (isa<memref::DeallocOp>(user) ||
                  copy->isBeforeInBlock(entry.findAncestorOpInBlock(*user)));
        }))
      continue;
    alloc.replaceAllUsesWith(Value(target));
    copy.erase();
    alloc.erase();
  }
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

    // Elementwise fusion. The unit dimensions that the legalization leaves
    // (`4096x1`, `1x100`, reductions over one element) are folded first, so
    // that the reshapes around them do not stop the fusion; the pads become
    // gathers, the reductions start from their constants, and the other
    // fills become generics; then producers of several uses are copied into
    // their consumers and fused, until nothing changes.
    if (fuse) {
      PassManager unit(context);
      unit.addPass(createLinalgFoldUnitExtentDimsPass());
      unit.addPass(createCanonicalizerPass());
      if (failed(unit.run(*bodies)))
        return signalPassFailure();
      for (func::FuncOp function : bodies->getOps<func::FuncOp>()) {
        if (failed(padsAsGathers(function)))
          return signalPassFailure();
        startReductionsFromConstants(function);
        generalizeFills(function);
      }
    }
    for (unsigned round = 0; round != (fuse ? 8 : 1); ++round) {
      PassManager pm(context);
      pm.addPass(createLinalgElementwiseOpFusionPass());
      pm.addPass(createCanonicalizerPass());
      pm.addPass(createCSEPass());
      if (failed(pm.run(*bodies)))
        return signalPassFailure();
      if (!fuse)
        break;
      bool copied = false;
      for (func::FuncOp function : bodies->getOps<func::FuncOp>())
        copied |= copyProducersOfSeveralUses(function);
      if (!copied)
        break;
    }

    PassManager pm(context);
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
    if (fuse)
      for (func::FuncOp function : bodies->getOps<func::FuncOp>())
        writeResultsInPlace(function);
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
