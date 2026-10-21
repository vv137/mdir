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
#include "mdir/Conversion/ReductionEmitter.h"

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
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
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
  return success();
}

/// Makes each global (a constant of the body) that a kernel reads be read
/// where the kernel is, so that the outlining takes it into the kernel's
/// module.
static void readGlobalsInside(gpu::LaunchOp launch) {
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

/// Whether fusing the producer of `operand` into its consumer evaluates the
/// producer no more often than it has elements: the consumer reads the
/// operand through a permutation of its loops. Otherwise the consumer reads
/// each element of the operand at several of its iterations, and the fused
/// producer would be evaluated at each: that is allowed only for a cheap
/// producer in a consumer without reductions (a broadcast), never into a
/// contraction, whose operands stay what it reads, so that its loop over
/// the contracted dimension does not evaluate them again for each column of
/// the result.
static bool fusesWithoutRecomputing(OpOperand *operand) {
  auto producer = operand->get().getDefiningOp<linalg::GenericOp>();
  auto consumer = dyn_cast<linalg::GenericOp>(operand->getOwner());
  if (!producer || !consumer)
    return false;
  if (consumer.getMatchingIndexingMap(operand).isPermutation())
    return true;
  return consumer.getNumReductionLoops() == 0 && isCheap(producer);
}

/// Elementwise fusion, the patterns of upstream's
/// `linalg-fuse-elementwise-ops` (its fusion, folding of reshapes by
/// expansion, and canonicalizations), with a control that fuses a producer
/// of one use where `fusesWithoutRecomputing` allows it.
static LogicalResult fuseElementwise(ModuleOp bodies) {
  MLIRContext *context = bodies.getContext();
  RewritePatternSet patterns(context);
  linalg::ControlFusionFn control = [](OpOperand *operand) {
    Operation *producer = operand->get().getDefiningOp();
    return producer && producer->hasOneUse() &&
           fusesWithoutRecomputing(operand);
  };
  linalg::ControlFusionFn oneUse = [](OpOperand *operand) {
    Operation *producer = operand->get().getDefiningOp();
    return producer && producer->hasOneUse();
  };
  linalg::populateElementwiseOpsFusionPatterns(patterns, control);
  linalg::populateFoldReshapeOpsByExpansionPatterns(patterns, oneUse);
  tensor::populateBubbleUpExpandShapePatterns(patterns);
  affine::AffineApplyOp::getCanonicalizationPatterns(patterns, context);
  linalg::GenericOp::getCanonicalizationPatterns(patterns, context);
  tensor::ExpandShapeOp::getCanonicalizationPatterns(patterns, context);
  tensor::CollapseShapeOp::getCanonicalizationPatterns(patterns, context);
  context->getLoadedDialect<linalg::LinalgDialect>()
      ->getCanonicalizationPatterns(patterns);
  linalg::populateConstantFoldLinalgOperations(patterns, oneUse);
  GreedyRewriteConfig config;
  config.setUseTopDownTraversal(true);
  return applyPatternsGreedily(bodies, std::move(patterns), config);
}

/// Gives each consumer that elementwise fusion would take a copy of its
/// elementwise producer when the producer has several uses: the fusion
/// takes a producer of one use only, and a producer of several stays its own
/// kernel, its result a round trip through global memory. A copy is made
/// where `fusesWithoutRecomputing` allows it; the original stays for the
/// uses that are not fused (a contraction, a slice, the return).
/// IREE clones into each dispatch region that uses them only the producers
/// that cost nothing to recompute (fills, splat constants, scalar
/// arithmetic; `isCloneableIntoDispatchOp` in iree-org/iree v3.12.0,
/// `compiler/src/iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.cpp`),
/// and fuses another elementwise producer of several uses into the consumer
/// that dominates the others, which then also yields the producer's result
/// (`DispatchCreation/FuseMultiUseElementwiseProducer.cpp`). Here such a
/// producer is evaluated again in each consumer instead: it reads its own
/// operands once per element either way, and the copy saves the write and
/// the read of its result. Returns whether anything was copied.
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
      if (!consumer || !linalg::areElementwiseOpsFusable(use) ||
          !fusesWithoutRecomputing(use))
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

/// The declaration of the runtime's function `name` in `bodies`.
static func::FuncOp getOrDeclareRuntime(ModuleOp bodies, StringRef name,
                                        FunctionType type) {
  if (auto known = bodies.lookupSymbol<func::FuncOp>(name))
    return known;
  OpBuilder builder = OpBuilder::atBlockEnd(bodies.getBody());
  auto declared = func::FuncOp::create(builder, bodies.getLoc(), name, type);
  declared.setPrivate();
  return declared;
}

/// A product of the shape of a matrix product in f32 (D302):
/// for each index of the batch dimension (if any), C (m x n) += A B over k,
/// A m x k and B k x n. Each of m and n is the product of the extents of the
/// loops it stands for, 1 for none; how each matrix lies in memory is read
/// from its buffer by `callGemms`.
struct Gemm {
  /// The input that is A, and the one that is B.
  unsigned a, b;
  /// The loops: the batch (at most one), the rows of C (those of A), the
  /// columns of C (those of B, at most one), and the reduced loop.
  SmallVector<unsigned> batch, rows, cols;
  unsigned reduced;
  /// Whether the payload starts the sum from zero at the first reduced
  /// iteration instead of reading the init (`startReductionsFromConstants`).
  bool fromZero = false;
};

/// The loop dimensions that `map` reads, in its order, if each of its
/// results is a dimension.
static std::optional<SmallVector<unsigned>> getDims(AffineMap map) {
  SmallVector<unsigned> out;
  for (AffineExpr expr : map.getResults()) {
    auto dim = dyn_cast<AffineDimExpr>(expr);
    if (!dim)
      return std::nullopt;
    out.push_back(dim.getPosition());
  }
  return out;
}

/// Whether `value`, in the payload of `op`, is the accumulator read from the
/// init, or (with `fromZero` set) a select of zero at the first iteration of
/// the reduced loop `reduced` and of the accumulator after it, as
/// `startReductionsFromConstants` writes it.
static bool isAccumulator(linalg::LinalgOp op, Value value, unsigned reduced,
                          bool &fromZero) {
  Value acc = op.getBlock()->getArgument(2);
  if (value == acc)
    return true;
  auto select = value.getDefiningOp<arith::SelectOp>();
  if (!select || select.getFalseValue() != acc ||
      !matchPattern(select.getTrueValue(), m_AnyZeroFloat()))
    return false;
  auto first = select.getCondition().getDefiningOp<arith::CmpIOp>();
  if (!first || first.getPredicate() != arith::CmpIPredicate::eq ||
      !matchPattern(first.getRhs(), m_Zero()))
    return false;
  auto index = first.getLhs().getDefiningOp<linalg::IndexOp>();
  if (!index || index.getDim() != reduced)
    return false;
  fromZero = true;
  return true;
}

/// `op` as a GEMM, batched or not: two operands of f32 and one result, of
/// static shapes; one reduced loop; every other loop indexes the result,
/// and either both operands (the batch: at most one, the result's first
/// dimension) or one of them; the result's dimensions after the batch are
/// those of the first operand that indexes them (A, the rows: several rows
/// dimensions are one, as IREE treats the three components of PaiNN's
/// vectors as rows) and then at most one of the other (B, the columns);
/// each operand indexed by its own loops, the batch, and the reduced loop,
/// in any order; rows or columns (not a dot product); and a payload that adds the product of the operands to the
/// result (or to zero at the first reduced iteration), or a `linalg.matmul`
/// or `linalg.batch_matmul`.
static std::optional<Gemm> matchGemm(linalg::LinalgOp op) {
  if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1 ||
      op.getNumLoops() < 2 || op.getNumReductionLoops() != 1 ||
      !isa<linalg::GenericOp, linalg::MatmulOp, linalg::BatchMatmulOp>(
          op.getOperation()))
    return std::nullopt;
  for (OpOperand &operand : op->getOpOperands()) {
    auto shaped = dyn_cast<ShapedType>(operand.get().getType());
    if (!shaped || !shaped.getElementType().isF32() ||
        !shaped.hasStaticShape())
      return std::nullopt;
  }
  SmallVector<int64_t> extents = op.getStaticLoopRanges();
  if (llvm::any_of(extents, ShapedType::isDynamic))
    return std::nullopt;
  unsigned numLoops = op.getNumLoops();
  auto out = getDims(op.getMatchingIndexingMap(op.getDpsInitOperand(0)));
  auto in0 = getDims(op.getMatchingIndexingMap(op.getDpsInputOperand(0)));
  auto in1 = getDims(op.getMatchingIndexingMap(op.getDpsInputOperand(1)));
  if (!out || !in0 || !in1 || out->size() != numLoops - 1)
    return std::nullopt;
  Gemm gemm;
  gemm.reduced = numLoops;
  SmallVector<utils::IteratorType> types = op.getIteratorTypesArray();
  for (unsigned d = 0; d != numLoops; ++d)
    if (types[d] == utils::IteratorType::reduction)
      gemm.reduced = d;
  if (llvm::is_contained(*out, gemm.reduced) ||
      !llvm::is_contained(*in0, gemm.reduced) ||
      !llvm::is_contained(*in1, gemm.reduced))
    return std::nullopt;
  // The loops of the result: the batch, then those of A, then those of B.
  ArrayRef<unsigned> rest(*out);
  if (!rest.empty() && llvm::is_contained(*in0, rest.front()) &&
      llvm::is_contained(*in1, rest.front())) {
    gemm.batch.push_back(rest.front());
    rest = rest.drop_front();
  }
  gemm.a = 0;
  if (!rest.empty() && !llvm::is_contained(*in0, rest.front()))
    gemm.a = 1;
  gemm.b = 1 - gemm.a;
  ArrayRef<unsigned> inA = gemm.a == 0 ? *in0 : *in1;
  ArrayRef<unsigned> inB = gemm.a == 0 ? *in1 : *in0;
  for (unsigned d : rest) {
    bool inAOnly = llvm::is_contained(inA, d) && !llvm::is_contained(inB, d);
    bool inBOnly = llvm::is_contained(inB, d) && !llvm::is_contained(inA, d);
    if (inAOnly && gemm.cols.empty())
      gemm.rows.push_back(d);
    else if (inBOnly && gemm.cols.empty())
      gemm.cols.push_back(d);
    else
      return std::nullopt;
  }
  // Each operand reads its own loops, the batch, and the reduced loop, each
  // once.
  auto reads = [&](ArrayRef<unsigned> in, ArrayRef<unsigned> own) {
    SmallVector<unsigned> expected(gemm.batch);
    expected.append(own.begin(), own.end());
    expected.push_back(gemm.reduced);
    SmallVector<unsigned> sortedIn(in), sortedExpected(expected);
    llvm::sort(sortedIn);
    llvm::sort(sortedExpected);
    return sortedIn == sortedExpected;
  };
  if (!reads(inA, gemm.rows) || !reads(inB, gemm.cols))
    return std::nullopt;
  // A product of 1 x k by k x 1 for each index of the batch is a dot product
  // of rows: a sum of rows (`warpRowSums`), not a GEMM.
  if (gemm.rows.empty() && gemm.cols.empty())
    return std::nullopt;
  if (isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op.getOperation()))
    return gemm;
  // The payload: (out or zero at the first reduced iteration) + a * b, in
  // either order of each.
  Block *body = op.getBlock();
  auto yield = cast<linalg::YieldOp>(body->getTerminator());
  if (yield.getNumOperands() != 1)
    return std::nullopt;
  auto add = yield.getOperand(0).getDefiningOp<arith::AddFOp>();
  if (!add || add->getBlock() != body)
    return std::nullopt;
  Value x = body->getArgument(0), y = body->getArgument(1);
  for (auto [product, start] : {std::make_pair(add.getLhs(), add.getRhs()),
                                std::make_pair(add.getRhs(), add.getLhs())}) {
    auto mul = product.getDefiningOp<arith::MulFOp>();
    if (!mul || !((mul.getLhs() == x && mul.getRhs() == y) ||
                  (mul.getLhs() == y && mul.getRhs() == x)))
      continue;
    bool fromZero = false;
    if (!isAccumulator(op, start, gemm.reduced, fromZero))
      continue;
    gemm.fromZero = fromZero;
    // Nothing else in the payload: the ops of the select, if any.
    for (Operation &inner : body->without_terminator())
      if (!isa<arith::MulFOp, arith::AddFOp, arith::SelectOp, arith::CmpIOp,
               arith::ConstantOp, linalg::IndexOp>(inner))
        return std::nullopt;
    return gemm;
  }
  return std::nullopt;
}

/// The device's address of the first element of the buffer `buffer` and the
/// distance in elements between consecutive indices of each of its
/// dimensions, as i64; a constant of the body (a global) is copied to the
/// device once by the runtime. Fails on a buffer whose layout is not static.
static FailureOr<std::pair<Value, SmallVector<int64_t>>>
deviceAddress(OpBuilder &builder, Location loc, Value buffer,
              ModuleOp bodies) {
  auto type = cast<MemRefType>(buffer.getType());
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(type.getStridesAndOffset(strides, offset)) ||
      llvm::any_of(strides, ShapedType::isDynamic))
    return failure();
  Type i64 = builder.getI64Type();
  auto metadata =
      memref::ExtractStridedMetadataOp::create(builder, loc, buffer);
  Value base = arith::IndexCastOp::create(
      builder, loc, i64,
      memref::ExtractAlignedPointerAsIndexOp::create(builder, loc, buffer));
  // A constant of the body, on the host: its copy on the device.
  Value root = buffer;
  while (Operation *def = root.getDefiningOp()) {
    if (auto view = dyn_cast<ViewLikeOpInterface>(def))
      root = view.getViewSource();
    else
      break;
  }
  if (auto global = root.getDefiningOp<memref::GetGlobalOp>()) {
    auto rootType = cast<MemRefType>(global.getType());
    int64_t bytes = rootType.getNumElements() * 4;
    Value hostBase = arith::IndexCastOp::create(
        builder, loc, i64,
        memref::ExtractAlignedPointerAsIndexOp::create(builder, loc, global));
    Value offsetInRoot = arith::SubIOp::create(builder, loc, base, hostBase);
    auto declare = getOrDeclareRuntime(bodies, "mdrtDeviceConstant",
                                       builder.getFunctionType({i64, i64}, {i64}));
    Value device =
        func::CallOp::create(builder, loc, declare,
                             ValueRange{hostBase,
                                        arith::ConstantIntOp::create(
                                            builder, loc, i64, bytes)})
            .getResult(0);
    base = arith::AddIOp::create(builder, loc, device, offsetInRoot);
  }
  Value elements = arith::IndexCastOp::create(builder, loc, i64,
                                              metadata.getOffset());
  Value address = arith::AddIOp::create(
      builder, loc, base,
      arith::MulIOp::create(builder, loc, elements,
                            arith::ConstantIntOp::create(builder, loc, i64, 4)));
  return std::make_pair(address, strides);
}

/// One matrix of a GEMM as cuBLAS takes it: row-major p x q with rows `ld`
/// apart, or (`transposed`) its transpose stored row-major, q x p.
struct Matrix {
  int64_t ld;
  bool transposed;
};

/// The matrix of extents p x q whose rows' and columns' distances in memory
/// are `rowStride` and `colStride` (nothing for an extent made of no loop),
/// or nothing if neither is contiguous. A dimension of extent 1 is
/// contiguous whatever its distance; cuBLAS requires the leading dimension
/// to be at least the extent of the contiguous one.
static std::optional<Matrix> asMatrix(int64_t p, std::optional<int64_t> rowStride,
                                      int64_t q,
                                      std::optional<int64_t> colStride,
                                      bool allowTranspose) {
  if (q == 1 || colStride == 1) {
    int64_t ld = p == 1 ? q : *rowStride;
    if (ld >= q)
      return Matrix{ld, false};
  }
  if (allowTranspose && (p == 1 || rowStride == 1)) {
    int64_t ld = q == 1 ? p : *colStride;
    if (ld >= p)
      return Matrix{ld, true};
  }
  return std::nullopt;
}

/// Replaces each GEMM of the bufferized body `function` with a call of
/// cuBLAS's f32 GEMM through the runtime, strided-batched for a GEMM with a
/// batch, on the stream of the kernels. A result that a `linalg.fill` of
/// zero initializes just before is not filled: the call overwrites it (beta
/// 0), as it does where the payload starts the sum from zero. A product
/// whose buffers cuBLAS cannot take (rows dimensions that are not one after
/// the other, neither dimension of a matrix contiguous) stays loops.
static LogicalResult callGemms(func::FuncOp function, ModuleOp bodies) {
  SmallVector<std::pair<linalg::LinalgOp, Gemm>> found;
  function.walk([&](linalg::LinalgOp op) {
    if (op.hasPureBufferSemantics() && op->getBlock() ==
                                           &function.getBody().front())
      if (std::optional<Gemm> gemm = matchGemm(op))
        found.push_back({op, *gemm});
  });
  Type i64 = IntegerType::get(function.getContext(), 64);
  Type i32 = IntegerType::get(function.getContext(), 32);
  Type f32 = Float32Type::get(function.getContext());
  for (auto [op, gemm] : found) {
    Location loc = op.getLoc();
    SmallVector<int64_t> extents = op.getStaticLoopRanges();
    // The distance of each loop in each operand, from its type.
    auto distances = [&](OpOperand *operand)
        -> std::optional<llvm::DenseMap<unsigned, int64_t>> {
      auto type = cast<MemRefType>(operand->get().getType());
      SmallVector<int64_t> strides;
      int64_t offset;
      if (failed(type.getStridesAndOffset(strides, offset)) ||
          llvm::any_of(strides, ShapedType::isDynamic))
        return std::nullopt;
      llvm::DenseMap<unsigned, int64_t> of;
      for (auto [i, d] : llvm::enumerate(
               *getDims(op.getMatchingIndexingMap(operand))))
        of[d] = strides[i];
      return of;
    };
    auto inA = distances(op.getDpsInputOperand(gemm.a));
    auto inB = distances(op.getDpsInputOperand(gemm.b));
    auto inC = distances(op.getDpsInitOperand(0));
    if (!inA || !inB || !inC)
      continue;
    // Several loops of rows are one if they lie one after the other in A
    // and in C; the distance of the rows is then that of the last.
    int64_t m = 1, n = 1, k = extents[gemm.reduced], batch = 1;
    bool joined = true;
    for (auto [i, d] : llvm::enumerate(gemm.rows)) {
      m *= extents[d];
      if (i + 1 == gemm.rows.size())
        continue;
      unsigned next = gemm.rows[i + 1];
      for (auto *of : {&*inA, &*inC})
        joined &= of->lookup(d) == of->lookup(next) * extents[next];
    }
    for (unsigned d : gemm.cols)
      n *= extents[d];
    for (unsigned d : gemm.batch)
      batch *= extents[d];
    if (!joined)
      continue;
    auto at = [](const llvm::DenseMap<unsigned, int64_t> &of,
                 ArrayRef<unsigned> dims) -> std::optional<int64_t> {
      if (dims.empty())
        return std::nullopt;
      return of.lookup(dims.back());
    };
    SmallVector<unsigned> reduced{gemm.reduced};
    std::optional<Matrix> a =
        asMatrix(m, at(*inA, gemm.rows), k, at(*inA, reduced), true);
    std::optional<Matrix> b =
        asMatrix(k, at(*inB, reduced), n, at(*inB, gemm.cols), true);
    std::optional<Matrix> c =
        asMatrix(m, at(*inC, gemm.rows), n, at(*inC, gemm.cols), false);
    if (!a || !b || !c)
      continue;
    auto batchStride = [&](const llvm::DenseMap<unsigned, int64_t> &of) {
      return gemm.batch.empty() ? int64_t(0) : of.lookup(gemm.batch.front());
    };

    OpBuilder builder(op);
    Value cBuffer = op.getDpsInitOperand(0)->get();
    // The fill of zero just before, of the same buffer, becomes beta 0.
    float beta = gemm.fromZero ? 0.0f : 1.0f;
    linalg::FillOp fill;
    for (Operation *user : cBuffer.getUsers())
      if (auto candidate = dyn_cast<linalg::FillOp>(user))
        if (candidate->getBlock() == op->getBlock() &&
            candidate->isBeforeInBlock(op) &&
            candidate.getOutputs()[0] == cBuffer &&
            matchPattern(candidate.getInputs()[0], m_AnyZeroFloat()))
          fill = candidate;
    // Nothing else touches the buffer between the fill and the product: a
    // view of it made there counts only if something there uses the view.
    std::function<bool(Value)> touchedBetween = [&](Value buffer) {
      return llvm::any_of(buffer.getUsers(), [&](Operation *user) {
        Operation *at = op->getBlock()->findAncestorOpInBlock(*user);
        if (user == fill || user == op || !at || !fill->isBeforeInBlock(at) ||
            !at->isBeforeInBlock(op))
          return false;
        if (isa<ViewLikeOpInterface>(user) && user->getNumResults() == 1)
          return touchedBetween(user->getResult(0));
        return true;
      });
    };
    if (!gemm.fromZero && fill && !touchedBetween(cBuffer)) {
      beta = 0.0f;
      fill.erase();
    }
    auto addrA = deviceAddress(builder, loc,
                               op.getDpsInputOperand(gemm.a)->get(), bodies);
    auto addrB = deviceAddress(builder, loc,
                               op.getDpsInputOperand(gemm.b)->get(), bodies);
    auto addrC = deviceAddress(builder, loc, cBuffer, bodies);
    if (failed(addrA) || failed(addrB) || failed(addrC))
      return op.emitOpError() << "is a product of the shape of a GEMM over "
                                 "a buffer of a layout that is not static";
    auto constant = [&](int64_t value) {
      return arith::ConstantIntOp::create(builder, loc, i64, value).getResult();
    };
    auto flag = [&](bool value) {
      return arith::ConstantIntOp::create(builder, loc, i32, value).getResult();
    };
    auto declare = getOrDeclareRuntime(
        bodies, "mdrtGemmF32",
        builder.getFunctionType({i64, i64, i64, i64, i64, i64, i32, i64, i64,
                                 i64, i32, i64, i64, i64, f32, i64},
                                {}));
    func::CallOp::create(
        builder, loc, declare,
        ValueRange{constant(m), constant(n), constant(k), addrA->first,
                   constant(a->ld), constant(batchStride(*inA)),
                   flag(a->transposed), addrB->first, constant(b->ld),
                   constant(batchStride(*inB)), flag(b->transposed),
                   addrC->first, constant(c->ld), constant(batchStride(*inC)),
                   arith::ConstantOp::create(builder, loc, f32,
                                             builder.getF32FloatAttr(beta)),
                   constant(batch)});
    op->erase();
  }
  return success();
}

/// The threads of a block of a kernel of row reductions: 8 warps, a row each.
constexpr int64_t kRowsBlock = 256;

/// The least extent of the reduced loop that a warp takes (one term a lane
/// at least).
constexpr int64_t kWarp = 32;

/// A sum of rows (D303): a generic of one result whose one
/// reduced loop, of `extent` iterations, is the last dimension of each
/// operand that it indexes, and whose payload yields `start + term`, with
/// `term` computed from the operands without the accumulator and `start`
/// the accumulator or, as `startReductionsFromConstants` writes it, the
/// constant `constant` at the first reduced iteration and the accumulator
/// after it.
struct RowSum {
  unsigned reduced;
  int64_t extent;
  Value term;
  Value constant;
};

/// `op` as a sum of rows in f32 or f64, on buffers. Short sums are kept
/// in one thread, while long sums require contiguous operands for a warp.
static std::optional<RowSum> matchRowSum(linalg::GenericOp op,
                                          bool shortSum = false) {
  if (!op.hasPureBufferSemantics() || op.getNumDpsInits() != 1 ||
      op.getNumReductionLoops() != 1)
    return std::nullopt;
  SmallVector<int64_t> extents = op.getStaticLoopRanges();
  if (llvm::any_of(extents, ShapedType::isDynamic))
    return std::nullopt;
  RowSum sum;
  for (auto [d, type] : llvm::enumerate(op.getIteratorTypesArray()))
    if (type == utils::IteratorType::reduction)
      sum.reduced = d;
  sum.extent = extents[sum.reduced];
  if (shortSum ? (sum.extent < 2 || sum.extent >= kWarp)
               : (sum.extent < kWarp))
    return std::nullopt;
  OpOperand *init = op.getDpsInitOperand(0);
  Type element = getElementTypeOrSelf(init->get().getType());
  if (!element.isF32() && !element.isF64())
    return std::nullopt;
  auto out = getDims(op.getMatchingIndexingMap(init));
  if (!out || out->size() != op.getNumLoops() - 1 ||
      llvm::is_contained(*out, sum.reduced))
    return std::nullopt;
  // The reduced loop runs along the contiguous dimension of what it reads,
  // so that the lanes of a warp read consecutive elements.
  for (OpOperand *input : op.getDpsInputOperands()) {
    auto dims = getDims(op.getMatchingIndexingMap(input));
    if (!dims)
      return std::nullopt;
    if (!shortSum && llvm::is_contained(*dims, sum.reduced) &&
        dims->back() != sum.reduced)
      return std::nullopt;
  }
  Block *body = op.getBody();
  // The explicit kernel clones only the term's dependencies. Reads are part
  // of a row term (including captured buffer loads in long PaiNN sums);
  // writes, unknown effects, and nested regions retain the loop lowering.
  for (Operation &inner : body->without_terminator())
    if ((!isMemoryEffectFree(&inner) && !isa<memref::LoadOp>(inner)) ||
        inner.getNumRegions() != 0)
      return std::nullopt;
  BlockArgument acc = op.getMatchingBlockArgument(init);
  auto yield = cast<linalg::YieldOp>(body->getTerminator());
  auto add = yield.getOperand(0).getDefiningOp<arith::AddFOp>();
  if (!add || add->getBlock() != body)
    return std::nullopt;
  for (auto [start, term] : {std::make_pair(add.getLhs(), add.getRhs()),
                             std::make_pair(add.getRhs(), add.getLhs())}) {
    Value constant;
    if (start != acc) {
      auto select = start.getDefiningOp<arith::SelectOp>();
      if (!select || select.getFalseValue() != acc ||
          !matchPattern(select.getTrueValue(), m_Constant()) ||
          !select->hasOneUse())
        continue;
      auto first = select.getCondition().getDefiningOp<arith::CmpIOp>();
      if (!first || first.getPredicate() != arith::CmpIPredicate::eq ||
          !matchPattern(first.getRhs(), m_Zero()))
        continue;
      auto index = first.getLhs().getDefiningOp<linalg::IndexOp>();
      if (!index || index.getDim() != sum.reduced)
        continue;
      constant = select.getTrueValue();
    }
    // The accumulator is read by `start` alone, and the term does not
    // depend on it.
    if (!llvm::all_of(acc.getUsers(), [&](Operation *user) {
          return user == add || user == start.getDefiningOp();
        }))
      continue;
    sum.term = term;
    sum.constant = constant;
    return sum;
  }
  return std::nullopt;
}

/// The pure operations of a row-sum term, shared by the register and warp
/// strategies. The matcher has already checked that the accumulator is not
/// one of their dependencies.
static llvm::SetVector<Operation *> rowTermOps(RowSum sum, Block *body) {
  llvm::SetVector<Operation *> needed;
  SmallVector<Value> pending{sum.term};
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    Operation *def = value.getDefiningOp();
    if (!def || def->getBlock() != body || !needed.insert(def))
      continue;
    for (Value operand : def->getOperands())
      pending.push_back(operand);
  }
  return needed;
}

static Value emitRowTerm(OpBuilder &builder, Location loc, linalg::GenericOp op,
                         RowSum sum, ArrayRef<Value> indices,
                         const llvm::SetVector<Operation *> &needed) {
  IRMapping mapping;
  for (OpOperand *input : op.getDpsInputOperands()) {
    SmallVector<Value> place;
    for (unsigned d : *getDims(op.getMatchingIndexingMap(input)))
      place.push_back(indices[d]);
    mapping.map(op.getMatchingBlockArgument(input),
                memref::LoadOp::create(builder, loc, input->get(), place));
  }
  for (Operation &inner : op.getBody()->without_terminator()) {
    if (!needed.contains(&inner))
      continue;
    if (auto idx = dyn_cast<linalg::IndexOp>(inner))
      mapping.map(idx.getResult(), indices[idx.getDim()]);
    else
      builder.clone(inner, mapping);
  }
  return mapping.lookupOrDefault(sum.term);
}

/// An elementwise epilogue that reads only this short sum, at the same
/// output indices. The sum is still stored in its buffer: another reader or
/// a later reuse of the allocation therefore sees the original value. The
/// epilogue writes a fresh buffer, so running it in each sum thread cannot
/// overwrite an operand needed by another thread.
static linalg::GenericOp matchShortRowEpilogue(linalg::GenericOp sum) {
  Operation *next = sum->getNextNode();
  while (next && (isa<memref::AllocOp>(next) || isa<gpu::AllocOp>(next)))
    next = next->getNextNode();
  auto epilogue = dyn_cast_or_null<linalg::GenericOp>(next);
  if (!epilogue || !epilogue.hasPureBufferSemantics() ||
      epilogue.getNumDpsInputs() != 1 || epilogue.getNumDpsInits() != 1 ||
      epilogue.getNumReductionLoops() != 0 ||
      epilogue.getDpsInputOperand(0)->get() !=
          sum.getDpsInitOperand(0)->get())
    return {};
  OpOperand *input = epilogue.getDpsInputOperand(0);
  OpOperand *output = epilogue.getDpsInitOperand(0);
  if (!epilogue.getMatchingIndexingMap(input).isIdentity() ||
      !epilogue.getMatchingIndexingMap(output).isIdentity() ||
      epilogue.getNumLoops() !=
          cast<MemRefType>(output->get().getType()).getRank() ||
      !epilogue.getMatchingBlockArgument(output).use_empty() ||
      !isa_and_nonnull<memref::AllocOp, gpu::AllocOp>(
          output->get().getDefiningOp()))
    return {};
  if (cast<MemRefType>(input->get().getType()).getShape() !=
      cast<MemRefType>(output->get().getType()).getShape())
    return {};
  for (Operation &inner : epilogue.getBody()->without_terminator())
    if (!isMemoryEffectFree(&inner) || inner.getNumRegions() != 0)
      return {};
  return epilogue;
}

/// Replaces each sum of rows of the bufferized body `function` with a kernel
/// of a warp for each element of the result (D303, the
/// reductions of the rows that IREE gives a warp each, `VectorDistribute`):
/// lane l adds the terms l, l + 32, l + 64, ... in that order, the warp
/// combines its lanes by the butterfly of D301 (`kernels::emitWarpReduce`),
/// and the first lane stores the start (the init's element, or the
/// payload's constant) plus that sum. The order is fixed by the lanes, so
/// the result is the same on every run, in every mode; the sum is taken in
/// the type of the result, as the loops took it.
static void warpRowSums(func::FuncOp function) {
  SmallVector<std::pair<linalg::GenericOp, RowSum>> found;
  function.walk([&](linalg::GenericOp op) {
    if (op->getBlock() == &function.getBody().front())
      if (std::optional<RowSum> sum = matchRowSum(op))
        found.push_back({op, *sum});
  });
  for (auto [op, sum] : found) {
    Location loc = op.getLoc();
    SmallVector<int64_t> extents = op.getStaticLoopRanges();
    unsigned numLoops = op.getNumLoops();
    int64_t rows = 1;
    for (unsigned d = 0; d != numLoops; ++d)
      if (d != sum.reduced)
        rows *= extents[d];
    OpBuilder builder(op);
    if (rows == 0) {
      op.erase();
      continue;
    }
    int64_t warps = kRowsBlock / kWarp;
    Value one = index(builder, loc, 1);
    auto launch = gpu::LaunchOp::create(
        builder, loc, index(builder, loc, (rows + warps - 1) / warps), one, one,
        index(builder, loc, kRowsBlock), one, one);
    OpBuilder kernel = OpBuilder::atBlockEnd(&launch.getBody().front());
    Value thread = launch.getThreadIds().x;
    Value row = arith::AddIOp::create(
        kernel, loc,
        arith::MulIOp::create(kernel, loc, launch.getBlockIds().x,
                              index(kernel, loc, warps)),
        arith::DivUIOp::create(kernel, loc, thread, index(kernel, loc, kWarp)));
    Value lane =
        arith::RemUIOp::create(kernel, loc, thread, index(kernel, loc, kWarp));
    // A warp is inside or outside as a whole: every lane of a warp inside
    // reaches the butterfly.
    Value inside = arith::CmpIOp::create(kernel, loc, arith::CmpIPredicate::ult,
                                         row, index(kernel, loc, rows));
    auto guard = scf::IfOp::create(kernel, loc, inside, /*withElse=*/false);
    OpBuilder then = guard.getThenBodyBuilder();
    // The indices of the parallel loops, the last fastest.
    SmallVector<Value> at(numLoops);
    Value rest = row;
    for (unsigned d = numLoops; d-- != 0;) {
      if (d == sum.reduced)
        continue;
      at[d] = arith::RemUIOp::create(then, loc, rest,
                                     index(then, loc, extents[d]));
      rest = arith::DivUIOp::create(then, loc, rest,
                                    index(then, loc, extents[d]));
    }
    // The ops of the payload that the term needs, in their order.
    llvm::SetVector<Operation *> needed = rowTermOps(sum, op.getBody());
    auto term = [&](OpBuilder &b, Value k) {
      SmallVector<Value> indices(at);
      indices[sum.reduced] = k;
      return emitRowTerm(b, loc, op, sum, indices, needed);
    };
    // Lane l: the terms l, l + 32, ..., in order (the extent is at least
    // 32, so each lane has one).
    Value first = term(then, lane);
    auto loop = scf::ForOp::create(
        then, loc, arith::AddIOp::create(then, loc, lane,
                                         index(then, loc, kWarp)),
        index(then, loc, sum.extent), index(then, loc, kWarp),
        ValueRange{first});
    {
      OpBuilder inner = OpBuilder::atBlockBegin(loop.getBody());
      Value next = term(inner, loop.getInductionVar());
      scf::YieldOp::create(
          inner, loc,
          ValueRange{arith::AddFOp::create(inner, loc,
                                           loop.getRegionIterArgs()[0], next)
                         .getResult()});
    }
    Value total = kernels::emitWarpReduce(then, loc, loop.getResult(0),
                                          kernels::Combiner::Add);
    Value isFirst = arith::CmpIOp::create(
        then, loc, arith::CmpIPredicate::eq, lane, index(then, loc, 0));
    auto store = scf::IfOp::create(then, loc, isFirst, /*withElse=*/false);
    OpBuilder last = store.getThenBodyBuilder();
    OpOperand *init = op.getDpsInitOperand(0);
    SmallVector<Value> place;
    for (unsigned d : *getDims(op.getMatchingIndexingMap(init)))
      place.push_back(at[d]);
    Value start = sum.constant
                      ? last.clone(*sum.constant.getDefiningOp())->getResult(0)
                      : memref::LoadOp::create(last, loc, init->get(), place)
                            .getResult();
    memref::StoreOp::create(
        last, loc, arith::AddFOp::create(last, loc, start, total),
        init->get(), place);
    gpu::TerminatorOp::create(kernel, loc);
    op.erase();
  }
}

/// A short reduction (in PaiNN, the three vector components) needs one
/// thread for each result. Unroll its terms into that thread so that its
/// accumulator stays in a register instead of being loaded and stored at
/// each iteration of the loop lowering. The order of the additions is the
/// original order, including the first term's constant initial value.
static void shortRowSums(func::FuncOp function, bool fuseElementwise) {
  struct Candidate {
    linalg::GenericOp op;
    RowSum sum;
    linalg::GenericOp epilogue;
  };
  SmallVector<Candidate> found;
  function.walk([&](linalg::GenericOp op) {
    if (op->getBlock() == &function.getBody().front())
      if (std::optional<RowSum> sum = matchRowSum(op, /*shortSum=*/true))
        found.push_back(
            {op, *sum, fuseElementwise ? matchShortRowEpilogue(op)
                                       : linalg::GenericOp{}});
  });
  for (auto [op, sum, epilogue] : found) {
    Location loc = op.getLoc();
    SmallVector<int64_t> extents = op.getStaticLoopRanges();
    int64_t rows = 1;
    for (unsigned d = 0; d != op.getNumLoops(); ++d)
      if (d != sum.reduced)
        rows *= extents[d];
    OpBuilder builder(epilogue ? epilogue.getOperation() : op.getOperation());
    if (rows == 0) {
      op.erase();
      continue;
    }
    Value one = index(builder, loc, 1);
    auto launch = gpu::LaunchOp::create(
        builder, loc, index(builder, loc, (rows + kRowsBlock - 1) / kRowsBlock),
        one, one, index(builder, loc, kRowsBlock), one, one);
    OpBuilder kernel = OpBuilder::atBlockEnd(&launch.getBody().front());
    Value row = arith::AddIOp::create(
        kernel, loc,
        arith::MulIOp::create(kernel, loc, launch.getBlockIds().x,
                              index(kernel, loc, kRowsBlock)),
        launch.getThreadIds().x);
    Value inside = arith::CmpIOp::create(kernel, loc, arith::CmpIPredicate::ult,
                                         row, index(kernel, loc, rows));
    auto guard = scf::IfOp::create(kernel, loc, inside, /*withElse=*/false);
    OpBuilder then = guard.getThenBodyBuilder();
    SmallVector<Value> at(op.getNumLoops());
    Value rest = row;
    for (unsigned d = op.getNumLoops(); d-- != 0;) {
      if (d == sum.reduced)
        continue;
      at[d] = arith::RemUIOp::create(then, loc, rest,
                                     index(then, loc, extents[d]));
      rest = arith::DivUIOp::create(then, loc, rest,
                                    index(then, loc, extents[d]));
    }
    llvm::SetVector<Operation *> needed = rowTermOps(sum, op.getBody());
    auto term = [&](OpBuilder &b, int64_t k) {
      SmallVector<Value> indices(at);
      indices[sum.reduced] = index(b, loc, k);
      return emitRowTerm(b, loc, op, sum, indices, needed);
    };
    OpOperand *init = op.getDpsInitOperand(0);
    SmallVector<Value> place;
    for (unsigned d : *getDims(op.getMatchingIndexingMap(init)))
      place.push_back(at[d]);
    Value value = sum.constant
                      ? then.clone(*sum.constant.getDefiningOp())->getResult(0)
                      : memref::LoadOp::create(then, loc, init->get(), place)
                            .getResult();
    value = kernels::emitOrderedRegisterSum(then, loc, value, sum.extent, term);
    memref::StoreOp::create(then, loc, value, init->get(), place);
    if (epilogue) {
      IRMapping mapping;
      mapping.map(epilogue.getMatchingBlockArgument(
                      epilogue.getDpsInputOperand(0)),
                  value);
      for (Operation &inner : epilogue.getBody()->without_terminator()) {
        if (auto idx = dyn_cast<linalg::IndexOp>(inner))
          mapping.map(idx.getResult(), place[idx.getDim()]);
        else
          then.clone(inner, mapping);
      }
      Value result = mapping.lookupOrDefault(
          cast<linalg::YieldOp>(epilogue.getBody()->getTerminator())
              .getOperand(0));
      memref::StoreOp::create(then, loc, result,
                              epilogue.getDpsInitOperand(0)->get(), place);
    }
    gpu::TerminatorOp::create(kernel, loc);
    op.erase();
    if (epilogue)
      epilogue.erase();
  }
}

/// Starts each reduction whose initial value is a `linalg.fill` of a constant
/// from that constant, read at the first iteration of its reduced loops,
/// instead of from a buffer that a kernel of its own fills: the init becomes
/// `tensor.empty`, whose contents the payload no longer reads. This is the
/// meaning of the generic unchanged: at the first reduced iteration the
/// accumulator is the fill's value, and after it the partial sum. With
/// `keepGemms`, the GEMMs are left as they are, for `callGemms`.
static void startReductionsFromConstants(func::FuncOp function,
                                         bool keepGemms) {
  SmallVector<linalg::GenericOp> reductions;
  function.walk([&](linalg::GenericOp op) {
    if (op.hasPureTensorSemantics() && op.getNumReductionLoops() != 0 &&
        !(keepGemms && matchGemm(op)))
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
/// (a product), or with `keepGemms` of a GEMM, as a generic, which
/// elementwise fusion takes into its consumers like any other elementwise
/// producer.
static void generalizeFills(func::FuncOp function, bool keepGemms) {
  SmallVector<linalg::FillOp> fills;
  function.walk([&](linalg::FillOp fill) {
    bool initOfNamed = llvm::any_of(fill->getUses(), [&](OpOperand &use) {
      auto user = dyn_cast<linalg::LinalgOp>(use.getOwner());
      return user && user.isDpsInit(&use) &&
             (!isa<linalg::GenericOp>(user) || (keepGemms && matchGemm(user)));
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

/// A final pointwise producer can write identical results to two caller
/// buffers in one launch. Bufferization otherwise copies its temporary to
/// each result in two further launches. Restrict the source reads to local
/// allocations: the caller may alias an output with one of its inputs.
static void writeDuplicatedResultsInPlace(func::FuncOp function) {
  // This promise is established only for stage references that have no
  // direct callers. The tiled stage lowering allocates a fresh device buffer
  // for each result in planTiles; arbitrary calls need not do so.
  if (!function->hasAttr("md.stage_disjoint_results"))
    return;
  auto localAllocation = [](Value value) {
    while (Operation *def = value.getDefiningOp()) {
      if (auto view = dyn_cast<memref::SubViewOp>(def))
        value = view.getSource();
      else if (auto view = dyn_cast<memref::ExpandShapeOp>(def))
        value = view.getSrc();
      else if (auto view = dyn_cast<memref::CollapseShapeOp>(def))
        value = view.getSrc();
      else if (auto view = dyn_cast<memref::CastOp>(def))
        value = view.getSource();
      else
        return isa<memref::AllocOp, gpu::AllocOp>(def);
    }
    return false;
  };
  SmallVector<memref::AllocOp> allocations;
  function.walk([&](memref::AllocOp alloc) { allocations.push_back(alloc); });
  for (memref::AllocOp alloc : allocations) {
    SmallVector<memref::CopyOp> copies;
    linalg::GenericOp producer;
    bool valid = true;
    for (Operation *user : alloc->getUsers()) {
      if (auto copy = dyn_cast<memref::CopyOp>(user)) {
        if (copy.getSource() != alloc)
          valid = false;
        copies.push_back(copy);
      } else if (auto generic = dyn_cast<linalg::GenericOp>(user)) {
        if (producer || generic.getNumDpsInits() != 1 ||
            generic.getDpsInitOperand(0)->get() != alloc)
          valid = false;
        producer = generic;
      } else {
        valid = false;
      }
    }
    if (!valid || !producer || copies.size() != 2 ||
        !producer.hasPureBufferSemantics() ||
        producer.getNumReductionLoops() != 0 ||
        !producer.getMatchingIndexingMap(producer.getDpsInitOperand(0))
             .isIdentity() ||
        !producer.getMatchingBlockArgument(producer.getDpsInitOperand(0))
             .use_empty())
      continue;
    Block &entry = function.getBody().front();
    for (memref::CopyOp copy : copies) {
      auto target = dyn_cast<BlockArgument>(copy.getTarget());
      if (!target || target.getOwner() != &entry || !target.hasOneUse() ||
          target.getType() != alloc.getType() ||
          copy->getBlock() != &entry ||
          !producer->isBeforeInBlock(copy))
        valid = false;
    }
    for (Value input : producer.getDpsInputs())
      valid &= localAllocation(input);
    for (Operation &inner : producer.getBody()->without_terminator()) {
      if (auto load = dyn_cast<memref::LoadOp>(inner))
        valid &= localAllocation(load.getMemRef());
      else if (!isMemoryEffectFree(&inner) || inner.getNumRegions() != 0)
        valid = false;
    }
    if (!valid)
      continue;
    OpBuilder builder(producer);
    SmallVector<AffineMap> maps(producer.getIndexingMapsArray());
    maps.push_back(maps.back());
    SmallVector<Value> outputs{copies[0].getTarget(), copies[1].getTarget()};
    auto original = producer.getBody();
    auto replacement = linalg::GenericOp::create(
        builder, producer.getLoc(), TypeRange{}, producer.getDpsInputs(),
        outputs, maps, producer.getIteratorTypesArray(),
        [&](OpBuilder &bodyBuilder, Location loc, ValueRange arguments) {
          IRMapping mapping;
          for (auto [oldArg, newArg] :
               llvm::zip(original->getArguments(), arguments))
            mapping.map(oldArg, newArg);
          for (Operation &inner : original->without_terminator())
            bodyBuilder.clone(inner, mapping);
          Value result = mapping.lookupOrDefault(
              cast<linalg::YieldOp>(original->getTerminator()).getOperand(0));
          linalg::YieldOp::create(bodyBuilder, loc, ValueRange{result, result});
        });
    (void)replacement;
    for (memref::CopyOp copy : copies)
      copy.erase();
    producer.erase();
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
      // A body used only through md.stage is called by the tiled lowering,
      // whose planTiles allocates a separate buffer for every result. Do not
      // assume that contract when another symbol user can call this body.
      if (auto uses = SymbolTable::getSymbolUses(function, module)) {
        bool onlyStages = !uses->empty();
        for (const SymbolTable::SymbolUse &use : *uses) {
          auto stage = dyn_cast<md::StageOp>(use.getUser());
          if (!stage || (use.getSymbolRef() != stage.getForwardAttr() &&
                         use.getSymbolRef() != stage.getVjpAttr()))
            onlyStages = false;
        }
        if (onlyStages)
          function->setAttr("md.stage_disjoint_results",
                            UnitAttr::get(context));
      }
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

    // The strategy of the products of the shape of a GEMM: calls of cuBLAS
    // on a device, or loops (D302).
    if (contractions != "gemm" && contractions != "loops") {
      module.emitError() << "md-exec-lower-stage-bodies: contractions is '"
                         << contractions << "'; expected 'gemm' or 'loops'";
      return signalPassFailure();
    }
    bool gemm = device && contractions == "gemm";

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
        startReductionsFromConstants(function, gemm);
        generalizeFills(function, gemm);
      }
    }
    for (unsigned round = 0; round != (fuse ? 8 : 1); ++round) {
      PassManager pm(context);
      if (!fuse)
        pm.addPass(createLinalgElementwiseOpFusionPass());
      pm.addPass(createCanonicalizerPass());
      pm.addPass(createCSEPass());
      if (fuse && failed(fuseElementwise(*bodies)))
        return signalPassFailure();
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

    // Each GEMM that starts from a fill of zero gets a fill of its own, of a
    // tensor of its own, which bufferizes to a buffer that only the fill
    // and the product write: `callGemms` then drops the fill (beta 0). CSE
    // merges the fills of equal operands, and the product that would share
    // one would start from a copy of it instead.
    // An init whose values the payload does not read (an elementwise op's)
    // is any tensor of its shape: an empty one, so that a fill shared with
    // it through CSE does not stay for it alone.
    if (fuse)
      bodies->walk([&](linalg::GenericOp op) {
        if (!op.hasPureTensorSemantics())
          return;
        for (OpOperand &init : op.getDpsInitsMutable()) {
          if (!op.getMatchingBlockArgument(&init).use_empty() ||
              init.get().getDefiningOp<tensor::EmptyOp>())
            continue;
          OpBuilder builder(op);
          auto type = cast<RankedTensorType>(init.get().getType());
          init.set(tensor::EmptyOp::create(builder, op.getLoc(),
                                           type.getShape(),
                                           type.getElementType()));
        }
      });
    if (gemm)
      bodies->walk([&](linalg::LinalgOp op) {
        if (!op.hasPureTensorSemantics() || !matchGemm(op))
          return;
        OpOperand *init = op.getDpsInitOperand(0);
        auto fill = init->get().getDefiningOp<linalg::FillOp>();
        if (!fill || !matchPattern(fill.getInputs()[0], m_AnyZeroFloat()))
          return;
        OpBuilder builder(op);
        auto type = cast<RankedTensorType>(init->get().getType());
        Value empty = tensor::EmptyOp::create(builder, op.getLoc(),
                                              type.getShape(),
                                              type.getElementType());
        init->set(linalg::FillOp::create(builder, op.getLoc(),
                                         fill.getInputs()[0], empty)
                      .getResult(0));
      });
    // The fills and empty tensors that the above left without a use, which
    // the bufferization would still allocate and fill.
    SmallVector<Operation *> dead;
    bodies->walk([&](linalg::FillOp op) {
      if (op->use_empty())
        dead.push_back(op);
    });
    for (Operation *op : dead)
      op->erase();
    dead.clear();
    bodies->walk([&](tensor::EmptyOp op) {
      if (op->use_empty())
        dead.push_back(op);
    });
    for (Operation *op : dead)
      op->erase();

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
    if (device && fuse && shortReductions)
      for (func::FuncOp function : bodies->getOps<func::FuncOp>())
        writeDuplicatedResultsInPlace(function);
    // The GEMMs as calls of cuBLAS (D302).
    if (gemm)
      for (func::FuncOp function :
           llvm::to_vector(bodies->getOps<func::FuncOp>()))
        if (!function.isExternal() && failed(callGemms(function, *bodies)))
          return signalPassFailure();
    // The sums of rows as a warp a row (D303), or a register-held
    // accumulator per result when the sum is short.
    if (device)
      for (func::FuncOp function : bodies->getOps<func::FuncOp>())
        if (!function.isExternal()) {
          warpRowSums(function);
          if (shortReductions)
            shortRowSums(function, fuse);
        }
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
      // The runtime's functions that the bodies call.
      if (function.isExternal())
        continue;
      if (device) {
        SmallVector<scf::ParallelOp> loops;
        function.walk([&](scf::ParallelOp loop) {
          if (!loop->getParentOfType<scf::ParallelOp>())
            loops.push_back(loop);
        });
        for (scf::ParallelOp loop : loops)
          if (failed(launchLoop(loop, blockSize)))
            return signalPassFailure();
        function.walk([](gpu::LaunchOp launch) { readGlobalsInside(launch); });
        if (failed(moveToDevice(function)))
          return signalPassFailure();
      } else {
        releaseOnHost(function);
      }
      function->setAttr("md.stage_body", UnitAttr::get(context));
      function->removeAttr("md.stage_disjoint_results");
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
