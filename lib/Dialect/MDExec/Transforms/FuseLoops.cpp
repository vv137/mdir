// Fusion of loops over the pairs of one neighbor structure.

#include "mdir/Dialect/MDExec/Transforms/Passes.h"

#include "mdir/Dialect/MDExec/MDExecDialect.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::md_exec;

/// Returns true if `a` and `b` run over the same pairs in the same way.
static bool haveSameDomain(PairForOp a, PairForOp b) {
  return a.getNeighbors() == b.getNeighbors() &&
         a.getPositions() == b.getPositions() && a.getCell() == b.getCell() &&
         a.getCutoffAttr() == b.getCutoffAttr() &&
         a.getTraversal() == b.getTraversal() &&
         a.getConflict() == b.getConflict();
}

/// Returns true if `op` or an op nested in it uses a result of `producer`.
static bool uses(Operation *op, Operation *producer) {
  bool found = false;
  op->walk([&](Operation *nested) {
    for (Value operand : nested->getOperands())
      found |= operand.getDefiningOp() == producer;
  });
  return found;
}

/// Returns true if every op that uses a result of `op` comes after `point`,
/// which is in the block of `op`.
static bool usersComeAfter(Operation *op, Operation *point) {
  for (Value result : op->getResults()) {
    for (Operation *user : result.getUsers()) {
      // The op of the block of `op` that holds the user.
      while (user->getBlock() != op->getBlock())
        user = user->getParentOp();
      if (!point->isBeforeInBlock(user))
        return false;
    }
  }
  return true;
}

/// The weights of the global sums of `op`, with 1 where none is given.
static void appendWeights(PairForOp op, SmallVectorImpl<double> &weights) {
  if (auto given = op.getWeights()) {
    weights.append(given->begin(), given->end());
    return;
  }
  weights.append(op.getReduce().size(), 1.0);
}

/// Replaces `first` and `second` with one loop, placed where `second` is.
static void fuse(PairForOp first, PairForOp second) {
  OpBuilder builder(second);
  Location loc = second.getLoc();

  // The fields that the kernels read. A field that both read is read once.
  SmallVector<Value> ins;
  llvm::DenseMap<Value, unsigned> position;
  auto addFields = [&](PairForOp op) {
    for (Value field : op.getIns())
      if (position.try_emplace(field, ins.size()).second)
        ins.push_back(field);
  };
  addFields(first);
  addFields(second);

  SmallVector<Value> outs(first.getOuts().begin(), first.getOuts().end());
  outs.append(second.getOuts().begin(), second.getOuts().end());
  SmallVector<Value> reduce(first.getReduce().begin(),
                            first.getReduce().end());
  reduce.append(second.getReduce().begin(), second.getReduce().end());

  SmallVector<double> weights;
  appendWeights(first, weights);
  appendWeights(second, weights);
  DenseF64ArrayAttr weightsAttr;
  if (llvm::any_of(weights, [](double weight) { return weight != 1.0; }))
    weightsAttr = builder.getDenseF64ArrayAttr(weights);

  SmallVector<Type> resultTypes;
  for (Value value : outs)
    resultTypes.push_back(value.getType());
  for (Value value : reduce)
    resultTypes.push_back(value.getType());

  auto fused = PairForOp::create(
      builder, loc, resultTypes, first.getNeighbors(), first.getPositions(),
      first.getCell(), ins, outs, reduce, /*scratch=*/ValueRange(),
      first.getCutoffAttr(), weightsAttr,
      /*overwrite=*/DenseBoolArrayAttr(), first.getTraversalAttr(),
      first.getConflictAttr());

  // The kernel: the first kernel, then the second.
  Block *block = new Block();
  fused.getKernel().push_back(block);
  Block &firstKernel = first.getKernel().front();
  for (unsigned i = 0; i != 2; ++i)
    block->addArgument(firstKernel.getArgument(i).getType(), loc);
  for (Value field : ins) {
    Type type = cast<md::FieldType>(field.getType()).getKernelValueType();
    block->addArgument(type, loc);
    block->addArgument(type, loc);
  }

  OpBuilder kernel(builder.getContext());
  kernel.setInsertionPointToEnd(block);
  SmallVector<Value> yieldedOuts, yieldedSums;
  auto addKernel = [&](PairForOp op) {
    Block &source = op.getKernel().front();
    IRMapping mapping;
    mapping.map(source.getArgument(0), block->getArgument(0));
    mapping.map(source.getArgument(1), block->getArgument(1));
    for (auto [index, field] : llvm::enumerate(op.getIns())) {
      unsigned target = 2 + 2 * position.lookup(field);
      mapping.map(source.getArgument(2 + 2 * index),
                  block->getArgument(target));
      mapping.map(source.getArgument(3 + 2 * index),
                  block->getArgument(target + 1));
    }
    for (Operation &nested : source.without_terminator())
      kernel.clone(nested, mapping);

    Operation *yield = source.getTerminator();
    unsigned numOuts = op.getOuts().size();
    for (unsigned i = 0, e = yield->getNumOperands(); i != e; ++i) {
      Value value = mapping.lookupOrDefault(yield->getOperand(i));
      (i < numOuts ? yieldedOuts : yieldedSums).push_back(value);
    }
  };
  addKernel(first);
  addKernel(second);

  SmallVector<Value> yielded(yieldedOuts);
  yielded.append(yieldedSums);
  YieldOp::create(kernel, loc, yielded);

  // Results: the destinations of the first, those of the second, the sums
  // of the first, those of the second.
  unsigned firstOuts = first.getOuts().size();
  unsigned secondOuts = second.getOuts().size();
  unsigned firstSums = first.getReduce().size();
  unsigned allOuts = firstOuts + secondOuts;
  for (unsigned i = 0, e = first.getNumResults(); i != e; ++i) {
    unsigned target = i < firstOuts ? i : allOuts + (i - firstOuts);
    first.getResult(i).replaceAllUsesWith(fused.getResult(target));
  }
  for (unsigned i = 0, e = second.getNumResults(); i != e; ++i) {
    unsigned target = i < secondOuts
                          ? firstOuts + i
                          : allOuts + firstSums + (i - secondOuts);
    second.getResult(i).replaceAllUsesWith(fused.getResult(target));
  }
  first.erase();
  second.erase();
}

/// Fuses two loops of `block`, if two can be fused. Returns true if it did.
static bool fuseOnce(Block &block) {
  SmallVector<PairForOp> loops;
  // Fusion relies on the value form: in the storage form, what a loop
  // depends on is not visible in its operands.
  for (Operation &op : block)
    if (auto loop = dyn_cast<PairForOp>(&op))
      if (!loop.isStorageForm())
        loops.push_back(loop);

  for (unsigned i = 0, e = loops.size(); i != e; ++i) {
    for (unsigned j = i + 1; j != e; ++j) {
      PairForOp first = loops[i];
      PairForOp second = loops[j];
      if (!haveSameDomain(first, second) || uses(second, first) ||
          !usersComeAfter(first, second))
        continue;
      fuse(first, second);
      return true;
    }
  }
  return false;
}

namespace mdir {
namespace md_exec {

#define GEN_PASS_DEF_FUSELOOPS
#include "mdir/Dialect/MDExec/Transforms/Passes.h.inc"

namespace {
class FuseLoops : public impl::FuseLoopsBase<FuseLoops> {
public:
  using impl::FuseLoopsBase<FuseLoops>::FuseLoopsBase;

  void runOnOperation() final {
    SmallVector<Block *> blocks;
    getOperation()->walk([&](Operation *op) {
      for (Region &region : op->getRegions())
        for (Block &block : region)
          blocks.push_back(&block);
    });
    for (Block *block : blocks)
      while (fuseOnce(*block))
        ;
  }
};
} // namespace

} // namespace md_exec
} // namespace mdir
