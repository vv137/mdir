// Accumulation of the destinations of loops that a loop over particles adds.
//
// See the description of the pass in Passes.td.

#include "mdir/Dialect/MDExec/Transforms/Passes.h"

#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::md_exec;

namespace {
/// A field that a loop over pairs or tuples gives, accumulated from zero.
struct Term {
  Operation *loop = nullptr;
  unsigned out = 0;
};
} // namespace

/// Returns the loop and destination that give `field`, if a loop over pairs
/// or tuples in the value form gives it from a field of zeros, and nothing
/// else uses it than `user`.
static std::optional<Term> getTerm(Value field, Operation *user) {
  auto result = dyn_cast<OpResult>(field);
  if (!result || !result.hasOneUse() || *result.getUsers().begin() != user)
    return std::nullopt;
  Operation *loop = result.getOwner();
  ValueRange outs;
  if (auto pair = dyn_cast<PairForOp>(loop)) {
    if (pair.isStorageForm())
      return std::nullopt;
    outs = pair.getOuts();
  } else if (auto tuple = dyn_cast<TupleForOp>(loop)) {
    if (tuple.isStorageForm())
      return std::nullopt;
    outs = tuple.getOuts();
  } else {
    return std::nullopt;
  }
  unsigned index = result.getResultNumber();
  if (index >= outs.size() || !outs[index].getDefiningOp<ZerosOp>() ||
      loop->getBlock() != user->getBlock())
    return std::nullopt;
  return Term{loop, index};
}

/// The destinations of `loop`, which can be set.
static MutableOperandRange getOutsMutable(Operation *loop) {
  if (auto pair = dyn_cast<PairForOp>(loop))
    return pair.getOutsMutable();
  return cast<TupleForOp>(loop).getOutsMutable();
}

/// Rewrites the first chain of sums in the kernel of `op` that it can.
/// Returns true if it did.
static bool accumulateOnce(ParticleForOp op) {
  Block &kernel = op.getKernel().front();
  auto argumentIndex = [&](Value value) -> int {
    auto argument = dyn_cast<BlockArgument>(value);
    return argument && argument.getOwner() == &kernel
               ? static_cast<int>(argument.getArgNumber())
               : -1;
  };
  for (Operation &start : kernel.without_terminator()) {
    auto first = dyn_cast<arith::AddFOp>(start);
    if (!first || argumentIndex(first.getLhs()) < 0 ||
        argumentIndex(first.getRhs()) < 0)
      continue;
    // The chain ((a0 + a1) + a2) + ..., each sum used by the next alone.
    SmallVector<unsigned> arguments = {
        static_cast<unsigned>(argumentIndex(first.getLhs())),
        static_cast<unsigned>(argumentIndex(first.getRhs()))};
    SmallVector<arith::AddFOp> sums = {first};
    while (sums.back()->hasOneUse()) {
      auto next = dyn_cast<arith::AddFOp>(*sums.back()->getUsers().begin());
      if (!next || next.getLhs() != sums.back().getResult() ||
          argumentIndex(next.getRhs()) < 0)
        break;
      arguments.push_back(argumentIndex(next.getRhs()));
      sums.push_back(next);
    }
    // The prefix whose fields are terms of loops in the order of the chain.
    SmallVector<Term> terms;
    for (unsigned index : arguments) {
      if (!kernel.getArgument(index).hasOneUse())
        break;
      std::optional<Term> term = getTerm(op.getIns()[index], op);
      if (!term || (!terms.empty() &&
                    !terms.back().loop->isBeforeInBlock(term->loop)))
        break;
      terms.push_back(*term);
    }
    if (terms.size() < 2)
      continue;

    // Each loop accumulates onto what the loop before it gave.
    for (unsigned k = 1; k != terms.size(); ++k) {
      Value previous = terms[k - 1].loop->getResult(terms[k - 1].out);
      MutableOperandRange outs = getOutsMutable(terms[k].loop);
      Operation *zeros = outs[terms[k].out].get().getDefiningOp();
      outs[terms[k].out].set(previous);
      if (zeros->use_empty())
        zeros->erase();
    }
    // The loop over particles reads the last, in place of the sums.
    unsigned kept = arguments[0];
    Value last = terms.back().loop->getResult(terms.back().out);
    arith::AddFOp through = sums[terms.size() - 2];
    through.getResult().replaceAllUsesWith(kernel.getArgument(kept));
    for (int k = static_cast<int>(terms.size()) - 2; k >= 0; --k)
      sums[k]->erase();
    op.getInsMutable()[kept].set(last);
    SmallVector<unsigned> dropped(arguments.begin() + 1,
                                  arguments.begin() + terms.size());
    llvm::sort(dropped, std::greater<unsigned>());
    for (unsigned index : dropped) {
      op.getInsMutable().erase(index);
      kernel.eraseArgument(index);
    }
    return true;
  }
  return false;
}

namespace mdir {
namespace md_exec {

#define GEN_PASS_DEF_ACCUMULATEDESTINATIONS
#include "mdir/Dialect/MDExec/Transforms/Passes.h.inc"

namespace {
class AccumulateDestinations
    : public impl::AccumulateDestinationsBase<AccumulateDestinations> {
public:
  using impl::AccumulateDestinationsBase<
      AccumulateDestinations>::AccumulateDestinationsBase;

  void runOnOperation() final {
    SmallVector<ParticleForOp> loops;
    getOperation()->walk([&](ParticleForOp op) {
      if (!op.isStorageForm())
        loops.push_back(op);
    });
    for (ParticleForOp op : loops)
      while (accumulateOnce(op)) {
      }
  }
};
} // namespace

} // namespace md_exec
} // namespace mdir
