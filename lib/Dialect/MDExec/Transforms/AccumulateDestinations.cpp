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

namespace {
/// A chain of sums ((a0 + a1) + a2) + ... in the kernel of a loop over
/// particles, each sum used by the next alone: the arguments of the kernel
/// that it adds, in order, and the sums, `sums[k]` adding argument k + 1.
struct Chain {
  SmallVector<unsigned> arguments;
  SmallVector<arith::AddFOp> sums;
};
} // namespace

/// The chains of sums of the kernel of `op`, from the first.
static SmallVector<Chain> getChains(ParticleForOp op) {
  Block &kernel = op.getKernel().front();
  auto argumentIndex = [&](Value value) -> int {
    auto argument = dyn_cast<BlockArgument>(value);
    return argument && argument.getOwner() == &kernel
               ? static_cast<int>(argument.getArgNumber())
               : -1;
  };
  SmallVector<Chain> chains;
  for (Operation &start : kernel.without_terminator()) {
    auto first = dyn_cast<arith::AddFOp>(start);
    if (!first || argumentIndex(first.getLhs()) < 0 ||
        argumentIndex(first.getRhs()) < 0)
      continue;
    Chain chain;
    chain.arguments = {static_cast<unsigned>(argumentIndex(first.getLhs())),
                       static_cast<unsigned>(argumentIndex(first.getRhs()))};
    chain.sums = {first};
    while (chain.sums.back()->hasOneUse()) {
      auto next =
          dyn_cast<arith::AddFOp>(*chain.sums.back()->getUsers().begin());
      if (!next || next.getLhs() != chain.sums.back().getResult() ||
          argumentIndex(next.getRhs()) < 0)
        break;
      chain.arguments.push_back(argumentIndex(next.getRhs()));
      chain.sums.push_back(next);
    }
    chains.push_back(std::move(chain));
  }
  return chains;
}

/// Replaces `loop` with a loop whose destination `into` holds the sum of
/// its destinations `into` and `from`, and that has no destination `from`:
/// the kernel yields the sum of the two contributions of each member.
/// `from` must have no use.
static void mergeOuts(TupleForOp loop, unsigned into, unsigned from) {
  OpBuilder builder(loop);
  Location loc = loop.getLoc();
  int64_t arity = loop.getArity();
  unsigned numOuts = loop.getOuts().size();

  SmallVector<Value> outs(loop.getOuts().begin(), loop.getOuts().end());
  Value dropped = outs[from];
  outs.erase(outs.begin() + from);
  SmallVector<Type> resultTypes;
  for (Value value : llvm::concat<Value>(outs, loop.getReduce()))
    resultTypes.push_back(value.getType());
  auto merged = TupleForOp::create(
      builder, loc, resultTypes, loop.getIncidence(), loop.getPositions(),
      loop.getCell(), loop.getIns(), loop.getParameters(), outs,
      loop.getReduce(), /*scratch=*/ValueRange(), loop.getCoordinateKindsAttr(),
      loop.getCoordinateMembersAttr(), loop.getArityAttr(),
      /*overwrite=*/DenseBoolArrayAttr(), loop.getDisjointAttr());
  merged.getKernel().takeBody(loop.getKernel());

  // Each member takes the sum of its two contributions, in the destination
  // `into`.
  auto yield = cast<YieldOp>(merged.getKernel().front().getTerminator());
  OpBuilder kernel(yield);
  SmallVector<Value> values(yield->getOperands());
  SmallVector<Value> yielded;
  for (unsigned out = 0; out != numOuts; ++out) {
    if (out == from)
      continue;
    for (int64_t member = 0; member != arity; ++member) {
      Value value = values[out * arity + member];
      if (out == into)
        value = arith::AddFOp::create(kernel, loc, value,
                                      values[from * arity + member]);
      yielded.push_back(value);
    }
  }
  yielded.append(values.begin() + numOuts * arity, values.end());
  yield->setOperands(yielded);

  for (unsigned i = 0, e = loop->getNumResults(); i != e; ++i) {
    if (i == from)
      continue;
    loop->getResult(i).replaceAllUsesWith(
        merged->getResult(i < from ? i : i - 1));
  }
  loop->erase();
  if (Operation *zeros = dropped.getDefiningOp(); zeros && zeros->use_empty())
    zeros->erase();
}

/// Merges two destinations of one loop over tuples that a chain of sums of
/// `op` adds one after the other, as the forces of the sums over the
/// centers of groups that one loop gives (D139): the loop gives their sum
/// in one destination, and the chain adds it once. A loop with one
/// destination less writes one field less, and may then accumulate onto
/// another (accumulateOnce). Returns true if it merged two.
static bool mergeOnce(ParticleForOp op) {
  Block &kernel = op.getKernel().front();
  for (Chain &chain : getChains(op)) {
    for (unsigned k = 1, e = chain.arguments.size(); k != e; ++k) {
      unsigned before = chain.arguments[k - 1], after = chain.arguments[k];
      if (!kernel.getArgument(before).hasOneUse() ||
          !kernel.getArgument(after).hasOneUse())
        continue;
      std::optional<Term> a = getTerm(op.getIns()[before], op);
      std::optional<Term> b = getTerm(op.getIns()[after], op);
      if (!a || !b || a->loop != b->loop || !isa<TupleForOp>(a->loop))
        continue;
      // The chain adds argument `after` no longer.
      arith::AddFOp sum = chain.sums[k - 1];
      sum.getResult().replaceAllUsesWith(sum.getLhs());
      sum->erase();
      op.getInsMutable().erase(after);
      kernel.eraseArgument(after);
      mergeOuts(cast<TupleForOp>(a->loop), a->out, b->out);
      return true;
    }
  }
  return false;
}

/// Rewrites the first chain of sums in the kernel of `op` that it can.
/// Returns true if it did.
static bool accumulateOnce(ParticleForOp op) {
  Block &kernel = op.getKernel().front();
  for (Chain &chain : getChains(op)) {
    ArrayRef<unsigned> arguments = chain.arguments;
    ArrayRef<arith::AddFOp> sums = chain.sums;
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
    for (ParticleForOp op : loops) {
      while (mergeOnce(op)) {
      }
      while (accumulateOnce(op)) {
      }
    }
  }
};
} // namespace

} // namespace md_exec
} // namespace mdir
