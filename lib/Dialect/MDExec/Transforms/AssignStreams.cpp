// Ops that run on a second stream beside those they are independent of.
//
// See the description of the pass in Passes.td, and D87 and
// D[concurrent-terms] in docs/decisions for the argument that it is
// correct.

#include "mdir/Dialect/MDExec/Transforms/Passes.h"

#include "mdir/Dialect/MDExec/Independence.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::md_exec;

/// The op before which `op` would stand if it moved up its block past the
/// ops that it is independent of and whose results it does not take, or
/// null for the front of the block; `blocking` is the op that stops it.
static Operation *findPlace(Operation *op, BufferAliases &aliases,
                            Operation *&blocking) {
  Operation *before = op->getPrevNode();
  while (before && !isa<JoinOp>(before) && !before->hasAttr(kSideAttrName) &&
         areIndependent(before, op, aliases))
    before = before->getPrevNode();
  blocking = before;
  return before ? before->getNextNode() : &op->getBlock()->front();
}

/// Moves `op` up its block past the ops that it is independent of and
/// whose results it does not take. Moving it past an independent op leaves
/// memory and every value as they were (D87).
static void hoist(Operation *op, BufferAliases &aliases) {
  Operation *blocking;
  Operation *place = findPlace(op, aliases, blocking);
  if (place != op)
    op->moveBefore(place);
}

namespace {
/// Where an op stood, to put it back.
struct Place {
  Operation *op;
  Operation *next;
};
} // namespace

/// Moves the loop over tuples `loop` up as `hoist` does, and first the
/// loops over particles that stop it, which write what it reads (the
/// positions in f32 that the tuples take), each as far as it may go
/// itself: moving an op up past ops it is independent of leaves the
/// program's results as they were, whichever op it is (D87). Records each
/// move in `moves`.
static void hoistWithProducers(Operation *loop, BufferAliases &aliases,
                               SmallVectorImpl<Place> &moves) {
  for (int attempt = 0; attempt != 4; ++attempt) {
    Operation *blocking;
    Operation *place = findPlace(loop, aliases, blocking);
    if (place != loop) {
      moves.push_back({loop, loop->getNextNode()});
      loop->moveBefore(place);
    }
    if (!blocking || !isa<ParticleForOp>(blocking) ||
        blocking->hasAttr(kSideAttrName))
      return;
    Operation *stop;
    Operation *above = findPlace(blocking, aliases, stop);
    if (above == blocking)
      return;
    moves.push_back({blocking, blocking->getNextNode()});
    blocking->moveBefore(above);
  }
}

namespace mdir {
namespace md_exec {

#define GEN_PASS_DEF_ASSIGNSTREAMS
#include "mdir/Dialect/MDExec/Transforms/Passes.h.inc"

namespace {
class AssignStreams : public impl::AssignStreamsBase<AssignStreams> {
public:
  using impl::AssignStreamsBase<AssignStreams>::AssignStreamsBase;

  void runOnOperation() final {
    SmallVector<Operation *> marked;
    getOperation()->walk([&](ReciprocalOp op) {
      if (reciprocal && op.isStorageForm())
        marked.push_back(op);
    });
    // All are marked first, so that no window takes in another.
    for (Operation *op : marked)
      op->setAttr(kSideAttrName, UnitAttr::get(op->getContext()));
    BufferAliases aliases;
    for (Operation *op : marked) {
      hoist(op, aliases);
      OpBuilder builder(findJoin(op, aliases));
      JoinOp::create(builder, op->getLoc());
    }

    // A loop over tuples goes to the second stream if it can move up past
    // a loop over pairs before it in its block: then it runs beside the
    // pairs (D[concurrent-terms]). The loops of one chain of tuples follow
    // each other there, in their order, and after the reciprocal sum, on
    // the same stream.
    if (tuples) {
      SmallVector<TupleForOp> loops;
      // A loop that reduces values, which the host may read, stays (the
      // energies of a step that writes them).
      getOperation()->walk([&](TupleForOp loop) {
        if (loop.isStorageForm() && loop.getReduce().empty())
          loops.push_back(loop);
      });
      SmallVector<Operation *> tupleLoops;
      for (TupleForOp loop : loops) {
        PairForOp pair;
        for (Operation *before = loop->getPrevNode(); before && !pair;
             before = before->getPrevNode())
          pair = dyn_cast<PairForOp>(before);
        if (!pair)
          continue;
        SmallVector<Place> moves;
        hoistWithProducers(loop, aliases, moves);
        if (loop->isBeforeInBlock(pair)) {
          // A stream of their own, beside the reciprocal sums.
          loop->setAttr(kSideAttrName, UnitAttr::get(loop->getContext()));
          loop->setAttr(kLaneAttrName,
                        IntegerAttr::get(IntegerType::get(loop->getContext(), 64), 2));
          tupleLoops.push_back(loop);
          continue;
        }
        // Nothing to run beside: every op goes back where it stood.
        for (const Place &move : llvm::reverse(moves)) {
          if (move.next)
            move.op->moveBefore(move.next);
          else
            move.op->moveBefore(move.op->getBlock(),
                                move.op->getBlock()->end());
        }
      }

      // The first stream waits before the first op that is not
      // independent of a loop, unless a join stands there already.
      for (Operation *op : tupleLoops) {
        Operation *join = findJoin(op, aliases);
        if (isa<JoinOp>(join))
          continue;
        OpBuilder builder(join);
        JoinOp::create(builder, op->getLoc());
      }
    }

    if (forkAfterPairs)
      issueAfterPairs();
  }

  /// The host issues the work of a step in the order of the program, and a
  /// launch takes it microseconds: marked ops at the top of a window delay
  /// the loops over pairs, the longest chain, by their launches. So each run
  /// of marked ops of one side stream before a loop over pairs of its
  /// window moves down to the join, and a fork where it stood lets it start
  /// from there still (md_exec.fork, D[concurrent-terms]). The ops it moves
  /// past are those of its window, independent of it. At the join, it
  /// leaves the loops of the first stream after the pairs next to them
  /// (the bonded publication of TOWERS, D436).
  void issueAfterPairs() {
    SmallVector<Block *> blocks;
    getOperation()->walk([&](Block *block) { blocks.push_back(block); });
    for (Block *block : blocks) {
      for (unsigned lane : {1u, 2u}) {
        Operation *op = block->empty() ? nullptr : &block->front();
        while (op) {
          if (getSideLane(op) != lane) {
            op = op->getNextNode();
            continue;
          }
          // The run of marked ops of the lane from `op`, up to the join,
          // and the last loop over pairs before the join.
          SmallVector<Operation *> run;
          Operation *pair = nullptr, *end = op;
          for (; end && !isa<JoinOp>(end) &&
                 !end->hasTrait<OpTrait::IsTerminator>();
               end = end->getNextNode()) {
            if (isa<PairForOp>(end))
              pair = end;
            else if (getSideLane(end) == lane && !pair)
              run.push_back(end);
          }
          if (pair && !run.empty() && end && isa<JoinOp>(end)) {
            OpBuilder builder(run.front());
            ForkOp::create(builder, run.front()->getLoc(),
                           builder.getI64IntegerAttr(lane));
            for (Operation *moved : run)
              moved->moveBefore(end);
          }
          op = end;
        }
      }
    }
  }
};
} // namespace

} // namespace md_exec
} // namespace mdir
