// Ops that run on a second stream beside those they are independent of.
//
// See the description of the pass in Passes.td, and D87 in
// docs/decisions.md for the argument that it is correct.

#include "mdir/Dialect/MDExec/Transforms/Passes.h"

#include "mdir/Dialect/MDExec/Independence.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::md_exec;

/// Moves `op` up its block past the ops that it is independent of and
/// whose results it does not take. Moving it past an independent op leaves
/// memory and every value as they were (D87).
static void hoist(Operation *op, BufferAliases &aliases) {
  Operation *before = op->getPrevNode();
  while (before && !isa<JoinOp>(before) && !before->hasAttr(kSideAttrName) &&
         areIndependent(before, op, aliases))
    before = before->getPrevNode();
  if (before)
    op->moveAfter(before);
  else
    op->moveBefore(&op->getBlock()->front());
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
  }
};
} // namespace

} // namespace md_exec
} // namespace mdir
