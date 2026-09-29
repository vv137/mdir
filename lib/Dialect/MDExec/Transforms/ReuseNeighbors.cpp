// Reuse of neighbor structures across the iterations of a loop.

#include "mdir/Dialect/MDExec/Transforms/Passes.h"

#include "mdir/Dialect/MDExec/MDExecDialect.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mdir/Dialect/MDRT/MDRTDialect.h"
#include "mdir/Dialect/MDRT/MDRTOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::md_exec;

namespace {

/// A neighbor structure that the body of a loop uses and that the loop can
/// carry.
struct Candidate {
  /// A structure that the body builds, or null.
  BuildNeighborsOp build;
  /// An empty structure that a loop in the body starts from, or null.
  EmptyNeighborsOp empty;
  /// The loop in the body that carries `empty`, and the position of the
  /// structure among the values it carries.
  scf::ForOp inner;
  unsigned position = 0;
};

} // namespace

/// Returns true if the iterations of `loop` are segments of a run. A
/// neighbor structure starts empty in every segment, so that a run that is
/// restarted from a checkpoint builds where the first run did.
static bool isSegmentLoop(scf::ForOp loop) {
  return loop->hasAttr(mdrt::getSegmentAttrName());
}

/// Finds the neighbor structures that `loop` can carry.
static SmallVector<Candidate> findCandidates(scf::ForOp loop) {
  SmallVector<Candidate> candidates;
  for (Operation &op : *loop.getBody()) {
    if (auto build = dyn_cast<BuildNeighborsOp>(&op)) {
      if (build.getCells().getDefiningOp<BuildCellsOp>()) {
        Candidate candidate;
        candidate.build = build;
        candidates.push_back(candidate);
      }
      continue;
    }

    // An empty structure that a loop in the body carries, and nothing else
    // uses: the structure can live on from one iteration to the next.
    auto empty = dyn_cast<EmptyNeighborsOp>(&op);
    if (!empty || empty.isStorageForm() || !empty.getResult().hasOneUse())
      continue;
    OpOperand &use = *empty.getResult().use_begin();
    auto inner = dyn_cast<scf::ForOp>(use.getOwner());
    if (!inner || inner->getBlock() != loop.getBody())
      continue;
    for (auto [index, init] : llvm::enumerate(inner.getInitArgs())) {
      if (init != empty.getResult())
        continue;
      Candidate candidate;
      candidate.empty = empty;
      candidate.inner = inner;
      candidate.position = index;
      candidates.push_back(candidate);
    }
  }
  return candidates;
}

/// Makes the neighbor structures that the body of `loop` uses values that
/// the loop carries.
static void reuseNeighbors(scf::ForOp loop) {
  if (isSegmentLoop(loop))
    return;
  SmallVector<Candidate> candidates = findCandidates(loop);
  if (candidates.empty())
    return;

  Location loc = loop.getLoc();
  OpBuilder builder(loop);

  // An empty structure for each candidate, and the loop with those as
  // additional loop-carried values.
  SmallVector<Value> inits(loop.getInitArgs().begin(),
                           loop.getInitArgs().end());
  for (Candidate candidate : candidates) {
    if (candidate.build)
      inits.push_back(EmptyNeighborsOp::create(
          builder, candidate.build.getLoc(),
          candidate.build.getResult().getType(), /*size=*/Value(),
          /*positions=*/TypeAttr(), candidate.build.getKindAttr(),
          candidate.build.getWidthAttr()));
    else
      inits.push_back(builder.clone(*candidate.empty)->getResult(0));
  }

  OperationState state(loc, scf::ForOp::getOperationName());
  state.addOperands(
      {loop.getLowerBound(), loop.getUpperBound(), loop.getStep()});
  state.addOperands(inits);
  for (Value init : inits)
    state.addTypes(init.getType());
  state.addAttributes(loop->getAttrs());
  state.addRegion();
  auto replacement = cast<scf::ForOp>(builder.create(state));
  replacement.getRegion().takeBody(loop.getRegion());

  Block &moved = *replacement.getBody();
  SmallVector<Value> yielded(moved.getTerminator()->getOperands().begin(),
                             moved.getTerminator()->getOperands().end());
  for (Candidate candidate : candidates) {
    if (candidate.build) {
      // The build becomes a refresh of the structure that the loop carries.
      BuildNeighborsOp build = candidate.build;
      auto cells = build.getCells().getDefiningOp<BuildCellsOp>();
      Value carried =
          moved.addArgument(build.getResult().getType(), build.getLoc());

      OpBuilder inner(build);
      Value refreshed = RefreshNeighborsOp::create(
          inner, build.getLoc(), carried.getType(), carried,
          build.getPositions(), build.getCell(), /*scratch=*/ValueRange(),
          build.getCutoffAttr(), build.getSkinAttr(), cells.getWidthAttr());
      build.getResult().replaceAllUsesWith(refreshed);
      build.erase();
      if (cells.use_empty())
        cells.erase();
      yielded.push_back(refreshed);
      continue;
    }

    // The loop in the body starts from the structure that this loop
    // carries, and hands it back.
    EmptyNeighborsOp empty = candidate.empty;
    Value carried =
        moved.addArgument(empty.getResult().getType(), empty.getLoc());
    empty.getResult().replaceAllUsesWith(carried);
    empty.erase();
    yielded.push_back(candidate.inner.getResult(candidate.position));
  }
  moved.getTerminator()->setOperands(yielded);

  for (unsigned i = 0, e = loop.getNumResults(); i != e; ++i)
    loop.getResult(i).replaceAllUsesWith(replacement.getResult(i));
  loop.erase();
}

namespace mdir {
namespace md_exec {

#define GEN_PASS_DEF_REUSENEIGHBORS
#include "mdir/Dialect/MDExec/Transforms/Passes.h.inc"

namespace {
class ReuseNeighbors : public impl::ReuseNeighborsBase<ReuseNeighbors> {
public:
  using impl::ReuseNeighborsBase<ReuseNeighbors>::ReuseNeighborsBase;

  void runOnOperation() final {
    // Loops inside other loops come first, so that a structure moves
    // outward one loop at a time. A loop is replaced when it takes a
    // structure, so the loops are looked up again after every change.
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<scf::ForOp> loops;
      getOperation()->walk([&](scf::ForOp loop) { loops.push_back(loop); });
      for (scf::ForOp loop : loops) {
        if (isSegmentLoop(loop) || findCandidates(loop).empty())
          continue;
        reuseNeighbors(loop);
        changed = true;
        break;
      }
    }
  }
};
} // namespace

} // namespace md_exec
} // namespace mdir
