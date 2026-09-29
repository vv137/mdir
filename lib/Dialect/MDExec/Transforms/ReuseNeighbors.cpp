// Reuse of neighbor structures across the iterations of a loop.

#include "mdir/Dialect/MDExec/Transforms/Passes.h"

#include "mdir/Dialect/MDExec/MDExecDialect.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mdir/Dialect/MDRT/MDRTDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::md_exec;

/// Makes the neighbor structures that are built in the body of `loop`
/// values that the loop carries. Returns the loop that replaces `loop`, or
/// `loop` itself if its body builds no neighbor structure.
static scf::ForOp reuseNeighbors(scf::ForOp loop) {
  Block &body = *loop.getBody();
  SmallVector<BuildNeighborsOp> builds;
  for (Operation &op : body)
    if (auto build = dyn_cast<BuildNeighborsOp>(&op))
      if (build.getCells().getDefiningOp<BuildCellsOp>())
        builds.push_back(build);
  if (builds.empty())
    return loop;

  Location loc = loop.getLoc();
  OpBuilder builder(loop);

  // An empty structure for each build, and the loop with those as
  // additional loop-carried values.
  SmallVector<Value> inits(loop.getInitArgs().begin(),
                           loop.getInitArgs().end());
  for (BuildNeighborsOp build : builds)
    inits.push_back(EmptyNeighborsOp::create(
        builder, build.getLoc(), build.getResult().getType(), /*size=*/Value(),
        /*element=*/TypeAttr(), build.getKindAttr(), build.getWidthAttr()));

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
  for (BuildNeighborsOp build : builds) {
    auto cells = build.getCells().getDefiningOp<BuildCellsOp>();
    Value carried =
        moved.addArgument(build.getResult().getType(), build.getLoc());

    OpBuilder inner(build);
    Value refreshed = RefreshNeighborsOp::create(
        inner, build.getLoc(), carried.getType(), carried,
        build.getPositions(), build.getCell(), build.getCutoffAttr(),
        build.getSkinAttr(), cells.getWidthAttr());
    build.getResult().replaceAllUsesWith(refreshed);
    build.erase();
    if (cells.use_empty())
      cells.erase();
    yielded.push_back(refreshed);
  }
  moved.getTerminator()->setOperands(yielded);

  for (unsigned i = 0, e = loop.getNumResults(); i != e; ++i)
    loop.getResult(i).replaceAllUsesWith(replacement.getResult(i));
  loop.erase();
  return replacement;
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
    SmallVector<scf::ForOp> loops;
    getOperation()->walk([&](scf::ForOp loop) { loops.push_back(loop); });
    for (scf::ForOp loop : loops)
      reuseNeighbors(loop);
  }
};
} // namespace

} // namespace md_exec
} // namespace mdir
