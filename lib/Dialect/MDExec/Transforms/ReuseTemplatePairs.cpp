// Persistent state for the template route only. Ordinary neighbor reuse
// retains its own classification, lifetime and refresh rules.
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mdir/Dialect/MDRT/MDRTOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include <algorithm>
#include <optional>

using namespace mlir;
using namespace mdir;
using namespace mdir::md_exec;

namespace {
struct Chain {
  EmptySegmentedPairsOp empty;
  Value last;
};

static bool sameMembership(EmptySegmentedPairsOp a, EmptySegmentedPairsOp b) {
  return a.getResult().getType() == b.getResult().getType() &&
         llvm::equal(a->getOperands(), b->getOperands()) &&
         a.getCutoffAttr() == b.getCutoffAttr() &&
         a.getSkinAttr() == b.getSkinAttr() &&
         a.getExtentAttr() == b.getExtentAttr() &&
         a.getAlphaAttr() == b.getAlphaAttr() &&
         a.getExcluded() == b.getExcluded() && a.getAnchored() == b.getAnchored() &&
         a.getRadiusAttr() == b.getRadiusAttr();
}

// Each refresh consumes the last state of its allocation. A nested loop's
// corresponding result continues its initial state. A fresh run starts a
// new chain even when its membership equals the preceding evaluation's.
static void link(Block &block) {
  SmallVector<Chain> chains;
  DenseMap<Value, unsigned> owners;
  for (Operation &operation : llvm::make_early_inc_range(block)) {
    if (auto empty = dyn_cast<EmptySegmentedPairsOp>(&operation)) {
      std::optional<unsigned> found;
      if (!empty->hasAttr(mdrt::getFreshAttrName()))
        for (unsigned i = chains.size(); i != 0; --i)
          if (sameMembership(chains[i - 1].empty, empty)) {
            found = i - 1;
            break;
          }
      if (found) {
        empty.getResult().replaceAllUsesWith(chains[*found].last);
        empty.erase();
      } else {
        owners[empty.getResult()] = chains.size();
        chains.push_back({empty, empty.getResult()});
      }
    } else if (auto refresh = dyn_cast<RefreshSegmentedPairsOp>(&operation)) {
      auto owner = owners.find(refresh.getList());
      if (owner == owners.end()) continue;
      unsigned index = owner->second;
      refresh.getListMutable().set(chains[index].last);
      owners[refresh.getResult()] = index;
      chains[index].last = refresh.getResult();
    } else if (auto loop = dyn_cast<scf::ForOp>(&operation)) {
      for (auto [i, init] : llvm::enumerate(loop.getInitArgs())) {
        auto owner = owners.find(init);
        if (owner == owners.end()) continue;
        unsigned index = owner->second;
        loop.getInitArgsMutable()[i].set(chains[index].last);
        owners[loop.getResult(i)] = index;
        chains[index].last = loop.getResult(i);
      }
    }
  }
}

static bool outside(Value value, scf::ForOp loop) {
  Operation *owner = value.getDefiningOp();
  if (!owner) owner = cast<BlockArgument>(value).getOwner()->getParentOp();
  return !loop->isAncestor(owner);
}

static bool hoist(scf::ForOp loop) {
  if (loop->hasAttr(mdrt::getSegmentAttrName())) return false;
  SmallVector<EmptySegmentedPairsOp> empties;
  for (Operation &op : *loop.getBody())
    if (auto empty = dyn_cast<EmptySegmentedPairsOp>(&op))
      if (!empty->hasAttr(mdrt::getFreshAttrName()) &&
          llvm::all_of(empty->getOperands(), [&](Value value) {
            return outside(value, loop);
          }))
        empties.push_back(empty);
  if (empties.empty()) return false;
  OpBuilder builder(loop);
  SmallVector<Value> inits(loop.getInitArgs());
  for (EmptySegmentedPairsOp empty : empties) {
    Operation *initial = builder.clone(*empty);
    if (loop->hasAttr(mdrt::getFreshAttrName()))
      initial->setAttr(mdrt::getFreshAttrName(), builder.getUnitAttr());
    inits.push_back(initial->getResult(0));
  }
  OperationState state(loop.getLoc(), scf::ForOp::getOperationName());
  state.addOperands({loop.getLowerBound(), loop.getUpperBound(), loop.getStep()});
  state.addOperands(inits);
  for (Value init : inits) state.addTypes(init.getType());
  state.addAttributes(loop->getAttrs());
  state.addRegion();
  auto replacement = cast<scf::ForOp>(builder.create(state));
  replacement.getRegion().takeBody(loop.getRegion());
  Block &body = *replacement.getBody();
  SmallVector<Value> yielded(body.getTerminator()->getOperands());
  for (EmptySegmentedPairsOp empty : empties) {
    Value current = body.addArgument(empty.getResult().getType(), loop.getLoc());
    empty.getResult().replaceAllUsesWith(current);
    empty.erase();
    for (Operation &op : body) {
      if (auto refresh = dyn_cast<RefreshSegmentedPairsOp>(&op)) {
        if (refresh.getList() == current) current = refresh.getResult();
      } else if (auto nested = dyn_cast<scf::ForOp>(&op)) {
        for (auto [i, init] : llvm::enumerate(nested.getInitArgs()))
          if (init == current) current = nested.getResult(i);
      }
    }
    yielded.push_back(current);
  }
  body.getTerminator()->setOperands(yielded);
  for (unsigned i = 0; i != loop.getNumResults(); ++i)
    loop.getResult(i).replaceAllUsesWith(replacement.getResult(i));
  loop.erase();
  return true;
}
} // namespace

namespace mdir::md_exec {
void reuseTemplatePairs(Operation *root, double pruneSkin) {
  SmallVector<BuildSegmentedPairsOp> builds;
  root->walk([&](BuildSegmentedPairsOp build) { builds.push_back(build); });
  for (BuildSegmentedPairsOp build : builds) {
    OpBuilder builder(build);
    auto empty = EmptySegmentedPairsOp::create(
        builder, build.getLoc(), build.getResult().getType(),
        build.getUniverse(), build.getMolecules(), build.getExcluded(),
        build.getAnchored(), build.getCutoffAttr(), build.getSkinAttr(),
        build.getExtentAttr(), build.getAlphaAttr(), build.getRadiusAttr());
    double margin = build.getSkin().convertToDouble();
    if (pruneSkin > 0.0) margin = std::min(margin, pruneSkin);
    auto refresh = RefreshSegmentedPairsOp::create(
        builder, build.getLoc(), build.getResult().getType(), empty,
        build.getPositions(), build.getCell(), build.getCutoffAttr(),
        build.getSkinAttr(), builder.getF64FloatAttr(margin),
        build.getExtentAttr(), build.getAlphaAttr());
    build.getResult().replaceAllUsesWith(refresh);
    build.erase();
  }
  // Postorder lets a list move through one enclosing loop at a time.
  bool changed;
  do {
    changed = false;
    SmallVector<Block *> blocks;
    root->walk([&](Operation *op) {
      for (Region &region : op->getRegions())
        for (Block &block : region) blocks.push_back(&block);
    });
    for (Block *block : blocks) link(*block);
    SmallVector<scf::ForOp> loops;
    root->walk<WalkOrder::PostOrder>([&](scf::ForOp loop) { loops.push_back(loop); });
    for (scf::ForOp loop : loops)
      if (hoist(loop)) {
        changed = true;
        break;
      }
  } while (changed);
}
} // namespace mdir::md_exec
