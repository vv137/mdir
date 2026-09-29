// Reuse of neighbor structures: across the iterations of a loop, and from
// one use to the next.

#include "mdir/Dialect/MDExec/Transforms/Passes.h"

#include "mdir/Dialect/MDExec/MDExecDialect.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mdir/Dialect/MDRT/MDRTDialect.h"
#include "mdir/Dialect/MDRT/MDRTOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::md_exec;

namespace {

/// What a neighbor structure is built with. Two structures with the same
/// parameters can share their storage if they are not in use at the same
/// time.
struct Parameters {
  bool isKnown = false;
  Type type;
  Value cell;
  /// The pairs that the structure leaves out.
  Value excluded;
  Attribute kind, width, cutoff, skin, cellWidth;

  bool operator==(const Parameters &other) const {
    return isKnown && other.isKnown && type == other.type &&
           cell == other.cell && excluded == other.excluded &&
           kind == other.kind &&
           width == other.width && cutoff == other.cutoff &&
           skin == other.skin && cellWidth == other.cellWidth;
  }
};

/// A place in a block where a neighbor structure comes into being: a build,
/// or an empty structure that a loop in the block starts from.
struct Source {
  /// The op of the block that the structure is first used by.
  Operation *anchor = nullptr;
  BuildNeighborsOp build;
  EmptyNeighborsOp empty;
  /// The loop that carries `empty`, and the position of the structure
  /// among the values it carries.
  scf::ForOp loop;
  unsigned position = 0;
  Parameters parameters;
};

/// Structures of one block that follow one another in the same storage.
struct Chain {
  Parameters parameters;
  SmallVector<Source, 2> sources;
  /// The structure as the last source leaves it.
  Value last;
};

} // namespace

/// Returns true if the iterations of `loop` are segments of a run. A
/// neighbor structure starts empty in every segment, so that a run that is
/// restarted from a checkpoint builds where the first run did.
static bool isSegmentLoop(scf::ForOp loop) {
  return loop->hasAttr(mdrt::getSegmentAttrName());
}

/// Adds to `found` the parameters of the refreshes of `structure`, which a
/// loop carries.
static void findParameters(Value structure,
                           SmallVectorImpl<Parameters> &found) {
  for (OpOperand &use : structure.getUses()) {
    Operation *user = use.getOwner();
    if (auto refresh = dyn_cast<RefreshNeighborsOp>(user)) {
      if (refresh.getNeighbors() != structure)
        continue;
      Parameters parameters;
      parameters.isKnown = true;
      parameters.type = structure.getType();
      parameters.cell = refresh.getCell();
      parameters.cutoff = refresh.getCutoffAttr();
      parameters.skin = refresh.getSkinAttr();
      parameters.cellWidth = refresh.getCellWidthAttr();
      found.push_back(parameters);
      findParameters(refresh.getResult(), found);
      continue;
    }
    if (auto loop = dyn_cast<scf::ForOp>(user))
      for (auto [init, argument] :
           llvm::zip(loop.getInitArgs(), loop.getRegionIterArgs()))
        if (init == structure)
          findParameters(argument, found);
  }
}

/// Finds the places in `block` where a neighbor structure comes into being.
static SmallVector<Source, 4> findSources(Block &block) {
  SmallVector<Source, 4> sources;
  for (Operation &op : block) {
    if (auto build = dyn_cast<BuildNeighborsOp>(&op)) {
      auto cells = build.getCells().getDefiningOp<BuildCellsOp>();
      if (!cells)
        continue;
      Source source;
      source.anchor = build;
      source.build = build;
      source.parameters.isKnown = true;
      source.parameters.type = build.getResult().getType();
      source.parameters.cell = build.getCell();
      source.parameters.excluded = build.getExcluded();
      source.parameters.kind = build.getKindAttr();
      source.parameters.width = build.getWidthAttr();
      source.parameters.cutoff = build.getCutoffAttr();
      source.parameters.skin = build.getSkinAttr();
      source.parameters.cellWidth = cells.getWidthAttr();
      sources.push_back(source);
      continue;
    }

    // An empty structure that a loop of the block carries, and nothing
    // else uses.
    auto empty = dyn_cast<EmptyNeighborsOp>(&op);
    if (!empty || empty.isStorageForm() || !empty.getResult().hasOneUse())
      continue;
    OpOperand &use = *empty.getResult().use_begin();
    auto loop = dyn_cast<scf::ForOp>(use.getOwner());
    if (!loop || loop->getBlock() != &block)
      continue;
    for (auto [index, init] : llvm::enumerate(loop.getInitArgs())) {
      if (init != empty.getResult())
        continue;
      Source source;
      source.anchor = loop;
      source.empty = empty;
      source.loop = loop;
      source.position = index;

      // The structure is built with the parameters of its refreshes, if
      // they agree.
      SmallVector<Parameters> found;
      findParameters(loop.getRegionIterArgs()[index], found);
      if (!found.empty() && llvm::all_of(found, [&](Parameters &other) {
            return other.cell == found.front().cell &&
                   other.cutoff == found.front().cutoff &&
                   other.skin == found.front().skin &&
                   other.cellWidth == found.front().cellWidth;
          })) {
        source.parameters = found.front();
        source.parameters.excluded = empty.getExcluded();
        source.parameters.kind = empty.getKindAttr();
        source.parameters.width = empty.getWidthAttr();
      }
      source.parameters.type = empty.getResult().getType();
      sources.push_back(source);
    }
  }
  return sources;
}

/// Returns true if every use of `value` comes before `point`, which is in
/// the block `block`.
static bool isDeadBefore(Value value, Operation *point, Block &block) {
  for (Operation *user : value.getUsers()) {
    while (user->getBlock() != &block)
      user = user->getParentOp();
    if (!user->isBeforeInBlock(point))
      return false;
  }
  return true;
}

/// The structure as `source` leaves it.
static Value getLast(const Source &source) {
  Source copy = source;
  if (copy.build)
    return copy.build.getResult();
  return copy.loop.getResult(copy.position);
}

/// Groups the sources of `block` into chains: a source joins the chain of
/// an earlier source with the same parameters whose structure is no longer
/// in use.
static SmallVector<Chain, 2> findChains(Block &block) {
  SmallVector<Chain, 2> chains;
  for (const Source &source : findSources(block)) {
    Chain *joined = nullptr;
    for (Chain &chain : chains)
      if (chain.parameters == source.parameters &&
          isDeadBefore(chain.last, source.anchor, block)) {
        joined = &chain;
        break;
      }
    if (!joined) {
      chains.emplace_back();
      joined = &chains.back();
      joined->parameters = source.parameters;
    }
    joined->sources.push_back(source);
    joined->last = getLast(source);
  }
  return chains;
}

/// Makes `source` continue in the structure `current` and returns the
/// structure as the source leaves it.
static Value continueIn(Source source, Value current) {
  if (source.build) {
    // The build becomes a refresh.
    BuildNeighborsOp build = source.build;
    auto cells = build.getCells().getDefiningOp<BuildCellsOp>();
    OpBuilder builder(build);
    Value refreshed = RefreshNeighborsOp::create(
        builder, build.getLoc(), current.getType(), current,
        build.getPositions(), build.getCell(), /*scratch=*/ValueRange(),
        /*moved=*/Value(), build.getCutoffAttr(), build.getSkinAttr(),
        cells.getWidthAttr());
    build.getResult().replaceAllUsesWith(refreshed);
    build.erase();
    if (cells.use_empty())
      cells.erase();
    return refreshed;
  }

  // The loop starts from the structure.
  source.empty.getResult().replaceAllUsesWith(current);
  source.empty.erase();
  return source.loop.getResult(source.position);
}

/// Makes the sources of `chain` after the first continue in the structure
/// of the first.
static void linkChain(Chain &chain) {
  Value current = getLast(chain.sources.front());
  for (const Source &source : llvm::drop_begin(chain.sources))
    current = continueIn(source, current);
}

/// Makes the neighbor structures that the body of `loop` uses values that
/// the loop carries.
static void reuseNeighbors(scf::ForOp loop) {
  SmallVector<Chain, 2> chains = findChains(*loop.getBody());

  Location loc = loop.getLoc();
  OpBuilder builder(loop);

  // An empty structure for each chain, and the loop with those as
  // additional loop-carried values.
  SmallVector<Value> inits(loop.getInitArgs().begin(),
                           loop.getInitArgs().end());
  for (Chain &chain : chains) {
    Source first = chain.sources.front();
    if (first.build)
      inits.push_back(EmptyNeighborsOp::create(
          builder, first.build.getLoc(), first.build.getResult().getType(),
          /*size=*/Value(), first.build.getExcluded(),
          /*positions=*/TypeAttr(), first.build.getKindAttr(),
          first.build.getWidthAttr()));
    else
      inits.push_back(builder.clone(*first.empty)->getResult(0));
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
  for (Chain &chain : chains) {
    Value current = moved.addArgument(chain.parameters.type, loc);
    for (const Source &source : chain.sources)
      current = continueIn(source, current);
    yielded.push_back(current);
  }
  moved.getTerminator()->setOperands(yielded);

  for (unsigned i = 0, e = loop.getNumResults(); i != e; ++i)
    loop.getResult(i).replaceAllUsesWith(replacement.getResult(i));
  loop.erase();
}

/// Returns true if the body of `loop` has a structure that the loop can
/// carry.
static bool hasWork(scf::ForOp loop) {
  return !isSegmentLoop(loop) && !findSources(*loop.getBody()).empty();
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
        if (!hasWork(loop))
          continue;
        reuseNeighbors(loop);
        changed = true;
        break;
      }
    }

    // What no loop carries: the bodies of functions and of loops whose
    // iterations are segments. Structures that follow one another there
    // share their storage all the same.
    SmallVector<Block *> blocks;
    getOperation()->walk([&](Operation *op) {
      if (auto function = dyn_cast<func::FuncOp>(op)) {
        if (!function.isExternal())
          blocks.push_back(&function.getBody().front());
      } else if (auto loop = dyn_cast<scf::ForOp>(op)) {
        if (isSegmentLoop(loop))
          blocks.push_back(loop.getBody());
      }
    });
    for (Block *block : blocks) {
      SmallVector<Chain, 2> chains = findChains(*block);
      for (Chain &chain : chains)
        if (chain.sources.size() > 1)
          linkChain(chain);
    }
  }
};
} // namespace

} // namespace md_exec
} // namespace mdir
