// convert-md-exec-to-md-alg: the loops over pairs over groups become
// md_alg.traverse_pairs (D[md-alg-traverse], #886, step 2 of #877). The
// facts of md_exec that the lowering needs are stated on the op as the IR
// moves down: the visit and the combination from the policy and the
// algebra (D301), the contract in the type of the list (D413), and the
// execution choices that need md_exec's view of the program (the frames
// that a pruning leaves, D268, and the pruning in the loop, D284).

#include "mdir/Conversion/Passes.h"

#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MDAlg/MDAlgDialect.h"
#include "mdir/Dialect/MDAlg/MDAlgOps.h"
#include "mdir/Dialect/MDExec/Algebra.h"
#include "mdir/Dialect/MDExec/Independence.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace mdir {
#define GEN_PASS_DEF_CONVERTMDEXECTOMDALG
#include "mdir/Conversion/Passes.h.inc"
} // namespace mdir

using namespace mlir;
using namespace mdir;

namespace {

/// The loops over pairs of groups that follow a refresh of their dual list
/// at the same positions and in the same cell, with nothing between that
/// may write the positions (D268): they take the positions in the frames
/// that the refresh's pruning left.
llvm::DenseSet<Operation *>
findSharedFrames(func::FuncOp function, md_exec::BufferAliases &aliases) {
  llvm::DenseSet<Operation *> shares;
  function.walk([&](md_exec::PairForOp pair) {
    if (pair.getTraversal() != md_exec::Traversal::Unique)
      return;
    auto refresh =
        pair.getNeighbors().getDefiningOp<md_exec::RefreshNeighborsOp>();
    if (!refresh || !refresh.getPruneSkin() ||
        refresh->getBlock() != pair->getBlock() ||
        refresh.getPositions() != pair.getPositions() ||
        refresh.getCellMutable().get() != pair.getCellMutable().get())
      return;
    Value positions = pair.getPositions();
    for (Operation *crossed = refresh->getNextNode(); crossed != pair;
         crossed = crossed->getNextNode()) {
      // An op whose effects are not all declared, or that may write or
      // free the positions, ends the sharing.
      std::optional<SmallVector<MemoryEffects::EffectInstance>> effects =
          getEffectsRecursively(crossed);
      if (!effects)
        return;
      for (const MemoryEffects::EffectInstance &effect : *effects) {
        if (!isa<MemoryEffects::Write, MemoryEffects::Free>(
                effect.getEffect()))
          continue;
        Value value = effect.getValue();
        if (!value || aliases.mayAlias(value, positions))
          return;
      }
    }
    shares.insert(pair.getOperation());
  });
  return shares;
}

/// The loops of `shares` that also prune for their refresh (D284): the
/// first op that takes the refresh's result, so that nothing reads the
/// inner list before the loop writes it. Each with its reach.
llvm::DenseMap<Operation *, double>
findPrunedInLoop(const llvm::DenseSet<Operation *> &shares) {
  llvm::DenseMap<Operation *, double> pruned;
  for (Operation *op : shares) {
    auto pair = cast<md_exec::PairForOp>(op);
    auto refresh =
        pair.getNeighbors().getDefiningOp<md_exec::RefreshNeighborsOp>();
    if (!refresh.getStale())
      continue;
    Value result = refresh.getResult();
    bool first = true;
    for (Operation *crossed = refresh->getNextNode(); first && crossed != pair;
         crossed = crossed->getNextNode())
      crossed->walk([&](Operation *inner) {
        first &= !llvm::is_contained(inner->getOperands(), result);
      });
    if (!first)
      continue;
    pruned[op] = refresh.getCutoff().convertToDouble() +
                 refresh.getPruneSkin()->convertToDouble();
  }
  return pruned;
}

class ConvertMDExecToMDAlg
    : public mdir::impl::ConvertMDExecToMDAlgBase<ConvertMDExecToMDAlg> {
public:
  using mdir::impl::ConvertMDExecToMDAlgBase<
      ConvertMDExecToMDAlg>::ConvertMDExecToMDAlgBase;

  void runOnOperation() final {
    SmallVector<func::FuncOp> functions;
    getOperation().walk([&](func::FuncOp function) {
      functions.push_back(function);
    });
    for (func::FuncOp function : functions)
      if (failed(convert(function)))
        return signalPassFailure();
  }

private:
  LogicalResult convert(func::FuncOp function) {
    // The analyses run on the loops of md_exec before any is converted, so
    // that each sees the program as it was.
    md_exec::BufferAliases aliases;
    llvm::DenseSet<Operation *> shares = findSharedFrames(function, aliases);
    llvm::DenseMap<Operation *, double> pruned;
    if (fusePruning)
      pruned = findPrunedInLoop(shares);

    SmallVector<md_exec::PairForOp> loops;
    function.walk([&](md_exec::PairForOp loop) {
      if (loop.getTraversal() == md_exec::Traversal::Unique)
        loops.push_back(loop);
    });
    for (md_exec::PairForOp loop : loops)
      if (failed(convert(loop, shares.contains(loop.getOperation()),
                         pruned.lookup(loop.getOperation()),
                         pruned.contains(loop.getOperation()))))
        return failure();
    return success();
  }

  LogicalResult convert(md_exec::PairForOp loop, bool frames, double reach,
                        bool fused) {
    if (!loop.isStorageForm())
      return loop.emitOpError()
             << "is not in the storage form; run 'md-exec-assign-storage' "
                "first";
    std::optional<SmallVector<md_exec::Algebra>> algebras =
        md_exec::algebra::getAlgebras(loop);
    if (!algebras)
      return loop.emitOpError() << md_exec::algebra::getMissingMessage();
    OpBuilder builder(loop);
    SmallVector<Attribute> algebra, exchange;
    bool ordered = false;
    for (md_exec::Algebra value : *algebras) {
      algebra.push_back(builder.getStringAttr(md_exec::stringifyAlgebra(value)));
      ordered |= value == md_exec::Algebra::FixedOrder;
    }
    unsigned values = loop.getOuts().size() + loop.getReduce().size();
    for (unsigned i = 0; i != values; ++i)
      exchange.push_back(
          builder.getStringAttr(md::stringifyExchange(loop.getExchange(i))));
    auto refresh =
        loop.getNeighbors().getDefiningOp<md_exec::RefreshNeighborsOp>();
    bool inner = refresh && refresh.getPruneSkin();

    MLIRContext *context = loop.getContext();
    auto op = md_alg::TraversePairsOp::create(
        builder, loop.getLoc(), loop->getResultTypes(), loop.getNeighbors(),
        inner ? builder.getUnitAttr() : UnitAttr(), /*lists=*/ValueRange(),
        /*sizes=*/ValueRange(), /*contract=*/TypeAttr(), /*reach=*/FloatAttr(),
        loop.getPositions(), loop.getCellMutable().get(), loop.getIns(),
        loop.getOuts(), loop.getReduce(), loop.getScratch(),
        md_alg::MultiplicityAttr::get(
            context, loop.getTraversal() == md_exec::Traversal::Unique
                         ? md_alg::Multiplicity::Unique
                         : md_alg::Multiplicity::Directed),
        md_alg::VisitAttr::get(context, ordered ? md_alg::Visit::Stable
                                                : md_alg::Visit::Unordered),
        md_alg::CombineAttr::get(
            context, loop.getConflict() == md_exec::Conflict::Atomic
                         ? md_alg::Combine::Atomic
                         : md_alg::Combine::Owner),
        builder.getArrayAttr(algebra), builder.getArrayAttr(exchange),
        loop.getCutoffAttr(), loop.getWeightsAttr(), loop.getOverwriteAttr(),
        loop.getScatterAttr(), frames ? builder.getUnitAttr() : UnitAttr(),
        fused ? builder.getI64IntegerAttr(fusePruningGroups) : IntegerAttr(),
        fused ? builder.getF64FloatAttr(reach) : FloatAttr());
    // The region moves as it is, its terminator made md_alg's.
    op.getKernel().takeBody(loop.getKernel());
    Block &body = op.getKernel().front();
    Operation *terminator = body.getTerminator();
    OpBuilder end(terminator);
    md_alg::YieldOp::create(end, terminator->getLoc(),
                            terminator->getOperands());
    terminator->erase();
    loop->replaceAllUsesWith(op->getResults());
    loop->erase();
    return success();
  }
};

} // namespace
