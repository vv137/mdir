// convert-md-exec-to-md-alg: the loops over pairs over groups become
// md_alg.traverse_pairs (D415, #886, step 2 of #877). The
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
#include "mdir/Dialect/MDExec/MDExecOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

namespace mdir {
#define GEN_PASS_DEF_CONVERTMDEXECTOMDALG
#include "mdir/Conversion/Passes.h.inc"
} // namespace mdir

using namespace mlir;
using namespace mdir;

namespace {

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
    SmallVector<md_exec::PairForOp> loops;
    function.walk([&](md_exec::PairForOp loop) {
      if (loop.getTraversal() == md_exec::Traversal::Unique)
        loops.push_back(loop);
    });
    for (md_exec::PairForOp loop : loops)
      if (failed(convert(loop)))
        return failure();
    return success();
  }

  LogicalResult convert(md_exec::PairForOp loop) {
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
        loop.getScatterAttr(), /*frames=*/UnitAttr(),
        /*pruneGroups=*/IntegerAttr(), /*pruneReach=*/FloatAttr());
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
