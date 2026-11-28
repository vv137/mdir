// The test of validity of a neighbor structure, as a loop over particles.

#include "mdir/Dialect/MDExec/Transforms/Passes.h"

#include "mdir/Dialect/MD/MDTypes.h"
#include "mdir/Dialect/MDExec/MDExecDialect.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::md_exec;

/// Emits, before `refresh`, the loop over particles that tells whether a
/// particle has moved too far for the structure since it was built, and
/// returns its result. A barostat scales the positions with the cell; the
/// test compares them with the reference scaled as the cell was,
/// m ⊙ x_ref with m = L / L_ref, against half of min(m) R − r_c (D80).
/// With `pruned`, the test of the inner list of a dual list: against the
/// configuration and the cell of the last pruning, with the reach
/// R_in = r_c + prune_skin (D114).
static Value emitTest(RefreshNeighborsOp refresh, bool pruned) {
  Location loc = refresh.getLoc();
  OpBuilder builder(refresh);
  Value positions = refresh.getPositions();
  auto field = cast<md::FieldType>(positions.getType());
  Type real = field.getElementType();

  Value reference = ReferencePositionsOp::create(
      builder, loc, field, refresh.getNeighbors(), pruned);
  auto edgesType = VectorType::get({3}, builder.getF64Type());
  Value edges = CellEdgesOp::create(builder, loc, edgesType,
                                    refresh.getCell());
  Value built = ReferenceCellOp::create(
      builder, loc, refresh.getCell().getType(), refresh.getNeighbors(),
      pruned);
  Value builtEdges = CellEdgesOp::create(builder, loc, edgesType, built);
  Value scale = arith::DivFOp::create(builder, loc, edges, builtEdges);
  Value least = vector::ReductionOp::create(
      builder, loc, vector::CombiningKind::MINNUMF, scale);
  double cutoff = refresh.getCutoff().convertToDouble();
  double reach = cutoff + (pruned ? refresh.getPruneSkin()->convertToDouble()
                                  : refresh.getSkin().convertToDouble());
  auto constant = [&](double value) -> Value {
    return arith::ConstantOp::create(builder, loc, builder.getF64Type(),
                                     builder.getF64FloatAttr(value));
  };
  Value margin = arith::SubFOp::create(
      builder, loc, arith::MulFOp::create(builder, loc, least, constant(reach)),
      constant(cutoff));
  Value radii = refresh.getAnchorRadii();
  if (radii) {
    // A contraction spends skin at the largest covering reach. On an
    // expansion the smallest reach gives a conservative bound.
    Value change = arith::SubFOp::create(builder, loc, least, constant(1.0));
    Value contraction =
        arith::MinimumFOp::create(builder, loc, change, constant(0.0));
    Value loss = arith::MulFOp::create(
        builder, loc, contraction,
        constant(2.0 * refresh.getMaxAnchorRadius()->convertToDouble()));
    margin = arith::AddFOp::create(builder, loc, margin, loss);
  }
  Value half = arith::MulFOp::create(
      builder, loc, arith::MaximumFOp::create(builder, loc, margin,
                                              constant(0.0)),
      constant(0.5));
  // A cell that is not a number, before the first build, gives a limit
  // that is not a number, and the test fails; the build does not count it.
  Value limit2 = arith::MulFOp::create(builder, loc, half, half);
  Value no = arith::ConstantOp::create(builder, loc, builder.getI1Type(),
                                       builder.getBoolAttr(false));
  Value zeroMargin = radii ? constant(0.0) : Value();
  SmallVector<Value> inputs{positions, reference};
  if (radii)
    inputs.push_back(radii);
  auto loop = ParticleForOp::create(
      builder, loc, TypeRange{builder.getI1Type()},
      inputs, /*outs=*/ValueRange(),
      /*reduce=*/ValueRange{no}, /*scratch=*/ValueRange());

  Block *block = new Block();
  loop.getKernel().push_back(block);
  Type value = field.getKernelValueType();
  Value now = block->addArgument(value, loc);
  Value then = block->addArgument(value, loc);

  Value radius;
  if (radii)
    radius = block->addArgument(
        cast<md::FieldType>(radii.getType()).getKernelValueType(), loc);

  OpBuilder kernel = OpBuilder::atBlockEnd(block);
  Value scaled = arith::MulFOp::create(kernel, loc, then, scale);
  Value moved = arith::SubFOp::create(kernel, loc, now, scaled);
  Value squares = arith::MulFOp::create(kernel, loc, moved, moved);
  Value distance2 = vector::ReductionOp::create(
      kernel, loc, vector::CombiningKind::ADD, squares);
  Value limit = limit2;
  (void)real;
  // The structure is valid while no displacement exceeds the limit. A
  // displacement that is not a number exceeds it.
  Value far = arith::CmpFOp::create(kernel, loc, arith::CmpFPredicate::UGT,
                                    distance2, limit);
  if (radius) {
    Value zero = arith::ConstantOp::create(
        kernel, loc, kernel.getFloatAttr(radius.getType(), 0.0));
    // The sign marks the members. Lone sites have zero radius and are
    // anchors in their own right.
    Value anchor = arith::CmpFOp::create(kernel, loc, arith::CmpFPredicate::OGE,
                                       radius, zero);
    far = arith::AndIOp::create(kernel, loc, far, anchor);
    // A negative margin is invalid even if no anchor has moved.
    Value contracted = arith::CmpFOp::create(
        kernel, loc, arith::CmpFPredicate::ULT, margin, zeroMargin);
    far = arith::OrIOp::create(kernel, loc, far, contracted);
  }
  YieldOp::create(kernel, loc, ValueRange{far});
  return loop.getResult(0);
}

/// Hands `refresh` the tests of its structure: of the outer list, and of
/// the inner one of a dual list.
static void exposeTest(RefreshNeighborsOp refresh) {
  refresh.getMovedMutable().assign(emitTest(refresh, /*pruned=*/false));
  if (refresh.getPruneSkin())
    refresh.getStaleMutable().assign(emitTest(refresh, /*pruned=*/true));
}

/// Gives a refresh of a list of units the test of its writer
/// (D422): the writer comes right after the op that
/// gives the refresh's positions, the last op that writes them, in the
/// refresh's block (or where the block begins, for positions from outside
/// it), so that it follows every update of the step, the constraints and a
/// barostat's scaling included, and comes before the reciprocal sum and the
/// loops over the list. An empty list is built whatever has moved.
static void exposeWriter(RefreshSegmentedPairsOp refresh) {
  if (refresh.getList().getDefiningOp<EmptySegmentedPairsOp>())
    return;
  // After the latest of the ops in the block that define its operands.
  Block *block = refresh->getBlock();
  Operation *last = nullptr;
  Value operands[] = {refresh.getPositions(), refresh.getList(),
                      refresh.getCellMutable().get()};
  for (Value value : operands) {
    Operation *def = value.getDefiningOp();
    if (def && def->getBlock() == block &&
        (!last || last->isBeforeInBlock(def)))
      last = def;
  }
  OpBuilder builder(refresh->getContext());
  if (last)
    builder.setInsertionPointAfter(last);
  else
    builder.setInsertionPointToStart(block);
  auto writer = WriteUnitFramesOp::create(
      builder, refresh.getLoc(), builder.getI1Type(), builder.getI1Type(),
      refresh.getList(), refresh.getPositions(), refresh.getCellMutable().get(),
      refresh.getCutoffAttr(), refresh.getSkinAttr(),
      refresh.getPruneSkinAttr());
  refresh.getMovedMutable().assign(writer.getMoved());
  refresh.getStaleMutable().assign(writer.getStale());
}

namespace mdir {
namespace md_exec {

#define GEN_PASS_DEF_EXPOSEVALIDITY
#include "mdir/Dialect/MDExec/Transforms/Passes.h.inc"

namespace {
class ExposeValidity : public impl::ExposeValidityBase<ExposeValidity> {
public:
  using impl::ExposeValidityBase<ExposeValidity>::ExposeValidityBase;

  void runOnOperation() final {
    SmallVector<RefreshNeighborsOp> refreshes;
    getOperation()->walk([&](RefreshNeighborsOp refresh) {
      if (!refresh.isStorageForm() && !refresh.getMoved() &&
          refresh.getPolicy() == RebuildPolicy::Check)
        refreshes.push_back(refresh);
    });

    SmallVector<RefreshSegmentedPairsOp> lists;
    getOperation()->walk([&](RefreshSegmentedPairsOp refresh) {
      if (!isa<MemRefType>(refresh.getPositions().getType()) &&
          !refresh.getMoved())
        lists.push_back(refresh);
    });
    for (RefreshSegmentedPairsOp refresh : lists)
      exposeWriter(refresh);

    for (RefreshNeighborsOp refresh : refreshes) {
      // An empty structure is built whatever has moved.
      if (refresh.getNeighbors().getDefiningOp<EmptyNeighborsOp>())
        refresh.setPolicy(RebuildPolicy::Always);
      else
        exposeTest(refresh);
    }
  }
};
} // namespace

} // namespace md_exec
} // namespace mdir
