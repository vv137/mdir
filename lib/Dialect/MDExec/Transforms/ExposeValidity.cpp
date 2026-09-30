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
/// hands the result to `refresh`. A barostat scales the positions with the
/// cell; the test compares them with the reference scaled as the cell was,
/// m ⊙ x_ref with m = L / L_ref, against half of min(m) R − r_c (D80).
static void exposeTest(RefreshNeighborsOp refresh) {
  Location loc = refresh.getLoc();
  OpBuilder builder(refresh);
  Value positions = refresh.getPositions();
  auto field = cast<md::FieldType>(positions.getType());
  Type real = field.getElementType();

  Value reference = ReferencePositionsOp::create(
      builder, loc, field, refresh.getNeighbors());
  auto edgesType = VectorType::get({3}, builder.getF64Type());
  Value edges = CellEdgesOp::create(builder, loc, edgesType,
                                    refresh.getCell());
  Value built = ReferenceCellOp::create(
      builder, loc, refresh.getCell().getType(), refresh.getNeighbors());
  Value builtEdges = CellEdgesOp::create(builder, loc, edgesType, built);
  Value scale = arith::DivFOp::create(builder, loc, edges, builtEdges);
  Value least = vector::ReductionOp::create(
      builder, loc, vector::CombiningKind::MINNUMF, scale);
  double cutoff = refresh.getCutoff().convertToDouble();
  double reach = cutoff + refresh.getSkin().convertToDouble();
  auto constant = [&](double value) -> Value {
    return arith::ConstantOp::create(builder, loc, builder.getF64Type(),
                                     builder.getF64FloatAttr(value));
  };
  Value margin = arith::SubFOp::create(
      builder, loc, arith::MulFOp::create(builder, loc, least, constant(reach)),
      constant(cutoff));
  Value half = arith::MulFOp::create(
      builder, loc, arith::MaximumFOp::create(builder, loc, margin,
                                              constant(0.0)),
      constant(0.5));
  // A cell that is not a number, before the first build, gives a limit
  // that is not a number, and the test fails; the build does not count it.
  Value limit2 = arith::MulFOp::create(builder, loc, half, half);
  Value no = arith::ConstantOp::create(builder, loc, builder.getI1Type(),
                                       builder.getBoolAttr(false));
  auto loop = ParticleForOp::create(
      builder, loc, TypeRange{builder.getI1Type()},
      ValueRange{positions, reference}, /*outs=*/ValueRange(),
      /*reduce=*/ValueRange{no}, /*scratch=*/ValueRange());

  Block *block = new Block();
  loop.getKernel().push_back(block);
  Type value = field.getKernelValueType();
  Value now = block->addArgument(value, loc);
  Value then = block->addArgument(value, loc);

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
  YieldOp::create(kernel, loc, ValueRange{far});

  refresh.getMovedMutable().assign(loop.getResult(0));
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
