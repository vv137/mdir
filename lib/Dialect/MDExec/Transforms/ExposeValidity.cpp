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
/// particle has moved more than half the skin since the structure was
/// built, and hands the result to `refresh`.
static void exposeTest(RefreshNeighborsOp refresh) {
  Location loc = refresh.getLoc();
  OpBuilder builder(refresh);
  Value positions = refresh.getPositions();
  auto field = cast<md::FieldType>(positions.getType());
  Type real = field.getElementType();

  Value reference = ReferencePositionsOp::create(
      builder, loc, field, refresh.getNeighbors());
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
  Value moved = arith::SubFOp::create(kernel, loc, now, then);
  Value squares = arith::MulFOp::create(kernel, loc, moved, moved);
  Value distance2 = vector::ReductionOp::create(
      kernel, loc, vector::CombiningKind::ADD, squares);
  double skin = refresh.getSkin().convertToDouble();
  Value limit = arith::ConstantOp::create(
      kernel, loc, real, kernel.getFloatAttr(real, 0.25 * skin * skin));
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
