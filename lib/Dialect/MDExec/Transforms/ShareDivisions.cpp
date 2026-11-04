// Divisions shared in the f64 kernels of loops over pairs, under fast_math
// (D[shared-divisions], #731).
//
// See the description of the pass in Passes.td.

#include "mdir/Dialect/MDExec/Transforms/Passes.h"

#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::md_exec;

namespace mdir {
namespace md_exec {

#define GEN_PASS_DEF_SHAREDIVISIONS
#include "mdir/Dialect/MDExec/Transforms/Passes.h.inc"

namespace {
/// Whether `value` is the constant 1.
bool isOne(Value value) { return matchPattern(value, m_OneFloat()); }

/// The rewrite of the divisions of one block.
class Sharer {
public:
  explicit Sharer(Block &block) : block(block) {}

  void run();

private:
  /// The point right after the definition of `value`, or the start of the
  /// block if it is defined outside the block's ops.
  OpBuilder getBuilderAfter(Value value);

  /// The reciprocal of `divisor`, created once.
  Value getReciprocal(Value divisor);

  /// 1 / sqrt(x) as sqrt(x) times the reciprocal of `x`, created once.
  Value getInverseRoot(math::SqrtOp root);

  /// The fastmath flags that every division by `divisor` has.
  arith::FastMathFlags getCommonFlags(Value divisor);

  Block &block;
  /// The divisions of the block by each divisor, in the order of the block.
  llvm::MapVector<Value, SmallVector<arith::DivFOp>> divisions;
  DenseMap<Value, Value> reciprocals, inverseRoots;
};

OpBuilder Sharer::getBuilderAfter(Value value) {
  OpBuilder builder(block.getParentOp()->getContext());
  Operation *definition = value.getDefiningOp();
  if (definition && definition->getBlock() == &block)
    builder.setInsertionPointAfter(definition);
  else
    builder.setInsertionPointToStart(&block);
  return builder;
}

arith::FastMathFlags Sharer::getCommonFlags(Value divisor) {
  auto flags = arith::FastMathFlags::fast;
  auto found = divisions.find(divisor);
  if (found != divisions.end())
    for (arith::DivFOp division : found->second)
      flags = flags & division.getFastmath();
  return found == divisions.end() ? arith::FastMathFlags::none : flags;
}

Value Sharer::getReciprocal(Value divisor) {
  Value &reciprocal = reciprocals[divisor];
  if (reciprocal)
    return reciprocal;
  OpBuilder builder = getBuilderAfter(divisor);
  Location loc = divisor.getLoc();
  Value one = arith::ConstantOp::create(
      builder, loc, builder.getFloatAttr(divisor.getType(), 1.0));
  reciprocal = arith::DivFOp::create(builder, loc, one, divisor,
                                     getCommonFlags(divisor));
  return reciprocal;
}

Value Sharer::getInverseRoot(math::SqrtOp root) {
  Value &inverse = inverseRoots[root.getResult()];
  if (inverse)
    return inverse;
  Value reciprocal = getReciprocal(root.getOperand());
  // After the root and after the reciprocal: the reciprocal follows the
  // radicand, which precedes the root, unless the root is outside the block.
  OpBuilder builder = getBuilderAfter(root.getResult());
  if (root->getBlock() != &block)
    builder.setInsertionPointAfter(reciprocal.getDefiningOp());
  inverse = arith::MulFOp::create(builder, root.getLoc(), root.getResult(),
                                  reciprocal,
                                  getCommonFlags(root.getResult()));
  return inverse;
}

void Sharer::run() {
  for (Operation &op : block)
    if (auto division = dyn_cast<arith::DivFOp>(&op);
        division && division.getType().isF64())
      divisions[division.getRhs()].push_back(division);

  // Which divisions take which factor, decided before anything is created,
  // so that the decision does not depend on what the rewrite adds.
  SmallVector<std::pair<arith::DivFOp, Value>> rewrites;
  for (auto &[divisor, list] : divisions) {
    // A division by sqrt(x) in a block that also divides by x: 1/sqrt(x) is
    // sqrt(x) · (1/x), with the reciprocal of x shared.
    auto root = divisor.getDefiningOp<math::SqrtOp>();
    if (root && divisions.count(root.getOperand())) {
      for (arith::DivFOp division : list)
        rewrites.push_back({division, root.getResult()});
      continue;
    }
    bool shared = list.size() > 1;
    // The divisor is the radicand of a root that another division divides
    // by: its reciprocal serves both.
    for (auto &[other, unused] : divisions)
      if (auto otherRoot = other.getDefiningOp<math::SqrtOp>())
        shared |= otherRoot.getOperand() == divisor;
    if (!shared)
      continue;
    for (arith::DivFOp division : list)
      rewrites.push_back({division, divisor});
  }

  for (auto [division, divisor] : rewrites) {
    auto root = divisor.getDefiningOp<math::SqrtOp>();
    Value factor = root && divisions.count(root.getOperand())
                       ? getInverseRoot(root)
                       : getReciprocal(divisor);
    Value result = factor;
    if (!isOne(division.getLhs())) {
      OpBuilder builder(division);
      result = arith::MulFOp::create(builder, division.getLoc(),
                                     division.getLhs(), factor,
                                     division.getFastmath());
    }
    division.getResult().replaceAllUsesWith(result);
    division.erase();
  }
}

class ShareDivisions : public impl::ShareDivisionsBase<ShareDivisions> {
public:
  void runOnOperation() final {
    SmallVector<Block *> blocks;
    getOperation()->walk([&](PairForOp loop) {
      loop.getRegion().walk([&](Block *block) { blocks.push_back(block); });
    });
    for (Block *block : blocks)
      Sharer(*block).run();
  }
};
} // namespace

} // namespace md_exec
} // namespace mdir
