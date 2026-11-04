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
/// The rewrite of the divisions of one block.
class Sharer {
public:
  Sharer(Block &block, bool rsqrt) : block(block), rsqrt(rsqrt) {}

  void run();

private:
  /// A builder at the point right after the definition of `value`, or at
  /// the start of the block if it is defined outside the block's ops.
  OpBuilder getBuilderAfter(Value value);

  /// The factor that replaces a division by `divisor`: 1/divisor, or
  /// rsqrt(x) for divisor = sqrt(x), created once.
  Value getFactor(Value divisor);

  Block &block;
  bool rsqrt;
  /// The divisions of the block by each divisor, in the order of the block.
  llvm::MapVector<Value, SmallVector<arith::DivFOp>> divisions;
  DenseMap<Value, Value> factors;
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

Value Sharer::getFactor(Value divisor) {
  Value &factor = factors[divisor];
  if (factor)
    return factor;
  // The fastmath flags that every division by `divisor` has.
  auto flags = arith::FastMathFlags::fast;
  for (arith::DivFOp division : divisions.lookup(divisor))
    flags = flags & division.getFastmath();
  OpBuilder builder = getBuilderAfter(divisor);
  Location loc = divisor.getLoc();
  if (auto root = divisor.getDefiningOp<math::SqrtOp>(); root && rsqrt) {
    factor = math::RsqrtOp::create(builder, loc, root.getOperand(), flags);
    return factor;
  }
  Value one = arith::ConstantOp::create(
      builder, loc, builder.getFloatAttr(divisor.getType(), 1.0));
  factor = arith::DivFOp::create(builder, loc, one, divisor, flags);
  return factor;
}

void Sharer::run() {
  for (Operation &op : block)
    if (auto division = dyn_cast<arith::DivFOp>(&op);
        division && division.getType().isF64())
      divisions[division.getRhs()].push_back(division);

  SmallVector<math::SqrtOp> roots;
  for (auto &[divisor, list] : divisions) {
    // A division by sqrt(x) becomes a product by rsqrt(x), with `rsqrt`;
    // any other divisor is shared when two or more divisions take it.
    auto root = divisor.getDefiningOp<math::SqrtOp>();
    if (!(root && rsqrt) && list.size() < 2)
      continue;
    if (root && rsqrt)
      roots.push_back(root);
    for (arith::DivFOp division : list) {
      Value factor = getFactor(divisor);
      Value result = factor;
      if (!matchPattern(division.getLhs(), m_OneFloat())) {
        OpBuilder builder(division);
        result = arith::MulFOp::create(builder, division.getLoc(),
                                       division.getLhs(), factor,
                                       division.getFastmath());
      }
      division.getResult().replaceAllUsesWith(result);
      division.erase();
    }
  }
  // A root that only the divisions took.
  for (math::SqrtOp root : roots)
    if (root->getBlock() == &block && root->use_empty())
      root.erase();
}

class ShareDivisions : public impl::ShareDivisionsBase<ShareDivisions> {
public:
  using impl::ShareDivisionsBase<ShareDivisions>::ShareDivisionsBase;

  void runOnOperation() final {
    SmallVector<Block *> blocks;
    getOperation()->walk([&](PairForOp loop) {
      loop.getRegion().walk([&](Block *block) { blocks.push_back(block); });
    });
    for (Block *block : blocks)
      Sharer(*block, rsqrt).run();
  }
};
} // namespace

} // namespace md_exec
} // namespace mdir
