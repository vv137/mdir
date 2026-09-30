// Bypass of the updates of the other sets of a disjoint union.
//
// See docs/decisions.md, D83.

#include "mdir/Dialect/MD/Transforms/Passes.h"

#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;
using namespace mdir::md;

/// The tuple set of the relation of `op`.
static FlatSymbolRefAttr getTupleSet(Operation *op) {
  auto relation = cast<RelationType>(op->getOperand(0).getType());
  return relation.getTupleSet();
}

/// If `map` adds to one of its two fields what a loop over the tuples of
/// a set of `unionOp` other than `set` gathered, returns the field it adds
/// to; that is where the two agree at the members of the tuples of `set`.
static Value getBypassed(MapParticlesOp map, DisjointUnionOp unionOp,
                         FlatSymbolRefAttr set) {
  if (map.getGathered().size() != 2)
    return {};
  Block &kernel = map.getKernel().front();
  if (!llvm::hasSingleElement(kernel.without_terminator()))
    return {};
  auto add = dyn_cast<arith::AddFOp>(kernel.front());
  Operation *yield = kernel.getTerminator();
  if (!add || yield->getNumOperands() != 1 ||
      yield->getOperand(0) != add.getResult())
    return {};
  // The kernel argument that each operand of the sum is.
  auto getIndex = [&](Value value) -> int {
    auto argument = dyn_cast<BlockArgument>(value);
    return argument && argument.getOwner() == &kernel
               ? static_cast<int>(argument.getArgNumber())
               : -1;
  };
  int lhs = getIndex(add.getLhs()), rhs = getIndex(add.getRhs());
  if (lhs < 0 || rhs < 0 || lhs == rhs)
    return {};
  // The field that is zero at the members of `set`: gathered over the
  // tuples of an other set of the union.
  auto isZeroThere = [&](Value field) {
    auto gather = field.getDefiningOp<GatherTuplesOp>();
    if (!gather)
      return false;
    FlatSymbolRefAttr other = getTupleSet(gather);
    return other && other != set && unionOp.contains(other);
  };
  Value first = map.getGathered()[lhs], second = map.getGathered()[rhs];
  if (isZeroThere(second))
    return first;
  if (isZeroThere(first))
    return second;
  return {};
}

namespace mdir {
namespace md {

#define GEN_PASS_DEF_BYPASSUPDATES
#include "mdir/Dialect/MD/Transforms/Passes.h.inc"

namespace {
class BypassUpdates : public impl::BypassUpdatesBase<BypassUpdates> {
public:
  using impl::BypassUpdatesBase<BypassUpdates>::BypassUpdatesBase;

  void runOnOperation() final {
    // The union of each tuple set that is in one.
    llvm::DenseMap<Attribute, DisjointUnionOp> unions;
    getOperation()->walk([&](DisjointUnionOp op) {
      for (Attribute set : op.getTupleSets())
        unions[set] = op;
    });
    if (unions.empty())
      return;

    getOperation()->walk([&](Operation *op) {
      if (!isa<GatherTuplesOp, SumTuplesOp>(op))
        return;
      FlatSymbolRefAttr set = getTupleSet(op);
      if (!set)
        return;
      DisjointUnionOp unionOp = unions.lookup(set);
      if (!unionOp)
        return;
      // The loop is the next link of a chain of updates when a map adds
      // what it gathers to a field that it reads: only then does reading
      // the field before the updates of the other sets let the links run
      // at once. A loop that is no link, such as one over velocities that
      // reads the positions that a chain gave, keeps its fields; reading
      // an earlier one would only keep that one alive.
      auto isLinkOf = [&](Value field) {
        auto gather = dyn_cast<GatherTuplesOp>(op);
        if (!gather)
          return false;
        for (Operation *user : gather.getResult().getUsers()) {
          auto map = dyn_cast<MapParticlesOp>(user);
          if (map && map.getGathered().size() == 2 &&
              llvm::is_contained(map.getGathered(), field) &&
              getBypassed(map, unionOp, FlatSymbolRefAttr()) == field)
            return true;
        }
        return false;
      };
      // The positions and the gathered fields, which the loop reads at
      // the members of its tuples only.
      unsigned gathered =
          isa<GatherTuplesOp>(op)
              ? cast<GatherTuplesOp>(op).getGathered().size()
              : cast<SumTuplesOp>(op).getGathered().size();
      for (unsigned index = 1; index != 3 + gathered; ++index) {
        if (index == 2)
          continue; // The cell.
        Value field = op->getOperand(index);
        if (!isLinkOf(field))
          continue;
        while (auto map = field.getDefiningOp<MapParticlesOp>()) {
          Value bypassed = getBypassed(map, unionOp, set);
          if (!bypassed)
            break;
          field = bypassed;
        }
        op->setOperand(index, field);
      }
    });
  }
};
} // namespace

} // namespace md
} // namespace mdir
