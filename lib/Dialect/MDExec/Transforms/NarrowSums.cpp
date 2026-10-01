// Vector sums of loops narrowed to the elements that are used.
//
// See the description of the pass in Passes.td.

#include "mdir/Dialect/MDExec/Transforms/Passes.h"

#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallBitVector.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::md_exec;

namespace mdir {
namespace md_exec {

#define GEN_PASS_DEF_NARROWSUMS
#include "mdir/Dialect/MDExec/Transforms/Passes.h.inc"

namespace {
/// The vectors of one dimension of floating-point numbers, whose elements
/// the pass follows.
VectorType getTracked(Type type) {
  auto vector = dyn_cast<VectorType>(type);
  if (!vector || vector.getRank() != 1 || vector.isScalable() ||
      !isa<FloatType>(vector.getElementType()))
    return VectorType();
  return vector;
}

class Demand {
public:
  /// The elements of `value` that its users take.
  const llvm::SmallBitVector &get(Value value) {
    auto found = demand.find(value);
    if (found != demand.end())
      return found->second;
    int64_t size = getTracked(value.getType()).getNumElements();
    // A cycle, through the arguments of a loop, takes every element.
    demand[value] = llvm::SmallBitVector(size, true);
    llvm::SmallBitVector used(size);
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      if (auto extract = dyn_cast<vector::ExtractOp>(user)) {
        std::optional<int64_t> index;
        if (extract.getStaticPosition().size() == 1 &&
            extract.getStaticPosition()[0] != ShapedType::kDynamic)
          index = extract.getStaticPosition()[0];
        if (!index) {
          used.set();
          break;
        }
        used.set(*index);
        continue;
      }
      if (isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
              arith::NegFOp>(user) &&
          user->getNumResults() == 1 &&
          user->getResult(0).getType() == value.getType()) {
        used |= get(user->getResult(0));
        continue;
      }
      used.set();
      break;
    }
    demand[value] = used;
    return demand[value];
  }

private:
  llvm::DenseMap<Value, llvm::SmallBitVector> demand;
};

/// Narrows the sum of the loop `loop` that is its result `resultIndex`, its
/// operand `reduceOperand`, and the value `yieldIndex` of its kernel, to the
/// elements `kept`.
void narrow(Operation *loop, unsigned resultIndex, unsigned reduceOperand,
            unsigned yieldIndex, const llvm::SmallBitVector &kept) {
  Value result = loop->getResult(resultIndex);
  auto type = getTracked(result.getType());
  SmallVector<int64_t> lanes;
  for (int64_t i : kept.set_bits())
    lanes.push_back(i);
  auto narrowType =
      VectorType::get({static_cast<int64_t>(lanes.size())},
                      type.getElementType());
  Location loc = loop->getLoc();
  auto select = [&](OpBuilder &b, Value full) -> Value {
    SmallVector<Value> parts;
    for (int64_t lane : lanes)
      parts.push_back(vector::ExtractOp::create(b, loc, full, lane));
    return vector::FromElementsOp::create(b, loc, narrowType, parts);
  };
  // The initial value.
  OpBuilder before(loop);
  Value init = loop->getOperand(reduceOperand);
  loop->setOperand(reduceOperand, select(before, init));
  // The value of the kernel. A kernel that computes in a narrower type
  // and widens its value keeps doing so, so that the loop still sums in the
  // narrower type and widens at the end.
  Operation *yield = loop->getRegion(0).front().getTerminator();
  OpBuilder inside(yield);
  Value yielded = yield->getOperand(yieldIndex);
  if (auto widen = yielded.getDefiningOp<arith::ExtFOp>()) {
    auto narrowSource = cast<VectorType>(widen.getIn().getType());
    SmallVector<Value> parts;
    for (int64_t lane : lanes)
      parts.push_back(vector::ExtractOp::create(inside, loc, widen.getIn(),
                                                lane));
    Value narrowed = vector::FromElementsOp::create(
        inside, loc,
        VectorType::get({static_cast<int64_t>(lanes.size())},
                        narrowSource.getElementType()),
        parts);
    yield->setOperand(yieldIndex, arith::ExtFOp::create(inside, loc,
                                                        narrowType, narrowed));
  } else {
    yield->setOperand(yieldIndex, select(inside, yielded));
  }
  // The result, and a vector of the old type for its users.
  result.setType(narrowType);
  OpBuilder after(loop->getContext());
  after.setInsertionPointAfter(loop);
  Value zero = arith::ConstantOp::create(
      after, loc, type.getElementType(),
      after.getFloatAttr(type.getElementType(), 0.0));
  SmallVector<Value> parts(type.getNumElements(), zero);
  llvm::SmallPtrSet<Operation *, 8> extracts;
  for (auto [k, lane] : llvm::enumerate(lanes)) {
    auto extract = vector::ExtractOp::create(after, loc, result, k);
    extracts.insert(extract);
    parts[lane] = extract;
  }
  Value full = vector::FromElementsOp::create(after, loc, type, parts);
  result.replaceUsesWithIf(full, [&](OpOperand &use) {
    return !extracts.contains(use.getOwner());
  });
}

/// The number of the operand of `loop` that is the initial value of its
/// sum `index`.
unsigned getReduceOperand(Operation *loop, unsigned index) {
  if (auto pair = dyn_cast<PairForOp>(loop))
    return pair.getReduceMutable()[index].getOperandNumber();
  if (auto tuple = dyn_cast<TupleForOp>(loop))
    return tuple.getReduceMutable()[index].getOperandNumber();
  return cast<ParticleForOp>(loop).getReduceMutable()[index].getOperandNumber();
}

class NarrowSums : public impl::NarrowSumsBase<NarrowSums> {
public:
  void runOnOperation() final {
    Demand demand;
    struct Candidate {
      Operation *loop;
      unsigned result, operand, yield;
      llvm::SmallBitVector kept;
    };
    SmallVector<Candidate> candidates;
    getOperation()->walk([&](Operation *op) {
      ValueRange reduce;
      unsigned numOuts = 0;
      if (auto pair = dyn_cast<PairForOp>(op)) {
        if (pair.isStorageForm())
          return;
        reduce = pair.getReduce();
        numOuts = pair.getOuts().size();
      } else if (auto tuple = dyn_cast<TupleForOp>(op)) {
        if (tuple.isStorageForm())
          return;
        reduce = tuple.getReduce();
        numOuts = tuple.getOuts().size();
      } else if (auto particle = dyn_cast<ParticleForOp>(op)) {
        if (particle.isStorageForm())
          return;
        reduce = particle.getReduce();
        numOuts = particle.getOuts().size();
      } else {
        return;
      }
      Operation *yield = op->getRegion(0).front().getTerminator();
      unsigned firstSum = yield->getNumOperands() - reduce.size();
      for (auto [i, init] : llvm::enumerate(reduce)) {
        Value result = op->getResult(numOuts + i);
        auto type = getTracked(result.getType());
        if (!type)
          continue;
        llvm::SmallBitVector kept = demand.get(result);
        if (kept.none() || kept.all())
          continue;
        unsigned operand = getReduceOperand(op, i);
        candidates.push_back(
            {op, numOuts + static_cast<unsigned>(i), operand,
             firstSum + static_cast<unsigned>(i), kept});
      }
    });
    for (Candidate &candidate : candidates)
      narrow(candidate.loop, candidate.result, candidate.operand,
             candidate.yield, candidate.kept);
  }
};
} // namespace

} // namespace md_exec
} // namespace mdir
