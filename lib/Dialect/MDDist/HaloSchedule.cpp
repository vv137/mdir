// A reusable proof of start and completion edges for semantic halo transfers
// (D308, docs/distributed-ml2.md, Section 5).

#include "mdir/Dialect/MDDist/MDDistOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mdir::md_dist;

FailureOr<CenterPartitionRecord>
mdir::md_dist::proveCenterPartition(PartitionCentersOp op) {
  auto relation = op.getRelation().getDefiningOp<NeighborhoodOp>();
  if (!relation) {
    op.emitOpError("expected a direct md_dist.neighborhood relation");
    return failure();
  }
  if (relation->getBlock() != op->getBlock() ||
      !relation->isBeforeInBlock(op)) {
    op.emitOpError(
        "cannot prove relation in the same block before partition");
    return failure();
  }
  if (relation.getHalo() != op.getHalo() ||
      layoutOf(relation.getHalo()) != op.getLayout()) {
    op.emitOpError(
        "relation, halo, and layout must have the same ownership epoch");
    return failure();
  }
  auto layout = op.getLayout().getDefiningOp<PartitionOp>();
  auto halo = op.getHalo().getDefiningOp<HaloOp>();
  if (!layout || !halo || layout->getBlock() != op->getBlock() ||
      halo->getBlock() != op->getBlock() ||
      !layout->isBeforeInBlock(op) || !halo->isBeforeInBlock(op)) {
    op.emitOpError("cannot prove layout and halo in the same block");
    return failure();
  }
  auto copy = relation.getGhosts().getDefiningOp<HaloCopyOp>();
  Operation *owned = relation.getPositions().getDefiningOp();
  if (!copy || !owned || copy->getBlock() != op->getBlock() ||
      owned->getBlock() != op->getBlock() ||
      !copy->isBeforeInBlock(relation) ||
      !owned->isBeforeInBlock(relation)) {
    op.emitOpError("cannot prove completed relation positions in the same block");
    return failure();
  }
  auto validIndicator = [&](FieldType type) {
    return type.isOwned() && !type.getPartial() &&
           type.getDecomposition() ==
               op.getRelation().getType().getDecomposition() &&
           type.getNumComponents() == 1 &&
           type.getElementType().isSignlessInteger(32);
  };
  if (!validIndicator(op.getInterior().getType()) ||
      !validIndicator(op.getBoundary().getType())) {
    op.emitOpError("expected complete owned scalar i32 indicators of the "
                   "relation decomposition");
    return failure();
  }
  for (Value result : op->getResults())
    for (Operation *user : result.getUsers())
      if (user->getBlock() != op->getBlock()) {
        op.emitOpError("cannot prove partition use in the same block");
        return failure();
      }
  return CenterPartitionRecord{relation, op.getLayout(), op.getHalo(),
                               op.getInterior(), op.getBoundary()};
}

LogicalResult mdir::md_dist::proveHaloSchedule(
    Block &block, SmallVectorImpl<HaloScheduleRecord> &records) {
  records.clear();
  llvm::DenseMap<Operation *, unsigned> order;
  unsigned next = 0;
  for (Operation &op : block)
    order[&op] = next++;

  for (Operation &op : block) {
    auto copy = dyn_cast<HaloCopyOp>(&op);
    auto sum = dyn_cast<HaloSumOp>(&op);
    if (!copy && !sum)
      continue;
    // These are also the checks used before the synchronous lowering.
    if ((copy && failed(copy.verify())) || (sum && failed(sum.verify())))
      return failure();

    HaloScheduleRecord record;
    record.transfer = &op;
    record.halo = copy ? copy.getHalo() : sum.getHalo();
    record.layout = layoutOf(record.halo);
    record.result = copy ? copy.getResult() : sum.getResult();
    if (!record.layout)
      return op.emitOpError("cannot prove the halo ownership epoch");
    if (copy) {
      record.sources.push_back(copy.getField());
      if (copy.getImage())
        record.sources.push_back(copy.getImage());
    } else {
      record.sources.push_back(sum.getOwned());
      record.sources.push_back(sum.getGhosts());
    }
    record.sources.push_back(record.halo);

    llvm::SmallDenseSet<Operation *, 4> seen;
    for (Value source : record.sources) {
      Operation *producer = source.getDefiningOp();
      if (!producer) {
        auto argument = dyn_cast<BlockArgument>(source);
        if (!argument || argument.getOwner() != &block)
          return op.emitOpError(
              "cannot prove a start prerequisite in the same block");
        record.readyAtEntry.push_back(source);
        continue;
      }
      if (producer->getBlock() != &block ||
          order.lookup(producer) >= order.lookup(&op))
        return op.emitOpError(
            "cannot prove a start prerequisite in the same block");
      if (seen.insert(producer).second)
        record.startAfter.push_back(producer);
    }

    llvm::SmallDenseSet<Operation *, 4> seenUsers;
    for (Operation *user : record.result.getUsers()) {
      if (!seenUsers.insert(user).second)
        continue;
      if (user->getBlock() != &block ||
          order.lookup(user) <= order.lookup(&op))
        return op.emitOpError(
            "cannot prove completion before a result use in the same block");
      record.completeBefore.push_back(user);
    }
    llvm::sort(record.startAfter, [&](Operation *a, Operation *b) {
      return order.lookup(a) < order.lookup(b);
    });
    llvm::sort(record.completeBefore, [&](Operation *a, Operation *b) {
      return order.lookup(a) < order.lookup(b);
    });
    if (!record.completeBefore.empty())
      record.firstCompletion = record.completeBefore.front();

    unsigned end = record.completeBefore.empty()
                       ? next
                       : order.lookup(record.completeBefore.front());
    for (Operation &candidate : block) {
      unsigned position = order.lookup(&candidate);
      if (position <= order.lookup(&op) || position >= end)
        continue;
      if (!isa<MapParticlesOp, SumParticlesOp>(&candidate))
        continue;
      bool ownedOnly = true;
      for (Value operand : candidate.getOperands()) {
        FieldType field = dyn_cast<FieldType>(operand.getType());
        if (field && !field.isOwned())
          ownedOnly = false;
      }
      if (ownedOnly)
        record.independentOwnedWork.push_back(&candidate);
    }
    records.push_back(std::move(record));
  }
  return success();
}
