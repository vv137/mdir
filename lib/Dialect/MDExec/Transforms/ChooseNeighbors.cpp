// The kind of each neighbor structure and the policy of the loops over it.
//
// See the description of the pass in Passes.td, and D89 in
// docs/decisions.md.

#include "mdir/Dialect/MDExec/Transforms/Passes.h"

#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::md_exec;

namespace mdir {
namespace md_exec {

#define GEN_PASS_DEF_CHOOSENEIGHBORS
#include "mdir/Dialect/MDExec/Transforms/Passes.h.inc"

namespace {
/// Adds to `beginnings` the ops where the structure `value` may begin.
/// Returns false if one cannot be found.
bool findBeginnings(Value value, llvm::SetVector<Operation *> &beginnings,
                    llvm::DenseSet<Value> &visited) {
  if (!visited.insert(value).second)
    return true;
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    auto loop = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
    if (!loop || argument.getArgNumber() == 0)
      return false;
    unsigned index = argument.getArgNumber() - 1;
    return findBeginnings(loop.getInitArgs()[index], beginnings, visited) &&
           findBeginnings(loop.getYieldedValues()[index], beginnings,
                          visited);
  }
  Operation *op = value.getDefiningOp();
  if (auto refresh = dyn_cast<RefreshNeighborsOp>(op))
    return findBeginnings(refresh.getNeighbors(), beginnings, visited);
  if (auto loop = dyn_cast<scf::ForOp>(op)) {
    unsigned index = cast<OpResult>(value).getResultNumber();
    return findBeginnings(loop.getYieldedValues()[index], beginnings,
                          visited);
  }
  if (isa<EmptyNeighborsOp, BuildNeighborsOp>(op)) {
    beginnings.insert(op);
    return true;
  }
  return false;
}

/// Returns true if every destination of `loop` is symmetric or
/// antisymmetric and every sum symmetric.
bool allowsUnique(PairForOp loop) {
  unsigned numOuts = loop.getOuts().size();
  for (unsigned i = 0, e = numOuts + loop.getReduce().size(); i != e; ++i) {
    md::Exchange exchange = loop.getExchange(i);
    if (exchange != md::Exchange::Symmetric &&
        !(i < numOuts && exchange == md::Exchange::Antisymmetric))
      return false;
  }
  return true;
}

class ChooseNeighbors : public impl::ChooseNeighborsBase<ChooseNeighbors> {
public:
  using impl::ChooseNeighborsBase<ChooseNeighbors>::ChooseNeighborsBase;

  void runOnOperation() final {
    if (kind == "matrix")
      return;
    if (kind != "groups") {
      getOperation()->emitError()
          << "expected the kind 'groups' or 'matrix', got '" << kind << "'";
      return signalPassFailure();
    }

    // The loops over each beginning, and the beginnings that must stay
    // matrices: those of a loop that cannot take each pair once, and those
    // that share a loop with one that must.
    llvm::MapVector<Operation *, SmallVector<PairForOp>> loops;
    llvm::DenseSet<Operation *> keep;
    SmallVector<std::pair<PairForOp, llvm::SetVector<Operation *>>> found;
    bool unknown = false;
    getOperation()->walk([&](PairForOp loop) {
      llvm::SetVector<Operation *> beginnings;
      llvm::DenseSet<Value> visited;
      if (!findBeginnings(loop.getNeighbors(), beginnings, visited)) {
        unknown = true;
        return;
      }
      found.push_back({loop, beginnings});
    });
    // The triplets are found in the rows of a matrix, which hold every
    // neighbor of a particle (D160).
    getOperation()->walk([&](BuildTripletsOp triplets) {
      llvm::DenseSet<Value> visited;
      llvm::SetVector<Operation *> beginnings;
      if (!findBeginnings(triplets.getNeighbors(), beginnings, visited))
        unknown = true;
      keep.insert(beginnings.begin(), beginnings.end());
    });
    // A loop over a structure whose beginnings are unknown may be over any
    // of them: all stay matrices.
    if (unknown)
      return;
    for (auto &[loop, beginnings] : found)
      for (Operation *beginning : beginnings) {
        loops[beginning].push_back(loop);
        if (!allowsUnique(loop))
          keep.insert(beginning);
      }
    // Every beginning of a loop must have the same kind.
    for (bool changed = true; changed;) {
      changed = false;
      for (auto &[loop, beginnings] : found)
        if (llvm::any_of(beginnings,
                         [&](Operation *op) { return keep.contains(op); }))
          for (Operation *beginning : beginnings)
            changed |= keep.insert(beginning).second;
    }

    MLIRContext *context = &getContext();
    auto groups = NeighborKindAttr::get(context, NeighborKind::Groups);
    for (auto &[beginning, over] : loops) {
      if (keep.contains(beginning))
        continue;
      if (auto empty = dyn_cast<EmptyNeighborsOp>(beginning))
        empty.setKindAttr(groups);
      else
        cast<BuildNeighborsOp>(beginning).setKindAttr(groups);
      for (PairForOp loop : over) {
        loop.setTraversal(Traversal::Unique);
        loop.setConflict(Conflict::Atomic);
      }
    }
  }
};
} // namespace

} // namespace md_exec
} // namespace mdir
