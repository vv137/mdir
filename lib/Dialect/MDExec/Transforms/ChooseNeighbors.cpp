// The kind of each neighbor structure and the policy of the loops over it.
//
// See the description of the pass in Passes.td, and D89 in
// docs/decisions.md.

#include "mdir/Dialect/MDExec/Transforms/Passes.h"

#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mdir/Dialect/MDExec/NeighborContract.h"
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

/// The values that are one structure: a refresh's operand and result; the
/// initial value, the iteration argument, the yielded value, and the result
/// of a loop that carries it; and the result of an scf.if with the values
/// that its branches yield. They have one type, which states the
/// guarantee of the structure (D[neighbors-contract]).
class Webs {
public:
  explicit Webs(Operation *root) {
    root->walk([&](Operation *op) {
      for (Value result : op->getResults())
        add(result);
      for (Region &region : op->getRegions())
        for (Block &block : region)
          for (BlockArgument argument : block.getArguments())
            add(argument);
    });
    for (Value value : values) {
      if (auto refresh = value.getDefiningOp<RefreshNeighborsOp>())
        join(value, refresh.getNeighbors());
      if (auto loop = value.getDefiningOp<scf::ForOp>()) {
        unsigned index = cast<OpResult>(value).getResultNumber();
        join(value, loop.getInitArgs()[index]);
        join(value, loop.getRegionIterArgs()[index]);
        join(value, loop.getYieldedValues()[index]);
      }
      // The two branches of an scf.if yield one type.
      if (auto branch = value.getDefiningOp<scf::IfOp>()) {
        unsigned index = cast<OpResult>(value).getResultNumber();
        join(value, branch.thenYield().getOperand(index));
        if (!branch.getElseRegion().empty())
          join(value, branch.elseYield().getOperand(index));
      }
    }
  }

  /// The representative of the structure of `value`.
  Value find(Value value) {
    Value parent = parents.lookup(value);
    if (!parent || parent == value)
      return value;
    Value root = find(parent);
    parents[value] = root;
    return root;
  }

  /// Returns true if `value` has a use whose type is fixed elsewhere: a
  /// call or a return, whose function's signature fixes it, or any op
  /// that does not carry, choose, refresh, or read a structure.
  static bool escapes(Value value) {
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      if (isa<PairForOp, ReferencePositionsOp, ReferenceCellOp,
              ResetNeighborsOp, RebuildCountOp, BuildTripletsOp,
              BuildEdgesOp>(user))
        continue;
      if (auto refresh = dyn_cast<RefreshNeighborsOp>(user))
        if (&use == &refresh.getNeighborsMutable())
          continue;
      if (auto loop = dyn_cast<scf::ForOp>(user))
        if (use.getOperandNumber() >= loop.getNumControlOperands())
          continue;
      if (isa<scf::YieldOp>(user) &&
          isa<scf::ForOp, scf::IfOp>(user->getParentOp()))
        continue;
      return true;
    }
    return false;
  }

  llvm::SetVector<Value> values;

private:
  void add(Value value) {
    if (isa<mdrt::NeighborsType>(value.getType()))
      values.insert(value);
  }
  void join(Value a, Value b) {
    Value ra = find(a), rb = find(b);
    if (ra != rb)
      parents[ra] = rb;
  }
  llvm::DenseMap<Value, Value> parents;
};

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
    // So are the pairs of images of a learned potential
    // (D277).
    getOperation()->walk([&](BuildEdgesOp edges) {
      llvm::DenseSet<Value> visited;
      llvm::SetVector<Operation *> beginnings;
      if (!findBeginnings(edges.getNeighbors(), beginnings, visited))
        unknown = true;
      keep.insert(beginnings.begin(), beginnings.end());
    });
    // A loop over a structure whose beginnings are unknown may be over any
    // of them: all stay matrices.
    if (unknown)
      return keepOneList([](Value) { return true; });
    for (auto &[loop, beginnings] : found)
      for (Operation *beginning : beginnings) {
        loops[beginning].push_back(loop);
        // The rows of a domain stay a matrix (D290): groups take
        // every row as a center, in the minimum image.
        bool domainRows = loop.getCenters() ||
                          loop.getImages() == Images::Explicit ||
                          loop.getConflict() == Conflict::Atomic;
        if (!allowsUnique(loop) || domainRows)
          keep.insert(beginning);
      }
    // The values of one structure have one type (D[neighbors-contract]):
    // the beginnings of a structure that escapes, or that has a beginning
    // no loop over pairs takes, stay matrices with the rest of it.
    Webs webs(getOperation());
    llvm::MapVector<Value, llvm::SetVector<Operation *>> beginningsOf;
    llvm::DenseSet<Value> escaping;
    for (Value value : webs.values) {
      Value web = webs.find(value);
      Operation *op = value.getDefiningOp();
      if (isa_and_nonnull<EmptyNeighborsOp, BuildNeighborsOp>(op))
        beginningsOf[web].insert(op);
      else
        beginningsOf[web];
      // A value made by any other op, or an argument of any other block
      // than the body of a loop that carries it, has a type that its op
      // or its block fixes.
      auto argument = dyn_cast<BlockArgument>(value);
      bool carried =
          argument && argument.getArgNumber() != 0 &&
          isa<scf::ForOp>(argument.getOwner()->getParentOp()) &&
          argument.getOwner() ==
              cast<scf::ForOp>(argument.getOwner()->getParentOp()).getBody();
      bool made = op && isa<EmptyNeighborsOp, BuildNeighborsOp,
                            RefreshNeighborsOp, scf::ForOp, scf::IfOp>(op);
      if (Webs::escapes(value) || !(carried || made))
        escaping.insert(web);
    }
    for (auto &[web, beginnings] : beginningsOf)
      if (escaping.contains(web) ||
          llvm::any_of(beginnings,
                       [&](Operation *op) { return !loops.count(op); }))
        keep.insert(beginnings.begin(), beginnings.end());
    // Every beginning of a loop, and of a structure, must have the same
    // kind.
    for (bool changed = true; changed;) {
      changed = false;
      for (auto &[loop, beginnings] : found)
        if (llvm::any_of(beginnings,
                         [&](Operation *op) { return keep.contains(op); }))
          for (Operation *beginning : beginnings)
            changed |= keep.insert(beginning).second;
      for (auto &[web, beginnings] : beginningsOf)
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
    // The values of each structure of groups take its type.
    for (Value value : webs.values) {
      auto beginnings = beginningsOf.find(webs.find(value));
      if (beginnings == beginningsOf.end() || beginnings->second.empty() ||
          keep.contains(beginnings->second.front()) ||
          !loops.count(beginnings->second.front()))
        continue;
      auto type = cast<mdrt::NeighborsType>(value.getType());
      value.setType(getBuildType(type.getParticleSet(), NeighborKind::Groups,
                                 Images::Minimum));
    }
    keepOneList([&](Value structure) {
      llvm::SetVector<Operation *> beginnings;
      llvm::DenseSet<Value> visited;
      return !findBeginnings(structure, beginnings, visited) ||
             llvm::any_of(beginnings, [&](Operation *op) {
               return keep.contains(op) || !loops.count(op);
             });
    });
  }

  /// A structure that stays a matrix keeps one list: only the groups have
  /// an inner list to prune (D114). Its refreshes lose the reach and the
  /// test of the inner list that `md-exec-reuse-neighbors` and
  /// `md-exec-expose-validity` gave every refresh of the run, and what the
  /// test read of the pruning reads the build instead, so that a test
  /// which a fused loop still computes is of the one list
  /// (D245: the loops of the derivative in the tunables keep
  /// the matrix beside the groups of the forces).
  void keepOneList(llvm::function_ref<bool(Value)> staysMatrix) {
    getOperation()->walk([&](Operation *op) {
      if (auto refresh = dyn_cast<RefreshNeighborsOp>(op)) {
        if (!refresh.getPruneSkin() || !staysMatrix(refresh.getNeighbors()))
          return;
        refresh.getStaleMutable().clear();
        refresh.removePruneSkinAttr();
      } else if (auto positions = dyn_cast<ReferencePositionsOp>(op)) {
        if (positions.getPruned() && staysMatrix(positions.getNeighbors()))
          positions.setPruned(false);
      } else if (auto cell = dyn_cast<ReferenceCellOp>(op)) {
        if (cell.getPruned() && staysMatrix(cell.getNeighbors()))
          cell.setPruned(false);
      }
    });
  }
};
} // namespace

} // namespace md_exec
} // namespace mdir
