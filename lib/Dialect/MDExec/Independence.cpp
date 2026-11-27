// Independence of ops in the storage form of md_exec.
//
// See Independence.h, and D87 in docs/decisions.md for the argument that
// running independent ops at once is correct.

#include "mdir/Dialect/MDAlg/MDAlgOps.h"
#include "mdir/Dialect/MDExec/Independence.h"

#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MD/MDTypes.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::md_exec;

/// Returns true if `value` is an argument of the body of a function.
static bool isFunctionArgument(Value value) {
  auto argument = dyn_cast<BlockArgument>(value);
  return argument && argument.getOwner()->isEntryBlock() &&
         isa<FunctionOpInterface>(argument.getOwner()->getParentOp());
}

/// Returns true if `value` is the result of an op that allocates it.
static bool isAllocated(Value value) {
  auto result = dyn_cast<OpResult>(value);
  if (!result)
    return false;
  auto effects = dyn_cast<MemoryEffectOpInterface>(result.getOwner());
  return effects &&
         effects.getEffectOnValue<MemoryEffects::Allocate>(value).has_value();
}

std::optional<SmallVector<unsigned>>
BufferAliases::getPermutation(Operation *loop) {
  auto like = cast<LoopLikeOpInterface>(loop);
  Block::BlockArgListType arguments = like.getRegionIterArgs();
  std::optional<MutableArrayRef<OpOperand>> yielded =
      like.getYieldedValuesMutable();
  if (!yielded || yielded->size() != arguments.size())
    return std::nullopt;
  SmallVector<unsigned> permutation(arguments.size());
  SmallVector<bool> taken(arguments.size(), false);
  for (auto [i, operand] : llvm::enumerate(*yielded)) {
    auto argument = dyn_cast<BlockArgument>(resolveLoopResult(operand.get()));
    if (!argument || arguments.empty() ||
        argument.getOwner() != arguments.front().getOwner())
      return std::nullopt;
    unsigned j = argument.getArgNumber() - arguments.front().getArgNumber();
    if (j >= arguments.size() || taken[j])
      return std::nullopt;
    taken[j] = true;
    permutation[i] = j;
  }
  return permutation;
}

// A nested loop can yield a permutation to its enclosing loop. After a
// known trip count its result is precisely one initial operand. Reduce the
// count modulo the length of the orbit, including for zero iterations.
Value BufferAliases::resolveLoopResult(Value value) {
  auto result = dyn_cast<OpResult>(value);
  auto loop = result ? dyn_cast<scf::ForOp>(result.getOwner()) : scf::ForOp();
  if (!loop)
    return value;
  auto found = resolvedResults.find(value);
  if (found != resolvedResults.end())
    return found->second;
  Value resolved = value;
  if (auto count = loop.getStaticTripCount()) {
    if (auto permutation = getPermutation(loop)) {
      unsigned start = result.getResultNumber();
      SmallVector<unsigned> orbit;
      unsigned index = start;
      do {
        orbit.push_back(index);
        index = (*permutation)[index];
      } while (index != start);
      unsigned source = orbit[count->urem(orbit.size())];
      resolved = resolveLoopResult(loop.getInitArgs()[source]);
    }
  }
  return resolvedResults[value] = resolved;
}

std::optional<BufferAliases::Roots> BufferAliases::computeRoots(Value value) {
  Value resolved = resolveLoopResult(value);
  if (resolved != value)
    return getRoots(resolved);
  if (isFunctionArgument(value) || isAllocated(value) ||
      value.getDefiningOp<memref::GetGlobalOp>()) {
    Roots single;
    single.insert(value);
    return single;
  }

  // An argument that a loop carries, or a result of the loop, which is the
  // argument after the last iteration.
  Operation *loop = nullptr;
  unsigned index = 0;
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    auto like = dyn_cast<LoopLikeOpInterface>(
        argument.getOwner()->getParentOp());
    if (!like)
      return std::nullopt;
    Block::BlockArgListType carried = like.getRegionIterArgs();
    if (carried.empty() || argument.getOwner() != carried.front().getOwner() ||
        argument.getArgNumber() < carried.front().getArgNumber())
      return std::nullopt;
    loop = like;
    index = argument.getArgNumber() - carried.front().getArgNumber();
  } else {
    auto result = cast<OpResult>(value);
    Operation *owner = result.getOwner();
    if (auto view = dyn_cast<ViewLikeOpInterface>(owner))
      return getRoots(view.getViewSource());
    if (auto refresh = dyn_cast<RefreshNeighborsOp>(owner))
      return getRoots(refresh.getNeighbors());
    if (auto reference = dyn_cast<ReferencePositionsOp>(owner))
      return getRoots(reference.getNeighbors());
    auto like = dyn_cast<LoopLikeOpInterface>(owner);
    if (!like || !like.getLoopResults())
      return std::nullopt;
    loop = like;
    index = result.getResultNumber();
  }
  auto like = cast<LoopLikeOpInterface>(loop);
  if (index >= like.getInitsMutable().size())
    return std::nullopt;

  // The values that the argument takes: the initial values of the arguments
  // in its orbit under the permutation of the iterations.
  std::optional<SmallVector<unsigned>> permutation = getPermutation(loop);
  if (!permutation)
    return std::nullopt;
  Roots all;
  unsigned k = index;
  do {
    const std::optional<Roots> &part =
        getRoots(like.getInitsMutable()[k].get());
    if (!part)
      return std::nullopt;
    all.insert(part->begin(), part->end());
    k = (*permutation)[k];
  } while (k != index);
  return all;
}

const std::optional<BufferAliases::Roots> &
BufferAliases::getRoots(Value value) {
  auto found = roots.find(value);
  if (found != roots.end())
    return found->second;
  static const std::optional<Roots> unknown;
  // A cycle that does not pass through a loop that permutes: not traced.
  if (!visiting.insert(value).second)
    return unknown;
  std::optional<Roots> computed = computeRoots(value);
  // A neighbor structure keeps the buffers of its excluded pairs, which the
  // ops that read it read.
  if (computed && isa<mdrt::NeighborsType>(value.getType())) {
    SmallVector<Value> structures;
    for (Value root : *computed)
      if (isa<mdrt::NeighborsType>(root.getType()))
        structures.push_back(root);
    for (Value structure : structures) {
      SmallVector<Value> forms = {structure};
      // The forms of the structure, through its refreshes. A form that
      // another op takes, a loop that carries it for one, may be reset
      // where this does not see it: its roots are not known.
      for (unsigned i = 0; i != forms.size() && computed; ++i)
        for (Operation *user : forms[i].getUsers()) {
          if (auto refresh = dyn_cast<RefreshNeighborsOp>(user)) {
            if (refresh.getNeighbors() == forms[i])
              forms.push_back(refresh.getResult());
            continue;
          }
          if (auto reset = dyn_cast<ResetNeighborsOp>(user)) {
            if (Value excluded = reset.getExcluded()) {
              const std::optional<Roots> &kept = getRoots(excluded);
              if (!kept) {
                computed.reset();
                break;
              }
              computed->insert(kept->begin(), kept->end());
            }
            continue;
          }
          // md_alg.traverse_pairs reads a structure as the loop over pairs
          // it was made from (D415).
          if (!isa<PairForOp, BuildTripletsOp, BuildEdgesOp,
                   ReferencePositionsOp, ReferenceCellOp, RebuildCountOp,
                   md_alg::TraversePairsOp>(user)) {
            computed.reset();
            break;
          }
        }
      if (!computed)
        break;
    }
  }
  visiting.erase(value);
  return roots[value] = std::move(computed);
}

bool BufferAliases::mayAlias(Value a, Value b) {
  a = resolveLoopResult(a);
  b = resolveLoopResult(b);
  if (a == b)
    return true;

  // Two arguments carried by one loop that permutes them are distinct at
  // every iteration when the initial values they may take at once are.
  auto getCarried = [](Value value) -> std::pair<Operation *, unsigned> {
    if (auto argument = dyn_cast<BlockArgument>(value))
      if (auto like = dyn_cast<LoopLikeOpInterface>(
              argument.getOwner()->getParentOp())) {
        Block::BlockArgListType carried = like.getRegionIterArgs();
        if (!carried.empty() &&
            argument.getOwner() == carried.front().getOwner() &&
            argument.getArgNumber() >= carried.front().getArgNumber())
          return {like,
                  argument.getArgNumber() - carried.front().getArgNumber()};
      }
    return {nullptr, 0};
  };
  auto [loopA, i] = getCarried(a);
  auto [loopB, j] = getCarried(b);
  if (loopA && loopA == loopB && i != j) {
    auto like = cast<LoopLikeOpInterface>(loopA);
    if (std::optional<SmallVector<unsigned>> permutation =
            getPermutation(loopA)) {
      // After k iterations the two are the initial values at pi^k(i) and
      // pi^k(j), which differ: every pair of distinct places in the orbits
      // of i and of j (the orbits may differ in length, so pairs further
      // than one turn of the shorter matter too).
      auto getOrbit = [&](unsigned start) {
        SmallVector<unsigned> orbit;
        unsigned k = start;
        do {
          orbit.push_back(k);
          k = (*permutation)[k];
        } while (k != start);
        return orbit;
      };
      bool distinct = true;
      for (unsigned m : getOrbit(i))
        for (unsigned n : getOrbit(j))
          distinct &= m == n || !mayAlias(like.getInitsMutable()[m].get(),
                                          like.getInitsMutable()[n].get());
      if (distinct)
        return false;
    }
  }

  const std::optional<Roots> rootsA = getRoots(a);
  const std::optional<Roots> &rootsB = getRoots(b);
  if (!rootsA || !rootsB)
    return true;
  for (Value root : *rootsA)
    if (rootsB->contains(root))
      return true;
  // Global tables in a pair body cannot alias a live allocation. Two
  // handles to external storage (arguments or globals) may coincide.
  auto external = [](Value v) { return !isAllocated(v); };
  return llvm::any_of(*rootsA, external) && llvm::any_of(*rootsB, external);
}

/// Returns true if `op` or an op inside it has a value in the value form,
/// whose effects on memory are not declared.
static bool hasValueForm(Operation *op) {
  bool found = false;
  op->walk([&](Operation *nested) {
    for (Type type : llvm::concat<Type>(nested->getOperandTypes(),
                                        nested->getResultTypes()))
      found |= isValueFormType(type);
  });
  return found;
}

/// Returns true if an op in `inner` (or `inner` itself) uses a result of
/// `outer`, or of an op inside it.
static bool usesResultOf(Operation *inner, Operation *outer) {
  bool uses = false;
  inner->walk([&](Operation *nested) {
    for (Value operand : nested->getOperands())
      if (Operation *definition = operand.getDefiningOp())
        uses |= outer->isAncestor(definition);
  });
  return uses;
}

// A table lookup is pure in the value form. In the storage form its
// captured operand is a buffer read, just like a vector.load of a table.
static auto storageEffects(Operation *op) {
  auto effects = getEffectsRecursively(op);
  if (effects)
    op->walk([&](md::LookupOp lookup) {
      if (isa<MemRefType>(lookup.getTable().getType()))
        effects->emplace_back(MemoryEffects::Read::get(),
                              &lookup.getTableMutable());
    });
  return effects;
}

bool md_exec::areIndependent(Operation *a, Operation *b,
                             BufferAliases &aliases) {
  if (usesResultOf(a, b) || usesResultOf(b, a))
    return false;
  if (hasValueForm(a) || hasValueForm(b))
    return false;
  std::optional<SmallVector<MemoryEffects::EffectInstance>> effectsA =
      storageEffects(a);
  std::optional<SmallVector<MemoryEffects::EffectInstance>> effectsB =
      storageEffects(b);
  if (!effectsA || !effectsB)
    return false;
  auto changes = [](const MemoryEffects::EffectInstance &effect) {
    return isa<MemoryEffects::Write, MemoryEffects::Free>(
        effect.getEffect());
  };
  for (const MemoryEffects::EffectInstance &x : *effectsA)
    for (const MemoryEffects::EffectInstance &y : *effectsB) {
      // An allocation gives new memory, which the other op can reach only
      // through the value it gives.
      if (isa<MemoryEffects::Allocate>(x.getEffect()) ||
          isa<MemoryEffects::Allocate>(y.getEffect()))
        continue;
      if (!changes(x) && !changes(y))
        continue;
      if (x.getResource() != y.getResource())
        continue;
      Value p = x.getValue(), q = y.getValue();
      if (!p || !q || aliases.mayAlias(p, q))
        return false;
    }
  return true;
}

unsigned md_exec::getSideLane(Operation *op) {
  if (!op->hasAttr(kSideAttrName))
    return 0;
  if (auto lane = op->getAttrOfType<IntegerAttr>(kLaneAttrName))
    return static_cast<unsigned>(lane.getInt());
  return 1;
}

Operation *md_exec::findJoin(Operation *side, BufferAliases &aliases) {
  Operation *next = side->getNextNode();
  for (; next && !next->hasTrait<OpTrait::IsTerminator>();
       next = next->getNextNode()) {
    // An op marked for the same side stream runs after it there, in the
    // order of the program (D[concurrent-terms]).
    if (getSideLane(next) == getSideLane(side))
      continue;
    if (isa<JoinOp>(next) || !areIndependent(side, next, aliases))
      break;
  }
  return next;
}

LogicalResult md_exec::verifySide(Operation *side, BufferAliases &aliases) {
  for (Operation *next = side->getNextNode(); next;
       next = next->getNextNode()) {
    if (isa<JoinOp>(next))
      return success();
    // Another op marked for the same side stream runs after this one
    // there; the ops of the other streams after it must still be
    // independent of this one, which may be running then
    // (D[concurrent-terms]).
    if (getSideLane(next) == getSideLane(side))
      continue;
    if (!areIndependent(side, next, aliases)) {
      InFlightDiagnostic diagnostic =
          side->emitOpError() << "runs on a second stream beside an op that "
                                 "is not independent of it, before a join "
                                 "(D87)";
      diagnostic.attachNote(next->getLoc()) << "the op";
      return diagnostic;
    }
  }
  return side->emitOpError()
         << "runs on a second stream, but no join follows it in its block";
}
