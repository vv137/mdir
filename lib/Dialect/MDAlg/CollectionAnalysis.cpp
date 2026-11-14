// The shared analysis of the collections of md_alg (D409);
// see include/mdir/Dialect/MDAlg/CollectionAnalysis.h.
//
// Stage A works on memrefs: a producer writes a buffer, and its consumers
// read the buffer, or a view of it, wherever they stand (in the host's
// code or in the body of a launch). The flow is through memory, not from
// the uses of SSA values to their definitions, so the analysis is a walk
// from each producer over the aliases of its buffer, each use classified
// by the effects its op declares (MemoryEffectOpInterface). The order
// requirement is propagated from the consumers back to the producer
// [Steuwer et al. 2017; Henriksen et al. 2017]. Every use that the
// analysis cannot classify requires the full layout.

#include "mdir/Dialect/MDAlg/CollectionAnalysis.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mdir::md_alg;

//===----------------------------------------------------------------------===//
// The lattice
//===----------------------------------------------------------------------===//

bool OrderSpec::isMetBy(const OrderSpec &other) const {
  if (kind == Kind::None || other.kind == Kind::Layout)
    return true;
  if (kind == Kind::Layout || other.kind == Kind::None)
    return false;
  if (kind == Kind::Membership)
    return true;
  if (kind > other.kind)
    return false;
  // Both name a group (and, from sorted_within on, a key): they must be the
  // same group, and the same key.
  if (group != other.group)
    return false;
  if (kind >= Kind::SortedWithin && key != other.key)
    return false;
  return true;
}

OrderSpec OrderSpec::join(const OrderSpec &a, const OrderSpec &b) {
  if (a.isMetBy(b))
    return b;
  if (b.isMetBy(a))
    return a;
  return layout();
}

static std::string name(Value v, AsmState &state) {
  if (!v)
    return "item";
  std::string text;
  llvm::raw_string_ostream os(text);
  v.printAsOperand(os, state);
  return text;
}

void OrderSpec::print(llvm::raw_ostream &os, AsmState &state) const {
  switch (kind) {
  case Kind::None:
    os << "none";
    return;
  case Kind::Membership:
    os << "membership";
    return;
  case Kind::GroupedBy:
    os << "grouped_by(" << name(group, state) << ")";
    return;
  case Kind::SortedWithin:
    os << "sorted_within(" << name(group, state) << ", " << name(key, state)
       << ")";
    return;
  case Kind::Stable:
    os << "stable(" << name(group, state) << ", " << name(key, state) << ")";
    return;
  case Kind::Layout:
    os << "layout";
    return;
  }
}

//===----------------------------------------------------------------------===//
// Roots and aliases
//===----------------------------------------------------------------------===//

/// The views that the analysis follows to their source: the same elements
/// under another type or memory space, or a part of them. Any other op
/// that takes a buffer and gives a value is an escape.
static bool isFollowedView(Operation *op) {
  return isa<memref::MemorySpaceCastOp, memref::CastOp, memref::SubViewOp,
             memref::ReinterpretCastOp>(op);
}

/// The buffer that `v` views, through the followed views.
static Value rootOf(Value v) {
  while (Operation *def = v.getDefiningOp()) {
    if (!isFollowedView(def))
      break;
    v = def->getOperand(0);
  }
  return v;
}

static bool hasEffectOn(Operation *op, Value v, bool (*match)(
                                                    MemoryEffects::Effect *)) {
  auto effects = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effects)
    return false;
  SmallVector<MemoryEffects::EffectInstance> list;
  effects.getEffectsOnValue(v, list);
  return llvm::any_of(list, [&](MemoryEffects::EffectInstance &e) {
    return match(e.getEffect());
  });
}

static bool isAllocate(MemoryEffects::Effect *e) {
  return isa<MemoryEffects::Allocate>(e);
}
static bool isFree(MemoryEffects::Effect *e) {
  return isa<MemoryEffects::Free>(e);
}
static bool isRead(MemoryEffects::Effect *e) {
  return isa<MemoryEffects::Read>(e);
}
static bool isWrite(MemoryEffects::Effect *e) {
  return isa<MemoryEffects::Write>(e);
}

//===----------------------------------------------------------------------===//
// Producers
//===----------------------------------------------------------------------===//

namespace {
struct Produced {
  Value buffer;
  std::string role;
  OrderSpec guarantee;
  Cardinality cardinality;
};
} // namespace

static Cardinality runTime(Value v, int64_t offset = 0) {
  Cardinality c;
  c.kind = Cardinality::Kind::RunTime;
  c.value = v;
  c.offset = offset;
  return c;
}

static Cardinality bound(Value v) {
  Cardinality c;
  c.kind = Cardinality::Kind::Bound;
  c.value = v;
  return c;
}

/// Exact where `c` is of a constant, or where the buffer has a static
/// size; as it is otherwise.
static Cardinality settle(Cardinality c, Value buffer) {
  IntegerAttr constant;
  if (c.value && c.kind != Cardinality::Kind::Bound &&
      matchPattern(c.value, m_Constant(&constant))) {
    c.kind = Cardinality::Kind::Exact;
    c.exact = constant.getInt() + c.offset;
    return c;
  }
  if (!c.value)
    if (auto type = dyn_cast<MemRefType>(buffer.getType());
        type && type.hasStaticShape()) {
      c.kind = Cardinality::Kind::Exact;
      c.exact = type.getNumElements();
    }
  return c;
}

/// Whether every item of the counting sort `op` has a bin.
static bool everyKey(SortByKeyOp op) {
  if (op.getKey().empty())
    return op.getEveryKey();
  bool none = false;
  op.getKey().walk([&](NoBinOp) { none = true; });
  return !none;
}

/// The collections that `op` produces. `groupOfCursor` gives the root of
/// the offsets whose cursor a scatter takes (a scan's start), if known.
static SmallVector<Produced>
producedBy(Operation *op, function_ref<Value(Value)> groupOfCursor) {
  SmallVector<Produced> out;
  auto add = [&](Value buffer, StringRef role, OrderSpec guarantee,
                 Cardinality c) {
    if (buffer)
      out.push_back({buffer, role.str(), guarantee, settle(c, buffer)});
  };
  OrderSpec layout = OrderSpec::layout();
  TypeSwitch<Operation *>(op)
      .Case([&](ScanOp scan) {
        add(scan.getStart(), "start", layout, runTime(scan.getN(), 1));
        add(scan.getCursor(), "cursor", layout, runTime(scan.getN()));
      })
      .Case([&](ScatterToBinsOp scatter) {
        Value cursor = rootOf(scatter.getCursor());
        Value group = groupOfCursor(cursor);
        add(scatter.getOut(), "out",
            OrderSpec::groupedBy(group ? group : cursor),
            scatter.getEveryKey() ? runTime(scatter.getN())
                                  : bound(scatter.getN()));
        add(scatter.getCursor(), "cursor", layout, runTime(Value()));
      })
      .Case([&](SortBinsOp sort) {
        Value by = sort.getBy() ? rootOf(sort.getBy()) : Value();
        Value result = sort.getOut() ? sort.getOut() : sort.getItems();
        add(result, sort.getOut() ? "out" : "items",
            OrderSpec::stable(rootOf(sort.getStart()), by), runTime(Value()));
      })
      .Case([&](SortByKeyOp sort) {
        Value n = sort.getN(), bins = sort.getBins();
        Value group = rootOf(sort.getStart());
        Value by = sort.getBy() ? rootOf(sort.getBy()) : Value();
        if (!sort.getKey().empty())
          add(sort.getKeys(), "keys", layout, runTime(n));
        add(sort.getCounts(), "counts", layout, runTime(bins));
        add(sort.getStart(), "start", layout, runTime(bins, 1));
        add(sort.getCursor(), "cursor", layout, runTime(bins));
        add(sort.getOut(), "out",
            sort.getOrder() == Order::Stable ? OrderSpec::stable(group, by)
                                             : OrderSpec::groupedBy(group),
            everyKey(sort) ? runTime(n) : bound(n));
        for (Value w : sort.getWrites())
          add(w, "writes", layout, runTime(Value()));
        for (Value a : sort.getAdds())
          add(a, "adds", layout, runTime(Value()));
      })
      .Case([&](AppendOp append) {
        for (Value into : append.getInto())
          add(into, "into", OrderSpec::membership(), runTime(Value()));
        add(append.getUnits(), "units", OrderSpec::membership(),
            bound(append.getCapacity()));
        add(append.getOrdinals(), "ordinals", OrderSpec::membership(),
            bound(append.getCapacity()));
        add(append.getPool(), "pool", layout, runTime(Value()));
      })
      .Case([&](BinMembersOp members) {
        for (Value w : members.getWrites())
          add(w, "writes", layout, runTime(Value()));
        for (Value a : members.getAdds())
          add(a, "adds", layout, runTime(Value()));
      });
  return out;
}

//===----------------------------------------------------------------------===//
// Consumers
//===----------------------------------------------------------------------===//

/// The use of `operand`, a view of the collection's root, by its owner.
static CollectionUse classify(OpOperand &operand) {
  Operation *op = operand.getOwner();
  Value v = operand.get();
  CollectionUse use;
  use.op = op;
  auto layout = [&](CollectionUse::Role role, Access access, StringRef why) {
    use.role = role;
    use.required = OrderSpec::layout();
    use.access = access;
    use.why = why.str();
    return use;
  };
  auto declaredWrite = [&](StringRef why) {
    use.role = CollectionUse::Role::Write;
    use.required = OrderSpec::none();
    use.exactOrUnordered = true;
    use.why = why.str();
    return use;
  };
  // The readers that state what they observe.
  if (auto members = dyn_cast<BinMembersOp>(op)) {
    if (&operand == &members.getItemsMutable()) {
      Value group = rootOf(members.getStart());
      Value by = members.getBy() ? rootOf(members.getBy()) : Value();
      use.role = CollectionUse::Role::Read;
      use.access = Access::Streamed;
      use.exactOrUnordered = members.combinesExactly() ||
                             cast<DeterminismInterface>(members.getOperation()).getRequiredDeterminism() ==
                                 Determinism::Unordered;
      switch (members.getVisit()) {
      case Visit::Unordered:
        use.required = OrderSpec::groupedBy(group);
        use.why = "visits the bins in any order";
        break;
      case Visit::Sorted:
        use.required = OrderSpec::sortedWithin(group, by);
        use.why = "visits the bins sorted";
        break;
      case Visit::Stable:
        use.required = OrderSpec::stable(group, by);
        use.why = "visits the bins stably";
        break;
      }
      if (!use.exactOrUnordered)
        use.why += ", combines floats under '" +
                   getDeterminismName(cast<DeterminismInterface>(members.getOperation()).getRequiredDeterminism()).str() +
                   "'";
      return use;
    }
    if (llvm::is_contained(members.getWrites(), v) ||
        llvm::is_contained(members.getAdds(), v))
      return declaredWrite("a declared side output");
    use.exactOrUnordered = true;
    return layout(CollectionUse::Role::Read, Access::Gathered,
                  "reads at a place");
  }
  if (auto sort = dyn_cast<SortBinsOp>(op)) {
    if (&operand == &sort.getItemsMutable()) {
      use.exactOrUnordered = true;
      if (sort.getBy())
        return layout(CollectionUse::Role::Read, Access::Streamed,
                      "sorts by 'by', ties in the order it reads");
      use.role = CollectionUse::Role::Read;
      use.access = Access::Streamed;
      use.required = OrderSpec::groupedBy(rootOf(sort.getStart()));
      use.why = "orders each bin itself";
      return use;
    }
    if (sort.getOut() && &operand == &sort.getOutMutable()[0])
      return declaredWrite("a declared output");
  }
  if (auto sort = dyn_cast<SortByKeyOp>(op)) {
    bool output =
        v == sort.getStart() || v == sort.getCursor() || v == sort.getOut() ||
        (!sort.getKey().empty() && v == sort.getKeys()) ||
        (!sort.getCleared() && v == sort.getCounts()) ||
        v == sort.getSortBuffer() || llvm::is_contained(sort.getWrites(), v) ||
        llvm::is_contained(sort.getAdds(), v);
    if (output)
      return declaredWrite("a declared output");
  }
  if (isa_and_nonnull<MDAlgDialect>(op->getDialect())) {
    // Every other operand of an md_alg op: read at the places its meaning
    // names, exactly, or written as its declared output.
    use.exactOrUnordered = true;
    bool reads = hasEffectOn(op, v, isRead);
    if (!reads && hasEffectOn(op, v, isWrite))
      return declaredWrite("a declared output");
    bool streamed = isa<ScanOp>(op) ||
                    (isa<ScatterToBinsOp>(op) &&
                     v == cast<ScatterToBinsOp>(op).getKeys()) ||
                    (isa<SortBinsOp>(op) &&
                     v == cast<SortBinsOp>(op).getStart()) ||
                    (isa<SortByKeyOp>(op) &&
                     v == cast<SortByKeyOp>(op).getKeys());
    return layout(CollectionUse::Role::Read,
                  streamed ? Access::Streamed : Access::Gathered,
                  isa<BinItemOp>(op) ? "reads the item at a place"
                                     : "reads it by index");
  }
  // Every other op: the conservative rules.
  if (isa<memref::DimOp>(op)) {
    use.role = CollectionUse::Role::Shape;
    use.required = OrderSpec::none();
    use.exactOrUnordered = true;
    use.why = "its shape";
    return use;
  }
  if (hasEffectOn(op, v, isFree) && op->getNumResults() == 0) {
    use.role = CollectionUse::Role::Release;
    use.required = OrderSpec::none();
    use.exactOrUnordered = true;
    use.why = "releases it";
    return use;
  }
  if (isa<CallOpInterface>(op))
    return layout(CollectionUse::Role::Escape, Access::Random,
                  "passed to a call");
  if (op->hasTrait<OpTrait::ReturnLike>())
    return layout(CollectionUse::Role::Escape, Access::Random, "returned");
  if (auto store = dyn_cast<memref::StoreOp>(op);
      store && store.getValueToStore() == v)
    return layout(CollectionUse::Role::Escape, Access::Random, "stored");
  if (isa<memref::ExtractAlignedPointerAsIndexOp,
          memref::ExtractStridedMetadataOp>(op))
    return layout(CollectionUse::Role::Escape, Access::Random,
                  "its address taken");
  auto effects = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effects)
    return layout(CollectionUse::Role::Escape, Access::Random,
                  "an op whose effects are unknown");
  bool reads = hasEffectOn(op, v, isRead);
  bool writes = hasEffectOn(op, v, isWrite);
  if (reads)
    return layout(CollectionUse::Role::Read, Access::Random,
                  "a read by index");
  if (writes)
    return layout(CollectionUse::Role::Write, Access::Random,
                  "a write by index");
  return layout(CollectionUse::Role::Escape, Access::Random,
                "forwarded by an op that is not a followed view");
}

//===----------------------------------------------------------------------===//
// The analysis
//===----------------------------------------------------------------------===//

/// Whether `op` stands in a loop of its function.
static Operation *outermostLoop(Operation *op) {
  Operation *loop = nullptr;
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isa<FunctionOpInterface>(parent))
      break;
    if (isa<LoopLikeOpInterface>(parent))
      loop = parent;
  }
  return loop;
}

CollectionAnalysis::CollectionAnalysis(Operation *root) : root(root) {
  // The collection of each verdict, by index until every function's
  // collections are in place.
  SmallVector<size_t> results;
  SmallVector<FunctionOpInterface> functions;
  root->walk([&](FunctionOpInterface f) { functions.push_back(f); });
  for (FunctionOpInterface function : functions) {
    int64_t next = 0;
    function->walk<WalkOrder::PreOrder>(
        [&](Operation *op) { position[op] = next++; });
    auto end = [&](Operation *op) {
      int64_t last = position.lookup(op);
      op->walk([&](Operation *inner) {
        last = std::max(last, position.lookup(inner));
      });
      return last;
    };

    // The scans of the function, for the offsets of a scatter's cursor.
    SmallVector<ScanOp> scans;
    function->walk([&](ScanOp scan) { scans.push_back(scan); });
    auto groupOfCursor = [&](Value cursor) -> Value {
      for (ScanOp scan : scans)
        if (scan.getCursor() && rootOf(scan.getCursor()) == cursor)
          return rootOf(scan.getStart());
      return Value();
    };

    size_t first = collections.size();
    function->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (!isa_and_nonnull<MDAlgDialect>(op->getDialect()))
        return;
      for (Produced &p : producedBy(op, groupOfCursor)) {
        Collection c;
        c.producer = op;
        c.role = p.role;
        c.buffer = p.buffer;
        c.root = rootOf(p.buffer);
        c.guaranteed = p.guarantee;
        c.cardinality = p.cardinality;
        c.produced = position.lookup(op);
        if (auto deterministic = dyn_cast<DeterminismInterface>(op))
          c.determinism = deterministic.getRequiredDeterminism();
        collections.push_back(std::move(c));
      }
    });

    for (size_t i = first; i < collections.size(); ++i) {
      Collection &c = collections[i];
      // The root: an argument of the function escapes; an allocation is
      // local; anything else came from where the analysis cannot see.
      if (auto argument = dyn_cast<BlockArgument>(c.root)) {
        if (argument.getOwner()->isEntryBlock() &&
            argument.getOwner()->getParentOp() == function)
          c.escape = "an argument of the function";
        else
          c.escape = "an argument of a block";
      } else if (Operation *def = c.root.getDefiningOp()) {
        if (hasEffectOn(def, c.root, isAllocate))
          c.allocated = position.lookup(def);
        else
          c.escape = ("defined by " + def->getName().getStringRef()).str();
      }
      // The aliases, through the followed views.
      SmallVector<Value> work = {c.root};
      SmallVector<Value> all;
      while (!work.empty()) {
        Value v = work.pop_back_val();
        all.push_back(v);
        for (Operation *user : v.getUsers())
          if (isFollowedView(user) && user->getOperand(0) == v)
            for (Value r : user->getResults())
              work.push_back(r);
      }
      c.aliases.assign(all.begin() + 1, all.end());

      // The uses after the producer; every use, where it stands in a loop.
      bool looped = outermostLoop(c.producer) != nullptr;
      c.required = OrderSpec::none();
      c.lastUse = c.produced;
      for (Value v : all) {
        for (OpOperand &operand : v.getUses()) {
          Operation *user = operand.getOwner();
          if (user == c.producer || c.producer->isAncestor(user))
            continue;
          if (isFollowedView(user) && operand.getOperandNumber() == 0)
            continue;
          if (!looped && position.lookup(user) < c.produced)
            continue;
          CollectionUse use = classify(operand);
          if (use.role == CollectionUse::Role::Release) {
            c.released = std::max(c.released, position.lookup(user));
          } else {
            int64_t at = position.lookup(user);
            if (Operation *loop = outermostLoop(user))
              at = std::max(at, end(loop));
            c.lastUse = std::max(c.lastUse, at);
          }
          if (use.role == CollectionUse::Role::Escape && c.escape.empty())
            c.escape = use.why;
          if (use.role == CollectionUse::Role::Read ||
              (use.role == CollectionUse::Role::Write &&
               use.required.kind == OrderSpec::Kind::Layout) ||
              use.role == CollectionUse::Role::Escape) {
            Access a = use.access;
            if (!c.access || static_cast<int>(a) > static_cast<int>(*c.access))
              c.access = a;
          }
          c.required = OrderSpec::join(c.required, use.required);
          c.uses.push_back(std::move(use));
        }
      }
      // An escape from the root: whatever the code outside reads.
      if (!c.escape.empty()) {
        c.required = OrderSpec::layout();
        if (!c.access)
          c.access = Access::Random;
      }
      std::stable_sort(c.uses.begin(), c.uses.end(),
                       [&](const CollectionUse &a, const CollectionUse &b) {
                         return position.lookup(a.op) <
                                position.lookup(b.op);
                       });
    }

    // The sorts.
    for (size_t i = first; i < collections.size(); ++i) {
      const Collection &c = collections[i];
      SortVerdict verdict;
      if (auto sort = dyn_cast<SortByKeyOp>(c.producer)) {
        if (sort.getOrder() != Order::Stable || c.role != "out")
          continue;
        verdict.without = OrderSpec::groupedBy(rootOf(sort.getStart()));
      } else if (auto sort = dyn_cast<SortBinsOp>(c.producer)) {
        // Without the sort, the result is its input: what the last
        // producer of the items before it guarantees, or no more than
        // which items are there.
        Value input = rootOf(sort.getItems());
        verdict.without = OrderSpec::membership();
        for (size_t j = first; j < collections.size(); ++j) {
          const Collection &before = collections[j];
          if (before.root == input && before.produced < c.produced &&
              before.producer != c.producer)
            verdict.without = before.guaranteed;
        }
      } else {
        continue;
      }
      verdict.sort = c.producer;
      results.push_back(i);
      // The layout is what the program reads now, with the sort.
      verdict.sufficient = c.required.kind == OrderSpec::Kind::Layout ||
                           c.required.isMetBy(c.guaranteed);
      verdict.orderMetWithout = c.required.isMetBy(verdict.without);
      verdict.exactOrUnordered =
          c.escape.empty() &&
          llvm::all_of(c.uses, [](const CollectionUse &u) {
            return u.exactOrUnordered;
          });
      sorts.push_back(verdict);
    }
  }
  for (auto [verdict, index] : llvm::zip(sorts, results))
    verdict.result = &collections[index];
}

const Collection *CollectionAnalysis::lookup(Operation *producer,
                                             StringRef role) const {
  for (const Collection &c : collections)
    if (c.producer == producer && c.role == role)
      return &c;
  return nullptr;
}

//===----------------------------------------------------------------------===//
// The printer
//===----------------------------------------------------------------------===//

static StringRef accessName(Access a) {
  switch (a) {
  case Access::Streamed:
    return "streamed";
  case Access::Gathered:
    return "gathered";
  case Access::Random:
    return "random";
  }
  llvm_unreachable("an access");
}

void CollectionAnalysis::print(llvm::raw_ostream &os) const {
  AsmState state(root);
  auto at = [&](Operation *op) {
    std::string text;
    llvm::raw_string_ostream s(text);
    s << op->getName() << " at " << position.lookup(op);
    return text;
  };
  auto spec = [&](const OrderSpec &o) {
    std::string text;
    llvm::raw_string_ostream s(text);
    o.print(s, state);
    return text;
  };
  auto cardinality = [&](const Cardinality &c) {
    std::string text;
    llvm::raw_string_ostream s(text);
    switch (c.kind) {
    case Cardinality::Kind::Exact:
      s << "exact " << c.exact;
      break;
    case Cardinality::Kind::Bound:
      s << "bound " << name(c.value, state);
      break;
    case Cardinality::Kind::RunTime:
      s << "run-time";
      if (c.value) {
        s << " " << name(c.value, state);
        if (c.offset)
          s << " + " << c.offset;
      }
      break;
    }
    return text;
  };
  FunctionOpInterface current;
  for (const Collection &c : collections) {
    auto function = c.producer->getParentOfType<FunctionOpInterface>();
    if (function != current) {
      current = function;
      os << "md_alg collections of @" << function.getName() << "\n";
    }
    os << "  collection " << c.role << " of " << at(c.producer) << ": "
       << name(c.buffer, state) << "\n";
    os << "    root: " << name(c.root, state) << ", "
       << (c.allocated >= 0 ? "allocated at " + std::to_string(c.allocated)
                            : std::string("not allocated here"))
       << ", aliases " << c.aliases.size() << "\n";
    os << "    escapes: " << (c.escape.empty() ? "no" : c.escape) << "\n";
    os << "    guarantees: " << spec(c.guaranteed) << "\n";
    for (const CollectionUse &use : c.uses) {
      os << "    use: " << at(use.op) << ": ";
      switch (use.role) {
      case CollectionUse::Role::Release:
        os << "release";
        break;
      case CollectionUse::Role::Shape:
        os << "shape";
        break;
      case CollectionUse::Role::Write:
        os << "write, requires " << spec(use.required);
        break;
      case CollectionUse::Role::Read:
        os << "read, requires " << spec(use.required) << ", "
           << accessName(use.access);
        break;
      case CollectionUse::Role::Escape:
        os << "escape, requires " << spec(use.required);
        break;
      }
      os << " (" << use.why << ")\n";
    }
    os << "    requires: " << spec(c.required) << "\n";
    os << "    access: " << (c.access ? accessName(*c.access) : "none") << "\n";
    os << "    cardinality: " << cardinality(c.cardinality) << "\n";
    os << "    lifetime: produced at " << c.produced << ", last use at "
       << c.lastUse << ", "
       << (c.released >= 0 ? "released at " + std::to_string(c.released)
                           : std::string("not released here"))
       << "\n";
    os << "    determinism: "
       << (c.determinism ? getDeterminismName(*c.determinism) : "none")
       << "\n";
    for (const SortVerdict &v : sorts) {
      if (v.result != &c)
        continue;
      os << "  sort " << at(v.sort) << ": "
         << (v.redundant() ? "redundant" : "not redundant") << ": requires "
         << spec(c.required) << ", without the sort " << spec(v.without)
         << ", with it " << spec(c.guaranteed) << (v.sufficient ? "" : " (not sufficient)");
      if (!v.orderMetWithout)
        os << "; the order is not met without it";
      if (!v.exactOrUnordered)
        os << "; a consumer does not combine exactly";
      os << "\n";
    }
  }
}
