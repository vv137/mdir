// The liveness of the storage of md_exec within the loop over steps
// (D[md-exec-storage-liveness]); see
// include/mdir/Dialect/MDExec/StorageLiveness.h.
//
// Three steps:
// 1. The aliases. Each memref value of the function is mapped to the
//    buffers it may view: a view (a cast, a change of memory space, a
//    subview, a reinterpret cast, ...) views those of its source, an
//    argument or a result of a loop or a branch those of every value that
//    flows into it, and the result of any other op those of all its memref
//    operands.
// 2. The effects. Each op of the host's code is a point; a launch, or any
//    op with a region that is not control flow, is one point, with the
//    effects of every op in its body. An effect reads, writes, or touches a
//    buffer in a way the analysis cannot tell (a raw pointer, an external
//    function); only a write through a view of the whole buffer by an op
//    that does not read it kills. An asynchronous op uses its buffers again
//    where its token is waited for, and an op on a side stream where the
//    side stream is joined.
// 3. The dataflow. Liveness backward over the points to a fixpoint, then
//    the occupancy of each point (live in, live out, or touched there) and
//    the interference of buffers occupied at one point, as for the
//    registers of Chaitin, SIGPLAN Notices 17(6), 98 (1982).

#include "mdir/Dialect/MDExec/StorageLiveness.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mdir::md_exec;

namespace mdir::md_exec {

namespace {

/// The effects of one point, or of a whole function on its arguments, by
/// buffer.
struct Effects {
  llvm::BitVector read, write, unknown, release, partialWrite, provenWrite;
  /// Escapes: the buffer is live everywhere.
  llvm::DenseMap<unsigned, std::string> pins;

  explicit Effects(unsigned n = 0)
      : read(n), write(n), unknown(n), release(n), partialWrite(n),
        provenWrite(n) {}
  llvm::BitVector touched() const {
    llvm::BitVector t = read;
    t |= write;
    t |= unknown;
    t |= release;
    return t;
  }
  bool any() const { return touched().any() || !pins.empty(); }
};

/// Whether a value views a whole buffer: the buffer itself, or a cast of it
/// that keeps its extent.
static bool isMemRef(Value v) { return isa<BaseMemRefType>(v.getType()); }

static bool isFullView(Operation *op) {
  return isa<memref::CastOp, memref::MemorySpaceCastOp>(op);
}

static bool isView(Operation *op) {
  return isa<ViewLikeOpInterface, memref::CastOp, memref::MemorySpaceCastOp,
             memref::ReinterpretCastOp, memref::CollapseShapeOp,
             memref::ExpandShapeOp, memref::ExtractStridedMetadataOp,
             arith::SelectOp, UnrealizedConversionCastOp>(op);
}

/// Ops that read only the shape of a memref, not its contents.
static bool isShape(Operation *op) {
  return isa<memref::DimOp, memref::RankOp>(op);
}

static bool isStreamCall(Operation *op, StringRef name) {
  auto call = dyn_cast<func::CallOp>(op);
  return call && call.getCallee() == name;
}

static bool isStructural(Operation *op) {
  return isa<scf::ForOp, scf::WhileOp, scf::IfOp, scf::IndexSwitchOp,
             scf::ExecuteRegionOp>(op);
}

/// The aliases of the memref values of one function over a list of roots.
class Aliases {
public:
  Aliases(Region &body, llvm::ArrayRef<Value> roots) : n(roots.size()) {
    for (auto [i, root] : llvm::enumerate(roots)) {
      llvm::BitVector bits(n);
      bits.set(i);
      map[root] = bits;
      full.insert(root);
    }
    // To a fixpoint: values flow around loops.
    bool changed = true;
    while (changed) {
      changed = false;
      body.walk([&](Operation *op) { changed |= visit(op); });
    }
  }

  const llvm::BitVector *lookup(Value v) const {
    auto it = map.find(v);
    return it == map.end() ? nullptr : &it->second;
  }
  /// Whether `v` views all of exactly one buffer.
  bool isFull(Value v) const {
    const llvm::BitVector *bits = lookup(v);
    return bits && bits->count() == 1 && full.contains(v);
  }
  unsigned size() const { return n; }

private:
  unsigned n;
  llvm::DenseMap<Value, llvm::BitVector> map;
  llvm::DenseSet<Value> full;

  bool join(Value to, Value from) {
    if (!isMemRef(to))
      return false;
    const llvm::BitVector *bits = lookup(from);
    if (!bits || bits->none())
      return false;
    llvm::BitVector &mine = map.try_emplace(to, n).first->second;
    llvm::BitVector before = mine;
    mine |= *bits;
    return mine != before;
  }

  bool visit(Operation *op) {
    bool changed = false;
    if (auto loop = dyn_cast<scf::ForOp>(op)) {
      auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
      for (auto [i, init] : llvm::enumerate(loop.getInitArgs())) {
        Value arg = loop.getRegionIterArgs()[i];
        changed |= join(arg, init) | join(arg, yield.getOperand(i)) |
                   join(loop.getResult(i), init) |
                   join(loop.getResult(i), yield.getOperand(i));
      }
      return changed;
    }
    if (auto loop = dyn_cast<scf::WhileOp>(op)) {
      auto condition = loop.getConditionOp();
      auto yield = loop.getYieldOp();
      for (auto [i, init] : llvm::enumerate(loop.getInits())) {
        Value arg = loop.getBeforeArguments()[i];
        changed |= join(arg, init) | join(arg, yield.getOperand(i));
      }
      for (auto [i, value] : llvm::enumerate(condition.getArgs())) {
        changed |= join(loop.getAfterArguments()[i], value) |
                   join(loop.getResult(i), value);
      }
      return changed;
    }
    if (isa<scf::IfOp, scf::IndexSwitchOp, scf::ExecuteRegionOp>(op)) {
      for (Region &region : op->getRegions())
        for (Block &block : region)
          if (auto yield = dyn_cast<scf::YieldOp>(block.getTerminator()))
            for (auto [i, value] : llvm::enumerate(yield.getOperands()))
              changed |= join(op->getResult(i), value);
      return changed;
    }
    if (isa<gpu::AllocOp, memref::AllocOp, memref::AllocaOp>(op))
      return false;
    // Any other op: a memref result may view any memref operand. A full
    // view of a whole buffer is itself whole.
    for (Value result : op->getResults()) {
      if (!isMemRef(result))
        continue;
      for (Value operand : op->getOperands())
        changed |= join(result, operand);
      if (isFullView(op) && op->getNumOperands() >= 1 &&
          full.contains(op->getOperand(0)) && !full.contains(result)) {
        full.insert(result);
        changed = true;
      }
    }
    return changed;
  }
};

} // namespace

class StorageLivenessBuilder {
public:
  StorageLivenessBuilder(StorageLiveness &result) : r(result) {
    module = r.function->getParentOfType<ModuleOp>();
  }

  void run();

private:
  StorageLiveness &r;
  ModuleOp module;
  unsigned n = 0;
  std::unique_ptr<Aliases> aliases;
  /// The effects of each point op of the function.
  llvm::DenseMap<Operation *, Effects> effects;
  /// The buffers each op uses again: an asynchronous op's at its wait, a
  /// side stream's at its join.
  llvm::DenseMap<Operation *, llvm::BitVector> completions;
  /// The summaries of the functions of the module, by their arguments.
  llvm::DenseMap<Operation *, Effects> summaries;
  llvm::DenseSet<Operation *> summarizing;
  llvm::BitVector pinned;
  llvm::DenseMap<unsigned, std::string> pinReasons;

  Effects effectsOf(Operation *point, const Aliases &aliases, bool host);
  void addEffect(Effects &e, const Aliases &aliases, Value v, bool read,
                 bool write, bool unknown, bool release, Operation *op);
  const Effects &summarize(func::FuncOp callee);
  void pin(Effects &e, const Aliases &aliases, Value v, std::string why);
  void findCompletions(Region &body);
  llvm::SmallVector<Operation *> waitsOf(Value token, bool &resolved);

  unsigned addPoint(Operation *op, int step) {
    StoragePoint p;
    p.op = op;
    p.stepIndex = step;
    p.use.resize(n);
    p.kill.resize(n);
    p.access.resize(n);
    p.liveIn.resize(n);
    p.liveOut.resize(n);
    r.points.push_back(std::move(p));
    return r.points.size() - 1;
  }
  void link(unsigned from, unsigned to) {
    r.points[from].successors.push_back(to);
  }
  int stepIndex(bool inStep) { return inStep ? r.stepPoints++ : -1; }
  /// Builds the points of a block after `entry`; returns the last point.
  unsigned buildBlock(Block &block, unsigned entry, bool inStep);
  unsigned buildOp(Operation *op, unsigned entry, bool inStep);
};

void StorageLivenessBuilder::pin(Effects &e, const Aliases &a, Value v,
                                 std::string why) {
  if (const llvm::BitVector *bits = a.lookup(v))
    for (unsigned i : bits->set_bits())
      e.pins.try_emplace(i, why);
}

void StorageLivenessBuilder::addEffect(Effects &e, const Aliases &a, Value v,
                                       bool read, bool write, bool unknown,
                                       bool release, Operation *op) {
  const llvm::BitVector *bits = a.lookup(v);
  if (!bits || bits->none())
    return;
  if (read)
    e.read |= *bits;
  if (unknown)
    e.unknown |= *bits;
  if (release)
    e.release |= *bits;
  if (write) {
    e.write |= *bits;
    if (!a.isFull(v))
      e.partialWrite |= *bits;
    else if (isa<gpu::MemcpyOp, gpu::MemsetOp>(op))
      e.provenWrite |= *bits;
  }
}

/// The waits of a token: the `gpu.wait`s without a result it reaches,
/// directly or through other tokens. `resolved` is false when the token
/// leaves the code that the analysis follows (a yield, a call).
llvm::SmallVector<Operation *>
StorageLivenessBuilder::waitsOf(Value token, bool &resolved) {
  llvm::SmallVector<Operation *> waits;
  llvm::SmallVector<Value> work{token};
  llvm::DenseSet<Value> seen;
  bool any = false;
  while (!work.empty()) {
    Value t = work.pop_back_val();
    if (!seen.insert(t).second)
      continue;
    for (Operation *user : t.getUsers()) {
      auto async = dyn_cast<gpu::AsyncOpInterface>(user);
      if (auto wait = dyn_cast<gpu::WaitOp>(user); wait && !wait.getAsyncToken()) {
        waits.push_back(user);
        any = true;
      } else if (async && async.getAsyncToken()) {
        work.push_back(async.getAsyncToken());
      } else {
        resolved = false;
      }
    }
  }
  if (!any)
    resolved = false;
  return waits;
}

const Effects &StorageLivenessBuilder::summarize(func::FuncOp callee) {
  auto it = summaries.find(callee);
  if (it != summaries.end())
    return it->second;
  unsigned arity = callee.getNumArguments();
  llvm::SmallVector<Value> args(callee.getArguments());
  Effects all(arity);
  if (!summarizing.insert(callee).second) {
    // A recursive call: anything.
    all.unknown.set();
    return summaries.try_emplace(callee, all).first->second;
  }
  Aliases a(callee.getBody(), args);
  callee.getBody().walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op->getParentOfType<gpu::LaunchOp>())
      return WalkResult::skip();
    if (isStructural(op))
      return WalkResult::advance();
    Effects e = effectsOf(op, a, /*host=*/true);
    all.read |= e.read;
    all.write |= e.write;
    all.unknown |= e.unknown;
    all.release |= e.release;
    all.partialWrite |= e.partialWrite;
    all.provenWrite |= e.provenWrite;
    for (auto &[i, why] : e.pins)
      all.pins.try_emplace(i, "in @" + callee.getSymName().str() + ": " + why);
    // An asynchronous op of the callee: its buffers stay in use until a
    // wait in the callee; a token that leaves it pins them.
    if (auto async = dyn_cast<gpu::AsyncOpInterface>(op);
        async && async.getAsyncToken() && e.touched().any()) {
      bool resolved = true;
      (void)waitsOf(async.getAsyncToken(), resolved);
      if (!resolved)
        for (unsigned i : e.touched().set_bits())
          all.pins.try_emplace(i, "in @" + callee.getSymName().str() +
                                      ": a token not waited for");
    }
    return isa<gpu::LaunchOp>(op) || op->getNumRegions() ? WalkResult::skip()
                                                         : WalkResult::advance();
  });
  summarizing.erase(callee);
  return summaries.try_emplace(callee, std::move(all)).first->second;
}

/// The effects of a point op: of the op and, for a launch or another op
/// with regions, of every op in its body.
Effects StorageLivenessBuilder::effectsOf(Operation *point, const Aliases &a,
                                          bool host) {
  Effects e(a.size());
  auto visit = [&](Operation *op, bool atHost) {
    if (isa<gpu::AllocOp, memref::AllocOp, memref::AllocaOp>(op) ||
        isShape(op) || isa<scf::YieldOp, scf::ConditionOp>(op))
      return;
    if (isa<func::ReturnOp>(op)) {
      for (Value v : op->getOperands())
        pin(e, a, v, "returned");
      return;
    }
    if (auto call = dyn_cast<CallOpInterface>(op)) {
      func::FuncOp callee;
      if (auto sym = llvm::dyn_cast_if_present<SymbolRefAttr>(
              call.getCallableForCallee()))
        callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, sym);
      if (callee && !callee.isExternal()) {
        const Effects &s = summarize(callee);
        for (auto [i, v] : llvm::enumerate(call.getArgOperands())) {
          if (!isMemRef(v) || i >= s.read.size())
            continue;
          addEffect(e, a, v, s.read[i], s.write[i], s.unknown[i],
                    s.release[i], op);
          // A write that is whole in the callee is whole here only if the
          // argument views the whole buffer.
          const llvm::BitVector *bits = a.lookup(v);
          if (bits && s.partialWrite[i])
            e.partialWrite |= *bits;
          if (bits && s.provenWrite[i] && a.isFull(v) && !s.partialWrite[i])
            e.provenWrite |= *bits;
          if (auto pinIt = s.pins.find(i); pinIt != s.pins.end())
            pin(e, a, v, pinIt->second);
        }
        return;
      }
      // An external function: it may read and write each buffer it is
      // given, during the call.
      for (Value v : call.getArgOperands())
        if (isMemRef(v))
          addEffect(e, a, v, false, false, true, false, op);
      return;
    }
    if (isa<gpu::LaunchFuncOp>(op)) {
      for (Value v : op->getOperands())
        if (isMemRef(v))
          addEffect(e, a, v, false, false, true, false, op);
      return;
    }
    if (isView(op))
      return;
    // An op that turns a buffer into something that is not a memref (a raw
    // pointer): an access the analysis cannot follow; in the host's code,
    // the pointer may outlive the op.
    bool opaque = llvm::any_of(op->getResults(), [](Value v) {
      return !isMemRef(v) && !isa<gpu::AsyncTokenType>(v.getType());
    });
    if ((opaque && !isa<MemoryEffectOpInterface>(op)) ||
        isa<memref::ExtractAlignedPointerAsIndexOp>(op)) {
      for (Value v : op->getOperands()) {
        if (!isMemRef(v))
          continue;
        addEffect(e, a, v, false, false, true, false, op);
        if (atHost)
          pin(e, a, v, "a raw pointer in the host's code");
      }
      return;
    }
    auto iface = dyn_cast<MemoryEffectOpInterface>(op);
    if (!iface) {
      for (Value v : op->getOperands())
        if (isMemRef(v))
          addEffect(e, a, v, false, false, true, false, op);
      return;
    }
    llvm::SmallVector<MemoryEffects::EffectInstance> list;
    iface.getEffects(list);
    llvm::DenseSet<Value> named;
    for (auto &effect : list) {
      Value v = effect.getValue();
      if (!v || !isMemRef(v))
        continue;
      named.insert(v);
      bool isRead = isa<MemoryEffects::Read>(effect.getEffect());
      bool isWrite = isa<MemoryEffects::Write>(effect.getEffect());
      bool isFree = isa<MemoryEffects::Free>(effect.getEffect());
      if (isa<MemoryEffects::Allocate>(effect.getEffect()))
        continue;
      addEffect(e, a, v, isRead, isWrite, false, isFree, op);
    }
    // A memref operand whose effect the op does not name: unknown, unless
    // the op has no effect at all.
    if (!list.empty() || !isMemoryEffectFree(op))
      for (Value v : op->getOperands())
        if (isMemRef(v) && !named.contains(v))
          addEffect(e, a, v, false, false, true, false, op);
  };
  if (point->getNumRegions() == 0) {
    visit(point, host);
    return e;
  }
  point->walk([&](Operation *op) {
    if (op != point)
      visit(op, /*atHost=*/false);
  });
  // The operands of the point itself (a launch's sizes, a call's
  // arguments are walked above when it has no regions).
  for (Value v : point->getOperands())
    if (isMemRef(v))
      addEffect(e, a, v, false, false, true, false, point);
  return e;
}

/// The completions: for each asynchronous op that touches a buffer, its
/// buffers are used again at every wait its token reaches; for each op on
/// a side stream (between `mdrtSideBegin` and `mdrtSideEnd`), at the next
/// `mdrtSideJoin` of its block or of a block around it.
void StorageLivenessBuilder::findCompletions(Region &body) {
  body.walk([&](Operation *op) {
    auto it = effects.find(op);
    if (it == effects.end())
      return;
    llvm::BitVector touched = it->second.touched();
    if (touched.none())
      return;
    if (auto async = dyn_cast<gpu::AsyncOpInterface>(op);
        async && async.getAsyncToken()) {
      bool resolved = true;
      for (Operation *wait : waitsOf(async.getAsyncToken(), resolved))
        completions.try_emplace(wait, n).first->second |= touched;
      if (!resolved)
        for (unsigned i : touched.set_bits())
          pinReasons.try_emplace(i, "a token not waited for");
    }
  });
  // Side streams.
  body.walk([&](Block *block) {
    bool side = false;
    llvm::BitVector pending(n);
    for (Operation &op : *block) {
      if (isStreamCall(&op, "mdrtSideBegin"))
        side = true;
      else if (isStreamCall(&op, "mdrtSideEnd"))
        side = false;
      else if (isStreamCall(&op, "mdrtSideJoin") && pending.any()) {
        completions.try_emplace(&op, n).first->second |= pending;
        pending.reset();
      } else if (side) {
        op.walk([&](Operation *inner) {
          if (auto it = effects.find(inner); it != effects.end())
            pending |= it->second.touched();
        });
      }
    }
    if (pending.none())
      return;
    // No join in the block: the next one after it, in the blocks around.
    Operation *at = block->getParentOp();
    while (at && at != r.function.getOperation()) {
      for (Operation *next = at->getNextNode(); next;
           next = next->getNextNode())
        if (isStreamCall(next, "mdrtSideJoin")) {
          completions.try_emplace(next, n).first->second |= pending;
          return;
        }
      at = at->getParentOp();
    }
    for (unsigned i : pending.set_bits())
      pinReasons.try_emplace(i, "a side stream never joined");
  });
}

unsigned StorageLivenessBuilder::buildOp(Operation *op, unsigned entry,
                                         bool inStep) {
  if (auto loop = dyn_cast<scf::ForOp>(op)) {
    bool isStep = op == r.stepLoop;
    bool body = inStep || isStep;
    unsigned head = addPoint(op, stepIndex(body));
    link(entry, head);
    unsigned last = buildBlock(*loop.getBody(), head, body);
    link(last, head);
    unsigned exit = addPoint(op, stepIndex(inStep));
    link(head, exit);
    return exit;
  }
  if (auto loop = dyn_cast<scf::WhileOp>(op)) {
    bool body = inStep || op == r.stepLoop;
    unsigned head = addPoint(op, stepIndex(body));
    link(entry, head);
    unsigned before = buildBlock(loop.getBefore().front(), head, body);
    unsigned after = buildBlock(loop.getAfter().front(), before, body);
    link(after, head);
    unsigned exit = addPoint(op, stepIndex(inStep));
    link(before, exit);
    return exit;
  }
  if (isa<scf::IfOp, scf::IndexSwitchOp>(op)) {
    unsigned branch = addPoint(op, stepIndex(inStep));
    link(entry, branch);
    llvm::SmallVector<unsigned> ends;
    bool hasEmpty = false;
    for (Region &region : op->getRegions()) {
      if (region.empty()) {
        hasEmpty = true;
        continue;
      }
      ends.push_back(buildBlock(region.front(), branch, inStep));
    }
    unsigned join = addPoint(op, stepIndex(inStep));
    for (unsigned end : ends)
      link(end, join);
    if (hasEmpty)
      link(branch, join);
    return join;
  }
  if (isa<scf::ExecuteRegionOp>(op) && op->getRegion(0).hasOneBlock())
    return buildBlock(op->getRegion(0).front(), entry, inStep);

  auto it = effects.find(op);
  auto done = completions.find(op);
  auto alloc = dyn_cast<gpu::AllocOp>(op);
  int own = -1;
  if (alloc)
    for (StorageBuffer &b : r.buffers)
      if (b.alloc == alloc)
        own = b.index;
  if (it == effects.end() && done == completions.end() && own < 0)
    return entry;
  unsigned id = addPoint(op, stepIndex(inStep));
  link(entry, id);
  StoragePoint &p = r.points[id];
  if (own >= 0) {
    // The allocation begins the buffer's life.
    p.kill.set(own);
    p.access.set(own);
  }
  if (it != effects.end()) {
    const Effects &e = it->second;
    p.access |= e.touched();
    p.use |= e.read;
    p.use |= e.unknown;
    // A release ends it.
    p.kill |= e.release;
    llvm::BitVector writeOnly = e.write;
    llvm::BitVector notWhole = e.read;
    notWhole |= e.unknown;
    notWhole |= e.partialWrite;
    writeOnly.reset(notWhole);
    if (r.kills == StorageKills::Proven)
      writeOnly &= e.provenWrite;
    p.kill |= writeOnly;
  }
  if (done != completions.end()) {
    p.use |= done->second;
    p.access |= done->second;
  }
  return id;
}

unsigned StorageLivenessBuilder::buildBlock(Block &block, unsigned entry,
                                            bool inStep) {
  unsigned last = entry;
  for (Operation &op : block)
    last = buildOp(&op, last, inStep);
  return last;
}

void StorageLivenessBuilder::run() {
  Block &body = r.function.getBody().front();
  // The loop over steps: the loop of the body that launches the most
  // kernels and calls the most functions of the module (a loop of the setup,
  // such as the one of the minimization of the constraints, launches
  // fewer).
  unsigned most = 0;
  for (Operation &op : body) {
    if (!isa<scf::ForOp, scf::WhileOp>(op))
      continue;
    unsigned work = 0;
    op.walk([&](Operation *inner) {
      if (isa<gpu::LaunchOp, gpu::LaunchFuncOp>(inner))
        ++work;
      if (auto call = dyn_cast<func::CallOp>(inner))
        if (auto f = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
                inner, call.getCalleeAttr());
            f && !f.isExternal())
          ++work;
    });
    if (work > most) {
      most = work;
      r.stepLoop = &op;
    }
  }
  if (!r.stepLoop)
    return;
  llvm::SmallVector<Value> roots;
  for (Operation &op : body) {
    if (&op == r.stepLoop)
      break;
    if (auto alloc = dyn_cast<gpu::AllocOp>(op)) {
      StorageBuffer b;
      b.alloc = alloc;
      b.index = r.buffers.size();
      r.buffers.push_back(std::move(b));
      roots.push_back(alloc.getMemref());
    }
  }
  n = roots.size();
  pinned.resize(n);
  aliases = std::make_unique<Aliases>(r.function.getBody(), roots);
  // The effects of every point op.
  r.function.getBody().walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isStructural(op) || op == r.function.getOperation())
      return WalkResult::advance();
    Effects e = effectsOf(op, *aliases, /*host=*/true);
    for (auto &[i, why] : e.pins)
      pinReasons.try_emplace(i, why);
    if (e.any())
      effects.try_emplace(op, std::move(e));
    return op->getNumRegions() ? WalkResult::skip() : WalkResult::advance();
  });
  findCompletions(r.function.getBody());
  for (auto &[i, why] : pinReasons) {
    pinned.set(i);
    r.buffers[i].pinned = why;
  }
  for (auto &[op, e] : effects)
    for (unsigned i : e.release.set_bits())
      r.buffers[i].released = true;

  unsigned entry = addPoint(r.function, -1);
  buildBlock(body, entry, /*inStep=*/false);

  // Liveness, backward, to a fixpoint; the function's end has nothing live.
  bool changed = true;
  while (changed) {
    changed = false;
    for (int i = r.points.size() - 1; i >= 0; --i) {
      StoragePoint &p = r.points[i];
      llvm::BitVector out(n);
      for (unsigned s : p.successors)
        out |= r.points[s].liveIn;
      llvm::BitVector in = out;
      in.reset(p.kill);
      in |= p.use;
      if (in != p.liveIn || out != p.liveOut) {
        p.liveIn = std::move(in);
        p.liveOut = std::move(out);
        changed = true;
      }
    }
  }

  // The occupancy of each point, and the interference.
  for (StorageBuffer &b : r.buffers) {
    b.stepOccupied.resize(r.stepPoints);
    b.occupied.resize(r.points.size());
    b.stepInterferes.resize(n);
    b.interferes.resize(n);
  }
  unsigned head = 0, exit = 0;
  for (auto [i, p] : llvm::enumerate(r.points)) {
    if (p.op == r.stepLoop) {
      if (!head)
        head = i;
      else
        exit = i;
    }
  }
  for (auto [i, p] : llvm::enumerate(r.points)) {
    llvm::BitVector occupied = p.liveIn;
    occupied |= p.liveOut;
    occupied |= p.access;
    // A pinned buffer is live everywhere after its allocation: in the step,
    // at least.
    if (p.stepIndex >= 0)
      occupied |= pinned;
    for (unsigned b : occupied.set_bits()) {
      StorageBuffer &buffer = r.buffers[b];
      buffer.occupied.set(i);
      buffer.interferes |= occupied;
      if (p.stepIndex >= 0) {
        buffer.stepOccupied.set(p.stepIndex);
        buffer.stepInterferes |= occupied;
      }
    }
    if (p.stepIndex >= 0)
      for (unsigned b : p.access.set_bits()) {
        StorageBuffer &buffer = r.buffers[b];
        if (buffer.firstAccess < 0)
          buffer.firstAccess = p.stepIndex;
        buffer.lastAccess = p.stepIndex;
      }
  }
  // Pinned buffers interfere with every buffer live at any point.
  for (unsigned b : pinned.set_bits())
    for (StorageBuffer &other : r.buffers)
      if (other.occupied.any()) {
        other.interferes.set(b);
        r.buffers[b].interferes.set(other.index);
      }
  for (StorageBuffer &b : r.buffers) {
    b.stepInterferes.reset(b.index);
    b.interferes.reset(b.index);
    // Carried: live into the first point of the step's body, which the
    // head links first (the head itself when the body touches nothing).
    const StoragePoint &h = r.points[head];
    if (!h.successors.empty() && h.successors.front() != head &&
        r.points[h.successors.front()].liveIn.test(b.index))
      b.carried = true;
    if (pinned.test(b.index))
      b.carried = true;
    if (exit && r.points[exit].liveIn.test(b.index))
      b.liveAfterLoop = true;
  }
}

} // namespace mdir::md_exec

StorageLiveness::StorageLiveness(func::FuncOp function, StorageKills kills)
    : function(function), kills(kills) {
  StorageLivenessBuilder(*this).run();
}

llvm::SmallVector<std::pair<int, llvm::BitVector>>
StorageLiveness::liveAtCalls(StringRef callee) const {
  llvm::SmallVector<std::pair<int, llvm::BitVector>> result;
  for (const StoragePoint &p : points) {
    auto call = dyn_cast_or_null<func::CallOp>(p.op);
    if (!call || call.getCallee() != callee)
      continue;
    llvm::BitVector live = p.liveIn;
    live |= p.liveOut;
    live |= p.access;
    for (const StorageBuffer &b : buffers)
      if (!b.pinned.empty() && b.occupied.test(&p - points.data()))
        live.set(b.index);
    result.push_back({p.stepIndex, std::move(live)});
  }
  return result;
}

static void printRanges(llvm::raw_ostream &os, const llvm::BitVector &bits) {
  bool first = true;
  int start = -1;
  for (int i = 0, e = bits.size(); i <= e; ++i) {
    bool on = i < e && bits.test(i);
    if (on && start < 0)
      start = i;
    if (!on && start >= 0) {
      os << (first ? "" : ",") << start;
      if (i - 1 > start)
        os << "-" << i - 1;
      first = false;
      start = -1;
    }
  }
  if (first)
    os << "none";
}

/// A size of an allocation, by where it comes from: a dimension of an
/// argument of the function (`arg0.dim0`), a constant, or `?`.
static std::string describeSize(Value size, unsigned depth = 0) {
  if (depth > 8)
    return "?";
  if (auto c = size.getDefiningOp<arith::ConstantIndexOp>())
    return std::to_string(c.value());
  auto dim = size.getDefiningOp<memref::DimOp>();
  if (!dim)
    return "?";
  std::optional<int64_t> index = dim.getConstantIndex();
  if (!index)
    return "?";
  Value source = dim.getSource();
  while (Operation *def = source.getDefiningOp()) {
    if (isFullView(def)) {
      source = def->getOperand(0);
      continue;
    }
    if (auto alloc = dyn_cast<gpu::AllocOp>(def)) {
      auto type = cast<MemRefType>(alloc.getMemref().getType());
      if (!type.isDynamicDim(*index))
        return std::to_string(type.getDimSize(*index));
      return describeSize(
          alloc.getDynamicSizes()[type.getDynamicDimIndex(*index)], depth + 1);
    }
    return "?";
  }
  auto arg = dyn_cast<BlockArgument>(source);
  if (!arg || !isa<FunctionOpInterface>(arg.getOwner()->getParentOp()))
    return "?";
  return "arg" + std::to_string(arg.getArgNumber()) + ".dim" +
         std::to_string(*index);
}

static void printSet(llvm::raw_ostream &os, const llvm::BitVector &bits) {
  if (bits.none()) {
    os << " none";
    return;
  }
  for (unsigned i : bits.set_bits())
    os << " s" << i;
}

void StorageLiveness::print(llvm::raw_ostream &os, AsmState &state,
                            StringRef at) const {
  func::FuncOp fn = function;
  os << "storage liveness of @" << fn.getSymName() << " (kills="
     << (kills == StorageKills::Proven ? "proven" : "stores") << "): ";
  if (!stepLoop) {
    os << "no loop over steps\n";
    return;
  }
  os << buffers.size() << " buffers, " << stepPoints
     << " points in the step\n";
  for (const StorageBuffer &b : buffers) {
    gpu::AllocOp alloc = b.alloc;
    os << "  s" << b.index << " ";
    alloc.getMemref().printAsOperand(os, state);
    os << " : " << alloc.getMemref().getType();
    if (!alloc.getDynamicSizes().empty()) {
      os << " sizes(";
      llvm::interleaveComma(alloc.getDynamicSizes(), os, [&](Value v) {
        os << describeSize(v);
      });
      os << ")";
    }
    os << " step(";
    printRanges(os, b.stepOccupied);
    os << ")";
    if (b.firstAccess >= 0)
      os << " first " << b.firstAccess << " last " << b.lastAccess;
    else
      os << " untouched";
    if (b.carried)
      os << " carried";
    if (b.liveAfterLoop)
      os << " after-loop";
    if (b.released)
      os << " released";
    if (alloc.getMemref().use_empty())
      os << " unused";
    if (!b.pinned.empty())
      os << " pinned(" << b.pinned << ")";
    os << "\n";
  }
  for (const StorageBuffer &b : buffers) {
    os << "  s" << b.index << " interferes in the step with";
    printSet(os, b.stepInterferes);
    os << "\n";
  }
  for (const StorageBuffer &b : buffers) {
    os << "  s" << b.index << " interferes in the function with";
    printSet(os, b.interferes);
    os << "\n";
  }
  if (!at.empty()) {
    auto calls = liveAtCalls(at);
    if (calls.empty())
      os << "  live at @" << at << ": no call\n";
    for (auto &[step, live] : calls) {
      os << "  live at @" << at << " (";
      if (step < 0)
        os << "outside the step";
      else
        os << "step " << step;
      os << "):";
      printSet(os, live);
      os << "\n";
    }
  }
}
