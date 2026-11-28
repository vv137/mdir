// Lower the plain constrained step of the fixed-cell Gaussian PME NVT
// protocol after its storage lifetime and checks are explicit.
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/MapVector.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include <array>
#include <functional>
#include <set>

using namespace mlir;
namespace {
// A value of the host that a captured kernel takes by value is fixed at
// capture. The cell of a barostat is the one such value that changes
// between chunks: the host keeps it in its own memory, loads it before each
// chunk, and derives the launch arguments from it with pure arithmetic
// (D450). Each such argument gets a slot of a device mirror:
// before each chunk the host computes the arguments from its memory, as
// the launches of the step would, and writes them there on the stream; the
// captured kernels read their slots instead of their arguments.
struct Sink {
  std::function<bool(Value)> isLeaf;
  // The value that replaces a varying `v` in a kernel.
  std::function<Value(OpBuilder &, Location, Value)> slot;
  DenseMap<Value, bool> memo;

  static bool pure(Operation *op) {
    return op->getNumRegions() == 0 && isMemoryEffectFree(op) &&
           isa<arith::ArithDialect, vector::VectorDialect, math::MathDialect>(op->getDialect());
  }
  bool varying(Value v) {
    if (auto it = memo.find(v); it != memo.end()) return it->second;
    bool result = isLeaf(v);
    if (!result)
      if (Operation *def = v.getDefiningOp(); def && pure(def))
        result = llvm::any_of(def->getOperands(), [&](Value x) { return varying(x); });
    return memo[v] = result;
  }
  // Replace each varying value that `region` takes from above.
  void sink(Region &region, Location loc) {
    llvm::SetVector<Value> used;
    getUsedValuesDefinedAbove(region, used);
    OpBuilder k(&region.front(), region.front().begin());
    for (Value v : used) {
      if (!varying(v)) continue;
      Value inner = slot(k, loc, v);
      v.replaceUsesWithIf(inner, [&](OpOperand &use) {
        return region.isAncestor(use.getOwner()->getParentRegion());
      });
    }
  }
};

// The words of a slot: 64 bits for each scalar of `type`.
static int64_t slotWords(Type type) {
  if (auto vec = dyn_cast<VectorType>(type)) return vec.getNumElements();
  return 1;
}
static bool slotType(Type type) {
  Type e = getElementTypeOrSelf(type);
  if (auto vec = dyn_cast<VectorType>(type); vec && vec.getRank() != 1) return false;
  return e.isIndex() || (e.isIntOrFloat() && e.getIntOrFloatBitWidth() <= 64 &&
                         (e.isInteger() || e.isF32() || e.isF64()));
}
static Value toWord(OpBuilder &b, Location loc, Value x) {
  Type t = x.getType(), i64 = b.getI64Type();
  if (t.isIndex()) return arith::IndexCastOp::create(b, loc, i64, x);
  if (t.isF64()) return arith::BitcastOp::create(b, loc, i64, x);
  if (t.isF32()) x = arith::BitcastOp::create(b, loc, b.getI32Type(), x);
  if (x.getType().getIntOrFloatBitWidth() < 64) return arith::ExtUIOp::create(b, loc, i64, x);
  return x;
}
static Value fromWord(OpBuilder &b, Location loc, Value w, Type t) {
  if (t.isIndex()) return arith::IndexCastOp::create(b, loc, t, w);
  if (t.isF64()) return arith::BitcastOp::create(b, loc, t, w);
  if (t.isF32())
    return arith::BitcastOp::create(b, loc, t, arith::TruncIOp::create(b, loc, b.getI32Type(), w));
  if (t.getIntOrFloatBitWidth() < 64) return arith::TruncIOp::create(b, loc, t, w);
  return w;
}

struct Chunk {
  ModuleOp module;
  scf::ForOp loop;
  OpBuilder b;
  Location loc;
  Type i64;
  Value cache;
  SmallVector<Value> flags;
  SmallVector<memref::StoreOp> counts;
  Value positions, certificate;
  gpu::GPUFuncOp finish;
  int64_t atoms;
  // The device mirror of the cell's arguments that the chunk's kernels
  // read (D450), and how the host computes each slot.
  static constexpr int64_t mirrorWords = 256;
  Value mirror;
  int64_t slots = 0;
  DenseMap<Value, int64_t> slotOf;
  SmallVector<std::function<LogicalResult(OpBuilder &, SmallVector<Value> &)>> producers;
  Region *record = nullptr;
  llvm::SmallDenseSet<Value> storedInLoop;
  Sink cells;
  int variants = 0;

  Chunk(ModuleOp mod, scf::ForOp loop, Value cache)
      : module(mod), loop(loop), b(loop), loc(loop.getLoc()), i64(b.getI64Type()),
        cache(cache), atoms(mod->getAttrOfType<DenseI64ArrayAttr>("mdir.towers_grid")[0]) {}

  Value word(int64_t n) { return arith::ConstantIntOp::create(b, loc, n, 64); }
  Value address(Value v) {
    Value ptr = memref::ExtractAlignedPointerAsIndexOp::create(b, loc, v);
    return arith::IndexCastOp::create(b, loc, i64, ptr);
  }
  func::CallOp call(StringRef name, ValueRange args) {
    return func::CallOp::create(b, loc, module.lookupSymbol<func::FuncOp>(name), args);
  }
  // Host memory whose loads make an argument varying: a small array of
  // static shape defined before the chunk, which the chunk does not write.
  bool hostCell(Value memory) {
    auto type = dyn_cast<MemRefType>(memory.getType());
    if (!type || !type.hasStaticShape() || type.getMemorySpace() || type.getNumElements() > 64)
      return false;
    return loop.isDefinedOutsideOfLoop(memory) && !storedInLoop.contains(memory);
  }
  Value mirrorMemory() {
    if (mirror) return mirror;
    Operation *outer = loop;
    while (auto parent = outer->getParentOfType<scf::ForOp>()) outer = parent;
    auto host = MemRefType::get({mirrorWords}, i64);
    auto device = MemRefType::get({mirrorWords}, i64, MemRefLayoutAttrInterface(), b.getI64IntegerAttr(1));
    OpBuilder hoist(outer);
    mirror = gpu::AllocOp::create(hoist, loc, device, Type(), ValueRange{}, ValueRange{},
                                  ValueRange{}, false).getMemref();
    hoist.setInsertionPointAfter(outer);
    Value freed = memref::MemorySpaceCastOp::create(hoist, loc, host, mirror);
    gpu::DeallocOp::create(hoist, loc, Type(), ValueRange{}, freed);
    return mirror;
  }
  // The value of slot `base` of `type` in a kernel, from `memory`.
  Value loadSlot(OpBuilder &k, Location at, Value memory, int64_t base, Type type) {
    SmallVector<Value> parts;
    Type e = getElementTypeOrSelf(type);
    for (int64_t j = 0; j < slotWords(type); ++j) {
      Value index = arith::ConstantIndexOp::create(k, at, base + j);
      Value w = memref::LoadOp::create(k, at, memory, ValueRange{index});
      parts.push_back(fromWord(k, at, w, e));
    }
    if (isa<VectorType>(type)) return vector::FromElementsOp::create(k, at, cast<VectorType>(type), parts);
    return parts.front();
  }
  // Recompute `v` at the chunk site. Values defined in `inside` are cloned
  // (pure ops, constants and loads of the host's cell); the others already
  // dominate the site.
  // An argument of the region's entry block is resolved by `argument`.
  FailureOr<Value> recompute(OpBuilder &h, Value v, IRMapping &m, Region *inside,
                             function_ref<FailureOr<Value>(BlockArgument)> argument = nullptr) {
    if (Value done = m.lookupOrNull(v)) return done;
    if (!inside->isAncestor(v.getParentRegion())) return v;
    if (auto arg = dyn_cast<BlockArgument>(v); arg && argument && arg.getOwner() == &inside->front()) {
      auto x = argument(arg);
      if (succeeded(x)) m.map(v, *x);
      return x;
    }
    Operation *def = v.getDefiningOp();
    if (!def || !(Sink::pure(def) || def->hasTrait<OpTrait::ConstantLike>() ||
                  (isa<memref::LoadOp>(def) && hostCell(cast<memref::LoadOp>(def).getMemref()))))
      return failure();
    for (Value x : def->getOperands())
      if (failed(recompute(h, x, m, inside, argument))) return failure();
    Operation *copy = h.clone(*def, m);
    return copy->getResult(cast<OpResult>(v).getResultNumber());
  }
  // A slot for `v`, filled at the chunk site by `value`.
  FailureOr<int64_t> addSlot(Type type, std::function<FailureOr<Value>(OpBuilder &)> value) {
    if (!slotType(type) || slots + slotWords(type) > mirrorWords) return failure();
    int64_t base = slots;
    slots += slotWords(type);
    producers.push_back([value](OpBuilder &h, SmallVector<Value> &words) -> LogicalResult {
      auto x = value(h);
      if (failed(x)) return failure();
      Location at = x->getLoc();
      if (auto vec = dyn_cast<VectorType>(x->getType()))
        for (int64_t j = 0; j < vec.getNumElements(); ++j)
          words.push_back(toWord(h, at, vector::ExtractOp::create(h, at, *x, j)));
      else
        words.push_back(toWord(h, at, *x));
      return success();
    });
    return base;
  }
  bool slotFailed = false;
  void initCells() {
    loop.walk([&](memref::StoreOp st) { storedInLoop.insert(st.getMemref()); });
    cells.isLeaf = [this](Value v) {
      auto load = v.getDefiningOp<memref::LoadOp>();
      return load && hostCell(load.getMemref());
    };
    cells.slot = [this](OpBuilder &k, Location at, Value v) -> Value {
      auto it = slotOf.find(v);
      if (it == slotOf.end()) {
        auto base = addSlot(v.getType(), [this, v](OpBuilder &h) {
          IRMapping m;
          return recompute(h, v, m, record);
        });
        if (failed(base)) { slotFailed = true; return v; }
        it = slotOf.insert({v, *base}).first;
      }
      return loadSlot(k, at, mirrorMemory(), it->second, v.getType());
    };
  }
  // A non-pure op of the host outside the kernels must not take a varying
  // value: what it computed at capture would be replayed.
  LogicalResult checkHost(Region &region, Sink &sink, const llvm::SmallDenseSet<Operation *> &done) {
    WalkResult r = region.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (isa<gpu::LaunchOp>(op)) {
        for (Value x : op->getOperands())
          if (sink.varying(x)) {
            op->emitError("step_graph: the cell sets a launch dimension in the TOWERS chunk");
            return WalkResult::interrupt();
          }
        return WalkResult::skip();
      }
      if (Sink::pure(op) || done.contains(op)) return WalkResult::advance();
      for (Value x : op->getOperands())
        if (sink.varying(x)) {
          op->emitError("step_graph: the cell reaches host work in the TOWERS chunk");
          return WalkResult::interrupt();
        }
      return WalkResult::advance();
    });
    return failure(r.wasInterrupted());
  }
  // Each kernel of the clone reads its cell arguments from the mirror. A
  // PME wrapper that takes the cell gets a variant whose kernels do the
  // same; the host computes their slots through the wrapper's arithmetic.
  LogicalResult sinkCells(Operation *clone, llvm::SmallDenseSet<Operation *> &done) {
    if (auto launch = dyn_cast<gpu::LaunchFuncOp>(clone)) {
      for (Value x : launch.getKernelOperands())
        if (cells.varying(x))
          return launch.emitError("step_graph: an outlined TOWERS kernel takes the cell by value");
      return success();
    }
    if (auto call = dyn_cast<func::CallOp>(clone); call && call.getCallee().starts_with("mdrt_gpu_pme_")) {
      llvm::SmallDenseSet<unsigned> taken;
      for (auto [i, x] : llvm::enumerate(call.getArgOperands()))
        if (cells.varying(x)) taken.insert(i);
      if (taken.empty()) return success();
      auto original = module.lookupSymbol<func::FuncOp>(call.getCallee());
      auto variant = original.clone();
      std::string name;
      do name = (call.getCallee() + "_cell" + Twine(++variants)).str();
      while (module.lookupSymbol(name));
      variant.setSymName(name);
      module.push_back(variant);
      Value device = mirrorMemory();
      unsigned n = variant.getNumArguments();
      if (failed(variant.insertArgument(n, device.getType(), {}, loc))) return failure();
      Value memory = variant.getArgument(n);
      call.getArgOperandsMutable().append(device);
      Region *body = &variant.getBody();
      Sink local;
      local.isLeaf = [&](Value v) {
        auto arg = dyn_cast<BlockArgument>(v);
        return arg && arg.getOwner() == &body->front() && taken.contains(arg.getArgNumber());
      };
      DenseMap<Value, int64_t> own;
      local.slot = [&](OpBuilder &k, Location at, Value v) -> Value {
        auto it = own.find(v);
        if (it == own.end()) {
          Operation *site = call;
          auto base = addSlot(v.getType(), [this, v, site, body](OpBuilder &h) -> FailureOr<Value> {
            // The wrapper's arguments are the call's operands, recomputed
            // at the chunk site.
            IRMapping m, outer;
            return recompute(h, v, m, body, [&](BlockArgument arg) {
              return recompute(h, site->getOperand(arg.getArgNumber()), outer, record);
            });
          });
          if (failed(base)) { slotFailed = true; return v; }
          it = own.insert({v, *base}).first;
        }
        return loadSlot(k, at, memory, it->second, v.getType());
      };
      variant.walk([&](gpu::LaunchOp launch) { local.sink(launch.getBody(), loc); });
      if (slotFailed) return call.emitError("step_graph: the cell of a PME wrapper has no slot");
      if (failed(checkHost(variant.getBody(), local, {}))) return failure();
      call.setCallee(variant.getSymName());
      done.insert(call);
      return success();
    }
    clone->walk([&](gpu::LaunchOp launch) { cells.sink(launch.getBody(), loc); });
    if (slotFailed) return clone->emitError("step_graph: a cell argument has no slot");
    return success();
  }

  LogicalResult examine() {
    if (!loop.getInductionVar().use_empty() || !matchPattern(loop.getLowerBound(), m_Zero()) ||
        !matchPattern(loop.getStep(), m_One()))
      return loop.emitError("step_graph: unsupported induction in the TOWERS chunk");
    for (auto &op : *loop.getBody()) {
      if (auto f = dyn_cast<func::CallOp>(op)) {
        StringRef n = f.getCallee();
        if (n == "mdrtFlagStart" || n == "mdrtFlagTake") {
          Value v = f.getOperand(0);
          for (int i = 0; i < 3; ++i) {
            Operation *d = v.getDefiningOp();
            if (!d || !isa<LLVM::IntToPtrOp, arith::IndexCastOp,
                           memref::ExtractAlignedPointerAsIndexOp>(d))
              return f.emitError("step_graph: expected a mapped constraint flag");
            v = d->getOperand(0);
          }
          if (!loop.isDefinedOutsideOfLoop(v)) return f.emitError("step_graph: unstable flag");
          flags.push_back(v);
        }
        if (n.starts_with("mdrt_gpu_pme_weights")) positions = f.getOperand(0);
      }
      if (auto st = dyn_cast<memref::StoreOp>(op)) {
        auto add = st.getValue().getDefiningOp<arith::AddIOp>();
        auto ld = add ? add.getLhs().getDefiningOp<memref::LoadOp>() : nullptr;
        if (!ld || ld.getMemref() != st.getMemref() || !matchPattern(add.getRhs(), m_One()))
          return st.emitError("step_graph: unrecognized host state in TOWERS chunk");
        counts.push_back(st);
      }
      if (auto launch = dyn_cast<gpu::LaunchFuncOp>(op))
        if (launch.getKernelName().getValue().starts_with("towers_finish")) {
          certificate = launch.getKernelOperands()[5];
          finish = SymbolTable::lookupNearestSymbolFrom<gpu::GPUFuncOp>(launch, launch.getKernelAttr());
        }
    }
    SmallVector<Value> bySet(4);
    unsigned next = 0;
    auto checked = loop.walk([&](func::CallOp f) {
      if (f.getCallee() != "mdrtStopConstraintFailure") return WalkResult::advance();
      IntegerAttr set;
      if (!matchPattern(f.getOperand(1), m_Constant(&set)) || set.getInt() < 0 ||
          set.getInt() >= 4 || bySet[set.getInt()] || next >= flags.size())
        return WalkResult::interrupt();
      bySet[set.getInt()] = flags[next++];
      return WalkResult::advance();
    });
    if (checked.wasInterrupted() || next != flags.size() || !next)
      return loop.emitError("step_graph: unsupported constraint sets in plain TOWERS step");
    flags = std::move(bySet);
    if (!finish || !certificate || !positions || counts.size() != 2)
      return loop.emitError("step_graph: unsupported TOWERS check or bookkeeping layout");
    return success();
  }

  // The flag is uniform across a launch, including cooperative grids.
  static void guardBody(Block &block, Location loc, Value fatal) {
    OpBuilder b(&block, block.begin());
    Value ptr = LLVM::IntToPtrOp::create(b, loc, LLVM::LLVMPointerType::get(b.getContext()), fatal);
    Value status = LLVM::LoadOp::create(b, loc, b.getI32Type(), ptr);
    Value ok = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, status,
                                     arith::ConstantIntOp::create(b, loc, 0, 32));
    auto cond = scf::IfOp::create(b, loc, ok, false);
    cond.getThenRegion().front().getOperations().splice(
        cond.thenYield()->getIterator(), block.getOperations(), std::next(cond->getIterator()),
        block.getTerminator()->getIterator());
  }

  LogicalResult guardLaunches(Operation *op, Value fatal) {
    if (auto call = dyn_cast<func::CallOp>(op)) {
      if (!call.getCallee().starts_with("mdrt_gpu_pme_")) return success();
      // Host boundary steps share the original PME wrapper.
      auto original = module.lookupSymbol<func::FuncOp>(call.getCallee());
      std::string name = (call.getCallee() + "_guarded").str();
      auto guarded = module.lookupSymbol<func::FuncOp>(name);
      if (!guarded) {
        guarded = original.clone();
        guarded.setSymName(name);
        module.push_back(guarded);
        unsigned n = guarded.getNumArguments();
        if (failed(guarded.insertArgument(n, i64, {}, loc))) return failure();
        guarded.walk([&](gpu::LaunchOp launch) {
          guardBody(launch.getBody().front(), loc, guarded.getArgument(n));
        });
      }
      call.setCallee(name);
      call.getArgOperandsMutable().append(fatal);
    } else if (auto launch = dyn_cast<gpu::LaunchFuncOp>(op)) {
      auto kernel = SymbolTable::lookupNearestSymbolFrom<gpu::GPUFuncOp>(launch, launch.getKernelAttr());
      unsigned n = kernel.getFunctionType().getNumInputs();
      if (failed(kernel.insertArgument(n, i64, {}, loc))) return failure();
      guardBody(kernel.getBody().front(), loc, kernel.getArgument(n));
      launch.getKernelOperandsMutable().append(fatal);
    } else {
      op->walk([&](gpu::LaunchOp launch) { guardBody(launch.getBody().front(), loc, fatal); });
    }
    return success();
  }

  // Preserve the host descriptor's shape and strides when changing its
  // physical bank. The owning allocations outlive the schedule.
  Value bankView(Value bank, Value raw) {
    auto type = cast<MemRefType>(bank.getType());
    auto ptr = LLVM::LLVMPointerType::get(b.getContext(), type.getMemorySpaceAsInt());
    auto dims = LLVM::LLVMArrayType::get(i64, type.getRank());
    auto desc = LLVM::LLVMStructType::getLiteral(b.getContext(), {ptr, ptr, i64, dims, dims});
    Value addr = LLVM::IntToPtrOp::create(b, loc, ptr, raw);
    Value value = UnrealizedConversionCastOp::create(b, loc, desc, bank).getResult(0);
    for (int64_t field : {0, 1})
      value = LLVM::InsertValueOp::create(b, loc, value, addr, ArrayRef<int64_t>{field});
    return UnrealizedConversionCastOp::create(b, loc, type, value).getResult(0);
  }

  LogicalResult lower(int64_t site) {
    if (failed(examine())) return failure();
    // The finish entry is private to this plain force site. Publication
    // already skips a fatal result; the guarded kick reads its status.
    SmallVector<LLVM::InlineAsmOp> traps;
    finish.walk([&](LLVM::InlineAsmOp op) { if (op.getAsmString() == "trap;") traps.push_back(op); });
    if (traps.size() != 1) return loop.emitError("step_graph: expected one fatal force publication path");
    traps.front().erase();

    Value k = arith::IndexCastOp::create(b, loc, i64, loop.getUpperBound());
    SmallVector<Value> old;
    for (auto st : counts) {
      // The zero index from inside the original loop cannot escape it.
      SmallVector<Value> indices;
      for (Value index : st.getIndices()) {
        if (!matchPattern(index, m_Zero())) return st.emitError("step_graph: nonconstant counter index");
        indices.push_back(arith::ConstantIndexOp::create(b, loc, 0));
      }
      old.push_back(memref::LoadOp::create(b, loc, st.getMemref(), indices));
    }
    auto args = loop.getRegionIterArgs();
    auto init = loop.getInitArgs();
    auto yielded = loop.getBody()->getTerminator()->getOperands();
    IRMapping map;
    for (auto [arg, v] : llvm::zip(args, init)) map.map(arg, v);
    // Record the actual force inputs as well as the rotating state. An
    // enclosing output or coupling loop can supply a new scratch binding.
    llvm::SetVector<Value> external;
    getUsedValuesDefinedAbove(loop.getRegion(), external);
    SmallVector<Value> bindings;
    for (Value v : external)
      if (isa<MemRefType>(v.getType())) bindings.push_back(v);
    auto workspaces = loop.getBody()->getOps<gpu::AllocOp>();
    size_t workspaceCount = std::distance(workspaces.begin(), workspaces.end());
    Value table;
    {
      OpBuilder::InsertionGuard at(b);
      b.setInsertionPointAfterValue(cache);
      table = memref::AllocaOp::create(b, loc, MemRefType::get({int64_t(3*args.size() + bindings.size() + workspaceCount)}, i64));
    }
    auto recordWrites = [&](Operation &op, llvm::SmallDenseSet<Value> &banks) {
      op.walk([&](memref::StoreOp store) { banks.insert(store.getMemref()); });
      if (auto call = dyn_cast<func::CallOp>(op))
        if (call.getCallee().starts_with("mdrt_gpu_pme_gather_weights"))
          banks.insert(call.getOperands().back());
      if (auto launch = dyn_cast<gpu::LaunchFuncOp>(op))
        if (launch.getKernelName().getValue().starts_with("towers_finish"))
          banks.insert(launch.getKernelOperands().back());
    };
    llvm::SmallDenseSet<Value> written;
    for (auto &op : loop.getBody()->without_terminator()) recordWrites(op, written);
    llvm::SmallDenseSet<Value> sources;
    for (size_t i = 0; i < args.size(); ++i) {
      auto type = dyn_cast<MemRefType>(args[i].getType());
      auto src = llvm::find(args, yielded[i]);
      if (!type || !type.getLayout().isIdentity() || src == args.end() ||
          !sources.insert(*src).second || src->getType() != type)
        return loop.emitError("step_graph: expected unstrided permutation of particle banks");
      Value bytes = word(type.getElementTypeBitWidth()/8);
      for (unsigned axis = 0; axis < type.getRank(); ++axis) {
        Value dim = memref::DimOp::create(b, loc, init[i], axis);
        bytes = arith::MulIOp::create(b, loc, bytes, arith::IndexCastOp::create(b, loc, i64, dim));
      }
      int64_t field = src - args.begin();
      if (written.contains(args[i])) field += args.size();
      SmallVector<Value> fields{address(init[i]), bytes, word(field)};
      for (auto [j, v] : llvm::enumerate(fields))
        memref::StoreOp::create(b, loc, v, table, ValueRange{arith::ConstantIndexOp::create(b, loc, 3*i+j)});
    }
    // Hoist per-half work arrays above the outer schedule loop. Bind every
    // captured use to this allocation until the last graph use in the run.
    Operation *outer = loop;
    while (auto parent = outer->getParentOfType<scf::ForOp>()) outer = parent;
    for (auto alloc : workspaces) {
      auto type = alloc.getType();
      if (type.getRank() != 1 || !type.getElementType().isInteger(32) ||
          alloc.getDynamicSizes().size() != 1 ||
          !alloc.getDynamicSizes()[0].getDefiningOp<memref::DimOp>())
        return alloc.emitError("step_graph: unsupported per-step workspace");
      OpBuilder hoist(outer);
      Value dim = arith::ConstantIndexOp::create(hoist, loc, atoms);
      Value buf = gpu::AllocOp::create(hoist, loc, type, Type(), ValueRange{},
                                        ValueRange{dim}, ValueRange{}, false).getMemref();
      map.map(alloc.getMemref(), buf);
      bindings.push_back(buf);
      hoist.setInsertionPointAfter(outer);
      auto generic = MemRefType::get(type.getShape(), type.getElementType());
      Value freed = memref::MemorySpaceCastOp::create(hoist, loc, generic, buf);
      gpu::DeallocOp::create(hoist, loc, Type(), ValueRange{}, freed);
    }

    for (auto [i, v] : llvm::enumerate(bindings)) {
      Value at = arith::ConstantIndexOp::create(b, loc, 3*args.size()+i);
      memref::StoreOp::create(b, loc, address(v), table, ValueRange{at});
    }
    Value miss = call("mdrtStepBegin", {cache, word(site), word(args.size()),
                                       word(bindings.size()), address(table)}).getResult(0);
    Value pred = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne, miss,
                                        arith::ConstantIntOp::create(b, loc, 0, 32));
    auto record = scf::IfOp::create(b, loc, pred, false);
    b.setInsertionPoint(record.thenYield());
    // Create and retain the exact cuFFT plans before starting stream capture.
    // Gaussian inverse callbacks bind the fixed-cell influence table.
    for (auto fft : loop.getBody()->getOps<func::CallOp>()) {
      StringRef name = fft.getCallee();
      if (!name.starts_with("mdrtCudaFFT")) continue;
      SmallVector<Value> inputs{cache};
      for (Value dim : fft.getOperands().take_back(3)) {
        IntegerAttr size;
        if (!matchPattern(dim, m_Constant(&size)))
          return fft.emitError("step_graph: FFT grid must be fixed");
        inputs.push_back(word(size.getInt()));
      }
      inputs.push_back(word(name.starts_with("mdrtCudaFFTForward")));
      inputs.push_back(word(name.ends_with("F32")));
      Value tables = word(0);
      if (name == "mdrtCudaFFTGaussianBackwardF32") {
        Value table = fft.getOperand(2);
        if (!loop.isDefinedOutsideOfLoop(table))
          return fft.emitError("step_graph: callback table must be loop invariant");
        tables = address(table);
      }
      inputs.push_back(tables);
      call("mdrtStepPlan", inputs);
    }
    call("mdrtStepCapture", cache);
    Value row;
    {
      Value zero = arith::ConstantIndexOp::create(b, loc, 0);
      Value one = arith::ConstantIndexOp::create(b, loc, 1);
      Value count = call("mdrtStepPhases", cache).getResult(0);
      Value end = arith::IndexCastOp::create(b, loc, b.getIndexType(), count);
      auto phases = scf::ForOp::create(b, loc, zero, end, one);
      b.setInsertionPoint(phases.getBody()->getTerminator());
      row = call("mdrtStepBankRow", cache).getResult(0);
      for (auto [i, arg] : llvm::enumerate(args)) {
        Value raw = call("mdrtStepBank", {cache, word(i), word(1)}).getResult(0);
        map.map(arg, bankView(init[i], raw));
      }
    }
    Value fatal = call("mdrtStepFatalAddress", cache).getResult(0);
    this->record = &record.getThenRegion();
    initCells();
    llvm::SmallDenseSet<Operation *> sunk;
    llvm::SmallDenseSet<Value> defined;
    bool drift = true;
    bool guardIntegration = false;
    // Whether a side stream was forked and not yet joined.
    bool sideOpen = false;
    for (Operation &op : loop.getBody()->without_terminator()) {
      if (isa<gpu::AllocOp, gpu::DeallocOp, memref::StoreOp>(op)) continue;
      if (auto f = dyn_cast<func::CallOp>(op)) {
        StringRef n = f.getCallee();
        if (n == "mdrtFlagStart") {
          if (sideOpen)
            return f.emitError("step_graph: a side-stream window crosses "
                               "the end of the drift phase");
          drift = false;
          SmallVector<Value> inputs{cache,address(map.lookupOrDefault(positions)),word(atoms)};
          for (Value v : flags) inputs.push_back(v ? address(v) : word(0));
          call("mdrtStepDrift", inputs);
          map.map(f.getResult(0), word(0));
          continue;
        }
        if (n == "mdrtFlagFinish" || n == "mdrtFlagTake") {
          if (n == "mdrtFlagFinish") {
            guardIntegration = true;
            call("mdrtStepForce", {cache,address(map.lookupOrDefault(certificate))});
          }
          map.map(f.getResult(0), arith::ConstantIntOp::create(b, loc, 0, 32));
          continue;
        }
        if (n == "mdrtDeferStops" || n == "mdrtStopPending" || n == "mdrtCountBuild") continue;
        // The forks and joins of the side streams are event records and
        // waits, which the capture follows into the side streams; a window
        // must open and close within one phase (D449).
        bool sideCall = n == "mdrtSideBegin" || n == "mdrtSideEnd" ||
                        n == "mdrtSide2Begin" || n == "mdrtSide2End" ||
                        n == "mdrtSideJoin";
        if (n == "mdrtSideBegin" || n == "mdrtSide2Begin")
          sideOpen = true;
        else if (n == "mdrtSideJoin")
          sideOpen = false;
        if (!(n.starts_with("mdrt_gpu_pme_") || n.starts_with("mdrtCudaFFT") ||
              n == "mdrtCooperativeGridSize" || sideCall))
          return f.emitError("step_graph: unexpected host call in plain TOWERS graph");
      }
      // A fork or join inside a region would escape the phase check above.
      if (op.getNumRegions() &&
          op.walk([](func::CallOp f) {
              return f.getCallee().starts_with("mdrtSide")
                         ? WalkResult::interrupt()
                         : WalkResult::advance();
            }).wasInterrupted())
        return op.emitError("step_graph: a side-stream call inside a region "
                            "of the plain TOWERS step");
      if (auto cond = dyn_cast<scf::IfOp>(op))
        if (cond.walk([](func::CallOp f) {
              return f.getCallee() == "mdrtStopConstraintFailure" ? WalkResult::interrupt() :
                                                                  WalkResult::advance();
            }).wasInterrupted())
          continue;
      auto *clone = b.clone(op, map);
      if (failed(sinkCells(clone, sunk))) return failure();
      // A kernel reads the most recent definition of each field and
      // writes into the proposal. Constraint kernels consume the
      // unconstrained fields produced by the preceding launch.
      if (auto launch = dyn_cast<gpu::LaunchOp>(clone); launch && (drift || guardIntegration)) {
        OpBuilder::InsertionGuard at(b);
        b.setInsertionPointToStart(&launch.getBody().front());
        DenseMap<unsigned, Value> views;
        auto bind = [&](Value bank, bool read) -> Value {
          auto it = llvm::find(args, bank);
          if (it == args.end()) return {};
          unsigned i = it - args.begin();
          unsigned slot = read && !defined.contains(bank) ? i : args.size() + i;
          auto &view = views[slot];
          if (!view) {
            Value address = arith::AddIOp::create(b, loc, row, word(slot * sizeof(int64_t)));
            Value ptr = LLVM::IntToPtrOp::create(b, loc, LLVM::LLVMPointerType::get(b.getContext()), address);
            Value raw = LLVM::LoadOp::create(b, loc, i64, ptr);
            view = bankView(init[i], raw);
          }
          return view;
        };
        op.walk([&](memref::LoadOp load) {
          if (Value view = bind(load.getMemref(), true))
            cast<memref::LoadOp>(map.lookup(load.getOperation())).getMemrefMutable().assign(view);
        });
        op.walk([&](memref::StoreOp store) {
          if (Value view = bind(store.getMemref(), false))
            cast<memref::StoreOp>(map.lookup(store.getOperation())).getMemrefMutable().assign(view);
        });
      }
      recordWrites(op, defined);
      if (failed(guardLaunches(clone, fatal))) return failure();
      if (guardIntegration) {
        if (isa<gpu::LaunchFuncOp>(clone))
          return clone->emitError("step_graph: expected inline integration kernel before outlining");
        clone->walk([&](gpu::LaunchOp launch) {
          // All threads observe the same force certificate before any
          // shared-memory operation, barrier or particle store.
          Block &block = launch.getBody().front();
          OpBuilder guard(&block, block.begin());
          Value zero = arith::ConstantIndexOp::create(guard, loc, 0);
          Value status = memref::LoadOp::create(guard, loc, map.lookupOrDefault(certificate), zero);
          Value ok = arith::CmpIOp::create(guard, loc, arith::CmpIPredicate::eq, status,
                                           arith::ConstantIntOp::create(guard, loc, 0, 32));
          auto cond = scf::IfOp::create(guard, loc, ok, false);
          auto &ops = block.getOperations();
          cond.getThenRegion().front().getOperations().splice(
              cond.thenYield()->getIterator(), ops, std::next(cond->getIterator()),
              block.getTerminator()->getIterator());
        });
      }
    }
    if (sideOpen)
      return loop.emitError("step_graph: a side-stream window is not joined "
                            "within the step");
    call("mdrtStepEnd", cache);
    if (failed(checkHost(record.getThenRegion(), cells, sunk))) return failure();
    b.setInsertionPointAfter(record);
    // The host computes the cell's arguments of this chunk, as the step's
    // launches would, and the graph reads them from the mirror.
    if (slots) {
      Value staging;
      {
        OpBuilder::InsertionGuard at(b);
        b.setInsertionPointAfterValue(cache);
        staging = memref::AllocaOp::create(b, loc, MemRefType::get({mirrorWords}, i64));
      }
      SmallVector<Value> words;
      for (auto &produce : producers)
        if (failed(produce(b, words)))
          return loop.emitError("step_graph: the host cannot compute a cell argument at the chunk");
      for (auto [i, w] : llvm::enumerate(words))
        memref::StoreOp::create(b, loc, w, staging, ValueRange{arith::ConstantIndexOp::create(b, loc, i)});
      call("mdrtStepMirror", {cache, address(mirror), address(staging), word(8 * slots)});
    }
    Value accepted = call("mdrtStepRun", {cache,old.front(),k}).getResult(0);
    call("mdrtStepCheck", {cache, accepted});
    for (auto [st, prev] : llvm::zip(counts, old)) {
      Value next = st == counts.front() ? accepted : arith::AddIOp::create(b, loc, prev, k).getResult();
      SmallVector<Value> indices(st.getIndices().size(), arith::ConstantIndexOp::create(b, loc, 0));
      memref::StoreOp::create(b, loc, next, st.getMemref(), indices);
    }
    for (auto [i, res] : llvm::enumerate(loop.getResults()))
      res.replaceAllUsesWith(bankView(init[i], call("mdrtStepBank", {cache, word(i), word(2)}).getResult(0)));
    loop.erase();
    return success();
  }
};
}

namespace mdir {
LogicalResult lowerTowersChunks(ModuleOp module, func::FuncOp fn, ArrayRef<scf::ForOp> loops) {
  OpBuilder b(fn);
  auto i64 = b.getI64Type();
  auto declare = [&](StringRef n, unsigned count, TypeRange out, unsigned doubles = 0) {
    b.setInsertionPointToEnd(module.getBody());
    SmallVector<Type> types(count, i64);
    types.append(doubles, b.getF64Type());
    auto f = func::FuncOp::create(b, fn.getLoc(), n, b.getFunctionType(types, out));
    f.setPrivate();
    return f;
  };
  auto create = declare("mdrtStepCreate", 0, i64);
  declare("mdrtStepPlan", 7, {});
  declare("mdrtStepCapture", 1, {});
  declare("mdrtStepBank", 3, i64);
  declare("mdrtStepBankRow", 1, i64);
  declare("mdrtStepPhases", 1, i64);
  declare("mdrtStepFatalAddress", 1, i64);
  declare("mdrtStepBegin", 5, b.getI32Type());
  declare("mdrtStepDrift", 7, {});
  declare("mdrtStepForce", 2, {});
  declare("mdrtStepEnd", 1, {});
  declare("mdrtStepRun", 3, i64);
  declare("mdrtStepCheck", 2, {});
  declare("mdrtStepMirror", 4, {});
  auto seed = declare("mdrtStepSeed", 2, {}, 2);
  auto boundary = declare("mdrtStepBoundary", 1, {});
  auto read = declare("mdrtStepRead", 2, i64);
  auto destroy = declare("mdrtStepDestroy", 1, {});
  b.setInsertionPointToStart(&fn.front());
  Value cache = func::CallOp::create(b, fn.getLoc(), create).getResult(0);
  Value dt = arith::ConstantOp::create(b, fn.getLoc(), module->getAttrOfType<FloatAttr>("mdir.towers_dt"));
  auto originAttr = module->getAttrOfType<FloatAttr>("mdir.towers_start_time");
  Value origin = arith::ConstantOp::create(b, fn.getLoc(),
                              originAttr ? originAttr : b.getF64FloatAttr(0));
  Chunk first(module, loops.front(), cache);
  if (failed(first.examine())) return failure();
  Value stepMemory = first.counts.front().getMemref();
  memref::StoreOp initial;
  for (auto store : fn.front().getOps<memref::StoreOp>())
    if (store.getMemref() == stepMemory) { initial = store; break; }
  if (!initial) return fn.emitError("step_graph: missing TOWERS graph restart counter");
  b.setInsertionPointAfter(initial);
  func::CallOp::create(b, fn.getLoc(), seed, ValueRange{cache, initial.getValue(), origin, dt});
  int number = 0;
  for (auto loop : loops)
    if (failed(Chunk(module, loop, cache).lower(++number))) return failure();
  // The host launch path runs coupling and output steps. Advance the same
  // clock there and give the observed accepted number to Philox and output.
  SmallVector<func::CallOp> calls;
  fn.walk([&](func::CallOp call) { calls.push_back(call); });
  // The second draw of a coupling step on the device takes the step that the
  // first read, which precedes it in its block (D452).
  DenseMap<std::pair<Block *, Value>, Value> drawn;
  for (auto f : calls) {
    StringRef name = f.getCallee();
    if (name == "mdrtStopPending") {
      b.setInsertionPointAfter(f);
      func::CallOp::create(b, f.getLoc(), boundary, ValueRange{cache});
    } else if (name == "mdrtBussiFactor" || name == "mdrtBussiNormal" ||
               name == "mdrtBussiRest" || name == "mdrtWriteEnergies") {
      // The draws of a coupling step on the device take the step as the
      // factor did (D452).
      unsigned slot = name == "mdrtWriteEnergies" ? 0 : 1;
      std::pair<Block *, Value> key{f->getBlock(), f.getOperand(slot)};
      if (name == "mdrtBussiRest")
        if (Value step = drawn.lookup(key)) {
          f.setOperand(slot, step);
          continue;
        }
      b.setInsertionPoint(f);
      auto n = func::CallOp::create(b, f.getLoc(), read, ValueRange{cache, f.getOperand(slot)});
      f.setOperand(slot, n.getResult(0));
      if (name == "mdrtBussiNormal")
        drawn[key] = n.getResult(0);
    }
  }
  fn.walk([&](func::ReturnOp ret) {
    b.setInsertionPoint(ret);
    func::CallOp::create(b, fn.getLoc(), destroy, ValueRange{cache});
  });
  return success();
}
}
