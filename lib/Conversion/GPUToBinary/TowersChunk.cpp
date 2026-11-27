// Lower the plain constrained step of the fixed-cell Gaussian PME NVT
// protocol after its storage lifetime and checks are explicit.
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/RegionUtils.h"
#include <array>
#include <set>

using namespace mlir;
namespace {
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
    b.setInsertionPointAfter(record);
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
  for (auto f : calls) {
    StringRef name = f.getCallee();
    if (name == "mdrtStopPending") {
      b.setInsertionPointAfter(f);
      func::CallOp::create(b, f.getLoc(), boundary, ValueRange{cache});
    } else if (name == "mdrtBussiFactor" || name == "mdrtWriteEnergies") {
      unsigned slot = name == "mdrtBussiFactor" ? 1 : 0;
      b.setInsertionPoint(f);
      auto n = func::CallOp::create(b, f.getLoc(), read, ValueRange{cache, f.getOperand(slot)});
      f.setOperand(slot, n.getResult(0));
    }
  }
  fn.walk([&](func::ReturnOp ret) {
    b.setInsertionPoint(ret);
    func::CallOp::create(b, fn.getLoc(), destroy, ValueRange{cache});
  });
  return success();
}
}
