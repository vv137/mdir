// The coupling step on the device (D[device-coupling], issue #1026).
//
// The host computes the factor of stochastic velocity rescaling, the heat
// that goes to the bath, and (under a barostat) the strain and the cell
// from global sums of the step, so every coupling step copied the sums to
// the host and waited for them. Here the arithmetic on the sums is left to
// the device:
//
// - the draws of the factor depend only on the seed and the step, so the
//   host makes them as before (mdrtBussiNormal, mdrtBussiRest) and the rest
//   of mdrtBussiFactor is written as ops, in its order of operations;
// - the factor is computed once, by one thread, into a slot of the results
//   of the function on the device, which the kernels that scale read;
// - the bath is a slot of the same buffer: one thread adds each heat to it,
//   in the order of the program, from the value of the host, which takes it
//   back before every row, checkpoint and return.
//
// keepReadbacksOnDevice then clones what the one-thread kernels take from
// the sums into them, and removes the copies that the host no longer reads.
// The kernels are marked as host arithmetic: no contraction and no explicit
// fused multiply-add, so their rounding is that of the host.
#include "DeviceCoupling.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;

namespace {
func::FuncOp declare(ModuleOp module, StringRef name, FunctionType type) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(name))
    return existing;
  OpBuilder b = OpBuilder::atBlockBegin(module.getBody());
  auto function = func::FuncOp::create(b, module.getLoc(), name, type);
  function.setPrivate();
  return function;
}

/// A kernel of one thread at `b` that `body` fills.
gpu::LaunchOp launchOne(OpBuilder &b, Location loc,
                        function_ref<void(OpBuilder &)> body) {
  Value one = arith::ConstantIndexOp::create(b, loc, 1);
  auto launch = gpu::LaunchOp::create(b, loc, one, one, one, one, one, one);
  OpBuilder kernel = OpBuilder::atBlockEnd(&launch.getBody().front());
  body(kernel);
  gpu::TerminatorOp::create(kernel, loc);
  launch->setAttr("mdir.frame_host_arithmetic", b.getUnitAttr());
  launch->setAttr("mdir.coupling_step", b.getUnitAttr());
  return launch;
}

/// Whether `op` is a launch or holds one.
bool launches(Operation *op) {
  return op->walk([](gpu::LaunchOp) { return WalkResult::interrupt(); })
      .wasInterrupted();
}

/// The ops of mdrtBussiFactor (runtime/mdrt.c) after its draws, in the
/// order in which C evaluates them, at `call`:
///   next = K + (1 - c) (T (r1² + rest) / N - K)
///          + 2 r1 √(K T / N (1 - c) c),
///   α = √(next / K), or 1 unless K > 0 and N > 0.
Value expandBussi(func::CallOp call) {
  ModuleOp module = call->getParentOfType<ModuleOp>();
  OpBuilder b(call);
  Location loc = call.getLoc();
  Type f64 = b.getF64Type(), i64 = b.getI64Type();
  Value seed = call.getOperand(0), step = call.getOperand(1),
        kinetic = call.getOperand(2), target = call.getOperand(3),
        freedom = call.getOperand(4), decay = call.getOperand(5);
  auto normal = declare(module, "mdrtBussiNormal",
                        b.getFunctionType({i64, i64}, {f64}));
  auto rest = declare(module, "mdrtBussiRest",
                      b.getFunctionType({i64, i64, f64}, {f64}));
  Value r1 = func::CallOp::create(b, loc, normal, ValueRange{seed, step})
                 .getResult(0);
  Value others =
      func::CallOp::create(b, loc, rest, ValueRange{seed, step, freedom})
          .getResult(0);
  auto real = [&](double v) -> Value {
    return arith::ConstantOp::create(b, loc, b.getF64FloatAttr(v));
  };
  auto add = [&](Value x, Value y) -> Value {
    return arith::AddFOp::create(b, loc, x, y);
  };
  auto sub = [&](Value x, Value y) -> Value {
    return arith::SubFOp::create(b, loc, x, y);
  };
  auto mul = [&](Value x, Value y) -> Value {
    return arith::MulFOp::create(b, loc, x, y);
  };
  auto div = [&](Value x, Value y) -> Value {
    return arith::DivFOp::create(b, loc, x, y);
  };
  Value squares = add(mul(r1, r1), others);
  Value mean = div(mul(target, squares), freedom);
  Value keep = sub(real(1.0), decay);
  Value drift = mul(keep, sub(mean, kinetic));
  Value spread = mul(mul(div(mul(kinetic, target), freedom), keep), decay);
  Value noise = mul(mul(real(2.0), r1), math::SqrtOp::create(b, loc, spread));
  Value next = add(add(kinetic, drift), noise);
  Value factor = math::SqrtOp::create(b, loc, div(next, kinetic));
  Value positive = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OGT,
                                         kinetic, real(0.0));
  Value counted = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OGT,
                                        freedom, real(0.0));
  Value valid = arith::AndIOp::create(b, loc, positive, counted);
  auto result = arith::SelectOp::create(b, loc, valid, factor, real(1.0));
  result->setAttr("mdir.device_scalar", b.getUnitAttr());
  call.getResult(0).replaceAllUsesWith(result.getResult());
  call.erase();
  return result;
}

/// Replaces the uses of `value` in every kernel of `function` but
/// `producer`, which stores it into `slot` of `device`, with a load of the
/// slot.
void replaceInKernels(func::FuncOp function, Value value,
                      gpu::LaunchOp producer, Value device, int64_t slot) {
  Location loc = value.getLoc();
  function.walk([&](gpu::LaunchOp launch) {
    if (launch == producer)
      return;
    Region &body = launch.getBody();
    auto inside = [&](OpOperand &use) {
      return body.isAncestor(use.getOwner()->getParentRegion());
    };
    if (!llvm::any_of(value.getUses(), inside))
      return;
    OpBuilder k = OpBuilder::atBlockBegin(&body.front());
    Value at = arith::ConstantIndexOp::create(k, loc, slot);
    Value read = memref::LoadOp::create(k, loc, device, at);
    value.replaceUsesWithIf(read, inside);
  });
}

/// The operand of a strain of stochastic cell rescaling that is the
/// pressure (runtime/mdrt.c).
unsigned pressureOperand(func::CallOp call) {
  return call.getCallee() == "mdrtBarostatStrainAxis" ? 3 : 2;
}

bool isStrain(StringRef name) {
  return name == "mdrtBarostatStrain" || name == "mdrtBarostatStrainArea" ||
         name == "mdrtBarostatStrainHeight" || name == "mdrtBarostatStrainAxis";
}

/// The strains of one scaling: those of a block with only ops without
/// effects between them.
SmallVector<SmallVector<func::CallOp>>
groupStrains(ArrayRef<func::CallOp> strains) {
  SmallVector<SmallVector<func::CallOp>> groups;
  for (func::CallOp call : strains) {
    if (!groups.empty()) {
      func::CallOp last = groups.back().back();
      bool together = last->getBlock() == call->getBlock();
      for (Operation *op = last->getNextNode(); together && op != call;
           op = op->getNextNode())
        together = op->getNumRegions() == 0 && isMemoryEffectFree(op);
      if (together) {
        groups.back().push_back(call);
        continue;
      }
    }
    groups.push_back({call});
  }
  return groups;
}
} // namespace

std::string mdir::kernels::placeCouplingOnDevice(func::FuncOp function,
                                                 CouplingResults &results) {
  if (function->hasAttr("mdir.frame_batch"))
    return "";
  SmallVector<func::CallOp> factors, baths, readers, strains;
  StringRef refusal;
  function.walk([&](func::CallOp call) {
    StringRef name = call.getCallee();
    if (name == "mdrtBussiFactor")
      factors.push_back(call);
    else if (name == "mdrtAddBath")
      baths.push_back(call);
    else if (isStrain(name))
      strains.push_back(call);
    else if (name == "mdrtNoseHooverFactor")
      refusal = "the Nose-Hoover chain runs on the host";
    else if (name == "mdrtWriteEnergies" || name == "mdrtFinish" ||
             name.starts_with("mdrtWriteCheckpoint"))
      readers.push_back(call);
  });
  if (factors.empty() && baths.empty() && strains.empty())
    return "";
  if (!refusal.empty())
    return refusal.str();
  // The factor of each coupling and the bath take a slot each.
  if (results.room() < int64_t(factors.size()) + 1)
    return "the results of the function fill their buffer";
  for (func::CallOp call : factors)
    if (call->getParentOfType<gpu::LaunchOp>())
      return "a coupling inside a kernel";

  ModuleOp module = function->getParentOfType<ModuleOp>();
  auto [device, host] = results.cells();

  // The virial that a TOWERS dispatch observes for a scaling arrives in a
  // buffer of its own (TowersDispatch.cpp), which the host copies: as a
  // readback, so that kernels read it on the device, as they read the sums.
  // Each copy takes the loads that follow it in its block, up to the next
  // copy into the same memory; memory with a load that no copy takes so is
  // left as it is.
  llvm::MapVector<Value, SmallVector<gpu::MemcpyOp>> observed;
  function.walk([&](gpu::MemcpyOp copy) {
    auto observer = copy.getSrc().getDefiningOp<gpu::AllocOp>();
    auto attr = observer ? observer->getAttrOfType<StringAttr>(
                               "mdir.towers_buffer")
                         : StringAttr();
    if (attr && attr.getValue() == "observer" && !copy->hasAttr("mdir.readback") &&
        copy.getDst().getDefiningOp<memref::AllocaOp>())
      observed[copy.getDst()].push_back(copy);
  });
  for (auto &[memory, copies] : observed) {
    llvm::DenseMap<Operation *, SmallVector<Operation *>> groups;
    llvm::DenseSet<Operation *> taken;
    for (gpu::MemcpyOp copy : copies) {
      SmallVector<Operation *> &group = groups[copy];
      group.push_back(copy);
      for (Value token : copy.getAsyncDependencies())
        if (auto wait = token.getDefiningOp<gpu::WaitOp>())
          group.push_back(wait);
      for (Operation *user : copy->getUsers())
        if (isa<gpu::WaitOp>(user))
          group.push_back(user);
      for (Operation *op = copy->getNextNode(); op; op = op->getNextNode()) {
        if (auto other = dyn_cast<gpu::MemcpyOp>(op); other && other.getDst() == memory)
          break;
        auto load = dyn_cast<memref::LoadOp>(op);
        if (!load || load.getMemRef() != memory)
          continue;
        taken.insert(load);
        group.push_back(load);
        for (Value i : load.getIndices())
          if (Operation *def = i.getDefiningOp();
              def && def->getNumOperands() == 0 && def->hasOneUse())
            group.push_back(def);
      }
    }
    bool all = llvm::all_of(memory.getUsers(), [&](Operation *user) {
      return !isa<memref::LoadOp>(user) || taken.contains(user);
    });
    if (!all)
      continue;
    // Another evaluation of the step may write the buffer again before a
    // kernel reads it (the virials before and after a scaling), so each copy
    // takes a snapshot of its own on the device, which only it writes, as
    // the helpers of device frames do. Marked apart as well: the copy stays
    // where the dispatch put it.
    auto type = cast<MemRefType>(memory.getType());
    if (!type.hasStaticShape() || type.getRank() != 1 || type.getNumElements() > 16)
      continue;
    for (gpu::MemcpyOp copy : copies) {
      OpBuilder top = OpBuilder::atBlockBegin(&function.getBody().front());
      auto sourceType = cast<MemRefType>(copy.getSrc().getType());
      Value stable = gpu::AllocOp::create(top, copy.getLoc(), sourceType, Type(),
                                          ValueRange(), ValueRange(), ValueRange())
                         .getMemref();
      OpBuilder before(copy);
      for (Value token : copy.getAsyncDependencies())
        if (Operation *wait = token.getDefiningOp<gpu::WaitOp>())
          before.setInsertionPoint(wait);
      Value source = copy.getSrc();
      auto snapshot = launchOne(before, copy.getLoc(), [&](OpBuilder &k) {
        for (int64_t n = 0, e = sourceType.getNumElements(); n != e; ++n) {
          Value i = arith::ConstantIndexOp::create(k, copy.getLoc(), n);
          Value x = memref::LoadOp::create(k, copy.getLoc(), source, i);
          memref::StoreOp::create(k, copy.getLoc(), x, stable, i);
        }
      });
      results.bringIn(snapshot);
      copy.getSrcMutable().assign(stable);
      OpBuilder at(copy);
      IntegerAttr id = at.getI64IntegerAttr(results.readback());
      for (Operation *op : groups[copy]) {
        op->setAttr("mdir.readback", id);
        op->setAttr("mdir.observer_readback", at.getUnitAttr());
      }
    }
  }
  Location loc = function.getLoc();
  OpBuilder b(function.getContext());
  Type f64 = b.getF64Type();
  auto index = [&](OpBuilder &at, int64_t k) -> Value {
    return arith::ConstantIndexOp::create(at, loc, k);
  };

  // The bath: from the host's value at the entry, after the buffer is
  // cleared.
  int64_t bath = results.slot();
  Block &entry = function.getBody().front();
  Operation *cleared = nullptr;
  for (Operation &op : entry) {
    auto launch = dyn_cast<gpu::LaunchOp>(op);
    if (!launch)
      continue;
    bool stores = false;
    launch.getBody().walk([&](memref::StoreOp store) {
      stores |= store.getMemRef() == device;
    });
    if (stores) {
      cleared = launch;
      break;
    }
  }
  if (!cleared)
    return "the results of the function are not cleared at its entry";
  b.setInsertionPointAfter(cleared);
  auto value = declare(module, "mdrtBath", b.getFunctionType({}, {f64}));
  Value start = func::CallOp::create(b, loc, value, ValueRange{}).getResult(0);
  results.bringIn(launchOne(b, loc, [&](OpBuilder &k) {
    memref::StoreOp::create(k, loc, start, device, index(k, bath));
  }));

  // Each heat, added on the device in the order of the program.
  llvm::SetVector<gpu::LaunchOp> bathKernels;
  for (func::CallOp call : baths) {
    OpBuilder at(call);
    Value heat = call.getOperand(0);
    auto launch = launchOne(at, call.getLoc(), [&](OpBuilder &k) {
      Value slot = index(k, bath);
      Value before = memref::LoadOp::create(k, loc, device, slot);
      Value after = arith::AddFOp::create(k, loc, before, heat);
      memref::StoreOp::create(k, loc, after, device, slot);
    });
    results.bringIn(launch);
    bathKernels.insert(launch);
    call.erase();
  }

  // The host takes the bath back where it reads it.
  auto set = declare(module, "mdrtSetBath", b.getFunctionType({f64}, {}));
  auto giveBack = [&](Operation *before) {
    OpBuilder at(before);
    IntegerAttr id = at.getI64IntegerAttr(results.readback());
    Type token = gpu::AsyncTokenType::get(at.getContext());
    auto begin = gpu::WaitOp::create(at, loc, token, ValueRange());
    auto copy = gpu::MemcpyOp::create(at, loc, token,
                                      ValueRange{begin.getAsyncToken()}, host,
                                      device);
    auto done = gpu::WaitOp::create(at, loc, Type(),
                                    ValueRange{copy.getAsyncToken()});
    Value slot = index(at, bath);
    auto load = memref::LoadOp::create(at, loc, host, slot);
    for (Operation *op : {begin.getOperation(), copy.getOperation(),
                          done.getOperation(), load.getOperation(),
                          slot.getDefiningOp()})
      op->setAttr("mdir.readback", id);
    func::CallOp::create(at, loc, set, ValueRange{load.getResult()});
  };
  for (func::CallOp call : readers)
    giveBack(call);
  SmallVector<func::ReturnOp> returns;
  function.walk([&](func::ReturnOp ret) { returns.push_back(ret); });
  for (func::ReturnOp ret : returns)
    giveBack(ret);

  // Each factor, once, into a slot that the kernels that scale read: in the
  // kernel of the bath that follows it, if that comes first, or in one of
  // its own.
  for (func::CallOp call : factors) {
    Value factor = expandBussi(call);
    int64_t slot = results.slot();
    Operation *def = factor.getDefiningOp();
    Operation *first = def->getNextNode();
    while (first && !launches(first))
      first = first->getNextNode();
    auto store = [&](OpBuilder &k) {
      memref::StoreOp::create(k, loc, factor, device, index(k, slot));
    };
    gpu::LaunchOp producer;
    if (auto launch = dyn_cast_or_null<gpu::LaunchOp>(first);
        launch && bathKernels.contains(launch)) {
      producer = launch;
      OpBuilder k(launch.getBody().front().getTerminator());
      store(k);
    } else {
      OpBuilder at = first ? OpBuilder(first)
                           : OpBuilder(def->getBlock()->getTerminator());
      producer = launchOne(at, loc, store);
      results.bringIn(producer);
    }
    replaceInKernels(function, factor, producer, device, slot);
  }

  // The pressures of each scaling of the cell: computed on the device into
  // a few numbers, which the host copies back alone (a small page-locked
  // copy instead of the buffer of the results) and gives the strain, which
  // it computes as before. The host needs the new cell before it issues
  // the next step: the launches and the builds of the neighbor structures
  // take it by value.
  if (!strains.empty()) {
    auto [small, copy] = results.scalars();
    for (auto &group : groupStrains(strains)) {
      if (int64_t(group.size()) > CouplingResults::numScalars)
        continue;
      OpBuilder at(group.front());
      auto producer = launchOne(at, loc, [&](OpBuilder &k) {
        for (auto [n, call] : llvm::enumerate(group))
          memref::StoreOp::create(k, loc, call.getOperand(pressureOperand(call)),
                                  small, index(k, n));
      });
      results.bringIn(producer);
      IntegerAttr id = at.getI64IntegerAttr(results.readback());
      Type token = gpu::AsyncTokenType::get(at.getContext());
      auto begin = gpu::WaitOp::create(at, loc, token, ValueRange());
      auto transfer = gpu::MemcpyOp::create(
          at, loc, token, ValueRange{begin.getAsyncToken()}, copy, small);
      auto done = gpu::WaitOp::create(at, loc, Type(),
                                      ValueRange{transfer.getAsyncToken()});
      for (Operation *op : {begin.getOperation(), transfer.getOperation(),
                            done.getOperation()})
        op->setAttr("mdir.coupling_read", id);
      for (auto [n, call] : llvm::enumerate(group)) {
        Value read = memref::LoadOp::create(at, loc, copy, index(at, n));
        call->setOperand(pressureOperand(call), read);
      }
    }
  }
  // The small host buffers that the copies of the virials still reach, at
  // the outputs: page-locked, as the results are (the maintainer's addition
  // to #1026).
  SmallVector<gpu::MemcpyOp> small;
  function.walk([&](gpu::MemcpyOp copy) {
    if (copy->hasAttr("mdir.readback") || copy->hasAttr("mdir.coupling_read"))
      small.push_back(copy);
  });
  llvm::DenseSet<Value> pinned;
  for (gpu::MemcpyOp copy : small) {
    Value target = copy.getDst();
    Operation *made = target.getDefiningOp();
    auto type = dyn_cast<MemRefType>(target.getType());
    if (!made || !isa<memref::AllocOp, memref::AllocaOp>(made) || !type ||
        !type.getLayout().isIdentity() || type.getMemorySpace() || type.getRank() != 1 ||
        !type.hasStaticShape() || type.getNumElements() > 16 ||
        !type.getElementType().isF64() || pinned.contains(target))
      continue;
    bool plain = llvm::all_of(target.getUsers(), [](Operation *user) {
      return isa<memref::LoadOp, gpu::MemcpyOp, memref::DeallocOp>(user);
    });
    if (!plain)
      continue;
    Value page = results.pinned(type.getNumElements());
    SmallVector<Operation *> releases;
    for (Operation *user : target.getUsers())
      if (isa<memref::DeallocOp>(user))
        releases.push_back(user);
    for (Operation *op : releases)
      op->erase();
    target.replaceAllUsesWith(page);
    made->erase();
    pinned.insert(page);
  }
  return "";
}
