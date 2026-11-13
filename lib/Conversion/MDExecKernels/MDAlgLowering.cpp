// The lowerings of the md_alg dialect (D406); see
// include/mdir/Conversion/MDAlgLowering.h.
//
// The SIMT lowering puts each op's emitter where the op stands, in the order
// of a walk, without a rewrite driver, which would fold and reorder: the
// module is what the emitters wrote before the ops existed. The serial
// lowering is the definition of each op, written apart from the emitters
// so that the test of one against the other is a test.

#include "mdir/Conversion/MDAlgLowering.h"

#include "mdir/Conversion/CompactionEmitter.h"
#include "mdir/Conversion/GatherEmitter.h"
#include "mdir/Conversion/Passes.h"
#include "mdir/Conversion/RaggedRows.h"
#include "mdir/Dialect/MDAlg/MDAlgOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::md_alg;

namespace mdir {
#define GEN_PASS_DEF_CONVERTMDALGTOGPU
#define GEN_PASS_DEF_CONVERTMDALGTOSCF
#include "mdir/Conversion/Passes.h.inc"
} // namespace mdir

/// The ops of md_alg under `root`, in the order of a walk, each verified.
static FailureOr<SmallVector<Operation *>> collect(Operation *root) {
  SmallVector<Operation *> ops;
  root->walk([&](Operation *op) {
    // The bins of a key region are lowered with their sort.
    if (isa<MDAlgDialect>(op->getDialect()) && !isa<BinOp, NoBinOp>(op))
      ops.push_back(op);
  });
  for (Operation *op : ops)
    if (failed(mlir::verify(op)))
      return failure();
  return ops;
}

//===----------------------------------------------------------------------===//
// SIMT
//===----------------------------------------------------------------------===//

/// Whether `op` has the execution parameters of its SIMT lowering.
static LogicalResult checkExecution(Operation *op) {
  bool given = TypeSwitch<Operation *, bool>(op)
                   .Case([](ScanOp scan) { return !!scan.getStrategy(); })
                   .Case([](ScatterToBinsOp scatter) {
                     return !scatter.getLaunch().empty();
                   })
                   .Case([](SortBinsOp sort) { return !!sort.getStrategy(); })
                   .Case([](SortByKeyOp sort) {
                     return !sort.getLaunch().empty();
                   })
                   .Case([](AppendOp append) { return !!append.getLane(); })
                   .Default([](Operation *) { return true; });
  if (!given)
    return op->emitOpError()
           << "has no execution parameters for its SIMT lowering";
  if (auto append = dyn_cast<AppendOp>(op); append && append.getBlock() < 32)
    return op->emitOpError()
           << "takes blocks of at least 32 entries in its SIMT lowering, a "
              "warp's batch beginning at most two; not "
           << append.getBlock();
  return success();
}


namespace {
/// Code at a builder, with no attributes on loads and stores (those of a
/// template have none).
struct Emit {
  OpBuilder &b;
  Location loc;

  Value index(int64_t v) { return arith::ConstantIndexOp::create(b, loc, v); }
  Value i32(int64_t v) {
    return arith::ConstantOp::create(b, loc, b.getI32Type(),
                                     b.getI32IntegerAttr(v));
  }
  Value i64(int64_t v) {
    return arith::ConstantOp::create(b, loc, b.getI64Type(),
                                     b.getI64IntegerAttr(v));
  }
  Value f64(double v) {
    return arith::ConstantOp::create(b, loc, b.getF64Type(),
                                     b.getF64FloatAttr(v));
  }
  Value load(Value buffer, ValueRange at) {
    return memref::LoadOp::create(b, loc, buffer, at, BoolAttr(),
                                  IntegerAttr(), BoolAttr());
  }
  void store(Value v, Value buffer, ValueRange at) {
    memref::StoreOp::create(b, loc, v, buffer, at, BoolAttr(), IntegerAttr());
  }
  Value cast(Type type, Value v) {
    return arith::IndexCastOp::create(b, loc, type, v);
  }
  Value cmpi(arith::CmpIPredicate p, Value x, Value y) {
    return arith::CmpIOp::create(b, loc, p, x, y);
  }
  void ifThen(Value condition, function_ref<void()> body) {
    scf::IfOp::create(b, loc, condition, [&](OpBuilder &, Location) {
      body();
      scf::YieldOp::create(b, loc);
    });
  }
  /// A launch of `grid` blocks of `block` threads, a thread an item below
  /// `n`: body(item).
  void perItem(Value grid, Value block, Value one, Value n,
               function_ref<void(Value)> body) {
    auto op = gpu::LaunchOp::create(b, loc, grid, one, one, block, one, one);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToEnd(&op.getBody().front());
    Value base = arith::MulIOp::create(b, loc, op.getBlockIds().x, block);
    Value item = arith::AddIOp::create(b, loc, base, op.getThreadIds().x);
    Value inside = cmpi(arith::CmpIPredicate::ult, item, n);
    ifThen(inside, [&] { body(item); });
    gpu::TerminatorOp::create(b, loc);
  }
};
} // namespace

/// Moves the body of the key region of `op` to the builder, with `item` for
/// its argument, and replaces each bin in it with `bin(op, key)`, the key an
/// i32 or an index, or null for no bin.
static void inlineKey(OpBuilder &b, SortByKeyOp op, Value item,
                      function_ref<void(OpBuilder &, Operation *, Value)> bin) {
  Block &key = op.getKey().front();
  key.getArgument(0).replaceAllUsesWith(item);
  SmallVector<Operation *> bins;
  op.getKey().walk([&](Operation *inner) {
    if (isa<BinOp, NoBinOp>(inner))
      bins.push_back(inner);
  });
  Block *into = b.getInsertionBlock();
  into->getOperations().splice(b.getInsertionPoint(), key.getOperations());
  for (Operation *found : bins) {
    OpBuilder at(found);
    bin(at, found,
        isa<BinOp>(found) ? cast<BinOp>(found).getBin() : Value());
    found->erase();
  }
}

/// Whether the key region of `op` gives every item a bin.
static bool everyKey(SortByKeyOp op) {
  if (op.getKey().empty())
    return op.getEveryKey();
  bool none = false;
  op.getKey().walk([&](NoBinOp) { none = true; });
  return !none;
}

/// The SIMT code of a counting sort: the clear (unless `cleared`), the
/// kernel of the keys and their counts, then the scan, the scatter, and the
/// order within the bins by their emitters.
static void emitSortByKey(OpBuilder &b, Location loc, SortByKeyOp op) {
  using namespace kernels;
  Emit e{b, loc};
  OperandRange l = op.getLaunch();
  Value grid = l[0], block = l[1], one = l[2];
  Value counts = op.getCounts(), keys = op.getKeys();
  bool every = everyKey(op);
  if (!op.getCleared()) {
    Value last = arith::SubIOp::create(b, loc, block, one);
    Value padded = arith::AddIOp::create(b, loc, op.getBins(), last);
    Value blocks = arith::DivUIOp::create(b, loc, padded, block);
    Value binGrid = arith::MaxUIOp::create(b, loc, blocks, one);
    e.perItem(binGrid, block, one, op.getBins(), [&](Value bin) {
      e.store(e.i32(0), counts, bin);
    });
  }
  e.perItem(grid, block, one, op.getN(), [&](Value item) {
    if (op.getKey().empty()) {
      Value step = e.i32(1);
      Value key = e.cast(b.getIndexType(), e.load(keys, item));
      if (every) {
        emitDeviceAtomicAdd(b, loc, counts, key, step);
        return;
      }
      Value has = e.cmpi(arith::CmpIPredicate::sge, key, e.index(0));
      e.ifThen(has, [&] { emitDeviceAtomicAdd(b, loc, counts, key, step); });
      return;
    }
    inlineKey(b, op, item, [&](OpBuilder &at, Operation *, Value bin) {
      Emit here{at, loc};
      if (!bin) {
        here.store(here.i32(-1), keys, item);
        return;
      }
      Value one32 = here.i32(1);
      if (bin.getType().isIndex()) {
        here.store(here.cast(at.getI32Type(), bin), keys, item);
        emitDeviceAtomicAdd(at, loc, counts, bin, one32);
      } else {
        here.store(bin, keys, item);
        emitDeviceAtomicAdd(at, loc, counts,
                            here.cast(at.getIndexType(), bin), one32);
      }
    });
  });
  OperandRange s = op.getScanLaunch();
  switch (*op.getScanStrategy()) {
  case ScanStrategy::ReduceThenScan:
    emitDeviceScan(b, loc, counts, op.getStart(), op.getCursor(), op.getBins(),
                   op.getScanScratch(), s[0], s[1], s[2]);
    break;
  case ScanStrategy::SinglePass:
    emitSinglePassScan(b, loc, counts, op.getStart(), op.getCursor(),
                       op.getBins(), op.getScanScratch(), s[0], s[1], s[2]);
    break;
  case ScanStrategy::Chunked:
    emitChunkedScan(b, loc, counts, op.getStart(), op.getCursor(),
                    op.getScanScratch(), op.getBins(), s[0], s[1], s[2], s[3],
                    s[4]);
    break;
  }
  // The order within the bins is the op's contract, not its strategy's: an
  // unordered sort ends with the scatter.
  bool sorted = op.getOrder() == Order::Stable;
  bool rank =
      sorted && *op.getSortStrategy() == SortStrategy::StableRank;
  Value scattered = rank ? op.getSortBuffer() : op.getOut();
  emitCursorScatter(b, loc, keys, op.getCursor(), scattered, op.getN(), grid,
                    block, one, every);
  if (!sorted)
    return;
  OperandRange t = op.getSortLaunch();
  if (rank)
    emitRankInBins(b, loc, scattered, keys, op.getStart(), op.getOut(),
                   op.getBins(), t[0], t[1], t[2]);
  else
    emitSortBins(b, loc, op.getStart(), op.getOut(), op.getBins(), t[0], t[1],
                 t[2], op.getBy());
}

/// The address of `pool[at]` as a pointer of the device.
static Value poolPointer(Emit &e, Value pool, Value at) {
  OpBuilder &b = e.b;
  Value base = memref::ExtractAlignedPointerAsIndexOp::create(b, e.loc, pool);
  Value base64 = e.cast(b.getI64Type(), base);
  Value offset;
  if (std::optional<int64_t> c = getConstantIntValue(at))
    offset = e.i64(*c * 4);
  else
    offset = arith::MulIOp::create(b, e.loc, e.cast(b.getI64Type(), at),
                                   e.i64(4));
  Value address = arith::AddIOp::create(b, e.loc, base64, offset);
  auto space =
      cast<MemRefType>(pool.getType()).getMemorySpaceAsInt();
  return LLVM::IntToPtrOp::create(
      b, e.loc, LLVM::LLVMPointerType::get(b.getContext(), space), address);
}

/// The SIMT code of an append by the lanes of a warp (the groups build's
/// lists, D89): the ballot of the lanes that keep an entry, their ranks,
/// the blocks that the round begins taken by the first lane with one
/// atomic and shared, each entry written at its slot, and the new blocks
/// named. Gives the count and the last block after the round.
static SmallVector<Value> emitAppend(OpBuilder &b, Location loc,
                                     AppendOp op) {
  using namespace kernels;
  using P = arith::CmpIPredicate;
  Emit e{b, loc};
  int64_t block = op.getBlock();
  int64_t shift = llvm::Log2_64(block);
  Value zero = e.i32(0), oneI = e.i32(1);
  Value bits = e.i32(shift), mask = e.i32(block - 1);
  Value keep = op.getKeep(), cs = op.getCount(), csb = op.getLast();
  Value lane32 = op.getLane(), below = op.getBelow();
  Value capacity = op.getCapacity();
  auto add = [&](Value x, Value y) -> Value {
    return arith::AddIOp::create(b, loc, x, y);
  };
  auto andi = [&](Value x, Value y) -> Value {
    return arith::AndIOp::create(b, loc, x, y);
  };
  auto shr = [&](Value x) -> Value {
    return arith::ShRUIOp::create(b, loc, x, bits);
  };
  auto select = [&](Value c, Value x, Value y) -> Value {
    return arith::SelectOp::create(b, loc, c, x, y);
  };
  Value kept = emitWarpBallot(b, loc, keep);
  Value rank = emitWarpRank(b, loc, kept, below);
  Value slot = add(cs, rank);
  Value nKept = emitWarpCount(b, loc, kept);
  Value next = add(cs, nKept);
  Value any = e.cmpi(P::ne, nKept, zero);
  Value firstK = shr(cs);
  Value lastSlot = arith::SubIOp::create(b, loc, next, oneI);
  Value lastK = shr(lastSlot);
  Value csMod = andi(cs, mask);
  Value atStart = e.cmpi(P::eq, csMod, zero);
  Value needA = andi(any, atStart);
  Value crosses = e.cmpi(P::ugt, lastK, firstK);
  Value needB = andi(any, crosses);
  Value na = arith::ExtUIOp::create(b, loc, b.getI32Type(), needA);
  Value nb = arith::ExtUIOp::create(b, loc, b.getI32Type(), needB);
  Value taken = add(na, nb);
  Value takes = e.cmpi(P::ne, taken, zero);
  Value laneFirst = e.cmpi(P::eq, lane32, zero);
  Value allocates = andi(takes, laneFirst);
  auto take = scf::IfOp::create(b, loc, TypeRange{b.getI32Type()}, allocates,
                                /*addThenBlock=*/true, /*addElseBlock=*/true);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(take.thenBlock());
    Value ptr = poolPointer(e, op.getPool(), op.getAt());
    Value first = LLVM::AtomicRMWOp::create(
        b, loc, LLVM::AtomicBinOp::add, ptr, taken,
        LLVM::AtomicOrdering::monotonic, "device");
    scf::YieldOp::create(b, loc, first);
    b.setInsertionPointToStart(take.elseBlock());
    scf::YieldOp::create(b, loc, zero);
  }
  Value base0 = take.getResult(0);
  Value width = e.i32(32);
  Value takenAt = gpu::ShuffleOp::create(b, loc, base0, zero, width,
                                         gpu::ShuffleMode::IDX)
                      .getShuffleResult();
  Value blockA = select(needA, takenAt, csb);
  Value takenAt1 = add(takenAt, oneI);
  Value blockB = select(needA, takenAt1, takenAt);
  Value slotK = shr(slot);
  Value inA = e.cmpi(P::eq, slotK, firstK);
  Value slotBlock = select(inA, blockA, blockB);
  Value slotMod = andi(slot, mask);
  Value slotBase = arith::ShLIOp::create(b, loc, slotBlock, bits);
  Value pos32 = add(slotBase, slotMod);
  Value slotBlockI = e.cast(b.getIndexType(), slotBlock);
  Value fits = e.cmpi(P::ult, slotBlockI, capacity);
  Value storeIt = andi(keep, fits);
  e.ifThen(storeIt, [&] {
    Value pos = e.cast(b.getIndexType(), pos32);
    for (auto [v, into] : llvm::zip(op.getValues(), op.getInto()))
      e.store(v, into, pos);
  });
  auto name = [&](Value need, Value blockN, Value ordinal) {
    Value nameN = andi(need, laneFirst);
    Value at = e.cast(b.getIndexType(), blockN);
    Value atFits = e.cmpi(P::ult, at, capacity);
    Value write = andi(nameN, atFits);
    e.ifThen(write, [&] {
      e.store(op.getOwner(), op.getUnits(), at);
      e.store(ordinal, op.getOrdinals(), at);
    });
  };
  name(needA, blockA, firstK);
  name(needB, blockB, lastK);
  Value nextBlock = select(needB, blockB, blockA);
  return {next, nextBlock};
}

/// The counts of a grid plan, on the host: the same code for both
/// lowerings. The stages run across the edges (all the divisions, then all
/// the floors, ...), as the sites computed them.
static SmallVector<Value> emitGridPlan(OpBuilder &b, Location loc,
                                       GridPlanOp op) {
  Emit e{b, loc};
  auto real = cast<FloatType>(op.getLengths().front().getType());
  auto constant = [&](Attribute attr) -> Value {
    return arith::ConstantOp::create(b, loc, real, cast<FloatAttr>(attr));
  };
  Value width = op.getWidth();
  if (!width) {
    Value n64 = e.cast(b.getI64Type(), op.getPopulation());
    Value n = arith::SIToFPOp::create(b, loc, real, n64);
    Value density = arith::DivFOp::create(b, loc, n, op.getVolume());
    Value occupancy = constant(op.getOccupancyAttr());
    Value cube = arith::DivFOp::create(b, loc, occupancy, density);
    Value third = constant(b.getFloatAttr(real, 0.3333333333333333));
    width = math::PowFOp::create(b, loc, cube, third);
  }
  SmallVector<Value> v;
  for (Value length : op.getLengths())
    v.push_back(arith::DivFOp::create(b, loc, length, width));
  for (Value &x : v)
    x = math::FloorOp::create(b, loc, x);
  Value least = constant(op.getLeastAttr());
  Value most = constant(op.getMostAttr());
  for (Value &x : v)
    x = arith::MaxNumFOp::create(b, loc, x, least);
  for (Value &x : v)
    x = arith::MinNumFOp::create(b, loc, x, most);
  for (Value &x : v)
    x = arith::FPToSIOp::create(b, loc, b.getI64Type(), x);
  return v;
}

/// The emitter's code of `op` before it; its results.
static SmallVector<Value> emitSIMT(OpBuilder &b, Operation *op) {
  using namespace kernels;
  Location loc = op->getLoc();
  return TypeSwitch<Operation *, SmallVector<Value>>(op)
      .Case([&](ScanOp scan) {
        OperandRange l = scan.getLaunch();
        switch (*scan.getStrategy()) {
        case ScanStrategy::ReduceThenScan:
          emitDeviceScan(b, loc, scan.getCounts(), scan.getStart(),
                         scan.getCursor(), scan.getN(), scan.getScratch(),
                         l[0], l[1], l[2]);
          break;
        case ScanStrategy::SinglePass:
          emitSinglePassScan(b, loc, scan.getCounts(), scan.getStart(),
                             scan.getCursor(), scan.getN(),
                             scan.getScratch(), l[0], l[1], l[2]);
          break;
        case ScanStrategy::Chunked:
          emitChunkedScan(b, loc, scan.getCounts(), scan.getStart(),
                          scan.getCursor(), scan.getScratch(), scan.getN(),
                          l[0], l[1], l[2], l[3], l[4]);
          break;
        }
        return SmallVector<Value>{};
      })
      .Case([&](ScatterToBinsOp scatter) {
        OperandRange l = scatter.getLaunch();
        emitCursorScatter(b, loc, scatter.getKeys(), scatter.getCursor(),
                          scatter.getOut(), scatter.getN(), l[0], l[1], l[2],
                          scatter.getEveryKey());
        return SmallVector<Value>{};
      })
      .Case([&](SortBinsOp sort) {
        OperandRange l = sort.getLaunch();
        if (*sort.getStrategy() == SortStrategy::Insertion)
          emitSortBins(b, loc, sort.getStart(), sort.getItems(),
                       sort.getBins(), l[0], l[1], l[2], sort.getBy());
        else
          emitRankInBins(b, loc, sort.getItems(), sort.getKeys(),
                         sort.getStart(), sort.getOut(), sort.getBins(), l[0],
                         l[1], l[2]);
        return SmallVector<Value>{};
      })
      .Case([&](RaggedRowOp row) {
        RaggedRow found = emitRaggedRow(b, loc, row.getRows(), row.getRow(),
                                        row.getEntrySize(), row.getChecked());
        return SmallVector<Value>{found.first, found.count};
      })
      .Case([&](RaggedEntryOp entry) {
        return SmallVector<Value>{emitRaggedEntry(
            b, loc, entry.getRows(), entry.getFirst(), entry.getK(),
            entry.getColumn(), entry.getEntrySize())};
      })
      .Case([&](WrapOp wrap) {
        return SmallVector<Value>{
            emitWrappedGridAxis(b, loc, wrap.getStart(), wrap.getOffset(),
                                wrap.getSize(), wrap.getSignedCompare())};
      })
      .Case([&](SortByKeyOp sort) {
        emitSortByKey(b, loc, sort);
        return SmallVector<Value>{};
      })
      .Case([&](AppendOp append) { return emitAppend(b, loc, append); })
      .Case([&](GridPlanOp plan) { return emitGridPlan(b, loc, plan); })
      .Case([&](BinSpanOp span) {
        Emit e{b, loc};
        Value begin = e.load(span.getStart(), span.getFrom());
        Value end = e.load(span.getStart(), span.getTo());
        return SmallVector<Value>{begin, end};
      })
      .Case([&](BinItemOp item) {
        Emit e{b, loc};
        return SmallVector<Value>{e.load(item.getOut(), item.getPlace())};
      })
      .Case([&](GridReadOp read) {
        return SmallVector<Value>{emitGridGather(b, loc, read.getGrid(),
                                                 read.getBase(), read.getZ(),
                                                 read.getActive())};
      });
}

LogicalResult kernels::lowerAlgToSIMT(Operation *root) {
  FailureOr<SmallVector<Operation *>> ops = collect(root);
  if (failed(ops))
    return failure();
  for (Operation *op : *ops)
    if (failed(checkExecution(op)))
      return failure();
  for (Operation *op : *ops) {
    OpBuilder builder(op);
    SmallVector<Value> results = emitSIMT(builder, op);
    op->replaceAllUsesWith(results);
    op->erase();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Serial
//===----------------------------------------------------------------------===//

namespace {
/// The serial definitions, built at one place.
struct Serial {
  OpBuilder &b;
  Location loc;

  Value index(int64_t v) { return arith::ConstantIndexOp::create(b, loc, v); }
  Value i32(int64_t v) {
    return arith::ConstantOp::create(b, loc, b.getI32Type(),
                                     b.getI32IntegerAttr(v));
  }
  Value load(Value buffer, Value at) {
    return memref::LoadOp::create(b, loc, buffer, ValueRange{at});
  }
  void store(Value v, Value buffer, Value at) {
    memref::StoreOp::create(b, loc, v, buffer, ValueRange{at});
  }
  Value asIndex(Value v) {
    return arith::IndexCastOp::create(b, loc, b.getIndexType(), v);
  }
  /// for i in [from, to): body(i).
  void loop(Value from, Value to, function_ref<void(Value)> body) {
    auto op = scf::ForOp::create(b, loc, from, to, index(1));
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(op.getBody());
    body(op.getInductionVar());
  }
  void ifThen(Value condition, function_ref<void()> body) {
    auto op = scf::IfOp::create(b, loc, condition, /*withElseRegion=*/false);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(op.thenBlock());
    body();
  }

  /// start[i] = sum of counts[k < i], start[n] = the sum of all, and
  /// cursor[i] = start[i].
  void scan(ScanOp op) {
    scan(op.getCounts(), op.getN(), op.getStart(), op.getCursor());
  }
  void scan(Value counts, Value n, Value start, Value cursor) {
    Value i0 = index(0), i1 = index(1);
    auto sums = scf::ForOp::create(b, loc, i0, n, i1, ValueRange{i32(0)});
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(sums.getBody());
      Value i = sums.getInductionVar();
      Value before = sums.getRegionIterArg(0);
      store(before, start, i);
      if (cursor)
        store(before, cursor, i);
      Value next = arith::AddIOp::create(b, loc, before, load(counts, i));
      scf::YieldOp::create(b, loc, next);
    }
    store(sums.getResult(0), start, n);
  }

  /// Each item with a bin to the next slot of its bin, in the order of the
  /// items.
  void scatter(ScatterToBinsOp op) {
    scatter(op.getKeys(), op.getN(), op.getCursor(), op.getOut(),
            op.getEveryKey());
  }
  void scatter(Value keys, Value n, Value cursor, Value out, bool every) {
    loop(index(0), n, [&](Value i) {
      Value key = load(keys, i);
      auto place = [&]() {
        Value k = asIndex(key);
        Value slot = load(cursor, k);
        Value next = arith::AddIOp::create(b, loc, slot, i32(1));
        store(next, cursor, k);
        Value item = arith::IndexCastOp::create(b, loc, b.getI32Type(), i);
        store(item, out, asIndex(slot));
      };
      if (every) {
        place();
        return;
      }
      Value has = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge,
                                        key, i32(0));
      ifThen(has, place);
    });
  }

  /// Each bin sorted ascending, stably: the items copied to `out` first
  /// where it is given, then each item exchanged with the one before it
  /// while that one is greater.
  void sort(SortBinsOp op) {
    sort(op.getStart(), op.getBins(), op.getItems(), op.getBy(), op.getOut());
  }
  void sort(Value start, Value bins, Value items, Value by, Value out) {
    Value target = out ? out : items;
    auto order = [&](Value item) -> Value {
      return by ? load(by, asIndex(item)) : item;
    };
    loop(index(0), bins, [&](Value bin) {
      Value next = arith::AddIOp::create(b, loc, bin, index(1));
      Value begin = asIndex(load(start, bin));
      Value end = asIndex(load(start, next));
      if (out)
        loop(begin, end, [&](Value p) { store(load(items, p), target, p); });
      loop(begin, end, [&](Value p) {
        // q from p down to begin + 1: exchange q - 1 and q while the item
        // at q - 1 orders after the one at q.
        auto walk = scf::WhileOp::create(
            b, loc, TypeRange{b.getIndexType()}, ValueRange{p},
            [&](OpBuilder &w, Location, ValueRange q) {
              OpBuilder::InsertionGuard guard(b);
              b.setInsertionPointToEnd(w.getBlock());
              Value more = arith::CmpIOp::create(
                  b, loc, arith::CmpIPredicate::ugt, q[0], begin);
              auto test = scf::IfOp::create(b, loc, TypeRange{b.getI1Type()},
                                            more, /*withElseRegion=*/true);
              {
                OpBuilder::InsertionGuard inner(b);
                b.setInsertionPointToStart(test.thenBlock());
                Value before = arith::SubIOp::create(b, loc, q[0], index(1));
                Value left = order(load(target, before));
                Value right = order(load(target, q[0]));
                Value after = arith::CmpIOp::create(
                    b, loc, arith::CmpIPredicate::sgt, left, right);
                scf::YieldOp::create(b, loc, after);
                b.setInsertionPointToStart(test.elseBlock());
                Value no = arith::ConstantOp::create(b, loc, b.getBoolAttr(false));
                scf::YieldOp::create(b, loc, no);
              }
              scf::ConditionOp::create(b, loc, test.getResult(0), q);
            },
            [&](OpBuilder &w, Location, ValueRange q) {
              OpBuilder::InsertionGuard guard(b);
              b.setInsertionPointToEnd(w.getBlock());
              Value before = arith::SubIOp::create(b, loc, q[0], index(1));
              Value left = load(target, before);
              Value right = load(target, q[0]);
              store(right, target, before);
              store(left, target, q[0]);
              scf::YieldOp::create(b, loc, before);
            });
        (void)walk;
      });
    });
  }

  /// The counts cleared (unless `cleared`), each item's key and its
  /// count in the order of the items, then the scan, the scatter, and the
  /// sort within the bins as defined above, in place in `out`.
  void sortByKey(SortByKeyOp op) {
    Value counts = op.getCounts(), keys = op.getKeys();
    // Before the region's code moves into the loop.
    bool every = everyKey(op);
    if (!op.getCleared())
      loop(index(0), op.getBins(),
           [&](Value bin) { store(i32(0), counts, bin); });
    auto count = [&](OpBuilder &at, Value k) {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPoint(at.getInsertionBlock(), at.getInsertionPoint());
      Value c = load(counts, k);
      store(arith::AddIOp::create(b, loc, c, i32(1)), counts, k);
    };
    loop(index(0), op.getN(), [&](Value item) {
      if (op.getKey().empty()) {
        Value key = load(keys, item);
        auto add = [&] { count(b, asIndex(key)); };
        if (op.getEveryKey())
          add();
        else
          ifThen(arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge, key,
                                       i32(0)),
                 add);
        return;
      }
      inlineKey(b, op, item, [&](OpBuilder &at, Operation *, Value bin) {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPoint(at.getInsertionBlock(), at.getInsertionPoint());
        if (!bin) {
          store(i32(-1), keys, item);
          return;
        }
        Value k32 = bin.getType().isIndex()
                        ? arith::IndexCastOp::create(b, loc, b.getI32Type(),
                                                     bin)
                              .getResult()
                        : bin;
        store(k32, keys, item);
        count(b, asIndex(k32));
      });
    });
    scan(counts, op.getBins(), op.getStart(), op.getCursor());
    scatter(keys, op.getN(), op.getCursor(), op.getOut(), every);
    if (op.getOrder() == Order::Stable)
      sort(op.getStart(), op.getBins(), op.getOut(), op.getBy(), Value());
  }

  /// One item appended: a block taken where its slot begins one, named
  /// below the capacity, and the values written there below it.
  SmallVector<Value> append(AppendOp op) {
    using P = arith::CmpIPredicate;
    Type t = b.getI32Type();
    int64_t block = op.getBlock();
    Value count = op.getCount(), last = op.getLast();
    auto kept = scf::IfOp::create(b, loc, TypeRange{t, t}, op.getKeep(),
                                  /*addThenBlock=*/true, /*addElseBlock=*/true);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(kept.thenBlock());
      Value j = arith::RemUIOp::create(b, loc, count, i32(block));
      Value k = arith::DivUIOp::create(b, loc, count, i32(block));
      Value begins = arith::CmpIOp::create(b, loc, P::eq, j, i32(0));
      Value capacity = op.getCapacity();
      auto fits = [&](Value blockN) {
        return arith::CmpIOp::create(b, loc, P::ult, asIndex(blockN),
                                     capacity);
      };
      auto now = scf::IfOp::create(b, loc, TypeRange{t}, begins, true, true);
      {
        OpBuilder::InsertionGuard inner(b);
        b.setInsertionPointToStart(now.thenBlock());
        Value taken = load(op.getPool(), op.getAt());
        store(arith::AddIOp::create(b, loc, taken, i32(1)), op.getPool(),
              op.getAt());
        ifThen(fits(taken), [&] {
          store(op.getOwner(), op.getUnits(), asIndex(taken));
          store(k, op.getOrdinals(), asIndex(taken));
        });
        scf::YieldOp::create(b, loc, taken);
        b.setInsertionPointToStart(now.elseBlock());
        scf::YieldOp::create(b, loc, last);
      }
      Value holder = now.getResult(0);
      ifThen(fits(holder), [&] {
        Value base = arith::MulIOp::create(b, loc, holder, i32(block));
        Value at = asIndex(arith::AddIOp::create(b, loc, base, j));
        for (auto [v, into] : llvm::zip(op.getValues(), op.getInto()))
          store(v, into, at);
      });
      Value next = arith::AddIOp::create(b, loc, count, i32(1));
      scf::YieldOp::create(b, loc, ValueRange{next, holder});
      b.setInsertionPointToStart(kept.elseBlock());
      scf::YieldOp::create(b, loc, ValueRange{count, last});
    }
    return {kept.getResult(0), kept.getResult(1)};
  }
};
} // namespace

LogicalResult kernels::lowerAlgToSerial(Operation *root) {
  FailureOr<SmallVector<Operation *>> ops = collect(root);
  if (failed(ops))
    return failure();
  for (Operation *op : *ops) {
    OpBuilder builder(op);
    Serial serial{builder, op->getLoc()};
    SmallVector<Value> results;
    TypeSwitch<Operation *>(op)
        .Case([&](ScanOp scan) { serial.scan(scan); })
        .Case([&](ScatterToBinsOp scatter) { serial.scatter(scatter); })
        .Case([&](SortBinsOp sort) { serial.sort(sort); })
        .Case([&](SortByKeyOp sort) { serial.sortByKey(sort); })
        .Case([&](AppendOp append) { results = serial.append(append); })
        // The accessors compute one value where they stand; their serial
        // code is that of their emitters.
        .Default([&](Operation *op) { results = emitSIMT(builder, op); });
    op->replaceAllUsesWith(results);
    op->erase();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

namespace {
class ConvertMDAlgToGPU
    : public mdir::impl::ConvertMDAlgToGPUBase<ConvertMDAlgToGPU> {
public:
  void runOnOperation() final {
    if (failed(kernels::lowerAlgToSIMT(getOperation())))
      signalPassFailure();
  }
};

class ConvertMDAlgToSCF
    : public mdir::impl::ConvertMDAlgToSCFBase<ConvertMDAlgToSCF> {
public:
  void runOnOperation() final {
    if (failed(kernels::lowerAlgToSerial(getOperation())))
      signalPassFailure();
  }
};
} // namespace
