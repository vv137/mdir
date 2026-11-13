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
    if (isa<MDAlgDialect>(op->getDialect()))
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
                   .Default([](Operation *) { return true; });
  if (!given)
    return op->emitOpError()
           << "has no execution parameters for its SIMT lowering";
  return success();
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
    Value i0 = index(0), i1 = index(1);
    auto sums = scf::ForOp::create(b, loc, i0, op.getN(), i1,
                                   ValueRange{i32(0)});
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(sums.getBody());
      Value i = sums.getInductionVar();
      Value before = sums.getRegionIterArg(0);
      store(before, op.getStart(), i);
      if (op.getCursor())
        store(before, op.getCursor(), i);
      Value next = arith::AddIOp::create(
          b, loc, before, load(op.getCounts(), i));
      scf::YieldOp::create(b, loc, next);
    }
    store(sums.getResult(0), op.getStart(), op.getN());
  }

  /// Each item with a bin to the next slot of its bin, in the order of the
  /// items.
  void scatter(ScatterToBinsOp op) {
    loop(index(0), op.getN(), [&](Value i) {
      Value key = load(op.getKeys(), i);
      auto place = [&]() {
        Value k = asIndex(key);
        Value slot = load(op.getCursor(), k);
        Value next = arith::AddIOp::create(b, loc, slot, i32(1));
        store(next, op.getCursor(), k);
        Value item = arith::IndexCastOp::create(b, loc, b.getI32Type(), i);
        store(item, op.getOut(), asIndex(slot));
      };
      if (op.getEveryKey()) {
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
    Value target = op.getOut() ? op.getOut() : op.getItems();
    Value by = op.getBy();
    auto order = [&](Value item) -> Value {
      return by ? load(by, asIndex(item)) : item;
    };
    loop(index(0), op.getBins(), [&](Value bin) {
      Value next = arith::AddIOp::create(b, loc, bin, index(1));
      Value begin = asIndex(load(op.getStart(), bin));
      Value end = asIndex(load(op.getStart(), next));
      if (op.getOut())
        loop(begin, end,
             [&](Value p) { store(load(op.getItems(), p), target, p); });
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
