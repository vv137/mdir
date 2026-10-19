// The emitters of compaction, scans, and counting sorts on a device
// (D[compaction-emitters]); see include/mdir/Conversion/CompactionEmitter.h.

#include "mdir/Conversion/CompactionEmitter.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;

//===----------------------------------------------------------------------===//
// Warp compaction
//===----------------------------------------------------------------------===//

Value kernels::emitLanesBelow(OpBuilder &b, Location loc, Value lane,
                              Value one) {
  if (one)
    return arith::SubIOp::create(
        b, loc, arith::ShLIOp::create(b, loc, one, lane), one);
  auto constant32 = [&](OpBuilder &b, int64_t v) -> Value {
    return arith::ConstantOp::create(b, loc, b.getI32Type(),
                                     b.getI32IntegerAttr(v));
  };
  return arith::SubIOp::create(
      b, loc, arith::ShLIOp::create(b, loc, constant32(b, 1), lane),
      constant32(b, 1));
}

Value kernels::emitWarpBallot(OpBuilder &b, Location loc, Value predicate) {
  return gpu::BallotOp::create(b, loc, b.getI32Type(), predicate);
}

Value kernels::emitWarpRank(OpBuilder &b, Location loc, Value ballot,
                            Value below) {
  return math::CtPopOp::create(b, loc,
                               arith::AndIOp::create(b, loc, ballot, below));
}

Value kernels::emitWarpCount(OpBuilder &b, Location loc, Value ballot) {
  return math::CtPopOp::create(b, loc, ballot);
}

//===----------------------------------------------------------------------===//
// Scans
//===----------------------------------------------------------------------===//

Value kernels::emitWarpScan(OpBuilder &b, Location loc, Value x, Value lane,
                            Value zero, Value width, int64_t lanes,
                            bool unsignedLane) {
  Type i32 = b.getI32Type();
  for (int64_t o = 1; o < lanes; o *= 2) {
    Value offset =
        arith::ConstantOp::create(b, loc, i32, b.getI32IntegerAttr(o));
    Value up = gpu::ShuffleOp::create(b, loc, x, offset, width,
                                      gpu::ShuffleMode::UP)
                   .getShuffleResult();
    Value has = arith::CmpIOp::create(
        b, loc,
        unsignedLane ? arith::CmpIPredicate::uge : arith::CmpIPredicate::sge,
        lane, offset);
    Value add = arith::SelectOp::create(b, loc, has, up, zero);
    x = arith::AddIOp::create(b, loc, x, add);
  }
  return x;
}

BlockScan kernels::emitBlockScan(OpBuilder &b, Location loc, Value value,
                                 Value shared, Value thread) {
  Type i32 = b.getI32Type();
  int64_t warps = cast<MemRefType>(shared.getType()).getDimSize(0) - 1;
  auto index = [&](OpBuilder &b, int64_t v) -> Value {
    return arith::ConstantIndexOp::create(b, loc, v);
  };
  Value zero = arith::ConstantOp::create(b, loc, i32, b.getI32IntegerAttr(0));
  Value i0 = index(b, 0);
  Value lastWarp = index(b, warps - 1);
  Value totalAt = index(b, warps);
  Value lastLane = index(b, 31);
  Value warpSize = index(b, 32);
  Value width = arith::ConstantOp::create(b, loc, i32, b.getI32IntegerAttr(32));
  Value lane = arith::RemUIOp::create(b, loc, thread, warpSize);
  Value warp = arith::DivUIOp::create(b, loc, thread, warpSize);
  Value lane32 = arith::IndexCastOp::create(b, loc, i32, lane);
  // The scan of each warp; its last lane holds the warp's sum.
  Value x = arith::AddIOp::create(b, loc, value, zero);
  x = emitWarpScan(b, loc, x, lane32, zero, width, 32);
  Value last = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, lane,
                                     lastLane);
  scf::IfOp::create(b, loc, last, [&](OpBuilder &c, Location) {
    memref::StoreOp::create(c, loc, x, shared, ValueRange{warp});
    scf::YieldOp::create(c, loc);
  });
  gpu::BarrierOp::create(b, loc);
  // The first warp scans the sums of the warps: each takes the sum of the
  // warps before it, and the last element the sum of all.
  Value first =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, warp, i0);
  scf::IfOp::create(b, loc, first, [&](OpBuilder &c, Location) {
    Value inWarps = arith::CmpIOp::create(c, loc, arith::CmpIPredicate::ult,
                                          lane, totalAt);
    // Only the lanes of the sums read them: a lane past them that read
    // the first would race with the lane that writes it below.
    auto read = scf::IfOp::create(c, loc, TypeRange{i32}, inWarps,
                                  /*withElseRegion=*/true);
    {
      OpBuilder then = OpBuilder::atBlockEnd(read.thenBlock());
      Value w = memref::LoadOp::create(then, loc, shared, ValueRange{lane});
      scf::YieldOp::create(then, loc, w);
      OpBuilder otherwise = OpBuilder::atBlockEnd(read.elseBlock());
      scf::YieldOp::create(otherwise, loc, zero);
    }
    Value u0 = read.getResult(0);
    Value u = emitWarpScan(c, loc, u0, lane32, zero, width, warps);
    Value exclusive = arith::SubIOp::create(c, loc, u, u0);
    Value isLast = arith::CmpIOp::create(c, loc, arith::CmpIPredicate::eq,
                                         lane, lastWarp);
    scf::IfOp::create(c, loc, inWarps, [&](OpBuilder &d, Location) {
      memref::StoreOp::create(d, loc, exclusive, shared, ValueRange{lane});
      scf::YieldOp::create(d, loc);
    });
    scf::IfOp::create(c, loc, isLast, [&](OpBuilder &d, Location) {
      memref::StoreOp::create(d, loc, u, shared, ValueRange{totalAt});
      scf::YieldOp::create(d, loc);
    });
    scf::YieldOp::create(c, loc);
  });
  gpu::BarrierOp::create(b, loc);
  Value offset = memref::LoadOp::create(b, loc, shared, ValueRange{warp});
  Value total = memref::LoadOp::create(b, loc, shared, ValueRange{totalAt});
  Value inWarp = arith::SubIOp::create(b, loc, x, value);
  return {arith::AddIOp::create(b, loc, inWarp, offset), total};
}

void kernels::emitDeviceScan(OpBuilder &b, Location loc, Value counts,
                             Value start, Value cursor, Value n, Value sums,
                             Value blocks, Value threads, Value one) {
  MLIRContext *context = b.getContext();
  Type i32 = b.getI32Type();
  auto shared = MemRefType::get(
      {9}, i32, MemRefLayoutAttrInterface(),
      gpu::AddressSpaceAttr::get(context,
                                 gpu::GPUDialect::getWorkgroupAddressSpace()));
  auto index = [&](OpBuilder &b, int64_t v) -> Value {
    return arith::ConstantIndexOp::create(b, loc, v);
  };
  auto launch = [&](Value grid,
                    function_ref<void(OpBuilder &, Value, Value, Value)> body) {
    auto op = gpu::LaunchOp::create(
        b, loc, grid, one, one, threads, one, one,
        /*dynamicSharedMemorySize=*/nullptr, /*asyncTokenType=*/nullptr,
        /*asyncDependencies=*/{}, /*workgroupAttributions=*/TypeRange{shared});
    OpBuilder kernel = OpBuilder::atBlockEnd(&op.getBody().front());
    body(kernel, op.getBlockIds().x, op.getThreadIds().x,
         op.getWorkgroupAttributionBBArgs().front());
    gpu::TerminatorOp::create(kernel, loc);
  };
  // The first count of each thread, 4 a thread and 1024 a block.
  struct Counts {
    Value i0, i1, i4, none, first;
  };
  auto prologue = [&](OpBuilder &k, Value block, Value thread) {
    Counts c;
    c.i0 = index(k, 0);
    c.i1 = index(k, 1);
    c.i4 = index(k, 4);
    c.none = arith::ConstantOp::create(k, loc, i32, k.getI32IntegerAttr(0));
    Value perBlock = index(k, 1024);
    Value blockBase = arith::MulIOp::create(k, loc, block, perBlock);
    Value threadBase = arith::MulIOp::create(k, loc, thread, c.i4);
    c.first = arith::AddIOp::create(k, loc, blockBase, threadBase);
    return c;
  };
  // The sum of the 4 counts of a thread.
  auto sumOfFour = [&](OpBuilder &k, const Counts &c) -> Value {
    return scf::ForOp::create(
               k, loc, c.i0, c.i4, c.i1, ValueRange{c.none},
               [&](OpBuilder &f, Location, Value at, ValueRange s) {
                 Value item = arith::AddIOp::create(f, loc, c.first, at);
                 Value in = arith::CmpIOp::create(
                     f, loc, arith::CmpIPredicate::ult, item, n);
                 Value safe = arith::SelectOp::create(f, loc, in, item, c.i0);
                 Value v0 =
                     memref::LoadOp::create(f, loc, counts, ValueRange{safe});
                 Value v = arith::SelectOp::create(f, loc, in, v0, c.none);
                 scf::YieldOp::create(
                     f, loc,
                     ValueRange{arith::AddIOp::create(f, loc, s[0], v)});
               })
        .getResult(0);
  };

  // The sum of the counts of each block.
  launch(blocks, [&](OpBuilder &k, Value block, Value thread, Value wg) {
    Counts c = prologue(k, block, thread);
    Value mine = sumOfFour(k, c);
    BlockScan scan = emitBlockScan(k, loc, mine, wg, thread);
    Value lead = arith::CmpIOp::create(k, loc, arith::CmpIPredicate::eq,
                                       thread, c.i0);
    scf::IfOp::create(k, loc, lead, [&](OpBuilder &t, Location) {
      memref::StoreOp::create(t, loc, scan.total, sums, ValueRange{block});
      scf::YieldOp::create(t, loc);
    });
  });

  // One block: the sums of the blocks become their offsets, a span of
  // them a thread, and the sum of all goes to start[n].
  launch(one, [&](OpBuilder &k, Value, Value thread, Value wg) {
    Value i0 = index(k, 0);
    Value i1 = index(k, 1);
    Value none = arith::ConstantOp::create(k, loc, i32, k.getI32IntegerAttr(0));
    Value width = index(k, 256);
    Value span0 = arith::AddIOp::create(k, loc, blocks, width);
    Value span1 = arith::SubIOp::create(k, loc, span0, i1);
    Value span = arith::DivUIOp::create(k, loc, span1, width);
    Value begin0 = arith::MulIOp::create(k, loc, thread, span);
    Value begin = arith::MinUIOp::create(k, loc, begin0, blocks);
    Value end0 = arith::AddIOp::create(k, loc, begin, span);
    Value end = arith::MinUIOp::create(k, loc, end0, blocks);
    Value mine =
        scf::ForOp::create(
            k, loc, begin, end, i1, ValueRange{none},
            [&](OpBuilder &f, Location, Value at, ValueRange s) {
              Value v = memref::LoadOp::create(f, loc, sums, ValueRange{at});
              scf::YieldOp::create(
                  f, loc, ValueRange{arith::AddIOp::create(f, loc, s[0], v)});
            })
            .getResult(0);
    BlockScan scan = emitBlockScan(k, loc, mine, wg, thread);
    scf::ForOp::create(
        k, loc, begin, end, i1, ValueRange{scan.before},
        [&](OpBuilder &f, Location, Value at, ValueRange before) {
          Value v = memref::LoadOp::create(f, loc, sums, ValueRange{at});
          memref::StoreOp::create(f, loc, before[0], sums, ValueRange{at});
          scf::YieldOp::create(
              f, loc, ValueRange{arith::AddIOp::create(f, loc, before[0], v)});
        });
    Value lead =
        arith::CmpIOp::create(k, loc, arith::CmpIPredicate::eq, thread, i0);
    scf::IfOp::create(k, loc, lead, [&](OpBuilder &t, Location) {
      memref::StoreOp::create(t, loc, scan.total, start, ValueRange{n});
      scf::YieldOp::create(t, loc);
    });
  });

  // Each block scans its counts from the sum of the blocks before it.
  launch(blocks, [&](OpBuilder &k, Value block, Value thread, Value wg) {
    Counts c = prologue(k, block, thread);
    Value mine = sumOfFour(k, c);
    BlockScan scan = emitBlockScan(k, loc, mine, wg, thread);
    Value offset = memref::LoadOp::create(k, loc, sums, ValueRange{block});
    Value from = arith::AddIOp::create(k, loc, offset, scan.before);
    scf::ForOp::create(
        k, loc, c.i0, c.i4, c.i1, ValueRange{from},
        [&](OpBuilder &f, Location, Value at, ValueRange before) {
          Value item = arith::AddIOp::create(f, loc, c.first, at);
          Value in = arith::CmpIOp::create(f, loc, arith::CmpIPredicate::ult,
                                           item, n);
          Value safe = arith::SelectOp::create(f, loc, in, item, c.i0);
          Value v0 = memref::LoadOp::create(f, loc, counts, ValueRange{safe});
          Value v = arith::SelectOp::create(f, loc, in, v0, c.none);
          scf::IfOp::create(f, loc, in, [&](OpBuilder &s, Location) {
            memref::StoreOp::create(s, loc, before[0], start,
                                    ValueRange{item});
            if (cursor)
              memref::StoreOp::create(s, loc, before[0], cursor,
                                      ValueRange{item});
            scf::YieldOp::create(s, loc);
          });
          scf::YieldOp::create(
              f, loc, ValueRange{arith::AddIOp::create(f, loc, before[0], v)});
        });
  });
}

/// A launch of `grid` blocks of `block` threads, a thread an item below
/// `count`, running `body` for its item.
static void
launchPerItem(OpBuilder &b, Location loc, Value grid, Value block, Value one,
              Value count, function_ref<void(OpBuilder &, Value)> body) {
  auto op = gpu::LaunchOp::create(b, loc, grid, one, one, block, one, one);
  OpBuilder kernel = OpBuilder::atBlockEnd(&op.getBody().front());
  Value base =
      arith::MulIOp::create(kernel, loc, op.getBlockIds().x, block);
  Value item = arith::AddIOp::create(kernel, loc, base, op.getThreadIds().x);
  Value inside = arith::CmpIOp::create(kernel, loc, arith::CmpIPredicate::ult,
                                       item, count);
  scf::IfOp::create(kernel, loc, inside, [&](OpBuilder &then, Location) {
    body(then, item);
    scf::YieldOp::create(then, loc);
  });
  gpu::TerminatorOp::create(kernel, loc);
}

void kernels::emitSerialScan(OpBuilder &b, Location loc, Value sums,
                             Value count, Value out, Value at) {
  Value i0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value i1 = arith::ConstantIndexOp::create(b, loc, 1);
  Value none =
      arith::ConstantOp::create(b, loc, b.getI32Type(), b.getI32IntegerAttr(0));
  Value total =
      scf::ForOp::create(
          b, loc, i0, count, i1, ValueRange{none},
          [&](OpBuilder &f, Location, Value k, ValueRange before) {
            Value sum = memref::LoadOp::create(f, loc, sums, ValueRange{k});
            memref::StoreOp::create(f, loc, before[0], sums, ValueRange{k});
            scf::YieldOp::create(
                f, loc,
                ValueRange{arith::AddIOp::create(f, loc, before[0], sum)});
          })
          .getResult(0);
  memref::StoreOp::create(b, loc, total, out, ValueRange{at});
}

void kernels::emitChunkedScan(OpBuilder &b, Location loc, Value counts,
                              Value start, Value cursor, Value sums, Value n,
                              Value chunks, Value chunk, Value grid,
                              Value block, Value one) {
  // The counts of chunk `c`, from `begin` to `end`.
  auto bounds = [&](OpBuilder &t, Value c) {
    Value begin = arith::MulIOp::create(t, loc, c, chunk);
    Value full = arith::AddIOp::create(t, loc, begin, chunk);
    Value isShort =
        arith::CmpIOp::create(t, loc, arith::CmpIPredicate::ult, n, full);
    Value end = arith::SelectOp::create(t, loc, isShort, n, full);
    return std::make_pair(begin, end);
  };
  launchPerItem(b, loc, grid, block, one, chunks, [&](OpBuilder &t, Value c) {
    Value i1 = arith::ConstantIndexOp::create(t, loc, 1);
    Value none = arith::ConstantOp::create(t, loc, t.getI32Type(),
                                           t.getI32IntegerAttr(0));
    auto [begin, end] = bounds(t, c);
    Value sum =
        scf::ForOp::create(
            t, loc, begin, end, i1, ValueRange{none},
            [&](OpBuilder &f, Location, Value k, ValueRange partial) {
              Value v = memref::LoadOp::create(f, loc, counts, ValueRange{k});
              scf::YieldOp::create(
                  f, loc,
                  ValueRange{arith::AddIOp::create(f, loc, partial[0], v)});
            })
            .getResult(0);
    memref::StoreOp::create(t, loc, sum, sums, ValueRange{c});
  });
  {
    auto op = gpu::LaunchOp::create(b, loc, one, one, one, one, one, one);
    OpBuilder kernel = OpBuilder::atBlockEnd(&op.getBody().front());
    emitSerialScan(kernel, loc, sums, chunks, start, n);
    gpu::TerminatorOp::create(kernel, loc);
  }
  launchPerItem(b, loc, grid, block, one, chunks, [&](OpBuilder &t, Value c) {
    Value i1 = arith::ConstantIndexOp::create(t, loc, 1);
    auto [begin, end] = bounds(t, c);
    Value first = memref::LoadOp::create(t, loc, sums, ValueRange{c});
    scf::ForOp::create(
        t, loc, begin, end, i1, ValueRange{first},
        [&](OpBuilder &f, Location, Value k, ValueRange before) {
          Value v = memref::LoadOp::create(f, loc, counts, ValueRange{k});
          memref::StoreOp::create(f, loc, before[0], start, ValueRange{k});
          memref::StoreOp::create(f, loc, before[0], cursor, ValueRange{k});
          scf::YieldOp::create(
              f, loc, ValueRange{arith::AddIOp::create(f, loc, before[0], v)});
        });
  });
}

//===----------------------------------------------------------------------===//
// Counting sorts
//===----------------------------------------------------------------------===//

Value kernels::emitDeviceAtomicAdd(OpBuilder &b, Location loc, Value buffer,
                                   Value index, Value value) {
  Type wide = b.getI64Type();
  Value base = memref::ExtractAlignedPointerAsIndexOp::create(b, loc, buffer);
  Value baseInt = arith::IndexCastOp::create(b, loc, wide, base);
  Value indexInt = arith::IndexCastOp::create(b, loc, wide, index);
  Value four =
      arith::ConstantOp::create(b, loc, wide, b.getI64IntegerAttr(4));
  Value offset = arith::MulIOp::create(b, loc, indexInt, four);
  Value address = arith::AddIOp::create(b, loc, baseInt, offset);
  Value pointer = LLVM::IntToPtrOp::create(
      b, loc, LLVM::LLVMPointerType::get(b.getContext(), 1), address);
  return LLVM::AtomicRMWOp::create(b, loc, LLVM::AtomicBinOp::add, pointer,
                                   value, LLVM::AtomicOrdering::monotonic,
                                   StringRef("device"));
}

void kernels::emitCursorScatter(OpBuilder &b, Location loc, Value keys,
                                Value cursor, Value out, Value n, Value grid,
                                Value block, Value one, bool everyKey) {
  Type i32 = b.getI32Type();
  launchPerItem(b, loc, grid, block, one, n, [&](OpBuilder &t, Value item) {
    Value step = arith::ConstantOp::create(t, loc, i32, t.getI32IntegerAttr(1));
    Value zero =
        everyKey
            ? Value()
            : arith::ConstantOp::create(t, loc, i32, t.getI32IntegerAttr(0));
    Value key = memref::LoadOp::create(t, loc, keys, ValueRange{item});
    auto place = [&](OpBuilder &s) {
      Value k = arith::IndexCastOp::create(s, loc, s.getIndexType(), key);
      Value slot32 = emitDeviceAtomicAdd(s, loc, cursor, k, step);
      Value slot = arith::IndexCastOp::create(s, loc, s.getIndexType(), slot32);
      Value narrow = arith::IndexCastOp::create(s, loc, i32, item);
      memref::StoreOp::create(s, loc, narrow, out, ValueRange{slot});
    };
    if (everyKey) {
      place(t);
      return;
    }
    Value hasKey =
        arith::CmpIOp::create(t, loc, arith::CmpIPredicate::sge, key, zero);
    scf::IfOp::create(t, loc, hasKey, [&](OpBuilder &s, Location) {
      place(s);
      scf::YieldOp::create(s, loc);
    });
  });
}

void kernels::emitSortBins(OpBuilder &b, Location loc, Value start,
                           Value items, Value n, Value grid, Value block,
                           Value one, Value by) {
  Type index = b.getIndexType();
  // The value that orders `item` (i32).
  auto orderOf = [&](OpBuilder &t, Value item) -> Value {
    if (!by)
      return item;
    Value at = arith::IndexCastOp::create(t, loc, index, item);
    return memref::LoadOp::create(t, loc, by, ValueRange{at});
  };
  launchPerItem(b, loc, grid, block, one, n, [&](OpBuilder &t, Value bin) {
    Value i1 = arith::ConstantIndexOp::create(t, loc, 1);
    Value next = arith::AddIOp::create(t, loc, bin, i1);
    Value begin32 = memref::LoadOp::create(t, loc, start, ValueRange{bin});
    Value end32 = memref::LoadOp::create(t, loc, start, ValueRange{next});
    Value begin = arith::IndexCastOp::create(t, loc, index, begin32);
    Value end = arith::IndexCastOp::create(t, loc, index, end32);
    Value first = arith::AddIOp::create(t, loc, begin, i1);
    scf::ForOp::create(
        t, loc, first, end, i1, ValueRange(),
        [&](OpBuilder &f, Location, Value p, ValueRange) {
          Value value = memref::LoadOp::create(f, loc, items, ValueRange{p});
          Value mine = orderOf(f, value);
          // The hole moves down past the larger items before it.
          auto loop = scf::WhileOp::create(
              f, loc, TypeRange{index}, ValueRange{p},
              [&](OpBuilder &w, Location, ValueRange q) {
                Value more = arith::CmpIOp::create(
                    w, loc, arith::CmpIPredicate::ugt, q[0], begin);
                Value before = arith::SubIOp::create(w, loc, q[0], i1);
                Value safe =
                    arith::SelectOp::create(w, loc, more, before, begin);
                Value left =
                    memref::LoadOp::create(w, loc, items, ValueRange{safe});
                Value larger = arith::CmpIOp::create(
                    w, loc, arith::CmpIPredicate::sgt, orderOf(w, left), mine);
                Value go = arith::AndIOp::create(w, loc, more, larger);
                scf::ConditionOp::create(w, loc, go, q);
              },
              [&](OpBuilder &w, Location, ValueRange q) {
                Value before = arith::SubIOp::create(w, loc, q[0], i1);
                Value left =
                    memref::LoadOp::create(w, loc, items, ValueRange{before});
                memref::StoreOp::create(w, loc, left, items, q);
                scf::YieldOp::create(w, loc, before);
              });
          memref::StoreOp::create(f, loc, value, items,
                                  ValueRange{loop.getResult(0)});
          scf::YieldOp::create(f, loc);
        });
  });
}

void kernels::emitRankInBins(OpBuilder &b, Location loc, Value items,
                             Value keys, Value start, Value ranked,
                             Value bins, Value grid, Value block, Value one) {
  Type index = b.getIndexType();
  auto op = gpu::LaunchOp::create(b, loc, grid, one, one, block, one, one);
  OpBuilder kernel = OpBuilder::atBlockEnd(&op.getBody().front());
  Value base = arith::MulIOp::create(kernel, loc, op.getBlockIds().x, block);
  Value p = arith::AddIOp::create(kernel, loc, base, op.getThreadIds().x);
  // The places beyond the items that have a bin hold no item.
  Value total32 = memref::LoadOp::create(kernel, loc, start, ValueRange{bins});
  Value total = arith::IndexCastOp::create(kernel, loc, index, total32);
  Value inside =
      arith::CmpIOp::create(kernel, loc, arith::CmpIPredicate::ult, p, total);
  scf::IfOp::create(kernel, loc, inside, [&](OpBuilder &t, Location) {
    Value i1 = arith::ConstantIndexOp::create(t, loc, 1);
    Value none = arith::ConstantOp::create(t, loc, t.getI32Type(),
                                           t.getI32IntegerAttr(0));
    Value value = memref::LoadOp::create(t, loc, items, ValueRange{p});
    Value item = arith::IndexCastOp::create(t, loc, index, value);
    Value k32 = memref::LoadOp::create(t, loc, keys, ValueRange{item});
    Value k = arith::IndexCastOp::create(t, loc, index, k32);
    Value next = arith::AddIOp::create(t, loc, k, i1);
    Value begin32 = memref::LoadOp::create(t, loc, start, ValueRange{k});
    Value end32 = memref::LoadOp::create(t, loc, start, ValueRange{next});
    Value begin = arith::IndexCastOp::create(t, loc, index, begin32);
    Value end = arith::IndexCastOp::create(t, loc, index, end32);
    Value rank32 =
        scf::ForOp::create(
            t, loc, begin, end, i1, ValueRange{none},
            [&](OpBuilder &f, Location, Value q, ValueRange before) {
              Value other =
                  memref::LoadOp::create(f, loc, items, ValueRange{q});
              Value smaller = arith::CmpIOp::create(
                  f, loc, arith::CmpIPredicate::slt, other, value);
              Value oneMore =
                  arith::ExtUIOp::create(f, loc, f.getI32Type(), smaller);
              scf::YieldOp::create(
                  f, loc,
                  ValueRange{arith::AddIOp::create(f, loc, before[0], oneMore)});
            })
            .getResult(0);
    Value rank = arith::IndexCastOp::create(t, loc, index, rank32);
    Value slot = arith::AddIOp::create(t, loc, begin, rank);
    memref::StoreOp::create(t, loc, value, ranked, ValueRange{slot});
    scf::YieldOp::create(t, loc);
  });
  gpu::TerminatorOp::create(kernel, loc);
}
