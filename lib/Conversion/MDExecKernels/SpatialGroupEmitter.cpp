#include "mdir/Conversion/SpatialGroupEmitter.h"
#include "mdir/Conversion/CompactionEmitter.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
using namespace mdir::kernels;

SpatialGroupPlanResult mdir::kernels::emitSpatialGroupPlan(
    OpBuilder &b, Location loc, const SpatialGroupPlanBuffers &v) {
  auto ix = [&](OpBuilder &at, int64_t n) { return createIndex(at, loc, n); };
  auto word = [&](OpBuilder &at, int64_t n) -> Value {
    return arith::ConstantIntOp::create(at, loc, n, 32);
  };
  auto index = [&](OpBuilder &at, Value value) -> Value {
    return arith::IndexCastUIOp::create(at, loc, at.getIndexType(), value);
  };
  auto toWord = [&](OpBuilder &at, Value value) -> Value {
    return arith::IndexCastOp::create(at, loc, at.getI32Type(), value);
  };
  auto key = [&](OpBuilder &at, Value keys, Value place) -> Value {
    Value unit = memref::LoadOp::create(at, loc, v.order, ValueRange{place});
    return memref::LoadOp::create(at, loc, keys, ValueRange{index(at, unit)});
  };
  Value one = ix(b, 1), block = ix(b, 256);
  Value scanSize = arith::MaxUIOp::create(b, loc, v.size, one);
  auto launch = [&](OpBuilder &host, Value count, auto body) {
    Value grid = arith::MaxUIOp::create(host, loc, one,
        arith::DivUIOp::create(host, loc,
            arith::AddIOp::create(host, loc, count, ix(host, 255)), block));
    auto kernel = gpu::LaunchOp::create(host, loc, grid, one, one, block, one, one);
    OpBuilder at = OpBuilder::atBlockEnd(&kernel.getBody().front());
    Value item = arith::AddIOp::create(at, loc,
        arith::MulIOp::create(at, loc, kernel.getBlockIds().x, block), kernel.getThreadIds().x);
    Value inside = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ult, item, count);
    scf::IfOp::create(at, loc, inside, [&](OpBuilder &in, Location) {
      body(in, item);
      scf::YieldOp::create(in, loc);
    });
    gpu::TerminatorOp::create(at, loc);
  };
  auto threshold = [&](OpBuilder &at, Value item,
                       const std::array<Value, 2> &values) -> Value {
    Value first = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ult,
        item, v.firstClassSize);
    return arith::SelectOp::create(at, loc, first, values[0], values[1]);
  };
  // The most units of the group that holds `item`: firstWidth or 16.
  auto width = [&](OpBuilder &at, Value item) -> Value {
    Value first = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ult,
        item, v.firstClassSize);
    return arith::SelectOp::create(at, loc, first, ix(at, v.firstWidth), ix(at, 16));
  };

  launch(b, scanSize, [&](OpBuilder &out, Value item) {
    // Word 4 of the controls records the last pass that split a group (one
    // based), so that the passes are known whatever number ran unread;
    // words 5 to 7 are unused and read back with it. Words 0 to 3 start
    // at zero for an empty input, which no later kernel of a pass visits.
    Value head = arith::CmpIOp::create(out, loc, arith::CmpIPredicate::eq, item, ix(out, 0));
    scf::IfOp::create(out, loc, head, [&](OpBuilder &at, Location) {
      for (int64_t slot = 0; slot != 8; ++slot)
        memref::StoreOp::create(at, loc, word(at, 0), v.control, ValueRange{ix(at, slot)});
      scf::YieldOp::create(at, loc);
    });
    // Cleared once a plan: the segments only grow from pass to pass, so
    // each pass rewrites every start, length and count an earlier one
    // wrote, and the tail beyond the segments stays zero for the scan of
    // the counts.
    for (Value field : {v.starts, v.lengths, v.counts})
      memref::StoreOp::create(out, loc, word(out, 0), field, ValueRange{item});
    Value valid = arith::CmpIOp::create(out, loc, arith::CmpIPredicate::ult, item, v.size);
    auto flag = scf::IfOp::create(out, loc, valid, [&](OpBuilder &at, Location) {
    Value first = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::eq, item, ix(at, 0));
    Value previous = arith::SelectOp::create(at, loc, first, item,
        arith::SubIOp::create(at, loc, item, one));
    Value columnChanged = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ne,
        key(at, v.columns, item), key(at, v.columns, previous));
    Value gap = arith::SubIOp::create(at, loc,
        key(at, v.zKeys, item), key(at, v.zKeys, previous));
    Value separated = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ugt,
        gap, threshold(at, item, v.gap));
    Value classBoundary = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::eq,
        item, v.firstClassSize);
    Value cut = arith::OrIOp::create(at, loc, arith::OrIOp::create(at, loc, first, classBoundary),
        arith::OrIOp::create(at, loc, columnChanged, separated));
    scf::YieldOp::create(at, loc,
        ValueRange{arith::ExtUIOp::create(at, loc, at.getI32Type(), cut)});
    }, [&](OpBuilder &empty, Location) {
      scf::YieldOp::create(empty, loc, ValueRange{word(empty, 0)});
    });
    memref::StoreOp::create(out, loc, flag.getResult(0), v.flags, ValueRange{item});
  });

  Value yes = arith::ConstantOp::create(b, loc, b.getBoolAttr(true));
  // One refinement pass. A pass after convergence changes nothing: the
  // flags only accumulate, and words 0 to 3 are recounted.
  auto pass = [&](OpBuilder &at, Value ordinal) {
    emitDeviceScan(at, loc, v.flags, v.segments, Value{}, scanSize,
        v.scanScratch, v.scanBlocks, block, one);
    launch(at, v.size, [&](OpBuilder &in, Value i) {
      Value prefix = memref::LoadOp::create(in, loc, v.segments, ValueRange{i});
      Value flag = memref::LoadOp::create(in, loc, v.flags, ValueRange{i});
      Value segment = arith::SubIOp::create(in, loc,
          arith::AddIOp::create(in, loc, prefix, flag), word(in, 1));
      // Normalize the scan in place into stable segment IDs before any
      // refinement changes flags. No later kernel reads a changing flag
      // to recover its segment ID.
      memref::StoreOp::create(in, loc, segment, v.segments, ValueRange{i});
      Value begins = arith::CmpIOp::create(in, loc, arith::CmpIPredicate::ne, flag, word(in, 0));
      scf::IfOp::create(in, loc, begins, [&](OpBuilder &start, Location) {
        memref::StoreOp::create(start, loc, toWord(start, i), v.starts,
            ValueRange{index(start, segment)});
        scf::YieldOp::create(start, loc);
      });
      // Word 0 is this pass's segment count; words 1 to 3 are cleared here
      // for the later kernels of the pass, which add to them.
      Value leader = arith::CmpIOp::create(in, loc, arith::CmpIPredicate::eq, i, ix(in, 0));
      scf::IfOp::create(in, loc, leader, [&](OpBuilder &total, Location) {
        Value count = memref::LoadOp::create(total, loc, v.segments, ValueRange{v.size});
        memref::StoreOp::create(total, loc, count, v.control, ValueRange{ix(total, 0)});
        for (int64_t slot = 1; slot != 4; ++slot)
          memref::StoreOp::create(total, loc, word(total, 0), v.control, ValueRange{ix(total, slot)});
        scf::YieldOp::create(total, loc);
      });
    });
    launch(at, v.size, [&](OpBuilder &in, Value i) {
      Value next = arith::AddIOp::create(in, loc, i, one);
      Value hasNext = arith::CmpIOp::create(in, loc, arith::CmpIPredicate::ult, next, v.size);
      Value safe = arith::SelectOp::create(in, loc, hasNext, next, i);
      Value nextFlag = memref::LoadOp::create(in, loc, v.flags, ValueRange{safe});
      Value last = arith::OrIOp::create(in, loc,
          arith::XOrIOp::create(in, loc, hasNext, yes),
          arith::CmpIOp::create(in, loc, arith::CmpIPredicate::ne, nextFlag, word(in, 0)));
      scf::IfOp::create(in, loc, last, [&](OpBuilder &end, Location) {
        Value segment = index(end,
            memref::LoadOp::create(end, loc, v.segments, ValueRange{i}));
        Value start = index(end, memref::LoadOp::create(end, loc, v.starts, ValueRange{segment}));
        Value length = arith::SubIOp::create(end, loc, next, start);
        Value w = width(end, i);
        Value count = arith::DivUIOp::create(end, loc,
            arith::AddIOp::create(end, loc, length, arith::SubIOp::create(end, loc, w, one)), w);
        memref::StoreOp::create(end, loc, toWord(end, length), v.lengths, ValueRange{segment});
        memref::StoreOp::create(end, loc, toWord(end, count), v.counts, ValueRange{segment});
        Value first = arith::CmpIOp::create(end, loc, arith::CmpIPredicate::ult, i, v.firstClassSize);
        Value slot = arith::SelectOp::create(end, loc, first, ix(end, 1), ix(end, 2));
        emitDeviceAtomicAdd(end, loc, v.control, slot, toWord(end, count));
        scf::YieldOp::create(end, loc);
      });
    });
    launch(at, v.size, [&](OpBuilder &in, Value i) {
      Value segment = index(in, memref::LoadOp::create(in, loc, v.segments, ValueRange{i}));
      Value start = index(in, memref::LoadOp::create(in, loc, v.starts, ValueRange{segment}));
      Value position = arith::SubIOp::create(in, loc, i, start);
      Value w = width(in, i);
      Value groupStart = arith::CmpIOp::create(in, loc, arith::CmpIPredicate::eq,
          arith::RemUIOp::create(in, loc, position, w), ix(in, 0));
      scf::IfOp::create(in, loc, groupStart, [&](OpBuilder &group, Location) {
        Value length = index(group,
            memref::LoadOp::create(group, loc, v.lengths, ValueRange{segment}));
        Value last = arith::MinUIOp::create(group, loc,
            arith::AddIOp::create(group, loc, i, arith::SubIOp::create(group, loc, w, one)),
            arith::SubIOp::create(group, loc,
                arith::AddIOp::create(group, loc, start, length), one));
        Value span = arith::SubIOp::create(group, loc,
            key(group, v.zKeys, last), key(group, v.zKeys, i));
        Value tooWide = arith::CmpIOp::create(group, loc, arith::CmpIPredicate::ugt,
            span, threshold(group, i, v.span));
        scf::IfOp::create(group, loc, tooWide, [&](OpBuilder &split, Location) {
          auto best = scf::ForOp::create(split, loc,
              arith::AddIOp::create(split, loc, i, one),
              arith::AddIOp::create(split, loc, last, one), one,
              ValueRange{word(split, 0), i},
              [&](OpBuilder &gap, Location, Value j, ValueRange state) {
            Value difference = arith::SubIOp::create(gap, loc,
                key(gap, v.zKeys, j), key(gap, v.zKeys,
                    arith::SubIOp::create(gap, loc, j, one)));
            Value take = arith::CmpIOp::create(gap, loc, arith::CmpIPredicate::uge,
                difference, state[0]);
            scf::YieldOp::create(gap, loc, ValueRange{
                arith::SelectOp::create(gap, loc, take, difference, state[0]),
                arith::SelectOp::create(gap, loc, take, j, state[1])});
          });
          memref::StoreOp::create(split, loc, word(split, 1), v.flags,
              ValueRange{best.getResult(1)});
          emitDeviceAtomicAdd(split, loc, v.control, ix(split, 3), word(split, 1));
          memref::AtomicRMWOp::create(split, loc, arith::AtomicRMWKind::maxu,
              toWord(split, arith::AddIOp::create(split, loc, ordinal, one)), v.control,
              ValueRange{ix(split, 4)});
          scf::YieldOp::create(split, loc);
        });
        scf::YieldOp::create(group, loc);
      });
    });
  };
  Value host = memref::AllocaOp::create(b, loc, MemRefType::get({8}, b.getI32Type()));
  auto readControls = [&](OpBuilder &at) -> Value {
    Type token = gpu::AsyncTokenType::get(at.getContext());
    Value ready = gpu::WaitOp::create(at, loc, token, ValueRange()).getAsyncToken();
    Value copied = gpu::MemcpyOp::create(at, loc, token, ValueRange{ready},
        host, v.control).getAsyncToken();
    gpu::WaitOp::create(at, loc, Type(), ValueRange{copied});
    Value changed = memref::LoadOp::create(at, loc, host, ValueRange{ix(at, 3)});
    return arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ne, changed, word(at, 0));
  };
  // The first `unread` passes run without a readback (at least one, at
  // most the bound of N+1); the controls are then read once, and passes
  // continue one readback each only while the last one split a group. A
  // caller that knows the passes of a similar plan (the previous build)
  // thus usually reads the controls once.
  Value limit = arith::AddIOp::create(b, loc, v.size, one);
  Value unread = arith::MinUIOp::create(b, loc, limit,
      arith::MaxUIOp::create(b, loc, one, v.unreadPasses ? v.unreadPasses : one));
  scf::ForOp::create(b, loc, ix(b, 0), unread, one, ValueRange{},
      [&](OpBuilder &at, Location, Value ordinal, ValueRange) {
    pass(at, ordinal);
    scf::YieldOp::create(at, loc);
  });
  Value initial = readControls(b);
  auto refined = scf::WhileOp::create(b, loc,
      TypeRange{b.getI1Type(), b.getIndexType()}, ValueRange{initial, unread},
      [&](OpBuilder &at, Location, ValueRange state) {
        Value bounded = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ult, state[1], limit);
        scf::ConditionOp::create(at, loc,
            arith::AndIOp::create(at, loc, state[0], bounded), state);
      }, [&](OpBuilder &at, Location, ValueRange state) {
    pass(at, state[1]);
    Value changed = readControls(at);
    scf::YieldOp::create(at, loc, ValueRange{changed,
        arith::AddIOp::create(at, loc, state[1], one)});
  });
  Value done = arith::XOrIOp::create(b, loc, refined.getResult(0), yes);
  cf::AssertOp::create(b, loc, done, "spatial group refinement did not converge");
  emitDeviceScan(b, loc, v.counts, v.offsets, Value{}, scanSize,
      v.scanScratch, v.scanBlocks, block, one);
  std::array<Value, 2> groups;
  for (int64_t kind = 0; kind != 2; ++kind)
    groups[kind] = index(b, memref::LoadOp::create(b, loc, host, ValueRange{ix(b, kind + 1)}));
  Value productive = index(b, memref::LoadOp::create(b, loc, host, ValueRange{ix(b, 4)}));
  return {groups, arith::AddIOp::create(b, loc, productive, one)};
}
