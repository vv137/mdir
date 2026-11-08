#include "mdir/Conversion/TemplateStageKernels.h"
#include "mdir/Conversion/CompactionEmitter.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mdir/Runtime/TemplatePairsABI.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
using namespace mdir::kernels;
namespace {
static Value wide(OpBuilder &b, Location loc, int64_t n) {
  return arith::ConstantIntOp::create(b, loc, n, 64);
}
static Value word(OpBuilder &b, Location loc, int64_t n) {
  return arith::ConstantIntOp::create(b, loc, n, 32);
}
static Value leader(OpBuilder &b, Location loc, Value lane) {
  return arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, lane, word(b, loc, 0));
}
static Value address(OpBuilder &b, Location loc, Value n) {
  return arith::IndexCastUIOp::create(b, loc, b.getIndexType(), n);
}
static Value broadcast(OpBuilder &b, Location loc, Value value) {
  Value shift = wide(b, loc, 32);
  auto take = [&](Value part) -> Value {
    return gpu::ShuffleOp::create(b, loc, part, word(b, loc, 0),
        word(b, loc, 32), gpu::ShuffleMode::IDX).getShuffleResult();
  };
  Value low = arith::TruncIOp::create(b, loc, b.getI32Type(), value);
  Value high = arith::TruncIOp::create(b, loc, b.getI32Type(),
      arith::ShRUIOp::create(b, loc, value, shift));
  return arith::OrIOp::create(b, loc,
      arith::ExtUIOp::create(b, loc, b.getI64Type(), take(low)),
      arith::ShLIOp::create(b, loc,
          arith::ExtUIOp::create(b, loc, b.getI64Type(), take(high)), shift));
}
static void flag(OpBuilder &b, Location loc, const TemplateStageBuffers &v, int64_t bit) {
  memref::AtomicRMWOp::create(b, loc, arith::AtomicRMWKind::ori,
      wide(b, loc, bit), v.counters,
      ValueRange{createIndex(b, loc, TEMPLATE_COUNTER_FLAGS)});
}
}

TemplateStageState mdir::kernels::emitAppendTemplateStage(OpBuilder &b, Location loc,
    const TemplateStageBuffers &v, TemplateStageState previous, Value keep,
    ValueRange record) {
  assert(record.size() == 4);
  Value ballot = emitWarpBallot(b, loc, keep);
  Value rank = arith::ExtUIOp::create(b, loc, b.getI64Type(),
      emitWarpRank(b, loc, ballot, emitLanesBelow(b, loc, v.lane)));
  Value kept = arith::ExtUIOp::create(b, loc, b.getI64Type(), emitWarpCount(b, loc, ballot));
  Value count = arith::AddIOp::create(b, loc, previous.count, kept);
  auto chunks = [&](Value n) -> Value {
    return arith::DivUIOp::create(b, loc,
        arith::AddIOp::create(b, loc, n, wide(b, loc, 63)), wide(b, loc, 64));
  };
  Value newChunks = chunks(count);
  Value needs = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ugt,
                                       newChunks, chunks(previous.count));
  auto reserved = scf::IfOp::create(b, loc, needs,
      [&](OpBuilder &at, Location) {
        auto ticket = scf::IfOp::create(at, loc, leader(at, loc, v.lane),
            [&](OpBuilder &one, Location) {
              Value id = memref::AtomicRMWOp::create(one, loc, arith::AtomicRMWKind::addi,
                  wide(one, loc, 1), v.counters,
                  ValueRange{createIndex(one, loc, TEMPLATE_COUNTER_CHUNKS)});
              Value overflow = arith::CmpIOp::create(one, loc, arith::CmpIPredicate::uge, id, v.capacity);
              scf::IfOp::create(one, loc, overflow, [&](OpBuilder &bad, Location) {
                flag(bad, loc, v, TEMPLATE_BUILD_CAPACITY_OVERFLOW);
                scf::YieldOp::create(bad, loc);
              });
              scf::YieldOp::create(one, loc, ValueRange{id});
            }, [&](OpBuilder &other, Location) {
              scf::YieldOp::create(other, loc, ValueRange{wide(other, loc, 0)});
            });
        scf::YieldOp::create(at, loc, ValueRange{broadcast(at, loc, ticket.getResult(0))});
      }, [&](OpBuilder &at, Location) {
        scf::YieldOp::create(at, loc, ValueRange{wide(at, loc, -1)});
      });
  Value nextChunk = reserved.getResult(0);
  Value newOrdinal = arith::SubIOp::create(b, loc, newChunks, wide(b, loc, 1));
  Value newBase = arith::MulIOp::create(b, loc, newOrdinal, wide(b, loc, 64));
  Value ordinal = arith::AddIOp::create(b, loc, previous.count, rank);
  Value inNew = arith::AndIOp::create(b, loc, needs,
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::uge, ordinal, newBase));
  Value destination = arith::SelectOp::create(b, loc, inNew, nextChunk, previous.chunk);
  Value fits = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, destination, v.capacity);
  Value writes = arith::AndIOp::create(b, loc, keep, fits);
  scf::IfOp::create(b, loc, writes, [&](OpBuilder &at, Location) {
    Value slot = arith::AddIOp::create(at, loc,
        arith::MulIOp::create(at, loc, destination, wide(at, loc, 64)),
        arith::RemUIOp::create(at, loc, ordinal, wide(at, loc, 64)));
    Value first = arith::MulIOp::create(at, loc, slot, wide(at, loc, 4));
    for (unsigned column = 0; column != 4; ++column)
      memref::StoreOp::create(at, loc, record[column], v.entries,
          ValueRange{address(at, loc, arith::AddIOp::create(at, loc, first, wide(at, loc, column)))});
    scf::YieldOp::create(at, loc);
  });
  Value nonempty = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne, kept, wide(b, loc, 0));
  Value publishes = arith::AndIOp::create(b, loc, nonempty, leader(b, loc, v.lane));
  scf::IfOp::create(b, loc, publishes, [&](OpBuilder &at, Location) {
    Value oldFits = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ult, previous.chunk, v.capacity);
    scf::IfOp::create(at, loc, oldFits, [&](OpBuilder &old, Location) {
      Value oldBase = arith::MulIOp::create(old, loc,
          arith::DivUIOp::create(old, loc,
              arith::SubIOp::create(old, loc, previous.count, wide(old, loc, 1)), wide(old, loc, 64)),
          wide(old, loc, 64));
      Value finalCount = arith::MinUIOp::create(old, loc, wide(old, loc, 64),
          arith::SubIOp::create(old, loc, count, oldBase));
      Value field = arith::AddIOp::create(old, loc,
          arith::MulIOp::create(old, loc, previous.chunk, wide(old, loc, 4)), wide(old, loc, 1));
      memref::StoreOp::create(old, loc,
          arith::TruncIOp::create(old, loc, old.getI32Type(), finalCount), v.units,
          ValueRange{address(old, loc, field)});
      scf::YieldOp::create(old, loc);
    });
    Value newFits = arith::AndIOp::create(at, loc, needs,
        arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ult, nextChunk, v.capacity));
    scf::IfOp::create(at, loc, newFits, [&](OpBuilder &fresh, Location) {
      Value fields[] = {v.row,
          arith::TruncIOp::create(fresh, loc, fresh.getI32Type(), arith::SubIOp::create(fresh, loc, count, newBase)),
          arith::TruncIOp::create(fresh, loc, fresh.getI32Type(), newOrdinal), word(fresh, loc, 0)};
      Value first = arith::MulIOp::create(fresh, loc, nextChunk, wide(fresh, loc, 4));
      for (unsigned column = 0; column != 4; ++column)
        memref::StoreOp::create(fresh, loc, fields[column], v.units,
            ValueRange{address(fresh, loc, arith::AddIOp::create(fresh, loc, first, wide(fresh, loc, column)))});
      scf::YieldOp::create(fresh, loc);
    });
    scf::YieldOp::create(at, loc);
  });
  return {count, arith::SelectOp::create(b, loc, needs, nextChunk, previous.chunk)};
}

void mdir::kernels::emitFinishTemplateStage(OpBuilder &b, Location loc,
    const TemplateStageBuffers &v, TemplateStageState state) {
  scf::IfOp::create(b, loc, leader(b, loc, v.lane), [&](OpBuilder &at, Location) {
    Value total = memref::AtomicRMWOp::create(at, loc, arith::AtomicRMWKind::addi,
        state.count, v.counters, ValueRange{createIndex(at, loc, TEMPLATE_COUNTER_ENTRIES)});
    total = arith::AddIOp::create(at, loc, total, state.count);
    Value limit = wide(at, loc, INT32_MAX);
    Value overflow = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ugt, total, limit);
    scf::IfOp::create(at, loc, overflow, [&](OpBuilder &bad, Location) {
      flag(bad, loc, v, TEMPLATE_BUILD_INDEX_OVERFLOW);
      scf::YieldOp::create(bad, loc);
    });
    Value bounded = arith::MinUIOp::create(at, loc, state.count, limit);
    memref::StoreOp::create(at, loc,
        arith::TruncIOp::create(at, loc, at.getI32Type(), bounded), v.counts,
        ValueRange{address(at, loc, arith::ExtUIOp::create(at, loc, at.getI64Type(), v.row))});
    scf::YieldOp::create(at, loc);
  });
}
