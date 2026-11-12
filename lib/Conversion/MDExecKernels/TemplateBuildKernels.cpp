#include "mdir/Conversion/TemplateBuildKernels.h"
#include "mdir/Conversion/CompactionEmitter.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mdir/Runtime/TemplatePairsABI.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
using namespace mdir::kernels;

void mdir::kernels::emitCompactTemplatePairs(OpBuilder &b, Location loc,
                                             const TemplateBuildBuffers &v) {
  auto ix = [&](OpBuilder &at, int64_t n) { return createIndex(at, loc, n); };
  auto index = [&](OpBuilder &at, Value n) -> Value {
    return arith::IndexCastUIOp::create(at, loc, at.getIndexType(), n);
  };
  auto word = [&](OpBuilder &at, Value n) -> Value {
    return arith::IndexCastOp::create(at, loc, at.getI32Type(), n);
  };
  auto add = [&](OpBuilder &at, Value a, Value c) -> Value {
    return arith::AddIOp::create(at, loc, a, c);
  };
  auto mul = [&](OpBuilder &at, Value a, int64_t n) -> Value {
    return arith::MulIOp::create(at, loc, a, ix(at, n));
  };
  auto load = [&](OpBuilder &at, Value buffer, Value i) -> Value {
    return memref::LoadOp::create(at, loc, buffer, ValueRange{i});
  };
  auto store = [&](OpBuilder &at, Value buffer, Value i, Value value) {
    memref::StoreOp::create(at, loc, value, buffer, ValueRange{i});
  };
  auto recordStore = [&](OpBuilder &at, Value buffer, Value record,
                          unsigned column, Value value) {
    store(at, buffer, add(at, mul(at, record, 4), ix(at, column)), value);
  };
  // The search fits when its flags are clear and its entries are within
  // the capacity: only then are the staged chunks and the scanned offsets
  // within the arena. An overflowed search publishes nothing here; the
  // caller grows the arena and searches again.
  auto fits = [&](OpBuilder &at) -> Value {
    if (!v.counters) return arith::ConstantOp::create(at, loc, at.getBoolAttr(true));
    Value flags = load(at, v.counters, ix(at, TEMPLATE_COUNTER_FLAGS));
    Value entries = load(at, v.counters, ix(at, TEMPLATE_COUNTER_ENTRIES));
    Value zero = arith::ConstantIntOp::create(at, loc, 0, 64);
    return arith::AndIOp::create(at, loc,
        arith::CmpIOp::create(at, loc, arith::CmpIPredicate::eq, flags, zero),
        arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ule, entries, v.entryCapacity));
  };
  Value one = ix(b, 1), block = ix(b, 256);
  auto launch = [&](Value n, auto body) {
    Value grid = arith::DivUIOp::create(b, loc, add(b, n, ix(b, 255)), block);
    grid = arith::MaxUIOp::create(b, loc, grid, one);
    auto kernel = gpu::LaunchOp::create(b, loc, grid, one, one, block, one, one);
    OpBuilder at = OpBuilder::atBlockEnd(&kernel.getBody().front());
    Value item = add(at, mul(at, kernel.getBlockIds().x, 256), kernel.getThreadIds().x);
    Value inside = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ult, item, n);
    scf::IfOp::create(at, loc, inside, [&](OpBuilder &in, Location) {
      body(in, item);
      scf::YieldOp::create(in, loc);
    });
    gpu::TerminatorOp::create(at, loc);
  };
  launch(v.rows, [&](OpBuilder &at, Value row) {
    Value count = index(at, load(at, v.counts, row));
    Value units = arith::DivUIOp::create(at, loc, add(at, count, ix(at, 63)), ix(at, 64));
    store(at, v.workCounts, row, word(at, units));
  });
  emitDeviceScan(b, loc, v.counts, v.entryOffsets, Value(), v.rows,
                  v.scanScratch, v.scanBlocks, block, one);
  emitDeviceScan(b, loc, v.workCounts, v.workOffsets, Value(), v.rows,
                  v.scanScratch, v.scanBlocks, block, one);
  Value waterRows = v.groups.waterGroups;
  Value atomRows = v.groups.atomGroups;
  Value ssBegin = add(b, waterRows, atomRows);
  auto classify = [&](OpBuilder &at, Value row) -> Value {
    Value ww = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ult, row, waterRows);
    Value sw = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ult, row, ssBegin);
    return arith::SelectOp::create(at, loc, ww, ix(at, 0),
        arith::SelectOp::create(at, loc, sw, ix(at, 1), ix(at, 2)));
  };
  auto firstPlace = [&](OpBuilder &at, Value row) -> Value {
    Value ww = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ult, row, waterRows);
    Value sw = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ult, row, ssBegin);
    Value atomRow = arith::SubIOp::create(at, loc, row,
        arith::SelectOp::create(at, loc, sw, waterRows, ssBegin));
    Value group = arith::SelectOp::create(at, loc, ww, row, add(at, v.waters, atomRow));
    return index(at, load(at, v.groups.starts, group));
  };
  Value rowsOrClasses = arith::MaxUIOp::create(b, loc, v.rows, ix(b, 3));
  launch(rowsOrClasses, [&](OpBuilder &gated, Value row) {
    scf::IfOp::create(gated, loc, fits(gated), [&](OpBuilder &at, Location) {
    Value exists = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ult, row, v.rows);
    scf::IfOp::create(at, loc, exists, [&](OpBuilder &in, Location) {
      Value fields[] = {word(in, classify(in, row)), word(in, firstPlace(in, row)),
                       load(in, v.entryOffsets, row), load(in, v.counts, row)};
      for (unsigned column = 0; column != 4; ++column) {
        recordStore(in, v.outerRows, row, column, fields[column]);
        recordStore(in, v.innerRows, row, column, fields[column]);
      }
      scf::YieldOp::create(in, loc);
    });
    Value header = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ult, row, ix(at, 3));
    scf::IfOp::create(at, loc, header, [&](OpBuilder &in, Location) {
      Value zero = arith::CmpIOp::create(in, loc, arith::CmpIPredicate::eq, row, ix(in, 0));
      scf::IfOp::create(in, loc, zero, [&](OpBuilder &total, Location) {
        store(total, v.arena, ix(total, TEMPLATE_PAIR_ROWS), word(total, v.rows));
        scf::YieldOp::create(total, loc);
      });
      Value first = arith::CmpIOp::create(in, loc, arith::CmpIPredicate::eq, row, ix(in, 1));
      Value begin = arith::SelectOp::create(in, loc, zero, ix(in, 0),
          arith::SelectOp::create(in, loc, first, waterRows, ssBegin));
      Value end = arith::SelectOp::create(in, loc, zero, waterRows,
          arith::SelectOp::create(in, loc, first, ssBegin, v.rows));
      Value fields[] = {word(in, begin), word(in, end), load(in, v.entryOffsets, begin),
          load(in, v.entryOffsets, end), load(in, v.workOffsets, begin), load(in, v.workOffsets, end)};
      for (unsigned column = 0; column != 6; ++column)
        for (int64_t partition : {TEMPLATE_PAIR_OUTER_RANGES, TEMPLATE_PAIR_INNER_RANGES})
          store(in, v.arena, add(in, ix(in, partition + column), mul(in, row, 6)), fields[column]);
      scf::YieldOp::create(in, loc);
    });
    scf::YieldOp::create(at, loc);
    });
  });
  // A warp per staging chunk: lane l copies records l and l+32, so each
  // store of a column is coalesced. The search already compacted the
  // members of each chunk with a ballot; lane 0 writes the work record.
  launch(mul(b, v.unitCapacity, 32), [&](OpBuilder &at, Value item) {
    Value chunk = arith::DivUIOp::create(at, loc, item, ix(at, 32));
    Value lane = arith::RemUIOp::create(at, loc, item, ix(at, 32));
    Value used = index(at, load(at, v.arena, ix(at, TEMPLATE_PAIR_STAGE_COUNT)));
    Value exists = arith::AndIOp::create(at, loc,
        arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ult, chunk, used), fits(at));
    scf::IfOp::create(at, loc, exists, [&](OpBuilder &in, Location) {
      Value base = mul(in, chunk, 4);
      Value row = index(in, load(in, v.stageUnits, base));
      Value count = index(in, load(in, v.stageUnits, add(in, base, ix(in, 1))));
      Value ordinal = index(in, load(in, v.stageUnits, add(in, base, ix(in, 2))));
      Value destination = add(in, index(in, load(in, v.entryOffsets, row)), mul(in, ordinal, 64));
      Value leader = arith::CmpIOp::create(in, loc, arith::CmpIPredicate::eq, lane, ix(in, 0));
      scf::IfOp::create(in, loc, leader, [&](OpBuilder &first, Location) {
        Value work = add(first, index(first, load(first, v.workOffsets, row)), ordinal);
        // The outer record names its slice of the row and the row's end,
        // which the prune re-chunks; the inner one its own entries.
        Value tag = classify(first, row);
        Value rowEnd = add(first, index(first, load(first, v.entryOffsets, row)),
                           index(first, load(first, v.counts, row)));
        Value fields[] = {word(first, tag), word(first, firstPlace(first, row)),
                         word(first, destination), word(first, add(first, destination, count))};
        Value outer[] = {word(first, add(first, tag, mul(first, ordinal, 4))), fields[1],
                         fields[2], word(first, rowEnd)};
        for (unsigned column = 0; column != 4; ++column) {
          recordStore(first, v.outerUnits, work, column, outer[column]);
          recordStore(first, v.innerUnits, work, column, fields[column]);
        }
        scf::YieldOp::create(first, loc);
      });
      for (int64_t half = 0; half != 2; ++half) {
        Value i = add(in, lane, ix(in, 32 * half));
        Value copies = arith::CmpIOp::create(in, loc, arith::CmpIPredicate::ult, i, count);
        scf::IfOp::create(in, loc, copies, [&](OpBuilder &copy, Location) {
          Value source = mul(copy, add(copy, mul(copy, chunk, 64), i), 4);
          Value target = mul(copy, add(copy, destination, i), 4);
          for (unsigned column = 0; column != 4; ++column) {
            Value value = load(copy, v.stageEntries, add(copy, source, ix(copy, column)));
            store(copy, v.outerEntries, add(copy, target, ix(copy, column)), value);
            store(copy, v.innerEntries, add(copy, target, ix(copy, column)), value);
          }
          scf::YieldOp::create(copy, loc);
        });
      }
      scf::YieldOp::create(in, loc);
    });
  });
}
