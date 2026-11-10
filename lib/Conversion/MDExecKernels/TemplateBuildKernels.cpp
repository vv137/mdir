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
  launch(rowsOrClasses, [&](OpBuilder &at, Value row) {
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
  });
  // A thread per staging chunk: copying at most 64 records is bounded.
  // The search already compacted the members of each chunk with a ballot.
  launch(v.unitCapacity, [&](OpBuilder &at, Value chunk) {
    Value used = index(at, load(at, v.arena, ix(at, TEMPLATE_PAIR_STAGE_COUNT)));
    Value exists = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ult, chunk, used);
    scf::IfOp::create(at, loc, exists, [&](OpBuilder &in, Location) {
      Value base = mul(in, chunk, 4);
      Value row = index(in, load(in, v.stageUnits, base));
      Value count = index(in, load(in, v.stageUnits, add(in, base, ix(in, 1))));
      Value ordinal = index(in, load(in, v.stageUnits, add(in, base, ix(in, 2))));
      Value destination = add(in, index(in, load(in, v.entryOffsets, row)), mul(in, ordinal, 64));
      Value work = add(in, index(in, load(in, v.workOffsets, row)), ordinal);
      Value fields[] = {word(in, classify(in, row)), word(in, firstPlace(in, row)),
                       word(in, destination), word(in, add(in, destination, count))};
      for (unsigned column = 0; column != 4; ++column) {
        recordStore(in, v.outerUnits, work, column, fields[column]);
        recordStore(in, v.innerUnits, work, column, fields[column]);
      }
      scf::ForOp::create(in, loc, ix(in, 0), count, ix(in, 1), ValueRange(),
          [&](OpBuilder &copy, Location, Value i, ValueRange) {
            Value source = mul(copy, add(copy, mul(copy, chunk, 64), i), 4);
            Value target = mul(copy, add(copy, destination, i), 4);
            for (unsigned column = 0; column != 4; ++column) {
              Value value = load(copy, v.stageEntries, add(copy, source, ix(copy, column)));
              store(copy, v.outerEntries, add(copy, target, ix(copy, column)), value);
              store(copy, v.innerEntries, add(copy, target, ix(copy, column)), value);
            }
            scf::YieldOp::create(copy, loc);
          });
      scf::YieldOp::create(in, loc);
    });
  });
}
