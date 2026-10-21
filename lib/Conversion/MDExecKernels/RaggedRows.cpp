// Ragged rows (D[ragged-incidence]); see include/mdir/Conversion/RaggedRows.h.

#include "mdir/Conversion/RaggedRows.h"
#include "mdir/Conversion/CompactionEmitter.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;

static Value constantIndex(OpBuilder &b, Location loc, int64_t value) {
  return arith::ConstantIndexOp::create(b, loc, value);
}

static Value loadAsIndex(OpBuilder &b, Location loc, Value buffer,
                         Value at) {
  Value narrow = memref::LoadOp::create(b, loc, buffer, ValueRange{at});
  return arith::IndexCastOp::create(b, loc, b.getIndexType(), narrow);
}

Value kernels::emitRaggedSize(OpBuilder &b, Location loc, Value rows,
                              Value entries, int64_t entrySize) {
  Value one = constantIndex(b, loc, 1);
  Value withZeros = arith::AddIOp::create(b, loc, entries, one);
  Value values = arith::MulIOp::create(b, loc, withZeros,
                                       constantIndex(b, loc, entrySize));
  Value header =
      arith::AddIOp::create(b, loc, rows, constantIndex(b, loc, 2));
  return arith::AddIOp::create(b, loc, header, values);
}

/// The row of a buffer that has it.
static RaggedRow loadRow(OpBuilder &b, Location loc, Value buffer, Value row,
                         Value entrySize) {
  Value one = constantIndex(b, loc, 1);
  Value two = constantIndex(b, loc, 2);
  Value rows = loadAsIndex(b, loc, buffer, constantIndex(b, loc, 0));
  Value at = arith::AddIOp::create(b, loc, row, one);
  Value begin = loadAsIndex(b, loc, buffer, at);
  Value end =
      loadAsIndex(b, loc, buffer, arith::AddIOp::create(b, loc, at, one));
  Value header = arith::AddIOp::create(b, loc, rows, two);
  Value first = arith::AddIOp::create(
      b, loc, header, arith::MulIOp::create(b, loc, begin, entrySize));
  return {first, arith::SubIOp::create(b, loc, end, begin)};
}

RaggedRow kernels::emitRaggedRow(OpBuilder &b, Location loc, Value buffer,
                                 Value row, Value entrySize, bool checked) {
  if (!checked)
    return loadRow(b, loc, buffer, row, entrySize);
  Type index = b.getIndexType();
  SmallVector<Type, 2> types = {index, index};
  Value zero = constantIndex(b, loc, 0);
  Value size = memref::DimOp::create(b, loc, buffer, zero);
  Value some =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne, size, zero);
  auto none = [&](scf::IfOp branch) {
    OpBuilder otherwise = OpBuilder::atBlockEnd(branch.elseBlock());
    Value nothing = constantIndex(otherwise, loc, 0);
    scf::YieldOp::create(otherwise, loc, ValueRange{nothing, nothing});
  };
  auto branch =
      scf::IfOp::create(b, loc, types, some, /*withElseRegion=*/true);
  {
    OpBuilder t = OpBuilder::atBlockEnd(branch.thenBlock());
    Value rows = loadAsIndex(t, loc, buffer, constantIndex(t, loc, 0));
    Value inside =
        arith::CmpIOp::create(t, loc, arith::CmpIPredicate::ult, row, rows);
    auto within =
        scf::IfOp::create(t, loc, types, inside, /*withElseRegion=*/true);
    OpBuilder u = OpBuilder::atBlockEnd(within.thenBlock());
    RaggedRow found = loadRow(u, loc, buffer, row, entrySize);
    scf::YieldOp::create(u, loc, ValueRange{found.first, found.count});
    none(within);
    scf::YieldOp::create(t, loc, within.getResults());
  }
  none(branch);
  return {branch.getResult(0), branch.getResult(1)};
}

RaggedRow kernels::emitRaggedRow(OpBuilder &b, Location loc, Value buffer,
                                 Value row, int64_t entrySize, bool checked) {
  return emitRaggedRow(b, loc, buffer, row, constantIndex(b, loc, entrySize),
                       checked);
}

Value kernels::emitRaggedEntry(OpBuilder &b, Location loc, Value buffer,
                               Value first, Value k, Value column,
                               Value entrySize) {
  Value entry = arith::AddIOp::create(
      b, loc, first, arith::MulIOp::create(b, loc, k, entrySize));
  Value at = arith::AddIOp::create(b, loc, entry, column);
  return memref::LoadOp::create(b, loc, buffer, ValueRange{at});
}

Value kernels::emitRaggedEntry(OpBuilder &b, Location loc, Value buffer,
                               Value first, Value k, int64_t column,
                               int64_t entrySize) {
  return emitRaggedEntry(b, loc, buffer, first, k,
                         constantIndex(b, loc, column),
                         constantIndex(b, loc, entrySize));
}

/// A launch of a thread for each of `count` items, in blocks of `block`.
static void launchEach(OpBuilder &b, Location loc, Value count, Value block,
                       Value one,
                       function_ref<void(OpBuilder &, Value)> body) {
  Value last = arith::SubIOp::create(b, loc, block, one);
  Value grid = arith::DivUIOp::create(
      b, loc, arith::AddIOp::create(b, loc, count, last), block);
  // A grid of no blocks is not a launch.
  grid = arith::MaxUIOp::create(b, loc, grid, one);
  auto op = gpu::LaunchOp::create(b, loc, grid, one, one, block, one, one);
  OpBuilder kernel = OpBuilder::atBlockEnd(&op.getBody().front());
  Value base = arith::MulIOp::create(kernel, loc, op.getBlockIds().x, block);
  Value item = arith::AddIOp::create(kernel, loc, base, op.getThreadIds().x);
  Value inside = arith::CmpIOp::create(kernel, loc, arith::CmpIPredicate::ult,
                                       item, count);
  scf::IfOp::create(kernel, loc, inside, [&](OpBuilder &then, Location) {
    body(then, item);
    scf::YieldOp::create(then, loc);
  });
  gpu::TerminatorOp::create(kernel, loc);
}

void kernels::emitDeviceRaggedRows(OpBuilder &b, Location loc, Value keys,
                                   Value m, Value n, Value counts,
                                   Value start, Value cursor, Value items,
                                   Value sums, Value blocks, Value block,
                                   Value one) {
  Type i32 = b.getI32Type();
  // The counts, from zero.
  launchEach(b, loc, n, block, one, [&](OpBuilder &t, Value row) {
    Value zero = arith::ConstantOp::create(t, loc, i32, t.getI32IntegerAttr(0));
    memref::StoreOp::create(t, loc, zero, counts, ValueRange{row});
  });
  // A count for each item, in its row.
  launchEach(b, loc, m, block, one, [&](OpBuilder &t, Value item) {
    Value step = arith::ConstantOp::create(t, loc, i32, t.getI32IntegerAttr(1));
    Value key = loadAsIndex(t, loc, keys, item);
    emitDeviceAtomicAdd(t, loc, counts, key, step);
  });
  // The offsets of the rows, and the cursors at their beginnings.
  emitDeviceScan(b, loc, counts, start, cursor, n, sums, blocks, block, one);
  // Each item at the next slot of its row, then each row in order.
  Value grid = arith::MaxUIOp::create(
      b, loc,
      arith::DivUIOp::create(
          b, loc,
          arith::AddIOp::create(b, loc, m,
                                arith::SubIOp::create(b, loc, block, one)),
          block),
      one);
  emitCursorScatter(b, loc, keys, cursor, items, m, grid, block, one,
                    /*everyKey=*/true);
  Value rowGrid = arith::MaxUIOp::create(
      b, loc,
      arith::DivUIOp::create(
          b, loc,
          arith::AddIOp::create(b, loc, n,
                                arith::SubIOp::create(b, loc, block, one)),
          block),
      one);
  emitSortBins(b, loc, start, items, n, rowGrid, block, one);
}
