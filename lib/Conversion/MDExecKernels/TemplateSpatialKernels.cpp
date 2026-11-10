// The column/z ordering of D89, shared by both kinds through a class key.
#include "mdir/Conversion/TemplateSpatialKernels.h"
#include "mdir/Conversion/CompactionEmitter.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mdir/Conversion/RaggedRows.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace mlir;
using namespace mdir::kernels;

std::array<Value, 2> mdir::kernels::emitTemplateColumnGrid(
    OpBuilder &b, Location loc, Value cell, Value units, Value bins) {
  auto ix = [&](int64_t n) { return createIndex(b, loc, n); };
  auto real = [&](double n) {
    return createReal(b, loc, b.getF64Type(), n);
  };
  // The prototype's (16/rho)^(1/3) scale, over the units in the shared
  // class-prefixed grid. Orthorhombic volume is the diagonal product too.
  Value volume = real(1);
  for (int64_t axis = 0; axis != 3; ++axis)
    volume = arith::MulFOp::create(b, loc, volume,
        vector::ExtractOp::create(b, loc, cell, axis));
  Value population = arith::UIToFPOp::create(b, loc, b.getF64Type(),
      arith::IndexCastOp::create(b, loc, b.getI64Type(),
          arith::MaxUIOp::create(b, loc, units, ix(1))));
  Value edge = math::CbrtOp::create(b, loc,
      arith::DivFOp::create(b, loc,
          arith::MulFOp::create(b, loc, real(16), volume), population));
  Value widths = isTriclinic(cell) ? emitFaceWidths(b, loc, cell) : cell;
  Value budget = arith::DivUIOp::create(b, loc, bins, ix(2));
  auto dimension = [&](int64_t axis, Value limit) -> Value {
    Value rounded = math::FloorOp::create(b, loc,
        arith::AddFOp::create(b, loc,
            arith::DivFOp::create(b, loc,
                vector::ExtractOp::create(b, loc, widths, axis), edge), real(0.5)));
    Value bound = arith::UIToFPOp::create(b, loc, b.getF64Type(),
        arith::IndexCastOp::create(b, loc, b.getI64Type(), limit));
    // Clamp before integer conversion, including highly anisotropic cells.
    Value safe = arith::MinimumFOp::create(b, loc, bound,
        arith::MaximumFOp::create(b, loc, real(1), rounded));
    return arith::IndexCastOp::create(b, loc, b.getIndexType(),
        arith::FPToUIOp::create(b, loc, b.getI64Type(), safe));
  };
  Value x = dimension(0, budget);
  Value y = dimension(1, arith::DivUIOp::create(b, loc, budget, x));
  return {x, y};
}

void mdir::kernels::emitTemplateSpatialOrder(OpBuilder &b, Location loc,
                                             const TemplateSpatialBuffers &v) {
  auto ix = [&](OpBuilder &at, int64_t n) { return createIndex(at, loc, n); };
  auto add = [&](OpBuilder &at, Value a, Value c) -> Value {
    return arith::AddIOp::create(at, loc, a, c);
  };
  auto sub = [&](OpBuilder &at, Value a, Value c) -> Value {
    return arith::SubIOp::create(at, loc, a, c);
  };
  auto mul = [&](OpBuilder &at, Value a, Value c) -> Value {
    return arith::MulIOp::create(at, loc, a, c);
  };
  auto index = [&](OpBuilder &at, Value word) -> Value {
    return arith::IndexCastUIOp::create(at, loc, at.getIndexType(), word);
  };
  auto word = [&](OpBuilder &at, Value n) -> Value {
    return arith::IndexCastOp::create(at, loc, at.getI32Type(), n);
  };
  Value one = ix(b, 1), block = ix(b, 256);
  Value size = add(b, v.waters, v.atoms);
  Value columns = mul(b, v.columnsX, v.columnsY);
  Value bins = mul(b, columns, ix(b, 2));
  auto gridFor = [&](Value n) -> Value {
    Value grid = arith::DivUIOp::create(b, loc, add(b, n, ix(b, 255)), block);
    return arith::MaxUIOp::create(b, loc, grid, one);
  };
  Value itemGrid = gridFor(size), binGrid = gridFor(bins);
  auto launch = [&](Value n, Value grid, auto body) {
    auto kernel = gpu::LaunchOp::create(b, loc, grid, one, one, block, one, one);
    OpBuilder at = OpBuilder::atBlockEnd(&kernel.getBody().front());
    Value item = add(at, mul(at, kernel.getBlockIds().x, block), kernel.getThreadIds().x);
    Value inside = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ult, item, n);
    scf::IfOp::create(at, loc, inside, [&](OpBuilder &in, Location) {
      body(in, item);
      scf::YieldOp::create(in, loc);
    });
    gpu::TerminatorOp::create(at, loc);
  };
  launch(size, itemGrid, [&](OpBuilder &at, Value unit) {
    Value original = loadElement(at, loc, v.points, unit);
    SmallVector<Value, 3> wrapped;
    for (unsigned axis = 0; axis != 3; ++axis) {
      Value component = vector::ExtractOp::create(at, loc, original, axis);
      wrapped.push_back(arith::SubFOp::create(at, loc, component,
          math::FloorOp::create(at, loc, component)));
    }
    Value point = convertReal(at, loc,
        vector::FromElementsOp::create(at, loc, original.getType(), wrapped), at.getF32Type());
    auto bin = [&](unsigned axis, Value count) -> Value {
      Value component = vector::ExtractOp::create(at, loc, point, axis);
      Value scale = arith::UIToFPOp::create(at, loc, at.getF32Type(), word(at, count));
      Value scaled = arith::MulFOp::create(at, loc, component, scale);
      Value ordinal = index(at, arith::FPToUIOp::create(at, loc, at.getI32Type(), scaled));
      return arith::MinUIOp::create(at, loc, ordinal, sub(at, count, ix(at, 1)));
    };
    Value x = bin(0, v.columnsX), y = bin(1, v.columnsY);
    Value odd = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ne,
        arith::AndIOp::create(at, loc, x, ix(at, 1)), ix(at, 0));
    Value reversed = sub(at, sub(at, v.columnsY, ix(at, 1)), y);
    Value serpentine = arith::SelectOp::create(at, loc, odd, reversed, y);
    Value column = add(at, mul(at, x, v.columnsY), serpentine);
    Value water = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ult, unit, v.waters);
    Value key = arith::SelectOp::create(at, loc, water, column, add(at, columns, column));
    memref::StoreOp::create(at, loc, word(at, key), v.keys, ValueRange{unit});
    memref::StoreOp::create(at, loc, word(at, bin(2, ix(at, 1 << 21))),
                            v.zKeys, ValueRange{unit});
  });
  emitDeviceRaggedRows(b, loc, v.keys, size, bins, v.counts, v.starts,
      v.cursors, v.items, v.scanScratch, v.scanBlocks, block, one);
  // Ragged rows first order equal-column items by fixed ID. The stable
  // z-key insertion keeps that tie order after the second key stage.
  emitSortBins(b, loc, v.starts, v.items, bins, binGrid, block, one, v.zKeys);
  Value waterPlaces = mul(b, arith::DivUIOp::create(b, loc,
      add(b, v.waters, ix(b, 15)), ix(b, 16)), ix(b, 16));
  Value atomPlaces = mul(b, arith::DivUIOp::create(b, loc,
      add(b, v.atoms, ix(b, 15)), ix(b, 16)), ix(b, 16));
  Value padded = add(b, waterPlaces, atomPlaces);
  launch(padded, gridFor(padded), [&](OpBuilder &at, Value place) {
    Value water = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ult, place, waterPlaces);
    Value active = arith::SelectOp::create(at, loc, water, v.waterActive, v.atomActive);
    scf::IfOp::create(at, loc, active, [&](OpBuilder &in, Location) {
      Value none = arith::ConstantIntOp::create(in, loc, -1, 32);
      memref::StoreOp::create(in, loc, none, v.order, ValueRange{place});
      scf::YieldOp::create(in, loc);
    });
  });
  launch(size, itemGrid, [&](OpBuilder &at, Value rank) {
    Value unit = memref::LoadOp::create(at, loc, v.items, ValueRange{rank});
    Value water = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ult, rank, v.waters);
    Value active = arith::SelectOp::create(at, loc, water, v.waterActive, v.atomActive);
    Value place = arith::SelectOp::create(at, loc, water, rank,
        add(at, waterPlaces, sub(at, rank, v.waters)));
    scf::IfOp::create(at, loc, active, [&](OpBuilder &in, Location) {
      memref::StoreOp::create(in, loc, unit, v.order, ValueRange{place});
      memref::StoreOp::create(in, loc, word(in, place), v.places, ValueRange{index(in, unit)});
      scf::YieldOp::create(in, loc);
    });
  });
}
