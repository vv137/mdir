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
  // pow is already lowered by the native host pipeline; introducing a
  // new math.cbrt here would require changing that shared pipeline.
  Value edge = math::PowFOp::create(b, loc,
      arith::DivFOp::create(b, loc,
          arith::MulFOp::create(b, loc, real(16), volume), population),
      real(1.0 / 3.0));
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
  // One stable LSD counting sort on (class-column,z), starting with fixed
  // IDs. Each digit scatters *input positions*, then the shared rank restores
  // their order in the bucket. Stability therefore preserves all lower
  // digits, including the initial fixed-ID tie order. No site-specific sort
  // kernel is introduced (docs/device-primitives.md, Section 4).
  auto scratch = [&](int64_t part) -> Value {
    return memref::SubViewOp::create(b, loc, v.sortScratch,
        SmallVector<OpFoldResult>{mul(b, size, ix(b, part))},
        SmallVector<OpFoldResult>{size},
        SmallVector<OpFoldResult>{b.getIndexAttr(1)});
  };
  Value input = scratch(0), output = scratch(1);
  Value digitKeys = scratch(2), ranked = scratch(3);
  launch(size, itemGrid, [&](OpBuilder &at, Value unit) {
    memref::StoreOp::create(at, loc, word(at, unit), input, ValueRange{unit});
  });
  // Use at most 256 buckets, bounded by the existing class-column storage.
  // Balance the significant joint-key bits over digits so that a final
  // one-bit digit does not put half of a large population in one rank bin.
  Value columnBits = ix(b, 0), radixBits = ix(b, 1);
  Value lastBin = sub(b, bins, one);
  for (int64_t bit = 0; bit != 31; ++bit) {
    Value present = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::uge,
        lastBin, ix(b, int64_t(1) << bit));
    columnBits = arith::SelectOp::create(b, loc, present, ix(b, bit + 1), columnBits);
    if (bit >= 1 && bit <= 8) {
      Value fits = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::uge,
          bins, ix(b, int64_t(1) << bit));
      radixBits = arith::SelectOp::create(b, loc, fits, ix(b, bit), radixBits);
    }
  }
  Value bits = add(b, columnBits, ix(b, 21));
  Value passes = arith::DivUIOp::create(b, loc,
      add(b, bits, sub(b, radixBits, one)), radixBits);
  Value shortWidth = arith::DivUIOp::create(b, loc, bits, passes);
  Value wider = arith::RemUIOp::create(b, loc, bits, passes);
  auto sorted = scf::ForOp::create(b, loc, ix(b, 0), passes, one,
      ValueRange{input, output, ix(b, 0)},
      [&](OpBuilder &at, Location, Value pass, ValueRange state) {
    Value extra = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ult, pass, wider);
    Value width = add(at, shortWidth,
        arith::SelectOp::create(at, loc, extra, one, ix(at, 0)));
    Value buckets = arith::ShLIOp::create(at, loc, one, width);
    Value mask = sub(at, buckets, one);
    // The launch helper must insert each digit's kernels in this loop.
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToEnd(at.getInsertionBlock());
    launch(buckets, gridFor(buckets), [&](OpBuilder &in, Value bucket) {
      memref::StoreOp::create(in, loc, word(in, ix(in, 0)), v.counts, ValueRange{bucket});
    });
    launch(size, itemGrid, [&](OpBuilder &in, Value position) {
      Value unit = index(in, memref::LoadOp::create(in, loc, state[0], ValueRange{position}));
      Value column = index(in, memref::LoadOp::create(in, loc, v.keys, ValueRange{unit}));
      Value z = index(in, memref::LoadOp::create(in, loc, v.zKeys, ValueRange{unit}));
      Value joint = arith::OrIOp::create(in, loc,
          arith::ShLIOp::create(in, loc, column, ix(in, 21)), z);
      Value digit = arith::AndIOp::create(in, loc,
          arith::ShRUIOp::create(in, loc, joint, state[2]), mask);
      memref::StoreOp::create(in, loc, word(in, digit), digitKeys, ValueRange{position});
      emitDeviceAtomicAdd(in, loc, v.counts, digit, word(in, one));
    });
    emitDeviceScan(b, loc, v.counts, v.starts, v.cursors, buckets,
        v.scanScratch, one, block, one);
    emitCursorScatter(b, loc, digitKeys, v.cursors, v.items, size,
        itemGrid, block, one, /*everyKey=*/true);
    emitRankInBins(b, loc, v.items, digitKeys, v.starts, ranked,
        buckets, itemGrid, block, one);
    launch(size, itemGrid, [&](OpBuilder &in, Value rank) {
      Value position = index(in,
          memref::LoadOp::create(in, loc, ranked, ValueRange{rank}));
      Value unit = memref::LoadOp::create(in, loc, state[0], ValueRange{position});
      memref::StoreOp::create(in, loc, unit, state[1], ValueRange{rank});
    });
    at.setInsertionPointToEnd(at.getInsertionBlock());
    scf::YieldOp::create(at, loc, ValueRange{state[1], state[0], add(at, state[2], width)});
  });
  launch(size, itemGrid, [&](OpBuilder &at, Value rank) {
    Value unit = memref::LoadOp::create(at, loc, sorted.getResult(0), ValueRange{rank});
    memref::StoreOp::create(at, loc, unit, v.items, ValueRange{rank});
  });
  // Publish the original class-column offsets required by the unchanged
  // covering search; the digit histograms above are only temporary.
  launch(bins, binGrid, [&](OpBuilder &at, Value bin) {
    memref::StoreOp::create(at, loc, word(at, ix(at, 0)), v.counts, ValueRange{bin});
  });
  launch(size, itemGrid, [&](OpBuilder &at, Value unit) {
    Value key = index(at, memref::LoadOp::create(at, loc, v.keys, ValueRange{unit}));
    emitDeviceAtomicAdd(at, loc, v.counts, key, word(at, one));
  });
  emitDeviceScan(b, loc, v.counts, v.starts, v.cursors, bins,
      v.scanScratch, v.scanBlocks, block, one);
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
