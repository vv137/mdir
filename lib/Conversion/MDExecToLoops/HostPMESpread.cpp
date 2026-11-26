#include "HostPME.h"
#include "../MDExecKernels/HostChargeStencil.h"

using namespace mlir;
using namespace mdir::kernels;

// Zeroes the fixed-point grid.
void HostPME::emitZero() {
  OpBuilder::InsertionGuard guard(b);
  auto fn =
      declare("mdrt.pme_zero",
              b.getFunctionType(
                  {MemRefType::get({ShapedType::kDynamic}, i64Type)}, {}));
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  Value grid = entry->getArgument(0);
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  Value none = arith::ConstantOp::create(b, loc, b.getIntegerAttr(i64Type, 0));
  Value points = memref::DimOp::create(b, loc, grid, c0);
  scf::ParallelOp::create(b, loc, ValueRange{c0}, ValueRange{points},
                          ValueRange{c1},
                          [&](OpBuilder &b, Location, ValueRange ivs) {
                            Value p = ivs[0];
                            store(none, grid, {p});
                          });
  func::ReturnOp::create(b, loc);
}

// Adds the charge `qs` of a particle with the weights `wx`, `wy`, `wz`
// from the points `sx`, `sy`, `sz` on to the fixed-point grid, with
// integer atomics: the threads of the spreading add to the same points, and
// an integer sum does not depend on their order (D70).
void HostPME::emitSpreadPoint() {
  OpBuilder::InsertionGuard guard(b);
  auto fn = declare(
      "mdrt.pme_spread_point",
      b.getFunctionType(
          {MemRefType::get({ShapedType::kDynamic}, i64Type), realType, idxType,
           idxType, idxType, MemRefType::get({8}, realType),
           MemRefType::get({8}, realType), MemRefType::get({8}, realType),
           idxType, idxType, idxType, idxType},
          {}));
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  Value grid = entry->getArgument(0);
  Value qs = entry->getArgument(1);
  Value placex = entry->getArgument(2);
  Value placey = entry->getArgument(3);
  Value placez = entry->getArgument(4);
  Value wx = entry->getArgument(5);
  Value wy = entry->getArgument(6);
  Value wz = entry->getArgument(7);
  Value k1 = entry->getArgument(8);
  Value k2 = entry->getArgument(9);
  Value k3 = entry->getArgument(10);
  Value n = entry->getArgument(11);
  Value sx = arith::RemUIOp::create(b, loc, placex, k1);
  Value sy = arith::RemUIOp::create(b, loc, placey, k2);
  Value sz = arith::RemUIOp::create(b, loc, placez, k3);
  Value scale = arith::ConstantOp::create(
      b, loc, b.getFloatAttr(realType, 1099511627776.0));
  Value qscaled = arith::MulFOp::create(b, loc, qs, scale);
  Value weights[] = {wx, wy, wz};
  emitHostChargeStencil(b, loc, {sx, sy, sz}, {k1, k2, k3}, n, qscaled,
      [&](OpBuilder &at, unsigned axis, Value i) -> Value {
        return memref::LoadOp::create(at, loc, weights[axis], ValueRange{i});
      },
      [&](OpBuilder &at, Value i, Value value) {
        Value rounded = math::RoundEvenOp::create(at, loc, value);
        Value word = arith::FPToSIOp::create(at, loc, i64Type, rounded);
        memref::AtomicRMWOp::create(at, loc, arith::AtomicRMWKind::addi,
                                    word, grid, ValueRange{i});
      });
  func::ReturnOp::create(b, loc);
}

// Spreads the charge of particle `i` to `grid`, in fixed point.
void HostPME::emitSpreadOne() {
  OpBuilder::InsertionGuard guard(b);
  auto fn =
      declare("mdrt.pme_spread_one",
              b.getFunctionType(
                  {MemRefType::get({ShapedType::kDynamic, 3}, positionType),
                   MemRefType::get({ShapedType::kDynamic}, chargeType),
                   VectorType::get({3}, f64Type),
                   MemRefType::get({ShapedType::kDynamic}, i64Type), idxType,
                   idxType, idxType, idxType,
                   MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic},
                                   f64Type),
                   idxType},
                  {}));
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  Value x = entry->getArgument(0);
  Value q = entry->getArgument(1);
  Value box = entry->getArgument(2);
  Value grid = entry->getArgument(3);
  Value k1 = entry->getArgument(4);
  Value k2 = entry->getArgument(5);
  Value k3 = entry->getArgument(6);
  Value n = entry->getArgument(7);
  Value moduli = entry->getArgument(8);
  Value i = entry->getArgument(9);
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  Value c2 = arith::ConstantIndexOp::create(b, loc, 2);
  Value lx = vector::ExtractOp::create(b, loc, box, ArrayRef<int64_t>{0});
  Value ly = vector::ExtractOp::create(b, loc, box, ArrayRef<int64_t>{1});
  Value lz = vector::ExtractOp::create(b, loc, box, ArrayRef<int64_t>{2});
  Value wx = memref::AllocaOp::create(b, loc, MemRefType::get({8}, realType));
  Value wy = memref::AllocaOp::create(b, loc, MemRefType::get({8}, realType));
  Value wz = memref::AllocaOp::create(b, loc, MemRefType::get({8}, realType));
  Value dx = memref::AllocaOp::create(b, loc, MemRefType::get({8}, realType));
  Value dy = memref::AllocaOp::create(b, loc, MemRefType::get({8}, realType));
  Value dz = memref::AllocaOp::create(b, loc, MemRefType::get({8}, realType));
  Value xs = load(x, {i, c0});
  Value ys = load(x, {i, c1});
  Value zs = load(x, {i, c2});
  Value qs = load(q, {i});
  Value xi = convert(f64Type, xs);
  Value yi = convert(f64Type, ys);
  Value zi = convert(f64Type, zs);
  Value qi = convert(realType, qs);
  auto sxCall = func::CallOp::create(b, loc, "mdrt.pme_place",
                                     TypeRange{idxType, f64Type},
                                     ValueRange{xi, lx, k1, n, moduli});
  Value sx = sxCall.getResult(0);
  Value fx = sxCall.getResult(1);
  auto syCall = func::CallOp::create(b, loc, "mdrt.pme_place",
                                     TypeRange{idxType, f64Type},
                                     ValueRange{yi, ly, k2, n, moduli});
  Value sy = syCall.getResult(0);
  Value fy = syCall.getResult(1);
  auto szCall = func::CallOp::create(b, loc, "mdrt.pme_place",
                                     TypeRange{idxType, f64Type},
                                     ValueRange{zi, lz, k3, n, moduli});
  Value sz = szCall.getResult(0);
  Value fz = szCall.getResult(1);
  func::CallOp::create(b, loc, "mdrt.pme_bspline", TypeRange{},
                       ValueRange{fx, n, wx, dx, moduli});
  func::CallOp::create(b, loc, "mdrt.pme_bspline", TypeRange{},
                       ValueRange{fy, n, wy, dy, moduli});
  func::CallOp::create(b, loc, "mdrt.pme_bspline", TypeRange{},
                       ValueRange{fz, n, wz, dz, moduli});
  func::CallOp::create(
      b, loc, "mdrt.pme_spread_point", TypeRange{},
      ValueRange{grid, qi, sx, sy, sz, wx, wy, wz, k1, k2, k3, n});
  func::ReturnOp::create(b, loc);
}

// Spreads the charge of particle `i` to `grid` in a triclinic cell, from
// its fractional coordinates s = x H⁻¹.
void HostPME::emitSpreadOneTriclinic() {
  OpBuilder::InsertionGuard guard(b);
  auto fn =
      declare("mdrt.pme_spread_one_triclinic",
              b.getFunctionType(
                  {MemRefType::get({ShapedType::kDynamic, 3}, positionType),
                   MemRefType::get({ShapedType::kDynamic}, chargeType),
                   VectorType::get({6}, f64Type),
                   MemRefType::get({ShapedType::kDynamic}, i64Type), idxType,
                   idxType, idxType, idxType,
                   MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic},
                                   f64Type),
                   idxType},
                  {}));
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  Value x = entry->getArgument(0);
  Value q = entry->getArgument(1);
  Value box = entry->getArgument(2);
  Value grid = entry->getArgument(3);
  Value k1 = entry->getArgument(4);
  Value k2 = entry->getArgument(5);
  Value k3 = entry->getArgument(6);
  Value n = entry->getArgument(7);
  Value moduli = entry->getArgument(8);
  Value i = entry->getArgument(9);
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  Value c2 = arith::ConstantIndexOp::create(b, loc, 2);
  Value unit = arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 1.0));
  auto h00Call = func::CallOp::create(
      b, loc, "mdrt.pme_inverse_cell",
      TypeRange{f64Type, f64Type, f64Type, f64Type, f64Type, f64Type},
      ValueRange{box});
  Value h00 = h00Call.getResult(0);
  Value h10 = h00Call.getResult(1);
  Value h11 = h00Call.getResult(2);
  Value h20 = h00Call.getResult(3);
  Value h21 = h00Call.getResult(4);
  Value h22 = h00Call.getResult(5);
  Value wx = memref::AllocaOp::create(b, loc, MemRefType::get({8}, realType));
  Value wy = memref::AllocaOp::create(b, loc, MemRefType::get({8}, realType));
  Value wz = memref::AllocaOp::create(b, loc, MemRefType::get({8}, realType));
  Value dx = memref::AllocaOp::create(b, loc, MemRefType::get({8}, realType));
  Value dy = memref::AllocaOp::create(b, loc, MemRefType::get({8}, realType));
  Value dz = memref::AllocaOp::create(b, loc, MemRefType::get({8}, realType));
  Value xs = load(x, {i, c0});
  Value ys = load(x, {i, c1});
  Value zs = load(x, {i, c2});
  Value qs = load(q, {i});
  Value xi = convert(f64Type, xs);
  Value yi = convert(f64Type, ys);
  Value zi = convert(f64Type, zs);
  Value qi = convert(realType, qs);
  auto [s1, s2, s3] =
      fractionalPosition(xi, yi, zi, h00, h10, h11, h20, h21, h22);
  auto sxCall = func::CallOp::create(b, loc, "mdrt.pme_place",
                                     TypeRange{idxType, f64Type},
                                     ValueRange{s1, unit, k1, n, moduli});
  Value sx = sxCall.getResult(0);
  Value fx = sxCall.getResult(1);
  auto syCall = func::CallOp::create(b, loc, "mdrt.pme_place",
                                     TypeRange{idxType, f64Type},
                                     ValueRange{s2, unit, k2, n, moduli});
  Value sy = syCall.getResult(0);
  Value fy = syCall.getResult(1);
  auto szCall = func::CallOp::create(b, loc, "mdrt.pme_place",
                                     TypeRange{idxType, f64Type},
                                     ValueRange{s3, unit, k3, n, moduli});
  Value sz = szCall.getResult(0);
  Value fz = szCall.getResult(1);
  func::CallOp::create(b, loc, "mdrt.pme_bspline", TypeRange{},
                       ValueRange{fx, n, wx, dx, moduli});
  func::CallOp::create(b, loc, "mdrt.pme_bspline", TypeRange{},
                       ValueRange{fy, n, wy, dy, moduli});
  func::CallOp::create(b, loc, "mdrt.pme_bspline", TypeRange{},
                       ValueRange{fz, n, wz, dz, moduli});
  func::CallOp::create(
      b, loc, "mdrt.pme_spread_point", TypeRange{},
      ValueRange{grid, qi, sx, sy, sz, wx, wy, wz, k1, k2, k3, n});
  func::ReturnOp::create(b, loc);
}

// The grid in the grid precision, from fixed point.
void HostPME::emitReal() {
  OpBuilder::InsertionGuard guard(b);
  auto fn = declare(
      "mdrt.pme_real",
      b.getFunctionType({MemRefType::get({ShapedType::kDynamic}, i64Type),
                         MemRefType::get({ShapedType::kDynamic}, realType)},
                        {}));
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  Value grid = entry->getArgument(0);
  Value real = entry->getArgument(1);
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  Value inverse = arith::ConstantOp::create(
      b, loc, b.getFloatAttr(realType, 9.094947017729282e-13));
  Value points = memref::DimOp::create(b, loc, grid, c0);
  scf::ParallelOp::create(
      b, loc, ValueRange{c0}, ValueRange{points}, ValueRange{c1},
      [&](OpBuilder &b, Location, ValueRange ivs) {
        Value p = ivs[0];
        Value fixed = load(grid, {p});
        Value value = arith::SIToFPOp::create(b, loc, realType, fixed);
        Value scaled = arith::MulFOp::create(b, loc, value, inverse);
        store(scaled, real, {p});
      });
  func::ReturnOp::create(b, loc);
}

// The number of the listed particles of `list` (D262):
// its entries before the first negative one.
void HostPME::emitCountListed() {
  OpBuilder::InsertionGuard guard(b);
  auto fn = declare(
      "mdrt.pme_count_listed",
      b.getFunctionType({MemRefType::get({ShapedType::kDynamic}, i32Type)},
                        {idxType}));
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  Value list = entry->getArgument(0);
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  Value none = arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 0));
  Value size = memref::DimOp::create(b, loc, list, c0);
  auto countOp = scf::ForOp::create(
      b, loc, c0, size, c1, ValueRange{c0},
      [&](OpBuilder &b, Location, Value k, ValueRange carried) {
        Value n = carried[0];
        Value entry = load(list, {k});
        Value listed = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge,
                                             entry, none);
        Value more = arith::AddIOp::create(b, loc, n, c1);
        Value next = arith::SelectOp::create(b, loc, listed, more, n);
        scf::YieldOp::create(b, loc, ValueRange{next});
      });
  Value count = countOp.getResult(0);
  func::ReturnOp::create(b, loc, ValueRange{count});
}
