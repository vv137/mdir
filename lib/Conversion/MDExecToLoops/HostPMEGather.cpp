#include "HostPME.h"

using namespace mlir;
using namespace mdir::kernels;

// The sums Σ_k φ(k) ∂θ_i(k)/∂u_a of particle `i` along the three edges,
// from its first points and its weights and slopes, in the grid precision.
void HostPME::emitGatherSums() {
  OpBuilder::InsertionGuard guard(b);
  auto fn = declare(
      "mdrt.pme_gather_sums",
      b.getFunctionType(
          {MemRefType::get({ShapedType::kDynamic}, realType), idxType, idxType,
           idxType, MemRefType::get({8}, realType),
           MemRefType::get({8}, realType), MemRefType::get({8}, realType),
           MemRefType::get({8}, realType), MemRefType::get({8}, realType),
           MemRefType::get({8}, realType), idxType, idxType, idxType, idxType},
          {f64Type, f64Type, f64Type}));
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  Value phi = entry->getArgument(0);
  Value placex = entry->getArgument(1);
  Value placey = entry->getArgument(2);
  Value placez = entry->getArgument(3);
  Value wx = entry->getArgument(4);
  Value wy = entry->getArgument(5);
  Value wz = entry->getArgument(6);
  Value dx = entry->getArgument(7);
  Value dy = entry->getArgument(8);
  Value dz = entry->getArgument(9);
  Value k1 = entry->getArgument(10);
  Value k2 = entry->getArgument(11);
  Value k3 = entry->getArgument(12);
  Value n = entry->getArgument(13);
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  Value sx = arith::RemUIOp::create(b, loc, placex, k1);
  Value sy = arith::RemUIOp::create(b, loc, placey, k2);
  Value sz = arith::RemUIOp::create(b, loc, placez, k3);
  Value zero = arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.0));
  auto gxOp = scf::ForOp::create(
      b, loc, c0, n, c1, ValueRange{zero, zero, zero},
      [&](OpBuilder &b, Location, Value j1, ValueRange carried) {
        Value ax = carried[0];
        Value ay = carried[1];
        Value az = carried[2];
        Value g1s = arith::AddIOp::create(b, loc, sx, j1);
        Value g1w = arith::SubIOp::create(b, loc, g1s, k1);
        Value g1o =
            arith::CmpIOp::create(b, loc, arith::CmpIPredicate::uge, g1s, k1);
        Value g1 = arith::SelectOp::create(b, loc, g1o, g1w, g1s);
        Value w1 = load(wx, {j1});
        Value d1 = load(dx, {j1});
        auto bxOp = scf::ForOp::create(
            b, loc, c0, n, c1, ValueRange{ax, ay, az},
            [&](OpBuilder &b, Location, Value j2, ValueRange carried) {
              Value cx = carried[0];
              Value cy = carried[1];
              Value cz = carried[2];
              Value g2s = arith::AddIOp::create(b, loc, sy, j2);
              Value g2w = arith::SubIOp::create(b, loc, g2s, k2);
              Value g2o = arith::CmpIOp::create(
                  b, loc, arith::CmpIPredicate::uge, g2s, k2);
              Value g2 = arith::SelectOp::create(b, loc, g2o, g2w, g2s);
              Value w2 = load(wy, {j2});
              Value d2 = load(dy, {j2});
              Value row1 = arith::MulIOp::create(b, loc, g1, k2);
              Value row = arith::AddIOp::create(b, loc, row1, g2);
              Value base = arith::MulIOp::create(b, loc, row, k3);
              Value d1w2 = arith::MulFOp::create(b, loc, d1, w2);
              Value w1d2 = arith::MulFOp::create(b, loc, w1, d2);
              Value w1w2 = arith::MulFOp::create(b, loc, w1, w2);
              auto exOp = scf::ForOp::create(
                  b, loc, c0, n, c1, ValueRange{cx, cy, cz},
                  [&](OpBuilder &b, Location, Value j3, ValueRange carried) {
                    Value tx = carried[0];
                    Value ty = carried[1];
                    Value tz = carried[2];
                    Value g3s = arith::AddIOp::create(b, loc, sz, j3);
                    Value g3w = arith::SubIOp::create(b, loc, g3s, k3);
                    Value g3o = arith::CmpIOp::create(
                        b, loc, arith::CmpIPredicate::uge, g3s, k3);
                    Value g3 = arith::SelectOp::create(b, loc, g3o, g3w, g3s);
                    Value w3 = load(wz, {j3});
                    Value d3 = load(dz, {j3});
                    Value at = arith::AddIOp::create(b, loc, base, g3);
                    Value p = load(phi, {at});
                    Value px = arith::MulFOp::create(b, loc, d1w2, w3);
                    Value py = arith::MulFOp::create(b, loc, w1d2, w3);
                    Value pz = arith::MulFOp::create(b, loc, w1w2, d3);
                    Value vx = arith::MulFOp::create(b, loc, p, px);
                    Value vy = arith::MulFOp::create(b, loc, p, py);
                    Value vz = arith::MulFOp::create(b, loc, p, pz);
                    Value nx = arith::AddFOp::create(b, loc, tx, vx);
                    Value ny = arith::AddFOp::create(b, loc, ty, vy);
                    Value nz = arith::AddFOp::create(b, loc, tz, vz);
                    scf::YieldOp::create(b, loc, ValueRange{nx, ny, nz});
                  });
              Value ex = exOp.getResult(0);
              Value ey = exOp.getResult(1);
              Value ez = exOp.getResult(2);
              scf::YieldOp::create(b, loc, ValueRange{ex, ey, ez});
            });
        Value bx = bxOp.getResult(0);
        Value by = bxOp.getResult(1);
        Value bz = bxOp.getResult(2);
        scf::YieldOp::create(b, loc, ValueRange{bx, by, bz});
      });
  Value gx = gxOp.getResult(0);
  Value gy = gxOp.getResult(1);
  Value gz = gxOp.getResult(2);
  Value gx64 = convert(f64Type, gx);
  Value gy64 = convert(f64Type, gy);
  Value gz64 = convert(f64Type, gz);
  func::ReturnOp::create(b, loc, ValueRange{gx64, gy64, gz64});
}

// The sum Σ_k φ(k) θ_i(k) of particle `i`, in the grid precision.
void HostPME::emitPotentialSum() {
  OpBuilder::InsertionGuard guard(b);
  auto fn = declare(
      "mdrt.pme_potential_sum",
      b.getFunctionType(
          {MemRefType::get({ShapedType::kDynamic}, realType), idxType, idxType,
           idxType, MemRefType::get({8}, realType),
           MemRefType::get({8}, realType), MemRefType::get({8}, realType),
           idxType, idxType, idxType, idxType},
          {f64Type}));
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  Value phi = entry->getArgument(0);
  Value placex = entry->getArgument(1);
  Value placey = entry->getArgument(2);
  Value placez = entry->getArgument(3);
  Value wx = entry->getArgument(4);
  Value wy = entry->getArgument(5);
  Value wz = entry->getArgument(6);
  Value k1 = entry->getArgument(7);
  Value k2 = entry->getArgument(8);
  Value k3 = entry->getArgument(9);
  Value n = entry->getArgument(10);
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  Value sx = arith::RemUIOp::create(b, loc, placex, k1);
  Value sy = arith::RemUIOp::create(b, loc, placey, k2);
  Value sz = arith::RemUIOp::create(b, loc, placez, k3);
  Value zero = arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.0));
  auto gOp = scf::ForOp::create(
      b, loc, c0, n, c1, ValueRange{zero},
      [&](OpBuilder &b, Location, Value j1, ValueRange carried) {
        Value a = carried[0];
        Value g1s = arith::AddIOp::create(b, loc, sx, j1);
        Value g1w = arith::SubIOp::create(b, loc, g1s, k1);
        Value g1o =
            arith::CmpIOp::create(b, loc, arith::CmpIPredicate::uge, g1s, k1);
        Value g1 = arith::SelectOp::create(b, loc, g1o, g1w, g1s);
        Value w1 = load(wx, {j1});
        auto accBOp = scf::ForOp::create(
            b, loc, c0, n, c1, ValueRange{a},
            [&](OpBuilder &b, Location, Value j2, ValueRange carried) {
              Value c = carried[0];
              Value g2s = arith::AddIOp::create(b, loc, sy, j2);
              Value g2w = arith::SubIOp::create(b, loc, g2s, k2);
              Value g2o = arith::CmpIOp::create(
                  b, loc, arith::CmpIPredicate::uge, g2s, k2);
              Value g2 = arith::SelectOp::create(b, loc, g2o, g2w, g2s);
              Value w2 = load(wy, {j2});
              Value row1 = arith::MulIOp::create(b, loc, g1, k2);
              Value row = arith::AddIOp::create(b, loc, row1, g2);
              Value base = arith::MulIOp::create(b, loc, row, k3);
              Value w1w2 = arith::MulFOp::create(b, loc, w1, w2);
              auto eOp = scf::ForOp::create(
                  b, loc, c0, n, c1, ValueRange{c},
                  [&](OpBuilder &b, Location, Value j3, ValueRange carried) {
                    Value t = carried[0];
                    Value g3s = arith::AddIOp::create(b, loc, sz, j3);
                    Value g3w = arith::SubIOp::create(b, loc, g3s, k3);
                    Value g3o = arith::CmpIOp::create(
                        b, loc, arith::CmpIPredicate::uge, g3s, k3);
                    Value g3 = arith::SelectOp::create(b, loc, g3o, g3w, g3s);
                    Value w3 = load(wz, {j3});
                    Value at = arith::AddIOp::create(b, loc, base, g3);
                    Value p = load(phi, {at});
                    Value w = arith::MulFOp::create(b, loc, w1w2, w3);
                    Value v = arith::MulFOp::create(b, loc, p, w);
                    Value next = arith::AddFOp::create(b, loc, t, v);
                    scf::YieldOp::create(b, loc, ValueRange{next});
                  });
              Value e = eOp.getResult(0);
              scf::YieldOp::create(b, loc, ValueRange{e});
            });
        Value accB = accBOp.getResult(0);
        scf::YieldOp::create(b, loc, ValueRange{accB});
      });
  Value g = gOp.getResult(0);
  Value g64 = convert(f64Type, g);
  func::ReturnOp::create(b, loc, ValueRange{g64});
}

// The force on particle `i`, F_i = −q_i Σ_k φ(k) ∇_i θ_i(k), from the
// potential `phi` on the grid, which is the backward transform of the
// product of the transform of the charges with the influence function. A
// function of its own, so that its arrays live on the stack of a call.
void HostPME::emitGatherOne() {
  OpBuilder::InsertionGuard guard(b);
  auto fn = declare(
      "mdrt.pme_gather_one",
      b.getFunctionType(
          {MemRefType::get({ShapedType::kDynamic, 3}, positionType),
           MemRefType::get({ShapedType::kDynamic}, chargeType),
           MemRefType::get({ShapedType::kDynamic}, realType),
           VectorType::get({3}, f64Type), idxType, idxType, idxType, idxType,
           MemRefType::get({ShapedType::kDynamic, 3}, realType), idxType,
           MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic},
                           f64Type)},
          {}));
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  Value x = entry->getArgument(0);
  Value q = entry->getArgument(1);
  Value phi = entry->getArgument(2);
  Value box = entry->getArgument(3);
  Value k1 = entry->getArgument(4);
  Value k2 = entry->getArgument(5);
  Value k3 = entry->getArgument(6);
  Value n = entry->getArgument(7);
  Value f = entry->getArgument(8);
  Value i = entry->getArgument(9);
  Value moduli = entry->getArgument(10);
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  Value c2 = arith::ConstantIndexOp::create(b, loc, 2);
  Value lx = vector::ExtractOp::create(b, loc, box, ArrayRef<int64_t>{0});
  Value ly = vector::ExtractOp::create(b, loc, box, ArrayRef<int64_t>{1});
  Value lz = vector::ExtractOp::create(b, loc, box, ArrayRef<int64_t>{2});
  Value k1i = arith::IndexCastOp::create(b, loc, i64Type, k1);
  Value k2i = arith::IndexCastOp::create(b, loc, i64Type, k2);
  Value k3i = arith::IndexCastOp::create(b, loc, i64Type, k3);
  Value k1f = arith::SIToFPOp::create(b, loc, f64Type, k1i);
  Value k2f = arith::SIToFPOp::create(b, loc, f64Type, k2i);
  Value k3f = arith::SIToFPOp::create(b, loc, f64Type, k3i);
  Value rx = arith::DivFOp::create(b, loc, k1f, lx);
  Value ry = arith::DivFOp::create(b, loc, k2f, ly);
  Value rz = arith::DivFOp::create(b, loc, k3f, lz);
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
  Value qi = convert(f64Type, qs);
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
  auto gxCall = func::CallOp::create(
      b, loc, "mdrt.pme_gather_sums", TypeRange{f64Type, f64Type, f64Type},
      ValueRange{phi, sx, sy, sz, wx, wy, wz, dx, dy, dz, k1, k2, k3, n});
  Value gx = gxCall.getResult(0);
  Value gy = gxCall.getResult(1);
  Value gz = gxCall.getResult(2);
  Value mq = arith::NegFOp::create(b, loc, qi);
  Value sxq = arith::MulFOp::create(b, loc, mq, rx);
  Value syq = arith::MulFOp::create(b, loc, mq, ry);
  Value szq = arith::MulFOp::create(b, loc, mq, rz);
  Value fx64 = arith::MulFOp::create(b, loc, sxq, gx);
  Value fy64 = arith::MulFOp::create(b, loc, syq, gy);
  Value fz64 = arith::MulFOp::create(b, loc, szq, gz);
  Value fxs = convert(realType, fx64);
  Value fys = convert(realType, fy64);
  Value fzs = convert(realType, fz64);
  store(fxs, f, {i, c0});
  store(fys, f, {i, c1});
  store(fzs, f, {i, c2});
  func::ReturnOp::create(b, loc);
}

// The potential of the grid at particle `i`, Σ_k φ(k) θ_i(k): the
// derivative of the energy of the sum in the charge of the particle, the
// fourth result of `md.reciprocal` (D231). The places
// and the splines are those of @mdrt.pme_gather_one; the buffer holds
// one number for each particle, in the type that the op stores it in.
void HostPME::emitPotentialOne() {
  OpBuilder::InsertionGuard guard(b);
  auto fn = declare(
      "mdrt.pme_potential_one",
      b.getFunctionType(
          {MemRefType::get({ShapedType::kDynamic, 3}, positionType),
           MemRefType::get({ShapedType::kDynamic}, chargeType),
           MemRefType::get({ShapedType::kDynamic}, realType),
           VectorType::get({3}, f64Type), idxType, idxType, idxType, idxType,
           MemRefType::get({ShapedType::kDynamic}, realType), idxType,
           MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic},
                           f64Type)},
          {}));
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  Value x = entry->getArgument(0);
  Value phi = entry->getArgument(2);
  Value box = entry->getArgument(3);
  Value k1 = entry->getArgument(4);
  Value k2 = entry->getArgument(5);
  Value k3 = entry->getArgument(6);
  Value n = entry->getArgument(7);
  Value f = entry->getArgument(8);
  Value i = entry->getArgument(9);
  Value moduli = entry->getArgument(10);
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
  Value xi = convert(f64Type, xs);
  Value yi = convert(f64Type, ys);
  Value zi = convert(f64Type, zs);
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
  Value g = func::CallOp::create(
                b, loc, "mdrt.pme_potential_sum", TypeRange{f64Type},
                ValueRange{phi, sx, sy, sz, wx, wy, wz, k1, k2, k3, n})
                .getResult(0);
  Value stored = convert(realType, g);
  store(stored, f, {i});
  func::ReturnOp::create(b, loc);
}

// The places and the weights of particle `i` in a triclinic cell, from its
// fractional coordinates s = x H⁻¹: its first points, written to `starts`.
void HostPME::emitSplinesTriclinic() {
  OpBuilder::InsertionGuard guard(b);
  auto fn = declare(
      "mdrt.pme_splines_triclinic",
      b.getFunctionType(
          {MemRefType::get({ShapedType::kDynamic, 3}, positionType),
           VectorType::get({6}, f64Type), idxType, idxType, idxType, idxType,
           idxType,
           MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic},
                           f64Type),
           MemRefType::get({8}, realType), MemRefType::get({8}, realType),
           MemRefType::get({8}, realType), MemRefType::get({8}, realType),
           MemRefType::get({8}, realType), MemRefType::get({8}, realType)},
          {idxType, idxType, idxType}));
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  Value x = entry->getArgument(0);
  Value box = entry->getArgument(1);
  Value k1 = entry->getArgument(2);
  Value k2 = entry->getArgument(3);
  Value k3 = entry->getArgument(4);
  Value n = entry->getArgument(5);
  Value i = entry->getArgument(6);
  Value moduli = entry->getArgument(7);
  Value wx = entry->getArgument(8);
  Value wy = entry->getArgument(9);
  Value wz = entry->getArgument(10);
  Value dx = entry->getArgument(11);
  Value dy = entry->getArgument(12);
  Value dz = entry->getArgument(13);
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
  Value xs = load(x, {i, c0});
  Value ys = load(x, {i, c1});
  Value zs = load(x, {i, c2});
  Value xi = convert(f64Type, xs);
  Value yi = convert(f64Type, ys);
  Value zi = convert(f64Type, zs);
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
  func::ReturnOp::create(b, loc, ValueRange{sx, sy, sz});
}

// As pme_gather_one, in a triclinic cell.
void HostPME::emitGatherOneTriclinic() {
  OpBuilder::InsertionGuard guard(b);
  auto fn = declare(
      "mdrt.pme_gather_one_triclinic",
      b.getFunctionType(
          {MemRefType::get({ShapedType::kDynamic, 3}, positionType),
           MemRefType::get({ShapedType::kDynamic}, chargeType),
           MemRefType::get({ShapedType::kDynamic}, realType),
           VectorType::get({6}, f64Type), idxType, idxType, idxType, idxType,
           MemRefType::get({ShapedType::kDynamic, 3}, realType), idxType,
           MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic},
                           f64Type)},
          {}));
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  Value x = entry->getArgument(0);
  Value q = entry->getArgument(1);
  Value phi = entry->getArgument(2);
  Value box = entry->getArgument(3);
  Value k1 = entry->getArgument(4);
  Value k2 = entry->getArgument(5);
  Value k3 = entry->getArgument(6);
  Value n = entry->getArgument(7);
  Value f = entry->getArgument(8);
  Value i = entry->getArgument(9);
  Value moduli = entry->getArgument(10);
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  Value c2 = arith::ConstantIndexOp::create(b, loc, 2);
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
  Value k1i = arith::IndexCastOp::create(b, loc, i64Type, k1);
  Value k2i = arith::IndexCastOp::create(b, loc, i64Type, k2);
  Value k3i = arith::IndexCastOp::create(b, loc, i64Type, k3);
  Value k1f = arith::SIToFPOp::create(b, loc, f64Type, k1i);
  Value k2f = arith::SIToFPOp::create(b, loc, f64Type, k2i);
  Value k3f = arith::SIToFPOp::create(b, loc, f64Type, k3i);
  Value wx = memref::AllocaOp::create(b, loc, MemRefType::get({8}, realType));
  Value wy = memref::AllocaOp::create(b, loc, MemRefType::get({8}, realType));
  Value wz = memref::AllocaOp::create(b, loc, MemRefType::get({8}, realType));
  Value dx = memref::AllocaOp::create(b, loc, MemRefType::get({8}, realType));
  Value dy = memref::AllocaOp::create(b, loc, MemRefType::get({8}, realType));
  Value dz = memref::AllocaOp::create(b, loc, MemRefType::get({8}, realType));
  Value qs = load(q, {i});
  Value qi = convert(f64Type, qs);
  auto sxCall = func::CallOp::create(
      b, loc, "mdrt.pme_splines_triclinic",
      TypeRange{idxType, idxType, idxType},
      ValueRange{x, box, k1, k2, k3, n, i, moduli, wx, wy, wz, dx, dy, dz});
  Value sx = sxCall.getResult(0);
  Value sy = sxCall.getResult(1);
  Value sz = sxCall.getResult(2);
  auto gxCall = func::CallOp::create(
      b, loc, "mdrt.pme_gather_sums", TypeRange{f64Type, f64Type, f64Type},
      ValueRange{phi, sx, sy, sz, wx, wy, wz, dx, dy, dz, k1, k2, k3, n});
  Value gx = gxCall.getResult(0);
  Value gy = gxCall.getResult(1);
  Value gz = gxCall.getResult(2);
  Value mq = arith::NegFOp::create(b, loc, qi);
  Value u1 = arith::MulFOp::create(b, loc, gx, k1f);
  Value u2 = arith::MulFOp::create(b, loc, gy, k2f);
  Value u3 = arith::MulFOp::create(b, loc, gz, k3f);
  Value gx_lab = arith::MulFOp::create(b, loc, u1, h00);
  Value gy_1 = arith::MulFOp::create(b, loc, u1, h10);
  Value gy_2 = arith::MulFOp::create(b, loc, u2, h11);
  Value gy_lab = arith::AddFOp::create(b, loc, gy_1, gy_2);
  Value gz_1 = arith::MulFOp::create(b, loc, u1, h20);
  Value gz_2 = arith::MulFOp::create(b, loc, u2, h21);
  Value gz_3 = arith::MulFOp::create(b, loc, u3, h22);
  Value gz_12 = arith::AddFOp::create(b, loc, gz_1, gz_2);
  Value gz_lab = arith::AddFOp::create(b, loc, gz_12, gz_3);
  Value fx64 = arith::MulFOp::create(b, loc, mq, gx_lab);
  Value fy64 = arith::MulFOp::create(b, loc, mq, gy_lab);
  Value fz64 = arith::MulFOp::create(b, loc, mq, gz_lab);
  Value fxs = convert(realType, fx64);
  Value fys = convert(realType, fy64);
  Value fzs = convert(realType, fz64);
  store(fxs, f, {i, c0});
  store(fys, f, {i, c1});
  store(fzs, f, {i, c2});
  func::ReturnOp::create(b, loc);
}

// The potential of the grid at particle `i` in a triclinic cell, as
// pme_potential_one gives it.
void HostPME::emitPotentialOneTriclinic() {
  OpBuilder::InsertionGuard guard(b);
  auto fn = declare(
      "mdrt.pme_potential_one_triclinic",
      b.getFunctionType(
          {MemRefType::get({ShapedType::kDynamic, 3}, positionType),
           MemRefType::get({ShapedType::kDynamic}, chargeType),
           MemRefType::get({ShapedType::kDynamic}, realType),
           VectorType::get({6}, f64Type), idxType, idxType, idxType, idxType,
           MemRefType::get({ShapedType::kDynamic}, realType), idxType,
           MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic},
                           f64Type)},
          {}));
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  Value x = entry->getArgument(0);
  Value phi = entry->getArgument(2);
  Value box = entry->getArgument(3);
  Value k1 = entry->getArgument(4);
  Value k2 = entry->getArgument(5);
  Value k3 = entry->getArgument(6);
  Value n = entry->getArgument(7);
  Value f = entry->getArgument(8);
  Value i = entry->getArgument(9);
  Value moduli = entry->getArgument(10);
  Value wx = memref::AllocaOp::create(b, loc, MemRefType::get({8}, realType));
  Value wy = memref::AllocaOp::create(b, loc, MemRefType::get({8}, realType));
  Value wz = memref::AllocaOp::create(b, loc, MemRefType::get({8}, realType));
  Value dx = memref::AllocaOp::create(b, loc, MemRefType::get({8}, realType));
  Value dy = memref::AllocaOp::create(b, loc, MemRefType::get({8}, realType));
  Value dz = memref::AllocaOp::create(b, loc, MemRefType::get({8}, realType));
  auto sxCall = func::CallOp::create(
      b, loc, "mdrt.pme_splines_triclinic",
      TypeRange{idxType, idxType, idxType},
      ValueRange{x, box, k1, k2, k3, n, i, moduli, wx, wy, wz, dx, dy, dz});
  Value sx = sxCall.getResult(0);
  Value sy = sxCall.getResult(1);
  Value sz = sxCall.getResult(2);
  Value g = func::CallOp::create(
                b, loc, "mdrt.pme_potential_sum", TypeRange{f64Type},
                ValueRange{phi, sx, sy, sz, wx, wy, wz, k1, k2, k3, n})
                .getResult(0);
  Value stored = convert(realType, g);
  store(stored, f, {i});
  func::ReturnOp::create(b, loc);
}
