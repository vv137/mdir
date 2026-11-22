#include "DevicePME.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;

// Cardinal B-spline recursion [Essmann1995, Eq. 2.28].
void DevicePME::splineWeights(Value fraction, Value pointCount, Value weights,
                              Value slopes) {
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  Value c2 = arith::ConstantIndexOp::create(b, loc, 2);
  Value c3 = arith::ConstantIndexOp::create(b, loc, 3);
  Value c8 = arith::ConstantIndexOp::create(b, loc, 8);
  Value zero = arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.0));
  Value one = arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 1.0));
  scf::ForOp::create(b, loc, c0, c8, c1, ValueRange{},
                     [&](OpBuilder &b, Location, Value j, ValueRange carried) {
                       store(zero, weights, {j});
                       scf::YieldOp::create(b, loc);
                     });
  Value omw = arith::SubFOp::create(b, loc, one, fraction);
  store(omw, weights, {c0});
  store(fraction, weights, {c1});
  scf::ForOp::create(b, loc, c3, pointCount, c1, ValueRange{},
                     [&](OpBuilder &b, Location, Value k, ValueRange carried) {
                       raiseSpline(fraction, k, weights, c0, c1, c2, one);
                       scf::YieldOp::create(b, loc);
                     });
  if (slopes) {
    Value v0 = load(weights, {c0});
    Value s0 = arith::NegFOp::create(b, loc, v0);
    store(s0, slopes, {c0});
    scf::ForOp::create(
        b, loc, c1, pointCount, c1, ValueRange{},
        [&](OpBuilder &b, Location, Value js, ValueRange carried) {
          Value jm1 = arith::SubIOp::create(b, loc, js, c1);
          Value before = load(weights, {jm1});
          Value here = load(weights, {js});
          Value slope = arith::SubFOp::create(b, loc, before, here);
          store(slope, slopes, {js});
          scf::YieldOp::create(b, loc);
        });
  }
  raiseSpline(fraction, pointCount, weights, c0, c1, c2, one);
}

// Wrap an axis before assigning its first mesh point and fraction.
std::pair<Value, Value> DevicePME::placeAxis(Value coordinate,
                                             Value inverseLength, Value size,
                                             Value pointCount) {
  Value s = arith::MulFOp::create(b, loc, coordinate, inverseLength);
  Value fs = math::FloorOp::create(b, loc, s);
  Value frac = arith::SubFOp::create(b, loc, s, fs);
  Value ki = arith::IndexCastOp::create(b, loc, i32Type, size);
  Value kf = arith::SIToFPOp::create(b, loc, realType, ki);
  Value fracr = convert(realType, frac);
  Value u = arith::MulFOp::create(b, loc, fracr, kf);
  Value fu0 = math::FloorOp::create(b, loc, u);
  Value nil = arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.0));
  Value fu = arith::MaxNumFOp::create(b, loc, fu0, nil);
  Value w = arith::SubFOp::create(b, loc, u, fu);
  Value bi32 = arith::FPToSIOp::create(b, loc, i32Type, fu);
  Value bi = arith::ExtSIOp::create(b, loc, i64Type, bi32);
  Value accB = arith::IndexCastOp::create(b, loc, idxType, bi);
  Value bk = arith::AddIOp::create(b, loc, accB, size);
  Value bkn = arith::SubIOp::create(b, loc, bk, pointCount);
  Value c1p = arith::ConstantIndexOp::create(b, loc, 1);
  Value start0 = arith::AddIOp::create(b, loc, bkn, c1p);
  Value start = arith::RemUIOp::create(b, loc, start0, size);
  return {start, w};
}

// Select a register-resident weight after unrolling the loop over support.
Value DevicePME::pickSpline(Value array, Value atIndex, Value pointCount) {
  Value none = arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.0));
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  auto valueOp = scf::ForOp::create(
      b, loc, c0, pointCount, c1, ValueRange{none},
      [&](OpBuilder &b, Location, Value e, ValueRange carried) {
        Value acc = carried[0];
        Value at = load(array, {e});
        Value here =
            arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, e, atIndex);
        Value chosen = arith::SelectOp::create(b, loc, here, at, acc);
        scf::YieldOp::create(b, loc, ValueRange{chosen});
      });
  Value value = valueOp.getResult(0);
  return value;
}

// Evaluate the Chebyshev window or its derivative at one point [Liang2026].
Value DevicePME::clenshaw(Value base, Value fraction, Value one, Value terms,
                          Value points, Value j, Value tables) {
  Value term_t2 = arith::AddFOp::create(b, loc, fraction, fraction);
  Value term_z =
      arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.0));
  auto term_b1Op = scf::ForOp::create(
      b, loc, one, terms, one, ValueRange{term_z, term_z},
      [&](OpBuilder &b, Location, Value term_r, ValueRange carried) {
        Value term_p1 = carried[0];
        Value term_p2 = carried[1];
        Value term_k = arith::SubIOp::create(b, loc, terms, term_r);
        Value term_kp = arith::MulIOp::create(b, loc, term_k, points);
        Value term_at0 = arith::AddIOp::create(b, loc, base, term_kp);
        Value term_at = arith::AddIOp::create(b, loc, term_at0, j);
        Value term_a = load(tables, {term_at});
        Value term_m = arith::MulFOp::create(b, loc, term_t2, term_p1);
        Value term_s = arith::SubFOp::create(b, loc, term_m, term_p2);
        Value term_n = arith::AddFOp::create(b, loc, term_s, term_a);
        scf::YieldOp::create(b, loc, ValueRange{term_n, term_p1});
      });
  Value term_b1 = term_b1Op.getResult(0);
  Value term_b2 = term_b1Op.getResult(1);
  Value term_at00 = arith::AddIOp::create(b, loc, base, j);
  Value term_a0 = load(tables, {term_at00});
  Value term_m0 = arith::MulFOp::create(b, loc, fraction, term_b1);
  Value term_s0 = arith::SubFOp::create(b, loc, term_m0, term_b2);
  Value result = arith::AddFOp::create(b, loc, term_s0, term_a0);
  return result;
}

// Signed frequency on a periodic grid.
Value DevicePME::waveNumber64(Value k, Value count, Value inverseLength,
                              Value cc2) {
  Value half = arith::DivUIOp::create(b, loc, count, cc2);
  Value ki = arith::IndexCastOp::create(b, loc, i64Type, k);
  Value ni = arith::IndexCastOp::create(b, loc, i64Type, count);
  Value beyond =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ugt, k, half);
  Value shifted = arith::SubIOp::create(b, loc, ki, ni);
  Value signedIndex = arith::SelectOp::create(b, loc, beyond, shifted, ki);
  Value sf = arith::SIToFPOp::create(b, loc, f64Type, signedIndex);
  Value m = arith::MulFOp::create(b, loc, sf, inverseLength);
  return m;
}
