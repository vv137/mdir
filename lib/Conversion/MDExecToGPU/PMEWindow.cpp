#include "DevicePME.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;

// Cubic cardinal weights in mesh order, and their fractional derivatives.
std::array<Value, 4> DevicePME::cubicWeights(Value u,
                                            std::array<Value, 4> *slopes) {
  auto constant = [&](double x) {
    return arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, x));
  };
  Value one = constant(1), three = constant(3), sixth = constant(1.0 / 6);
  Value v = subf(one, u), vv = mulf(v, v), uu = mulf(u, u);
  Value cube = mulf(uu, u), threeCube = mulf(three, cube);
  if (slopes) {
    Value half = constant(0.5);
    Value first = mulf(half, vv), last = mulf(half, uu);
    Value middle = subf(subf(one, first), last);
    *slopes = {subf(constant(0), first), subf(first, middle),
               subf(middle, last), last};
  }
  Value w1 = addf(subf(threeCube, mulf(constant(6), uu)), constant(4));
  Value w2 = addf(addf(subf(mulf(three, uu), threeCube), mulf(three, u)), one);
  return {mulf(mulf(vv, v), sixth), mulf(w1, sixth), mulf(w2, sixth),
          mulf(cube, sixth)};
}

Value DevicePME::cubicLane(const std::array<Value, 4> &weights, Value lane) {
  Value selected = weights.back();
  for (int j = 2; j >= 0; --j) {
    Value index = arith::ConstantOp::create(
        b, loc, b.getIntegerAttr(lane.getType(), j));
    selected = select(cmpi(arith::CmpIPredicate::eq, lane, index),
                      weights[j], selected);
  }
  return selected;
}

// Cardinal B-spline recursion [Essmann1995, Eq. 2.28].
void DevicePME::splineWeights(Value fraction, Value pointCount, Value weights,
                              Value slopes) {
  splineWindow<true>(fraction, pointCount, weights, slopes,
                     [&](Value k, Value c0, Value c1, Value c2, Value one) {
                       raiseSpline(fraction, k, weights, c0, c1, c2, one);
                     });
}

// Wrap an axis before assigning its first mesh point and fraction.
std::pair<Value, Value> DevicePME::placeAxis(Value coordinate,
                                             Value inverseLength, Value size,
                                             Value pointCount) {
  Value s = mulf(coordinate, inverseLength);
  return splitMeshAxis(meshCoordinate(s, size), size, pointCount);
}

Value DevicePME::meshCoordinate(Value s, Value size) {
  Value fs = math::FloorOp::create(b, loc, s);
  Value frac = subf(s, fs);
  Value ki = indexCast(i32Type, size);
  Value kf = sitofp(realType, ki);
  Value fracr = convert(realType, frac);
  return mulf(fracr, kf);
}

std::pair<Value, Value> DevicePME::splitMeshAxis(Value u, Value size,
                                                  Value pointCount, bool cubic) {
  Value fu0 = math::FloorOp::create(b, loc, u);
  Value nil = arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.0));
  Value fu = maxnumf(fu0, nil);
  Value w = subf(u, fu);
  Value bi32 = fptosi(i32Type, fu);
  if (cubic) {
    // The packed mesh coordinate lies in [0, K], including a point that
    // rounded up to K. Center the four-point stencil using i32 arithmetic.
    Value three = arith::ConstantIntOp::create(b, loc, 3, 32);
    Value n = indexCast(i32Type, size);
    Value raw = subi(bi32, three);
    Value upper = select(cmpi(arith::CmpIPredicate::sge, raw, n), subi(raw, n), raw);
    Value start = select(cmpi(arith::CmpIPredicate::slt, bi32, three), addi(raw, n), upper);
    return {indexCast(idxType, start), w};
  }
  Value bi = arith::ExtSIOp::create(b, loc, i64Type, bi32);
  Value accB = indexCast(idxType, bi);
  Value bk = addi(accB, size);
  Value bkn = subi(bk, pointCount);
  Value c1p = arith::ConstantIndexOp::create(b, loc, 1);
  Value start0 = addi(bkn, c1p);
  Value start = remui(start0, size);
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
        Value here = cmpi(arith::CmpIPredicate::eq, e, atIndex);
        Value chosen = select(here, at, acc);
        scf::YieldOp::create(b, loc, ValueRange{chosen});
      });
  Value value = valueOp.getResult(0);
  return value;
}

// Evaluate the Chebyshev window or its derivative at one point [Liang2026].
Value DevicePME::clenshaw(Value base, Value fraction, Value one, Value terms,
                          Value points, Value j, Value tables) {
  Value term_t2 = addf(fraction, fraction);
  Value term_z =
      arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.0));
  return evaluateSeries(terms, one, fraction, term_t2, term_z, [&](Value k) {
    Value at = base;
    if (k) {
      Value offset = muli(k, points);
      at = addi(base, offset);
    }
    at = addi(at, j);
    return load(tables, {at});
  });
}

// Signed frequency on a periodic grid.
Value DevicePME::waveNumber64(Value k, Value count, Value inverseLength,
                              Value cc2) {
  Value half = divui(count, cc2);
  Value ki = indexCast(i64Type, k);
  Value ni = indexCast(i64Type, count);
  Value beyond = cmpi(arith::CmpIPredicate::ugt, k, half);
  Value shifted = subi(ki, ni);
  Value signedIndex = select(beyond, shifted, ki);
  Value sf = sitofp(f64Type, signedIndex);
  Value m = mulf(sf, inverseLength);
  return m;
}

DevicePME::LaneWindow DevicePME::laneWindow(Value lane, Value i, Value places,
                                            Value tables, Value n_all, Value pe,
                                            Value c0w, Value two_i,
                                            Value three_i, bool slopes) {
  Value points =
      arith::ConstantOp::create(b, loc, b.getIntegerAttr(idxType, order));
  Value one = arith::ConstantIndexOp::create(b, loc, 1);
  Value lane_i = indexCast(idxType, lane);
  Value three_p = arith::ConstantIndexOp::create(b, loc, 3);
  Value tp = muli(three_p, points);
  Value active = cmpi(arith::CmpIPredicate::ult, lane_i, tp);
  Value lane_a = select(active, lane_i, c0w);
  Value axis = divui(lane_a, points);
  Value j = remui(lane_a, points);
  Value fat0 = addi(axis, three_i);
  Value fat1 = muli(fat0, n_all);
  Value fat = addi(fat1, i);
  Value frac = load(places, {fat});
  Value twor = arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 2.0));
  Value oner = arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 1.0));
  Value frac2 = mulf(frac, twor);
  Value t = subf(frac2, oner);
  Value meta_d = addi(pe, two_i);
  Value dr = load(tables, {meta_d});
  Value meta_n = load(tables, {pe});
  Value terms_i = fptosi(i32Type, dr);
  Value terms = indexCast(idxType, terms_i);
  Value n_i = fptosi(i32Type, meta_n);
  Value n_s = indexCast(idxType, n_i);
  Value n_2 = addi(n_s, n_s);
  Value wb0 = addi(pe, three_i);
  Value wbase = addi(wb0, n_2);
  Value wvalue0 = clenshaw(wbase, t, one, terms, points, j, tables);
  Value zero_w =
      arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.0));
  Value wvalue = select(active, wvalue0, zero_w);
  Value wslope;
  if (slopes) {
    Value pdw = muli(points, terms);
    Value sbase = addi(wbase, pdw);
    Value wslope0 = clenshaw(sbase, t, one, terms, points, j, tables);
    wslope = select(active, wslope0, zero_w);
  }
  return {wvalue, wslope, axis, zero_w, one};
}

Value DevicePME::placementStart(Value places, Value n, Value i, int axis) {
  Value a = arith::ConstantIndexOp::create(b, loc, axis);
  Value offset = muli(n, a);
  Value at = addi(offset, i);
  Value start = load(places, {at});
  return fptosi(i32Type, start);
}
