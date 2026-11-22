// Common PME arithmetic [Essmann1995].
#pragma once
#include "KernelCode.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/TypeUtilities.h"
#include <tuple>

namespace mdir::kernels::detail {
class PMECode : public KernelCode {
protected:
  PMECode(OpBuilder &builder, Type position, Type charge, Type force)
      : KernelCode(builder, builder.getUnknownLoc()), positionType(position),
        chargeType(charge), realType(force) {}
  Type positionType, chargeType, realType;
  Type idxType = b.getIndexType(), i32Type = b.getI32Type(),
       i64Type = b.getI64Type(), f64Type = b.getF64Type();
  Value convert(Type type, Value input) {
    if (type == input.getType())
      return arith::BitcastOp::create(b, loc, type, input);
    auto width = [](Type t) {
      return getElementTypeOrSelf(t).getIntOrFloatBitWidth();
    };
    if (width(type) > width(input.getType()))
      return arith::ExtFOp::create(b, loc, type, input);
    return arith::TruncFOp::create(b, loc, type, input);
  }

  std::tuple<Value, Value, Value, Value, Value, Value>
  inverseCell(Value box, Value lx, Value ly, Value lz) {
    Value tbx = extract(box, 3);
    Value tcx = extract(box, 4);
    Value tcy = extract(box, 5);
    Value unit =
        arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 1.0));
    Value h00 = divf(unit, lx);
    Value h11 = divf(unit, ly);
    Value h22 = divf(unit, lz);
    Value bx_h00 = mulf(tbx, h00);
    Value bx_h00_h11 = mulf(bx_h00, h11);
    Value h10 = arith::NegFOp::create(b, loc, bx_h00_h11);
    Value cy_h11 = mulf(tcy, h11);
    Value cy_h11_h22 = mulf(cy_h11, h22);
    Value h21 = arith::NegFOp::create(b, loc, cy_h11_h22);
    Value bxcy = mulf(tbx, tcy);
    Value bycx = mulf(ly, tcx);
    Value h20n = subf(bxcy, bycx);
    Value h20a = mulf(h20n, h00);
    Value h20b = mulf(h20a, h11);
    Value h20 = mulf(h20b, h22);
    return {h00, h10, h11, h20, h21, h22};
  }

  // Clenshaw evaluation with coefficients in host rows or device lanes.
  template <class Coefficient>
  Value evaluateSeries(Value count, Value c1, Value t, Value t2, Value zero,
                       Coefficient coefficient) {
    auto b1Op = scf::ForOp::create(
        b, loc, c1, count, c1, ValueRange{zero, zero},
        [&](OpBuilder &b, Location, Value i, ValueRange carried) {
          Value p1 = carried[0];
          Value p2 = carried[1];
          Value k = subi(count, i);
          Value a = coefficient(k);
          Value m = mulf(t2, p1);
          Value s = subf(m, p2);
          Value accB = addf(s, a);
          scf::YieldOp::create(b, loc, ValueRange{accB, p1});
        });
    Value b1 = b1Op.getResult(0);
    Value b2 = b1Op.getResult(1);
    Value a0 = coefficient(Value{});
    Value m0 = mulf(t, b1);
    Value s0 = subf(m0, b2);
    Value v = addf(s0, a0);
    return v;
  }

  template <bool inlineRaise, class Raise>
  void splineWindow(Value fraction, Value pointCount, Value weights,
                    Value slopes, Raise raise) {
    Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
    Value c2;
    if constexpr (inlineRaise)
      c2 = arith::ConstantIndexOp::create(b, loc, 2);
    Value c3 = arith::ConstantIndexOp::create(b, loc, 3);
    Value c8 = arith::ConstantIndexOp::create(b, loc, 8);
    Value zero =
        arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.0));
    Value one =
        arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 1.0));
    scf::ForOp::create(
        b, loc, c0, c8, c1, ValueRange{},
        [&](OpBuilder &b, Location, Value j, ValueRange carried) {
          store(zero, weights, {j});
          scf::YieldOp::create(b, loc);
        });
    Value omw = subf(one, fraction);
    store(omw, weights, {c0});
    store(fraction, weights, {c1});
    scf::ForOp::create(
        b, loc, c3, pointCount, c1, ValueRange{},
        [&](OpBuilder &b, Location, Value k, ValueRange carried) {
          raise(k, c0, c1, c2, one);
          scf::YieldOp::create(b, loc);
        });
    if (slopes) {
      Value v0 = load(weights, {c0});
      Value s0 = arith::NegFOp::create(b, loc, v0);
      store(s0, slopes, {c0});
      scf::ForOp::create(
          b, loc, c1, pointCount, c1, ValueRange{},
          [&](OpBuilder &b, Location, Value js, ValueRange carried) {
            Value jm1 = subi(js, c1);
            Value before = load(weights, {jm1});
            Value here = load(weights, {js});
            Value slope = subf(before, here);
            store(slope, slopes, {js});
            scf::YieldOp::create(b, loc);
          });
    }
    raise(pointCount, c0, c1, c2, one);
  }

  // Raise the in-place weights from order k-1 to k (Eq. 2.28).
  void raiseSpline(Value w, Value k, Value a, Value c0, Value c1, Value c2,
                   Value one) {
    Value km1 = subi(k, c1);
    Value km2 = subi(k, c2);
    Value km1i = indexCast(i64Type, km1);
    Value km1f = sitofp(realType, km1i);
    Value inverse = divf(one, km1f);
    Value ki = indexCast(i64Type, k);
    Value kf = sitofp(realType, ki);
    Value last = load(a, {km2});
    Value dw = mulf(inverse, w);
    Value top = mulf(dw, last);
    store(top, a, {km1});
    scf::ForOp::create(
        b, loc, c1, km1, c1, ValueRange{},
        [&](OpBuilder &b, Location, Value j, ValueRange carried) {
          Value ji = indexCast(i64Type, j);
          Value jf = sitofp(realType, ji);
          Value at = subi(km1, j);
          Value below = subi(at, c1);
          Value lo = load(a, {below});
          Value hi = load(a, {at});
          Value wj = addf(w, jf);
          Value kj = subf(kf, jf);
          Value kjw = subf(kj, w);
          Value t1 = mulf(wj, lo);
          Value t2 = mulf(kjw, hi);
          Value sum = addf(t1, t2);
          Value value = mulf(inverse, sum);
          store(value, a, {at});
          scf::YieldOp::create(b, loc);
        });
    if (!c0)
      c0 = arith::ConstantIndexOp::create(b, loc, 0);
    Value first = load(a, {c0});
    Value omw = subf(one, w);
    Value d0 = mulf(inverse, omw);
    Value bottom = mulf(d0, first);
    store(bottom, a, {c0});
  }
};
} // namespace mdir::kernels::detail
