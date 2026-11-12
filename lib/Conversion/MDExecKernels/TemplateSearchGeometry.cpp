// Cartesian covering geometry with a certified uncertainty shell.
// Standard forward rounding-error model: Higham, Accuracy and Stability of
// Numerical Algorithms, 2nd ed., Ch. 3. The original f64 predicate is the
// reference operation graph; these bounds never change its physical reach.
#include "mdir/Conversion/TemplateSearchGeometry.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include <cassert>
#include <cmath>
#include <limits>

using namespace mlir;

namespace {
// All certificate expressions below are nonnegative and monotone. Round
// EVERY elementary result outward, including underflow, rather than applying
// a fitted epsilon once at the end. A positive finite f64's next bit pattern
// is its next representable value; zero is canonicalized before increment.
class Bounds {
public:
  Bounds(OpBuilder &b, Location loc) : b(b), loc(loc) {}
  Value c(double x) { return arith::ConstantFloatOp::create(b, loc, b.getF64Type(), APFloat(x)); }
  Value cmp(arith::CmpFPredicate pred, Value a, Value d) {
    return arith::CmpFOp::create(b, loc, pred, a, d);
  }
  Value both(Value a, Value d) { return arith::AndIOp::create(b, loc, a, d); }
  Value finite(Value a) {
    return cmp(arith::CmpFPredicate::OLE, math::AbsFOp::create(b, loc, a),
               c(std::numeric_limits<double>::max()));
  }
  Value rawAdd(Value a, Value d) { return arith::AddFOp::create(b, loc, a, d); }
  Value rawSub(Value a, Value d) { return arith::SubFOp::create(b, loc, a, d); }
  Value rawMul(Value a, Value d) { return arith::MulFOp::create(b, loc, a, d); }
  Value up(Value a) {
    Value zero = c(0), infinity = c(std::numeric_limits<double>::infinity());
    Value valid = both(cmp(arith::CmpFPredicate::OGE, a, zero),
                       cmp(arith::CmpFPredicate::OLT, a, infinity));
    Value canonical = arith::SelectOp::create(b, loc,
        cmp(arith::CmpFPredicate::OEQ, a, zero), zero, a);
    Value bits = arith::BitcastOp::create(b, loc, b.getI64Type(), canonical);
    Value next = arith::AddIOp::create(b, loc, bits,
        arith::ConstantIntOp::create(b, loc, 1, 64));
    Value result = arith::BitcastOp::create(b, loc, b.getF64Type(), next);
    return arith::SelectOp::create(b, loc, valid, result, infinity);
  }
  // Clamp a negative lower bound to zero. Such a value may be used only
  // in nonnegative monotone expressions; final acceptance is separately
  // disabled unless the resulting lower threshold is strictly positive.
  Value down(Value a) {
    Value positive = cmp(arith::CmpFPredicate::OGT, a, c(0));
    Value bits = arith::BitcastOp::create(b, loc, b.getI64Type(), a);
    Value previous = arith::SubIOp::create(b, loc, bits,
        arith::ConstantIntOp::create(b, loc, 1, 64));
    Value result = arith::BitcastOp::create(b, loc, b.getF64Type(), previous);
    return arith::SelectOp::create(b, loc, positive, result, c(0));
  }
  Value add(Value a, Value d) { return up(rawAdd(a, d)); }
  Value mul(Value a, Value d) { return up(rawMul(a, d)); }
  Value square(Value a) { return mul(a, a); }
  Value addDown(Value a, Value d) { return down(rawAdd(a, d)); }
  Value subDown(Value a, Value d) { return down(rawSub(a, d)); }
  Value mulDown(Value a, Value d) { return down(rawMul(a, d)); }
  Value divUp(Value a, Value d) { return up(arith::DivFOp::create(b, loc, a, d)); }
  Value divDown(Value a, Value d) { return down(arith::DivFOp::create(b, loc, a, d)); }
  Value sqrtUp(Value a) { return up(math::SqrtOp::create(b, loc, a)); }
  Value sqrtDown(Value a) { return down(math::SqrtOp::create(b, loc, a)); }

  Value toF32(Value a, bool upward) {
    Value rounded = arith::TruncFOp::create(b, loc, b.getF32Type(), a);
    Value extended = arith::ExtFOp::create(b, loc, b.getF64Type(), rounded);
    Value adjust = cmp(upward ? arith::CmpFPredicate::OLT : arith::CmpFPredicate::OGT,
                       extended, a);
    Value bits = arith::BitcastOp::create(b, loc, b.getI32Type(), rounded);
    Value one = arith::ConstantIntOp::create(b, loc, 1, 32);
    Value changed = upward ? Value(arith::AddIOp::create(b, loc, bits, one))
                           : Value(arith::SubIOp::create(b, loc, bits, one));
    Value next = arith::BitcastOp::create(b, loc, b.getF32Type(), changed);
    return arith::SelectOp::create(b, loc, adjust, next, rounded);
  }

private:
  OpBuilder &b;
  Location loc;
};

// Numerator and denominator are exact dyadics here (n <= 9). One nextafter
// encloses the correctly rounded host division; no runtime bound is fitted.
double gamma(unsigned n, double u) {
  return std::nextafter((n * u) / (1 - n * u),
                        std::numeric_limits<double>::infinity());
}
double beta(unsigned n, double u, double nu) {
  return std::nextafter((n * nu) / (1 - n * u),
                        std::numeric_limits<double>::infinity());
}
} // namespace

mdir::kernels::TemplateSearchCertificate
mdir::kernels::emitTemplateSearchCertificate(OpBuilder &b, Location loc,
    Value cell, Value frame, Value imageEnvelope, Value reach2) {
  auto cellType = cast<VectorType>(cell.getType());
  assert(cellType.getElementType().isF64() &&
         (cellType.getNumElements() == 3 || cellType.getNumElements() == 6));
  assert(frame.getType() == VectorType::get({3}, b.getF64Type()));
  assert(imageEnvelope.getType() == frame.getType() && reach2.getType().isF64());
  Bounds q(b, loc);
  constexpr double uf = 0x1p-24, ud = 0x1p-53;
  constexpr double nf = 0x1p-126, nd = 0x1p-1022;
  Value zero = q.c(0), one = q.c(1), two = q.c(2), three = q.c(3);
  Value g2f = q.c(gamma(2, uf)), g3f = q.c(gamma(3, uf));
  Value g2d = q.c(gamma(2, ud)), g3d = q.c(gamma(3, ud));
  Value b9f = q.c(beta(9, uf, nf)), b9d = q.c(beta(9, ud, nd));
  Value b5f = q.c(beta(5, uf, nf)), b5d = q.c(beta(5, ud, nd));
  Value rho = q.c(std::nextafter(ud + nd, std::numeric_limits<double>::infinity()));
  Value a2f = q.c(std::nextafter((6 + 3 * uf) * nf, std::numeric_limits<double>::infinity()));
  Value a2d = q.c(std::nextafter((6 + 3 * ud) * nd, std::numeric_limits<double>::infinity()));
  Value enabled = q.both(q.finite(reach2),
      q.cmp(arith::CmpFPredicate::OGE, reach2, zero));
  Value h = zero, f = zero, s = zero;
  for (int64_t i = 0; i != cellType.getNumElements(); ++i) {
    Value component = vector::ExtractOp::create(b, loc, cell, i);
    enabled = q.both(enabled, q.finite(component));
    h = q.add(h, math::AbsFOp::create(b, loc, component));
  }
  for (int64_t i = 0; i != 3; ++i) {
    Value n = vector::ExtractOp::create(b, loc, frame, i);
    Value e = vector::ExtractOp::create(b, loc, imageEnvelope, i);
    enabled = q.both(enabled, q.both(q.finite(n), q.finite(e)));
    enabled = q.both(enabled, q.both(
        q.cmp(arith::CmpFPredicate::OGE, e, zero),
        q.cmp(arith::CmpFPredicate::OLE, e, q.c(0x1p24))));
    f = q.add(f, math::AbsFOp::create(b, loc, n));
    s = q.add(s, e);
  }
  // frame is integer-valued by the caller. F <= 2^52 and |s_i| <= 2^24
  // imply exact64(n_i+s_i), while every queried s_i is exact32.
  enabled = q.both(enabled, q.cmp(arith::CmpFPredicate::OLE, f, q.c(0x1p52)));
  Value maxFloat = q.c(std::numeric_limits<float>::max());
  enabled = q.both(enabled, q.cmp(arith::CmpFPredicate::OLE, h, maxFloat));

  // c64 = H * fl64(P-floor(P)); eta bounds c32 versus exact H*X.
  Value wrapError = q.mul(three, rho);
  Value alpha = q.add(q.add(q.mul(h, wrapError),
      q.mul(g3d, q.mul(h, q.add(three, wrapError)))), b9d);
  Value kappa = q.add(q.mul(q.c(uf), q.add(h, alpha)), q.c(3 * nf));
  Value eta = q.add(alpha, kappa);
  Value eh = q.add(q.mul(q.c(uf), h), q.c(6 * nf));
  Value shifted = q.add(q.add(q.mul(eh, s),
      q.mul(g3f, q.mul(q.add(h, eh), s))), b9f);
  Value pointMagnitude = q.add(h, eta);
  Value shiftMagnitude = q.add(q.mul(h, s), shifted);
  Value operandMagnitude = q.add(q.mul(two, pointMagnitude), shiftMagnitude);
  Value delta32 = q.add(q.add(q.mul(two, eta), shifted),
      q.add(q.mul(g2f, operandMagnitude), a2f));

  // Original: P - (fl64(J-floor(J)) + exact64(n+s)), then H and dot.
  // Its absolute-image addition/cancellation still depends on F.
  Value rawMagnitude = q.add(q.add(q.mul(two, f), s), q.add(q.c(6), wrapError));
  Value e64 = q.add(wrapError, q.add(q.mul(g2d, rawMagnitude), a2d));
  Value delta64 = q.add(q.add(q.mul(h, e64),
      q.mul(g3d, q.mul(h, q.add(q.add(three, s), e64)))), b9d);
  Value delta = q.add(delta32, delta64);

  // Reach-based shell avoids charging a large global coordinate norm to
  // every near-cutoff pair. |D32-D64| <= delta. The dot envelopes are
  // (1-g3)|D|^2-b5 <= Q <= (1+g3)|D|^2+b5.
  Value oldRadiusUpper = q.sqrtUp(q.divUp(q.add(reach2, b5d), q.subDown(one, g3d)));
  Value upperRadius = q.add(oldRadiusUpper, delta);
  Value upper64 = q.add(q.mul(q.add(one, g3f), q.square(upperRadius)), b5f);
  Value oldRadiusLower = q.sqrtDown(q.divDown(q.subDown(reach2, b5d), q.add(one, g3d)));
  Value lowerRadius = q.subDown(oldRadiusLower, delta);
  Value lower64 = q.subDown(q.mulDown(q.subDown(one, g3f),
      q.mulDown(lowerRadius, lowerRadius)), b5f);

  // Boxes share c64. Min/max changes each component by <= kappa. The
  // max-gap function is 1-Lipschitz; three axes turn the per-axis bound
  // into an L1 (therefore also L2) displacement envelope. Do not treat
  // extrema selected from different points as one point's L1 error.
  Value boxOperandMagnitude = q.add(q.mul(two, q.add(q.add(h, alpha), kappa)),
                                    shiftMagnitude);
  Value gapError = q.add(q.add(q.mul(two, kappa), shifted),
      q.add(q.mul(g2f, boxOperandMagnitude), q.c(std::nextafter((2 + uf) * nf, std::numeric_limits<double>::infinity()))));
  Value gapDelta = q.mul(three, gapError);
  // oldQ <= reach2 => |exact D| <= oldRadiusUpper+delta64. Native c64
  // endpoints for that same pair add <= 2*alpha; the box gap cannot exceed
  // this pair distance. Finally add the f32 max-gap error and dot envelope.
  Value boxRadius = q.add(q.add(q.add(oldRadiusUpper, delta64), q.mul(two, alpha)), gapDelta);
  Value boxUpper64 = q.add(q.mul(q.add(one, g3f), q.square(boxRadius)), b5f);

  // Certify cast/shift/add/sub/dot intermediates, not merely final threshold
  // finiteness. Three times this per-component envelope also covers box gaps.
  Value dMagnitude = q.add(q.mul(q.add(one, g2f), operandMagnitude), a2f);
  Value qMagnitude = q.add(q.mul(q.add(one, g3f), q.square(q.mul(three, dMagnitude))), b5f);
  enabled = q.both(enabled, q.cmp(arith::CmpFPredicate::OLE, qMagnitude, maxFloat));
  Value lower = q.toF32(lower64, false), upper = q.toF32(upper64, true);
  Value boxUpper = q.toF32(boxUpper64, true);
  for (Value value : {delta32, delta64, upper64, boxUpper64})
    enabled = q.both(enabled, q.finite(value));
  for (Value value : {upper, boxUpper})
    enabled = q.both(enabled, q.cmp(arith::CmpFPredicate::OLE,
        arith::ExtFOp::create(b, loc, b.getF64Type(), value), maxFloat));
  Value acceptEnabled = q.both(enabled,
      q.both(q.cmp(arith::CmpFPredicate::OGT, lowerRadius, zero),
             q.cmp(arith::CmpFPredicate::OGT, lower64, zero)));
  return {lower, upper, boxUpper, enabled, acceptEnabled};
}
