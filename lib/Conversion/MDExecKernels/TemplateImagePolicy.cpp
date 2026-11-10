// Refresh-time dispatch only: no image-policy branch is inserted in a search
// kernel. The c/b/a image argument is D241; forward rounding bounds use
// Higham2002, Ch. 3, with the original f64 predicate retained as the reference.
#include "mdir/Conversion/TemplateImagePolicy.h"
#include "mdir/Runtime/TemplatePairsABI.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include <cassert>
#include <cmath>
#include <limits>

using namespace mlir;

namespace {
class ImageBounds {
public:
  ImageBounds(OpBuilder &b, Location loc) : b(b), loc(loc) {}
  Value c(double x) {
    return arith::ConstantFloatOp::create(b, loc, b.getF64Type(), APFloat(x));
  }
  Value integer(int64_t x) { return arith::ConstantIntOp::create(b, loc, x, 64); }
  Value cmp(arith::CmpFPredicate pred, Value a, Value d) {
    return arith::CmpFOp::create(b, loc, pred, a, d);
  }
  Value both(Value a, Value d) { return arith::AndIOp::create(b, loc, a, d); }
  Value abs(Value x) { return math::AbsFOp::create(b, loc, x); }
  Value finite(Value x) {
    return cmp(arith::CmpFPredicate::OLE, abs(x), c(std::numeric_limits<double>::max()));
  }
  Value up(Value x) {
    Value zero = c(0), tiny = c(0x1p-1022);
    Value infinity = c(std::numeric_limits<double>::infinity());
    Value valid = both(cmp(arith::CmpFPredicate::OGE, x, zero),
                       cmp(arith::CmpFPredicate::OLT, x, infinity));
    Value canonical = arith::SelectOp::create(b, loc,
        cmp(arith::CmpFPredicate::OEQ, x, zero), zero, x);
    Value bits = arith::BitcastOp::create(b, loc, b.getI64Type(), canonical);
    Value nextBits = arith::AddIOp::create(b, loc, bits, integer(1));
    Value next = arith::BitcastOp::create(b, loc, b.getF64Type(), nextBits);
    // A rounded value below min-normal has exact positive value below it.
    // This fixed positive floor is outward and keeps bound outputs normal.
    Value result = arith::SelectOp::create(b, loc,
        cmp(arith::CmpFPredicate::OLT, canonical, tiny), tiny, next);
    return arith::SelectOp::create(b, loc, valid, result, infinity);
  }
  Value add(Value a, Value d) { return up(arith::AddFOp::create(b, loc, a, d)); }
  Value mul(Value a, Value d) { return up(arith::MulFOp::create(b, loc, a, d)); }
  Value sqrt(Value a) { return up(math::SqrtOp::create(b, loc, a)); }
private:
  OpBuilder &b;
  Location loc;
};

double gamma(unsigned n, double u) {
  return std::nextafter((n * u) / (1 - n * u),
                        std::numeric_limits<double>::infinity());
}
double beta(unsigned n, double u, double nu) {
  return std::nextafter((n * nu) / (1 - n * u),
                        std::numeric_limits<double>::infinity());
}
} // namespace

mdir::kernels::TemplateImagePolicy
mdir::kernels::emitTemplateImagePolicy(OpBuilder &b, Location loc, Value cell,
    Value maxReach, Value maxAbsFrame) {
  auto type = cast<VectorType>(cell.getType());
  assert(type.getElementType().isF64() &&
         (type.getNumElements() == 3 || type.getNumElements() == 6));
  assert(maxReach.getType().isF64() && maxAbsFrame.getType().isF64());
  ImageBounds q(b, loc);
  constexpr double u = 0x1p-53, nu = 0x1p-1022;
  Value zero = q.c(0), two = q.c(2), three = q.c(3);
  Value valid = q.both(q.finite(maxReach), q.finite(maxAbsFrame));
  valid = q.both(valid, q.both(
      q.cmp(arith::CmpFPredicate::OGE, maxReach, zero),
      q.cmp(arith::CmpFPredicate::OGE, maxAbsFrame, zero)));
  // Every provisional entry image is nI+s-nJ. With |s_i| <= 8 this
  // stronger ABI guard proves 2*M+8 < IMAGE_OFFSET, including before the
  // final old-query-domain filter. It cannot introduce an overflow flag
  // for an image that the general iterator would never visit. Larger
  // reference frames select the original general path, never refusal.
  constexpr int64_t safeFrame = (MDRT_TEMPLATE_IMAGE_OFFSET - 9) / 2;
  static_assert(safeFrame >= 0);
  valid = q.both(valid,
      q.cmp(arith::CmpFPredicate::OLE, maxAbsFrame, q.c(safeFrame)));

  SmallVector<Value, 6> h;
  Value sum = zero;
  for (int64_t i = 0; i != type.getNumElements(); ++i) {
    Value value = vector::ExtractOp::create(b, loc, cell, i);
    h.push_back(value);
    valid = q.both(valid, q.finite(value));
    sum = q.add(sum, q.abs(value));
  }
  Value least = h[0];
  for (int64_t i = 0; i != 3; ++i) {
    // This also makes multiplication of any diagonal by 1/2 exact and
    // normal, and its reciprocal finite, for the specialized image graph.
    valid = q.both(valid,
        q.cmp(arith::CmpFPredicate::OGT, h[i], q.c(2 * nu)));
    least = arith::SelectOp::create(b, loc,
        q.cmp(arith::CmpFPredicate::OLT, h[i], least), h[i], least);
  }
  if (type.getNumElements() == 6) {
    Value halfA = arith::MulFOp::create(b, loc, h[0], q.c(0.5));
    Value halfB = arith::MulFOp::create(b, loc, h[1], q.c(0.5));
    valid = q.both(valid, q.cmp(arith::CmpFPredicate::OLE, q.abs(h[3]), halfA));
    valid = q.both(valid, q.cmp(arith::CmpFPredicate::OLE, q.abs(h[4]), halfA));
    valid = q.both(valid, q.cmp(arith::CmpFPredicate::OLE, q.abs(h[5]), halfB));
  }
  // emitFaceWidths forms H^3 volume and H^4 cross-product norm terms.
  // Guard their noncancelling normal core, not just the final width: the
  // factor-eight upper slack covers at most six H^4 terms and rounding;
  // the factor-128 lower slack keeps (by*cz)^2 normal with room for minor
  // gradual-underflow terms. Small/huge cells retain the general iterator.
  Value h2 = arith::MulFOp::create(b, loc, sum, sum);
  Value h4 = arith::MulFOp::create(b, loc, h2, h2);
  Value d2 = arith::MulFOp::create(b, loc, least, least);
  Value d4 = arith::MulFOp::create(b, loc, d2, d2);
  for (Value value : {h2, h4, d2, d4})
    valid = q.both(valid, q.finite(value));
  valid = q.both(valid, q.cmp(arith::CmpFPredicate::OLE, h4,
      q.c(std::numeric_limits<double>::max() / 8)));
  valid = q.both(valid, q.cmp(arith::CmpFPredicate::OGE, d2, q.c(128 * nu)));
  valid = q.both(valid, q.cmp(arith::CmpFPredicate::OGE, d4, q.c(128 * nu)));
  valid = q.both(valid, q.finite(sum));
  valid = q.both(valid, q.finite(q.mul(maxReach, maxReach)));

  Value f = q.mul(three, maxAbsFrame);
  Value rho = q.c(std::nextafter(u + nu, std::numeric_limits<double>::infinity()));
  Value wrap = q.mul(three, rho);
  // Original P-(fl64(J-floor(J))+(n+s)), with |s_i| <= 8, S <= 24.
  // The integer sum n_i+s_i is exact64 when |n_i| <= 2^52.
  Value magnitude = q.add(q.add(q.mul(two, f), q.c(30)), wrap);
  Value rawError = q.add(wrap, q.add(
      q.mul(q.c(gamma(2, u)), magnitude),
      q.c(std::nextafter((6 + 3 * u) * nu, std::numeric_limits<double>::infinity()))));
  Value oldError = q.add(q.add(q.mul(sum, rawError),
      q.mul(q.c(gamma(3, u)), q.mul(sum, q.add(q.c(27), rawError)))),
      q.c(beta(9, u, nu)));

  // oldQ <= fl64(R*R) implies exact |D| <= R + oldError + normError.
  // sqrt((1+u)/(1-gamma3))-1 <= gamma8, and its absolute floor is
  // <= 2*sqrt(beta5), for u=2^-53. This includes radius-square rounding.
  Value normError = q.add(q.mul(q.c(gamma(8, u)), maxReach),
      q.mul(two, q.sqrt(q.c(beta(5, u, nu)))));

  // New wrapped-Cartesian c/b/a prefixes: a fixed integer leaf has at most
  // 64 rounded operations and weighted leaf magnitude <= h*(4F+64).
  // Doubling that budget covers fractional/quotient underflow scaled by H;
  // three coordinate absolute sums fit beta256. See the proof note for
  // the operation count and the induction proving all |s_i| <= 8.
  Value prefixError = q.add(q.mul(q.c(gamma(64, u)),
      q.mul(sum, q.add(q.c(128), q.mul(q.c(8), f)))),
      q.c(beta(256, u, nu)));
  Value totalError = q.add(q.add(oldError, normError), prefixError);

  // The caller may use ordinary RN64(classReach+margin). If E bounds
  // all preceding errors, 2*(E+u*maxReach+nu) ensures that addition is
  // >= classReach+E. Its upward value is also the selector's upper radius.
  Value margin = q.mul(two,
      q.add(q.add(totalError, q.mul(q.c(u), maxReach)), q.c(nu)));
  Value radius = q.add(maxReach, margin);
  valid = q.both(valid, q.both(q.finite(margin), q.finite(radius)));
  Value halfLeast = arith::MulFOp::create(b, loc, least, q.c(0.5));
  Value pass = q.both(valid, q.cmp(arith::CmpFPredicate::OLE, radius, halfLeast));
  Value neighbors = q.both(valid, q.cmp(arith::CmpFPredicate::OLT, radius, least));
  Value other = arith::SelectOp::create(b, loc, neighbors, q.integer(1), q.integer(2));
  Value path = arith::SelectOp::create(b, loc, pass, q.integer(0), other);
  return {path, margin};
}
