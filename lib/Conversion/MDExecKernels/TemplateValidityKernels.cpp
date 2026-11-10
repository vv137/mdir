#include "mdir/Conversion/TemplateValidityKernels.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include <array>
using namespace mlir;
using namespace mdir::kernels;

Value mdir::kernels::emitTemplateCellScaleLower(OpBuilder &b, Location loc,
                                               Value current, Value reference) {
  auto real = [&](double v) { return createReal(b, loc, b.getF64Type(), v); };
  auto at = [&](Value v, int64_t i) -> Value { return vector::ExtractOp::create(b, loc, v, i); };
  auto add = [&](Value a, Value c) -> Value { return arith::AddFOp::create(b, loc, a, c); };
  auto sub = [&](Value a, Value c) -> Value { return arith::SubFOp::create(b, loc, a, c); };
  auto mul = [&](Value a, Value c) -> Value { return arith::MulFOp::create(b, loc, a, c); };
  auto div = [&](Value a, Value c) -> Value { return arith::DivFOp::create(b, loc, a, c); };
  auto abs = [&](Value v) -> Value { return math::AbsFOp::create(b, loc, v); };
  auto max = [&](Value a, Value c) -> Value { return arith::MaximumFOp::create(b, loc, a, c); };
  Value one = real(1), zero = real(0), guard = real(0x1p-46);
  int64_t width = isTriclinic(current) ? 6 : 3;
  Value same = arith::ConstantOp::create(b, loc, b.getBoolAttr(true));
  for (int64_t i = 0; i != width; ++i)
    same = arith::AndIOp::create(b, loc, same,
        arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OEQ, at(current, i), at(reference, i)));
  Value lower;
  if (width == 3) {
    lower = div(at(current, 0), at(reference, 0));
    for (int64_t i = 1; i != 3; ++i)
      lower = arith::MinimumFOp::create(b, loc, lower, div(at(current, i), at(reference, i)));
    lower = mul(lower, sub(one, guard));
  } else {
    // With lattice vectors as columns, B = H_ref H_now^-1. Its upper triangle is found by triangular
    // substitution, keeping cancellation in the shear entries. Then
    // sigma_min(H_now H_ref^-1) >= 1/sqrt(||B||_1 ||B||_infinity).
    std::array<Value, 3> diagonal, error;
    for (unsigned i = 0; i != 3; ++i) {
      diagonal[i] = div(at(reference, i), at(current, i));
      error[i] = mul(guard, add(one, abs(diagonal[i])));
    }
    Value xy = div(sub(at(reference, 3), mul(diagonal[0], at(current, 3))), at(current, 1));
    Value yz = div(sub(at(reference, 5), mul(diagonal[1], at(current, 5))), at(current, 2));
    Value xz = div(sub(sub(at(reference, 4), mul(diagonal[0], at(current, 4))),
                       mul(xy, at(current, 5))), at(current, 2));
    // Positive error budgets include the magnitudes before cancellation
    // and propagated diagonal/shear error, not just the small residual.
    // 64 binary64 epsilons cover the short arithmetic chains, including
    // underflow by their absolute unit term; nonfinite budgets invalidate
    // the bound and hence the caller's ordered comparison.
    Value rxy = div(abs(at(current, 3)), at(current, 1));
    Value ryz = div(abs(at(current, 5)), at(current, 2));
    Value rxz = div(abs(at(current, 4)), at(current, 2));
    Value exy = add(mul(guard, add(one,
        add(div(abs(at(reference, 3)), at(current, 1)), mul(abs(diagonal[0]), rxy)))),
        mul(rxy, error[0]));
    Value eyz = add(mul(guard, add(one,
        add(div(abs(at(reference, 5)), at(current, 2)), mul(abs(diagonal[1]), ryz)))),
        mul(ryz, error[1]));
    Value exz = add(mul(guard, add(one,
        add(add(div(abs(at(reference, 4)), at(current, 2)), mul(abs(diagonal[0]), rxz)),
            mul(abs(xy), ryz)))), add(mul(rxz, error[0]), mul(ryz, exy)));
    Value a = add(abs(diagonal[0]), error[0]);
    Value d = add(abs(diagonal[1]), error[1]);
    Value f = add(abs(diagonal[2]), error[2]);
    Value cxy = add(abs(xy), exy), cxz = add(abs(xz), exz), cyz = add(abs(yz), eyz);
    Value norm1 = max(a, max(add(cxy, d), add(add(cxz, cyz), f)));
    Value normInfinity = max(add(add(a, cxy), cxz), max(add(d, cyz), f));
    norm1 = mul(norm1, add(one, guard));
    normInfinity = mul(normInfinity, add(one, guard));
    // Separate square roots avoid overflowing the norm product when the
    // inverse deformation itself is representable. A final downward
    // guard covers square-root, product and reciprocal rounding.
    Value denominator = mul(math::SqrtOp::create(b, loc, norm1),
                            math::SqrtOp::create(b, loc, normInfinity));
    lower = mul(div(one, denominator), sub(one, guard));
  }
  lower = max(zero, lower);
  return arith::SelectOp::create(b, loc, same, one, lower);
}

Value mdir::kernels::emitTemplateFreshReaches(OpBuilder &b, Location loc,
                                              Value physical, Value skin) {
  Value padding = vector::BroadcastOp::create(b, loc, physical.getType(), skin);
  return arith::AddFOp::create(b, loc, physical, padding);
}

std::array<Value, 2> mdir::kernels::emitTemplateGroupMotionLimits(
    OpBuilder &b, Location loc, Value current, ArrayRef<Value> classCells,
    Value built, Value physical, Value skin) {
  assert(classCells.size() == 3);
  Value fresh = emitTemplateFreshReaches(b, loc, physical, skin);
  std::array<Value, 3> margins;
  for (int64_t tag = 0; tag != 3; ++tag) {
    Value r = vector::ExtractOp::create(b, loc, physical, tag);
    Value oldReach = vector::ExtractOp::create(b, loc, built, tag);
    Value newReach = vector::ExtractOp::create(b, loc, fresh, tag);
    Value scale = emitTemplateCellScaleLower(b, loc, current, classCells[tag]);
    Value deformed = arith::SubFOp::create(b, loc,
        arith::MulFOp::create(b, loc, scale, oldReach), r);
    Value freshMargin = arith::SubFOp::create(b, loc, newReach, r);
    margins[tag] = arith::MinimumFOp::create(b, loc, deformed, freshMargin);
  }
  Value half = createReal(b, loc, b.getF64Type(), .5);
  return {arith::MulFOp::create(b, loc, half,
              arith::MinimumFOp::create(b, loc, margins[0], margins[1])),
          arith::MulFOp::create(b, loc, half,
              arith::MinimumFOp::create(b, loc, margins[1], margins[2]))};
}

Value mdir::kernels::emitTemplateMotionExceeded(OpBuilder &b, Location loc,
    Value anchor, Value reference, Value current, Value halfMargin,
    Value anchorL1, Value cellL1) {
  Value projected = emitLatticeShift(b, loc, reference, current);
  Value difference = arith::SubFOp::create(b, loc, anchor, projected);
  Value squares = arith::MulFOp::create(b, loc, difference, difference);
  Value distance2 = vector::ReductionOp::create(b, loc, vector::CombiningKind::ADD, squares);
  auto norm1 = [&](Value value) -> Value {
    return vector::ReductionOp::create(b, loc, vector::CombiningKind::ADD,
        math::AbsFOp::create(b, loc, value));
  };
  Value one = createReal(b, loc, b.getF64Type(), 1);
  Value zero = createReal(b, loc, b.getF64Type(), 0);
  Value guard = createReal(b, loc, b.getF64Type(), 0x1p-46);
  // The stored fractions define the reference used by the search. Bound
  // arithmetic error in projecting that point and subtracting the actual
  // framed anchor, rather than replacing its displacement by a minimum image.
  Value magnitude = arith::AddFOp::create(b, loc, one,
      arith::AddFOp::create(b, loc, anchorL1 ? anchorL1 : norm1(anchor),
          arith::MulFOp::create(b, loc, cellL1 ? cellL1 : norm1(current), norm1(reference))));
  Value error = arith::MulFOp::create(b, loc, guard, magnitude);
  Value budget = arith::SubFOp::create(b, loc, halfMargin, error);
  Value nonnegative = arith::MaximumFOp::create(b, loc, zero, budget);
  Value limit2 = arith::MulFOp::create(b, loc, nonnegative, nonnegative);
  limit2 = arith::MulFOp::create(b, loc, limit2, arith::SubFOp::create(b, loc, one, guard));
  distance2 = arith::MulFOp::create(b, loc, distance2, arith::AddFOp::create(b, loc, one, guard));
  return arith::OrIOp::create(b, loc,
      arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLT, budget, zero),
      arith::CmpFOp::create(b, loc, arith::CmpFPredicate::UGT, distance2, limit2));
}
