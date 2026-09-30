// Approximations of divisions and erfc in the f32 kernels of loops, under
// fast_math.
//
// See the description of the pass in Passes.td.

#include "mdir/Dialect/MDExec/Transforms/Passes.h"

#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"

#include <cmath>

using namespace mlir;
using namespace mdir;
using namespace mdir::md_exec;

namespace mdir {
namespace md_exec {

#define GEN_PASS_DEF_APPROXIMATE
#include "mdir/Dialect/MDExec/Transforms/Passes.h.inc"

namespace {
/// The coefficients of P, from the constant term up (scripts/fit-erfc.py).
constexpr double kErfcPolynomial[] = {
    -5.844017783662725e-05,
    0.28321992286605291,
    0.27292130542277271,
    0.28757948962599933,
    0.072615343657635112,
    0.21878733123987557,
    -0.021014074929707336,
    -0.25983392190712945,
    0.18656882080781992,
    -0.04078578921930151,
};

/// The value of the f32 constant `value`, if it is one.
std::optional<double> getConstant(Value value) {
  FloatAttr attr;
  if (!value.getType().isF32() || !matchPattern(value, m_Constant(&attr)))
    return std::nullopt;
  return attr.getValueAsDouble();
}

/// Moves `op` and the ops in `block` that it uses before `point`, if it is
/// after it, so that its result can be used there. Only a product of a
/// value that dominates `point` and a constant, and an exponential of it,
/// are moved.
void moveBefore(Operation *op, Operation *point) {
  if (op->getBlock() != point->getBlock() || op->isBeforeInBlock(point))
    return;
  for (Value operand : op->getOperands())
    if (Operation *def = operand.getDefiningOp())
      moveBefore(def, point);
  op->moveBefore(point);
}

/// Returns `exp(y * k)` in the block of `erfc`, with `k` within 4 ulp of
/// `minusC2`, if the block has one.
Value findExponential(Value y, double minusC2, math::ErfcOp erfc) {
  for (Operation *user : y.getUsers()) {
    auto product = dyn_cast<arith::MulFOp>(user);
    if (!product || product->getBlock() != erfc->getBlock())
      continue;
    Value other =
        product.getLhs() == y ? product.getRhs() : product.getLhs();
    std::optional<double> k = getConstant(other);
    if (!k || std::abs(*k - minusC2) >
                  4.0 * std::ldexp(std::abs(minusC2), -23))
      continue;
    for (Operation *use : product->getUsers())
      if (auto exp = dyn_cast<math::ExpOp>(use))
        if (exp->getBlock() == erfc->getBlock()) {
          moveBefore(exp, erfc);
          return exp.getResult();
        }
  }
  return Value();
}

/// Rewrites `erfc` if its argument is sqrt(y) * c with c > 0.
void approximate(math::ErfcOp erfc) {
  Value x = erfc.getOperand();
  auto product = x.getDefiningOp<arith::MulFOp>();
  if (!x.getType().isF32() || !product)
    return;
  Value root = product.getLhs();
  std::optional<double> c = getConstant(product.getRhs());
  if (!c) {
    root = product.getRhs();
    c = getConstant(product.getLhs());
  }
  auto sqrt = root.getDefiningOp<math::SqrtOp>();
  if (!c || !(*c > 0.0) || !sqrt)
    return;
  Value y = sqrt.getOperand();

  OpBuilder builder(erfc);
  Location loc = erfc.getLoc();
  Type f32 = builder.getF32Type();
  auto constant = [&](double value) -> Value {
    return arith::ConstantOp::create(builder, loc, f32,
                                     builder.getF32FloatAttr(value));
  };
  double minusC2 = -(*c) * (*c);
  Value exponential = findExponential(y, minusC2, erfc);
  auto approximately = arith::FastMathFlagsAttr::get(
      builder.getContext(), arith::FastMathFlags::afn);
  if (!exponential)
    exponential = math::ExpOp::create(
        builder, loc,
        arith::MulFOp::create(builder, loc, y, constant(minusC2)),
        approximately);
  Value denominator = arith::AddFOp::create(
      builder, loc,
      arith::MulFOp::create(builder, loc, x, constant(0.5)), constant(1.0));
  Value t = arith::DivFOp::create(builder, loc, constant(1.0), denominator,
                                  approximately);
  constexpr int degree = std::size(kErfcPolynomial) - 1;
  Value p = constant(kErfcPolynomial[degree]);
  for (int k = degree - 1; k >= 0; --k)
    p = math::FmaOp::create(builder, loc, p, t,
                            constant(kErfcPolynomial[k]));
  Value result = arith::MulFOp::create(builder, loc, exponential, p);
  erfc.replaceAllUsesWith(result);
  erfc.erase();
}

class Approximate : public impl::ApproximateBase<Approximate> {
public:
  void runOnOperation() final {
    SmallVector<Region *> kernels;
    getOperation()->walk([&](Operation *op) {
      if (isa<PairForOp, TupleForOp, ParticleForOp>(op))
        kernels.push_back(&op->getRegion(0));
    });
    auto isF32 = [](Type type) {
      return getElementTypeOrSelf(type).isF32();
    };
    for (Region *kernel : kernels) {
      SmallVector<math::ErfcOp> erfcs;
      kernel->walk([&](Operation *op) {
        if (auto erfc = dyn_cast<math::ErfcOp>(op))
          erfcs.push_back(erfc);
        if (auto exp = dyn_cast<math::ExpOp>(op)) {
          if (isF32(exp.getType()))
            exp.setFastmath(exp.getFastmath() | arith::FastMathFlags::afn);
          return;
        }
        auto divide = dyn_cast<arith::DivFOp>(op);
        if (!divide || !isF32(divide.getType()))
          return;
        divide.setFastmath(divide.getFastmath() | arith::FastMathFlags::afn);
      });
      for (math::ErfcOp erfc : erfcs)
        approximate(erfc);
    }
  }
};
} // namespace

} // namespace md_exec
} // namespace mdir
