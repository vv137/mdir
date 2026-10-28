// Shared scalar Lennard-Jones IR for primary and special 1-4 pairs.
#include "BuilderImpl.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
using namespace mdir::driver;
using namespace mdir::driver::builder;
using namespace mlir;

// 4ε (σ¹² Φ₁₂(r) − σ⁶ Φ₆(r)), with Φₙ = r⁻ⁿ truncated as `modifier`
// says, with rc the cutoff and rs the switching distance:
//
//   none             r⁻ⁿ
//   potential shift  r⁻ⁿ − rc⁻ⁿ
//   force switch     r⁻ⁿ − Cₙ − [r > rs] (Aₙ/3 (r − rs)³ + Bₙ/4 (r − rs)⁴),
//                    a cubic polynomial added to the force, so that it and
//                    its derivative vanish at rc [GromacsManual2025]
//   power force      r⁻ⁿ − (rs rc)^(−n/2)               for r <= rs
//   switch           kₙ (r^(−n/2) − rc^(−n/2))²          otherwise,
//                    kₙ = rc^(n/2) / (rc^(n/2) − rs^(n/2)): the force of
//                    each power times a switch linear in r^(n/2)
//                    [Steinbach1994]
//
// σⁿ Φₙ is written with s = σ/r and g = σ, so that no power of r alone
// leaves the range of f32.
mlir::Value mdir::driver::builder::buildLennardJonesIR(OpBuilder &builder,
                                                       ValueRange operands,
                                                       double cutoff,
                                                       double from,
                                                       Truncation modifier) {
  Value sigma = operands[0], epsilon = operands[1], distance = operands[2];
  Value physicalDistance = operands[3];
  auto loc = builder.getUnknownLoc();
  auto constant = [&](double value) -> Value {
    return arith::ConstantOp::create(builder, loc,
                                     builder.getF64FloatAttr(value));
  };
  auto integer = [&](int value) -> Value {
    return arith::ConstantOp::create(builder, loc,
                                     builder.getI32IntegerAttr(value));
  };
  auto add = [&](Value a, Value b) -> Value {
    return arith::AddFOp::create(builder, loc, a, b);
  };
  auto sub = [&](Value a, Value b) -> Value {
    return arith::SubFOp::create(builder, loc, a, b);
  };
  auto mul = [&](Value a, Value b) -> Value {
    return arith::MulFOp::create(builder, loc, a, b);
  };
  auto div = [&](Value a, Value b) -> Value {
    return arith::DivFOp::create(builder, loc, a, b);
  };
  auto power = [&](Value a, Value n) -> Value {
    return math::FPowIOp::create(builder, loc, a, n);
  };
  auto compare = [&](arith::CmpFPredicate predicate, Value a,
                     Value b) -> Value {
    return arith::CmpFOp::create(builder, loc, predicate, a, b);
  };
  auto select = [&](Value condition, Value yes, Value no) -> Value {
    return arith::SelectOp::create(builder, loc, condition, yes, no);
  };
  Value c4 = constant(4.0);
  Value e4 = mul(c4, epsilon);
  Value sr = div(sigma, distance);
  if (modifier == Truncation::None ||
      modifier == Truncation::SquaredDistanceSwitch) {
    Value i6 = integer(6);
    Value s6 = power(sr, i6);
    Value s12 = mul(s6, s6);
    Value t = sub(s12, s6);
    Value raw = mul(e4, t);
    if (modifier != Truncation::SquaredDistanceSwitch)
      return raw;
    // The cubic switching potential in r² [Brooks1983]. Preserve the
    // clamped (1-y)²(1+2y) operation sequence used by differentiation.
    Value rs2 = constant(from * from);
    Value widthInv = constant(1.0 / (cutoff * cutoff - from * from));
    Value zero = constant(0.0);
    Value one = constant(1.0);
    Value two = constant(2.0);
    Value r2 = mul(physicalDistance, physicalDistance);
    Value dr2 = sub(r2, rs2);
    Value yRaw = mul(dr2, widthInv);
    Value below = compare(arith::CmpFPredicate::OLT, yRaw, zero);
    Value yLo = select(below, zero, yRaw);
    Value above = compare(arith::CmpFPredicate::OGT, yLo, one);
    Value y = select(above, one, yLo);
    Value complement = sub(one, y);
    Value square = mul(complement, complement);
    Value twice = mul(two, y);
    Value factor = add(one, twice);
    Value switching = mul(square, factor);
    return mul(raw, switching);
  }
  Value i3 = integer(3);
  Value s3 = power(sr, i3);
  Value s6 = mul(s3, s3);
  Value s12 = mul(s6, s6);
  Value g3 = power(sigma, i3);
  Value g6 = mul(g3, g3);
  Value g12 = mul(g6, g6);
  auto subtract = [&](Value c12, Value c6) {
    Value gc12 = mul(g12, c12);
    Value a = sub(s12, gc12);
    Value gc6 = mul(g6, c6);
    Value b = sub(s6, gc6);
    return std::pair<Value, Value>{a, b};
  };
  Value a, b;
  if (modifier == Truncation::Shift) {
    Value rc12 = constant(std::pow(cutoff, -12.0));
    Value rc6 = constant(std::pow(cutoff, -6.0));
    std::tie(a, b) = subtract(rc12, rc6);
  } else if (modifier == Truncation::ForceSwitch) {
    double width = cutoff - from;
    Value rs = constant(from);
    Value zero = constant(0.0);
    Value dr = sub(distance, rs);
    Value beyond = compare(arith::CmpFPredicate::OGT, distance, rs);
    Value u = select(beyond, dr, zero);
    Value u2 = mul(u, u);
    Value u3 = mul(u2, u);
    Value p12, p6;
    for (int n : {12, 6}) {
      double an = -n * ((n + 4) * cutoff - (n + 1) * from) /
                  (std::pow(cutoff, n + 2) * width * width);
      double bn = n * ((n + 3) * cutoff - (n + 1) * from) /
                  (std::pow(cutoff, n + 2) * width * width * width);
      double cn = std::pow(cutoff, -n) - an / 3.0 * std::pow(width, 3) -
                  bn / 4.0 * std::pow(width, 4);
      Value an3 = constant(an / 3.0);
      Value bn4 = constant(bn / 4.0);
      Value c = constant(cn);
      Value pb = mul(bn4, u);
      Value pa = add(an3, pb);
      Value pu = mul(u3, pa);
      Value p = add(c, pu);
      (n == 12 ? p12 : p6) = p;
    }
    std::tie(a, b) = subtract(p12, p6);
  } else {
    Value in12 = constant(std::pow(from * cutoff, -6.0));
    Value in6 = constant(std::pow(from * cutoff, -3.0));
    std::tie(a, b) = subtract(in12, in6);
    Value k12 = constant(std::pow(cutoff, 6.0) /
                         (std::pow(cutoff, 6.0) - std::pow(from, 6.0)));
    Value k6 = constant(std::pow(cutoff, 3.0) /
                        (std::pow(cutoff, 3.0) - std::pow(from, 3.0)));
    Value rc6 = constant(std::pow(cutoff, -6.0));
    Value rc3 = constant(std::pow(cutoff, -3.0));
    Value gr6 = mul(g6, rc6);
    Value h6 = sub(s6, gr6);
    Value h6s = mul(h6, h6);
    Value aOut = mul(k12, h6s);
    Value gr3 = mul(g3, rc3);
    Value h3 = sub(s3, gr3);
    Value h3s = mul(h3, h3);
    Value bOut = mul(k6, h3s);
    Value rs = constant(from);
    Value inside = compare(arith::CmpFPredicate::OLE, distance, rs);
    a = select(inside, a, aOut);
    b = select(inside, b, bOut);
  }
  Value t = sub(a, b);
  return mul(e4, t);
}

void mdir::driver::builder::emitLennardJones(TermEmission &context,
                                             const Control &control,
                                             StringRef sigma, StringRef epsilon,
                                             StringRef result,
                                             Truncation modifier,
                                             StringRef distance) {
  double cutoff = control.cutoffDistance * units::length;
  double from = control.switchDistance * units::length;
  context.emitIRFragment(
      "    ", {result}, {"f64"}, {sigma, epsilon, distance, "%r"},
      {"f64", "f64", "f64", "f64"},
      [cutoff, from, modifier](OpBuilder &builder,
                               ValueRange operands) -> TermIRValues {
        return {buildLennardJonesIR(builder, operands, cutoff, from, modifier)};
      });
}
