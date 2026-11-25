// Emit the constrained splitting's cosine transform and k times its slope.
// The host and device emitters differ only in where they read the coefficients.
#pragma once

#include "mdir/Driver/ProlateSpectrum.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mdir::prolate {
template <class Load>
std::pair<mlir::Value, mlir::Value> emitPolynomialSpectrum(mlir::OpBuilder &b,
                                                   mlir::Location loc,
                                                   mlir::Value k, Load load) {
  using namespace mlir;
  Type t = k.getType();
  auto real = [&](double x) -> Value {
    return arith::ConstantOp::create(b, loc, b.getFloatAttr(t, x));
  };
  auto index = [&](int x) -> Value {
    return arith::ConstantIndexOp::create(b, loc, x);
  };
  auto add = [&](Value x, Value y) -> Value {
    return arith::AddFOp::create(b, loc, x, y);
  };
  auto sub = [&](Value x, Value y) -> Value {
    return arith::SubFOp::create(b, loc, x, y);
  };
  auto mul = [&](Value x, Value y) -> Value {
    return arith::MulFOp::create(b, loc, x, y);
  };
  auto div = [&](Value x, Value y) -> Value {
    return arith::DivFOp::create(b, loc, x, y);
  };
  Value edge = real(SpectrumTransition), one = real(1), two = real(2);
  Value low = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLT, k, edge);
  auto branch = scf::IfOp::create(b, loc, TypeRange{t, t}, low, true);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(branch.thenBlock());
    Value scaled = mul(k, real(SpectrumSamplesPerUnit));
    Value cell = arith::FPToSIOp::create(b, loc, b.getI32Type(), scaled);
    Value i = arith::IndexCastOp::create(b, loc, b.getIndexType(), cell);
    Value next = arith::AddIOp::create(b, loc, i, index(1));
    Value u = sub(scaled, arith::SIToFPOp::create(b, loc, t, cell));
    Value v = load(false, i), m = load(true, i);
    Value delta = sub(load(false, next), v), end = load(true, next);
    Value quadratic = sub(sub(mul(real(3), delta), mul(two, m)), end);
    Value cubic = sub(add(m, end), mul(two, delta));
    Value value = add(v, mul(u, add(m, mul(u, add(quadratic, mul(u, cubic))))));
    Value slope =
        add(m, mul(u, add(mul(two, quadratic), mul(real(3), mul(u, cubic)))));
    scf::YieldOp::create(b, loc, ValueRange{value, mul(scaled, slope)});
  }
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(branch.elseBlock());
    Value inv = div(one, k), ratio = mul(edge, inv), x = mul(ratio, ratio);
    Value zero = real(0);
    auto sum = scf::ForOp::create(
        b, loc, index(0), index(SpectrumBasis), index(1),
        ValueRange{zero, zero, zero, zero},
        [&](OpBuilder &body, Location, Value i, ValueRange args) {
          OpBuilder::InsertionGuard point(b);
          b.setInsertionPoint(body.getInsertionBlock(),
                              body.getInsertionPoint());
          Value j =
              arith::SubIOp::create(b, loc, index(SpectrumKnots + SpectrumBasis - 1), i);
          scf::YieldOp::create(b, loc,
                               ValueRange{add(load(false, j), mul(x, args[0])),
                                          add(load(true, j), mul(x, args[1])),
                                          add(args[0], mul(x, args[2])),
                                          add(args[1], mul(x, args[3]))});
        });
    Value a = mul(inv, sum.getResult(0)),
          bsin = mul(mul(inv, inv), sum.getResult(1));
    Value sin = math::SinOp::create(b, loc, k),
          cos = math::CosOp::create(b, loc, k);
    Value value = add(mul(sin, a), mul(cos, bsin));
    Value da = sub(zero, add(a, mul(mul(two, x), mul(inv, sum.getResult(2)))));
    Value db =
        sub(zero,
            mul(two, add(bsin, mul(x, mul(mul(inv, inv), sum.getResult(3))))));
    Value slope = add(mul(k, sub(mul(cos, a), mul(sin, bsin))),
                      add(mul(sin, da), mul(cos, db)));
    scf::YieldOp::create(b, loc, ValueRange{value, slope});
  }
  return {branch.getResult(0), branch.getResult(1)};
}
} // namespace mdir::prolate
