// Common PME arithmetic [Essmann1995].
#pragma once
#include "MemoryAccess.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/TypeUtilities.h"

namespace mdir::kernels::detail {
class PMECode : public MemoryAccess {
protected:
  PMECode(OpBuilder &builder, Type position, Type charge, Type force)
      : MemoryAccess(builder, builder.getUnknownLoc()), positionType(position),
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

  // Raise the in-place weights from order k-1 to k (Eq. 2.28).
  void raiseSpline(Value w, Value k, Value a, Value c0, Value c1, Value c2,
                   Value one) {
    Value km1 = arith::SubIOp::create(b, loc, k, c1);
    Value km2 = arith::SubIOp::create(b, loc, k, c2);
    Value km1i = arith::IndexCastOp::create(b, loc, i64Type, km1);
    Value km1f = arith::SIToFPOp::create(b, loc, realType, km1i);
    Value inverse = arith::DivFOp::create(b, loc, one, km1f);
    Value ki = arith::IndexCastOp::create(b, loc, i64Type, k);
    Value kf = arith::SIToFPOp::create(b, loc, realType, ki);
    Value last = load(a, {km2});
    Value dw = arith::MulFOp::create(b, loc, inverse, w);
    Value top = arith::MulFOp::create(b, loc, dw, last);
    store(top, a, {km1});
    scf::ForOp::create(
        b, loc, c1, km1, c1, ValueRange{},
        [&](OpBuilder &b, Location, Value j, ValueRange carried) {
          Value ji = arith::IndexCastOp::create(b, loc, i64Type, j);
          Value jf = arith::SIToFPOp::create(b, loc, realType, ji);
          Value at = arith::SubIOp::create(b, loc, km1, j);
          Value below = arith::SubIOp::create(b, loc, at, c1);
          Value lo = load(a, {below});
          Value hi = load(a, {at});
          Value wj = arith::AddFOp::create(b, loc, w, jf);
          Value kj = arith::SubFOp::create(b, loc, kf, jf);
          Value kjw = arith::SubFOp::create(b, loc, kj, w);
          Value t1 = arith::MulFOp::create(b, loc, wj, lo);
          Value t2 = arith::MulFOp::create(b, loc, kjw, hi);
          Value sum = arith::AddFOp::create(b, loc, t1, t2);
          Value value = arith::MulFOp::create(b, loc, inverse, sum);
          store(value, a, {at});
          scf::YieldOp::create(b, loc);
        });
    if (!c0)
      c0 = arith::ConstantIndexOp::create(b, loc, 0);
    Value first = load(a, {c0});
    Value omw = arith::SubFOp::create(b, loc, one, w);
    Value d0 = arith::MulFOp::create(b, loc, inverse, omw);
    Value bottom = arith::MulFOp::create(b, loc, d0, first);
    store(bottom, a, {c0});
  }
};
} // namespace mdir::kernels::detail
