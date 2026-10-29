// The existing virtual-site geometry, constructed at its original loop site.
#include "SiteIR.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"

using namespace mlir;

Value mdir::driver::builder::buildSitePlacement(
    OpBuilder &builder, bool amber, Value toJ, Value toK, Value a, Value b) {
  auto loc = builder.getUnknownLoc();
  auto vec = VectorType::get({3}, builder.getF64Type());
  auto add = [&](Value x, Value y) -> Value {
    return arith::AddFOp::create(builder, loc, x, y);
  };
  auto div = [&](Value x, Value y) -> Value {
    return arith::DivFOp::create(builder, loc, x, y);
  };
  auto splat = [&](Value x) -> Value {
    return vector::BroadcastOp::create(builder, loc, vec, x);
  };
  auto scale = [&](Value factor, Value x) -> Value {
    Value broadcast = splat(factor);
    return arith::MulFOp::create(builder, loc, broadcast, x);
  };
  auto norm = [&](Value x) -> Value {
    Value square = arith::MulFOp::create(builder, loc, x, x);
    Value sum = vector::ReductionOp::create(
        builder, loc, vector::CombiningKind::ADD, square);
    return math::SqrtOp::create(builder, loc, sum);
  };
  if (amber) {
    Value lengthU = norm(toJ), lengthV = norm(toK);
    Value u = div(toJ, splat(lengthU)), v = div(toK, splat(lengthV));
    Value sum = add(u, v), lengthSum = norm(sum);
    return scale(div(a, lengthSum), sum);
  }
  return add(scale(a, toJ), scale(b, toK));
}

Value mdir::driver::builder::buildAddChange(OpBuilder &builder, Value positions,
                                            Value change) {
  auto loc = builder.getUnknownLoc();
  auto vec = VectorType::get({3}, builder.getF64Type());
  auto mapped = mdir::md::MapParticlesOp::create(
      builder, loc, positions.getType(), ValueRange{positions, change});
  {
    OpBuilder::InsertionGuard guard(builder);
    Block *block = builder.createBlock(&mapped.getKernel(), {},
                                        {vec, vec}, {loc, loc});
    Value sum = arith::AddFOp::create(builder, loc, block->getArgument(0),
                                      block->getArgument(1));
    mdir::md::YieldOp::create(builder, loc, ValueRange{sum});
  }
  return mapped.getResult();
}
