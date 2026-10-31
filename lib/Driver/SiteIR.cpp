// The existing virtual-site geometry, constructed at its original loop site.
#include "SiteIR.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MD/MDCoordinates.h"
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

Value mdir::driver::builder::buildPlacedSites(
    OpBuilder &builder, const SitePlacementInputs &input, bool amber) {
  SmallVector<Value> operands{input.relation, input.positions, input.cell, input.a, input.b};
  ValueRange values(operands);
  using namespace mlir;
  auto loc = builder.getUnknownLoc();
  auto vec = VectorType::get({3}, builder.getF64Type());
  DenseI32ArrayAttr kinds;
  DenseI64ArrayAttr members;
  mdir::md::getCoordinateAttrs(
      builder, {{mdir::md::CoordinateKind::Displacement, {2, 1}},
                {mdir::md::CoordinateKind::Displacement, {3, 1}},
                {mdir::md::CoordinateKind::Displacement, {0, 1}}},
      kinds, members);
  auto delta = mdir::md::GatherTuplesOp::create(
      builder, loc, values[1].getType(), Type{}, values[0], values[1],
      values[2], ValueRange{}, values.slice(3, 2), kinds, members);
  {
    OpBuilder::InsertionGuard guard(builder);
    SmallVector<Type> types = {vec, vec, vec, builder.getF64Type(),
                               builder.getF64Type()};
    SmallVector<Location> locations(types.size(), loc);
    Block *block = builder.createBlock(&delta.getKernel(), {}, types,
                                        locations);
    auto args = block->getArguments();
    Value placed = buildSitePlacement(builder, amber, args[0], args[1],
                                       args[3], args[4]);
    Value change = arith::SubFOp::create(builder, loc, placed, args[2]);
    Value zero = arith::ConstantOp::create(
        builder, loc, DenseElementsAttr::get(vec, 0.0));
    mdir::md::YieldOp::create(builder, loc,
                              ValueRange{change, zero, zero, zero});
  }
  return buildAddChange(builder, values[1], delta.getResult());
}
