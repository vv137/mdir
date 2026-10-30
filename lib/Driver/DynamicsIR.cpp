// Mass-metric descent with the existing scaled 16-norm and rejected trials.
#include "DynamicsIR.h"
#include "mdir/Dialect/Dyn/DynOps.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include <limits>
using namespace mlir;
using namespace mdir::driver::builder;
namespace {
class Kernel {
public:
  explicit Kernel(OpBuilder &builder)
      : builder(builder), loc(builder.getUnknownLoc()),
        vec(VectorType::get({3}, builder.getF64Type())) {}
  Value mul(Value a, Value b) { return arith::MulFOp::create(builder, loc, a, b); }
  Value div(Value a, Value b) { return arith::DivFOp::create(builder, loc, a, b); }
  Value sqrt(Value a) { return math::SqrtOp::create(builder, loc, a); }
  Value squareNorm(Value a) {
    Value square = mul(a, a);
    return vector::ReductionOp::create(builder, loc, vector::CombiningKind::ADD, square);
  }
  Value sum(Value gradient, Value mean = {}) {
    auto sum = mdir::md::SumParticlesOp::create(builder, loc, builder.getF64Type(), ValueRange{gradient});
    {
      OpBuilder::InsertionGuard guard(builder);
      auto block = builder.createBlock(&sum.getKernel(), {}, {vec}, {loc});
      Value result = squareNorm(block->getArgument(0));
      if (mean) {
        Value r2 = div(result, mean);
        Value r4 = mul(r2, r2), r8 = mul(r4, r4);
        result = mul(r8, r8);
      }
      mdir::md::YieldOp::create(builder, loc, ValueRange{result});
    }
    return sum.getResult();
  }
private:
  OpBuilder &builder;
  Location loc;
  VectorType vec;
};
} // namespace
Value mdir::driver::builder::buildDescentPositions(
    OpBuilder &builder, Value positions, Value gradient, Value distance,
    Value count, Value tiny) {
  auto loc = builder.getUnknownLoc();
  auto vec = VectorType::get({3}, builder.getF64Type());
  Kernel k(builder);
  Value square = k.sum(gradient);
  Value mean = k.div(square, count);
  Value safe = arith::MaximumFOp::create(builder, loc, mean, tiny);
  Value rms = k.sqrt(safe);
  Value power = k.sum(gradient, safe);
  Value root2 = k.sqrt(power), root4 = k.sqrt(root2);
  Value root8 = k.sqrt(root4), root16 = k.sqrt(root8);
  Value norm = k.mul(rms, root16);
  Value scale = k.div(distance, norm);
  auto map = mdir::md::MapParticlesOp::create(builder, loc, positions.getType(), ValueRange{positions, gradient});
  {
    OpBuilder::InsertionGuard guard(builder);
    auto args = builder.createBlock(&map.getKernel(), {}, {vec, vec}, {loc, loc})->getArguments();
    Value broadcast = vector::BroadcastOp::create(builder, loc, vec, scale);
    Value change = k.mul(broadcast, args[1]);
    Value moved = arith::AddFOp::create(builder, loc, args[0], change);
    mdir::md::YieldOp::create(builder, loc, ValueRange{moved});
  }
  return map.getResult();
}
ConstrainedTrialIR mdir::driver::builder::buildConstrainedTrial(
    OpBuilder &builder, Value candidate, Value previous, Value broken, Value zero) {
  auto loc = builder.getUnknownLoc();
  auto vec = VectorType::get({3}, builder.getF64Type());
  Value intact = arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::OEQ, broken, zero);
  auto map = mdir::md::MapParticlesOp::create(builder, loc, candidate.getType(), ValueRange{candidate, previous});
  {
    OpBuilder::InsertionGuard guard(builder);
    auto args = builder.createBlock(&map.getKernel(), {}, {vec, vec}, {loc, loc})->getArguments();
    Value localZero = arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(0.0));
    Value test = arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::OEQ, broken, localZero);
    Value chosen = arith::SelectOp::create(builder, loc, test, args[0], args[1]);
    mdir::md::YieldOp::create(builder, loc, ValueRange{chosen});
  }
  return {map.getResult(), intact};
}
Value mdir::driver::builder::buildTrialEnergy(OpBuilder &builder, Value energy, Value intact) {
  auto loc = builder.getUnknownLoc();
  Value never = arith::ConstantOp::create(builder, loc,
      builder.getF64FloatAttr(std::numeric_limits<double>::infinity()));
  return arith::SelectOp::create(builder, loc, intact, energy, never);
}

Value mdir::driver::builder::buildKick(
    OpBuilder &builder, Value velocities, Value forces, Value masses, Value interval) {
  return mdir::dyn::KickOp::create(builder, builder.getUnknownLoc(), velocities.getType(),
                                    velocities, forces, masses, interval);
}
Value mdir::driver::builder::buildDrift(
    OpBuilder &builder, Value positions, Value velocities, Value interval) {
  return mdir::dyn::DriftOp::create(builder, builder.getUnknownLoc(), positions.getType(),
                                     positions, velocities, interval);
}
DriftScaleIR mdir::driver::builder::buildDriftScale(OpBuilder &builder, Value scale) {
  auto loc = builder.getUnknownLoc();
  auto vec = VectorType::get({3}, builder.getF64Type());
  Value unit = arith::ConstantOp::create(builder, loc, DenseElementsAttr::get(vec, 1.0));
  Value oneMinus = arith::SubFOp::create(builder, loc, unit, scale);
  Value inverse = arith::DivFOp::create(builder, loc, unit, scale);
  return {oneMinus, inverse};
}
Value mdir::driver::builder::buildVelocityScaling(OpBuilder &builder, Value velocities, Value scale) {
  auto loc = builder.getUnknownLoc();
  auto vec = VectorType::get({3}, builder.getF64Type());
  auto map = mdir::md::MapParticlesOp::create(builder, loc, velocities.getType(), ValueRange{velocities});
  {
    OpBuilder::InsertionGuard guard(builder);
    auto args = builder.createBlock(&map.getKernel(), {}, {vec}, {loc})->getArguments();
    Value result = arith::MulFOp::create(builder, loc, scale, args[0]);
    mdir::md::YieldOp::create(builder, loc, ValueRange{result});
  }
  return map.getResult();
}
Value mdir::driver::builder::buildFieldDifference(OpBuilder &builder, Value first, Value second) {
  auto loc = builder.getUnknownLoc();
  auto vec = VectorType::get({3}, builder.getF64Type());
  auto map = mdir::md::MapParticlesOp::create(builder, loc, first.getType(), ValueRange{first, second});
  {
    OpBuilder::InsertionGuard guard(builder);
    auto args = builder.createBlock(&map.getKernel(), {}, {vec, vec}, {loc, loc})->getArguments();
    Value result = arith::SubFOp::create(builder, loc, args[0], args[1]);
    mdir::md::YieldOp::create(builder, loc, ValueRange{result});
  }
  return map.getResult();
}
