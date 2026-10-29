// Preserve particle and arithmetic order in kinetic reductions [Jung2018].
#include "KineticIR.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace mdir::driver::builder;

namespace {
class Kernel {
public:
  explicit Kernel(OpBuilder &builder)
      : builder(builder), loc(builder.getUnknownLoc()),
        vec(VectorType::get({3}, builder.getF64Type())) {}
  Value constant(double value) {
    return arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(value));
  }
  Value mul(Value a, Value b) { return arith::MulFOp::create(builder, loc, a, b); }
  Value add(Value a, Value b) { return arith::AddFOp::create(builder, loc, a, b); }
  Value sub(Value a, Value b) { return arith::SubFOp::create(builder, loc, a, b); }
  Value div(Value a, Value b) { return arith::DivFOp::create(builder, loc, a, b); }
  Value broadcast(Value a) {
    return vector::BroadcastOp::create(builder, loc, vec, a);
  }
  Value reduce(Value a) {
    return vector::ReductionOp::create(builder, loc, vector::CombiningKind::ADD, a);
  }
  Block *block(Region &region) {
    return builder.createBlock(&region, {}, {vec, builder.getF64Type()}, {loc, loc});
  }

private:
  OpBuilder &builder;
  Location loc;
  VectorType vec;
};
} // namespace

Value mdir::driver::builder::buildKineticVector(OpBuilder &builder,
                                                Value velocities, Value masses) {
  auto loc = builder.getUnknownLoc();
  auto vec = VectorType::get({3}, builder.getF64Type());
  auto sum = mdir::md::SumParticlesOp::create(
      builder, loc, vec, ValueRange{velocities, masses});
  {
    OpBuilder::InsertionGuard guard(builder);
    Kernel k(builder);
    auto args = k.block(sum.getKernel())->getArguments();
    Value half = k.constant(0.5);
    Value mass = k.mul(half, args[1]);
    Value broadcast = k.broadcast(mass);
    Value square = k.mul(args[0], args[0]);
    Value energy = k.mul(broadcast, square);
    mdir::md::YieldOp::create(builder, loc, ValueRange{energy});
  }
  return sum.getResult();
}

Value mdir::driver::builder::buildKineticEnergy(OpBuilder &builder,
                                                Value velocities, Value masses) {
  auto loc = builder.getUnknownLoc();
  auto sum = mdir::md::SumParticlesOp::create(
      builder, loc, builder.getF64Type(), ValueRange{velocities, masses});
  {
    OpBuilder::InsertionGuard guard(builder);
    Kernel k(builder);
    auto args = k.block(sum.getKernel())->getArguments();
    Value half = k.constant(0.5);
    Value square = k.mul(args[0], args[0]);
    Value norm = k.reduce(square);
    Value weighted = k.mul(args[1], norm);
    Value energy = k.mul(half, weighted);
    mdir::md::YieldOp::create(builder, loc, ValueRange{energy});
  }
  return sum.getResult();
}

Value mdir::driver::builder::buildMomentum(OpBuilder &builder, Value velocities,
                                           Value masses) {
  auto loc = builder.getUnknownLoc();
  auto vec = VectorType::get({3}, builder.getF64Type());
  auto sum = mdir::md::SumParticlesOp::create(
      builder, loc, vec, ValueRange{velocities, masses});
  {
    OpBuilder::InsertionGuard guard(builder);
    Kernel k(builder);
    auto args = k.block(sum.getKernel())->getArguments();
    Value mass = k.broadcast(args[1]);
    Value moment = k.mul(mass, args[0]);
    mdir::md::YieldOp::create(builder, loc, ValueRange{moment});
  }
  return sum.getResult();
}

Value mdir::driver::builder::buildForceSquare(OpBuilder &builder, Value forces,
                                              Value masses) {
  auto loc = builder.getUnknownLoc();
  auto vec = VectorType::get({3}, builder.getF64Type());
  auto sum = mdir::md::SumParticlesOp::create(
      builder, loc, vec, ValueRange{forces, masses});
  {
    OpBuilder::InsertionGuard guard(builder);
    Kernel k(builder);
    auto args = k.block(sum.getKernel())->getArguments();
    Value square = k.mul(args[0], args[0]);
    Value zero = k.constant(0.0), one = k.constant(1.0);
    Value massless = arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::OEQ,
                                          args[1], zero);
    Value safe = arith::SelectOp::create(builder, loc, massless, one, args[1]);
    Value inverse = k.div(one, safe);
    Value weight = arith::SelectOp::create(builder, loc, massless, zero, inverse);
    Value broadcast = k.broadcast(weight);
    Value term = k.mul(square, broadcast);
    mdir::md::YieldOp::create(builder, loc, ValueRange{term});
  }
  return sum.getResult();
}

Value mdir::driver::builder::buildKineticExcess(OpBuilder &builder, Value forces,
                                                Value masses, Value dt) {
  Value sum = buildForceSquare(builder, forces, masses);
  Kernel k(builder);
  Value square = k.mul(dt, dt);
  Value eighth = k.constant(0.125);
  Value scale = k.mul(square, eighth);
  Value broadcast = k.broadcast(scale);
  return k.mul(sum, broadcast);
}

Value mdir::driver::builder::buildCorrectedHalfVelocity(
    OpBuilder &builder, Value kicked, Value corrected, Value drifted,
    Value interval) {
  auto loc = builder.getUnknownLoc();
  auto vec = VectorType::get({3}, builder.getF64Type());
  Kernel k(builder);
  Value one = k.constant(1.0);
  Value rate = k.div(one, interval);
  auto map = mdir::md::MapParticlesOp::create(
      builder, loc, kicked.getType(), ValueRange{kicked, corrected, drifted});
  {
    OpBuilder::InsertionGuard guard(builder);
    auto args = builder.createBlock(&map.getKernel(), {}, {vec, vec, vec},
                                      {loc, loc, loc})->getArguments();
    Value difference = k.sub(args[1], args[2]);
    Value broadcast = k.broadcast(rate);
    Value change = k.mul(broadcast, difference);
    Value result = k.add(args[0], change);
    mdir::md::YieldOp::create(builder, loc, ValueRange{result});
  }
  return map.getResult();
}

Value mdir::driver::builder::buildHalfStepKinetic(
    OpBuilder &builder, Value first, Value second, Value masses) {
  auto loc = builder.getUnknownLoc();
  auto f64 = builder.getF64Type();
  auto vec = VectorType::get({3}, f64);
  auto sum = mdir::md::SumParticlesOp::create(
      builder, loc, vec, ValueRange{first, second, masses});
  {
    OpBuilder::InsertionGuard guard(builder);
    Kernel k(builder);
    auto args = builder.createBlock(&sum.getKernel(), {}, {vec, vec, f64},
                                      {loc, loc, loc})->getArguments();
    Value quarter = k.constant(0.25);
    Value mass = k.mul(quarter, args[2]);
    Value broadcast = k.broadcast(mass);
    Value aa = k.mul(args[0], args[0]);
    Value bb = k.mul(args[1], args[1]);
    Value square = k.add(aa, bb);
    Value result = k.mul(broadcast, square);
    mdir::md::YieldOp::create(builder, loc, ValueRange{result});
  }
  return sum.getResult();
}

Value mdir::driver::builder::buildSolventKinetic(
    OpBuilder &builder, const SolventKineticInputs &in) {
  auto loc = builder.getUnknownLoc();
  auto f64 = builder.getF64Type();
  auto vec = VectorType::get({3}, f64);
  DenseI32ArrayAttr kinds;
  DenseI64ArrayAttr members;
  SmallVector<mdir::md::Coordinate> coords{
      {mdir::md::CoordinateKind::Displacement, {1, 0}}};
  mdir::md::getCoordinateAttrs(builder, coords, kinds, members);
  auto sum = mdir::md::SumTuplesOp::create(
      builder, loc, vec, in.relation, in.positions, in.cell,
      ValueRange{in.full, in.first, in.second, in.masses}, ValueRange{},
      kinds, members);
  {
    OpBuilder::InsertionGuard guard(builder);
    SmallVector<Type> types(10, vec);
    types.append(3, f64);
    SmallVector<Location> locations(types.size(), loc);
    auto args = builder.createBlock(&sum.getKernel(), {}, types, locations)
                    ->getArguments();
    Kernel k(builder);
    Value full, half;
    for (unsigned j = 0; j != 3; ++j) {
      Value ff = k.mul(args[1 + j], args[1 + j]);
      Value fs = k.reduce(ff);
      Value fm = k.mul(args[10 + j], fs);
      Value aa = k.mul(args[4 + j], args[4 + j]);
      Value bb = k.mul(args[7 + j], args[7 + j]);
      Value hh = k.add(aa, bb);
      Value hs = k.reduce(hh);
      Value hm = k.mul(args[10 + j], hs);
      if (j == 0) {
        full = fm;
        half = hm;
      } else {
        full = k.add(full, fm);
        half = k.add(half, hm);
      }
    }
    Value halfConstant = k.constant(0.5), quarter = k.constant(0.25);
    Value energy = k.mul(halfConstant, full);
    Value halfEnergy = k.mul(quarter, half);
    Value zero = k.constant(0.0);
    Value result = vector::FromElementsOp::create(
        builder, loc, vec, ValueRange{energy, halfEnergy, zero});
    mdir::md::YieldOp::create(builder, loc, ValueRange{result});
  }
  return sum.getResult();
}
