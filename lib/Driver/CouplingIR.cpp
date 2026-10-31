// Molecular pressure and rigid-center motion in their original shared regions.
#include "CouplingIR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "KineticIR.h"
#include "SiteIR.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace mlir;
using namespace mdir::driver::builder;

namespace {
class Kernel {
public:
  explicit Kernel(OpBuilder &builder)
      : builder(builder), loc(builder.getUnknownLoc()),
        vec(VectorType::get({3}, builder.getF64Type())) {}
  Value add(Value a, Value b) { return arith::AddFOp::create(builder, loc, a, b); }
  Value sub(Value a, Value b) { return arith::SubFOp::create(builder, loc, a, b); }
  Value mul(Value a, Value b) { return arith::MulFOp::create(builder, loc, a, b); }
  Value div(Value a, Value b) { return arith::DivFOp::create(builder, loc, a, b); }
  Value negate(Value a) { return arith::NegFOp::create(builder, loc, a); }
  Value broadcast(Value a) {
    return vector::BroadcastOp::create(builder, loc, vec, a);
  }
  Value scale(Value factor, Value a) { return mul(broadcast(factor), a); }
  Value vectorConstant(double value) {
    return arith::ConstantOp::create(builder, loc, DenseElementsAttr::get(vec, value));
  }
  Value constant(double value) {
    return arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(value));
  }

private:
  OpBuilder &builder;
  Location loc;
  VectorType vec;
};

Block *kernel(OpBuilder &builder, Region &region, unsigned vectors,
                unsigned scalars) {
  auto vec = VectorType::get({3}, builder.getF64Type());
  SmallVector<Type> types(vectors, vec);
  types.append(scalars, builder.getF64Type());
  SmallVector<Location> locations(types.size(), builder.getUnknownLoc());
  return builder.createBlock(&region, {}, types, locations);
}

void coordinates(OpBuilder &builder, unsigned count, DenseI32ArrayAttr &kinds,
                   DenseI64ArrayAttr &members) {
  SmallVector<mdir::md::Coordinate> coords;
  for (unsigned j = 1; j <= count; ++j)
    coords.push_back({mdir::md::CoordinateKind::Displacement, {j, 0}});
  mdir::md::getCoordinateAttrs(builder, coords, kinds, members);
}
} // namespace

Value mdir::driver::builder::buildMolecularDiagonal(
    OpBuilder &builder, const MolecularDiagonalInputs &in) {
  auto loc = builder.getUnknownLoc();
  auto vec = VectorType::get({3}, builder.getF64Type());
  Kernel k(builder);
  Value molecular = in.diagonal;
  for (const auto &group : in.groups) {
    unsigned count = group.arity - 1;
    DenseI32ArrayAttr kinds;
    DenseI64ArrayAttr members;
    coordinates(builder, count, kinds, members);
    auto sum = mdir::md::SumTuplesOp::create(
        builder, loc, vec, group.relation, in.positions, in.cell,
        ValueRange{in.forces, in.masses}, ValueRange{}, kinds, members);
    {
      OpBuilder::InsertionGuard guard(builder);
      auto args = kernel(builder, sum.getKernel(), count + group.arity,
                           group.arity)->getArguments();
      auto force = [&](unsigned j) { return args[count + j]; };
      auto mass = [&](unsigned j) { return args[count + group.arity + j]; };
      Value total = mass(0), moment = k.vectorConstant(0.0),
            resultant = force(0), places = k.vectorConstant(0.0);
      for (unsigned j = 1; j <= count; ++j) {
        total = k.add(total, mass(j));
        moment = k.add(moment, k.scale(mass(j), args[j - 1]));
        resultant = k.add(resultant, force(j));
        places = k.add(places, k.mul(args[j - 1], force(j)));
      }
      Value center = k.scale(k.div(k.constant(1.0), total), moment);
      Value about = k.sub(places, k.mul(center, resultant));
      mdir::md::YieldOp::create(builder, loc, ValueRange{about});
    }
    molecular = k.sub(molecular, sum.getResult());
  }
  return molecular;
}

Value mdir::driver::builder::buildGroupScaling(
    OpBuilder &builder, const GroupScalingInputs &in) {
  auto loc = builder.getUnknownLoc();
  Kernel k(builder);
  auto scaled = mdir::md::MapParticlesOp::create(
      builder, loc, in.positions.getType(), ValueRange{in.positions});
  {
    OpBuilder::InsertionGuard guard(builder);
    auto args = kernel(builder, scaled.getKernel(), 1, 0)->getArguments();
    Value value = k.mul(in.scale, args[0]);
    mdir::md::YieldOp::create(builder, loc, ValueRange{value});
  }
  Value positions = scaled.getResult();
  for (const auto &group : in.groups) {
    unsigned count = group.arity - 1;
    DenseI32ArrayAttr kinds;
    DenseI64ArrayAttr members;
    coordinates(builder, count, kinds, members);
    auto delta = mdir::md::GatherTuplesOp::create(
        builder, loc, in.positions.getType(), Type{}, group.relation,
        in.positions, in.cell, ValueRange{in.masses}, ValueRange{}, kinds, members);
    {
      OpBuilder::InsertionGuard guard(builder);
      auto args = kernel(builder, delta.getKernel(), count,
                           group.arity)->getArguments();
      auto mass = [&](unsigned j) { return args[count + j]; };
      Value total = mass(0), moment = k.vectorConstant(0.0);
      for (unsigned j = 1; j <= count; ++j) {
        total = k.add(total, mass(j));
        moment = k.add(moment, k.scale(mass(j), args[j - 1]));
      }
      Value one = k.constant(1.0);
      Value center = k.scale(k.div(one, total), moment);
      SmallVector<Value> results;
      for (unsigned j = 0; j <= count; ++j) {
        Value place = j == 0 ? k.negate(center) : k.sub(args[j - 1], center);
        results.push_back(k.mul(in.oneMinusScale, place));
      }
      mdir::md::YieldOp::create(builder, loc, results);
    }
    positions = buildAddChange(builder, positions, delta.getResult());
  }
  return positions;
}

GroupKineticResult mdir::driver::builder::buildGroupKinetic(
    OpBuilder &builder, const GroupKineticInputs &in) {
  auto loc = builder.getUnknownLoc();
  auto vec = VectorType::get({3}, builder.getF64Type());
  Kernel k(builder);
  Value kinetic = buildKineticVector(builder, in.velocities, in.masses);
  Value force = buildForceSquare(builder, in.forces, in.masses);
  for (const auto &group : in.groups) {
    for (int part = 0; part != 2; ++part) {
      DenseI32ArrayAttr kinds;
      DenseI64ArrayAttr members;
      // This otherwise unused coordinate retains the existing kernel signature.
      coordinates(builder, 1, kinds, members);
      auto sum = mdir::md::SumTuplesOp::create(
          builder, loc, vec, group.relation, in.positions, in.cell,
          ValueRange{part == 0 ? in.velocities : in.forces, in.masses},
          ValueRange{}, kinds, members);
      {
        OpBuilder::InsertionGuard guard(builder);
        auto args = kernel(builder, sum.getKernel(), 1 + group.arity,
                             group.arity)->getArguments();
        auto field = [&](unsigned j) { return args[1 + j]; };
        auto mass = [&](unsigned j) { return args[1 + group.arity + j]; };
        Value total = mass(0);
        for (unsigned j = 1; j != group.arity; ++j)
          total = k.add(total, mass(j));
        Value within;
        if (part == 0) {
          Value moment = k.scale(mass(0), field(0));
          for (unsigned j = 1; j != group.arity; ++j)
            moment = k.add(moment, k.scale(mass(j), field(j)));
          Value center = k.scale(k.div(k.constant(1.0), total), moment);
          for (unsigned j = 0; j != group.arity; ++j) {
            Value relative = k.sub(field(j), center);
            Value term = k.scale(mass(j), k.mul(relative, relative));
            within = within ? k.add(within, term) : term;
          }
        } else {
          Value resultant = field(0);
          for (unsigned j = 1; j != group.arity; ++j)
            resultant = k.add(resultant, field(j));
          for (unsigned j = 0; j != group.arity; ++j) {
            Value term = k.scale(k.div(k.constant(1.0), mass(j)),
                                  k.mul(field(j), field(j)));
            within = within ? k.add(within, term) : term;
          }
          Value whole = k.scale(k.div(k.constant(1.0), total),
                                 k.mul(resultant, resultant));
          within = k.sub(within, whole);
        }
        mdir::md::YieldOp::create(builder, loc, ValueRange{within});
      }
      if (part == 0) {
        Value half = k.vectorConstant(0.5);
        Value internal = k.mul(sum.getResult(), half);
        kinetic = k.sub(kinetic, internal);
      } else {
        force = k.sub(force, sum.getResult());
      }
    }
  }
  if (in.totalMass) {
    auto momentum = mdir::md::SumParticlesOp::create(
        builder, loc, vec, ValueRange{in.velocities, in.masses});
    {
      OpBuilder::InsertionGuard guard(builder);
      auto args = kernel(builder, momentum.getKernel(), 1, 1)->getArguments();
      Value broadcast = k.broadcast(args[1]);
      Value value = k.mul(broadcast, args[0]);
      mdir::md::YieldOp::create(builder, loc, ValueRange{value});
    }
    Value center = k.div(momentum.getResult(), in.totalMass);
    Value square = k.mul(momentum.getResult(), center);
    Value half = k.vectorConstant(0.5);
    Value com = k.mul(half, square);
    kinetic = k.sub(kinetic, com);
  }
  // The mean half-step center energy is K_g + dt^2 F_g^2/(8 M_g)
  // [Jung2020, eq. 42], after removing the internal group motion.
  Value dt2 = k.mul(in.timeStep, in.timeStep);
  Value coefficient = k.constant(0.125);
  Value scale = k.mul(dt2, coefficient);
  Value broadcast = k.broadcast(scale);
  Value forceTerm = k.mul(force, broadcast);
  Value half = k.add(kinetic, forceTerm);
  return {kinetic, forceTerm, half};
}

CenterMotionResult mdir::driver::builder::buildCenterMotion(
    OpBuilder &builder, const CenterMotionInputs &in) {
  Kernel k(builder);
  if (!in.totalMass)
    return {{}, {}, k.constant(0.0)};
  auto loc = builder.getUnknownLoc();
  Value momentum = buildMomentum(builder, in.velocities, in.masses);
  Value center = k.div(momentum, in.totalMass);
  Value product = k.mul(momentum, center);
  Value sum = vector::ReductionOp::create(builder, loc, vector::CombiningKind::ADD, product);
  Value energy = k.mul(in.half, sum);
  return {momentum, center, energy};
}

Value mdir::driver::builder::buildAxesWithoutCenter(
    OpBuilder &builder, Value kinetic, Value momentum, Value center, Value half) {
  Kernel k(builder);
  Value product = k.mul(momentum, center);
  Value energy = k.mul(half, product);
  return k.sub(kinetic, energy);
}

ThermostatIRResult mdir::driver::builder::buildThermostat(
    OpBuilder &builder, const ThermostatIRInputs &in, bool noseHoover) {
  auto loc = builder.getUnknownLoc();
  Kernel k(builder);
  Value kinetic = in.totalEnergy;
  if (in.centerEnergy)
    kinetic = k.sub(kinetic, in.centerEnergy);
  Value controlled = in.optimal ? in.optimal : kinetic;
  if (in.optimal && in.centerEnergy)
    controlled = k.sub(controlled, in.centerEnergy);
  Value factor;
  if (noseHoover) {
    factor = func::CallOp::create(builder, loc, "mdrtNoseHooverFactor",
        TypeRange{builder.getF64Type()}, ValueRange{in.step, controlled}).getResult(0);
  } else {
    factor = func::CallOp::create(builder, loc, "mdrtBussiFactor",
        TypeRange{builder.getF64Type()},
        ValueRange{in.seed, in.step, controlled, in.target, in.freedom, in.decay}).getResult(0);
  }
  Value squared = k.mul(factor, factor);
  Value after = k.mul(squared, kinetic);
  Value heat = k.sub(in.totalEnergy, after);
  return {factor, squared, heat};
}

Value mdir::driver::builder::buildCoupledVelocities(
    OpBuilder &builder, Value velocities, Value center, Value factor,
    Value inverse, Value bath) {
  auto loc = builder.getUnknownLoc();
  func::CallOp::create(builder, loc, "mdrtAddBath", TypeRange{}, ValueRange{bath});
  auto map = mdir::md::MapParticlesOp::create(builder, loc, velocities.getType(), ValueRange{velocities});
  {
    OpBuilder::InsertionGuard guard(builder);
    auto args = kernel(builder, map.getKernel(), 1, 0)->getArguments();
    Kernel k(builder);
    Value value = args[0];
    if (center)
      value = k.sub(value, center);
    if (factor) {
      Value broadcast = k.broadcast(factor);
      value = k.mul(broadcast, value);
    }
    if (inverse)
      value = k.mul(inverse, value);
    mdir::md::YieldOp::create(builder, loc, ValueRange{value});
  }
  return map.getResult();
}

CouplingMassIR mdir::driver::builder::buildCouplingMass(OpBuilder &builder, Value masses) {
  auto loc = builder.getUnknownLoc();
  Value half = arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(0.5));
  auto vec = VectorType::get({3}, builder.getF64Type());
  auto total = mdir::md::SumParticlesOp::create(builder, loc, vec, ValueRange{masses});
  {
    OpBuilder::InsertionGuard guard(builder);
    auto args = builder.createBlock(&total.getKernel(), {}, {builder.getF64Type()}, {loc})->getArguments();
    Value mass = vector::BroadcastOp::create(builder, loc, vec, args[0]);
    mdir::md::YieldOp::create(builder, loc, ValueRange{mass});
  }
  return {half, total.getResult()};
}
