// Constraint virial reductions with the original coordinates and arithmetic.
#include "ConstraintIR.h"
#include "SiteIR.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include <array>
#include <vector>

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
  Value dot(Value a, Value b) {
    Value product = mul(a, b);
    return vector::ReductionOp::create(builder, loc, vector::CombiningKind::ADD,
                                        product);
  }
  Value scale(Value factor, Value a) {
    Value broadcast = vector::BroadcastOp::create(builder, loc, vec, factor);
    return mul(broadcast, a);
  }
  Value norm(Value a) {
    Value square = dot(a, a);
    return math::SqrtOp::create(builder, loc, square);
  }
  Value unit(Value a) {
    Value length = norm(a);
    Value broadcast = vector::BroadcastOp::create(builder, loc, vec, length);
    return div(a, broadcast);
  }
  Value zero() {
    return arith::ConstantOp::create(builder, loc, DenseElementsAttr::get(vec, 0.0));
  }
  Value component(Value a, int index) {
    return vector::ExtractOp::create(builder, loc, a, index);
  }
  Value constant(double value) {
    return arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(value));
  }

private:
  OpBuilder &builder;
  Location loc;
  VectorType vec;
};

// Gaussian elimination without pivoting for the small, diagonally dominated
// constraint systems, retaining their row and back-substitution order.
std::vector<Value> smallSolve(Kernel &k, std::vector<std::vector<Value>> matrix,
                                std::vector<Value> rhs) {
  unsigned count = rhs.size();
  for (unsigned c = 0; c != count; ++c)
    for (unsigned r = c + 1; r != count; ++r) {
      Value factor = k.div(matrix[r][c], matrix[c][c]);
      for (unsigned d = c; d != count; ++d)
        matrix[r][d] = k.sub(matrix[r][d], k.mul(factor, matrix[c][d]));
      rhs[r] = k.sub(rhs[r], k.mul(factor, rhs[c]));
    }
  std::vector<Value> result(count);
  for (unsigned c = count; c-- != 0;) {
    Value sum = rhs[c];
    for (unsigned d = c + 1; d != count; ++d)
      sum = k.sub(sum, k.mul(matrix[c][d], result[d]));
    result[c] = k.div(sum, matrix[c][c]);
  }
  return result;
}

void coordinates(OpBuilder &builder, unsigned arity, DenseI32ArrayAttr &kinds,
                   DenseI64ArrayAttr &members) {
  SmallVector<mdir::md::Coordinate> values;
  for (unsigned j = 1; j != arity; ++j)
    values.push_back({mdir::md::CoordinateKind::Displacement, {j, 0}});
  mdir::md::getCoordinateAttrs(builder, values, kinds, members);
}

Block *kernel(OpBuilder &builder, Region &region, unsigned vectors,
                unsigned scalars) {
  auto vec = VectorType::get({3}, builder.getF64Type());
  SmallVector<Type> types(vectors, vec);
  types.append(scalars, builder.getF64Type());
  SmallVector<Location> locations(types.size(), builder.getUnknownLoc());
  return builder.createBlock(&region, {}, types, locations);
}
} // namespace

Value mdir::driver::builder::buildConstraintVirial(
    OpBuilder &builder, const ConstraintVirialInputs &in, unsigned arity) {
  auto loc = builder.getUnknownLoc();
  auto vec9 = VectorType::get({9}, builder.getF64Type());
  DenseI32ArrayAttr kinds;
  DenseI64ArrayAttr members;
  coordinates(builder, arity, kinds, members);
  auto sum = mdir::md::SumTuplesOp::create(
      builder, loc, vec9, in.relation, in.positions, in.cell,
      ValueRange{in.change, in.masses}, ValueRange{}, kinds, members);
  Kernel k(builder);
  {
    OpBuilder::InsertionGuard guard(builder);
    unsigned count = arity - 1;
    auto args = kernel(builder, sum.getKernel(), count + arity, arity)->getArguments();
    auto change = [&](unsigned j) { return args[count + j]; };
    auto mass = [&](unsigned j) { return args[count + arity + j]; };
    Value rate = k.div(k.constant(1.0), k.mul(in.timeStep, in.timeStep));
    SmallVector<Value> forces;
    for (unsigned j = 1; j <= count; ++j)
      forces.push_back(k.scale(k.mul(mass(j), rate), change(j)));
    SmallVector<Value> elements;
    for (int a = 0; a != 3; ++a) {
      Value row;
      for (unsigned j = 1; j <= count; ++j) {
        Value term = k.scale(k.component(args[j - 1], a), forces[j - 1]);
        row = row ? k.add(row, term) : term;
      }
      for (int b = 0; b != 3; ++b)
        elements.push_back(k.component(row, b));
    }
    Value virial = vector::FromElementsOp::create(builder, loc, vec9, elements);
    mdir::md::YieldOp::create(builder, loc, ValueRange{virial});
  }
  return k.add(in.virial, sum.getResult());
}

Value mdir::driver::builder::buildConstraintTrace(
    OpBuilder &builder, const ConstraintTraceInputs &in, unsigned arity,
    bool axes) {
  auto loc = builder.getUnknownLoc();
  DenseI32ArrayAttr kinds;
  DenseI64ArrayAttr members;
  coordinates(builder, arity, kinds, members);
  auto sum = mdir::md::SumTuplesOp::create(
      builder, loc, in.trace.getType(), in.relation, in.positions, in.cell,
      ValueRange{in.descent, in.forces, in.velocities, in.masses}, ValueRange{},
      kinds, members);
  Kernel k(builder);
  {
    OpBuilder::InsertionGuard guard(builder);
    unsigned count = arity - 1;
    auto args = kernel(builder, sum.getKernel(), count + 3 * arity,
                         arity)->getArguments();
    auto descent = [&](unsigned j) { return args[count + j]; };
    auto force = [&](unsigned j) { return args[count + arity + j]; };
    auto velocity = [&](unsigned j) { return args[count + 2 * arity + j]; };
    auto mass = [&](unsigned j) { return args[count + 3 * arity + j]; };
    Value arms;
    for (unsigned j = 1; j <= count; ++j) {
      Value constrained = k.sub(k.scale(mass(j), descent(j)), force(j));
      Value term = axes ? k.mul(args[j - 1], constrained)
                        : k.dot(args[j - 1], constrained);
      arms = arms ? k.add(arms, term) : term;
    }
    Value total = mass(0), moment = k.scale(mass(0), velocity(0));
    for (unsigned j = 1; j <= count; ++j) {
      total = k.add(total, mass(j));
      moment = k.add(moment, k.scale(mass(j), velocity(j)));
    }
    Value center = k.scale(k.div(k.constant(1.0), total), moment);
    Value twice;
    for (unsigned j = 0; j <= count; ++j) {
      Value relative = k.sub(velocity(j), center);
      Value term = axes ? k.scale(mass(j), k.mul(relative, relative))
                        : k.mul(mass(j), k.dot(relative, relative));
      twice = twice ? k.add(twice, term) : term;
    }
    Value virial = k.sub(arms, twice);
    mdir::md::YieldOp::create(builder, loc, ValueRange{virial});
  }
  return k.add(in.trace, sum.getResult());
}

ConstraintVelocityResult mdir::driver::builder::buildShakeVelocities(
    OpBuilder &builder, const ConstraintVelocityInputs &in, unsigned arity) {
  auto loc = builder.getUnknownLoc();
  auto vec9 = VectorType::get({9}, builder.getF64Type());
  unsigned count = arity - 1;
  DenseI32ArrayAttr kinds;
  DenseI64ArrayAttr members;
  coordinates(builder, arity, kinds, members);
  Kernel k(builder);
  auto change = mdir::md::GatherTuplesOp::create(
      builder, loc, in.velocities.getType(), Type{}, in.relation, in.positions,
      in.cell, ValueRange{in.velocities, in.masses}, in.parameters, kinds, members);
  {
    OpBuilder::InsertionGuard guard(builder);
    auto args = kernel(builder, change.getKernel(), count + arity,
                         arity + in.parameters.size())->getArguments();
    auto velocity = [&](unsigned j) { return args[count + j]; };
    auto mass = [&](unsigned j) { return args[count + arity + j]; };
    std::vector<Value> units, rhs;
    Value one = k.constant(1.0), zero = k.constant(0.0);
    Value inverseCenter = k.div(one, mass(0));
    std::vector<Value> inverse;
    for (unsigned j = 1; j <= count; ++j) {
      units.push_back(k.unit(args[j - 1]));
      inverse.push_back(k.div(one, mass(j)));
      rhs.push_back(k.sub(zero, k.dot(units.back(),
                                      k.sub(velocity(j), velocity(0)))));
    }
    std::vector<std::vector<Value>> matrix(count, std::vector<Value>(count));
    for (unsigned i = 0; i != count; ++i)
      for (unsigned j = 0; j != count; ++j) {
        Value coupling = k.mul(k.dot(units[i], units[j]), inverseCenter);
        matrix[i][j] = i == j ? k.add(coupling, inverse[i]) : coupling;
      }
    auto impulse = smallSolve(k, matrix, rhs);
    std::vector<Value> changes(arity);
    Value center = k.zero();
    for (unsigned j = 0; j != count; ++j) {
      Value push = k.scale(impulse[j], units[j]);
      changes[j + 1] = k.scale(inverse[j], push);
      center = k.sub(center, k.scale(inverseCenter, push));
    }
    changes[0] = center;
    mdir::md::YieldOp::create(builder, loc, changes);
  }
  Value virial = in.virial;
  if (in.virial) {
    auto sum = mdir::md::SumTuplesOp::create(
        builder, loc, vec9, in.relation, in.positions, in.cell,
        ValueRange{change.getResult(), in.masses}, in.parameters, kinds, members);
    {
      OpBuilder::InsertionGuard guard(builder);
      auto args = kernel(builder, sum.getKernel(), count + arity,
                           arity + in.parameters.size())->getArguments();
      auto velocity = [&](unsigned j) { return args[count + j]; };
      auto mass = [&](unsigned j) { return args[count + arity + j]; };
      Value one = k.constant(1.0);
      SmallVector<Value> forces;
      for (unsigned j = 1; j <= count; ++j)
        forces.push_back(k.scale(k.div(k.mul(one, mass(j)), in.timeStep), velocity(j)));
      SmallVector<Value> elements;
      for (int a = 0; a != 3; ++a) {
        Value row;
        for (unsigned j = 1; j <= count; ++j) {
          Value term = k.scale(k.component(args[j - 1], a), forces[j - 1]);
          row = row ? k.add(row, term) : term;
        }
        for (int b = 0; b != 3; ++b)
          elements.push_back(k.component(row, b));
      }
      Value result = vector::FromElementsOp::create(builder, loc, vec9, elements);
      mdir::md::YieldOp::create(builder, loc, ValueRange{result});
    }
    virial = k.add(in.virial, sum.getResult());
  }
  Value velocities = buildAddChange(builder, in.velocities, change.getResult());
  return {velocities, virial};
}

Value mdir::driver::builder::buildConstraintForceVirial(
    OpBuilder &builder, const ConstraintForceInputs &in, unsigned arity,
    llvm::ArrayRef<std::pair<unsigned, unsigned>> bonds) {
  auto loc = builder.getUnknownLoc();
  auto vec9 = VectorType::get({9}, builder.getF64Type());
  DenseI32ArrayAttr kinds;
  DenseI64ArrayAttr members;
  coordinates(builder, arity, kinds, members);
  auto sum = mdir::md::SumTuplesOp::create(
      builder, loc, vec9, in.relation, in.positions, in.cell,
      ValueRange{in.velocities, in.forces, in.masses}, ValueRange{}, kinds, members);
  Kernel k(builder);
  {
    OpBuilder::InsertionGuard guard(builder);
    unsigned count = arity - 1;
    auto args = kernel(builder, sum.getKernel(), count + 2 * arity,
                         arity)->getArguments();
    auto position = [&](unsigned j) { return args[j - 1]; };
    auto velocity = [&](unsigned j) { return args[count + j]; };
    auto force = [&](unsigned j) { return args[count + arity + j]; };
    auto mass = [&](unsigned j) { return args[count + 2 * arity + j]; };
    Value zero = k.constant(0.0), one = k.constant(1.0);
    std::vector<Value> acceleration, inverse;
    for (unsigned j = 0; j != arity; ++j) {
      inverse.push_back(k.div(one, mass(j)));
      acceleration.push_back(k.scale(inverse.back(), force(j)));
    }
    std::vector<Value> units, rhs;
    for (auto [p, q] : bonds) {
      Value d = p == 0 ? position(q) : k.sub(position(q), position(p));
      Value length = k.norm(d);
      units.push_back(k.unit(d));
      Value relative = k.sub(velocity(q), velocity(p));
      Value centripetal = k.div(k.dot(relative, relative), length);
      Value along = k.dot(units.back(), k.sub(acceleration[q], acceleration[p]));
      rhs.push_back(k.sub(zero, k.add(along, centripetal)));
    }
    size_t n = bonds.size();
    std::vector<std::vector<Value>> matrix(n, std::vector<Value>(n));
    for (size_t b = 0; b != n; ++b)
      for (size_t c = 0; c != n; ++c) {
        auto [pb, qb] = bonds[b];
        auto [pc, qc] = bonds[c];
        auto on = [&](unsigned x) {
          return (x == qc ? 1 : 0) - (x == pc ? 1 : 0);
        };
        Value coefficient = zero;
        if (on(qb) != 0)
          coefficient = on(qb) > 0 ? k.add(coefficient, inverse[qb])
                                    : k.sub(coefficient, inverse[qb]);
        if (on(pb) != 0)
          coefficient = on(pb) > 0 ? k.sub(coefficient, inverse[pb])
                                    : k.add(coefficient, inverse[pb]);
        matrix[b][c] = k.mul(k.dot(units[b], units[c]), coefficient);
      }
    auto tension = smallSolve(k, matrix, rhs);
    std::vector<Value> forces(arity);
    for (unsigned j = 1; j != arity; ++j) {
      Value force = k.zero();
      for (size_t c = 0; c != n; ++c) {
        auto [pc, qc] = bonds[c];
        if (j != pc && j != qc)
          continue;
        Value push = k.scale(tension[c], units[c]);
        force = j == qc ? k.add(force, push) : k.sub(force, push);
      }
      forces[j] = force;
    }
    SmallVector<Value> elements;
    for (int a = 0; a != 3; ++a) {
      Value row;
      for (unsigned j = 1; j != arity; ++j) {
        Value term = k.scale(k.component(position(j), a), forces[j]);
        row = row ? k.add(row, term) : term;
      }
      for (int b = 0; b != 3; ++b)
        elements.push_back(k.component(row, b));
    }
    Value virial = vector::FromElementsOp::create(builder, loc, vec9, elements);
    mdir::md::YieldOp::create(builder, loc, ValueRange{virial});
  }
  return k.add(in.virial, sum.getResult());
}

ConstraintVelocityResult mdir::driver::builder::buildSettleVelocities(
    OpBuilder &builder, const ConstraintVelocityInputs &in) {
  auto loc = builder.getUnknownLoc();
  auto vec9 = VectorType::get({9}, builder.getF64Type());
  DenseI32ArrayAttr kinds;
  DenseI64ArrayAttr members;
  mdir::md::getCoordinateAttrs(
      builder, {{mdir::md::CoordinateKind::Displacement, {1, 0}},
                {mdir::md::CoordinateKind::Displacement, {2, 0}},
                {mdir::md::CoordinateKind::Displacement, {2, 1}}}, kinds, members);
  Kernel k(builder);
  auto change = mdir::md::GatherTuplesOp::create(
      builder, loc, in.velocities.getType(), Type{}, in.relation, in.positions,
      in.cell, ValueRange{in.velocities, in.masses}, in.parameters, kinds, members);
  {
    OpBuilder::InsertionGuard guard(builder);
    auto args = kernel(builder, change.getKernel(), 6, 5)->getArguments();
    Value e0 = k.unit(args[0]), e1 = k.unit(args[1]), e2 = k.unit(args[2]);
    Value one = k.constant(1.0);
    Value io = k.div(one, args[6]), ih = k.div(one, args[7]);
    Value zero = k.constant(0.0);
    auto minus = [&](Value value) { return k.sub(zero, value); };
    Value r0 = minus(k.dot(e0, k.sub(args[4], args[3])));
    Value r1 = minus(k.dot(e1, k.sub(args[5], args[3])));
    Value r2 = minus(k.dot(e2, k.sub(args[5], args[4])));
    Value a00 = k.add(io, ih), a11 = a00;
    Value a22 = k.add(ih, ih);
    Value a01 = k.mul(k.dot(e0, e1), io);
    Value a02 = minus(k.mul(k.dot(e0, e2), ih));
    Value a12 = k.mul(k.dot(e1, e2), ih);
    auto det3 = [&](Value m00, Value m01, Value m02,
                    Value m10, Value m11, Value m12,
                    Value m20, Value m21, Value m22) {
      auto minor = [&](Value p, Value q, Value r, Value s) {
        return k.sub(k.mul(p, q), k.mul(r, s));
      };
      return k.add(
          k.sub(k.mul(m00, minor(m11, m22, m12, m21)),
                k.mul(m01, minor(m10, m22, m12, m20))),
          k.mul(m02, minor(m10, m21, m11, m20)));
    };
    Value det = det3(a00, a01, a02, a01, a11, a12, a02, a12, a22);
    Value t0 = k.div(det3(r0, a01, a02, r1, a11, a12, r2, a12, a22), det);
    Value t1 = k.div(det3(a00, r0, a02, a01, r1, a12, a02, r2, a22), det);
    Value t2 = k.div(det3(a00, a01, r0, a01, a11, r1, a02, a12, r2), det);
    Value i0 = k.scale(t0, e0), i1 = k.scale(t1, e1), i2 = k.scale(t2, e2);
    Value dv0 = k.negate(k.scale(io, k.add(i0, i1)));
    Value dv1 = k.scale(ih, k.sub(i0, i2));
    Value dv2 = k.scale(ih, k.add(i1, i2));
    mdir::md::YieldOp::create(builder, loc, ValueRange{dv0, dv1, dv2});
  }
  Value virial = in.virial;
  if (in.virial) {
    auto sum = mdir::md::SumTuplesOp::create(
        builder, loc, vec9, in.relation, in.positions, in.cell,
        ValueRange{change.getResult(), in.masses}, in.parameters, kinds, members);
    {
      OpBuilder::InsertionGuard guard(builder);
      auto args = kernel(builder, sum.getKernel(), 6, 5)->getArguments();
      Value factor = k.div(args[7], in.timeStep);
      std::array<Value, 2> arms = {args[0], args[1]};
      std::array<Value, 2> forces = {k.scale(factor, args[4]), k.scale(factor, args[5])};
      SmallVector<Value> elements;
      for (int a = 0; a != 3; ++a) {
        Value row;
        for (int m = 0; m != 2; ++m) {
          Value term = k.scale(k.component(arms[m], a), forces[m]);
          row = row ? k.add(row, term) : term;
        }
        for (int b = 0; b != 3; ++b)
          elements.push_back(k.component(row, b));
      }
      Value result = vector::FromElementsOp::create(builder, loc, vec9, elements);
      mdir::md::YieldOp::create(builder, loc, ValueRange{result});
    }
    virial = k.add(in.virial, sum.getResult());
  }
  Value velocities = buildAddChange(builder, in.velocities, change.getResult());
  return {velocities, virial};
}
