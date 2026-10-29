// Constraint kernels with their original coordinates, solves, and arithmetic.
#include "ConstraintIR.h"
#include "SiteIR.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
  Value root(Value value) { return math::SqrtOp::create(builder, loc, value); }
  Value compose(Value x, Value y, Value z) {
    return vector::FromElementsOp::create(builder, loc, vec, ValueRange{x, y, z});
  }
  Value cross(Value a, Value b) {
    Value a0 = component(a, 0), a1 = component(a, 1), a2 = component(a, 2),
          b0 = component(b, 0), b1 = component(b, 1), b2 = component(b, 2);
    auto term = [&](Value p, Value q, Value r, Value t) {
      return sub(mul(p, q), mul(r, t));
    };
    return compose(term(a1, b2, a2, b1), term(a2, b0, a0, b2),
                     term(a0, b1, a1, b0));
  }
  Value combine(Value a, Value x, Value b, Value y, Value c, Value z) {
    return add(add(scale(a, x), scale(b, y)), scale(c, z));
  }
  Value abs(Value value) { return math::AbsFOp::create(builder, loc, value); }
  Value compare(arith::CmpFPredicate predicate, Value a, Value b) {
    return arith::CmpFOp::create(builder, loc, predicate, a, b);
  }
  Value either(Value a, Value b) { return arith::OrIOp::create(builder, loc, a, b); }
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

Value lengthFailure(Kernel &k, ArrayRef<std::pair<Value, Value>> bonds,
                      double tolerance) {
  Value one = k.constant(1.0), limit = k.constant(tolerance);
  Value any;
  for (const auto &[r, length] : bonds) {
    Value d2 = k.mul(length, length);
    Value off = k.sub(k.div(k.dot(r, r), d2), one);
    Value magnitude = k.abs(off);
    Value bad = k.compare(arith::CmpFPredicate::UGT, magnitude, limit);
    any = any ? k.either(any, bad) : bad;
  }
  return any;
}

void yieldValues(OpBuilder &builder, Block &block, ValueRange values) {
  if (!block.empty() && isa<scf::YieldOp>(block.back()))
    block.back().erase();
  builder.setInsertionPointToEnd(&block);
  scf::YieldOp::create(builder, builder.getUnknownLoc(), values);
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

ConstraintPositionResult mdir::driver::builder::buildSettlePositions(
    OpBuilder &builder, const ConstraintPositionInputs &in, bool withFailure,
    double tolerance) {
  auto loc = builder.getUnknownLoc();
  DenseI32ArrayAttr kinds;
  DenseI64ArrayAttr members;
  mdir::md::getCoordinateAttrs(
      builder, {{mdir::md::CoordinateKind::Displacement, {1, 0}},
                {mdir::md::CoordinateKind::Displacement, {2, 0}}}, kinds, members);
  Type failedType = withFailure ? Type(builder.getI1Type()) : Type{};
  auto change = mdir::md::GatherTuplesOp::create(
      builder, loc, in.positions.getType(), failedType, in.relation, in.positions,
      in.cell, ValueRange{in.oldPositions, in.positions, in.masses},
      in.parameters, kinds, members);
  {
    OpBuilder::InsertionGuard guard(builder);
    auto args = kernel(builder, change.getKernel(), 8, 5)->getArguments();
    Kernel k(builder);
    // The old bonds, in the periods of the new ones.
    auto oldBond = [&](Value bond, Value o, Value n) {
      Value raw = k.sub( n, args[5]);
      Value shift = k.sub( bond, raw);
      return k.add( k.sub( o, args[2]), shift);
    };
    Value b0 = oldBond(args[0], args[3], args[6]);
    Value c0 = oldBond(args[1], args[4], args[7]);
    // The rigid triangle: the oxygen at ra from the center of mass along the
    // bisector, the hydrogens at rb behind it and rc to each side.
    Value two = k.constant(2.0), half = k.constant(0.5),
                one = k.constant(1.0);
    Value total =
        k.add( args[8], k.mul( two, args[9]));
    Value rc = k.mul( half, args[12]);
    Value height = k.root(k.sub( k.mul( args[11], args[11]), k.mul( rc, rc)));
    Value ra = k.div( k.mul( k.mul( two, args[9]), height), total);
    Value rb = k.sub( height, ra);
    // The new positions about their center of mass.
    Value center = k.scale(k.div( args[9], total),
                                 k.add( args[0], args[1]));
    Value a1 = k.negate(center);
    Value b1 = k.sub( args[0], center);
    Value c1 = k.sub( args[1], center);
    Value z = k.unit(k.cross(b0, c0));
    Value xAxis = k.unit(k.cross(a1, z));
    Value y = k.cross(z, xAxis);
    Value xb0 = k.dot(b0, xAxis), yb0 = k.dot(b0, y);
    Value xc0 = k.dot(c0, xAxis), yc0 = k.dot(c0, y);
    Value za1 = k.dot(a1, z);
    Value xb1 = k.dot(b1, xAxis), yb1 = k.dot(b1, y), zb1 = k.dot(b1, z);
    Value xc1 = k.dot(c1, xAxis), yc1 = k.dot(c1, y), zc1 = k.dot(c1, z);
    auto complement = [&](Value sine) {
      return k.root(k.sub( one, k.mul( sine, sine)));
    };
    Value sinPhi = k.div( za1, ra);
    Value cosPhi = complement(sinPhi);
    Value sinPsi =
        k.div( k.sub( zb1, zc1),
               k.mul( k.mul( two, rc), cosPhi));
    Value cosPsi = complement(sinPsi);
    Value ya2 = k.mul( ra, cosPhi);
    Value xb2 = k.mul( k.sub( k.constant(0.0), rc),
                             cosPsi);
    Value t1 = k.mul( k.sub( k.constant(0.0), rb),
                            cosPhi);
    Value t2 = k.mul( k.mul( rc, sinPsi), sinPhi);
    Value yb2 = k.sub( t1, t2), yc2 = k.add( t1, t2);
    auto sum3 = [&](Value a, Value b, Value c) {
      return k.add( k.add( a, b), c);
    };
    Value alpha = sum3(k.mul( xb2, k.sub( xb0, xc0)),
                             k.mul( yb0, yb2),
                             k.mul( yc0, yc2));
    Value beta = sum3(k.mul( xb2, k.sub( yc0, yb0)),
                            k.mul( xb0, yb2),
                            k.mul( xc0, yc2));
    Value gamma = k.add(
        k.sub( k.mul( xb0, yb1), k.mul( xb1, yb0)),
        k.sub( k.mul( xc0, yc1), k.mul( xc1, yc0)));
    Value both = k.add( k.mul( alpha, alpha),
                              k.mul( beta, beta));
    Value sinTheta = k.div(
        k.sub( k.mul( alpha, gamma),
               k.mul( beta,
                      k.root(k.sub( both,
                                    k.mul( gamma, gamma))))),
        both);
    Value cosTheta = complement(sinTheta);
    Value zero = k.constant(0.0);
    Value xa3 = k.sub( zero, k.mul( ya2, sinTheta));
    Value ya3 = k.mul( ya2, cosTheta);
    Value xb3 = k.sub( k.mul( xb2, cosTheta),
                             k.mul( yb2, sinTheta));
    Value yb3 = k.add( k.mul( xb2, sinTheta),
                             k.mul( yb2, cosTheta));
    Value xc3 = k.sub( k.sub( zero,
                                            k.mul( xb2, cosTheta)),
                             k.mul( yc2, sinTheta));
    Value yc3 = k.add( k.sub( zero,
                                            k.mul( xb2, sinTheta)),
                             k.mul( yc2, cosTheta));
    // Back to the cell, relative to the new oxygen.
    auto back = [&](Value px, Value py, Value pz) {
      return k.add( k.combine(px, xAxis, py, y, pz, z), center);
    };
    Value a3 = back(xa3, ya3, za1);
    Value b3 = back(xb3, yb3, zb1);
    Value c3 = back(xc3, yc3, zc1);
    Value db = k.sub( b3, args[0]);
    Value dc = k.sub( c3, args[1]);

    SmallVector<Value> results = {a3, db, dc};
    if (withFailure) {
      Value bad = lengthFailure(k, {{k.sub(b3, a3), args[11]},
                                     {k.sub(c3, a3), args[11]},
                                     {k.sub(c3, b3), args[12]}}, tolerance);
      results.push_back(bad);
    }
    mdir::md::YieldOp::create(builder, loc, results);
  }
  return {change.getResult(), withFailure ? change.getFailed() : Value{}};
}

ConstraintPositionResult mdir::driver::builder::buildBondConstraints(
    OpBuilder &builder, const ConstraintPositionInputs &in, unsigned arity,
    llvm::ArrayRef<ConstraintBond> bonds, const ConstraintPositionConfig &config) {
  auto loc = builder.getUnknownLoc();
  auto vec = VectorType::get({3}, builder.getF64Type());
  unsigned count = bonds.size();
  SmallVector<mdir::md::Coordinate> coords;
  for (const auto &bond : bonds)
    coords.push_back({mdir::md::CoordinateKind::Displacement,
                      {bond.second, bond.first}});
  DenseI32ArrayAttr kinds;
  DenseI64ArrayAttr members;
  mdir::md::getCoordinateAttrs(builder, coords, kinds, members);
  Type failedType = config.withFailure ? Type(builder.getI1Type()) : Type{};
  auto change = mdir::md::GatherTuplesOp::create(
      builder, loc, in.positions.getType(), failedType, in.relation, in.positions,
      in.cell, ValueRange{in.oldPositions, in.positions, in.masses},
      in.parameters, kinds, members);
  {
    OpBuilder::InsertionGuard guard(builder);
    auto args = kernel(builder, change.getKernel(), count + 2 * arity,
                         arity + in.parameters.size())->getArguments();
    auto old = [&](unsigned j) { return args[count + j]; };
    auto position = [&](unsigned j) { return args[count + arity + j]; };
    auto mass = [&](unsigned j) { return args[count + 2 * arity + j]; };
    auto length = [&](const ConstraintBond &bond) {
      return args[count + 3 * arity + bond.length];
    };
    Kernel k(builder);
    std::vector<Value> olds, inverse;
    Value one = k.constant(1.0), two = k.constant(2.0);
    for (unsigned j = 0; j != arity; ++j)
      inverse.push_back(k.div(one, mass(j)));
    for (auto [i, bond] : llvm::enumerate(bonds)) {
      Value raw = k.sub(position(bond.second), position(bond.first));
      Value shift = k.sub(args[i], raw);
      olds.push_back(k.add(k.sub(old(bond.second), old(bond.first)), shift));
    }
    Value zero = k.zero();
    // M-SHAKE linearizes each bond error [Krautler2001]:
    // sum_l 2 (r_k dot s_l) w_kl lambda_l = d_k^2 - r_k^2. Members move
    // by +/- lambda_l s_l / m along the old bonds, with signed shared
    // inverse masses w_kl. Carry displacements, not rounded positions.
    // Four Newton sweeps reach f64 rounding from a typical 1e-2 bond error;
    // six retain the original margin. Carry small displacements [Jung2026].
    auto newton = [&]() {
      Value first = arith::ConstantIndexOp::create(builder, loc, 0);
      Value step = arith::ConstantIndexOp::create(builder, loc, 1);
      Value iterations = arith::ConstantIndexOp::create(builder, loc, 6);
      std::vector<Value> initial(arity, zero);
      auto loop = scf::ForOp::create(builder, loc, first, iterations, step, initial);
      {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(loop.getBody());
        Kernel l(builder);
        auto carried = loop.getRegionIterArgs();
        std::vector<Value> moved(carried.begin(), carried.end());
        std::vector<Value> current, error;
        for (auto [i, bond] : llvm::enumerate(bonds)) {
          current.push_back(l.sub(l.add(args[i], moved[bond.second]),
                                   moved[bond.first]));
          error.push_back(l.sub(l.mul(length(bond), length(bond)),
                                 l.dot(current.back(), current.back())));
        }
        // Signed inverse masses shared by the two constraint bonds.
        auto weightOf = [&](const ConstraintBond &k, const ConstraintBond &bond) {
          Value sum;
          auto add = [&](bool negative, unsigned j) {
            if (!sum)
              sum = negative ? l.sub(l.constant(0.0), inverse[j]) : inverse[j];
            else
              sum = negative ? l.sub(sum, inverse[j]) : l.add(sum, inverse[j]);
          };
          if (k.second == bond.second)
            add(false, k.second);
          if (k.second == bond.first)
            add(true, k.second);
          if (k.first == bond.second)
            add(true, k.first);
          if (k.first == bond.first)
            add(false, k.first);
          return sum ? sum : l.constant(0.0);
        };
        std::vector<std::vector<Value>> matrix(count, std::vector<Value>(count));
        for (unsigned i = 0; i != count; ++i)
          for (unsigned j = 0; j != count; ++j)
            matrix[i][j] = l.mul(l.mul(two, l.dot(current[i], olds[j])),
                                  weightOf(bonds[i], bonds[j]));
        auto lambda = smallSolve(l, matrix, error);
        for (auto [j, bond] : llvm::enumerate(bonds)) {
          Value push = l.scale(lambda[j], olds[j]);
          moved[bond.second] = l.add(moved[bond.second],
                                      l.scale(inverse[bond.second], push));
          moved[bond.first] = l.sub(moved[bond.first],
                                     l.scale(inverse[bond.first], push));
        }
        yieldValues(builder, *loop.getBody(), moved);
      }
      auto results = loop.getResults();
      return std::vector<Value>(results.begin(), results.end());
    };
    std::vector<Value> results;
    // The rationalized near root of the single-bond quadratic avoids
    // cancellation. Retain its residual/direction checks and Newton fallback.
    if (config.analytic && count == 1) {
      const auto &bond = bonds.front();
      Value r = args[0], s = olds.front();
      Value d2 = k.mul(length(bond), length(bond));
      Value error = k.sub(d2, k.dot(r, r));
      Value rs = k.dot(r, s);
      Value discriminant = k.add(k.mul(rs, rs), k.mul(k.dot(s, s), error));
      Value t = k.div(error, k.add(rs, k.root(discriminant)));
      Value lambda = k.div(t, k.add(inverse[bond.first], inverse[bond.second]));
      Value push = k.scale(lambda, s);
      Value da = k.negate(k.scale(inverse[bond.first], push));
      Value db = k.scale(inverse[bond.second], push);
      Value corrected = k.sub(k.add(r, db), da);
      Value residual = k.sub(k.dot(corrected, corrected), d2);
      Value tolerance = k.mul(d2, k.constant(config.analyticTolerance));
      Value scalarZero = k.constant(0.0);
      Value magnitude = k.abs(residual);
      Value accurate = k.compare(arith::CmpFPredicate::OLE, magnitude, tolerance);
      Value forward = k.compare(arith::CmpFPredicate::OGT, rs, scalarZero);
      Value checked = arith::AndIOp::create(builder, loc, accurate, forward);
      auto branch = scf::IfOp::create(builder, loc, TypeRange{vec, vec}, checked, true);
      {
        OpBuilder::InsertionGuard guard(builder);
        yieldValues(builder, *branch.thenBlock(), ValueRange{da, db});
      }
      {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(branch.elseBlock());
        auto projected = newton();
        yieldValues(builder, *branch.elseBlock(), projected);
      }
      auto values = branch.getResults();
      results.assign(values.begin(), values.end());
    } else {
      results = newton();
    }
    if (config.withFailure) {
      Kernel test(builder);
      std::vector<std::pair<Value, Value>> lengths;
      for (auto [i, bond] : llvm::enumerate(bonds))
        lengths.push_back({test.sub(test.add(args[i], results[bond.second]),
                                     results[bond.first]), length(bond)});
      Value failed = lengthFailure(test, lengths, config.tolerance);
      results.push_back(failed);
    }
    mdir::md::YieldOp::create(builder, loc, results);
  }
  return {change.getResult(), config.withFailure ? change.getFailed() : Value{}};
}
