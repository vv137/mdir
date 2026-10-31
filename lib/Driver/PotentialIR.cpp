// Evaluations retain request/result order and the original physical state.
#include "PotentialIR.h"
#include "mdir/Driver/Builder.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MD/MDCoordinates.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include <cassert>
using namespace mlir;
using namespace mdir::driver::builder;
SmallVector<Value> mdir::driver::builder::potentialArguments(const PotentialStateIR &state) {
  SmallVector<Value> values{state.positions, state.cell};
  llvm::append_range(values, state.fields);
  if (state.time)
    values.push_back(state.time);
  return values;
}
EnergyResults mdir::driver::builder::buildEnergyEvaluation(
    OpBuilder &builder, const PotentialStateIR &state, const EnergyRequests &request) {
  assert((request.energy || request.forces || request.virial) && "evaluation needs a result");
  auto loc = builder.getUnknownLoc();
  SmallVector<Type> types;
  SmallVector<int32_t> kinds;
  if (request.energy) {
    types.push_back(builder.getF64Type());
    kinds.push_back(int32_t(mdir::md::Request::Energy));
  }
  if (request.forces) {
    types.push_back(state.positions.getType());
    kinds.push_back(int32_t(mdir::md::Request::Forces));
  }
  if (request.virial) {
    types.push_back(VectorType::get({9}, builder.getF64Type()));
    kinds.push_back(int32_t(mdir::md::Request::Virial));
  }
  SmallVector<int64_t> arguments(kinds.size(), -1);
  auto evaluation = mdir::md::EvaluateOp::create(
      builder, loc, types, FlatSymbolRefAttr::get(builder.getContext(), "energy"),
      potentialArguments(state), builder.getDenseI32ArrayAttr(kinds),
      builder.getDenseI64ArrayAttr(arguments));
  size_t index = 0;
  EnergyResults result;
  if (request.energy) result.energy = evaluation.getResult(index++);
  if (request.forces) result.forces = evaluation.getResult(index++);
  if (request.virial) result.virial = evaluation.getResult(index++);
  return result;
}
Value mdir::driver::builder::buildSimulationTime(OpBuilder &builder, Value step, Value interval) {
  auto loc = builder.getUnknownLoc();
  Value real = arith::SIToFPOp::create(builder, loc, builder.getF64Type(), step);
  return arith::MulFOp::create(builder, loc, real, interval);
}

SmallVector<Type> mdir::driver::builder::buildParameterTypes(
    MLIRContext &context, const mdir::driver::Program &program) {
  auto atoms = FlatSymbolRefAttr::get(&context, "atoms");
  auto f64 = Float64Type::get(&context);
  auto i32 = IntegerType::get(&context, 32);
  SmallVector<Type> types;
  for (const auto &field : program.fields)
    types.push_back(mdir::md::FieldType::get(&context, atoms, 1,
        field.isInteger ? Type(i32) : Type(f64)));
  for (const auto &table : program.tables)
    types.push_back(mdir::md::TableType::get(&context, 2, f64, table.isSquare()));
  for (const auto &set : program.tupleSets) {
    auto symbol = FlatSymbolRefAttr::get(&context, set.name);
    auto orientation = set.arity == 2 && !set.oriented
                           ? mdir::md::Orientation::Unordered
                           : set.reversible ? mdir::md::Orientation::Reversal
                                            : mdir::md::Orientation::Ordered;
    types.push_back(mdir::md::RelationType::get(&context, atoms, set.arity, orientation, symbol));
    for (unsigned i = 0; i != set.fields.size(); ++i)
      types.push_back(mdir::md::FieldType::get(&context, symbol, 1, f64));
  }
  return types;
}

mdir::md::PotentialOp mdir::driver::builder::buildScalarPotential(
    OpBuilder &builder, llvm::StringRef name, TypeRange inputs,
    llvm::function_ref<Value(OpBuilder &, ValueRange)> body) {
  auto loc = builder.getUnknownLoc();
  auto potential = mdir::md::PotentialOp::create(builder, loc, name,
      builder.getFunctionType(inputs, {builder.getF64Type()}), StringAttr{}, ArrayAttr{}, ArrayAttr{});
  OpBuilder::InsertionGuard guard(builder);
  SmallVector<Location> locations(inputs.size(), loc);
  auto args = builder.createBlock(&potential.getBody(), {}, inputs, locations)->getArguments();
  Value energy = body(builder, args);
  mdir::md::ReturnOp::create(builder, loc, ValueRange{energy});
  return potential;
}

SmallVector<Value> mdir::driver::builder::buildCentroidCoordinates(
    OpBuilder &builder, ValueRange values, size_t groups, unsigned arity) {
        using namespace mlir;
        auto loc = builder.getUnknownLoc();
        auto constant = [&](double value) -> Value {
          return arith::ConstantOp::create(builder, loc,
                                           builder.getF64FloatAttr(value));
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
        auto sum = [&](size_t slot, int component) -> Value {
          DenseI32ArrayAttr kinds;
          DenseI64ArrayAttr members;
          mdir::md::Coordinate coordinate{
              mdir::md::CoordinateKind::Displacement, {0, 1}};
          mdir::md::getCoordinateAttrs(builder, {coordinate}, kinds, members);
          auto sum = mdir::md::SumTuplesOp::create(
              builder, loc, builder.getF64Type(), values[0], values[1],
              values[2], ValueRange{}, ValueRange{values[3 + slot]}, kinds,
              members);
          {
            OpBuilder::InsertionGuard guard(builder);
            llvm::SmallVector<Type> types = {
                VectorType::get({3}, builder.getF64Type()),
                builder.getF64Type()};
            llvm::SmallVector<Location> locations(types.size(), loc);
            Block *block =
                builder.createBlock(&sum.getKernel(), {}, types, locations);
            Value displacement = vector::ExtractOp::create(
                builder, loc, block->getArgument(0), component);
            Value energy = mul(block->getArgument(1), displacement);
            mdir::md::YieldOp::create(builder, loc, ValueRange{energy});
          }
          return sum.getResult();
        };
        // Minimum-image member and reference links retain their original
        // loop order, allowing the same sums to fuse after inlining.
        using Vector = std::array<Value, 3>;
        Value angstrom = constant(1.0 / units::length);
        std::vector<Vector> centers;
        for (size_t g = 0; g != groups; ++g) {
          Vector position;
          for (int c = 0; c != 3; ++c) {
            Value value = sum(g, c);
            if (g > 0) {
              Value link = sum(groups + g - 1, c);
              value = add(link, value);
            }
            position[c] = mul(value, angstrom);
          }
          centers.push_back(position);
        }
        auto subtract = [&](const Vector &u, const Vector &v) {
          Vector w;
          for (int c = 0; c != 3; ++c)
            w[c] = sub(u[c], v[c]);
          return w;
        };
        auto dot = [&](const Vector &u, const Vector &v) {
          Value total = mul(u[0], v[0]);
          for (int c = 1; c != 3; ++c) {
            Value product = mul(u[c], v[c]);
            total = add(total, product);
          }
          return total;
        };
        auto cross = [&](const Vector &u, const Vector &v) {
          Vector w;
          for (int c = 0; c != 3; ++c) {
            int i = (c + 1) % 3, j = (c + 2) % 3;
            Value a = mul(u[i], v[j]);
            Value b = mul(u[j], v[i]);
            w[c] = sub(a, b);
          }
          return w;
        };
        auto scale = [&](Value factor, const Vector &u) {
          Vector w;
          for (int c = 0; c != 3; ++c)
            w[c] = mul(factor, u[c]);
          return w;
        };
        auto norm = [&](const Vector &u) -> Value {
          Value square = dot(u, u);
          return math::SqrtOp::create(builder, loc, square);
        };
        if (arity == 2) {
          Vector d = subtract(centers[1], centers[0]);
          Value r = norm(d);
          return {r, d[0], d[1], d[2]};
        }
        if (arity == 3) {
          Vector u = subtract(centers[0], centers[1]);
          Vector v = subtract(centers[2], centers[1]);
          Vector product = cross(u, v);
          Value a = norm(product);
          Value b = dot(u, v);
          return {math::Atan2Op::create(builder, loc, a, b)};
        }
        Vector b0 = subtract(centers[0], centers[1]);
        Vector b1 = subtract(centers[2], centers[1]);
        Vector b2 = subtract(centers[3], centers[2]);
        Value one = constant(1.0);
        Value length = norm(b1);
        Value inverse = arith::DivFOp::create(builder, loc, one, length);
        Vector axis = scale(inverse, b1);
        Value vProjection = dot(b0, axis);
        Vector vAxis = scale(vProjection, axis);
        Vector v = subtract(b0, vAxis);
        Value wProjection = dot(b2, axis);
        Vector wAxis = scale(wProjection, axis);
        Vector w = subtract(b2, wAxis);
        Vector product = cross(axis, v);
        Value a = dot(product, w);
        Value b = dot(v, w);
        return {math::Atan2Op::create(builder, loc, a, b)};
}
