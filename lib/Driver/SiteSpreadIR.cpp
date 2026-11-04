// Native virtual-site force transfer and its virial, in original loop order.
#include "SiteIR.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include <array>

using namespace mlir;
using namespace mdir::driver::builder;

namespace {
class Geometry {
public:
  explicit Geometry(OpBuilder &builder)
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
  Value norm(Value a) {
    Value square = dot(a, a);
    return math::SqrtOp::create(builder, loc, square);
  }
  Value splat(Value a) {
    return vector::BroadcastOp::create(builder, loc, vec, a);
  }
  Value scale(Value factor, Value a) { return mul(splat(factor), a); }
  Value component(Value a, int index) {
    return vector::ExtractOp::create(builder, loc, a, index);
  }
  Value zero() {
    return arith::ConstantOp::create(builder, loc, DenseElementsAttr::get(vec, 0.0));
  }

private:
  OpBuilder &builder;
  Location loc;
  VectorType vec;
};

// Unit bond directions and their bisector. Force transfer applies the
// transpose of the placement derivative: perpendicular to the bisector,
// then to each bond, so the site follows bond directions, not lengths.
struct SiteFrame {
  Value u, v, lengthU, lengthV, sum, lengthSum;
  SiteFrame(Geometry &k, Value toJ, Value toK) {
    lengthU = k.norm(toJ);
    lengthV = k.norm(toK);
    u = k.div(toJ, k.splat(lengthU));
    v = k.div(toK, k.splat(lengthV));
    sum = k.add(u, v);
    lengthSum = k.norm(sum);
  }
  std::array<Value, 3> spread(Geometry &k, Value force, Value distance) {
    Value bisector = k.div(sum, k.splat(lengthSum));
    Value across = k.sub(force, k.scale(k.dot(force, bisector), bisector));
    auto toHydrogen = [&](Value unit, Value length) {
      Value factor = k.div(distance, k.mul(length, lengthSum));
      Value perpendicular = k.sub(across, k.scale(k.dot(across, unit), unit));
      return k.scale(factor, perpendicular);
    };
    Value first = toHydrogen(u, lengthU), second = toHydrogen(v, lengthV);
    Value owner = k.sub(k.sub(force, first), second);
    return {owner, first, second};
  }
};

using Coordinate = mdir::md::Coordinate;
constexpr auto displacement = mdir::md::CoordinateKind::Displacement;

Block *kernel(OpBuilder &builder, Region &region, unsigned vectors,
                unsigned scalars = 0) {
  auto vec = VectorType::get({3}, builder.getF64Type());
  SmallVector<Type> types(vectors, vec);
  types.append(scalars, builder.getF64Type());
  SmallVector<Location> locations(types.size(), builder.getUnknownLoc());
  return builder.createBlock(&region, {}, types, locations);
}

mdir::md::GatherTuplesOp gather(OpBuilder &builder, const SiteSpreadInputs &in,
                                 ArrayRef<Coordinate> coordinates,
                                 bool parameters) {
  DenseI32ArrayAttr kinds;
  DenseI64ArrayAttr members;
  mdir::md::getCoordinateAttrs(builder, coordinates, kinds, members);
  SmallVector<Value> fields;
  if (parameters)
    fields = {in.a, in.b};
  return mdir::md::GatherTuplesOp::create(
      builder, builder.getUnknownLoc(), in.forces.getType(), Type{}, in.relation,
      in.positions, in.cell, ValueRange{in.forces}, fields, kinds, members);
}

mdir::md::SumTuplesOp sumVirial(OpBuilder &builder, const SiteSpreadInputs &in,
                                 ArrayRef<Coordinate> coordinates) {
  DenseI32ArrayAttr kinds;
  DenseI64ArrayAttr members;
  mdir::md::getCoordinateAttrs(builder, coordinates, kinds, members);
  auto virialType = VectorType::get({9}, builder.getF64Type());
  return mdir::md::SumTuplesOp::create(
      builder, builder.getUnknownLoc(), virialType, in.relation, in.positions,
      in.cell, ValueRange{in.forces}, ValueRange{in.a, in.b}, kinds, members);
}

Value composeVirial(OpBuilder &builder, ValueRange elements) {
  auto type = VectorType::get({9}, builder.getF64Type());
  return vector::FromElementsOp::create(builder, builder.getUnknownLoc(), type,
                                        elements);
}
} // namespace

SiteSpreadResult mdir::driver::builder::buildSiteSpread(
    OpBuilder &builder, const SiteSpreadInputs &in, bool amber, bool derived,
    llvm::StringRef potential) {
  auto loc = builder.getUnknownLoc();
  Geometry k(builder);
  if (!derived) {
    auto delta = gather(builder, in,
                        {{displacement, {2, 1}}, {displacement, {3, 1}}}, true);
    {
      OpBuilder::InsertionGuard guard(builder);
      auto args = kernel(builder, delta.getKernel(), 6, 2)->getArguments();
      std::array<Value, 3> spread;
      if (amber) {
        SiteFrame frame(k, args[0], args[1]);
        spread = frame.spread(k, args[2], args[6]);
      } else {
        Value first = k.scale(args[6], args[2]), second = k.scale(args[7], args[2]);
        spread = {k.sub(k.sub(args[2], first), second), first, second};
      }
      Value removed = k.negate(args[2]);
      mdir::md::YieldOp::create(builder, loc,
                                ValueRange{removed, spread[0], spread[1], spread[2]});
    }
    Value virial = in.virial;
    if (amber && in.virial) {
      auto change = sumVirial(
          builder, in, {{displacement, {2, 1}}, {displacement, {3, 1}},
                        {displacement, {1, 0}}, {displacement, {2, 0}},
                        {displacement, {3, 0}}});
      {
        OpBuilder::InsertionGuard guard(builder);
        auto args = kernel(builder, change.getKernel(), 9, 2)->getArguments();
        SiteFrame frame(k, args[0], args[1]);
        auto forces = frame.spread(k, args[5], args[9]);
        SmallVector<Value> elements;
        for (int a = 0; a != 3; ++a) {
          Value row;
          for (int m = 0; m != 3; ++m) {
            Value term = k.scale(k.component(args[2 + m], a), forces[m]);
            row = row ? k.add(row, term) : term;
          }
          for (int b = 0; b != 3; ++b)
            elements.push_back(k.component(row, b));
        }
        Value result = composeVirial(builder, elements);
        mdir::md::YieldOp::create(builder, loc, ValueRange{result});
      }
      virial = k.add(in.virial, change.getResult());
    }
    Value forces = buildAddChange(builder, in.forces, delta.getResult());
    return {forces, virial};
  }

  // The placement forces omit the owner's translation. Its complete
  // transpose adds that translation and subtracts the differentiated forces.
  SmallVector<int32_t> kinds = {int32_t(mdir::md::Request::Forces)};
  SmallVector<int64_t> arguments = {-1};
  SmallVector<Type> resultTypes = {in.forces.getType()};
  if (in.virial) {
    kinds.push_back(int32_t(mdir::md::Request::Virial));
    arguments.push_back(-1);
    resultTypes.push_back(VectorType::get({9}, builder.getF64Type()));
  }
  auto evaluation = mdir::md::EvaluateOp::create(
      builder, loc, resultTypes,
      FlatSymbolRefAttr::get(builder.getContext(), potential),
      ValueRange{in.positions, in.cell, in.relation, in.a, in.b, in.forces},
      builder.getDenseI32ArrayAttr(kinds), builder.getDenseI64ArrayAttr(arguments));
  auto moved = gather(builder, in, {{displacement, {2, 1}}}, false);
  {
    OpBuilder::InsertionGuard guard(builder);
    auto args = kernel(builder, moved.getKernel(), 5)->getArguments();
    Value removed = k.negate(args[1]), zero = k.zero();
    mdir::md::YieldOp::create(builder, loc, ValueRange{removed, args[1], zero, zero});
  }
  auto mapped = mdir::md::MapParticlesOp::create(
      builder, loc, in.forces.getType(),
      ValueRange{in.forces, moved.getResult(), evaluation.getResult(0)});
  {
    OpBuilder::InsertionGuard guard(builder);
    auto args = kernel(builder, mapped.getKernel(), 3)->getArguments();
    Value sum = k.add(args[0], args[1]);
    Value forces = k.sub(sum, args[2]);
    mdir::md::YieldOp::create(builder, loc, ValueRange{forces});
  }
  if (!in.virial)
    return {mapped.getResult(), {}};

  // W_atoms = -W_placement - sum(p tensor f), relative to the site.
  auto arm = sumVirial(builder, in,
                       {{displacement, {2, 1}}, {displacement, {3, 1}}});
  {
    OpBuilder::InsertionGuard guard(builder);
    auto args = kernel(builder, arm.getKernel(), 6, 2)->getArguments();
    Value placed = buildSitePlacement(builder, amber, args[0], args[1],
                                       args[6], args[7]);
    SmallVector<Value> elements;
    for (int a = 0; a != 3; ++a)
      for (int b = 0; b != 3; ++b)
        elements.push_back(k.mul(k.component(placed, a), k.component(args[2], b)));
    Value result = composeVirial(builder, elements);
    mdir::md::YieldOp::create(builder, loc, ValueRange{result});
  }
  Value total = k.add(evaluation.getResult(1), arm.getResult());
  Value virial = k.sub(in.virial, total);
  return {mapped.getResult(), virial};
}
