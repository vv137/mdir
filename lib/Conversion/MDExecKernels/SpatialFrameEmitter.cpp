#include "mdir/Conversion/SpatialFrameEmitter.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mdir/Conversion/ReductionEmitter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include <limits>

using namespace mlir;
using namespace mdir::kernels;

SpatialReferenceFrame mdir::kernels::emitSpatialReferenceFrame(
    OpBuilder &b, Location loc, Value point) {
  SmallVector<Value, 3> wrapped, lattice;
  for (int64_t axis = 0; axis != 3; ++axis) {
    Value component = vector::ExtractOp::create(b, loc, point, axis);
    Value frame = math::FloorOp::create(b, loc, component);
    lattice.push_back(frame);
    wrapped.push_back(arith::SubFOp::create(b, loc, component, frame));
  }
  return {vector::FromElementsOp::create(b, loc, point.getType(), wrapped),
          vector::FromElementsOp::create(b, loc, point.getType(), lattice)};
}

SpatialBoundingBox mdir::kernels::emitSpatialWarpBox(
    OpBuilder &b, Location loc, Value point, Value valid) {
  Value infinity = vector::BroadcastOp::create(b, loc, point.getType(),
      createReal(b, loc, b.getF64Type(), std::numeric_limits<double>::infinity()));
  Value minInput = arith::SelectOp::create(b, loc, valid, point, infinity);
  Value maxInput = arith::SelectOp::create(b, loc, valid, point,
      arith::NegFOp::create(b, loc, infinity));
  Value minimum = arith::NegFOp::create(b, loc,
      emitWarpReduce(b, loc, arith::NegFOp::create(b, loc, minInput), Combiner::Max));
  Value maximum = emitWarpReduce(b, loc, maxInput, Combiner::Max);
  return {minimum, maximum};
}

Value mdir::kernels::emitSpatialBoxesWithinReach(OpBuilder &b, Location loc,
    SpatialBoundingBox first, SpatialBoundingBox second, Value shift,
    Value reach, Value roundoff) {
  auto real = [&](double value) { return createReal(b, loc, b.getF64Type(), value); };
  auto broadcast = [&](Value value) {
    return vector::BroadcastOp::create(b, loc, first.minimum.getType(), value);
  };
  Value low = arith::SubFOp::create(b, loc, first.minimum,
      arith::AddFOp::create(b, loc, second.maximum, shift));
  Value high = arith::SubFOp::create(b, loc,
      arith::AddFOp::create(b, loc, second.minimum, shift), first.maximum);
  Value gap = arith::MaximumFOp::create(b, loc, low, high);
  gap = arith::MaximumFOp::create(b, loc,
      arith::SubFOp::create(b, loc, gap, broadcast(roundoff)), broadcast(real(0)));
  Value square = arith::MulFOp::create(b, loc, gap, gap);
  Value lower = vector::ReductionOp::create(b, loc, vector::CombiningKind::ADD, square);
  // Protect the final lower-bound square/sum as well as coordinate errors.
  Value enlarged = arith::MulFOp::create(b, loc, reach, real(1 + 0x1p-46));
  Value upper = arith::MulFOp::create(b, loc, enlarged, enlarged);
  // Overflow/indeterminate box arithmetic must fall through to the exact
  // pair test, rather than becoming a false rejection by the prefilter.
  return arith::CmpFOp::create(b, loc, arith::CmpFPredicate::ULE, lower, upper);
}

SpatialSingleImage mdir::kernels::emitSpatialSingleImage(OpBuilder &b,
    Location loc, SpatialBoundingBox first, SpatialBoundingBox second,
    Value radius, Value roundoff) {
  Value proved = arith::ConstantOp::create(b, loc, b.getBoolAttr(true));
  SmallVector<Value, 3> images;
  auto real = [&](double x) { return createReal(b, loc, b.getF64Type(), x); };
  for (int64_t axis = 0; axis != 3; ++axis) {
    auto component = [&](Value x) -> Value {
      return vector::ExtractOp::create(b, loc, x, axis);
    };
    Value margin = arith::AddFOp::create(b, loc, component(radius), component(roundoff));
    Value low = arith::SubFOp::create(b, loc,
        arith::SubFOp::create(b, loc, component(first.minimum), component(second.maximum)), margin);
    Value high = arith::AddFOp::create(b, loc,
        arith::SubFOp::create(b, loc, component(first.maximum), component(second.minimum)), margin);
    Value width = arith::SubFOp::create(b, loc, high, low);
    // Wrapped boxes are in [0,1]; this additional absolute envelope
    // dominates the endpoint/subtraction/center rounding at that scale.
    Value narrow = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLT,
        width, real(1 - 0x1p-40));
    proved = arith::AndIOp::create(b, loc, proved, narrow);
    Value center = arith::MulFOp::create(b, loc,
        arith::AddFOp::create(b, loc, low, high), real(0.5));
    images.push_back(math::FloorOp::create(b, loc,
        arith::AddFOp::create(b, loc, center, real(0.5))));
  }
  return {proved, vector::FromElementsOp::create(b, loc, first.minimum.getType(), images)};
}
