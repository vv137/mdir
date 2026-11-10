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
