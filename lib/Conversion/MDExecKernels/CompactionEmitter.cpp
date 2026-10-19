// The emitters of compaction, scans, and counting sorts on a device
// (D[compaction-emitters]); see include/mdir/Conversion/CompactionEmitter.h.

#include "mdir/Conversion/CompactionEmitter.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;

Value kernels::emitLanesBelow(OpBuilder &b, Location loc, Value lane) {
  auto constant32 = [&](OpBuilder &b, int64_t v) -> Value {
    return arith::ConstantOp::create(b, loc, b.getI32Type(),
                                     b.getI32IntegerAttr(v));
  };
  return arith::SubIOp::create(
      b, loc, arith::ShLIOp::create(b, loc, constant32(b, 1), lane),
      constant32(b, 1));
}

Value kernels::emitWarpBallot(OpBuilder &b, Location loc, Value predicate) {
  return gpu::BallotOp::create(b, loc, b.getI32Type(), predicate);
}

Value kernels::emitWarpRank(OpBuilder &b, Location loc, Value ballot,
                            Value below) {
  return math::CtPopOp::create(b, loc,
                               arith::AndIOp::create(b, loc, ballot, below));
}

Value kernels::emitWarpCount(OpBuilder &b, Location loc, Value ballot) {
  return math::CtPopOp::create(b, loc, ballot);
}
