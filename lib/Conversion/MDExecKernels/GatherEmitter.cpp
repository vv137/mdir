#include "mdir/Conversion/GatherEmitter.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;

Value mdir::kernels::emitWrappedGridAxis(OpBuilder &builder, Location loc,
                                         Value zStart, Value lane,
                                         Value zSize, bool signedCompare) {
  Value zSum = arith::AddIOp::create(builder, loc, zStart, lane);
  Value zWrapped = arith::SubIOp::create(builder, loc, zSum, zSize);
  Value over = arith::CmpIOp::create(builder, loc,
      signedCompare ? arith::CmpIPredicate::sge : arith::CmpIPredicate::uge,
      zSum, zSize);
  return arith::SelectOp::create(builder, loc, over, zWrapped, zSum);
}

Value mdir::kernels::emitGridGather(OpBuilder &builder, Location loc,
                                    Value grid, Value rowBase, Value z,
                                    Value active) {
  Value at = arith::AddIOp::create(builder, loc, rowBase, z);
  if (active)
    at = arith::SelectOp::create(builder, loc, active, at, rowBase);
  if (!at.getType().isIndex())
    at = arith::IndexCastOp::create(builder, loc, builder.getIndexType(), at);
  return memref::LoadOp::create(builder, loc, grid, ValueRange{at});
}
