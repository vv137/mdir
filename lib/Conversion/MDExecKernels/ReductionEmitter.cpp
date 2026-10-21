// The emitters of reductions on a device (D301); see
// include/mdir/Conversion/ReductionEmitter.h.

#include "mdir/Conversion/ReductionEmitter.h"

#include "mdir/Conversion/MDExecKernels.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;

Value kernels::emitShuffleXor(OpBuilder &builder, Location loc, Value value,
                              int64_t offset) {
  Type type = value.getType();
  if (auto vector = dyn_cast<VectorType>(type)) {
    SmallVector<Value> elements;
    for (int64_t i = 0, e = vector.getNumElements(); i != e; ++i)
      elements.push_back(emitShuffleXor(
          builder, loc, vector::ExtractOp::create(builder, loc, value, i),
          offset));
    return vector::FromElementsOp::create(builder, loc, vector, elements);
  }
  Type i32 = builder.getI32Type();
  Value distance = arith::ConstantOp::create(
      builder, loc, i32, builder.getI32IntegerAttr(offset));
  Value width =
      arith::ConstantOp::create(builder, loc, i32, builder.getI32IntegerAttr(32));
  auto shuffle = [&](Value word) {
    return gpu::ShuffleOp::create(builder, loc, word, distance, width,
                                  gpu::ShuffleMode::XOR)
        .getShuffleResult();
  };
  if (type.getIntOrFloatBitWidth() == 32)
    return shuffle(value);
  Type i64 = builder.getI64Type();
  Value bits = arith::BitcastOp::create(builder, loc, i64, value);
  Value shift =
      arith::ConstantOp::create(builder, loc, i64, builder.getI64IntegerAttr(32));
  Value low = arith::TruncIOp::create(builder, loc, i32, bits);
  Value high = arith::TruncIOp::create(
      builder, loc, i32, arith::ShRUIOp::create(builder, loc, bits, shift));
  Value lowBack =
      arith::ExtUIOp::create(builder, loc, i64, shuffle(low));
  Value highBack = arith::ShLIOp::create(
      builder, loc, arith::ExtUIOp::create(builder, loc, i64, shuffle(high)),
      shift);
  Value joined = arith::OrIOp::create(builder, loc, lowBack, highBack);
  return arith::BitcastOp::create(builder, loc, type, joined);
}

Value kernels::emitCombine(OpBuilder &builder, Location loc,
                           Combiner combiner, Value lhs, Value rhs) {
  if (combiner == Combiner::Add) {
    if (isa<IntegerType>(getElementTypeOrSelf(lhs.getType())))
      return arith::AddIOp::create(builder, loc, lhs, rhs);
    return arith::AddFOp::create(builder, loc, lhs, rhs);
  }
  Value larger = arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::OGT,
                                       lhs, rhs);
  return arith::SelectOp::create(builder, loc, larger, lhs, rhs);
}

Value kernels::emitOrderedRegisterSum(
    OpBuilder &builder, Location loc, Value initial, int64_t count,
    llvm::function_ref<Value(OpBuilder &, int64_t)> emitTerm) {
  Value value = initial;
  for (int64_t k = 0; k != count; ++k)
    value = emitCombine(builder, loc, Combiner::Add, value,
                        emitTerm(builder, k));
  return value;
}

Value kernels::emitWarpReduce(OpBuilder &builder, Location loc, Value value,
                              Combiner combiner, int64_t lanes) {
  for (int64_t offset = lanes / 2; offset >= 1; offset /= 2)
    value = emitCombine(builder, loc, combiner, value,
                        emitShuffleXor(builder, loc, value, offset));
  return value;
}

SmallVector<Value> kernels::emitBlockReduce(OpBuilder &b, Location loc,
                                            ValueRange values, Value thread,
                                            Value shared, Combiner combiner,
                                            int64_t blockSize) {
  int64_t warps = blockSize / 32;
  SmallVector<Value> numbers;
  for (Value value : values) {
    auto vector = dyn_cast<VectorType>(value.getType());
    if (!vector) {
      numbers.push_back(value);
      continue;
    }
    for (int64_t i = 0, e = vector.getNumElements(); i != e; ++i)
      numbers.push_back(vector::ExtractOp::create(b, loc, value, i));
  }
  for (Value &number : numbers)
    number = emitWarpReduce(b, loc, number, combiner);
  Value width = createIndex(b, loc, 32);
  Value lane = arith::RemUIOp::create(b, loc, thread, width);
  Value warp = arith::DivUIOp::create(b, loc, thread, width);
  Value count = createIndex(b, loc, numbers.size());
  Value leads = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, lane,
                                      createIndex(b, loc, 0));
  Type wide = b.getF64Type();
  scf::IfOp::create(b, loc, leads, [&](OpBuilder &then, Location) {
    Value base = arith::MulIOp::create(then, loc, warp, count);
    for (auto [k, number] : llvm::enumerate(numbers)) {
      Value place = arith::AddIOp::create(then, loc, base,
                                          createIndex(then, loc, k));
      Value stored = number;
      if (number.getType() != wide)
        stored = arith::ExtFOp::create(then, loc, wide, number);
      memref::StoreOp::create(then, loc, stored, shared, ValueRange{place});
    }
    scf::YieldOp::create(then, loc);
  });
  gpu::BarrierOp::create(b, loc);
  SmallVector<Value> totals;
  for (auto [k, number] : llvm::enumerate(numbers)) {
    Value total;
    for (int64_t w = 0; w != warps; ++w) {
      Value part = memref::LoadOp::create(
          b, loc, shared,
          ValueRange{createIndex(b, loc, w * numbers.size() + k)});
      if (number.getType() != wide)
        part = arith::TruncFOp::create(b, loc, number.getType(), part);
      total = w == 0 ? part : emitCombine(b, loc, combiner, total, part);
    }
    totals.push_back(total);
  }
  SmallVector<Value> results;
  size_t next = 0;
  for (Value value : values) {
    auto vector = dyn_cast<VectorType>(value.getType());
    if (!vector) {
      results.push_back(totals[next++]);
      continue;
    }
    ArrayRef<Value> elements(totals.begin() + next,
                             totals.begin() + next + vector.getNumElements());
    next += vector.getNumElements();
    results.push_back(
        vector::FromElementsOp::create(b, loc, vector, elements));
  }
  return results;
}
