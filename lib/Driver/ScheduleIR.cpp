// Preserve the counter arithmetic and convergence polling order.
#include "ScheduleIR.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MDRT/MDRTOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
using namespace mlir;
using namespace mdir::driver::builder;
Value mdir::driver::builder::buildStepNumber(
    OpBuilder &builder, llvm::ArrayRef<LoopIterationIR> loops, Value current,
    Value outerPeriod, Value period, Value one, Value start) {
  auto loc = builder.getUnknownLoc();
  auto add = [&](Value a, Value b) -> Value { return arith::AddIOp::create(builder, loc, a, b); };
  auto mul = [&](Value a, Value b) -> Value { return arith::MulIOp::create(builder, loc, a, b); };
  Value counted = loops.front().index;
  for (auto loop : loops.drop_front()) {
    Value scaled = mul(counted, loop.extent);
    counted = add(scaled, loop.index);
  }
  Value steps;
  if (outerPeriod) {
    Value before = mul(counted, outerPeriod);
    Value done = add(current, one);
    Value within = mul(done, period);
    steps = add(before, within);
  } else {
    Value done = add(counted, one);
    steps = mul(done, period);
  }
  Value since = arith::IndexCastOp::create(builder, loc, builder.getI64Type(), steps);
  return add(since, start);
}
MinimizationStopIR mdir::driver::builder::buildMinimizationStop(
    OpBuilder &builder, Value start, int64_t steps) {
  auto loc = builder.getUnknownLoc();
  Value memory = memref::AllocaOp::create(builder, loc, MemRefType::get({2}, builder.getI64Type()));
  Value flagIndex = arith::ConstantIndexOp::create(builder, loc, 0);
  Value stepIndex = arith::ConstantIndexOp::create(builder, loc, 1);
  Value notDone = arith::ConstantIntOp::create(builder, loc, 0, 64);
  Value count = arith::ConstantIntOp::create(builder, loc, steps, 64);
  Value last = arith::AddIOp::create(builder, loc, start, count);
  memref::StoreOp::create(builder, loc, notDone, memory, flagIndex);
  memref::StoreOp::create(builder, loc, last, memory, stepIndex);
  mdir::mdrt::HostCallOp::create(builder, loc, TypeRange{},
      FlatSymbolRefAttr::get(builder.getContext(), "mdrtCheckMinimization"),
      ValueRange{memory}, ValueRange{}, UnitAttr{});
  return {memory, flagIndex, stepIndex, notDone};
}
MinimizationBoundsIR mdir::driver::builder::buildMinimizationBounds(
    OpBuilder &builder, const MinimizationStopIR &stop, Value steps, Value zero, Value one) {
  auto loc = builder.getUnknownLoc();
  Value flag = memref::LoadOp::create(builder, loc, stop.memory, stop.flagIndex);
  Value open = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq, flag, stop.notDone);
  Value stepTrips = arith::SelectOp::create(builder, loc, open, steps, zero);
  Value rowTrips = arith::SelectOp::create(builder, loc, open, one, zero);
  return {stepTrips, rowTrips};
}
Value mdir::driver::builder::buildMinimizationFrameBound(
    OpBuilder &builder, const MinimizationStopIR &stop, Value step, Value zero, Value one) {
  auto loc = builder.getUnknownLoc();
  Value flag = memref::LoadOp::create(builder, loc, stop.memory, stop.flagIndex);
  Value at = memref::LoadOp::create(builder, loc, stop.memory, stop.stepIndex);
  Value open = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq, flag, stop.notDone);
  Value final = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq, at, step);
  Value due = arith::OrIOp::create(builder, loc, open, final);
  return arith::SelectOp::create(builder, loc, due, one, zero);
}

Value mdir::driver::builder::buildSegmentStep(
    OpBuilder &builder, Value outerCount, Value outerPeriod, Value tail,
    Value extra, Value start) {
  auto loc = builder.getUnknownLoc();
  Value before = arith::MulIOp::create(builder, loc, outerCount, outerPeriod);
  Value withTail = arith::AddIOp::create(builder, loc, before, tail);
  Value total = arith::AddIOp::create(builder, loc, withTail, extra);
  Value since = arith::IndexCastOp::create(builder, loc, builder.getI64Type(), total);
  return arith::AddIOp::create(builder, loc, start, since);
}
