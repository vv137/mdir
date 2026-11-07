#include "FrameBatchChainRuleIR.h"
#include "FrameBatchParameterPlan.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
using namespace mlir;
using namespace mdir::driver::builder;
namespace {
void rows(OpBuilder &builder, int64_t count,
          llvm::function_ref<void(OpBuilder &, Value)> body) {
  if (!count) return;
  auto loc = builder.getUnknownLoc();
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  Value threads = arith::ConstantIndexOp::create(builder, loc, 128);
  Value blocks = arith::ConstantIndexOp::create(builder, loc, count / 128 + (count % 128 != 0));
  Value size = arith::ConstantIndexOp::create(builder, loc, count);
  auto launch = gpu::LaunchOp::create(builder, loc, blocks, one, one, threads, one, one);
  launch->setAttr("mdir.frame_host_arithmetic", builder.getUnitAttr());
  OpBuilder at = OpBuilder::atBlockEnd(&launch.getBody().front());
  Value base = arith::MulIOp::create(at, loc, launch.getBlockIds().x, threads);
  Value index = arith::AddIOp::create(at, loc, base, launch.getThreadIds().x);
  Value valid = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ult, index, size);
  scf::IfOp::create(at, loc, valid, [&](OpBuilder &inside, Location) {
    body(inside, index);
    scf::YieldOp::create(inside, loc);
  });
  gpu::TerminatorOp::create(at, loc);
}
}
void mdir::driver::builder::buildFrameBatchRawCorrections(OpBuilder &builder,
    const FrameBatchParameterPlan &plan, Value raw, Value kinds,
    Value volumeConstants, Value fixedConstants, Value typeOffsets,
    Value particlesByType, Value volumeScale) {
  auto loc = builder.getUnknownLoc();
  for (auto [field, target] : llvm::enumerate(plan.typeSums)) {
    if (target < 0) continue;
    rows(builder, plan.types, [&](OpBuilder &at, Value type) {
      Value one = arith::ConstantIndexOp::create(at, loc, 1);
      Value next = arith::AddIOp::create(at, loc, type, one);
      Value begin = memref::LoadOp::create(at, loc, typeOffsets, type);
      Value end = memref::LoadOp::create(at, loc, typeOffsets, next);
      begin = arith::IndexCastOp::create(at, loc, at.getIndexType(), begin);
      end = arith::IndexCastOp::create(at, loc, at.getIndexType(), end);
      Value zero = arith::ConstantOp::create(at, loc, at.getF64FloatAttr(0));
      Value base = arith::ConstantIndexOp::create(at, loc, plan.fieldValues[field]);
      auto sum = scf::ForOp::create(at, loc, begin, end, one, ValueRange{zero},
          [&](OpBuilder &term, Location, Value item, ValueRange carried) {
        Value particle = memref::LoadOp::create(term, loc, particlesByType, item);
        particle = arith::IndexCastOp::create(term, loc, term.getIndexType(), particle);
        Value index = arith::AddIOp::create(term, loc, base, particle);
        Value value = memref::LoadOp::create(term, loc, raw, index);
        Value added = arith::AddFOp::create(term, loc, carried[0], value);
        scf::YieldOp::create(term, loc, added);
      });
      Value output = arith::ConstantIndexOp::create(at, loc, target);
      output = arith::AddIOp::create(at, loc, output, type);
      memref::StoreOp::create(at, loc, sum.getResult(0), raw, output);
    });
  }
  rows(builder, plan.rawValues, [&](OpBuilder &at, Value index) {
    Value kind = memref::LoadOp::create(at, loc, kinds, index);
    Value zero = arith::ConstantIntOp::create(at, loc, 0, 64);
    Value taken = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ne, kind, zero);
    scf::IfOp::create(at, loc, taken, [&](OpBuilder &write, Location) {
      Value original = memref::LoadOp::create(write, loc, raw, index);
      Value volume = memref::LoadOp::create(write, loc, volumeConstants, index);
      Value fixed = memref::LoadOp::create(write, loc, fixedConstants, index);
      volume = arith::MulFOp::create(write, loc, volume, volumeScale);
      Value shifted = arith::AddFOp::create(write, loc, original, volume);
      shifted = arith::AddFOp::create(write, loc, shifted, fixed);
      Value charge = arith::AddFOp::create(write, loc, fixed, volume);
      Value two = arith::ConstantIntOp::create(write, loc, 2, 64);
      Value isCharge = arith::CmpIOp::create(write, loc, arith::CmpIPredicate::eq, kind, two);
      Value result = arith::SelectOp::create(write, loc, isCharge, charge, shifted);
      memref::StoreOp::create(write, loc, result, raw, index);
      scf::YieldOp::create(write, loc);
    });
  });
}
