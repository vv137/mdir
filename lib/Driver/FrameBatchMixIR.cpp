#include "FrameBatchMixIR.h"
#include "FrameBatchOutputIR.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
using namespace mlir;
using namespace mdir::driver::builder;
namespace {
bool vector(Value value, Type element) {
  auto type = dyn_cast<MemRefType>(value.getType());
  return type && type.getRank() == 1 && type.getElementType() == element &&
      type.getLayout().isIdentity() && type.getMemorySpaceAsInt() == 1;
}
}
llvm::Error mdir::driver::builder::buildFrameBatchMixing(OpBuilder &builder,
    Value theta, Value typeMap, Value typeValues, Value pairMap, Value fixedPairs,
    Value destination, Value logicalDestination, const FrameBatchMixIR &config, Value status) {
  auto type = dyn_cast<MemRefType>(destination.getType());
  if (!type || !(type.getElementType().isF32() || type.getElementType().isF64()) ||
      !vector(destination, type.getElementType()) || !config.types ||
      !vector(theta, builder.getF64Type()) || !vector(typeMap, builder.getI64Type()) ||
      !vector(typeValues, builder.getF64Type()) || !vector(pairMap, builder.getI64Type()) ||
      !vector(fixedPairs, builder.getI64Type()) ||
      (logicalDestination && !vector(logicalDestination, builder.getF64Type())))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
        "frame combining rule requires contiguous device theta, type and pair metadata");
  auto loc = builder.getUnknownLoc();
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  Value threads = arith::ConstantIndexOp::create(builder, loc, 128);
  uint64_t count = uint64_t(config.types) * config.types;
  Value blocks = arith::ConstantIndexOp::create(builder, loc, count / 128 + (count % 128 != 0));
  Value size = arith::ConstantIndexOp::create(builder, loc, count);
  Value types = arith::ConstantIndexOp::create(builder, loc, config.types);
  auto launch = gpu::LaunchOp::create(builder, loc, blocks, one, one, threads, one, one);
  launch->setAttr("mdir.frame_host_arithmetic", builder.getUnitAttr());
  OpBuilder kernel = OpBuilder::atBlockEnd(&launch.getBody().front());
  Value base = arith::MulIOp::create(kernel, loc, launch.getBlockIds().x, threads);
  Value index = arith::AddIOp::create(kernel, loc, base, launch.getThreadIds().x);
  Value valid = arith::CmpIOp::create(kernel, loc, arith::CmpIPredicate::ult, index, size);
  scf::IfOp::create(kernel, loc, valid, [&](OpBuilder &at, Location) {
    Value a = arith::DivUIOp::create(at, loc, index, types);
    Value b = arith::RemUIOp::create(at, loc, index, types);
    Value zero = arith::ConstantIntOp::create(at, loc, 0, 64);
    Value nil = arith::ConstantOp::create(at, loc, at.getF64FloatAttr(0));
    Value half = arith::ConstantOp::create(at, loc, at.getF64FloatAttr(0.5));
    auto read = [&](Value site) -> Value {
      Value mapped = memref::LoadOp::create(at, loc, typeMap, site);
      Value taken = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::sge, mapped, zero);
      auto value = scf::IfOp::create(at, loc, TypeRange{at.getF64Type()}, taken, true);
      {
        OpBuilder yes = OpBuilder::atBlockEnd(value.thenBlock());
        Value slot = arith::IndexCastOp::create(yes, loc, yes.getIndexType(), mapped);
        Value loaded = memref::LoadOp::create(yes, loc, theta, slot);
        scf::YieldOp::create(yes, loc, loaded);
      }
      {
        OpBuilder no = OpBuilder::atBlockEnd(value.elseBlock());
        Value loaded = memref::LoadOp::create(no, loc, typeValues, site);
        scf::YieldOp::create(no, loc, loaded);
      }
      return value.getResult(0);
    };
    Value first = read(a), second = read(b);
    Value pair = memref::LoadOp::create(at, loc, pairMap, index);
    Value taken = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::sge, pair, zero);
    auto write = [&](OpBuilder &out, Value value) {
      if (logicalDestination) memref::StoreOp::create(out, loc, value, logicalDestination, index);
      if (type.getElementType().isF32()) value = arith::TruncFOp::create(out, loc, out.getF32Type(), value);
      memref::StoreOp::create(out, loc, value, destination, index);
    };
    if (config.derivative) {
      Value weight = half;
      if (config.geometric) {
        Value ratio = arith::DivFOp::create(at, loc, second, first);
        Value root = math::SqrtOp::create(at, loc, ratio);
        weight = arith::MulFOp::create(at, loc, half, root);
        Value isZero = arith::CmpFOp::create(at, loc, arith::CmpFPredicate::OEQ, first, nil);
        if (status) {
          Value other = arith::CmpFOp::create(at, loc, arith::CmpFPredicate::ONE, second, nil);
          Value invalid = arith::AndIOp::create(at, loc, isZero, other);
          invalid = arith::AndIOp::create(at, loc, invalid, taken);
          Value frame = arith::ConstantIndexOp::create(at, loc, 0);
          buildFrameBatchFailure(at, invalid, frame, status, 7);
        }
        weight = arith::SelectOp::create(at, loc, isZero, nil, weight);
      }
      weight = arith::SelectOp::create(at, loc, taken, weight, nil);
      write(at, weight);
    } else {
      scf::IfOp::create(at, loc, taken, [&](OpBuilder &out, Location) {
        Value slot = arith::IndexCastOp::create(out, loc, out.getIndexType(), pair);
        Value value = memref::LoadOp::create(out, loc, theta, slot);
        write(out, value);
        scf::YieldOp::create(out, loc);
      }, [&](OpBuilder &out, Location) {
        Value fixed = memref::LoadOp::create(out, loc, fixedPairs, index);
        Value change = arith::CmpIOp::create(out, loc, arith::CmpIPredicate::eq, fixed, zero);
        scf::IfOp::create(out, loc, change, [&](OpBuilder &mix, Location) {
          Value value;
          if (config.geometric) {
            Value product = arith::MulFOp::create(mix, loc, first, second);
            value = math::SqrtOp::create(mix, loc, product);
          } else {
            Value sum = arith::AddFOp::create(mix, loc, first, second);
            value = arith::MulFOp::create(mix, loc, half, sum);
          }
          write(mix, value);
          scf::YieldOp::create(mix, loc);
        });
        scf::YieldOp::create(out, loc);
      });
    }
    scf::YieldOp::create(at, loc);
  });
  gpu::TerminatorOp::create(kernel, loc);
  return llvm::Error::success();
}
