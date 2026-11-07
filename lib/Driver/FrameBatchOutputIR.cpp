#include "FrameBatchOutputIR.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include <limits>
using namespace mlir;
using namespace mdir::driver::builder;
namespace {
bool buffer(Value value, unsigned rank, Type element) {
  auto type = dyn_cast<MemRefType>(value.getType());
  return type && type.getRank() == rank && type.getElementType() == element &&
      type.getLayout().isIdentity() && type.getMemorySpaceAsInt() == 1;
}
bool statusBuffer(OpBuilder &builder, Value status) {
  return buffer(status, 1, builder.getI64Type());
}
llvm::Error bad() {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
      "frame output requires matching contiguous device buffers and an i64 status vector");
}
void launch(OpBuilder &builder, Value count,
            llvm::function_ref<void(OpBuilder &, Value)> body) {
  auto loc = builder.getUnknownLoc();
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  Value threads = arith::ConstantIndexOp::create(builder, loc, 128);
  Value blocks = arith::CeilDivUIOp::create(builder, loc, count, threads);
  Value nonempty = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ugt, count, zero);
  scf::IfOp::create(builder, loc, nonempty, [&](OpBuilder &at, Location) {
    auto work = gpu::LaunchOp::create(at, loc, blocks, one, one, threads, one, one);
    // This kernel replaces host output arithmetic, not a potential kernel.
    // Preserve its separate multiply/add roundings in deterministic mode.
    work->setAttr("mdir.frame_host_arithmetic", at.getUnitAttr());
    OpBuilder kernel = OpBuilder::atBlockEnd(&work.getBody().front());
    Value base = arith::MulIOp::create(kernel, loc, work.getBlockIds().x, threads);
    Value row = arith::AddIOp::create(kernel, loc, base, work.getThreadIds().x);
    Value valid = arith::CmpIOp::create(kernel, loc, arith::CmpIPredicate::ult, row, count);
    scf::IfOp::create(kernel, loc, valid, [&](OpBuilder &inside, Location) {
      body(inside, row);
      scf::YieldOp::create(inside, loc);
    });
    gpu::TerminatorOp::create(kernel, loc);
    scf::YieldOp::create(at, loc);
  });
}
Value finite(OpBuilder &builder, Value value) {
  auto loc = builder.getUnknownLoc();
  unsigned width = value.getType().getIntOrFloatBitWidth();
  Type integer = builder.getIntegerType(width);
  Value bits = arith::BitcastOp::create(builder, loc, integer, value);
  uint64_t exponent = width == 64 ? 0x7ff0000000000000ULL : 0x7f800000ULL;
  Value mask = arith::ConstantOp::create(builder, loc, builder.getIntegerAttr(integer, exponent));
  Value selected = arith::AndIOp::create(builder, loc, bits, mask);
  return arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ne, selected, mask);
}
void failure(OpBuilder &builder, Value invalid, Value frame, Value status, unsigned reason) {
  auto loc = builder.getUnknownLoc();
  scf::IfOp::create(builder, loc, invalid, [&](OpBuilder &at, Location) {
    Value index = arith::IndexCastOp::create(at, loc, at.getI64Type(), frame);
    Value shift = arith::ConstantIntOp::create(at, loc, 16, 64);
    Value high = arith::ShLIOp::create(at, loc, index, shift);
    unsigned priority = reason == 1 || reason == 7 || reason == 8 ||
        reason == 9 || reason == 10 ? 0 : 1;
    Value code = arith::ConstantIntOp::create(at, loc, (priority << 8) | reason, 64);
    Value encoded = arith::OrIOp::create(at, loc, high, code);
    Value zero = arith::ConstantIndexOp::create(at, loc, 0);
    memref::AtomicRMWOp::create(at, loc, arith::AtomicRMWKind::mins,
        encoded, status, ValueRange{zero});
    scf::YieldOp::create(at, loc);
  });
}
void checkFinite(OpBuilder &builder, Value value, Value frame, Value status,
                 unsigned reason) {
  auto loc = builder.getUnknownLoc();
  Value no = arith::ConstantIntOp::create(builder, loc, 0, 1);
  Value invalid = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
      finite(builder, value), no);
  failure(builder, invalid, frame, status, reason);
}
}
Value mdir::driver::builder::buildFrameBatchIsFinite(OpBuilder &builder, Value value) {
  return finite(builder, value);
}
void mdir::driver::builder::buildFrameBatchFailure(OpBuilder &builder, Value invalid,
    Value frame, Value status, unsigned reason) {
  failure(builder, invalid, frame, status, reason);
}
void mdir::driver::builder::buildFrameBatchFiniteCheck(OpBuilder &builder,
    Value value, Value frame, Value status, unsigned reason) {
  checkFinite(builder, value, frame, status, reason);
}
llvm::Error mdir::driver::builder::buildFrameBatchStatus(OpBuilder &builder, Value status,
    Value initial) {
  if (!statusBuffer(builder, status)) return bad();
  auto loc = builder.getUnknownLoc();
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  launch(builder, one, [&](OpBuilder &at, Value zero) {
    Value success = initial ? initial : Value(arith::ConstantIntOp::create(at, loc,
        std::numeric_limits<int64_t>::max(), 64));
    memref::StoreOp::create(at, loc, success, status, zero);
  });
  return llvm::Error::success();
}
llvm::Error mdir::driver::builder::buildFrameBatchPositionOutput(OpBuilder &builder,
    Value forces, Value identities, Value destination, Value frame, Value status) {
  auto sourceType = dyn_cast<MemRefType>(forces.getType());
  auto targetType = dyn_cast<MemRefType>(destination.getType());
  if (!sourceType || !targetType) return bad();
  Type from = sourceType.getElementType(), to = targetType.getElementType();
  if (!(from.isF32() || from.isF64()) || !(to.isF32() || to.isF64()) ||
      from.getIntOrFloatBitWidth() > to.getIntOrFloatBitWidth() ||
      !buffer(forces, 2, from) || sourceType.getDimSize(1) != 3 ||
      !buffer(destination, 2, to) || targetType.getDimSize(1) != 3 ||
      !buffer(identities, 1, builder.getI32Type()) || !frame.getType().isIndex() ||
      !statusBuffer(builder, status)) return bad();
  auto loc = builder.getUnknownLoc();
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value count = memref::DimOp::create(builder, loc, forces, zero);
  launch(builder, count, [&](OpBuilder &at, Value row) {
    Value identity = memref::LoadOp::create(at, loc, identities, row);
    identity = arith::IndexCastOp::create(at, loc, at.getIndexType(), identity);
    for (unsigned component = 0; component != 3; ++component) {
      Value index = arith::ConstantIndexOp::create(at, loc, component);
      Value force = memref::LoadOp::create(at, loc, forces, ValueRange{row, index});
      Value gradient = arith::NegFOp::create(at, loc, force);
      if (from != to) gradient = arith::ExtFOp::create(at, loc, to, gradient);
      memref::StoreOp::create(at, loc, gradient, destination, ValueRange{identity, index});
      checkFinite(at, gradient, frame, status, 3);
    }
  });
  return llvm::Error::success();
}
llvm::Error mdir::driver::builder::buildFrameBatchScalarOutput(OpBuilder &builder,
    Value scalar, Value destination, Value frame, Value status, unsigned reason,
    Value destinationIndex) {
  if (!scalar.getType().isF64() || !buffer(destination, 1, builder.getF64Type()) ||
      !frame.getType().isIndex() || !statusBuffer(builder, status) || reason >= 256) return bad();
  auto loc = builder.getUnknownLoc();
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  launch(builder, one, [&](OpBuilder &at, Value) {
    memref::StoreOp::create(at, loc, scalar, destination,
        destinationIndex ? destinationIndex : frame);
    checkFinite(at, scalar, frame, status, reason);
  });
  return llvm::Error::success();
}
llvm::Error mdir::driver::builder::buildFrameBatchFieldOutput(OpBuilder &builder,
    Value source, Value identities, Value destination, Value offset,
    Value frame, Value status) {
  auto type = dyn_cast<MemRefType>(source.getType());
  if (!type || !(type.getElementType().isF32() || type.getElementType().isF64()) ||
      !buffer(source, 1, type.getElementType()) ||
      !buffer(identities, 1, builder.getI32Type()) ||
      !buffer(destination, 1, builder.getF64Type()) || !statusBuffer(builder, status))
    return bad();
  auto loc = builder.getUnknownLoc();
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value count = memref::DimOp::create(builder, loc, source, zero);
  launch(builder, count, [&](OpBuilder &at, Value particle) {
    Value value = memref::LoadOp::create(at, loc, source, particle);
    if (type.getElementType().isF32())
      value = arith::ExtFOp::create(at, loc, at.getF64Type(), value);
    Value identity = memref::LoadOp::create(at, loc, identities, particle);
    Value index = arith::IndexCastOp::create(at, loc, at.getIndexType(), identity);
    index = arith::AddIOp::create(at, loc, offset, index);
    memref::StoreOp::create(at, loc, value, destination, index);
    checkFinite(at, value, frame, status, 4);
  });
  return llvm::Error::success();
}
llvm::Error mdir::driver::builder::buildFrameBatchStrainOutput(OpBuilder &builder,
    Value virial, Value correction, Value destination, Value frame, Value status) {
  if (virial.getType() != VectorType::get({9}, builder.getF64Type()) ||
      !correction.getType().isF64() || !buffer(destination, 2, builder.getF64Type()) ||
      cast<MemRefType>(destination.getType()).getDimSize(1) != 9 ||
      !frame.getType().isIndex() || !statusBuffer(builder, status)) return bad();
  auto loc = builder.getUnknownLoc();
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  launch(builder, one, [&](OpBuilder &at, Value) {
    for (int index = 0; index != 9; ++index) {
      Value value = vector::ExtractOp::create(at, loc, virial, (index % 3) * 3 + index / 3);
      value = arith::NegFOp::create(at, loc, value);
      if (index == 0 || index == 4 || index == 8)
        value = arith::SubFOp::create(at, loc, value, correction);
      Value column = arith::ConstantIndexOp::create(at, loc, index);
      memref::StoreOp::create(at, loc, value, destination, ValueRange{frame, column});
      checkFinite(at, value, frame, status, 4);
    }
  });
  return llvm::Error::success();
}
