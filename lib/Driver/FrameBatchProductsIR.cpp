#include "FrameBatchProductsIR.h"
#include "FrameBatchOutputIR.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
using namespace mlir;
using namespace mdir::driver::builder;
namespace {
void rows(OpBuilder &builder, Value count,
          llvm::function_ref<void(OpBuilder &, Value)> body) {
  auto loc = builder.getUnknownLoc();
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  Value threads = arith::ConstantIndexOp::create(builder, loc, 128);
  Value blocks = arith::CeilDivUIOp::create(builder, loc, count, threads);
  Value nonempty = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ugt, count, zero);
  scf::IfOp::create(builder, loc, nonempty, [&](OpBuilder &at, Location) {
    auto launch = gpu::LaunchOp::create(at, loc, blocks, one, one, threads, one, one);
    launch->setAttr("mdir.frame_host_arithmetic", at.getUnitAttr());
    OpBuilder kernel = OpBuilder::atBlockEnd(&launch.getBody().front());
    Value base = arith::MulIOp::create(kernel, loc, launch.getBlockIds().x, threads);
    Value index = arith::AddIOp::create(kernel, loc, base, launch.getThreadIds().x);
    Value valid = arith::CmpIOp::create(kernel, loc, arith::CmpIPredicate::ult, index, count);
    scf::IfOp::create(kernel, loc, valid, [&](OpBuilder &inside, Location) {
      body(inside, index);
      scf::YieldOp::create(inside, loc);
    });
    gpu::TerminatorOp::create(kernel, loc);
    scf::YieldOp::create(at, loc);
  });
}
}
void mdir::driver::builder::buildFrameBatchProductStart(OpBuilder &builder, Value vjp) {
  auto loc = builder.getUnknownLoc();
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value count = memref::DimOp::create(builder, loc, vjp, zero);
  rows(builder, count, [&](OpBuilder &at, Value index) {
    Value value = arith::ConstantOp::create(at, loc, at.getF64FloatAttr(0));
    memref::StoreOp::create(at, loc, value, vjp, index);
  });
}
void mdir::driver::builder::buildFrameBatchRowProducts(OpBuilder &builder,
    Value row, Value tangent, Value cotangent, Value jvp, Value vjp,
    Value frame, Value status) {
  auto loc = builder.getUnknownLoc();
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  Value count = memref::DimOp::create(builder, loc, row, zero);
  Value vjpSize = memref::DimOp::create(builder, loc, vjp, zero);
  Value takesVjp = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ugt, vjpSize, zero);
  rows(builder, count, [&](OpBuilder &at, Value index) {
    Value value = memref::LoadOp::create(at, loc, row, index);
    buildFrameBatchFiniteCheck(at, value, frame, status, 4);
    scf::IfOp::create(at, loc, takesVjp, [&](OpBuilder &write, Location) {
      Value weight = memref::LoadOp::create(write, loc, cotangent, frame);
      buildFrameBatchFiniteCheck(write, weight, frame, status, 10);
      Value product = arith::MulFOp::create(write, loc, weight, value);
      Value previous = memref::LoadOp::create(write, loc, vjp, index);
      Value sum = arith::AddFOp::create(write, loc, previous, product);
      memref::StoreOp::create(write, loc, sum, vjp, index);
      buildFrameBatchFiniteCheck(write, sum, frame, status, 5);
      scf::YieldOp::create(write, loc);
    });
  });
  Value jvpSize = memref::DimOp::create(builder, loc, jvp, zero);
  Value takesJvp = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ugt, jvpSize, zero);
  scf::IfOp::create(builder, loc, takesJvp, [&](OpBuilder &at, Location) {
    rows(at, one, [&](OpBuilder &kernel, Value) {
      Value nil = arith::ConstantOp::create(kernel, loc, kernel.getF64FloatAttr(0));
      auto sum = scf::ForOp::create(kernel, loc, zero, count, one, ValueRange{nil},
          [&](OpBuilder &term, Location, Value index, ValueRange carried) {
        Value value = memref::LoadOp::create(term, loc, row, index);
        Value weight = memref::LoadOp::create(term, loc, tangent, index);
        // A tangent belongs to the whole batch, not to any one frame.
        buildFrameBatchFiniteCheck(term, weight, zero, status, 9);
        Value product = arith::MulFOp::create(term, loc, value, weight);
        Value added = arith::AddFOp::create(term, loc, carried[0], product);
        scf::YieldOp::create(term, loc, added);
      });
      memref::StoreOp::create(kernel, loc, sum.getResult(0), jvp, frame);
      buildFrameBatchFiniteCheck(kernel, sum.getResult(0), frame, status, 5);
    });
    scf::YieldOp::create(at, loc);
  });
}
