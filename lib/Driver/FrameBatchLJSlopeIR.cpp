#include "FrameBatchLJSlopeIR.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
using namespace mlir;
using namespace mdir::driver::builder;
void mdir::driver::builder::buildFrameBatchLJSlopes(OpBuilder &builder,
    Value sigma, Value epsilon, Value orderedTypePairs, Value weights,
    Value volumeConstants, Value fixedConstants, const FrameBatchLJSlopeIR &config) {
  if (!config.types) return;
  auto loc = builder.getUnknownLoc();
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  Value types = arith::ConstantIndexOp::create(builder, loc, config.types);
  Value threads = arith::ConstantIndexOp::create(builder, loc, 128);
  Value blocks = arith::ConstantIndexOp::create(builder, loc,
      config.types / 128 + (config.types % 128 != 0));
  auto launch = gpu::LaunchOp::create(builder, loc, blocks, one, one, threads, one, one);
  launch->setAttr("mdir.frame_host_arithmetic", builder.getUnitAttr());
  OpBuilder kernel = OpBuilder::atBlockEnd(&launch.getBody().front());
  Value first = arith::MulIOp::create(kernel, loc, launch.getBlockIds().x, threads);
  Value a = arith::AddIOp::create(kernel, loc, first, launch.getThreadIds().x);
  Value valid = arith::CmpIOp::create(kernel, loc, arith::CmpIPredicate::ult, a, types);
  scf::IfOp::create(kernel, loc, valid, [&](OpBuilder &at, Location) {
    auto real = [&](double value) -> Value {
      return arith::ConstantOp::create(at, loc, at.getF64FloatAttr(value));
    };
    Value nil = real(0), factor = real(config.factor);
    Value coefficient = real(config.sigma ? 24 : 4);
    auto slope = [&](OpBuilder &term, Value row, Value column) -> Value {
      Value base = arith::MulIOp::create(term, loc, row, types);
      Value index = arith::AddIOp::create(term, loc, base, column);
      Value s = memref::LoadOp::create(term, loc, sigma, index);
      Value pairs = memref::LoadOp::create(term, loc, orderedTypePairs, index);
      Value s2 = arith::MulFOp::create(term, loc, s, s);
      Value s4 = arith::MulFOp::create(term, loc, s2, s2);
      Value power = arith::MulFOp::create(term, loc, config.sigma ? s : s2, s4);
      Value multiplier = coefficient;
      if (config.sigma) {
        Value e = memref::LoadOp::create(term, loc, epsilon, index);
        multiplier = arith::MulFOp::create(term, loc, multiplier, e);
      }
      Value c6 = arith::MulFOp::create(term, loc, multiplier, power);
      Value scaled = arith::MulFOp::create(term, loc, factor, pairs);
      return arith::MulFOp::create(term, loc, scaled, c6);
    };
    auto sum = scf::ForOp::create(at, loc, zero, types, one, ValueRange{nil},
        [&](OpBuilder &term, Location, Value b, ValueRange carried) {
      Value base = arith::MulIOp::create(term, loc, a, types);
      Value index = arith::AddIOp::create(term, loc, base, b);
      Value weight = memref::LoadOp::create(term, loc, weights, index);
      Value nonzero = arith::CmpFOp::create(term, loc, arith::CmpFPredicate::ONE, weight, nil);
      auto next = scf::IfOp::create(term, loc, TypeRange{term.getF64Type()}, nonzero, true);
      {
        OpBuilder yes = OpBuilder::atBlockEnd(next.thenBlock());
        Value ab = slope(yes, a, b), ba = slope(yes, b, a);
        Value pair = arith::AddFOp::create(yes, loc, ab, ba);
        Value weighted = arith::MulFOp::create(yes, loc, pair, weight);
        Value result = arith::AddFOp::create(yes, loc, carried[0], weighted);
        scf::YieldOp::create(yes, loc, result);
      }
      {
        OpBuilder no = OpBuilder::atBlockEnd(next.elseBlock());
        scf::YieldOp::create(no, loc, carried);
      }
      scf::YieldOp::create(term, loc, next.getResult(0));
    });
    Value base = arith::ConstantIndexOp::create(at, loc, config.rawBase);
    Value index = arith::AddIOp::create(at, loc, base, a);
    Value scaled = arith::MulFOp::create(at, loc, sum.getResult(0), real(2));
    Value fixed = arith::MulFOp::create(at, loc, sum.getResult(0), real(config.fixedFactor));
    memref::StoreOp::create(at, loc, scaled, volumeConstants, index);
    memref::StoreOp::create(at, loc, fixed, fixedConstants, index);
    scf::YieldOp::create(at, loc);
  });
  gpu::TerminatorOp::create(kernel, loc);
}
