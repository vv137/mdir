#include "FrameBatchPMEIR.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
using namespace mlir;
using namespace mdir::driver::builder;
void mdir::driver::builder::buildFrameBatchPMEConstants(OpBuilder &builder,
    Value charges, Value output, Value rawVolume, Value rawFixed,
    const FrameBatchPMEConstants &config) {
  auto loc = builder.getUnknownLoc();
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  Value count = memref::DimOp::create(builder, loc, charges, zero);
  auto launch = gpu::LaunchOp::create(builder, loc, one, one, one, one, one, one);
  launch->setAttr("mdir.frame_host_arithmetic", builder.getUnitAttr());
  OpBuilder kernel = OpBuilder::atBlockEnd(&launch.getBody().front());
  auto real = [&](double value) -> Value {
    return arith::ConstantOp::create(kernel, loc, kernel.getF64FloatAttr(value));
  };
  Value nil = real(0);
  auto sums = scf::ForOp::create(kernel, loc, zero, count, one, ValueRange{nil, nil},
      [&](OpBuilder &at, Location, Value index, ValueRange carried) {
    Value q = memref::LoadOp::create(at, loc, charges, index);
    Value square = arith::MulFOp::create(at, loc, q, q);
    Value squares = arith::AddFOp::create(at, loc, carried[0], square);
    Value net = arith::AddFOp::create(at, loc, carried[1], q);
    scf::YieldOp::create(at, loc, ValueRange{squares, net});
  });
  Value self = arith::MulFOp::create(kernel, loc, real(config.selfMultiplier), sums.getResult(0));
  Value background = arith::MulFOp::create(kernel, loc, real(config.backgroundMultiplier), sums.getResult(1));
  background = arith::MulFOp::create(kernel, loc, background, sums.getResult(1));
  background = arith::MulFOp::create(kernel, loc, background, real(config.backgroundFactor));
  background = arith::DivFOp::create(kernel, loc, background, real(config.volume));
  Value total = arith::AddFOp::create(kernel, loc, self, background);
  Value virial = arith::MulFOp::create(kernel, loc, real(3), background);
  memref::StoreOp::create(kernel, loc, self, output, zero);
  memref::StoreOp::create(kernel, loc, total, output, one);
  Value two = arith::ConstantIndexOp::create(kernel, loc, 2);
  memref::StoreOp::create(kernel, loc, virial, output, two);
  Value volume = arith::MulFOp::create(kernel, loc,
      real(config.gradientBackgroundMultiplier), sums.getResult(1));
  volume = arith::MulFOp::create(kernel, loc, volume, real(config.backgroundFactor));
  volume = arith::DivFOp::create(kernel, loc, volume, real(config.volume));
  Value multiplier = real(config.gradientSelfMultiplier);
  Value base = arith::ConstantIndexOp::create(kernel, loc, config.rawBase);
  scf::ForOp::create(kernel, loc, zero, count, one, ValueRange{},
      [&](OpBuilder &at, Location, Value particle, ValueRange) {
    Value q = memref::LoadOp::create(at, loc, charges, particle);
    Value fixed = arith::MulFOp::create(at, loc, multiplier, q);
    Value index = arith::AddIOp::create(at, loc, base, particle);
    memref::StoreOp::create(at, loc, fixed, rawFixed, index);
    memref::StoreOp::create(at, loc, volume, rawVolume, index);
    scf::YieldOp::create(at, loc);
  });
  gpu::TerminatorOp::create(kernel, loc);
}
