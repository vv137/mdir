#include "FrameBatchLJIR.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include <cmath>
using namespace mlir;
using namespace mdir::driver::builder;

llvm::Error mdir::driver::builder::buildFrameBatchLJConstants(
    OpBuilder &builder, Value sigma, Value epsilon, Value counts,
    Value excluded, Value output, const FrameBatchLJConstants &config) {
  auto vector = [&](Value value, Type element) {
    auto type = dyn_cast<MemRefType>(value.getType());
    return type && type.getRank() == 1 && type.getLayout().isIdentity() &&
        type.getMemorySpaceAsInt() == 1 && type.getElementType() == element;
  };
  if (!vector(sigma, builder.getF64Type()) || !vector(epsilon, builder.getF64Type()) ||
      !vector(counts, builder.getF64Type()) || !vector(excluded, builder.getI64Type()) ||
      !vector(output, builder.getF64Type()) || !config.types ||
      !std::isfinite(config.particles) || !std::isfinite(config.orderedPairs) ||
      !std::isfinite(config.cutoff) || !std::isfinite(config.volume) ||
      !(config.particles > 0) || !(config.cutoff > 0) || !(config.volume > 0))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
        "LJ frame constants require logical f64 device tables, counts and output, "
        "i64 exclusions, and positive type count, particle count, cutoff and volume");
  auto loc = builder.getUnknownLoc();
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  Value types = arith::ConstantIndexOp::create(builder, loc, config.types);
  Value excludedSize = memref::DimOp::create(builder, loc, excluded, zero);
  auto launch = gpu::LaunchOp::create(builder, loc, one, one, one, one, one, one);
  OpBuilder kernel = OpBuilder::atBlockEnd(&launch.getBody().front());
  auto real = [&](double value) -> Value {
    return arith::ConstantOp::create(kernel, loc, kernel.getF64FloatAttr(value));
  };
  Value nil = real(0), two = real(2), four = real(4);
  auto coefficient = [&](OpBuilder &at, Value a, Value b) {
    Value row = arith::MulIOp::create(at, loc, a, types);
    Value index = arith::AddIOp::create(at, loc, row, b);
    Value s = memref::LoadOp::create(at, loc, sigma, index);
    Value e = memref::LoadOp::create(at, loc, epsilon, index);
    // Integer powers are multiplications, never libdevice pow.
    Value s2 = arith::MulFOp::create(at, loc, s, s);
    Value s4 = arith::MulFOp::create(at, loc, s2, s2);
    Value s6 = arith::MulFOp::create(at, loc, s2, s4);
    Value factor = arith::MulFOp::create(at, loc, four, e);
    return Value(arith::MulFOp::create(at, loc, factor, s6));
  };
  auto rows = scf::ForOp::create(kernel, loc, zero, types, one, ValueRange{nil},
      [&](OpBuilder &row, Location, Value a, ValueRange carried) {
    Value na = memref::LoadOp::create(row, loc, counts, a);
    auto columns = scf::ForOp::create(row, loc, zero, types, one, carried,
        [&](OpBuilder &at, Location, Value b, ValueRange sum) {
      Value nb = memref::LoadOp::create(at, loc, counts, b);
      Value same = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::eq, a, b);
      Value realOne = arith::ConstantOp::create(at, loc, at.getF64FloatAttr(1));
      Value diagonal = arith::SelectOp::create(at, loc, same, realOne, nil);
      Value others = arith::SubFOp::create(at, loc, nb, diagonal);
      Value pairs = arith::MulFOp::create(at, loc, na, others);
      Value contribution = arith::MulFOp::create(at, loc, pairs, coefficient(at, a, b));
      Value total = arith::AddFOp::create(at, loc, sum[0], contribution);
      scf::YieldOp::create(at, loc, total);
    });
    scf::YieldOp::create(row, loc, columns.getResult(0));
  });
  Value step = arith::ConstantIndexOp::create(kernel, loc, 2);
  auto corrected = scf::ForOp::create(kernel, loc, zero, excludedSize, step,
      rows.getResults(), [&](OpBuilder &at, Location, Value index, ValueRange sum) {
    Value next = arith::AddIOp::create(at, loc, index, one);
    Value a = memref::LoadOp::create(at, loc, excluded, index);
    Value b = memref::LoadOp::create(at, loc, excluded, next);
    a = arith::IndexCastOp::create(at, loc, at.getIndexType(), a);
    b = arith::IndexCastOp::create(at, loc, at.getIndexType(), b);
    Value pair = coefficient(at, a, b);
    Value remove = arith::MulFOp::create(at, loc, two, pair);
    Value total = arith::SubFOp::create(at, loc, sum[0], remove);
    scf::YieldOp::create(at, loc, total);
  });
  Value mean = nil;
  if (config.orderedPairs > 0)
    mean = arith::DivFOp::create(kernel, loc, corrected.getResult(0), real(config.orderedPairs));
  Value factor = real(-2.0 * M_PI / (3.0 * config.volume) * config.particles * config.particles);
  Value numerator = arith::MulFOp::create(kernel, loc, factor, mean);
  Value rc3 = real(config.cutoff * config.cutoff * config.cutoff);
  Value tail = arith::DivFOp::create(kernel, loc, numerator, rc3);
  double shift = (1.0 - config.volume / (config.particles * 4.0 * M_PI / 3.0 *
      config.cutoff * config.cutoff * config.cutoff)) - 1.0;
  Value fixed = arith::MulFOp::create(kernel, loc, tail, real(shift));
  Value doubled = arith::AddFOp::create(kernel, loc, tail, tail);
  Value virial = arith::MulFOp::create(kernel, loc, real(6), tail);
  SmallVector<Value> results{config.shifted ? doubled : tail,
      config.shifted ? fixed : nil, config.shifted ? nil : tail,
      config.shifted ? nil : fixed, virial};
  for (auto [index, value] : llvm::enumerate(results)) {
    Value slot = arith::ConstantIndexOp::create(kernel, loc, index);
    memref::StoreOp::create(kernel, loc, value, output, slot);
  }
  gpu::TerminatorOp::create(kernel, loc);
  return llvm::Error::success();
}
