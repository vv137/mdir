// Philox/Box-Muller noise [Salmon2011], preserving word and arithmetic order.
#include "NoiseIR.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include <array>

using namespace mlir;
using namespace mdir::driver::builder;

Value mdir::driver::builder::buildNoise(OpBuilder &builder,
                                        const NoiseIRInputs &in,
                                        const NoiseIRConfig &config) {
  auto loc = builder.getUnknownLoc();
  auto f64 = builder.getF64Type();
  auto vec = VectorType::get({3}, f64);
  auto map = mdir::md::MapParticlesOp::create(
      builder, loc, in.velocities.getType(), ValueRange{in.velocities, in.masses, in.ids});
  {
    OpBuilder::InsertionGuard guard(builder);
    auto args = builder.createBlock(&map.getKernel(), {},
                                      {vec, f64, builder.getI32Type()},
                                      {loc, loc, loc})->getArguments();
    auto integer = [&](uint32_t value) -> Value {
      return arith::ConstantIntOp::create(builder, loc, static_cast<int32_t>(value), 32);
    };
    auto real = [&](double value) -> Value {
      return arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(value));
    };
    auto mul = [&](Value a, Value b) -> Value {
      return arith::MulFOp::create(builder, loc, a, b);
    };
    auto add = [&](Value a, Value b) -> Value {
      return arith::AddFOp::create(builder, loc, a, b);
    };
    Value bits = arith::ConstantIntOp::create(builder, loc, 32, 64);
    Value shifted = arith::ShRUIOp::create(builder, loc, in.step, bits);
    Value c0 = arith::TruncIOp::create(builder, loc, builder.getI32Type(), in.step);
    Value c1 = arith::TruncIOp::create(builder, loc, builder.getI32Type(), shifted);
    Value c3 = integer(2u << 24);
    Value ma = integer(0xD2511F53u), mb = integer(0xCD9E8D57u);
    uint32_t k0 = static_cast<uint32_t>(config.seed);
    uint32_t k1 = static_cast<uint32_t>(config.seed >> 32);
    Value c2 = args[2];
    for (int round = 0; round != 10; ++round) {
      if (round != 0) {
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
      }
      auto first = arith::MulUIExtendedOp::create(builder, loc, ma, c0);
      auto second = arith::MulUIExtendedOp::create(builder, loc, mb, c2);
      Value key0 = integer(k0), key1 = integer(k1);
      Value a = arith::XOrIOp::create(builder, loc, second.getHigh(), c1);
      Value b = arith::XOrIOp::create(builder, loc, a, key0);
      Value c = arith::XOrIOp::create(builder, loc, first.getHigh(), c3);
      Value d = arith::XOrIOp::create(builder, loc, c, key1);
      c0 = b;
      c1 = second.getLow();
      c2 = d;
      c3 = first.getLow();
    }
    Value half = real(0.5), scale = real(0x1.0p-32), minusTwo = real(-2.0),
          twoPi = real(6.283185307179586);
    std::array<Value, 4> words = {c0, c1, c2, c3}, uniform;
    for (int word = 0; word != 4; ++word) {
      Value floating = arith::UIToFPOp::create(builder, loc, f64, words[word]);
      Value centered = add(floating, half);
      uniform[word] = mul(centered, scale);
    }
    std::array<Value, 2> cosines;
    Value sine;
    for (int pair = 0; pair != 2; ++pair) {
      Value logarithm = math::LogOp::create(builder, loc, uniform[2 * pair]);
      Value square = mul(minusTwo, logarithm);
      Value radius = math::SqrtOp::create(builder, loc, square);
      Value theta = mul(twoPi, uniform[2 * pair + 1]);
      Value cosine = math::CosOp::create(builder, loc, theta);
      cosines[pair] = mul(radius, cosine);
      if (pair == 0) {
        Value value = math::SinOp::create(builder, loc, theta);
        sine = mul(radius, value);
      }
    }
    Value noise = vector::FromElementsOp::create(
        builder, loc, vec, ValueRange{cosines[0], sine, cosines[1]});
    Value zero = real(0.0);
    Value massive = arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::OGT,
                                          args[1], zero);
    Value one = real(1.0);
    Value safe = arith::SelectOp::create(builder, loc, massive, args[1], one);
    Value spread = real(config.spread);
    Value root = math::SqrtOp::create(builder, loc, safe);
    Value sigma = arith::DivFOp::create(builder, loc, spread, root);
    Value broadcast = vector::BroadcastOp::create(builder, loc, vec, sigma);
    Value kick = mul(broadcast, noise);
    Value damping = arith::ConstantOp::create(
        builder, loc, DenseElementsAttr::get(vec, config.damping));
    Value damped = mul(damping, args[0]);
    Value updated = add(damped, kick);
    Value result = arith::SelectOp::create(builder, loc, massive, updated, args[0]);
    mdir::md::YieldOp::create(builder, loc, ValueRange{result});
  }
  return map.getResult();
}

Value mdir::driver::builder::buildBrownianDrift(
    OpBuilder &builder, Value forces, Value masses, double gamma) {
  auto loc = builder.getUnknownLoc();
  auto f64 = builder.getF64Type();
  auto vec = VectorType::get({3}, f64);
  auto map = mdir::md::MapParticlesOp::create(
      builder, loc, forces.getType(), ValueRange{forces, masses});
  {
    OpBuilder::InsertionGuard guard(builder);
    auto args = builder.createBlock(&map.getKernel(), {}, {vec, f64},
                                      {loc, loc})->getArguments();
    auto real = [&](double value) -> Value {
      return arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(value));
    };
    Value zero = real(0.0);
    Value massive = arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::OGT,
                                          args[1], zero);
    Value one = real(1.0);
    Value safe = arith::SelectOp::create(builder, loc, massive, args[1], one);
    Value friction = real(gamma);
    Value mass = arith::MulFOp::create(builder, loc, safe, friction);
    Value broadcast = vector::BroadcastOp::create(builder, loc, vec, mass);
    Value drift = arith::DivFOp::create(builder, loc, args[0], broadcast);
    Value still = arith::ConstantOp::create(builder, loc, DenseElementsAttr::get(vec, 0.0));
    Value result = arith::SelectOp::create(builder, loc, massive, drift, still);
    mdir::md::YieldOp::create(builder, loc, ValueRange{result});
  }
  return map.getResult();
}
