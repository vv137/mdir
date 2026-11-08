// Straight-line site geometry, kept visible to differentiation and lowering.
#include "WaterTemplateIR.h"
#include "mdir/Driver/WaterTemplate.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include <array>

using namespace mlir;
using namespace mdir::driver;
using namespace mdir::driver::builder;

namespace {
struct WindowBuilder {
  OpBuilder &builder;
  Location loc;
  VectorType vector;
  Type real;

  WindowBuilder(OpBuilder &builder, Value displacement)
      : builder(builder), loc(builder.getUnknownLoc()),
        vector(cast<VectorType>(displacement.getType())),
        real(vector.getElementType()) {}
  Value constant(double value) {
    return arith::ConstantOp::create(builder, loc,
                                    FloatAttr::get(real, value));
  }
  Value type(unsigned value) {
    return arith::ConstantIntOp::create(builder, loc, value, 32);
  }
  Value zero() {
    return arith::ConstantOp::create(
        builder, loc, DenseElementsAttr::get(vector, builder.getZeroAttr(real)));
  }
  Value add(Value a, Value b) {
    return arith::AddFOp::create(builder, loc, a, b);
  }
  Value sub(Value a, Value b) {
    return arith::SubFOp::create(builder, loc, a, b);
  }
  std::array<Value, 4> offsets(const WaterTemplateModel &model,
                              const WaterTemplateFrame &frame) {
    std::array<Value, 4> values{zero(), frame.h1, frame.h2, {}};
    if (model.sites == 4) {
      Value weight = vector::BroadcastOp::create(
          builder, loc, vector, constant(model.alpha));
      values[3] = arith::MulFOp::create(
          builder, loc, weight, add(frame.h1, frame.h2));
    }
    return values;
  }
};
} // namespace

llvm::SmallVector<WaterTemplateSitePair, 10>
mdir::driver::builder::buildWaterWaterWindow(
    OpBuilder &builder, const WaterTemplateModel &model, Value base,
    const WaterTemplateFrame &a, const WaterTemplateFrame &b) {
  WindowBuilder window(builder, base);
  auto offsetsA = window.offsets(model, a), offsetsB = window.offsets(model, b);
  std::array<Value, 4> types, charges;
  for (unsigned i = 0; i != model.sites; ++i) {
    types[i] = window.type(model.types[i]);
    charges[i] = window.constant(model.charges[i]);
  }
  llvm::SmallVector<WaterTemplateSitePair, 10> pairs;
  for (unsigned i = 0; i != model.sites; ++i)
    for (unsigned j = 0; j != model.sites; ++j) {
      bool lj = i == 0 && j == 0 && model.epsilon != 0.0;
      bool q = model.charges[i] != 0.0 && model.charges[j] != 0.0;
      if (!lj && !q)
        continue;
      // Do not reassociate this as base + (offsetA - offsetB): the window
      // arithmetic and its mixed-precision rounding are part of the contract.
      Value displacement = window.sub(window.add(base, offsetsA[i]), offsetsB[j]);
      pairs.push_back({displacement, types[i], types[j], charges[i], charges[j], lj, q});
    }
  return pairs;
}

llvm::SmallVector<WaterTemplateSitePair, 4>
mdir::driver::builder::buildSoluteWaterWindow(
    OpBuilder &builder, const WaterTemplateModel &model, Value base,
    Value soluteType, Value soluteCharge, const WaterTemplateFrame &frame) {
  WindowBuilder window(builder, base);
  auto offsets = window.offsets(model, frame);
  llvm::SmallVector<WaterTemplateSitePair, 4> pairs;
  for (unsigned j = 0; j != model.sites; ++j) {
    bool lj = j == 0 && model.epsilon != 0.0;
    bool q = model.charges[j] != 0.0;
    if (!lj && !q)
      continue;
    pairs.push_back({window.sub(base, offsets[j]), soluteType,
                     window.type(model.types[j]), soluteCharge,
                     window.constant(model.charges[j]), lj, q});
  }
  return pairs;
}
