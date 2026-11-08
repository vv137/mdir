// The actual packing/coordinate emitter, executable without a device.
#include "mdir/Conversion/TemplateWindowKernels.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace mlir;
using namespace mdir::kernels;

void appendWaterTemplatePackingTests(OpBuilder &builder) {
  auto loc = builder.getUnknownLoc();
  auto wide = VectorType::get({3}, builder.getF64Type());
  auto narrow = VectorType::get({3}, builder.getF32Type());
  auto pack = func::FuncOp::create(builder, loc, "pack_template_frame",
      builder.getFunctionType({wide, wide, wide, wide}, {narrow, narrow, narrow}));
  auto *block = pack.addEntryBlock();
  builder.setInsertionPointToEnd(block);
  SmallVector<Value> whole(block->getArguments().take_front(3));
  auto packed = emitPackTemplateFrame(builder, loc, whole,
      block->getArgument(3), builder.getF32Type());
  func::ReturnOp::create(builder, loc, ValueRange(packed));
  builder.setInsertionPointAfter(pack);

  auto main = func::FuncOp::create(builder, loc, "water_pack_main",
                                   builder.getFunctionType({}, {}));
  builder.setInsertionPointToEnd(main.addEntryBlock());
  auto vector = [&](std::array<double, 3> values) -> Value {
    return arith::ConstantOp::create(builder, loc,
        DenseElementsAttr::get(wide, ArrayRef<double>(values)));
  };
  // At 2^30, f32 cannot retain a 1/8 local offset. Applying the lattice
  // shift in f64 first must retain it exactly. This is D95's ordering.
  constexpr double base = 1073741824.0;
  Value shift = vector({-base, -base, -base});
  auto frame = [&](double x) {
    SmallVector<Value> arguments{
        vector({base + x, base, base}),
        vector({base + x + 0.125, base, base}),
        vector({base + x, base + 0.125, base}), shift};
    auto call = func::CallOp::create(builder, loc, "pack_template_frame",
        TypeRange{narrow, narrow, narrow}, arguments);
    return SmallVector<Value>(call.getResults());
  };
  auto first = frame(0.0), second = frame(0.5);
  auto check = [&](Value actual, std::array<double, 3> expected) {
    for (unsigned axis = 0; axis != 3; ++axis) {
      Value component = vector::ExtractOp::create(builder, loc, actual, axis);
      Value reference = arith::ConstantOp::create(builder, loc,
          builder.getF32FloatAttr(expected[axis]));
      Value equal = arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::OEQ,
                                          component, reference);
      cf::AssertOp::create(builder, loc, equal, "template packing/coordinate mismatch");
    }
  };
  check(first[0], {0.0, 0.0, 0.0});
  check(first[1], {0.125, 0.0, 0.0});
  check(first[2], {0.0, 0.125, 0.0});
  check(second[0], {0.5, 0.0, 0.0});
  check(second[1], {0.625, 0.0, 0.0});
  check(second[2], {0.5, 0.125, 0.0});
  Value image = arith::ConstantOp::create(builder, loc,
      DenseElementsAttr::get(narrow, ArrayRef<float>{-2.0f, 0.0f, 0.0f}));
  for (auto kind : {TemplatePairClass::WaterWater,
                    TemplatePairClass::SoluteWater, TemplatePairClass::SoluteSolute}) {
    ArrayRef<Value> left(first), right(second);
    if (kind != TemplatePairClass::WaterWater) left = left.take_front(1);
    if (kind == TemplatePairClass::SoluteSolute) right = right.take_front(1);
    auto coordinates = emitTemplateWindowCoordinates(builder, loc, kind, left, right, image);
    check(coordinates[0], {1.5, 0.0, 0.0});
    for (unsigned i = 1; i < coordinates.size(); ++i)
      check(coordinates[i], i % 2 ? std::array<double, 3>{0.125, 0.0, 0.0}
                                  : std::array<double, 3>{0.0, 0.125, 0.0});
  }
  func::ReturnOp::create(builder, loc);
  builder.setInsertionPointAfter(main);
}
