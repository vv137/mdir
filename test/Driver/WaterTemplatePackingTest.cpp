// The actual packing/coordinate emitter, executable without a device.
#include "mdir/Conversion/TemplateWindowKernels.h"
#include "mdir/Conversion/MDExecKernels.h"
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

  // Dynamic arguments keep whole-frame reconstruction in the executable
  // helper, rather than evaluating these answers in the test generator.
  for (bool tilted : {false, true}) {
    Type cellType = VectorType::get({tilted ? 6 : 3}, builder.getF64Type());
    auto prepare = func::FuncOp::create(builder, loc,
        tilted ? "prepare_template_triclinic" : "prepare_template_orthorhombic",
        builder.getFunctionType({wide, wide, wide, cellType},
                                 {narrow, narrow, narrow, wide}));
    auto *entry = prepare.addEntryBlock();
    builder.setInsertionPointToEnd(entry);
    SmallVector<Value> positions(entry->getArguments().take_front(3));
    Value cell = entry->getArgument(3);
    auto prepared = emitPrepareTemplateFrame(builder, loc, positions, cell,
        createInverse(builder, loc, cell), builder.getF32Type());
    func::ReturnOp::create(builder, loc, ValueRange{prepared.packed[0],
        prepared.packed[1], prepared.packed[2], prepared.latticeShift});
    builder.setInsertionPointAfter(prepare);
  }
  auto prepareMain = func::FuncOp::create(builder, loc, "water_prepare_main",
                                           builder.getFunctionType({}, {}));
  builder.setInsertionPointToEnd(prepareMain.addEntryBlock());
  for (bool tilted : {false, true}) {
    SmallVector<double> edges = tilted ? SmallVector<double>{4, 4, 4, 1, .5, 1}
                                       : SmallVector<double>{4, 4, 4};
    Value cell = arith::ConstantOp::create(builder, loc,
        DenseElementsAttr::get(VectorType::get({int64_t(edges.size())},
            builder.getF64Type()), ArrayRef<double>(edges)));
    for (bool winding : {false, true}) {
      std::array<double, 3> oxygen, h1, h2, expectedO, expectedH1, expectedH2;
      if (winding) {
        oxygen = {base + .25, base + .25, base + .25};
        // Each hydrogen has an independent lattice image. The rational
        // tilted lattice makes every operation and expected answer exact.
        h1 = {base + .375 + (tilted ? 7.5 : 8),
              base + .25 + (tilted ? -3 : -4), base + 4.25};
        h2 = {base + .25 + (tilted ? -2.5 : -4),
              base + .375 + (tilted ? 7 : 8), base - 3.75};
        expectedO = {.25, .25, .25};
        expectedH1 = {.375, .25, .25};
        expectedH2 = {.25, .375, .25};
      } else if (tilted) {
        oxygen = {.25, 1.9375, .25};
        h1 = {-.75, -1.9375, .25};
        h2 = {.375, 1.9375, .25};
        expectedO = oxygen;
        expectedH1 = {.25, 2.0625, .25};
        expectedH2 = h2;
      } else {
        oxygen = {1.9375, 0, 0};
        h1 = {-1.9375, 0, 0};
        h2 = {1.9375, .125, 0};
        expectedO = oxygen;
        expectedH1 = {2.0625, 0, 0};
        expectedH2 = h2;
      }
      auto prepared = func::CallOp::create(builder, loc,
          tilted ? "prepare_template_triclinic" : "prepare_template_orthorhombic",
          TypeRange{narrow, narrow, narrow, wide},
          ValueRange{vector(oxygen), vector(h1), vector(h2), cell});
      check(prepared.getResult(0), expectedO);
      check(prepared.getResult(1), expectedH1);
      check(prepared.getResult(2), expectedH2);
      for (unsigned axis = 0; axis != 3; ++axis) {
        Value actual = vector::ExtractOp::create(builder, loc,
                                                 prepared.getResult(3), axis);
        Value expected = createReal(builder, loc, builder.getF64Type(),
                                     winding ? -base : 0);
        Value same = arith::CmpFOp::create(builder, loc,
                                            arith::CmpFPredicate::OEQ, actual, expected);
        cf::AssertOp::create(builder, loc, same, "template lattice shift mismatch");
      }
    }
  }
  func::ReturnOp::create(builder, loc);
  builder.setInsertionPointAfter(prepareMain);
}
