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

  // Retained build frames must survive a central-image face crossing and
  // scale their integer shift with a changed cell. These dynamic helpers
  // exercise the production scalar emitters without decimal serialization.
  for (bool tilted : {false, true}) {
    Type cellType = VectorType::get({tilted ? 6 : 3}, builder.getF64Type());
    auto choose = func::FuncOp::create(builder, loc,
        tilted ? "choose_template_cells_tilted" : "choose_template_cells_orthogonal",
        builder.getFunctionType({wide, cellType}, {wide}));
    auto *entry = choose.addEntryBlock();
    builder.setInsertionPointToEnd(entry);
    Value cell = entry->getArgument(1);
    Value cells = emitTemplatePackingCells(builder, loc, entry->getArgument(0),
        cell, createInverse(builder, loc, cell));
    func::ReturnOp::create(builder, loc, cells);
    builder.setInsertionPointAfter(choose);
    auto retain = func::FuncOp::create(builder, loc,
        tilted ? "retain_template_tilted" : "retain_template_orthogonal",
        builder.getFunctionType({wide, wide, wide, cellType, wide},
                                 {narrow, narrow, narrow, wide, wide}));
    entry = retain.addEntryBlock();
    builder.setInsertionPointToEnd(entry);
    cell = entry->getArgument(3);
    Value inverse = createInverse(builder, loc, cell);
    SmallVector<Value> positions(entry->getArguments().take_front(3));
    auto retained = emitRetainedTemplateFrame(builder, loc, positions, cell,
        inverse, entry->getArgument(4), builder.getF32Type());
    Value anchor = arith::AddFOp::create(builder, loc, positions[0], retained.latticeShift);
    Value fractional = emitTemplateFractional(builder, loc, anchor, cell, inverse);
    func::ReturnOp::create(builder, loc, ValueRange{retained.packed[0], retained.packed[1],
        retained.packed[2], fractional, retained.latticeShift});
    builder.setInsertionPointAfter(retain);
  }
  auto retainMain = func::FuncOp::create(builder, loc, "water_retained_main",
                                         builder.getFunctionType({}, {}));
  builder.setInsertionPointToEnd(retainMain.addEntryBlock());
  auto checkWide = [&](Value actual, std::array<double, 3> expected) {
    for (unsigned axis = 0; axis != 3; ++axis) {
      Value component = vector::ExtractOp::create(builder, loc, actual, axis);
      Value reference = createReal(builder, loc, builder.getF64Type(), expected[axis]);
      Value same = arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::OEQ,
                                         component, reference);
      cf::AssertOp::create(builder, loc, same, "retained template fractional frame mismatch");
    }
  };
  for (bool tilted : {false, true}) {
    auto makeCell = [&](double scale) -> Value {
      SmallVector<double> edges{4 * scale, 4 * scale, 4 * scale};
      if (tilted) edges.append({scale, .5 * scale, scale});
      return arith::ConstantOp::create(builder, loc,
          DenseElementsAttr::get(VectorType::get({int64_t(edges.size())},
              builder.getF64Type()), ArrayRef<double>(edges)));
    };
    // Independently choose an exact winding (17,-9,5). Matrix products in
    // this generator form inputs only, never the tested helper's result.
    auto translate = [&](std::array<double, 3> winding, double scale) {
      return std::array<double, 3>{
        scale * (4 * winding[0] + (tilted ? winding[1] + .5 * winding[2] : 0)),
        scale * (4 * winding[1] + (tilted ? winding[2] : 0)),
        scale * 4 * winding[2]};
    };
    auto offset = translate({17, -9, 5}, 1);
    Value initial = vector({offset[0] + 1.9375, offset[1] + .25, offset[2] + .25});
    Value cells = func::CallOp::create(builder, loc,
        tilted ? "choose_template_cells_tilted" : "choose_template_cells_orthogonal",
        TypeRange{wide}, ValueRange{initial, makeCell(1)}).getResult(0);
    for (double scale : {1.0, 2.0}) {
      checkWide(cells, {-17, 9, -5});
      offset = translate({17, -9, 5}, scale);
      std::array<double, 3> expected{2.0625 * scale, .25 * scale, .25 * scale};
      std::array<double, 3> oxygen;
      for (unsigned k = 0; k != 3; ++k) oxygen[k] = offset[k] + expected[k];
      auto image1 = translate({0, 1, -1}, scale);
      auto image2 = translate({1, 0, 1}, scale);
      std::array<double, 3> hydrogen1, hydrogen2;
      for (unsigned k = 0; k != 3; ++k) {
        hydrogen1[k] = oxygen[k] + image1[k] + (k == 0 ? .125 : 0);
        hydrogen2[k] = oxygen[k] + image2[k] + (k == 1 ? .125 : 0);
      }
      auto call = func::CallOp::create(builder, loc,
          tilted ? "retain_template_tilted" : "retain_template_orthogonal",
          TypeRange{narrow, narrow, narrow, wide, wide},
          ValueRange{vector(oxygen), vector(hydrogen1), vector(hydrogen2), makeCell(scale), cells});
      check(call.getResult(0), expected);
      auto h1 = expected, h2 = expected;
      h1[0] += .125; h2[1] += .125;
      check(call.getResult(1), h1); check(call.getResult(2), h2);
      checkWide(call.getResult(3), tilted ? std::array<double, 3>{.49609375, .046875, .0625}
                                         : std::array<double, 3>{.515625, .0625, .0625});
      checkWide(call.getResult(4), {-offset[0], -offset[1], -offset[2]});
    }
  }
  func::ReturnOp::create(builder, loc);
  builder.setInsertionPointAfter(retainMain);
}
