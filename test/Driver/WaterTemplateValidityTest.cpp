// Executable checks of the production cell-deformation lower bound.
#include "mdir/Conversion/TemplateValidityKernels.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>
#include <cstdlib>
using namespace mlir;
using namespace mdir::kernels;

void appendWaterTemplateValidityTests(OpBuilder &b) {
  Location loc = b.getUnknownLoc();
  for (int64_t width : {3, 6}) {
    auto cell = VectorType::get({width}, b.getF64Type());
    auto f = func::FuncOp::create(b, loc, width == 3 ? "template_scale_rectangular" : "template_scale_tilted",
        b.getFunctionType({cell, cell}, {b.getF64Type()}));
    auto *entry = f.addEntryBlock();
    b.setInsertionPointToEnd(entry);
    Value lower = emitTemplateCellScaleLower(b, loc, entry->getArgument(0), entry->getArgument(1));
    func::ReturnOp::create(b, loc, lower);
    b.setInsertionPointAfter(f);
    auto point = VectorType::get({3}, b.getF64Type());
    auto limits = func::FuncOp::create(b, loc,
        width == 3 ? "template_limits_rectangular" : "template_limits_tilted",
        b.getFunctionType({cell, cell, cell, cell, point, point, b.getF64Type()},
                           {b.getF64Type(), b.getF64Type()}));
    entry = limits.addEntryBlock();
    b.setInsertionPointToEnd(entry);
    SmallVector<Value> cells{entry->getArgument(1), entry->getArgument(2), entry->getArgument(3)};
    auto budgets = emitTemplateGroupMotionLimits(b, loc, entry->getArgument(0), cells,
        entry->getArgument(4), entry->getArgument(5), entry->getArgument(6));
    func::ReturnOp::create(b, loc, ValueRange{budgets[0], budgets[1]});
    b.setInsertionPointAfter(limits);
    auto motion = func::FuncOp::create(b, loc,
        width == 3 ? "template_motion_rectangular" : "template_motion_tilted",
        b.getFunctionType({point, point, cell, b.getF64Type()}, {b.getI1Type()}));
    entry = motion.addEntryBlock();
    b.setInsertionPointToEnd(entry);
    Value moved = emitTemplateMotionExceeded(b, loc, entry->getArgument(0),
        entry->getArgument(1), entry->getArgument(2), entry->getArgument(3));
    func::ReturnOp::create(b, loc, moved);
    b.setInsertionPointAfter(motion);
  }
  auto print = func::FuncOp::create(b, loc, "printI64", b.getFunctionType({b.getI64Type()}, {}));
  print.setPrivate();
  auto newline = func::FuncOp::create(b, loc, "printNewline", b.getFunctionType({}, {}));
  newline.setPrivate();
  auto main = func::FuncOp::create(b, loc, "water_scale_main", b.getFunctionType({}, {}));
  b.setInsertionPointToEnd(main.addEntryBlock());
  unsigned number = 0;
  auto check = [&](ArrayRef<double> current, ArrayRef<double> reference, double expected, bool exact = false) {
    auto type = VectorType::get({int64_t(current.size())}, b.getF64Type());
    Value now = arith::ConstantOp::create(b, loc, DenseElementsAttr::get(type, current));
    Value old = arith::ConstantOp::create(b, loc, DenseElementsAttr::get(type, reference));
    Value actual = func::CallOp::create(b, loc,
        current.size() == 3 ? "template_scale_rectangular" : "template_scale_tilted",
        TypeRange{b.getF64Type()}, ValueRange{now, old}).getResult(0);
    Value lo = createReal(b, loc, b.getF64Type(), exact ? expected : expected - 1e-11);
    Value hi = createReal(b, loc, b.getF64Type(), expected);
    Value good = arith::AndIOp::create(b, loc,
        arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OGE, actual, lo),
        arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLE, actual, hi));
    Value bad = arith::XOrIOp::create(b, loc, good,
        arith::ConstantOp::create(b, loc, b.getBoolAttr(true)));
    scf::IfOp::create(b, loc, bad, [&](OpBuilder &out, Location) {
      Value label = arith::ConstantIntOp::create(out, loc, number, 64);
      func::CallOp::create(out, loc, print, ValueRange{label});
      func::CallOp::create(out, loc, newline, ValueRange{});
      for (Value v : {actual, lo, hi}) {
        Value bits = arith::BitcastOp::create(out, loc, out.getI64Type(), v);
        func::CallOp::create(out, loc, print, ValueRange{bits});
        func::CallOp::create(out, loc, newline, ValueRange{});
      }
      scf::YieldOp::create(out, loc);
    });
    cf::AssertOp::create(b, loc, good, "template affine-scale lower bound differs");
    ++number;
  };
  check({4, 4, 4}, {4, 4, 4}, 1, true);
  check({4, 4, 4, 1, .5, 1}, {4, 4, 4, 1, .5, 1}, 1, true);
  check({8, 4, 2}, {4, 4, 4}, .5);
  check({8, 8, 8, 2, 1, 2}, {4, 4, 4, 1, .5, 1}, 2);
  check({8, 4, 2, 2, 1, 1}, {4, 4, 4, 1, .5, 1}, .5);
  // For unit simple shear, the inverse norm bound is 1/(1+|s|).
  check({4, 4, 4, 1, 0, 0}, {4, 4, 4, 0, 0, 0}, .8);
  check({4, 4, 4, -1, 0, 0}, {4, 4, 4, 0, 0, 0}, .8);
  check({4, 4, 4, 1, .5, 1}, {4, 4, 4, 0, 0, 0}, 16.0 / 21);
  check({4, 4, 4, 0, 0, 0}, {4, 4, 4, 1, .5, 1}, 8.0 / 11);
  check({8, 4, 4, 4, 0, 0}, {4, 4, 4, 0, 0, 0}, std::sqrt(2.0 / 3));
  func::ReturnOp::create(b, loc);
  b.setInsertionPointAfter(main);

  auto vector = [&](ArrayRef<double> values) -> Value {
    return arith::ConstantOp::create(b, loc,
        DenseElementsAttr::get(VectorType::get({int64_t(values.size())}, b.getF64Type()), values));
  };
  auto require = [&](Value good, ValueRange values, StringRef message) {
    Value bad = arith::XOrIOp::create(b, loc, good,
        arith::ConstantOp::create(b, loc, b.getBoolAttr(true)));
    scf::IfOp::create(b, loc, bad, [&](OpBuilder &out, Location) {
      Value label = arith::ConstantIntOp::create(out, loc, number, 64);
      func::CallOp::create(out, loc, print, ValueRange{label});
      func::CallOp::create(out, loc, newline, ValueRange{});
      for (Value v : values) {
        Value bits = v.getType().isInteger(1)
            ? Value(arith::ExtUIOp::create(out, loc, out.getI64Type(), v))
            : Value(arith::BitcastOp::create(out, loc, out.getI64Type(), v));
        func::CallOp::create(out, loc, print, ValueRange{bits});
        func::CallOp::create(out, loc, newline, ValueRange{});
      }
      scf::YieldOp::create(out, loc);
    });
    cf::AssertOp::create(b, loc, good, message);
    ++number;
  };
  auto intervals = func::FuncOp::create(b, loc, "water_limits_main", b.getFunctionType({}, {}));
  b.setInsertionPointToEnd(intervals.addEntryBlock());
  Value current = vector({4, 4, 4}), physical = vector({1, .8, .6});
  Value skin = createReal(b, loc, b.getF64Type(), .4);
  Value built = emitTemplateFreshReaches(b, loc, physical, skin);
  auto group = [&](double ww, double sw, double ss) {
    auto call = func::CallOp::create(b, loc, "template_limits_rectangular",
        TypeRange{b.getF64Type(), b.getF64Type()},
        ValueRange{current, vector({ww, ww, ww}), vector({sw, sw, sw}),
                   vector({ss, ss, ss}), built, physical, skin});
    return std::array<Value, 2>{call.getResult(0), call.getResult(1)};
  };
  auto range = [&](Value actual, double expected) {
    Value lo = createReal(b, loc, b.getF64Type(), expected - 1e-11);
    Value hi = createReal(b, loc, b.getF64Type(), expected);
    Value good = arith::AndIOp::create(b, loc,
        arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OGE, actual, lo),
        arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLE, actual, hi));
    require(good, ValueRange{actual, lo, hi}, "template mobility-group half-margin differs");
  };
  for (auto [cells, expected] : {
      std::pair{std::array<double, 3>{4, 4, 4}, std::array<double, 2>{.2, .2}},
      std::pair{std::array<double, 3>{2, 2, 2}, std::array<double, 2>{.2, .2}},
      std::pair{std::array<double, 3>{8, 8, 8}, std::array<double, 2>{-.15, -.1}},
      std::pair{std::array<double, 3>{2, 4, 8}, std::array<double, 2>{.2, -.05}}}) {
    auto budget = group(cells[0], cells[1], cells[2]);
    range(budget[0], expected[0]); range(budget[1], expected[1]);
  }
  // A fresh SW epoch cannot invalidate an inactive group, whether the old
  // SW cell was expanded or contracted. Each keeps its own reference point.
  for (double oldSW : {2.0, 4.25}) {
    auto before = group(4, oldSW, 4), after = group(4, 4, 4);
    for (unsigned i = 0; i != 2; ++i) {
      Value safe = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OGE, after[i], before[i]);
      require(safe, ValueRange{before[i], after[i]}, "partial SW refresh shrank an inactive group allowance");
    }
  }
  func::ReturnOp::create(b, loc);
  b.setInsertionPointAfter(intervals);

  auto motion = func::FuncOp::create(b, loc, "water_motion_main", b.getFunctionType({}, {}));
  b.setInsertionPointToEnd(motion.addEntryBlock());
  auto moved = [&](ArrayRef<double> anchor, ArrayRef<double> reference,
                   ArrayRef<double> cell, double budget, bool expected) {
    Value result = func::CallOp::create(b, loc,
        cell.size() == 3 ? "template_motion_rectangular" : "template_motion_tilted",
        TypeRange{b.getI1Type()}, ValueRange{vector(anchor), vector(reference), vector(cell),
            createReal(b, loc, b.getF64Type(), budget)}).getResult(0);
    Value wanted = arith::ConstantOp::create(b, loc, b.getBoolAttr(expected));
    Value same = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, result, wanted);
    require(same, ValueRange{result, wanted}, "template displacement predicate differs");
  };
  moved({.25, .5, .75}, {.0625, .125, .1875}, {4, 4, 4}, .01, false);
  moved({.2578125, .5, .75}, {.0625, .125, .1875}, {4, 4, 4}, .01, false);
  moved({.265625, .5, .75}, {.0625, .125, .1875}, {4, 4, 4}, .01, true);
  moved({.25, .5, .75}, {.0625, .125, .1875}, {4, 4, 4}, -.01, true);
  moved({.25, .5, .75}, {.0625, .125, .1875}, {4, 4, 4}, 0, true);
  moved({.46875, .6875, .75}, {.0625, .125, .1875}, {4, 4, 4, 1, .5, 1}, .01, false);
  moved({.59375, .6875, .75}, {.0625, .125, .1875}, {4, 4, 4, 1, .5, 1}, .13, false);
  moved({.59375, .6875, .75}, {.0625, .125, .1875}, {4, 4, 4, 1, .5, 1}, .12, true);
  moved({-2.03125, .6875, .75}, {-.5625, .125, .1875}, {4, 4, 4, 1, .5, 1}, .01, false);
  moved({1.96875, .6875, .75}, {-.5625, .125, .1875}, {4, 4, 4, 1, .5, 1}, .01, true);
  func::ReturnOp::create(b, loc);
  b.setInsertionPointAfter(motion);
}

int runWaterTemplateValidityTest() {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, cf::ControlFlowDialect,
      func::FuncDialect, math::MathDialect, scf::SCFDialect, vector::VectorDialect>();
  OpBuilder builder(&context);
  auto module = ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(module.getBody());
  appendWaterTemplateValidityTests(builder);
  if (failed(verify(module))) {
    llvm::errs() << "template validity IR verification failed\n";
    std::exit(1);
  }
  module.print(llvm::outs());
  module.erase();
  return 0;
}
