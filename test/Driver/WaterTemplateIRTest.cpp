// Native construction of the explicit window, before list specialization.
#include "../../lib/Driver/WaterTemplateIR.h"
#include "mdir/Driver/WaterTemplate.h"
#include "mdir/Dialect/MD/Transforms/DerivativeInterface.h"
#include "mdir/Dialect/MD/Transforms/ScalarDerivative.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>

using namespace mlir;
using namespace mdir::driver;
using namespace mdir::driver::builder;

static void require(bool condition, const char *message) {
  if (!condition) { llvm::errs() << message << '\n'; std::exit(1); }
}

int runWaterTemplateIRTest() {
  DialectRegistry registry;
  mdir::md::registerDerivativeInterfaces(registry);
  MLIRContext context(registry);
  context.loadDialect<arith::ArithDialect, func::FuncDialect,
                      vector::VectorDialect>();
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto module = ModuleOp::create(loc);
  builder.setInsertionPointToEnd(module.getBody());
  for (bool four : {false, true}) {
    auto vec = VectorType::get({3}, builder.getF64Type());
    SmallVector<Type> arguments(5, vec);
    arguments.push_back(builder.getI32Type());
    arguments.push_back(builder.getF64Type());
    auto function = func::FuncOp::create(
        builder, loc, four ? "four_site_window" : "three_site_window",
        builder.getFunctionType(arguments, {}));
    auto *block = function.addEntryBlock();
    builder.setInsertionPointToEnd(block);
    WaterTemplateModel model;
    model.sites = four ? 4 : 3;
    model.charges = {four ? 0.0 : -1.0, 0.5, 0.5, -1.0};
    model.types = {0, 1, 1, 2};
    model.alpha = 0.125;
    model.epsilon = 0.7;
    WaterTemplateFrame a{block->getArgument(1), block->getArgument(2)};
    WaterTemplateFrame b{block->getArgument(3), block->getArgument(4)};
    auto ww = buildWaterWaterWindow(builder, model, block->getArgument(0), a, b);
    auto sw = buildSoluteWaterWindow(builder, model, block->getArgument(0),
                                    block->getArgument(5), block->getArgument(6), b);
    require(ww.size() == (four ? 10 : 9), "WW structural zero elimination");
    require(sw.size() == model.sites, "SW structural zero elimination");
    unsigned lj = 0, q = 0;
    for (auto pair : ww) { lj += pair.lennardJones; q += pair.coulomb; }
    require(lj == 1 && q == 9, "WW term partition");
    lj = 0; q = 0;
    for (auto pair : sw) {
      require(pair.typeI == block->getArgument(5) &&
              pair.chargeI == block->getArgument(6), "solute parameters were folded");
      lj += pair.lennardJones; q += pair.coulomb;
    }
    require(lj == 1 && q == 3, "SW term partition");
    SmallVector<Value> returned;
    for (auto pair : ww) returned.push_back(pair.displacement);
    for (auto pair : sw) returned.push_back(pair.displacement);
    SmallVector<Type> resultTypes(returned.size(), vec);
    function.setFunctionType(builder.getFunctionType(arguments, resultTypes));
    func::ReturnOp::create(builder, loc, returned);
    builder.setInsertionPointAfter(function);
  }
  // An exact dyadic alpha makes the symbolic transpose coefficients exact:
  // (1 - 2 alpha, alpha, alpha) = (3/4, 1/8, 1/8) on each molecule.
  // Differentiate the emitted placement rather than coding a second force
  // formula. Stored sites are absent from both the arguments and results.
  {
    auto vec = VectorType::get({3}, builder.getF64Type());
    SmallVector<Type> types(6, vec);
    auto function = func::FuncOp::create(
        builder, loc, "constant_transpose", builder.getFunctionType(types, types));
    auto *block = function.addEntryBlock();
    builder.setInsertionPointToEnd(block);
    auto delta = [&](unsigned i, unsigned j) -> Value {
      return arith::SubFOp::create(builder, loc, block->getArgument(i),
                                   block->getArgument(j));
    };
    WaterTemplateModel model;
    model.sites = 4; model.alpha = 0.125;
    model.charges = {0.0, 0.5, 0.5, -1.0};
    model.epsilon = 0.7;
    auto pairs = buildWaterWaterWindow(builder, model, delta(0, 3),
                                       {delta(1, 0), delta(2, 0)},
                                       {delta(4, 3), delta(5, 3)});
    Value siteSite = pairs.back().displacement;
    SmallVector<Value> tangents;
    for (Value argument : block->getArguments()) {
      mdir::md::ScalarDerivative derivative(builder, argument);
      Value tangent;
      require(succeeded(derivative.get(siteSite, tangent)) && tangent,
              "constant transpose derivative");
      tangents.push_back(tangent);
    }
    func::ReturnOp::create(builder, loc, tangents);
    builder.setInsertionPointAfter(function);
  }
  require(succeeded(verify(module)), "window IR verification");
  module.print(llvm::outs());
  module.erase();
  return 0;
}
