// Native construction of the explicit window, before list specialization.
#include "../../lib/Driver/WaterTemplateIR.h"
#include "mdir/Driver/WaterTemplate.h"
#include "mdir/Dialect/MD/Transforms/DerivativeInterface.h"
#include "mdir/Dialect/MD/Transforms/ScalarDerivative.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MD/MDDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <tuple>

using namespace mlir;
using namespace mdir::driver;
using namespace mdir::driver::builder;

static void require(bool condition, const char *message) {
  if (!condition) { llvm::errs() << message << '\n'; std::exit(1); }
}
void appendWaterTemplatePackingTests(OpBuilder &);

int runWaterTemplateIRTest() {
  DialectRegistry registry;
  mdir::md::registerDerivativeInterfaces(registry);
  MLIRContext context(registry);
  context.loadDialect<arith::ArithDialect, func::FuncDialect,
                      vector::VectorDialect, cf::ControlFlowDialect,
                      mdir::md::MDDialect>();
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
    model.epsilon = 0.0;
    auto crossOnly = buildSoluteWaterWindow(builder, model, block->getArgument(0),
        block->getArgument(5), block->getArgument(6), b);
    require(crossOnly.front().lennardJones, "zero O-O epsilon erased an unproved O-solute row");
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
  {
    namespace md = mdir::md;
    md::ParticleSetOp::create(builder, loc, "atoms");
    md::TupleSetOp::create(builder, loc, "water", "atoms", 4, md::Orientation::Ordered, true);
    for (auto [name, arity] : {std::pair<const char *, unsigned>{"ww", 6}, {"sw", 4}, {"ss", 2}})
      md::TupleSetOp::create(builder, loc, name, "atoms", arity,
          arity == 2 ? md::Orientation::Unordered : md::Orientation::Ordered, false);
    auto atoms = FlatSymbolRefAttr::get(&context, "atoms");
    auto positions = md::FieldType::get(&context, atoms, 3, builder.getF64Type());
    auto types = md::FieldType::get(&context, atoms, 1, builder.getI32Type());
    auto charges = md::FieldType::get(&context, atoms, 1, builder.getF64Type());
    auto domain = [&](StringRef name, unsigned arity) {
      return md::RelationType::get(&context, atoms, arity,
          arity == 2 ? md::Orientation::Unordered : md::Orientation::Ordered,
          FlatSymbolRefAttr::get(&context, name));
    };
    SmallVector<Type> arguments{positions, md::CellType::get(&context), types, charges, domain("water", 4)};
    WaterTemplateModel model;
    model.sites = 4; model.alpha = 0.75; model.extent = 0.1001;
    model.charges = {0.0, 0.5, 0.5, -1.0}; model.epsilon = 0.7;
    for (unsigned mask = 0; mask != 4; ++mask) {
      auto potential = md::PotentialOp::create(builder, loc,
          "factory_windows_" + std::to_string(mask),
          builder.getFunctionType(arguments, {builder.getF64Type()}),
          StringAttr{}, ArrayAttr{}, ArrayAttr{});
      SmallVector<Location> locations(arguments.size(), loc);
      auto *block = builder.createBlock(&potential.getBody(), {}, arguments, locations);
      auto domains = md::TemplatePairsOp::create(builder, loc, domain("ww", 6),
          domain("sw", 4), domain("ss", 2), block->getArgument(2),
          block->getArgument(4), Value());
      Value total = arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(0.0));
      for (auto [index, relation] : llvm::enumerate(domains.getResults())) {
        WaterTemplateReductionInputs input{relation, block->getArgument(0),
            block->getArgument(1), block->getArgument(2), block->getArgument(3),
            1.0, bool(mask & 1), bool(mask & 2)};
        Value term = buildWaterTemplateReduction(builder, model, input,
            [&](OpBuilder &b, const WaterTemplateSitePair &pair) -> Value {
              Value square = arith::MulFOp::create(b, loc, pair.displacement, pair.displacement);
              return vector::ReductionOp::create(b, loc, vector::CombiningKind::ADD, square);
            });
        if (auto sum = term.getDefiningOp<md::SumTemplateWindowOp>()) {
          unsigned expected = index == 0 ? (mask == 3 ? 10 : mask == 2 ? 9 : 1)
                              : index == 1 ? (mask == 3 ? 4 : mask == 2 ? 3 : 1) : 1;
          require(sum.getKernel().front().getTerminator()->getNumOperands() == 2 * expected,
                  "native reduction lost or duplicated a site channel");
        } else require(mask == 0, "nonempty reduction was erased");
        total = arith::AddFOp::create(builder, loc, total, term);
      }
      md::ReturnOp::create(builder, loc, total);
      builder.setInsertionPointAfter(potential);
    }
  }
  {
    auto vec = VectorType::get({3}, builder.getF64Type());
    for (auto [name, alpha, sites] : {
        std::tuple<const char *, double, unsigned>{"extent_half", 0.5, 4},
        {"extent_threequarters", 0.75, 4}, {"extent_three_site", 0.75, 3}}) {
      auto function = func::FuncOp::create(builder, loc, name,
          builder.getFunctionType({vec, vec}, {builder.getF64Type()}));
      auto *entry = function.addEntryBlock();
      builder.setInsertionPointToEnd(entry);
      WaterTemplateModel model;
      model.sites = sites; model.alpha = alpha;
      Value extent2 = buildWaterTemplateExtentSquared(builder, model,
          {entry->getArgument(0), entry->getArgument(1)});
      func::ReturnOp::create(builder, loc, extent2);
      builder.setInsertionPointAfter(function);
    }
    auto main = func::FuncOp::create(builder, loc, "water_extent_main",
                                     builder.getFunctionType({}, {}));
    builder.setInsertionPointToEnd(main.addEntryBlock());
    Value h = arith::ConstantOp::create(builder, loc,
        DenseElementsAttr::get(vec, ArrayRef<double>{0.125, 0.0, 0.0}));
    Value turned = arith::ConstantOp::create(builder, loc,
        DenseElementsAttr::get(vec, ArrayRef<double>{0.0, 0.125, 0.0}));
    // Independent exact dyadic answers: aligned H offsets give O-H^2=1/64
    // and the alpha=.75 site's squared extent 9/256. For perpendicular
    // offsets its squared extent is 9/512. Every arithmetic value is exact.
    auto checkExtent = [&](StringRef name, Value second, double reference) {
      Value actual = func::CallOp::create(builder, loc, name,
          TypeRange{builder.getF64Type()}, ValueRange{h, second}).getResult(0);
      Value expected = arith::ConstantOp::create(builder, loc,
                                                 builder.getF64FloatAttr(reference));
      Value equal = arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::OEQ,
                                          actual, expected);
      cf::AssertOp::create(builder, loc, equal, "water-template extent mismatch");
    };
    checkExtent("extent_half", h, 1.0 / 64.0);
    checkExtent("extent_threequarters", h, 9.0 / 256.0);
    checkExtent("extent_threequarters", turned, 9.0 / 512.0);
    checkExtent("extent_three_site", h, 1.0 / 64.0);
    func::ReturnOp::create(builder, loc);
    builder.setInsertionPointAfter(main);
  }
  appendWaterTemplatePackingTests(builder);
  require(succeeded(verify(module)), "window IR verification");
  module.print(llvm::outs());
  module.erase();
  return 0;
}
