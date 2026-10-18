// A test pass that writes the two sides of the adjoint identity of each
// neural stage (D288, docs/ml4-ir.md, Section 6):
//
//   <ȳ, J v> = <J^T ȳ, v>,
//
// J v from MDIR's forward rules (D183) applied to the forward body, and
// J^T ȳ from the vector-Jacobian product that the stage binds. The two
// come from different code, so that their agreement tests the binding.

#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MD/Transforms/ScalarDerivative.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mdir::md;

namespace {

/// For each `md.stage` with a vector-Jacobian product and bodies in the
/// module, creates `func.func @<stage>.adjoint`, or gives the body to a
/// declaration of that name, whose arguments are the
/// operands of the stage, then a direction v for each differentiable
/// operand, then a cotangent ȳ for each result, and which returns
/// <ȳ, J v> and <J^T ȳ, v>. The `md.stage` ops are then erased.
struct TestStageAdjoint
    : public PassWrapper<TestStageAdjoint, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestStageAdjoint)

  StringRef getArgument() const final { return "test-md-stage-adjoint"; }
  StringRef getDescription() const final {
    return "Writes both sides of the adjoint identity of each neural stage";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() override;
};

/// Clones the body of `function` at the end of `block` with its arguments
/// mapped to `arguments`, and returns the values that it returns.
SmallVector<Value> inlineBody(func::FuncOp function, Block &block,
                              ValueRange arguments) {
  IRMapping mapping;
  Block &body = function.getBody().front();
  mapping.map(body.getArguments(), arguments);
  OpBuilder builder = OpBuilder::atBlockEnd(&block);
  for (Operation &op : body.without_terminator())
    builder.clone(op, mapping);
  SmallVector<Value> results;
  for (Value value : body.getTerminator()->getOperands())
    results.push_back(mapping.lookupOrDefault(value));
  return results;
}

/// The inner product of `a` and `b`, scalars or vectors of f64; null for a
/// null `a` or `b`, which stands for zero.
Value dot(OpBuilder &builder, Location loc, Value a, Value b) {
  if (!a || !b)
    return Value();
  Value product = arith::MulFOp::create(builder, loc, a, b);
  if (!isa<VectorType>(product.getType()))
    return product;
  return vector::ReductionOp::create(builder, loc,
                                     vector::CombiningKind::ADD, product);
}

} // namespace

void TestStageAdjoint::runOnOperation() {
  ModuleOp module = getOperation();
  SmallVector<StageOp> stages(module.getOps<StageOp>());
  OpBuilder builder = OpBuilder::atBlockEnd(module.getBody());
  for (StageOp stage : stages) {
    if (!stage.getVjpAttr())
      continue;
    auto forward =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(stage,
                                                           stage.getForwardAttr());
    auto vjp = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        stage, stage.getVjpAttr());
    if (!forward || !vjp || forward.isExternal() || vjp.isExternal()) {
      stage.emitOpError() << "has no bodies in the module to test";
      return signalPassFailure();
    }
    Location loc = stage.getLoc();
    FunctionType type = stage.getFunctionType();
    SmallVector<Type> inputs(type.getInputs());
    SmallVector<unsigned> differentiable;
    for (unsigned i = 0, e = type.getNumInputs(); i != e; ++i)
      if (stage.isDifferentiable(i)) {
        differentiable.push_back(i);
        inputs.push_back(type.getInput(i));
      }
    llvm::append_range(inputs, type.getResults());
    Type f64 = builder.getF64Type();
    std::string name = (stage.getSymName() + ".adjoint").str();
    FunctionType functionType = builder.getFunctionType(inputs, {f64, f64});
    // A declaration of that name, which a caller needs to verify, receives
    // the body.
    auto function = module.lookupSymbol<func::FuncOp>(name);
    if (function) {
      if (!function.isExternal() || function.getFunctionType() != functionType) {
        function.emitOpError() << "expected a declaration of the type "
                               << functionType;
        return signalPassFailure();
      }
      function.setVisibility(SymbolTable::Visibility::Public);
    } else {
      function = func::FuncOp::create(builder, loc, name, functionType);
    }
    Block *block = function.addEntryBlock();
    unsigned operands = type.getNumInputs();
    ValueRange arguments = block->getArguments();
    ValueRange values = arguments.take_front(operands);
    ValueRange directions =
        arguments.slice(operands, differentiable.size());
    ValueRange cotangents =
        arguments.take_back(type.getNumResults());

    // <ȳ, J v>: the tangent of each result along v, from the forward body.
    SmallVector<Value> outputs = inlineBody(forward, *block, values);
    OpBuilder kernel = OpBuilder::atBlockEnd(block);
    ScalarEmitter emit(kernel, loc);
    Value left;
    for (auto [k, place] : llvm::enumerate(differentiable)) {
      ScalarDerivative derivative(kernel, values[place], nullptr,
                                  directions[k]);
      for (auto [output, cotangent] : llvm::zip_equal(outputs, cotangents)) {
        Value tangent;
        if (failed(derivative.get(output, tangent)))
          return signalPassFailure();
        left = emit.add(left, dot(kernel, loc, cotangent, tangent));
      }
    }

    // <J^T ȳ, v>: the bound product.
    SmallVector<Value> operandsAndCotangents(values);
    llvm::append_range(operandsAndCotangents, cotangents);
    SmallVector<Value> products =
        inlineBody(vjp, *block, operandsAndCotangents);
    kernel.setInsertionPointToEnd(block);
    Value right;
    for (auto [product, direction] : llvm::zip_equal(products, directions))
      right = emit.add(right, dot(kernel, loc, product, direction));

    Value zero = emit.constant(0.0, f64);
    func::ReturnOp::create(kernel, loc,
                           ValueRange{left ? left : zero,
                                      right ? right : zero});
  }
  // The declarations of the stages, which the lowering of functions does
  // not take; the bodies stay, as functions.
  for (StageOp stage : stages)
    stage.erase();
}

namespace mdir {
namespace test {
void registerTestStageAdjoint() { PassRegistration<TestStageAdjoint>(); }
} // namespace test
} // namespace mdir
