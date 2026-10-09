// Makes every sine and cosine a call of its own to the C library (#243).

#include "mdir/Conversion/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;

namespace mdir {

#define GEN_PASS_DEF_SEPARATESINCOS
#include "mdir/Conversion/Passes.h.inc"

namespace {
class SeparateSinCos : public impl::SeparateSinCosBase<SeparateSinCos> {
public:
  void runOnOperation() final {
    ModuleOp module = getOperation();
    SmallVector<Operation *> found;
    module.walk([&](Operation *op) {
      if (isa<math::SinOp, math::CosOp>(op))
        found.push_back(op);
    });
    for (Operation *op : found) {
      Type type = op->getResult(0).getType();
      Type element = type;
      auto vector = dyn_cast<VectorType>(type);
      if (vector) {
        // A vector of a fixed length is computed element by element.
        if (vector.getRank() != 1 || vector.isScalable())
          continue;
        element = vector.getElementType();
      }
      if (!element.isF32() && !element.isF64())
        continue;
      std::string name = isa<math::SinOp>(op) ? "sin" : "cos";
      if (element.isF32())
        name += "f";
      func::FuncOp callee = getOrDeclare(module, name, element);
      // A symbol of that name that is not such a function: the op stays.
      if (!callee)
        continue;
      OpBuilder builder(op);
      Location loc = op->getLoc();
      Value argument = op->getOperand(0), result;
      auto call = [&](Value value) {
        return func::CallOp::create(builder, loc, callee, ValueRange{value})
            .getResult(0);
      };
      if (!vector) {
        result = call(argument);
      } else {
        SmallVector<Value> elements;
        for (int64_t i = 0, e = vector.getNumElements(); i != e; ++i)
          elements.push_back(
              call(vector::ExtractOp::create(builder, loc, argument, i)));
        result = vector::FromElementsOp::create(builder, loc, vector, elements);
      }
      op->getResult(0).replaceAllUsesWith(result);
      op->erase();
    }
  }

private:
  /// The declaration of the function `name` of the C library,
  /// `(real) -> real`, without attributes: the code generator makes the
  /// call of a function that it knows to read no memory a node that it
  /// may merge again.
  func::FuncOp getOrDeclare(ModuleOp module, StringRef name, Type real) {
    if (Operation *existing = module.lookupSymbol(name)) {
      auto function = dyn_cast<func::FuncOp>(existing);
      if (function &&
          function.getFunctionType() ==
              FunctionType::get(module.getContext(), {real}, {real}))
        return function;
      return nullptr;
    }
    OpBuilder builder(module.getBody(), module.getBody()->begin());
    auto function = func::FuncOp::create(
        builder, module.getLoc(), name,
        builder.getFunctionType({real}, {real}));
    function.setPrivate();
    return function;
  }
};
} // namespace

} // namespace mdir
