// Executes the actual window body and cutoff emitter independently of lists.
#include "mdir/Dialect/MD/MDDialect.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MD/Transforms/TemplateWindows.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace mdir::md;
namespace {
struct TestTemplateWindows
    : PassWrapper<TestTemplateWindows, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTemplateWindows)
  StringRef getArgument() const final { return "test-md-template-windows"; }
  StringRef getDescription() const final {
    return "Outlines proved template windows with their actual site guards";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    scf::SCFDialect, vector::VectorDialect>();
  }
  void runOnOperation() final {
    ModuleOp module = getOperation();
    SmallVector<Operation *> windows;
    module.walk([&](FunctionOp function) {
      function.walk([&](Operation *op) {
        if (isa<SumTemplateWindowOp, GatherTemplateWindowOp>(op))
          windows.push_back(op);
      });
    });
    for (Operation *op : windows) {
      Block &body = op->getRegion(0).front();
      auto yield = cast<YieldOp>(body.getTerminator());
      bool gather = isa<GatherTemplateWindowOp>(op);
      bool tensor = isa<VectorType>(op->getResult(0).getType());
      StringRef name = gather ? "window_forces" : tensor ? "window_virial" : "window_energy";
      unsigned stride = gather
          ? cast<RelationType>(op->getOperand(0).getType()).getArity() : 1;
      unsigned pairs = yield.getNumOperands() / (stride + 1);
      SmallVector<Type> inputs(body.getArgumentTypes());
      llvm::SetVector<Value> outer;
      for (Operation &inner : body)
        for (Value operand : inner.getOperands())
          if (operand.getParentBlock() != &body) outer.insert(operand);
      for (Value value : outer) inputs.push_back(value.getType());
      SmallVector<Type> outputs;
      for (Value value : yield.getOperands().take_front(stride * pairs))
        outputs.push_back(value.getType());
      FunctionType type = FunctionType::get(module.getContext(), inputs, outputs);
      auto existing = SymbolTable::lookupSymbolIn(module, name);
      auto function = dyn_cast_or_null<func::FuncOp>(existing);
      if (existing && (!function || !function.isExternal() || function.getFunctionType() != type)) {
        op->emitError("test window declaration has the wrong type or a duplicate definition");
        return signalPassFailure();
      }
      if (!function) {
        function = func::FuncOp::create(op->getLoc(), name, type);
        module.push_back(function);
      }
      Block *entry = function.addEntryBlock();
      OpBuilder builder = OpBuilder::atBlockEnd(entry);
      IRMapping mapping;
      unsigned index = 0;
      for (Value argument : body.getArguments())
        mapping.map(argument, entry->getArgument(index++));
      for (Value value : outer) mapping.map(value, entry->getArgument(index++));
      auto values = emitTemplateWindowBody(builder, op, mapping);
      func::ReturnOp::create(builder, op->getLoc(), values);
    }
    for (Operation &op : llvm::make_early_inc_range(module))
      if (isa<MDDialect>(op.getDialect())) op.erase();
  }
};
} // namespace
namespace mdir::test {
void registerTestTemplateWindows() { PassRegistration<TestTemplateWindows>(); }
}
