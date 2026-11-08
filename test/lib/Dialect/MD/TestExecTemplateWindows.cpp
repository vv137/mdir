// Exercise the actual execution-body emitter independently of device lists.
#include "mdir/Conversion/TemplateWindowKernels.h"
#include "mdir/Dialect/MD/MDDialect.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
namespace {
struct TestExecTemplateWindows
    : PassWrapper<TestExecTemplateWindows, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestExecTemplateWindows)
  StringRef getArgument() const final { return "test-md-exec-template-windows"; }
  StringRef getDescription() const final {
    return "Outlines execution template windows with their actual site guards";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    scf::SCFDialect, vector::VectorDialect>();
  }
  void runOnOperation() final {
    ModuleOp module = getOperation();
    SmallVector<mdir::md_exec::TemplateForOp> windows;
    module.walk([&](mdir::md_exec::TemplateForOp op) { windows.push_back(op); });
    llvm::SetVector<Operation *> sources;
    for (auto op : windows) {
      if (op.getOuts().size() + op.getReduce().size() != 1 ||
          op.getPairs().size() != 3 || !op.getPairs()[0] ||
          op.getPairs()[1] || op.getPairs()[2]) {
        op.emitError("test outliner expects one active WW output per loop");
        return signalPassFailure();
      }
      sources.insert(op->getParentOfType<func::FuncOp>());
      Block &body = op.getWater().front();
      bool forces = !op.getOuts().empty();
      bool tensor = !forces && isa<VectorType>(op.getReduce()[0].getType());
      StringRef name = forces ? "window_forces" : tensor ? "window_virial" : "window_energy";
      SmallVector<Type> inputs(body.getArgumentTypes()), outputs;
      llvm::SetVector<Value> captures;
      body.walk([&](Operation *inner) {
        for (Value value : inner->getOperands())
          if (!body.getParent()->isAncestor(value.getParentRegion())) captures.insert(value);
      });
      for (Value value : captures) inputs.push_back(value.getType());
      if (forces)
        outputs.assign(6, mdir::md_exec::getKernelValueType(op.getOuts()[0].getType()));
      else
        outputs.push_back(op.getReduce()[0].getType());
      FunctionType type = FunctionType::get(module.getContext(), inputs, outputs);
      Operation *existing = SymbolTable::lookupSymbolIn(module, name);
      auto function = dyn_cast_or_null<func::FuncOp>(existing);
      if (existing && (!function || !function.isExternal() || function.getFunctionType() != type)) {
        op.emitError("execution test declaration has the wrong type or duplicate definition");
        return signalPassFailure();
      }
      if (!function) {
        function = func::FuncOp::create(op.getLoc(), name, type);
        module.push_back(function);
      }
      Block *entry = function.addEntryBlock();
      OpBuilder builder = OpBuilder::atBlockEnd(entry);
      IRMapping mapping;
      unsigned index = 0;
      for (Value value : body.getArguments()) mapping.map(value, entry->getArgument(index++));
      for (Value value : captures) mapping.map(value, entry->getArgument(index++));
      auto contribution = mdir::kernels::emitTemplateWindowContributions(
          builder, op.getLoc(), op, 0, mapping);
      SmallVector<Value> values(contribution.firstForces.begin(), contribution.firstForces.end());
      llvm::append_range(values, contribution.secondForces);
      llvm::append_range(values, contribution.sums);
      func::ReturnOp::create(builder, op.getLoc(), values);
    }
    for (Operation *source : sources) source->erase();
    for (Operation &op : llvm::make_early_inc_range(module))
      if (isa<mdir::md::MDDialect>(op.getDialect())) op.erase();
  }
};
} // namespace
namespace mdir::test {
void registerTestExecTemplateWindows() { PassRegistration<TestExecTemplateWindows>(); }
}
