// Exercise the actual execution-body emitter independently of device lists.
#include "mdir/Conversion/TemplateWindowKernels.h"
#include "mdir/Dialect/MD/MDDialect.h"
#include "mdir/Dialect/MD/MDTemplateWindow.h"
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
  TestExecTemplateWindows() = default;
  TestExecTemplateWindows(const TestExecTemplateWindows &pass) : PassWrapper(pass) {}
  Option<bool> cacheSites{*this, "cache-sites",
      llvm::cl::desc("Reuse prepared formed-site geometry"), llvm::cl::init(false)};
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
      // One class is active: WW, or, for forces alone, SW or SS, whose
      // function returns the untransposed site forces.
      unsigned active = llvm::count_if(op.getPairs(), [](int32_t n) { return n != 0; });
      unsigned tag = op.getPairs().size() == 3 && op.getPairs()[0] ? 0
          : op.getPairs().size() == 3 && op.getPairs()[1] ? 1 : 2;
      if (op.getOuts().size() + op.getReduce().size() != 1 ||
          op.getPairs().size() != 3 || active != 1 ||
          (tag != 0 && op.getOuts().empty())) {
        op.emitError("test outliner expects one active WW output, or SW or SS forces, per loop");
        return signalPassFailure();
      }
      sources.insert(op->getParentOfType<func::FuncOp>());
      Block &body = op->getRegion(tag).front();
      bool forces = !op.getOuts().empty();
      bool tensor = !forces && isa<VectorType>(op.getReduce()[0].getType());
      StringRef name = tag == 1 ? "window_forces_mixed" : tag == 2 ? "window_forces_atoms"
          : forces ? "window_forces" : tensor ? "window_virial" : "window_energy";
      SmallVector<Type> inputs(body.getArgumentTypes()), outputs;
      llvm::SetVector<Value> captures;
      body.walk([&](Operation *inner) {
        for (Value value : inner->getOperands())
          if (!body.getParent()->isAncestor(value.getParentRegion())) captures.insert(value);
      });
      for (Value value : captures) inputs.push_back(value.getType());
      if (forces)
        outputs.assign(tag == 0 ? 6 : tag == 1 ? 4 : 2,
                       mdir::md_exec::getKernelValueType(op.getOuts()[0].getType()));
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
      if (cacheSites && tag == 0) {
        auto vector = cast<VectorType>(entry->getArgument(1).getType());
        Value weight = arith::ConstantOp::create(builder, op.getLoc(),
            DenseElementsAttr::get(vector, FloatAttr::get(vector.getElementType(),
                op.getAlpha().convertToDouble())));
        SmallVector<Value> sites;
        for (unsigned first : {1u, 3u})
          sites.push_back(arith::MulFOp::create(builder, op.getLoc(), weight,
              arith::AddFOp::create(builder, op.getLoc(),
                  entry->getArgument(first), entry->getArgument(first + 1))));
        unsigned bound = mdir::kernels::mapTemplateSiteOffsets(op, 0, sites, mapping);
        function->setAttr("test.cached_sites", builder.getI32IntegerAttr(bound));
      }
      auto contribution = mdir::kernels::emitTemplateWindowContributions(
          builder, op.getLoc(), op, tag, mapping);
      if (tag != 0) {
        SmallVector<Value> values(contribution.firstForces.begin(), contribution.firstForces.end());
        llvm::append_range(values, contribution.secondForces);
        if (values.size() != outputs.size()) {
          op.emitError("test outliner expects SW or SS forces without formed sites");
          return signalPassFailure();
        }
        func::ReturnOp::create(builder, op.getLoc(), values);
        continue;
      }
      auto values = mdir::md::emitTemplateForceTranspose(builder, op.getLoc(),
          contribution.firstForces, op.getAlpha().convertToDouble());
      llvm::append_range(values, mdir::md::emitTemplateForceTranspose(builder,
          op.getLoc(), contribution.secondForces, op.getAlpha().convertToDouble()));
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
