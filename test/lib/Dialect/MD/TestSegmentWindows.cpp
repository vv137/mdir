// Executes the actual segment window bodies and their per-site cutoffs
// independently of lists (D[kind-pairs]).
//
// Each md.sum_segment_window and md.gather_segment_window of an md.function
// becomes a func.func @window_<f>_<g>_<energy|forces|virial> whose
// arguments are the body's arguments, then the other values it captures.
// A table that the body reads (md.lookup) is not passed: each lookup
// becomes a call of @window_table<n>(indices...) -> value, n the number of
// the table's argument in its function, which the caller defines. The md
// ops are removed.
#include "mdir/Dialect/MD/MDDialect.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MD/Transforms/SegmentWindows.h"
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

static func::FuncOp declare(ModuleOp module, Location loc, StringRef name,
                            FunctionType type, bool &failed) {
  auto existing = SymbolTable::lookupSymbolIn(module, name);
  auto function = dyn_cast_or_null<func::FuncOp>(existing);
  if (existing && (!function || !function.isExternal() ||
                   function.getFunctionType() != type)) {
    emitError(loc) << "test window declaration @" << name
                   << " has the wrong type or a duplicate definition";
    failed = true;
    return {};
  }
  if (!function) {
    function = func::FuncOp::create(loc, name, type);
    function.setPrivate();
    module.push_back(function);
  }
  return function;
}

struct TestSegmentWindows
    : PassWrapper<TestSegmentWindows, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestSegmentWindows)
  StringRef getArgument() const final { return "test-md-segment-windows"; }
  StringRef getDescription() const final {
    return "Outlines proved segment windows with their actual site guards";
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
        if (isa<SumSegmentWindowOp, GatherSegmentWindowOp>(op))
          windows.push_back(op);
      });
    });
    bool failed = false;
    for (Operation *op : windows) {
      Block &body = op->getRegion(0).front();
      // Table lookups become calls that the caller defines.
      for (Operation &inner : llvm::make_early_inc_range(body)) {
        auto lookup = dyn_cast<LookupOp>(inner);
        auto table = lookup ? dyn_cast<BlockArgument>(lookup.getTable()) : BlockArgument();
        if (!lookup || !table || table.getParentBlock() == &body) continue;
        SmallVector<Type> indices(lookup.getIndices().getTypes());
        auto type = FunctionType::get(module.getContext(), indices,
                                      lookup.getResult().getType());
        auto callee = declare(module, op->getLoc(),
                              ("window_table" + llvm::Twine(table.getArgNumber())).str(),
                              type, failed);
        if (failed) return signalPassFailure();
        OpBuilder at(lookup);
        auto call = func::CallOp::create(at, lookup.getLoc(), callee, lookup.getIndices());
        lookup.getResult().replaceAllUsesWith(call.getResult(0));
        lookup.erase();
      }
      bool gather = isa<GatherSegmentWindowOp>(op);
      bool tensor = isa<VectorType>(op->getResult(0).getType());
      int64_t f = op->getAttrOfType<IntegerAttr>("f").getInt();
      int64_t g = op->getAttrOfType<IntegerAttr>("g").getInt();
      std::string name = ("window_" + llvm::Twine(f) + "_" + llvm::Twine(g) + "_" +
                          (gather ? "forces" : tensor ? "virial" : "energy")).str();
      SmallVector<Type> inputs(body.getArgumentTypes());
      llvm::SetVector<Value> outer;
      for (Operation &inner : body)
        for (Value operand : inner.getOperands())
          if (operand.getParentBlock() != &body) outer.insert(operand);
      for (Value value : outer) inputs.push_back(value.getType());
      IRMapping mapping;
      // The result types are those that the emitter returns.
      auto function = func::FuncOp::create(op->getLoc(), name,
          FunctionType::get(module.getContext(), inputs, {}));
      Block *entry = function.addEntryBlock();
      OpBuilder builder = OpBuilder::atBlockEnd(entry);
      unsigned index = 0;
      for (Value argument : body.getArguments())
        mapping.map(argument, entry->getArgument(index++));
      for (Value value : outer) mapping.map(value, entry->getArgument(index++));
      auto values = emitSegmentWindowBody(builder, op, mapping);
      func::ReturnOp::create(builder, op->getLoc(), values);
      SmallVector<Type> outputs;
      for (Value value : values) outputs.push_back(value.getType());
      function.setFunctionType(FunctionType::get(module.getContext(), inputs, outputs));
      if (SymbolTable::lookupSymbolIn(module, name)) {
        auto existing = dyn_cast<func::FuncOp>(SymbolTable::lookupSymbolIn(module, name));
        if (!existing || !existing.isExternal() ||
            existing.getFunctionType() != function.getFunctionType()) {
          op->emitError("test window declaration @")
              << name << " has the wrong type or a duplicate definition";
          function.erase();
          return signalPassFailure();
        }
        existing.erase();
      }
      module.push_back(function);
    }
    for (Operation &op : llvm::make_early_inc_range(module))
      if (isa<MDDialect>(op.getDialect())) op.erase();
  }
};
} // namespace
namespace mdir::test {
void registerTestSegmentWindows() { PassRegistration<TestSegmentWindows>(); }
}
