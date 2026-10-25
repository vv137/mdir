// Prints the derived loop effect interface for focused tests.

#include "mdir/Dialect/MDExec/LoopEffects.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mdir;

namespace mdir::test {
void registerTestLoopEffects();
}

namespace {
StringRef effectName(analysis::Effect effect) {
  switch (effect) {
  case analysis::Effect::Read:
    return "read";
  case analysis::Effect::Write:
    return "write";
  case analysis::Effect::Accumulate:
    return "accumulate";
  }
  llvm_unreachable("unknown effect");
}

StringRef rowsName(analysis::Rows rows) {
  switch (rows) {
  case analysis::Rows::Own:
    return "own";
  case analysis::Rows::Neighbor:
    return "neighbor";
  case analysis::Rows::TupleMembers:
    return "tuple_members";
  case analysis::Rows::Indexed:
    return "indexed";
  case analysis::Rows::Whole:
    return "whole";
  case analysis::Rows::Unknown:
    return "unknown";
  }
  llvm_unreachable("unknown row scope");
}

struct TestLoopEffects
    : public PassWrapper<TestLoopEffects, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestLoopEffects)

  StringRef getArgument() const final { return "test-md-exec-loop-effects"; }
  StringRef getDescription() const final {
    return "Print derived execution loop access summaries";
  }

  void runOnOperation() override {
    getOperation()->walk([](Operation *op) {
      if (!isa<md_exec::LoopEffectOpInterface>(op))
        return;
      md_exec::LoopEffectSummary summary = md_exec::summarizeLoop(op);
      llvm::outs() << op->getName() << " complete=" << summary.complete
                   << " accesses=" << summary.accesses.size()
                   << " reductions=" << summary.reductions.size() << "\n";
      for (const analysis::Access &access : summary.accesses) {
        llvm::outs() << "  operand=" << access.operand->getOperandNumber()
                     << " effect=" << effectName(access.effect)
                     << " rows=" << rowsName(access.rows)
                     << " set=";
        if (access.set)
          llvm::outs() << access.set;
        else
          llvm::outs() << "unknown";
        llvm::outs() << "\n";
      }
    });
  }
};
} // namespace

void mdir::test::registerTestLoopEffects() {
  PassRegistration<TestLoopEffects>();
}
