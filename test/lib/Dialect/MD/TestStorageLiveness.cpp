// A test pass that prints the liveness of the storage of md_exec within the
// loop over steps (D[md-exec-storage-liveness], #880): for each host
// function with a loop over steps, each buffer allocated before the loop,
// the points of the step where it is live, its first and last access there,
// whether the loop carries it, whether it is read after the loop or pinned,
// the buffers it interferes with in the step and in the function, and,
// with `at`, the buffers live at the first call to that function. It
// changes nothing.

#include "mdir/Dialect/MDExec/StorageLiveness.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct TestPrintStorageLiveness
    : public PassWrapper<TestPrintStorageLiveness, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestPrintStorageLiveness)

  TestPrintStorageLiveness() = default;
  TestPrintStorageLiveness(const TestPrintStorageLiveness &other)
      : PassWrapper(other) {}

  StringRef getArgument() const final {
    return "test-md-exec-print-storage-liveness";
  }
  StringRef getDescription() const final {
    return "Prints the liveness of the storage of md_exec in the loop over "
           "steps";
  }
  Option<std::string> kills{
      *this, "kills",
      llvm::cl::desc("What kills a buffer: stores (a launch that writes it "
                     "whole and reads none of it, assumed) or proven (a "
                     "copy or a fill of it)"),
      llvm::cl::init("stores")};
  Option<std::string> at{
      *this, "at",
      llvm::cl::desc("Also print the buffers live at the first call to "
                     "this function"),
      llvm::cl::init("")};

  void runOnOperation() final {
    if (kills != "stores" && kills != "proven") {
      getOperation().emitError("kills must be stores or proven");
      return signalPassFailure();
    }
    auto mode = kills == "proven" ? mdir::md_exec::StorageKills::Proven
                                  : mdir::md_exec::StorageKills::Stores;
    AsmState state(getOperation());
    for (auto function : getOperation().getOps<func::FuncOp>()) {
      if (function.isExternal())
        continue;
      mdir::md_exec::StorageLiveness liveness(function, mode);
      if (liveness.hasStepLoop() && !liveness.getBuffers().empty())
        liveness.print(llvm::outs(), state, at);
    }
    markAllAnalysesPreserved();
  }
};

} // namespace

namespace mdir::test {
void registerTestPrintStorageLiveness() {
  PassRegistration<TestPrintStorageLiveness>();
}
} // namespace mdir::test
