// A test pass that prints the shared analysis of the collections of md_alg
// (D409, #866): for each collection, its root, aliases,
// escape, guarantee, uses with what each requires, the joined requirement,
// access, cardinality, lifetime, and determinism, and the verdict on each
// sort. It changes nothing.

#include "mdir/Dialect/MDAlg/CollectionAnalysis.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct TestPrintCollections
    : public PassWrapper<TestPrintCollections, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestPrintCollections)

  StringRef getArgument() const final {
    return "test-md-alg-print-collections";
  }
  StringRef getDescription() const final {
    return "Prints the analysis of the collections of md_alg";
  }
  void runOnOperation() final {
    getAnalysis<mdir::md_alg::CollectionAnalysis>().print(llvm::outs());
    markAllAnalysesPreserved();
  }
};

} // namespace

namespace mdir::test {
void registerTestPrintCollections() {
  PassRegistration<TestPrintCollections>();
}
} // namespace mdir::test
