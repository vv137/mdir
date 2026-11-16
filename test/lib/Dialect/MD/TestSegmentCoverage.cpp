// A test pass that runs the coverage query of lists of units,
// md_exec::verifySegmentCoverage (D357, D417 Section 3), on each loop of
// md_alg.traverse_pairs over a list of units, apart from the lowering that
// takes it (#898). It reports a proof as a remark and a failure as the
// query's diagnostic, and changes nothing.

#include "mdir/Dialect/MDAlg/MDAlgOps.h"
#include "mdir/Dialect/MDExec/NeighborCoverage.h"
#include "mdir/Dialect/MDRT/SegmentRegions.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct TestSegmentCoverage
    : public PassWrapper<TestSegmentCoverage, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestSegmentCoverage)

  StringRef getArgument() const final {
    return "test-md-exec-segment-coverage";
  }
  StringRef getDescription() const final {
    return "Runs the coverage query of lists of units on each loop over "
           "their segments";
  }
  void runOnOperation() final {
    bool failedAny = false;
    getOperation().walk([&](mdir::md_alg::TraversePairsOp op) {
      if (!op.isSegmented())
        return;
      SmallVector<std::pair<unsigned, unsigned>> visited;
      for (unsigned r = 0, e = op.getSegmentKernels().size(); r != e; ++r)
        visited.push_back(mdir::mdrt::getSegmentKinds(op.getSegmentsAttr(), r));
      if (failed(mdir::md_exec::verifySegmentCoverage(
              op, op.getNeighbors(), op.getCutoff().convertToDouble(),
              visited)))
        failedAny = true;
      else
        op.emitRemark() << "segment coverage proved";
    });
    if (failedAny)
      signalPassFailure();
  }
};

} // namespace

namespace mdir::test {
void registerTestSegmentCoverage() { PassRegistration<TestSegmentCoverage>(); }
} // namespace mdir::test
