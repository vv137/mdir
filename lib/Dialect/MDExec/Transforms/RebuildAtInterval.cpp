// Neighbor structures rebuilt at a fixed interval, whether they are valid
// or not.
//
// NOT A DEFAULT: the structure may leave out pairs within the cutoff
// between builds. A run takes the pass only when its control file asks,
// and the driver warns. See the description of the pass in Passes.td, and
// D88 in docs/decisions.md.

#include "mdir/Dialect/MDExec/Transforms/Passes.h"

#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::md_exec;

namespace mdir {
namespace md_exec {

#define GEN_PASS_DEF_REBUILDATINTERVAL
#include "mdir/Dialect/MDExec/Transforms/Passes.h.inc"

namespace {
class RebuildAtInterval
    : public impl::RebuildAtIntervalBase<RebuildAtInterval> {
public:
  using impl::RebuildAtIntervalBase<RebuildAtInterval>::RebuildAtIntervalBase;

  void runOnOperation() final {
    if (interval < 1) {
      getOperation()->emitError()
          << "expected a positive interval of rebuilds, got "
          << static_cast<int64_t>(interval);
      return signalPassFailure();
    }
    Builder builder(&getContext());
    getOperation()->walk([&](RefreshNeighborsOp refresh) {
      // A refresh whose test a loop has made keeps it.
      if (refresh.getPolicy() != RebuildPolicy::Check || refresh.getMoved())
        return;
      refresh.setPolicy(RebuildPolicy::Interval);
      refresh.setIntervalAttr(builder.getI64IntegerAttr(interval));
    });
  }
};
} // namespace

} // namespace md_exec
} // namespace mdir
