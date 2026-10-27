// A read-only inspection of copy readiness and lifetime.

#include "mdir/Dialect/MDExec/Transforms/Passes.h"
#include "mdir/Dialect/MDExec/Transfers.h"
#include "mlir/IR/BuiltinOps.h"

namespace mdir::md_exec {

#define GEN_PASS_DEF_CHECKTRANSFERS
#include "mdir/Dialect/MDExec/Transforms/Passes.h.inc"

namespace {
class CheckTransfers : public impl::CheckTransfersBase<CheckTransfers> {
public:
  void runOnOperation() final {
    if (failed(verifyTransfers(getOperation())))
      signalPassFailure();
  }
};
} // namespace
} // namespace mdir::md_exec
