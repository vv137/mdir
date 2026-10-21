// Proof of exchange contracts.
//
// See docs/ops.md, Section 4.7. The proof is in lib/Dialect/MD/MDExchange.cpp.

#include "mdir/Dialect/MD/Transforms/Passes.h"
#include "mdir/Dialect/MD/MDExchange.h"

#include "mdir/Dialect/MD/MDOps.h"

using namespace mlir;
using namespace mdir::md;

/// Proves the exchange contract of `op`, or emits a diagnostic.
template <typename OpTy>
static LogicalResult checkContract(OpTy op) {
  if (op.getExchangeBasis() != ExchangeBasis::Proof ||
      op.getExchange() == Exchange::None)
    return success();

  Block &kernel = op.getKernel().front();
  Value value = cast<YieldOp>(kernel.getTerminator()).getOperand(0);

  int expected = op.getExchange() == Exchange::Symmetric ? 1 : -1;
  if (proveExchange(kernel, value) == expected)
    return success();

  InFlightDiagnostic diagnostic = op.emitOpError()
                                  << "cannot prove that the kernel is "
                                  << stringifyExchange(op.getExchange())
                                  << " under exchange of the two particles";
  diagnostic.attachNote()
      << "if the contract holds, state it as 'exchange("
      << stringifyExchange(op.getExchange()) << ", asserted)'";
  return diagnostic;
}

namespace mdir {
namespace md {

#define GEN_PASS_DEF_CHECKEXCHANGE
#include "mdir/Dialect/MD/Transforms/Passes.h.inc"

namespace {
class CheckExchange : public impl::CheckExchangeBase<CheckExchange> {
public:
  using impl::CheckExchangeBase<CheckExchange>::CheckExchangeBase;

  void runOnOperation() final {
    bool failed = false;
    getOperation()->walk([&](Operation *op) {
      if (auto sum = dyn_cast<SumRelationOp>(op))
        failed |= mlir::failed(checkContract(sum));
      else if (auto gather = dyn_cast<GatherRelationOp>(op))
        failed |= mlir::failed(checkContract(gather));
    });
    if (failed)
      signalPassFailure();
  }
};
} // namespace

} // namespace md
} // namespace mdir
