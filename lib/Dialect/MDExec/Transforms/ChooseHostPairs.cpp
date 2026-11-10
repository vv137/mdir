// The lanes of the rows of the loops over pairs on the host.
//
// See the description of the pass in Passes.td and D[host-pair-vector].
// The precondition is include/mdir/Dialect/MDExec/HostPairs.h, which the
// lowering checks with the same function; what this pass adds is the
// width and the policy: whether to take the lanes on this host.

#include "mdir/Dialect/MDExec/Transforms/Passes.h"

#include "mdir/Dialect/MDExec/Algebra.h"
#include "mdir/Dialect/MDExec/HostPairs.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::md_exec;

namespace mdir {
namespace md_exec {

#define GEN_PASS_DEF_CHOOSEHOSTPAIRS
#include "mdir/Dialect/MDExec/Transforms/Passes.h.inc"

namespace {
class ChooseHostPairs : public impl::ChooseHostPairsBase<ChooseHostPairs> {
public:
  using impl::ChooseHostPairsBase<ChooseHostPairs>::ChooseHostPairsBase;

  void runOnOperation() final {
    Operation *module = getOperation();
    if (mode != "auto" && mode != "lanes" && mode != "scalar") {
      module->emitError() << "expected 'auto', 'lanes', or 'scalar' for "
                             "'mode', got '"
                          << static_cast<std::string>(mode) << "'";
      return signalPassFailure();
    }
    if (width != 0)
      if (hostpairs::Refusal refused = hostpairs::checkWidth(width)) {
        module->emitError() << "expected a width of 0 or of lanes: "
                            << *refused;
        return signalPassFailure();
      }
    if (vectorBits < 1) {
      module->emitError() << "expected vector-bits of at least 1";
      return signalPassFailure();
    }
    deterministic =
        module->hasAttr(algebra::kDeterministicAttrName);
    module->walk([&](PairForOp op) { choose(op); });
  }

private:
  bool deterministic = false;

  void explainRefused(Operation *op, StringRef why) {
    if (explain)
      op->emitRemark() << "host pairs: scalar: " << why;
  }

  int64_t getWidth(PairForOp op) {
    if (width != 0)
      return width;
    Type computed = hostpairs::getComputedType(op);
    if (!computed || !computed.isIntOrFloat())
      return 1;
    if (deterministic)
      return hostpairs::getDeterministicLanes(computed);
    return hostpairs::getLanes(vectorBits, computed);
  }

  void choose(PairForOp op) {
    if (mode == "scalar")
      return explainRefused(op, "mode=scalar");
    int64_t lanes = getWidth(op);
    if (hostpairs::Refusal refused = hostpairs::checkLanes(op, lanes))
      return explainRefused(op, "no lanes(" + std::to_string(lanes) +
                                    "): " + *refused);
    if (mode == "auto") {
      std::optional<hostpairs::Policy> policy = hostpairs::findPolicy(cpu);
      if (!policy)
        return explainRefused(
            op, "the host '" + static_cast<std::string>(cpu) +
                    "' is not measured, which keeps the scalar row "
                    "(mode=auto)");
      if (!policy->lanes)
        return explainRefused(op, "the scalar row is the policy of '" +
                                      policy->cpu.str() + "' (" +
                                      policy->measured.str() + ")");
    }
    op.setHostLanes(lanes);
    if (explain)
      op->emitRemark() << "host pairs: lanes(" << lanes << ")";
  }
};
} // namespace

} // namespace md_exec
} // namespace mdir
