// The algebra of each reduced value, stated, and the check that every op
// that reduces states it (D[reduction-algebra]).
//
// See the descriptions of the passes in Passes.td and
// include/mdir/Dialect/MDExec/Algebra.h. The mode decides which algebra a
// floating-point sum takes; from here on the passes and the lowerings read
// the algebra of each sum, not the mode.

#include "mdir/Dialect/MDExec/Transforms/Passes.h"

#include "mdir/Dialect/MDExec/Algebra.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::md_exec;

namespace mdir {
namespace md_exec {

#define GEN_PASS_DEF_ASSIGNALGEBRA
#define GEN_PASS_DEF_CHECKALGEBRA
#include "mdir/Dialect/MDExec/Transforms/Passes.h.inc"

namespace {
class AssignAlgebra : public impl::AssignAlgebraBase<AssignAlgebra> {
public:
  using impl::AssignAlgebraBase<AssignAlgebra>::AssignAlgebraBase;

  void runOnOperation() final {
    // The mode of the program, which the verifier holds the algebras to:
    // no floating-point sum may be `any_order` in it.
    if (deterministic)
      getOperation()->setAttr(algebra::kDeterministicAttrName,
                              UnitAttr::get(&getContext()));
    else
      getOperation()->removeAttr(algebra::kDeterministicAttrName);
    getOperation()->walk([&](Operation *op) {
      if (!algebra::needsAlgebra(op))
        return;
      // A sum across domains has one lowering, in the order of the domains,
      // in every mode: `fixed_order` (for a given decomposition, #409).
      if (algebra::isDistributed(op)) {
        algebra::stateDistributed(op);
        return;
      }
      SmallVector<Algebra> values;
      for (Type type : algebra::getReducedTypes(op))
        values.push_back(algebra::getAssigned(type, deterministic));
      algebra::setAlgebras(op, values);
    });
  }
};

class CheckAlgebra : public impl::CheckAlgebraBase<CheckAlgebra> {
  void runOnOperation() final {
    bool invalid = false;
    getOperation()->walk([&](Operation *op) {
      if (!algebra::needsAlgebra(op) || algebra::getAlgebras(op))
        return;
      op->emitOpError() << algebra::getMissingMessage();
      invalid = true;
    });
    if (invalid)
      signalPassFailure();
  }
};
} // namespace

} // namespace md_exec
} // namespace mdir
