#include "mdir/Dialect/MD/Transforms/Passes.h"
#include "mdir/Dialect/MD/Transforms/DerivativeInterface.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MDDist/MDDistDialect.h"
using namespace mlir;
using namespace mdir::md;
namespace md_dist = mdir::md_dist;
namespace mdir { namespace md {
#define GEN_PASS_DEF_CHECKDERIVATIVECOVERAGE
#include "mdir/Dialect/MD/Transforms/Passes.h.inc"
namespace {
class CheckDerivativeCoverage : public impl::CheckDerivativeCoverageBase<CheckDerivativeCoverage> {
  void runOnOperation() final {
    bool invalid = false;
    getOperation()->walk([&](PotentialOp potential) {
      potential.walk([&](Operation *op) {
        if (op == potential || isa<ReturnOp, YieldOp>(op) ||
            isa<NeighborhoodOp, SumRelationOp, GatherRelationOp, SumTuplesOp,
                GatherTuplesOp, SumParticlesOp, MapParticlesOp, ReciprocalOp,
                OrthorhombicCellOp, TriclinicCellOp, TripletsOp>(op) ||
            op->getName().getStringRef() == "md_exec.cell_edges")
          return;
        // The ops of md_dist are semantic: md-differentiate takes the
        // transfers by their declared transposes and the structure as
        // inactive, and refuses what it cannot (D276).
        if (isa_and_nonnull<md_dist::MDDistDialect>(op->getDialect()))
          return;
        if (!isa<DerivativeOpInterface>(op)) {
          op->emitError() << "no derivative rule or declared zero for '"
                          << op->getName() << "' in potential '"
                          << potential.getSymName() << "'";
          invalid = true;
        }
      });
    });
    if (invalid) signalPassFailure();
  }
};

} } }
