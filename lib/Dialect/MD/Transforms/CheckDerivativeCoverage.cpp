#include "mdir/Dialect/MD/Transforms/Passes.h"
#include "mdir/Dialect/MD/Transforms/DerivativeInterface.h"
#include "mdir/Dialect/MD/MDOps.h"
using namespace mlir;
using namespace mdir::md;
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
