// The check that every op of the structure of a potential declares its
// transpose (D295).
//
// md-differentiate carries adjoints back through the ops that move and
// reduce the values of fields: the sums, gathers, and maps of md, and the
// transfers and loops of md_dist. Each must declare its transpose through
// AdjointOpInterface, so that the derivative of a composition is a
// composition of declared transposes (docs/primitives-map.md, Section 1.3).
// The scalar ops inside kernels are the business of
// --md-check-derivative-coverage (D183).

#include "mdir/Dialect/MD/Transforms/AdjointInterface.h"
#include "mdir/Dialect/MD/Transforms/Passes.h"

#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MDDist/MDDistOps.h"

using namespace mlir;
using namespace mdir::md;
namespace md_dist = mdir::md_dist;

namespace mdir {
namespace md {
#define GEN_PASS_DEF_CHECKTRANSPOSES
#include "mdir/Dialect/MD/Transforms/Passes.h.inc"

namespace {

/// Whether `op` moves or reduces the values of fields, so that an adjoint
/// passes through it: an op of the structure.
static bool isOfTheStructure(Operation *op) {
  return isa<SumRelationOp, GatherRelationOp, SumTuplesOp, GatherTuplesOp,
             SumParticlesOp, MapParticlesOp, md_dist::HaloCopyOp,
             md_dist::HaloSumOp, md_dist::DistributeOp, md_dist::CollectOp,
             md_dist::AllReduceOp, md_dist::SumPairsOp, md_dist::GatherPairsOp,
             md_dist::SumParticlesOp, md_dist::MapParticlesOp>(op);
}

class CheckTransposes : public impl::CheckTransposesBase<CheckTransposes> {
  void runOnOperation() final {
    bool invalid = false;
    getOperation()->walk([&](PotentialOp potential) {
      potential.walk([&](Operation *op) {
        if (!isOfTheStructure(op) || isa<AdjointOpInterface>(op))
          return;
        op->emitError() << "no declared transpose for '" << op->getName()
                        << "' in potential '" << potential.getSymName()
                        << "'";
        invalid = true;
      });
    });
    if (invalid)
      signalPassFailure();
  }
};

} // namespace
} // namespace md
} // namespace mdir
