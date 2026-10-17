// The forces of a directed traversal, computed at the owners.
//
// See docs/distributed-ml2.md, Sections 11 and 14 (D276).
// The derivative of a directed md_dist.sum_pairs is a gather_pairs with
// two partial results, half of every pair at each of its particles,
// completed by md_dist.halo_sum (G^T). Every pair of a directed relation is
// at the owner of each of its particles, so the gather with one complete
// result gives the same field and nothing has to return to the owners.
//
// Claim: halo_sum(p, q) = a, with (p, q) the two-ended gather of the
// kernel k over a directed relation D and a the one-ended gather of k over
// D. Proof: let i be owned by d and j any particle with r_ij < r_c. The
// pair is in D_d as (i, k), k the image of j nearest to i, and in D_{o(j)}
// as (j, k'), k' the image of i nearest to j (coverage, Section 4.3). The
// two-ended gather adds ½ k(i, k) at i from the first and ½ σ k(j, k') at
// k', which G^T returns to i, from the second; with the exchange contract
// k(j, k') = σ k(i, k), the two are ½ k(i, k) each, and their sum is the
// term of a at i. Premises, each checked here with the code of the
// verifier: the relation is directed (its type); its halo's reach is at
// least its cutoff (V4, md_dist::verifyCoverage); the kernel has the
// contract (V8, md_dist::proveContract); halo_sum completes the two results
// of the gather through the relation's halo (V7, the verifier of
// halo_sum).

#include "mdir/Dialect/MD/Transforms/Passes.h"

#include "mdir/Dialect/MDDist/MDDistDialect.h"
#include "mdir/Dialect/MDDist/MDDistOps.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;
namespace md_dist = mdir::md_dist;

namespace mdir {
namespace md {

#define GEN_PASS_DEF_DISTOWNERCOMPUTES
#include "mdir/Dialect/MD/Transforms/Passes.h.inc"

namespace {

/// Replaces `sum`, the completion of the two-ended gather over a directed
/// relation, by the one-ended gather. Returns failure after a diagnostic if
/// a premise does not hold; leaves an op that does not match as it is.
LogicalResult rewrite(md_dist::HaloSumOp sum) {
  auto gather = sum.getOwned().getDefiningOp<md_dist::GatherPairsOp>();
  if (!gather || gather.getResults().size() != 2 ||
      sum.getGhosts().getDefiningOp() != gather.getOperation())
    return success();
  if (gather.getRelation().getType().getTraversal() !=
      md_dist::Traversal::Directed)
    return success();
  auto neighborhood =
      gather.getRelation().getDefiningOp<md_dist::NeighborhoodOp>();
  if (!neighborhood)
    return gather.emitOpError()
           << "cannot show that every pair is at both of its owners: the "
              "relation does not come from md_dist.neighborhood";
  if (neighborhood.getHalo() != sum.getHalo())
    return sum.emitOpError()
           << "sums through another halo than that of the relation";
  if (failed(md_dist::verifyCoverage(gather, neighborhood)) ||
      failed(md_dist::proveContract(gather, gather.getKernel().front(),
                                    gather.getExchange())))
    return failure();
  if (gather.getExchange() == md::Exchange::None)
    return gather.emitOpError()
           << "has no exchange contract, which gives the value at the "
              "second particle of a pair";

  OpBuilder b(gather);
  OperationState state(gather.getLoc(), gather->getName());
  state.addOperands(gather->getOperands());
  state.addAttributes(gather->getAttrs());
  state.addTypes(sum.getResult().getType());
  state.addRegion();
  Operation *single = b.create(state);
  IRMapping mapping;
  gather.getKernel().cloneInto(&single->getRegion(0), mapping);
  sum.getResult().replaceAllUsesWith(single->getResult(0));
  sum.erase();
  gather.erase();
  return success();
}

class DistOwnerComputes
    : public impl::DistOwnerComputesBase<DistOwnerComputes> {
public:
  using impl::DistOwnerComputesBase<DistOwnerComputes>::DistOwnerComputesBase;

  void runOnOperation() final {
    SmallVector<md_dist::HaloSumOp> sums;
    getOperation()->walk([&](md_dist::HaloSumOp op) { sums.push_back(op); });
    for (md_dist::HaloSumOp sum : sums)
      if (failed(rewrite(sum)))
        return signalPassFailure();
  }
};

} // namespace
} // namespace md
} // namespace mdir
