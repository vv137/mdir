// The declared transposes of the transfers of md_dist (D[ml2-distribute]).
//
// See docs/distributed-ml2.md, Sections 3, 4.5, and 11. G, the copy of
// owned values to the ghosts, and G^T, the sum of the ghosts' values at
// their owners, are adjoint for the inner products of owned and of ghost
// rows; md-differentiate carries adjoints back through these rules.
//
//   op                         linear in        transpose
//   halo_copy a (G)            a                halo_sum(p, b̄): ā = p + G^T b̄
//   halo_copy x image (G_x)    x (affine)       the same, G^T
//   halo_sum a, b (a + G^T b)  a, b             ā = r̄, b̄ = G r̄ (halo_copy)
//   distribute f               f                collect
//   collect a                  a                distribute
//   all_reduce p               p                each partial takes r̄ once
//
// The cotangent of a replicated value is given once to every domain's
// partial sum, not divided among them: the replicas are one value.

#include "mdir/Dialect/MD/Transforms/AdjointInterface.h"

#include "mdir/Dialect/MD/MDTypes.h"
#include "mdir/Dialect/MDDist/MDDistDialect.h"
#include "mdir/Dialect/MDDist/MDDistOps.h"
#include "mdir/Dialect/MDDist/MDDistTypes.h"
#include "mlir/IR/DialectRegistry.h"

using namespace mlir;
namespace md_dist = mdir::md_dist;

#include "mdir/Dialect/MD/Transforms/AdjointInterface.cpp.inc"

namespace {

using mdir::md::AdjointOpInterface;

/// The owned rows of the layout of `halo`, zero, of the type of `like`
/// with owned rows.
static Value ownedZeros(OpBuilder &b, Location loc, Value halo,
                        md_dist::FieldType like) {
  Value layout = md_dist::layoutOf(halo);
  return md_dist::ZerosOp::create(b, loc, like.with(md_dist::Role::Owned, false),
                                  layout);
}

struct HaloCopyAdjoint
    : AdjointOpInterface::ExternalModel<HaloCopyAdjoint, md_dist::HaloCopyOp> {
  bool isLinearOperand(Operation *, unsigned index) const { return index == 0; }

  // (G^T b̄)_i = Σ_{g : s(g) = i} b̄_g, added to the owned adjoint `partner`.
  Value emitTranspose(Operation *op, OpBuilder &b, unsigned index,
                      Value adjoint, Value partner) const {
    auto copy = cast<md_dist::HaloCopyOp>(op);
    auto type = cast<md_dist::FieldType>(adjoint.getType());
    if (index != 0 || !type.isGhost()) {
      op->emitOpError() << "expected the adjoint of ghost rows";
      return Value();
    }
    if (type.getPartial() && !partner) {
      op->emitOpError()
          << "the adjoint of the ghost rows is partial, and the owned "
             "contributions of its traversal are not given";
      return Value();
    }
    Value owned = partner ? partner
                          : ownedZeros(b, op->getLoc(), copy.getHalo(), type);
    return md_dist::HaloSumOp::create(
        b, op->getLoc(), type.with(md_dist::Role::Owned, false), owned, adjoint,
        copy.getHalo());
  }
};

struct HaloSumAdjoint
    : AdjointOpInterface::ExternalModel<HaloSumAdjoint, md_dist::HaloSumOp> {
  bool isLinearOperand(Operation *, unsigned index) const { return index < 2; }

  // ā = r̄ for the owned operand; b̄ = G r̄ for the ghost operand.
  Value emitTranspose(Operation *op, OpBuilder &b, unsigned index,
                      Value adjoint, Value partner) const {
    auto sum = cast<md_dist::HaloSumOp>(op);
    if (partner) {
      op->emitOpError() << "has no adjoint to sum with another in one op";
      return Value();
    }
    if (index == 0)
      return adjoint;
    auto type = cast<md_dist::FieldType>(adjoint.getType());
    return md_dist::HaloCopyOp::create(
        b, op->getLoc(), type.with(md_dist::Role::Ghost, false), adjoint,
        sum.getHalo(), Value());
  }
};

struct DistributeAdjoint
    : AdjointOpInterface::ExternalModel<DistributeAdjoint,
                                        md_dist::DistributeOp> {
  bool isLinearOperand(Operation *, unsigned index) const { return index == 0; }

  // The wrapping into the cell is a translation by a lattice vector, whose
  // derivative is the identity: the transpose is the collection.
  Value emitTranspose(Operation *op, OpBuilder &b, unsigned index,
                      Value adjoint, Value partner) const {
    auto distribute = cast<md_dist::DistributeOp>(op);
    if (partner) {
      op->emitOpError() << "has no adjoint to sum with another in one op";
      return Value();
    }
    return md_dist::CollectOp::create(b, op->getLoc(),
                                      distribute.getField().getType(), adjoint,
                                      distribute.getLayout());
  }
};

struct CollectAdjoint
    : AdjointOpInterface::ExternalModel<CollectAdjoint, md_dist::CollectOp> {
  bool isLinearOperand(Operation *, unsigned index) const { return index == 0; }

  Value emitTranspose(Operation *op, OpBuilder &b, unsigned index,
                      Value adjoint, Value partner) const {
    auto collect = cast<md_dist::CollectOp>(op);
    if (partner) {
      op->emitOpError() << "has no adjoint to sum with another in one op";
      return Value();
    }
    return md_dist::DistributeOp::create(
        b, op->getLoc(),
        collect.getField().getType().with(md_dist::Role::Owned, false),
        adjoint, collect.getLayout(), Value());
  }
};

struct AllReduceAdjoint
    : AdjointOpInterface::ExternalModel<AllReduceAdjoint,
                                        md_dist::AllReduceOp> {
  bool isLinearOperand(Operation *, unsigned index) const { return index == 0; }

  // Every domain's partial sum takes the cotangent of the sum, once.
  Value emitTranspose(Operation *op, OpBuilder &, unsigned, Value adjoint,
                      Value partner) const {
    if (partner) {
      op->emitOpError() << "has no adjoint to sum with another in one op";
      return Value();
    }
    return adjoint;
  }
};

} // namespace

void mdir::md::registerAdjointInterfaces(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *context, md_dist::MDDistDialect *) {
        md_dist::HaloCopyOp::attachInterface<HaloCopyAdjoint>(*context);
        md_dist::HaloSumOp::attachInterface<HaloSumAdjoint>(*context);
        md_dist::DistributeOp::attachInterface<DistributeAdjoint>(*context);
        md_dist::CollectOp::attachInterface<CollectAdjoint>(*context);
        md_dist::AllReduceOp::attachInterface<AllReduceAdjoint>(*context);
      });
}
