// The declared transposes of the transfers of md_dist (D276) and of the
// structure of the ops of md with a kernel (D[md-transposes]).
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
//
// An op of md with a kernel gathers its operands, evaluates the kernel,
// and reduces (docs/primitives-map.md, Section 1.2). Its declared
// transpose is that of the structure; md-differentiate puts the reverse
// derivative of the kernel (D183) in the kernel of the op created here:
//
//   op               operands to the kernel   op of the transpose
//   sum_relation     positions, gathered      gather_relation, the kernel
//                                             copied, the weight captured
//   gather_relation  positions, gathered      gather_relation (or its sum
//                                             over the pairs), the kernel
//                                             from both ends, the adjoint
//                                             read at both
//   sum_tuples       positions, gathered      gather_tuples, the kernel
//                                             copied, the weight captured
//   sum_particles    every operand            map_particles, the weight
//                                             captured
//   map_particles    every operand            map_particles, the adjoint
//                                             read

#include "mdir/Dialect/MD/Transforms/AdjointInterface.h"

#include "mdir/Dialect/MD/MDDialect.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MD/MDTypes.h"
#include "mdir/Dialect/MDDist/MDDistDialect.h"
#include "mdir/Dialect/MDDist/MDDistOps.h"
#include "mdir/Dialect/MDDist/MDDistTypes.h"
#include "mdir/Dialect/MDExec/MDExecDialect.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/DialectRegistry.h"

using namespace mlir;
namespace md = mdir::md;
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

//===----------------------------------------------------------------------===//
// The structure of the ops of md with a kernel
//===----------------------------------------------------------------------===//

using md::TransposeKernel;

/// An op named `opName` over the relation, the positions, and the cell of
/// `source`, an op over a relation, then `gathered` and `extra`, with the
/// exchange contract `exchange`, derived, and one region.
static Operation *createPairOp(OpBuilder &b, Location loc, StringRef opName,
                               Operation *source, ValueRange gathered,
                               ValueRange extra, md::Exchange exchange,
                               TypeRange resultTypes) {
  MLIRContext *context = b.getContext();
  OperationState state(loc, opName);
  state.addOperands({source->getOperand(0), source->getOperand(1),
                     source->getOperand(2)});
  state.addOperands(gathered);
  state.addOperands(extra);
  state.addAttribute("exchange", md::ExchangeAttr::get(context, exchange));
  state.addAttribute(
      "exchange_basis",
      md::ExchangeBasisAttr::get(context, md::ExchangeBasis::Derived));
  state.addRegion();
  state.addTypes(resultTypes);
  return b.create(state);
}

/// Whether `index` is the positions or a gathered field of an op over a
/// relation or over tuples: operand 1, or 3 to 3 + `gathered` − 1.
static bool isPairOrTupleKernelOperand(unsigned index, unsigned gathered) {
  return index == 1 || (index >= 3 && index < 3 + gathered);
}

/// Checks that `index` reaches the kernel of `op` and that the summed form
/// is asked only where `allowSummed`.
static LogicalResult checkTranspose(Operation *op, unsigned index,
                                    bool summed, bool allowSummed) {
  if (!cast<md::AdjointOpInterface>(op).isKernelOperand(index))
    return op->emitOpError()
           << "has no transpose in operand " << index
           << ", which does not reach its kernel";
  if (summed && !allowSummed)
    return op->emitOpError() << "has no summed form of its transpose";
  return success();
}

struct SumRelationTranspose
    : md::AdjointOpInterface::ExternalModel<SumRelationTranspose,
                                            md::SumRelationOp> {
  bool isKernelOperand(Operation *op, unsigned index) const {
    return isPairOrTupleKernelOperand(
        index, cast<md::SumRelationOp>(op).getGathered().size());
  }

  // S = Σ_(i,j) u(i, j): the weight w of S at each pair, the reverse
  // derivative of u there, and its values at the center of each pair added
  // at the center; the relation is symmetric, so that the gather at the
  // center over the directed pairs takes the derivative at both ends.
  Operation *createTranspose(Operation *op, OpBuilder &b, Location loc,
                             unsigned index, Value, md::Exchange exchange,
                             TypeRange resultTypes, bool summed,
                             TransposeKernel &kernel) const {
    auto sum = cast<md::SumRelationOp>(op);
    if (failed(checkTranspose(op, index, summed, /*allowSummed=*/false)))
      return nullptr;
    Operation *gather =
        createPairOp(b, loc, md::GatherRelationOp::getOperationName(), op,
                     sum.getGathered(), ValueRange(), exchange, resultTypes);
    IRMapping mapping;
    sum.getKernel().cloneInto(&gather->getRegion(0), mapping);
    kernel.block = &gather->getRegion(0).front();
    kernel.cloned = true;
    return gather;
  }
};

struct GatherRelationTranspose
    : md::AdjointOpInterface::ExternalModel<GatherRelationTranspose,
                                            md::GatherRelationOp> {
  bool isKernelOperand(Operation *op, unsigned index) const {
    return isPairOrTupleKernelOperand(
        index, cast<md::GatherRelationOp>(op).getGathered().size());
  }

  // G_a = Σ_b k(a, b) with a at the center: the adjoint Ḡ at both ends of
  // each pair, the reverse derivative of k from each end, and the values
  // added at the particle. The kernel is evaluated with the center at the
  // center (`forward`) and with the ends exchanged (`backward`), where the
  // displacement is d_ji = −d_ij.
  Operation *createTranspose(Operation *op, OpBuilder &b, Location loc,
                             unsigned index, Value cotangent,
                             md::Exchange exchange, TypeRange resultTypes,
                             bool summed, TransposeKernel &kernel) const {
    auto gather = cast<md::GatherRelationOp>(op);
    if (failed(checkTranspose(op, index, summed, /*allowSummed=*/true)))
      return nullptr;
    if (!cotangent || !isa<md::FieldType>(cotangent.getType())) {
      op->emitOpError() << "expected the adjoint of its result, a field";
      return nullptr;
    }
    Operation *result = createPairOp(
        b, loc,
        summed ? md::SumRelationOp::getOperationName()
               : md::GatherRelationOp::getOperationName(),
        op, gather.getGathered(), cotangent, exchange, resultTypes);
    result->getRegion(0).push_back(new Block());
    Block &block = result->getRegion(0).front();
    Block &source = gather.getKernel().front();
    Value r = block.addArgument(source.getArgument(0).getType(), loc);
    Value d = block.addArgument(source.getArgument(1).getType(), loc);
    unsigned count = gather.getGathered().size();
    SmallVector<Value> ends;
    for (unsigned k = 0; k != 2 * count; ++k)
      ends.push_back(
          block.addArgument(source.getArgument(2 + k).getType(), loc));
    Type valueType =
        cast<md::FieldType>(cotangent.getType()).getKernelValueType();
    kernel.cotangentI = block.addArgument(valueType, loc);
    kernel.cotangentJ = block.addArgument(valueType, loc);
    OpBuilder inside = OpBuilder::atBlockEnd(&block);
    Value negated = arith::NegFOp::create(inside, loc, d);
    kernel.forward.map(source.getArgument(0), r);
    kernel.backward.map(source.getArgument(0), r);
    kernel.forward.map(source.getArgument(1), d);
    kernel.backward.map(source.getArgument(1), negated);
    for (unsigned k = 0; k != count; ++k) {
      kernel.forward.map(source.getArgument(2 + 2 * k), ends[2 * k]);
      kernel.forward.map(source.getArgument(3 + 2 * k), ends[2 * k + 1]);
      kernel.backward.map(source.getArgument(2 + 2 * k), ends[2 * k + 1]);
      kernel.backward.map(source.getArgument(3 + 2 * k), ends[2 * k]);
    }
    kernel.block = &block;
    return result;
  }
};

struct SumTuplesTranspose
    : md::AdjointOpInterface::ExternalModel<SumTuplesTranspose,
                                            md::SumTuplesOp> {
  bool isKernelOperand(Operation *op, unsigned index) const {
    return isPairOrTupleKernelOperand(
        index, cast<md::SumTuplesOp>(op).getGathered().size());
  }

  // S = Σ_t u(t): the weight at each tuple, the reverse derivative of u in
  // the values of each member, and those added at the members.
  Operation *createTranspose(Operation *op, OpBuilder &b, Location loc,
                             unsigned index, Value, md::Exchange,
                             TypeRange resultTypes, bool summed,
                             TransposeKernel &kernel) const {
    auto sum = cast<md::SumTuplesOp>(op);
    if (failed(checkTranspose(op, index, summed, /*allowSummed=*/false)))
      return nullptr;
    OperationState state(loc, md::GatherTuplesOp::getOperationName());
    state.addOperands(op->getOperands());
    state.addAttributes(op->getAttrDictionary().getValue());
    state.addRegion();
    state.addTypes(resultTypes);
    Operation *gather = b.create(state);
    IRMapping mapping;
    sum.getKernel().cloneInto(&gather->getRegion(0), mapping);
    kernel.block = &gather->getRegion(0).front();
    kernel.cloned = true;
    return gather;
  }
};

/// A map over the operands of `op`, a sum or a map over particles, then
/// the cotangent if it is a field, with the arguments of the source kernel,
/// then the cotangent's value at the particle.
static Operation *createParticleTranspose(Operation *op, Block &source,
                                          OpBuilder &b, Location loc,
                                          Value cotangent,
                                          TypeRange resultTypes,
                                          TransposeKernel &kernel) {
  bool isField = cotangent && isa<md::FieldType>(cotangent.getType());
  OperationState state(loc, md::MapParticlesOp::getOperationName());
  state.addOperands(op->getOperands());
  if (isField)
    state.addOperands(cotangent);
  state.addRegion();
  state.addTypes(resultTypes);
  Operation *map = b.create(state);
  Block *block = new Block();
  map->getRegion(0).push_back(block);
  for (BlockArgument argument : source.getArguments())
    kernel.forward.map(argument, block->addArgument(argument.getType(), loc));
  kernel.cotangentI =
      isField ? block->addArgument(
                    cast<md::FieldType>(cotangent.getType())
                        .getKernelValueType(),
                    loc)
              : cotangent;
  kernel.block = block;
  return map;
}

struct SumParticlesTranspose
    : md::AdjointOpInterface::ExternalModel<SumParticlesTranspose,
                                            md::SumParticlesOp> {
  bool isKernelOperand(Operation *op, unsigned index) const {
    return index < op->getNumOperands();
  }

  // S = Σ_i k(a_i, ...): the weight at each particle, which the kernel
  // captures, and the reverse derivative of k there.
  Operation *createTranspose(Operation *op, OpBuilder &b, Location loc,
                             unsigned index, Value cotangent, md::Exchange,
                             TypeRange resultTypes, bool summed,
                             TransposeKernel &kernel) const {
    if (failed(checkTranspose(op, index, summed, /*allowSummed=*/false)))
      return nullptr;
    return createParticleTranspose(
        op, cast<md::SumParticlesOp>(op).getKernel().front(), b, loc,
        cotangent, resultTypes, kernel);
  }
};

struct MapParticlesTranspose
    : md::AdjointOpInterface::ExternalModel<MapParticlesTranspose,
                                            md::MapParticlesOp> {
  bool isKernelOperand(Operation *op, unsigned index) const {
    return index < op->getNumOperands();
  }

  // B_i = g(a_i, ...): the adjoint B̄_i at each particle, read beside the
  // operands, and the reverse derivative of g there.
  Operation *createTranspose(Operation *op, OpBuilder &b, Location loc,
                             unsigned index, Value cotangent, md::Exchange,
                             TypeRange resultTypes, bool summed,
                             TransposeKernel &kernel) const {
    if (failed(checkTranspose(op, index, summed, /*allowSummed=*/false)))
      return nullptr;
    return createParticleTranspose(
        op, cast<md::MapParticlesOp>(op).getKernel().front(), b, loc,
        cotangent, resultTypes, kernel);
  }
};

//===----------------------------------------------------------------------===//
// The scatter-accumulate of md_exec (D291)
//===----------------------------------------------------------------------===//

// a' = a + Σ_{k : idx_k = i} v_k is linear in the values and in the
// destination. The transpose in the destination is the identity, ā = ā';
// that in the values is a gather by the indices, v̄_k = ā'_{idx_k}, for
// which md_exec has no op yet: it comes with the first derivative through
// a scatter (the ruling of 2026-10-19 on #459).
struct ScatterAddAdjoint
    : md::AdjointOpInterface::ExternalModel<ScatterAddAdjoint,
                                            mdir::md_exec::ScatterAddOp> {
  bool isLinearOperand(Operation *, unsigned index) const {
    return index == 0 || index == 2;
  }

  Value emitTranspose(Operation *op, OpBuilder &, unsigned index,
                      Value adjoint, Value partner) const {
    auto scatter = cast<mdir::md_exec::ScatterAddOp>(op);
    if (partner) {
      op->emitOpError() << "has no adjoint to sum with another in one op";
      return Value();
    }
    // The value form, which has no overwrite(true): the destination
    // always reaches the result.
    if (scatter.isStorageForm()) {
      op->emitOpError() << "has no transpose in the storage form";
      return Value();
    }
    if (index == 2)
      return adjoint;
    op->emitOpError()
        << "has no transpose in its values yet: it is a gather by the "
           "indices, which md_exec has no op for";
    return Value();
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
  registry.addExtension(+[](MLIRContext *context, md::MDDialect *) {
    md::SumRelationOp::attachInterface<SumRelationTranspose>(*context);
    md::GatherRelationOp::attachInterface<GatherRelationTranspose>(*context);
    md::SumTuplesOp::attachInterface<SumTuplesTranspose>(*context);
    md::SumParticlesOp::attachInterface<SumParticlesTranspose>(*context);
    md::MapParticlesOp::attachInterface<MapParticlesTranspose>(*context);
  });
  registry.addExtension(
      +[](MLIRContext *context, mdir::md_exec::MDExecDialect *) {
        mdir::md_exec::ScatterAddOp::attachInterface<ScatterAddAdjoint>(
            *context);
      });
}
