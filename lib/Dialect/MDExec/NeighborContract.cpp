// The guarantees of the builds and the requirements of the consumers of a
// neighbor structure; see include/mdir/Dialect/MDExec/NeighborContract.h.

#include "mdir/Dialect/MDExec/NeighborContract.h"

#include "mdir/Dialect/MDExec/Algebra.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::md_exec;

using Contract = mdrt::NeighborsContract;

Contract md_exec::getBuildGuarantee(NeighborKind kind, Images images) {
  if (kind == NeighborKind::Groups)
    return Contract::groups();
  return Contract::matrix(images == Images::Explicit
                              ? Contract::Images::Explicit
                              : Contract::Images::Minimum);
}

mdrt::NeighborsType md_exec::getBuildType(FlatSymbolRefAttr particleSet,
                                          NeighborKind kind, Images images) {
  return mdrt::NeighborsType::get(particleSet.getContext(), particleSet,
                                  getBuildGuarantee(kind, images));
}

Contract md_exec::getRequirement(PairForOp loop) {
  bool unique = loop.getTraversal() == Traversal::Unique;
  Contract required;
  required.traversal =
      unique ? Contract::Traversal::Unique : Contract::Traversal::Directed;
  required.images = loop.getImages() == Images::Explicit
                        ? Contract::Images::Explicit
                        : Contract::Images::Minimum;
  required.exclusions =
      unique ? Contract::Exclusions::Removed : Contract::Exclusions::Masked;
  if (loop.getConflict() == Conflict::OwnerOnly)
    required = Contract::withOrganization(
        required, Contract::Organization::GroupedBy, Contract::Group::Owner,
        Contract::Order::None);
  std::optional<SmallVector<Algebra>> algebras =
      algebra::getAlgebras(loop.getOperation());
  if (algebras && !algebra::allowReorder(*algebras)) {
    Contract ordered = Contract::withOrganization(
        required, Contract::Organization::Stable,
        unique ? Contract::Group::Group : Contract::Group::Owner,
        Contract::Order::Search);
    // The two organizations have the same group, so they join.
    if (std::optional<Contract> joined =
            Contract::joinOrganization(required, ordered))
      required = *joined;
    else
      required = ordered;
  }
  return required;
}

Contract md_exec::getRowsRequirement() {
  return Contract::matrix(Contract::Images::Minimum);
}

LogicalResult md_exec::verifyRequirement(Operation *op, Value neighbors,
                                         const Contract &required) {
  auto type = cast<mdrt::NeighborsType>(neighbors.getType());
  if (std::optional<std::string> reason =
          type.getContract().whyNot(required))
    return op->emitOpError()
           << "takes a neighbor structure that does not meet its "
              "requirement: "
           << *reason;
  return success();
}

LogicalResult md_exec::verifyGuarantee(Operation *op, Type result,
                                       const Contract &guaranteed) {
  auto type = cast<mdrt::NeighborsType>(result);
  if (!guaranteed.meets(type.getContract()))
    return op->emitOpError()
           << "has the type of a structure that guarantees '"
           << type.getContract().str() << "', more than its build does: '"
           << guaranteed.str() << "'";
  return success();
}
