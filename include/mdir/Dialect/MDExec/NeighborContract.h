// What the builds of md_exec guarantee about a neighbor structure, and what
// its consumers require (D413, #881, step 1 of #877).
//
// The guarantee is a parameter of `!mdrt.neighbors`
// (include/mdir/Dialect/MDRT/NeighborsContract.h). A build or an empty
// structure may state no more than its kind and its images guarantee; a
// consumer states its requirement from its policy and its algebra (D301),
// and its verifier refuses a structure whose guarantee does not meet it.

#ifndef MDIR_DIALECT_MDEXEC_NEIGHBORCONTRACT_H
#define MDIR_DIALECT_MDEXEC_NEIGHBORCONTRACT_H

#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mdir/Dialect/MDRT/NeighborsContract.h"

namespace mdir {
namespace md_exec {

/// What a build of `kind` with `images` guarantees: for a matrix the rows of
/// the centers in the order of the search, with the excluded pairs that a
/// device enters as the particle itself (`masked`); for groups each pair
/// once in the frame of its group (the minimum image), with no bit for an
/// excluded pair (`removed`), the list of each group in the order of its
/// search (I1, I2, I4 of docs/neighbor-groups.md). A dual list keeps the
/// order of its outer list (I5), so its guarantee is that of the groups.
mdrt::NeighborsContract getBuildGuarantee(NeighborKind kind, Images images);

/// The type of a structure on `particleSet` that a build of `kind` with
/// `images` makes, with the guarantee of getBuildGuarantee.
mdrt::NeighborsType getBuildType(mlir::FlatSymbolRefAttr particleSet,
                                 NeighborKind kind, Images images);

/// What the loop `loop` requires of its structure:
/// - its own traversal and images;
/// - the exclusions `masked` for a directed loop, which skips an entry equal
///   to its center, and `removed` for a loop that takes each pair once,
///   which reads the bits of its entries and skips nothing;
/// - the organization `grouped_by(owner)` under `owner_only`, whose center
///   adds its partners in its registers and stores once, and `none` under
///   `atomic`; joined, where a destination or a sum states `fixed_order`
///   (D301), with `stable(owner, search)` for a directed loop and
///   `stable(group, search)` for one that takes each pair once. A loop that
///   states no algebra yet takes nothing from it.
mdrt::NeighborsContract getRequirement(PairForOp loop);

/// What `build_triplets` and `build_edges` require: the rows of a matrix,
/// which hold every neighbor of a particle (D160, D277), in the order of
/// the search, since the order of the tuples they make is the order of
/// later sums.
mdrt::NeighborsContract getRowsRequirement();

/// Refuses on `op` a structure `neighbors` whose type does not meet
/// `required`.
mlir::LogicalResult verifyRequirement(mlir::Operation *op,
                                      mlir::Value neighbors,
                                      const mdrt::NeighborsContract &required);

/// Refuses on `op` a result type that states more than `guaranteed`.
mlir::LogicalResult verifyGuarantee(mlir::Operation *op, mlir::Type result,
                                    const mdrt::NeighborsContract &guaranteed);

} // namespace md_exec
} // namespace mdir

#endif // MDIR_DIALECT_MDEXEC_NEIGHBORCONTRACT_H
