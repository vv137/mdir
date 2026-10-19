// The algebra of each reduced value as a fact of the IR
// (D301, docs/device-primitives.md, Section 5).
//
// A reduction or a scatter-accumulate combines values in an order that its
// lowering chooses. Whether another order is acceptable is a property of
// the sum, not of the lowering: an integer sum or an `or` is exact in any
// order; a floating-point sum is not associative, and another order changes
// its rounding, which the default mode accepts and the deterministic mode
// (D84) does not. `md-exec-assign-algebra` states it on each op that
// reduces, as `md_exec.algebra`, one entry a reduced value; the chooser of
// strategies and the lowerings read it from here, and refuse an op that
// states none (docs/principles.md, principles 1 and 2).

#ifndef MDIR_DIALECT_MDEXEC_ALGEBRA_H
#define MDIR_DIALECT_MDEXEC_ALGEBRA_H

#include "mdir/Dialect/MDExec/MDExecOps.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>

namespace mdir {
namespace md_exec {
namespace algebra {

/// The name of the discardable attribute that states the algebras.
inline constexpr llvm::StringLiteral kAlgebraAttrName = "md_exec.algebra";

/// Whether `op` reduces values, and so states their algebra: the loops
/// over particles, pairs, and tuples, `md_exec.scatter_add`, and
/// `md_exec.reciprocal`; `md_dist.all_reduce` and `md_dist.halo_sum`.
bool reduces(mlir::Operation *op);

/// Whether `op` is a sum across the domains of `md_dist`, whose lowering
/// adds the domains in their order (D275, D276): its algebra is
/// `fixed_order` in every mode, where the fixed order means the same bits
/// on every run for the same decomposition and build, not independence of
/// the decomposition (#409).
bool isDistributed(mlir::Operation *op);

/// States `fixed_order` for every value that `op`, a sum across domains,
/// reduces. Only `md-exec-assign-algebra` states an algebra; every other
/// pass carries it unchanged.
void stateDistributed(mlir::Operation *op);

/// The name of the unit attribute on the module of a program built for the
/// deterministic mode, which `md-exec-assign-algebra{deterministic}` sets.
/// Its verifier refuses `any_order` on a floating-point value anywhere in
/// the module.
inline constexpr llvm::StringLiteral kDeterministicAttrName =
    "md_exec.deterministic";

/// Verifies `md_exec.deterministic` on `module`: no floating-point value is
/// `any_order` there.
mlir::LogicalResult verifyDeterministic(mlir::Operation *module);

/// The element type of each value that `op` reduces, in the order of the
/// attribute:
/// - `particle_for`: its `reduce`;
/// - `pair_for`, `tuple_for`: its destinations (each a sum over the pairs
///   or the tuples of a particle), then its `reduce`;
/// - `scatter_add`: its destination;
/// - `md_dist.all_reduce`, `md_dist.halo_sum`: the sum;
/// - `reciprocal`: the spreading into its grid (in the type of its forces,
///   D271, or of its positions where it yields none), its energy, its
///   virial.
/// Empty for an op that does not reduce.
llvm::SmallVector<mlir::Type> getReducedTypes(mlir::Operation *op);

/// The algebra that `md-exec-assign-algebra` states for a value of
/// `element`: `any_order_exact` for an integer, and for a floating-point number
/// `fixed_order` in the deterministic mode, `any_order` otherwise.
Algebra getAssigned(mlir::Type element, bool deterministic);

/// Whether `op` reduces at least one value, and so must state an algebra.
bool needsAlgebra(mlir::Operation *op);

/// Gives `to`, which replaces `from` with the same reduced values (the
/// storage form of an op), the algebras that `from` states, if any.
void carry(mlir::Operation *from, mlir::Operation *to);

/// States `values` on `op`.
void setAlgebras(mlir::Operation *op, llvm::ArrayRef<Algebra> values);

/// The algebras that `op` states, or nothing.
std::optional<llvm::SmallVector<Algebra>> getAlgebras(mlir::Operation *op);

/// Why a value has no algebra to read, or nothing where it has one.
using Refusal = std::optional<std::string>;

/// The algebra of the reduced value `index` of `op`. `refusal` takes why
/// there is none: the op states no algebra.
std::optional<Algebra> getAlgebra(mlir::Operation *op, unsigned index,
                                  std::string *refusal = nullptr);

/// The algebras of the destinations of a loop over pairs or tuples, or of
/// `scatter_add`: the first entries of the list. Nothing if the op states
/// none.
std::optional<llvm::SmallVector<Algebra>>
getDestinationAlgebras(mlir::Operation *op);

/// Whether a sum of this algebra may be combined in an order that the
/// threads decide: `any_order_exact` and `any_order`, not `fixed_order`.
inline bool allowsReorder(Algebra algebra) {
  return algebra != Algebra::FixedOrder;
}

/// Whether every one of `values` allows reordering.
bool allowReorder(llvm::ArrayRef<Algebra> values);

/// The refusal of a lowering where `op` states no algebra.
std::string getMissingMessage();

/// Verifies `md_exec.algebra` on `op`: an op that reduces, one entry a
/// reduced value, and `any_order_exact` only on integers.
mlir::LogicalResult verify(mlir::Operation *op, mlir::Attribute attr);

} // namespace algebra
} // namespace md_exec
} // namespace mdir

#endif // MDIR_DIALECT_MDEXEC_ALGEBRA_H
