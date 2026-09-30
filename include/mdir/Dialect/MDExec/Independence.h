// Independence of ops in the storage form of md_exec.
//
// Two ops are independent when running them in either order, or at once,
// leaves memory and every value as running them in the order of the
// program does. The ops that run beside others on a second stream must be
// independent of them (D87). The analysis decides it from the memory
// effects that the ops declare and from the buffers that the values may
// be, and it is conservative: where it cannot tell, the ops are dependent.

#ifndef MDIR_DIALECT_MDEXEC_INDEPENDENCE_H
#define MDIR_DIALECT_MDEXEC_INDEPENDENCE_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

namespace mdir {
namespace md_exec {

/// Which values may be the same memory, or overlap it.
///
/// A value is traced to its roots: the allocations and the arguments of the
/// function that it may be at some point of the run. The results of
/// distinct allocations are distinct memory while both are live, and the
/// allocations are not arguments; two arguments may be the same memory. A
/// view is its source. A neighbor structure is its first form, and its
/// roots include the buffers that it keeps (the excluded pairs), which the
/// ops that read the structure read. An argument of a loop that its
/// iterations carry is any of the values that it may take: its initial
/// value, and those that the iterations yield. What cannot be traced has
/// no roots that are known, and may be any memory.
///
/// Two arguments carried by one loop that are the values of distinct
/// allocations at the start, and that each iteration permutes, are distinct
/// at every iteration: by induction, after k iterations they are the
/// initial values at pi^k(i) and pi^k(j), which differ because the
/// permutation pi is a bijection.
class BufferAliases {
public:
  /// Returns true if `a` and `b` may be the same memory, or overlap.
  bool mayAlias(mlir::Value a, mlir::Value b);

private:
  using Roots = llvm::SetVector<mlir::Value>;

  /// The roots of `value`, or nothing if they are not known.
  const std::optional<Roots> &getRoots(mlir::Value value);
  std::optional<Roots> computeRoots(mlir::Value value);

  /// The permutation of the arguments that the loop `loop` carries, which
  /// its iterations apply, if they apply one: the argument `i` takes the
  /// value of the argument `permutation[i]` at the next iteration.
  std::optional<llvm::SmallVector<unsigned>>
  getPermutation(mlir::Operation *loop);

  llvm::DenseMap<mlir::Value, std::optional<Roots>> roots;
  /// Values whose roots are being computed, which a cycle reaches again.
  llvm::DenseSet<mlir::Value> visiting;
};

/// Returns true if `a` and `b` are independent: neither uses a value that
/// the other gives, both declare every memory effect of theirs and of the
/// ops inside them, and no memory that one writes or frees may be memory
/// that the other reads, writes or frees. Ops in the value form, whose
/// fields are not memory yet, are dependent on every op.
bool areIndependent(mlir::Operation *a, mlir::Operation *b,
                    BufferAliases &aliases);

/// The name of the unit attribute that marks an op that runs on a second
/// stream (md_exec.join, D87).
inline constexpr llvm::StringLiteral kSideAttrName = "md_exec.side";

/// Returns the first op after `side` in its block that must not run beside
/// it: one that is not independent of it, a join, another marked op, or the
/// terminator.
mlir::Operation *findJoin(mlir::Operation *side, BufferAliases &aliases);

/// Checks that the op `side`, marked to run on a second stream, is followed
/// in its block by a join, that every op between the two is independent of
/// it, and that none is marked. Reports an error on `side` if not.
mlir::LogicalResult verifySide(mlir::Operation *side, BufferAliases &aliases);

} // namespace md_exec
} // namespace mdir

#endif // MDIR_DIALECT_MDEXEC_INDEPENDENCE_H
