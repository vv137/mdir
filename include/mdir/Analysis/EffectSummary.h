// Access facts shared by execution and distributed scheduling.

#ifndef MDIR_ANALYSIS_EFFECTSUMMARY_H
#define MDIR_ANALYSIS_EFFECTSUMMARY_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"

namespace mdir::analysis {

/// The rows touched by one access. Unknown is never evidence of separation.
enum class Rows { Own, Neighbor, TupleMembers, Indexed, Whole, Unknown };
enum class Effect { Read, Write, Accumulate };

struct Access {
  /// The exact operand through which the operation observes or changes data.
  mlir::OpOperand *operand;
  Effect effect;
  Rows rows;
  mlir::Attribute set;
};

/// A derived record, not an assertion that two operations are independent.
/// An unknown body effect or range makes `complete` false. Readiness and
/// completion belong to separate contracts.
struct EffectSummary {
  llvm::SmallVector<Access> accesses;
  bool complete = true;

  void add(mlir::OpOperand &operand, Effect effect, Rows rows,
           mlir::Attribute set = {}) {
    accesses.push_back({&operand, effect, rows, set});
  }

  /// True only when every observed field access has an owned-row footprint.
  bool usesOwnedRowsOnly() const {
    if (!complete)
      return false;
    for (const Access &access : accesses)
      if (access.rows != Rows::Own)
        return false;
    return true;
  }
};

} // namespace mdir::analysis

#endif // MDIR_ANALYSIS_EFFECTSUMMARY_H
