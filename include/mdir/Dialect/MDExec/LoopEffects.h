// One effect record for md_exec execution loops (D[loop-effect-summary]).

#ifndef MDIR_DIALECT_MDEXEC_LOOPEFFECTS_H
#define MDIR_DIALECT_MDEXEC_LOOPEFFECTS_H

#include "mdir/Analysis/EffectSummary.h"
#include "mlir/IR/OpDefinition.h"
#include <cstdint>
#include <optional>

namespace mdir::md_exec {

enum class Algebra : uint32_t;

struct Reduction {
  mlir::OpOperand *operand;
  std::optional<Algebra> algebra;
  analysis::Rows rows;
};

struct LoopEffectSummary : analysis::EffectSummary {
  llvm::SmallVector<Reduction> reductions;
};

/// The summary returned by each loop interface, including an opaque-body bit.
LoopEffectSummary summarizeLoop(mlir::Operation *op);

} // namespace mdir::md_exec

#include "mdir/Dialect/MDExec/LoopEffects.h.inc"

#endif // MDIR_DIALECT_MDEXEC_LOOPEFFECTS_H
