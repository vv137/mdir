// Storage and lowering must agree on the cell-ordered cubic force path.
#pragma once
#include "mdir/Dialect/MDExec/MDExecOps.h"

namespace mdir::md_exec {
inline constexpr int PMECellEdge = 8;
inline bool canUsePMECellOrder(ReciprocalOp op) {
  // D262 lists the particles whose coefficient can be nonzero. A gather
  // over all atoms can include its zero coefficients in the packed frame.
  return (!op.getListed() || op.getGatherAll()) && op.getOrder() == 4 &&
         !op.getPotential() && !op.getProlate() && !op.getDispersion() &&
         !op.getP3m() &&
         llvm::all_of(op.getGrid(), [](int64_t k) { return k >= 7; });
}
} // namespace mdir::md_exec
