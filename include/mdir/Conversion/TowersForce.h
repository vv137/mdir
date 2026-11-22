// Force specialization of the rectangular TOWERS traversal.
#pragma once

#include "mdir/Conversion/TowersBuild.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"

namespace mdir::kernels {
enum class TowersOutput { Force, Energy, Virial };
struct TowersForceShape {
  int panel, stripe, workers, records, residency;
  bool packedChargeType, certified;
  int exclusions, split;
  TowersOutput output = TowersOutput::Force;
  // The caller has established that its direct potential is finite at the
  // cutoff. Unknown pair expressions retain the D159 endpoint guard.
  bool finiteAtCutoff = false;
};

mlir::LogicalResult emitTowersForce(mlir::ModuleOp module, TowersLayout layout,
                                    md_exec::PairForOp term,
                                    TowersForceShape shape, bool probe,
                                    md_exec::PairForOp correction);
} // namespace mdir::kernels
