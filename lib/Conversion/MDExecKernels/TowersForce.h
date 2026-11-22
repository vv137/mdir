// Force specialization of the rectangular TOWERS traversal.
#pragma once

#include "mdir/Conversion/TowersBuild.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"

namespace mdir::kernels {
struct TowersForceShape {
  int panel, stripe, workers, records, residency;
  bool packedChargeType;
};

mlir::LogicalResult emitTowersForce(mlir::ModuleOp module, TowersLayout layout,
                                    md_exec::PairForOp term,
                                    TowersForceShape shape, bool probe);
} // namespace mdir::kernels
