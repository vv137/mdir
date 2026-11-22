// Force specialization of the rectangular TOWERS traversal.
#pragma once

#include "mdir/Conversion/TowersBuild.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"

namespace mdir::kernels {
mlir::LogicalResult emitTowersForce(mlir::ModuleOp module, TowersLayout layout,
                                    md_exec::PairForOp term, bool probe);
} // namespace mdir::kernels
