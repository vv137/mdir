// Force specialization of the rectangular TOWERS traversal.
#pragma once

#include "mdir/Conversion/TowersBuild.h"
#include "mdir/Conversion/TowersLaunch.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"

namespace mdir::kernels {
enum class TowersOutput { Force, Energy, Virial, DiagonalVirial, TensorVirial };
struct TowersForceShape {
  int panel, stripe, workers, records, residency;
  bool packedChargeType, certified;
  int exclusions, split;
  TowersOutput output = TowersOutput::Force;
  unsigned yieldIndex = 0;
  bool indexedExclusions = false;
  int virialYield = -1;
  bool movingDispatch = false;
  mlir::Type cartesianPositions = {};
  int64_t typeCount = 0;

  // Compact only lists that can fill 128 four-row blocks. Smaller lists
  // need the parallelism of one thread per pair (D431).
  int exclusionRowsPerThread() const {
    return towersExclusionRows(exclusions, workers);
  }
};

mlir::LogicalResult emitTowersForce(mlir::ModuleOp module, TowersLayout layout,
                                    md_exec::PairForOp term,
                                    TowersForceShape shape, bool probe,
                                    md_exec::PairForOp correction);
} // namespace mdir::kernels
