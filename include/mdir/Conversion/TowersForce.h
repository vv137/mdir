// Force specialization of the rectangular TOWERS traversal.
#pragma once

#include "mdir/Conversion/TowersBuild.h"
#include "mdir/Conversion/TowersLaunch.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mdir/Support/ForceWords.h"

namespace mdir::kernels {
enum class TowersOutput { Force, Energy, Virial, DiagonalVirial, TensorVirial };
// The forward index serves the physical-pair traversal on wide replay.
struct TowersExclusionIndex {
  enum Column { ForwardMask, Maximum, Columns };
};

// Coordinates use Q32 in nm and virial forces use Q28. The three-limb
// accumulator keeps those additional bits until rounding the total to Q32.
inline constexpr int towersVirialGuardBits = 28;
inline constexpr int towersVirialLimbs = 3;

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
  bool emitRetryCounter = false;
  /// Leave certified per-atom force words for the following bonded kernel.
  bool keepForceWords = false;
  bool scalarScreen = false;

  // Compact only lists that can fill 128 four-row blocks. Smaller lists
  // need the parallelism of one thread per pair (D431).
  int exclusionRowsPerThread() const {
    return towersExclusionRows(exclusions, workers);
  }
};

mlir::Value decodeTowersSlotVirial(mlir::OpBuilder &b, mlir::Location loc,
                                  mlir::Value limbs, int component);

mlir::LogicalResult emitTowersForce(mlir::ModuleOp module, TowersLayout layout,
                                    md_exec::PairForOp term,
                                    TowersForceShape shape, bool probe,
                                    md_exec::PairForOp correction);
} // namespace mdir::kernels
