// Construction coverage across checked SSA boundaries (D357).

#ifndef MDIR_DIALECT_MDEXEC_NEIGHBORCOVERAGE_H
#define MDIR_DIALECT_MDEXEC_NEIGHBORCOVERAGE_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"

namespace mdir::md_exec {
class PairForOp;

/// Proves the requested cutoff, exact centers, and images through direct
/// builds/refreshes, scf.for initial/yielded paths, and every direct call
/// site of a private function. Unknown boundaries are diagnosed. This is
/// construction coverage, not current-state freshness (#571).
mlir::LogicalResult verifyNeighborCoverage(PairForOp loop);

/// The same for another consumer of a structure, `op`, which takes
/// `neighbors` with `cutoff`, no centers, and the minimum image: the
/// traversal of md_alg (D415).
mlir::LogicalResult verifyNeighborCoverage(mlir::Operation *op,
                                           mlir::Value neighbors,
                                           double cutoff);

/// The covering reach of segment s(f, g) of a list of units for the reach
/// `reach` of its reference points: reach + e_f + e_g, added in this order,
/// e_f the covering radius of the larger kind f (D417, Section 1). The
/// build, the pruning, and the coverage query take it from here (#795, U1),
/// so that one order of additions holds throughout.
double getSegmentCoveringReach(double reach, double ef, double eg);

/// The covering radius of kind `kind` (rank K - kind) of the list type
/// `type`: the extent of a taken kind, or `restRadius` for the rest, kind
/// K.
double getSegmentRadius(mlir::Type type, unsigned kind, double restRadius);

/// Proves for the consumer `op`, which takes the list of units `list` with
/// `cutoff` and visits the segments `segments` ((f, g) pairs), that each
/// segment's requirement, cutoff + e_f + e_g, is within the covering reach
/// of the cutoff that the list was built with, through the boundaries of
/// D357. A failure names the segment.
mlir::LogicalResult
verifySegmentCoverage(mlir::Operation *op, mlir::Value list, double cutoff,
                      llvm::ArrayRef<std::pair<unsigned, unsigned>> segments);

} // namespace mdir::md_exec

#endif // MDIR_DIALECT_MDEXEC_NEIGHBORCOVERAGE_H
