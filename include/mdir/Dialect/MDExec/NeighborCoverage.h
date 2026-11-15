// Construction coverage across checked SSA boundaries (D357).

#ifndef MDIR_DIALECT_MDEXEC_NEIGHBORCOVERAGE_H
#define MDIR_DIALECT_MDEXEC_NEIGHBORCOVERAGE_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

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

} // namespace mdir::md_exec

#endif // MDIR_DIALECT_MDEXEC_NEIGHBORCOVERAGE_H
