// Construction coverage across checked SSA boundaries (D[neighbor-coverage]).

#ifndef MDIR_DIALECT_MDEXEC_NEIGHBORCOVERAGE_H
#define MDIR_DIALECT_MDEXEC_NEIGHBORCOVERAGE_H

#include "mlir/Support/LogicalResult.h"

namespace mdir::md_exec {
class PairForOp;

/// Proves the requested cutoff, exact centers, and images through direct
/// builds/refreshes, scf.for initial/yielded paths, and every direct call
/// site of a private function. Unknown boundaries are diagnosed. This is
/// construction coverage, not current-state freshness (#571).
mlir::LogicalResult verifyNeighborCoverage(PairForOp loop);

} // namespace mdir::md_exec

#endif // MDIR_DIALECT_MDEXEC_NEIGHBORCOVERAGE_H
