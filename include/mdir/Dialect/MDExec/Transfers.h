// Checked copy readiness and buffer borrows (D358).

#ifndef MDIR_DIALECT_MDEXEC_TRANSFERS_H
#define MDIR_DIALECT_MDEXEC_TRANSFERS_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

namespace mdir::md_exec {

/// Checks every gpu.memcpy beneath root without changing the IR. Async
/// copies require a recognized blocking completion in the same block.
/// Unknown required effects or alias relations are unsupported contracts.
mlir::LogicalResult verifyTransfers(mlir::Operation *root);

} // namespace mdir::md_exec

#endif // MDIR_DIALECT_MDEXEC_TRANSFERS_H
