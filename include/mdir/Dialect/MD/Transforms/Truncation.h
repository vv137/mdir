// Expansion of truncation attributes.

#ifndef MDIR_DIALECT_MD_TRANSFORMS_TRUNCATION_H
#define MDIR_DIALECT_MD_TRANSFORMS_TRUNCATION_H

#include "mdir/Dialect/MD/MDOps.h"

namespace mdir {
namespace md {

/// Rewrites the kernel of `op` so that it computes the truncated energy, and
/// removes the truncation from `op`. Does nothing if `op` has no truncation.
mlir::LogicalResult expandTruncation(SumRelationOp op);

} // namespace md
} // namespace mdir

#endif // MDIR_DIALECT_MD_TRANSFORMS_TRUNCATION_H
