// Ops of the md_dist dialect.

#ifndef MDIR_DIALECT_MDDIST_MDDISTOPS_H
#define MDIR_DIALECT_MDDIST_MDDISTOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mdir/Dialect/MD/MDEnums.h"
#include "mdir/Dialect/MD/MDTypes.h"
#include "mdir/Dialect/MDDist/MDDistTypes.h"

#define GET_OP_CLASSES
#include "mdir/Dialect/MDDist/MDDistOps.h.inc"

namespace mdir {
namespace md_dist {

// The provenance of distributed values (docs/distributed-ml2.md, Section
// 4.8). Ownership, halos, and versions are SSA values; these functions
// follow the operands that state them. Each returns a null value where the
// IR does not show it (a block argument, an op of another dialect): a
// verifier then rejects the op with that reason rather than assume.

/// The layout of `value`: itself for a layout; the layout a halo was built
/// for; the layout of the rows of an owned field.
mlir::Value layoutOf(mlir::Value value);

/// The halo of `value`: itself for a halo; the halo whose ghost rows a
/// ghost field holds; the halo of a relation.
mlir::Value haloOf(mlir::Value value);

/// The owned value that the ghost field `ghosts` is a copy of, or null if
/// `ghosts` is not the result of `md_dist.halo_copy`.
mlir::Value copySourceOf(mlir::Value ghosts);

/// Checks that `ghosts` is the copy of `owned` through `halo`: V1 (a ghost
/// value is given), V2 (it is a copy of exactly `owned`, the version that
/// the op reads at its centers), and V3 (through `halo`, with images for
/// positions). `what` names the field in diagnostics.
mlir::LogicalResult verifyGhostCopy(mlir::Operation *op, llvm::StringRef what,
                                    mlir::Value owned, mlir::Value ghosts,
                                    mlir::Value halo, bool positions);

/// Checks V7 for the results of `op` of partial type: each has exactly one
/// use, by `md_dist.halo_sum` for a field and by `md_dist.all_reduce` for a
/// number.
mlir::LogicalResult verifyPartialUses(mlir::Operation *op);

/// Checks V4 for the relation of `neighborhood`: the reach of its halo is
/// at least its cutoff, so that every pair within the cutoff is in the
/// rows of the owner of each of its particles (docs/distributed-ml2.md,
/// Section 4.3). Diagnoses on `op`. The verifier of md_dist.neighborhood
/// and md-dist-owner-computes share it (D276).
mlir::LogicalResult verifyCoverage(mlir::Operation *op,
                                   NeighborhoodOp neighborhood);

/// Proves that the value that the pair kernel `kernel` yields has the
/// exchange contract `exchange` (V8), or diagnoses on `op`. The verifiers
/// of the pair ops and md-dist-owner-computes share it.
mlir::LogicalResult proveContract(mlir::Operation *op, mlir::Block &kernel,
                                  mdir::md::Exchange exchange);

} // namespace md_dist
} // namespace mdir

#endif // MDIR_DIALECT_MDDIST_MDDISTOPS_H
