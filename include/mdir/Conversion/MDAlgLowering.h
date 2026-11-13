// The lowerings of the md_alg dialect (D406,
// docs/device-primitives.md, Section 4): the serial one, which defines each
// op, and the SIMT one, which calls the emitters of the device primitives
// (CompactionEmitter.h, RaggedRows.h, GatherEmitter.h).

#ifndef MDIR_CONVERSION_MDALGLOWERING_H
#define MDIR_CONVERSION_MDALGLOWERING_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"

namespace mdir {
namespace kernels {

/// The module's record of the policy that each op with a determinism
/// aspect was lowered with (D[md-alg-ir]): an array of dictionaries of
/// strings (`function`, `op`, `determinism`, and its `order`,
/// strategies, and launches), in the order of the lowering. `mdir run`
/// takes it off the module once the passes have run, for its log and its
/// manifest.
inline constexpr llvm::StringLiteral kPolicyAttrName = "mdir.alg_policy";

/// Replaces every md_alg op under `root` with the code of its emitter,
/// where the op stands, in the order of a walk: the code of a device
/// primitive at a site is that of one emitter. An op without the execution
/// parameters that its SIMT lowering takes, or whose determinism
/// requirement its SIMT code does not meet, is an error, and no op is
/// lowered then. The policy of each op with a determinism aspect is added
/// to the record `kPolicyAttrName` of the module that holds `root`.
/// The pass `convert-md-alg-to-gpu` calls this on the module, for the ops
/// of the neighbor matrix's template and the groups build;
/// `convert-md-exec-to-gpu` on the templates of PME, whose accessors it
/// lowers before it unrolls their loops, and on the counting sort of the
/// device's ragged rows, where it takes its launches.
mlir::LogicalResult lowerAlgToSIMT(mlir::Operation *root);

/// Replaces every md_alg op under `root` with its serial definition in
/// `scf` loops, on whatever memory its buffers are in: the oracle of the
/// tests of the SIMT lowering. The execution parameters are ignored. The
/// pass `convert-md-alg-to-scf`.
mlir::LogicalResult lowerAlgToSerial(mlir::Operation *root);

} // namespace kernels
} // namespace mdir

#endif // MDIR_CONVERSION_MDALGLOWERING_H
