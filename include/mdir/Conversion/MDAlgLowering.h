// The lowerings of the md_alg dialect (D[md-alg],
// docs/device-primitives.md, Section 4): the serial one, which defines each
// op, and the SIMT one, which calls the emitters of the device primitives
// (CompactionEmitter.h, RaggedRows.h, GatherEmitter.h).

#ifndef MDIR_CONVERSION_MDALGLOWERING_H
#define MDIR_CONVERSION_MDALGLOWERING_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

namespace mdir {
namespace kernels {

/// Replaces every md_alg op under `root` with the code of its emitter,
/// where the op stands, in the order of a walk: the code of a device
/// primitive at a site is that of one emitter. An op without the execution
/// parameters that its SIMT lowering takes is an error, and no op is
/// lowered then. `convert-md-exec-to-gpu` calls this on each template it
/// parses and on the groups build; the pass `convert-md-alg-to-gpu` on a
/// module.
mlir::LogicalResult lowerAlgToSIMT(mlir::Operation *root);

/// Replaces every md_alg op under `root` with its serial definition in
/// `scf` loops, on whatever memory its buffers are in: the oracle of the
/// tests of the SIMT lowering. The execution parameters are ignored. The
/// pass `convert-md-alg-to-scf`.
mlir::LogicalResult lowerAlgToSerial(mlir::Operation *root);

} // namespace kernels
} // namespace mdir

#endif // MDIR_CONVERSION_MDALGLOWERING_H
