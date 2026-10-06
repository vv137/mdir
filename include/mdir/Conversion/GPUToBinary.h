// The serialization of the GPU modules (D[gpu-module-compile]).
#ifndef MDIR_CONVERSION_GPUTOBINARY_H
#define MDIR_CONVERSION_GPUTOBINARY_H

#include "mdir/Compiler/CompileCache.h"

namespace mlir {
class MLIRContext;
} // namespace mlir

namespace mdir {

/// What `mdir-gpu-module-to-binary` did in `context` since the last call:
/// the `gpu*` fields of CompileStats. Taking them resets them.
compiler::CompileStats takeGpuModuleStats(mlir::MLIRContext &context);

} // namespace mdir

#endif // MDIR_CONVERSION_GPUTOBINARY_H
