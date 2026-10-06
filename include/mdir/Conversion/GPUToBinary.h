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

/// The architecture (`sm_XY`) of the CUDA device that runs the kernels:
/// the visible device `device`, or the one that MDRT_DEVICE names, as the
/// runtime chooses it. Empty when the driver or the device cannot be
/// queried, or when LLVM's NVPTX backend does not know the architecture;
/// the kernels are then PTX for the default architecture, which the driver
/// compiles at load. The driver is initialized by the query. Asked once per
/// process and device.
std::string getGpuChip(int64_t device);

} // namespace mdir

#endif // MDIR_CONVERSION_GPUTOBINARY_H
