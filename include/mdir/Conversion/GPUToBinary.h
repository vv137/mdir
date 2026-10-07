// The serialization of the GPU modules (D214).
#ifndef MDIR_CONVERSION_GPUTOBINARY_H
#define MDIR_CONVERSION_GPUTOBINARY_H

#include "mdir/Compiler/CompileCache.h"
#include "llvm/Support/Error.h"
#include <string>

namespace mlir {
class MLIRContext;
} // namespace mlir

namespace mdir {

/// What `mdir-gpu-module-to-binary` did in `context` since the last call:
/// the `gpu*` fields of CompileStats. Taking them resets them.
compiler::CompileStats takeGpuModuleStats(mlir::MLIRContext &context);

/// The options of `mdir-gpu-lower-to-nvvm` for the kernels of the visible
/// CUDA device `device`, or of the one that MDRT_DEVICE names, as the
/// runtime chooses it (D214):
///
/// - MDIR_GPU_BINARY=auto (the default): cubins for the device's
///   architecture, PTX where ptxas cannot make them; PTX for the default
///   architecture when the architecture is not known.
/// - MDIR_GPU_BINARY=cubin: cubins; an unknown architecture or a missing
///   ptxas is an error.
/// - MDIR_GPU_BINARY=ptx: PTX, which the driver compiles at load.
///
/// The architecture is MDIR_GPU_ARCH (`sm_XY`), else the compute capability
/// that NVML gives for the device, if LLVM's NVPTX back end knows it. The
/// CUDA driver is not initialized, so a process may fork after it
/// compiles. NVML is asked once per process and selection of devices.
llvm::Expected<std::string> getGpuPipelineOptions(int64_t device);

} // namespace mdir

#endif // MDIR_CONVERSION_GPUTOBINARY_H
