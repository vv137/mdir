// Passes that convert one dialect to another.

#ifndef MDIR_CONVERSION_PASSES_H
#define MDIR_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace arith {
class ArithDialect;
} // namespace arith
namespace func {
class FuncDialect;
} // namespace func
namespace gpu {
class GPUDialect;
} // namespace gpu
namespace LLVM {
class LLVMDialect;
} // namespace LLVM
namespace memref {
class MemRefDialect;
} // namespace memref
namespace scf {
class SCFDialect;
} // namespace scf
namespace math {
class MathDialect;
} // namespace math
namespace vector {
class VectorDialect;
} // namespace vector
} // namespace mlir

namespace mdir {
namespace md_exec {
class MDExecDialect;
} // namespace md_exec
namespace mdrt {
class MDRTDialect;
} // namespace mdrt

#define GEN_PASS_DECL
#include "mdir/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "mdir/Conversion/Passes.h.inc"

/// Registers `mdir-gpu-lower-to-nvvm`: upstream `gpu-lower-to-nvvm-pipeline`,
/// with the same options, whose GPU modules are serialized by
/// `mdir-gpu-module-to-binary` (D214).
void registerGpuLowerToNVVMPipeline();

} // namespace mdir

#endif // MDIR_CONVERSION_PASSES_H
