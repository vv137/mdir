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

} // namespace mdir

#endif // MDIR_CONVERSION_PASSES_H
