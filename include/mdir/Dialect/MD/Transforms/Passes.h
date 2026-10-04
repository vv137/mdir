// Passes of the md dialect.

#ifndef MDIR_DIALECT_MD_TRANSFORMS_PASSES_H
#define MDIR_DIALECT_MD_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mdir { namespace mdrt { class MDRTDialect; } }
namespace mlir {
namespace memref { class MemRefDialect; }
namespace scf { class SCFDialect; }
namespace func { class FuncDialect; }
namespace arith {
class ArithDialect;
} // namespace arith
namespace math {
class MathDialect;
} // namespace math
namespace vector {
class VectorDialect;
} // namespace vector
} // namespace mlir

namespace mdir {
namespace md {

#define GEN_PASS_DECL
#include "mdir/Dialect/MD/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "mdir/Dialect/MD/Transforms/Passes.h.inc"

} // namespace md
} // namespace mdir

#endif // MDIR_DIALECT_MD_TRANSFORMS_PASSES_H
