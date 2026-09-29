// Passes of the md_exec dialect.

#ifndef MDIR_DIALECT_MDEXEC_TRANSFORMS_PASSES_H
#define MDIR_DIALECT_MDEXEC_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mdir {
namespace mdrt {
class MDRTDialect;
} // namespace mdrt

namespace md_exec {
class MDExecDialect;

#define GEN_PASS_DECL
#include "mdir/Dialect/MDExec/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "mdir/Dialect/MDExec/Transforms/Passes.h.inc"

} // namespace md_exec
} // namespace mdir

#endif // MDIR_DIALECT_MDEXEC_TRANSFORMS_PASSES_H
