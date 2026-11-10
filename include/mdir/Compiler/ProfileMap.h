// The profile map of a lowered program (D398, #840): for each
// parallel region of the host, the function that the translation to LLVM IR
// outlines it into (`mdir_run..omp_par.N`, `mdrt.pme_spread_..omp_par`) and
// the record that its lowering gave it
// (include/mdir/Dialect/MDExec/ProfileMap.h).
#ifndef MDIR_COMPILER_PROFILEMAP_H
#define MDIR_COMPILER_PROFILEMAP_H

#include "mdir/Driver/Control.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Error.h"

#include <string>

namespace mdir {
namespace compiler {

/// The profile map of `module`, lowered for the host by the pipeline of
/// `control`, as JSON (docs/profile-map.md). The module is not changed: a
/// copy of it is translated with a mark in each region, which tells the
/// function the region is outlined into; the names are those of the
/// translation of the module itself, whose order of outlining the marks do
/// not change.
llvm::Expected<std::string> getProfileMap(mlir::ModuleOp module,
                                          const driver::Control &control);

} // namespace compiler
} // namespace mdir

#endif // MDIR_COMPILER_PROFILEMAP_H
