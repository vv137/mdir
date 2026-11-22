// Host PME line transforms, parallel over disjoint planes or column blocks.
#ifndef MDIR_CONVERSION_MDEXECTOLOOPS_PMEFFTS_H
#define MDIR_CONVERSION_MDEXECTOLOOPS_PMEFFTS_H

#include "mlir/IR/BuiltinOps.h"

namespace mdir::kernels {
void emitHostPMEFFTs(mlir::ModuleOp module);
}
#endif
