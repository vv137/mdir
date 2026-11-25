// The device neighbor-matrix build, emitted before precision specialization.
#ifndef MDIR_CONVERSION_MATRIXBUILD_H
#define MDIR_CONVERSION_MATRIXBUILD_H

#include "mlir/IR/BuiltinOps.h"

namespace mdir::kernels {
void emitMatrixBuild(mlir::ModuleOp module, bool explicitImages = false);
}

#endif
