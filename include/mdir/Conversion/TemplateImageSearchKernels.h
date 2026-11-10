// Bounded-image candidate kernels; selection occurs outside device kernels.
#ifndef MDIR_CONVERSION_TEMPLATEIMAGESEARCHKERNELS_H
#define MDIR_CONVERSION_TEMPLATEIMAGESEARCHKERNELS_H
#include "mdir/Conversion/TemplateSearchKernels.h"
namespace mdir::kernels {
void emitSpecializedTemplateSearch(mlir::OpBuilder &, mlir::Location,
    const TemplateSearchBuffers &, bool conditional);
}
#endif
