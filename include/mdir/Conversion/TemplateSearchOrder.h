// Restore canonical covering-search order after specialized image search.
#ifndef MDIR_CONVERSION_TEMPLATESEARCHORDER_H
#define MDIR_CONVERSION_TEMPLATESEARCHORDER_H
#include "mdir/Conversion/TemplateBuildKernels.h"
#include "mdir/Conversion/TemplateSpatialKernels.h"
namespace mdir::kernels {
/// Called after compaction, before publication. Active rows carry a private
/// image/frame tag in entry word three; retained rows are already canonical.
/// Reuses dead staging and inner payload scratch without changing counts.
void emitOrderTemplateSearch(mlir::OpBuilder &, mlir::Location,
    const TemplateBuildBuffers &, const TemplateSpatialBuffers &);
}
#endif
