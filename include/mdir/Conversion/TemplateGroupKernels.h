// Retained-class publication of the shared spatial group plan.
#ifndef MDIR_CONVERSION_TEMPLATEGROUPKERNELS_H
#define MDIR_CONVERSION_TEMPLATEGROUPKERNELS_H
#include "mdir/Conversion/TemplateSpatialKernels.h"
#include "mlir/IR/BuiltinOps.h"
namespace mdir::kernels {
struct TemplateGroupBuffers {
  mlir::Value starts, lengths, columns, atStart;
  mlir::Value waterGroups, atomGroups, rows;
};
/// Select a common grid without moving a retained class, sort the compact
/// class orders, compose the shared cut plan, and publish only active
/// descriptors. Reacquires spatial views if row capacity grows; no ordinary
/// neighbor structure is selected or changed by this adapter.
TemplateGroupBuffers emitPrepareTemplateGroups(mlir::OpBuilder &,
    mlir::Location, mlir::ModuleOp, mlir::Value handle,
    TemplateSpatialBuffers &, mlir::Value cell);
}
#endif
