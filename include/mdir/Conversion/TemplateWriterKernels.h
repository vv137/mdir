// Packing and displacement tests in the final particle position writer.
#ifndef MDIR_CONVERSION_TEMPLATEWRITERKERNELS_H
#define MDIR_CONVERSION_TEMPLATEWRITERKERNELS_H
#include "mdir/Conversion/TemplateParticleKernels.h"
#include <array>
namespace mdir::kernels {
struct TemplateWriterBuffers {
  mlir::Value frames, packingCells, currentPoints, outerReference, innerReference;
  mlir::Value waters, cell, inverse;
  std::array<mlir::Value, 2> outerLimits, innerLimits;
  mlir::Value built, constrained;
  unsigned positionOutput = 0;
  bool hasSite = false;
  double alpha = 0, extent = 0;
};
struct TemplateWriterFlags {
  // Only the owning mobility group's booleans can be true. The caller
  // reduces these through the position writer's existing flag machinery.
  mlir::Value waterOuter, atomOuter, waterInner, atomInner, extent;
};
/// Consume final SSA coordinates from emitTemplateParticleKernel. No
/// position-buffer reads, launches, synchronization or readback are added.
/// An unbuilt list skips all uninitialized epoch views. Extra points do
/// not own packed frames. Constraint history is function scoped, so a
/// fresh list cannot reopen the initial-input extent exception.
TemplateWriterFlags emitUpdateTemplateFrame(mlir::OpBuilder &, mlir::Location,
    const TemplateWriterBuffers &, mlir::Value unit,
    llvm::ArrayRef<TemplateParticleResult>);
}
#endif
