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
  /// Optional fixed solute-unit tables (TEMPLATE_ATOM_ANCHORS/DEPENDENTS).
  /// With them, a member of a solute unit publishes its frame alone; its
  /// anchor publishes the reference point of the whole unit, in each
  /// member's packing frame, and alone takes the displacement tests.
  mlir::Value atomAnchors, atomDependents;
  /// Added to every fractional reference point, on each axis: 1/2 makes a
  /// build's points lie in [0,1), one lattice frame for all units, while
  /// the packed frames keep their central image. Displacement tests
  /// compare the framed position shifted by the same lattice fraction.
  double pointOffset = 0.0;
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
