// Packing and displacement tests in the final particle position writer.
#ifndef MDIR_CONVERSION_TEMPLATEWRITERKERNELS_H
#define MDIR_CONVERSION_TEMPLATEWRITERKERNELS_H
#include "mdir/Conversion/TemplateParticleKernels.h"
#include <array>
namespace mdir::kernels {
struct TemplateWriterBuffers {
  mlir::Value frames, packingCells, outerReference, innerReference;
  mlir::Value waters, cell, inverse;
  /// The f32 squares of the motion limits of each mobility group (water,
  /// atoms), from emitTemplateMotionLimitSquared, for the launch.
  std::array<mlir::Value, 2> outerLimits, innerLimits;
  mlir::Value built, constrained;
  /// Optional fixed solute-unit table (TEMPLATE_ATOM_ANCHORS). With it, a
  /// member of a solute unit publishes its frame alone; its anchor alone
  /// takes the displacement tests.
  mlir::Value atomAnchors;
  /// Added to every fractional reference point, on each axis: 1/2 makes a
  /// build's points lie in [0,1), one lattice frame for all units, while
  /// the packed frames keep their central image. Displacement tests
  /// compare the framed position shifted by the same lattice fraction.
  double pointOffset = 0.0;
  unsigned positionOutput = 0;
  bool hasSite = false;
  double alpha = 0, extent = 0;
  /// The launch's constants in f32, from prepareTemplateWriterConstants:
  /// the cell, its inverse, and H t for the point offset t.
  mlir::Value narrowCell, narrowInverse, narrowOffset;
};
/// Narrows the cell, its inverse, and the cell's image of the point offset
/// once for a launch, outside the kernel.
void prepareTemplateWriterConstants(mlir::OpBuilder &, mlir::Location,
                                    TemplateWriterBuffers &);
struct TemplateWriterFlags {
  // Only the owning mobility group's booleans can be true. The caller
  // reduces these through the position writer's existing flag machinery.
  mlir::Value waterOuter, atomOuter, waterInner, atomInner, extent;
};
/// Consume final coordinates of one unit. No position-buffer reads,
/// launches, synchronization or readback are added. An unbuilt list skips
/// all uninitialized epoch views. Extra points do not own packed frames.
/// Constraint history is function scoped, so a fresh list cannot reopen
/// the initial-input extent exception. The relative geometry, the extent
/// test and the displacement tests are in f32, with bounds that keep them
/// conservative; the anchor's lattice shift is added in the state's
/// precision. The writer stores no reference point: a prune takes the
/// current points (emitTemplateCurrentPoint) in a kernel before it.
TemplateWriterFlags emitUpdateTemplateFrame(mlir::OpBuilder &, mlir::Location,
    const TemplateWriterBuffers &, mlir::Value unit,
    llvm::ArrayRef<TemplateParticleResult>);
/// The current reference points, which a prune (or a test) needs: for each
/// unit, the fraction of its anchor's position framed at the anchor's
/// packing cells, plus the point offset. A water's anchor is its O; a
/// solute unit's member takes its anchor's point, in the anchor's frame.
struct TemplatePointBuffers {
  /// `members` maps real-site rows to particles; without it a row is the
  /// particle. `atomAnchors` is optional, as in TemplateWriterBuffers.
  mlir::Value positions, members, packingCells, currentPoints, atomAnchors;
  mlir::Value waters, cell, inverse;
  double pointOffset = 0.0;
};
/// One unit's current point (an index-typed `unit`).
void emitTemplateCurrentPoint(mlir::OpBuilder &, mlir::Location,
                              const TemplatePointBuffers &, mlir::Value unit);
}
#endif
