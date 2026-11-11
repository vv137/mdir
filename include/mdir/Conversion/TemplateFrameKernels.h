// Whole-frame snapshots at a stale outer-list rebuild.
#ifndef MDIR_CONVERSION_TEMPLATEFRAMEKERNELS_H
#define MDIR_CONVERSION_TEMPLATEFRAMEKERNELS_H
#include "mlir/IR/Builders.h"
namespace mdir::kernels {
struct TemplateFrameBuffers {
  mlir::Value positions, members, frames;
  mlir::Value outerReference, packingCells, currentPoints, counters;
  mlir::Value waters, atoms, cell, inverse;
  bool hasSite = false;
  double alpha = 0;
  /// Optional TEMPLATE_ATOM_ANCHORS: a member of a solute unit takes its
  /// anchor's packing cell, corrected by whole lattice vectors so that the
  /// member lies in the anchor's image, and the anchor's point in its own
  /// frame. Counter TEMPLATE_COUNTER_ANCHOR_EXTENT receives the largest
  /// squared member-anchor distance (unsigned max of f64 bits).
  mlir::Value atomAnchors;
};
/// Prepare only the requested mobility groups at an outer build. Choose
/// their new retained shifts, recover whole waters before f32 packing,
/// and publish outer/current fractional anchors. Inactive views remain
/// untouched. A cold control reset and unsigned max of nonnegative f64
/// extent bits publish the measured squared extent in counter 3; this is
/// zero if no water was requested. The caller checks finiteness and either
/// widens an initial build or enforces the post-constraint bound. No such
/// measurement/readback belongs to the per-step position writer. The
/// existing reset launch initializes all four controls before a cold
/// owner may copy that packet; no previous search counters are retained.
/// Before packing, bad real-particle positions (D107) increment counter 1
/// and their unit is skipped. The host consumes that count before search
/// reuses counter 1 as flags; it must stop before ordering invalid anchors.
void emitSnapshotTemplateFrames(mlir::OpBuilder &, mlir::Location,
    const TemplateFrameBuffers &, mlir::Value waterActive, mlir::Value atomActive);
}
#endif
