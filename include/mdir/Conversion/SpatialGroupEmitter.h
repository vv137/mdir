// Stable column/gap/span grouping composed from shared device scans.
#ifndef MDIR_CONVERSION_SPATIALGROUPEMITTER_H
#define MDIR_CONVERSION_SPATIALGROUPEMITTER_H
#include "mlir/IR/Builders.h"
#include <array>
namespace mdir::kernels {
struct SpatialGroupPlanBuffers {
  /// Unit permutation and per-unit integer column/z keys. The permutation
  /// is lexicographic by (class,column,z,ID).
  mlir::Value order, columns, zKeys, size, firstClassSize;
  /// Nonnegative i32 key thresholds, one for each class.
  std::array<mlir::Value, 2> gap, span;
  /// Independent scratch, M=max(N,1): M flags, M+1 prefix/segment IDs,
  /// M starts, lengths and group counts, M+1 group offsets, eight i32
  /// controls, and max(ceil(N/1024),1) scan words. None aliases another
  /// live field.
  mlir::Value flags, segments, starts, lengths, counts, offsets;
  mlir::Value control, scanScratch, scanBlocks;
  /// Optional host index: the passes to run before the controls are first
  /// read (clamped to [1, N+1]; one when absent). Each later pass is read.
  mlir::Value unreadPasses;
  /// The most units of a group of the first class (the second has 16).
  unsigned firstWidth = 16;
};
struct SpatialGroupPlanResult {
  /// Host index values, available after the control readback. Per-segment
  /// group offsets and IDs remain on the device for stable publication.
  std::array<mlir::Value, 2> groups;
  /// The passes the plan needs (the last that split a group, plus the one
  /// that found none), however many ran before the first readback.
  mlir::Value passes;
};
/// Cut at a column change or a gap greater than the threshold. A group of
/// up to sixteen whose span is too large splits at its rightmost largest
/// gap; refine until no such group remains. Flags only accumulate, giving
/// termination in at most N+1 passes, including an empty input. Passes
/// beyond convergence change nothing, so the first `unreadPasses` run
/// without a readback and only the later ones read the controls. The helper
/// uses shared scans, not another sorting or scan implementation. It does
/// not publish a class's order/maps or decide its ownership.
SpatialGroupPlanResult emitSpatialGroupPlan(mlir::OpBuilder &,
    mlir::Location, const SpatialGroupPlanBuffers &);
}
#endif
