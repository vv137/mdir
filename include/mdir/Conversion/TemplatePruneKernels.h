// D284-style fixed-block pruning inside template evaluation.
#ifndef MDIR_CONVERSION_TEMPLATEPRUNEKERNELS_H
#define MDIR_CONVERSION_TEMPLATEPRUNEKERNELS_H
#include "mlir/IR/Builders.h"
namespace mdir::kernels {
struct TemplatePruneBuffers {
  mlir::Value outerUnits, innerUnits, outerEntries, innerEntries, order;
  mlir::Value currentPoints, innerReference, waters, cell, reaches;
  mlir::Value waterActive, atomActive;
  mlir::Value groupLengths;
  /// Optional: the cell narrowed to f32 and the sum of its magnitudes,
  /// computed outside the kernel (then kernel parameters).
  mlir::Value narrowCell, cellMagnitude;
};
/// All 32 lanes own one outer work record (at most 64 entries). Filter its
/// masks using hybrid reference points and compact into its existing
/// inner block. Unrequested classes stay untouched. Work/row ranges retain
/// their outer storage spans; each inner work record's end is authoritative.
/// The owner publishes active currentPoints as new innerReference in the
/// evaluation prologue. Active groups are read only from currentPoints
/// here, so that publication cannot race a reference read by another warp.
/// A warp barrier orders the stores before evaluation reads the block.
/// No launch, allocation, host control or arithmetic from a pair body.
void emitPruneTemplateWork(mlir::OpBuilder &, mlir::Location,
    const TemplatePruneBuffers &, mlir::Value work, mlir::Value lane);
}
#endif
