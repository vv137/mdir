// Host orchestration of the template route's shared pair arena.
#ifndef MDIR_CONVERSION_TEMPLATEREFRESHKERNELS_H
#define MDIR_CONVERSION_TEMPLATEREFRESHKERNELS_H
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <array>
namespace mdir::kernels {
struct TemplateRefreshInputs {
  mlir::Value handle, points, cell, baseReach, extent;
  mlir::Value columnsX, columnsY, waterActive, atomActive;
  mlir::Value reaches;
};
/// One combined sort/search and compaction for an already stale list.
/// This is a cold build operation, not a per-step validity check: its
/// control readback is allowed only after the fused validity selected it.
/// `prepare` updates reference points and retained packing shifts for the
/// requested groups. On capacity growth it is called with both groups
/// active, and all arena views are reacquired before rebuilding all classes.
/// Exact search totals size one growth; a second overflow is a fatal error.
/// Returns the groups rebuilt by the successful attempt. The caller uses
/// these to publish its reference epochs and incident-class build cells.
std::array<mlir::Value, 2> emitRefreshTemplatePairArena(
    mlir::OpBuilder &, mlir::Location, mlir::ModuleOp,
    const TemplateRefreshInputs &,
    llvm::function_ref<void(mlir::OpBuilder &, mlir::Value, mlir::Value)> prepare);
}
#endif
