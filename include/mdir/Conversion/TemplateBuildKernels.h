// Compaction of the single search's staged chunks in one pair arena.
#ifndef MDIR_CONVERSION_TEMPLATEBUILDKERNELS_H
#define MDIR_CONVERSION_TEMPLATEBUILDKERNELS_H
#include "mdir/Conversion/TemplateGroupKernels.h"
#include "mlir/IR/Builders.h"
namespace mdir::kernels {
struct TemplateBuildBuffers {
  mlir::Value arena, stageUnits, stageEntries;
  mlir::Value counts, entryOffsets, workCounts, workOffsets, scanScratch;
  mlir::Value outerRows, innerRows, outerUnits, innerUnits, outerEntries, innerEntries;
  /// All sizes are index values, from host allocation metadata.
  mlir::Value waters, atoms, rows, unitCapacity, scanBlocks;
  TemplateGroupBuffers groups;
};
/// Compact staged chunks by the fixed row/class order. No spatial search
/// is repeated. Every buffer is a zero-offset view of the same arena.
/// Starts the inner view as the full outer list; the fused loop's pruning
/// can then shorten its blocks without allocating another list.
/// The caller must first handle any search overflow through arena growth;
/// row counts and staged chunk counts must fit their advertised capacities.
void emitCompactTemplatePairs(mlir::OpBuilder &, mlir::Location,
                               const TemplateBuildBuffers &, bool innerOnly = false);
}
#endif
