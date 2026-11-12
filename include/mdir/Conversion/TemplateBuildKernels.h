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
  /// Optional: the search's four i64 controls and the entry capacity (an
  /// i64 host value). With them the compaction publishes nothing unless
  /// the search's flags are clear and its entries fit, so that it may run
  /// before the host has read the controls.
  mlir::Value counters, entryCapacity;
};
/// Compact staged chunks by the fixed row/class order. No spatial search
/// is repeated. Every buffer is a zero-offset view of the same arena.
/// Starts the inner view as the full outer list; the fused loop's pruning
/// can then shorten its blocks without allocating another list.
/// Without `counters`, the caller must first handle any search overflow
/// through arena growth; with them, it handles the overflow after the
/// compaction, which then wrote no rows, units or entries. Row counts must
/// fit their advertised capacity either way.
void emitCompactTemplatePairs(mlir::OpBuilder &, mlir::Location,
                               const TemplateBuildBuffers &);
}
#endif
