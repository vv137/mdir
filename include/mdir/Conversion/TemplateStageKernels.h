// Ballot-compacted chunks produced by a single spatial search.
#ifndef MDIR_CONVERSION_TEMPLATESTAGEKERNELS_H
#define MDIR_CONVERSION_TEMPLATESTAGEKERNELS_H
#include "mlir/IR/Builders.h"
namespace mdir::kernels {
struct TemplateStageBuffers {
  mlir::Value units, entries, counters, counts;
  /// i64 capacity, i32 fixed row and lane. All lanes of one warp own a row.
  mlir::Value capacity, row, lane;
};
struct TemplateStageState {
  /// Warp-uniform i64 row-entry count and last reserved chunk (-1 initially).
  mlir::Value count, chunk;
};
/// Append zero or one four-i32 record per lane. All 32 lanes participate;
/// kept records retain lane order. A partial chunk carries across rounds,
/// and a round reserves at most one new chunk. Out-of-capacity writes are
/// skipped and flagged; counts remain exact for the caller's growth path.
TemplateStageState emitAppendTemplateStage(mlir::OpBuilder &, mlir::Location,
    const TemplateStageBuffers &, TemplateStageState, mlir::Value keep,
    mlir::ValueRange record);
/// Publish the row count once and add it to the i64 build total. Counts
/// beyond the final i32 payload domain set the index-overflow flag; the
/// caller must stop before compaction instead of narrowing such a build.
void emitFinishTemplateStage(mlir::OpBuilder &, mlir::Location,
    const TemplateStageBuffers &, TemplateStageState);
}
#endif
