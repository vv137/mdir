// Bounded, ordered work windows for one cooperative spatial-search row.
#ifndef MDIR_CONVERSION_ORDEREDROWWORKEMITTER_H
#define MDIR_CONVERSION_ORDEREDROWWORKEMITTER_H
#include "mdir/Conversion/TemplateStageKernels.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
namespace mdir::kernels {
inline constexpr int64_t ORDERED_ROW_JOBS = 128;
inline constexpr int64_t ORDERED_ROW_WARPS = 4;
inline constexpr int64_t ORDERED_ROW_MEMBERS = 16;
// Declared source upper bound: record/keep payload, group/query headers,
// carry and the search's three32-anchor caches. Actual compiled shared
// allocation may remove unused cache slots and is measured separately.
inline constexpr int64_t ORDERED_ROW_SHARED_BYTES =
    128 * 16 * 4 * 4 + 128 * 16 * 4 + 128 * 4 + 128 * 3 * 8 + 2 * 8 + (32 * 3 * 8 + 32 * 4 + 32 * 3 * 4);
static_assert(ORDERED_ROW_SHARED_BYTES == 45840);
static_assert(ORDERED_ROW_SHARED_BYTES <= 48 * 1024);
static_assert(ORDERED_ROW_JOBS % ORDERED_ROW_WARPS == 0);
struct OrderedRowWorkBuffers {
  /// Workgroup buffers: i32[128], i64[128,3], i32[128,16,4],
  /// i32[128,16] and i64[2]. Thread is the block-local index0..127.
  mlir::Value groups, images, records, keeps, carry, thread;
};
struct OrderedRowWorkState {
  TemplateStageState stage;
  /// Block-uniform index; carried through the canonical producer iterator.
  mlir::Value queued;
};
struct OrderedRowRecord {
  mlir::Value keep;
  llvm::SmallVector<mlir::Value, 4> fields;
};
using OrderedRowWorker = llvm::function_ref<OrderedRowRecord(
    mlir::OpBuilder &, mlir::Value group, mlir::ValueRange queryImages)>;
using OrderedRowAppend = llvm::function_ref<TemplateStageState(
    mlir::OpBuilder &, TemplateStageState, mlir::Value keep,
    mlir::ValueRange fields)>;
/// All128threads must enter at the same producer cursor with uniform state.
/// Worker callbacks are warp-local, finite, and contain no block barrier.
/// One job emits at most16 records. Only ownerwarp invokes append; its carry
/// crosses every window and frame. Full/tail windows share the same flush.
OrderedRowWorkState emitFlushOrderedRowWork(mlir::OpBuilder &, mlir::Location,
    const OrderedRowWorkBuffers &, OrderedRowWorkState,
    OrderedRowWorker, OrderedRowAppend);
/// Enqueue one block-uniform raw descriptor/query; only thread0 writes it.
/// No group/pair rejection belongs in the producer. Flushes exactly at128.
OrderedRowWorkState emitEnqueueOrderedRowWork(mlir::OpBuilder &, mlir::Location,
    const OrderedRowWorkBuffers &, OrderedRowWorkState, mlir::Value group,
    mlir::ValueRange queryImages, OrderedRowWorker, OrderedRowAppend);
}
#endif
