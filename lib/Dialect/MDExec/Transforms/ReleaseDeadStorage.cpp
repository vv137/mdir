// The release, before the loop over steps, of the device buffers that the
// rest of the function never uses (D[md-exec-release-dead], #885).
//
// See the description of the pass in Passes.td. The proof is the liveness
// of the storage (StorageLiveness.h, D412) in the mode `kills=proven`. A
// buffer is released after the op of the function's body that holds its
// last occupied point: a point where it is touched, live in, or live out,
// with the waits and the joins that complete work on it counted as
// touches. Occupancy, not liveness alone, decides: a buffer that a later op
// overwrites whole is dead before the write but its storage is still used
// there.

#include "mdir/Dialect/MDExec/Transforms/Passes.h"

#include "mdir/Dialect/MDExec/StorageLiveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::md_exec;

namespace mdir {
namespace md_exec {

#define GEN_PASS_DEF_RELEASEDEADSTORAGE
#include "mdir/Dialect/MDExec/Transforms/Passes.h.inc"

namespace {

/// The op of the function's body after which `buffer` may be released, or
/// null if it must keep its storage to the end.
static Operation *findRelease(const StorageLiveness &liveness,
                              const StorageBuffer &buffer, Block &body) {
  gpu::AllocOp alloc = buffer.alloc;
  // Kept: an escape or a use that the analysis cannot bound, a buffer the
  // loop reads at the start of a step, one read after the loop, one that
  // the code releases itself, and one that nothing uses (the lowering
  // removes it). Memory shared with the host and an allocation on a stream
  // are not the pool's blocks of the storage.
  if (!buffer.pinned.empty() || buffer.carried || buffer.liveAfterLoop ||
      buffer.released || alloc.getMemref().use_empty() ||
      alloc.getHostShared() || alloc.getAsyncToken())
    return nullptr;
  // Occupied at no point of the step.
  if (buffer.stepOccupied.any())
    return nullptr;
  int last = buffer.occupied.find_last();
  if (last < 0)
    return nullptr;
  Operation *at = liveness.getPoints()[last].op;
  if (!at || isa<func::FuncOp>(at))
    return nullptr;
  // The points follow the order of the code, so no point after the last
  // occupied one occupies the buffer: the release goes after the op of the
  // body that holds that point, which must come before the loop.
  Operation *top = body.findAncestorOpInBlock(*at);
  Operation *loop = liveness.getStepLoop();
  if (!top || top == loop || !top->isBeforeInBlock(loop))
    return nullptr;
  return top;
}

class ReleaseDeadStorage
    : public impl::ReleaseDeadStorageBase<ReleaseDeadStorage> {
public:
  using impl::ReleaseDeadStorageBase<
      ReleaseDeadStorage>::ReleaseDeadStorageBase;

  void runOnOperation() final {
    for (auto function : getOperation().getOps<func::FuncOp>()) {
      if (function.isExternal())
        continue;
      StorageLiveness liveness(function, StorageKills::Proven);
      if (!liveness.hasStepLoop())
        continue;
      Block &body = function.getBody().front();
      // The releases after one op keep the order of the buffers.
      llvm::DenseMap<Operation *, Operation *> lastRelease;
      for (const StorageBuffer &buffer : liveness.getBuffers()) {
        Operation *after = findRelease(liveness, buffer, body);
        if (!after)
          continue;
        auto it = lastRelease.try_emplace(after, after).first;
        OpBuilder builder(function.getContext());
        builder.setInsertionPointAfter(it->second);
        // gpu.dealloc takes the buffer without its memory space, as the
        // lowering's own releases do.
        gpu::AllocOp alloc = buffer.alloc;
        Value memref = alloc.getMemref();
        auto type = cast<MemRefType>(memref.getType());
        if (type.getMemorySpace())
          memref = memref::MemorySpaceCastOp::create(
              builder, alloc.getLoc(),
              MemRefType::get(type.getShape(), type.getElementType(),
                              type.getLayout()),
              memref);
        it->second = gpu::DeallocOp::create(builder, alloc.getLoc(),
                                            /*asyncToken=*/Type(),
                                            ValueRange(), memref);
      }
    }
  }
};

} // namespace

} // namespace md_exec
} // namespace mdir
