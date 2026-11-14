// The liveness of the storage of md_exec within the loop over steps
// (D[md-exec-storage-liveness], #880).
//
// The storage of md_exec is the device buffers that `md-exec-assign-storage`
// and `convert-md-exec-to-gpu` allocate in the host function before its loop
// over steps and keep to its end. The analysis runs on the host code as
// `convert-md-exec-to-gpu` leaves it (the slot of
// `md-alg-transform-collections`): every buffer is a `gpu.alloc`, every
// kernel a `gpu.launch` whose body names the buffers it touches, and the
// host's order of its launches is the order of the code. It is a backward
// dataflow of liveness over the host's control flow, with buffers in place
// of the variables of register allocation (Chaitin, SIGPLAN Notices 17(6),
// 98 (1982)): a use is
// a launch, a call, or a copy that reads a buffer or a view of it, and a kill
// is one that overwrites the whole buffer without reading it. Two buffers
// interfere when one is in use where the other is live; buffers that do not
// interfere could share one allocation, which is the bound of reuse by
// lifetime. The analysis changes no IR.

#ifndef MDIR_DIALECT_MDEXEC_STORAGELIVENESS_H
#define MDIR_DIALECT_MDEXEC_STORAGELIVENESS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/AsmState.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace mdir::md_exec {

/// What counts as a kill, an overwrite of a whole buffer that makes its
/// earlier contents dead.
enum class StorageKills {
  /// A copy or a fill of the whole buffer (`gpu.memcpy` into it,
  /// `gpu.memset`), which is proven.
  Proven,
  /// Also a launch or a call that writes the buffer through a view of all of
  /// it and reads none of it. That the kernel writes every element is
  /// assumed, not proven: the bound it gives is an upper bound.
  Stores,
};

/// One buffer of the storage: a `gpu.alloc` in the function's body before
/// its loop over steps.
struct StorageBuffer {
  mlir::gpu::AllocOp alloc;
  /// Its position among the buffers, in the order of the code.
  unsigned index = 0;
  /// Live at every point: the reason (a raw pointer that leaves the host
  /// code, a token never waited for, a return, ...), or empty.
  std::string pinned;
  /// A `gpu.dealloc` of it, or of a view of it, in the function.
  bool released = false;
  /// Live where the loop over steps begins an iteration: its value at the
  /// start of the step is read in the step (a buffer carried by the loop,
  /// or one that a step reads before it overwrites it).
  bool carried = false;
  /// Live where the loop exits: read after the loop.
  bool liveAfterLoop = false;
  /// Live at the points of the step, and at all the points of the function.
  llvm::BitVector stepOccupied, occupied;
  /// The first and the last point of the step that touch it, or -1.
  int firstAccess = -1, lastAccess = -1;
  /// The buffers it interferes with, in the step and in the function.
  llvm::BitVector stepInterferes, interferes;
};

/// A point of the host's control flow that touches the storage, or joins
/// the flow.
struct StoragePoint {
  mlir::Operation *op = nullptr;
  /// In the loop over steps, and its position there (-1 outside).
  int stepIndex = -1;
  llvm::BitVector use, kill, access;
  llvm::BitVector liveIn, liveOut;
  llvm::SmallVector<unsigned, 2> successors;
};

/// The liveness of the storage of one host function with a loop over
/// steps.
class StorageLiveness {
public:
  StorageLiveness(mlir::func::FuncOp function, StorageKills kills);

  /// Whether the function has a loop over steps: a loop in its body that
  /// launches a kernel or calls a function of the module.
  bool hasStepLoop() const { return stepLoop != nullptr; }
  mlir::Operation *getStepLoop() const { return stepLoop; }
  llvm::ArrayRef<StorageBuffer> getBuffers() const { return buffers; }
  llvm::ArrayRef<StoragePoint> getPoints() const { return points; }
  /// The number of points in the loop over steps.
  unsigned getStepPoints() const { return stepPoints; }
  /// The buffers live at each call to `callee`, in the order of the points,
  /// with the call's position in the step (-1 outside it).
  llvm::SmallVector<std::pair<int, llvm::BitVector>>
  liveAtCalls(llvm::StringRef callee) const;

  void print(llvm::raw_ostream &os, mlir::AsmState &state,
             llvm::StringRef at = {}) const;

private:
  mlir::func::FuncOp function;
  StorageKills kills;
  mlir::Operation *stepLoop = nullptr;
  llvm::SmallVector<StorageBuffer, 0> buffers;
  llvm::SmallVector<StoragePoint, 0> points;
  unsigned stepPoints = 0;

  friend class StorageLivenessBuilder;
};

} // namespace mdir::md_exec

#endif // MDIR_DIALECT_MDEXEC_STORAGELIVENESS_H
