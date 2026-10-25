// The emitters of the privatized scatter-accumulate on a device
// (D291, docs/device-primitives.md, Section 2.6).
// Portability: docs/device-primitives.md, Section 6 (D340).
//
// A site that adds values into destinations at indices that the threads of
// a block share emits its kernel and calls these to give the block a copy of
// what it touches in block memory: a table keyed by the index, cleared at
// the start of the kernel, added to by the threads, and flushed to the
// destinations with one atomic of the device for each index it holds. This
// is the atomic specialization of CUB's BlockHistogram [CUB], with a value
// in place of a count and the keys found by hashing; the strategy is
// `scatter(hash(S, P, L))` (include/mdir/Dialect/MDExec/Scatter.h holds its
// preconditions, which the chooser and the sites check). Every site of the
// strategy calls the same emitters, so that a change of the strategy's code
// is made once.

#ifndef MDIR_CONVERSION_SCATTEREMITTER_H
#define MDIR_CONVERSION_SCATTEREMITTER_H

#include "mdir/Dialect/MDExec/Scatter.h"
#include "mlir/IR/Builders.h"

#include "llvm/ADT/DenseMap.h"

namespace mdir {
namespace kernels {

//===----------------------------------------------------------------------===//
// Block memory
//===----------------------------------------------------------------------===//

/// The address of element `index` of `buffer`, a buffer in block memory,
/// as a number of 64 bits, for the instructions of PTX that take it.
mlir::Value getSharedAddress(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::Value buffer, mlir::Value index);

/// Adds `value`, of f32 or f64, to block memory at `address` (from
/// getSharedAddress), with no result: PTX's `red.shared.add`.
void emitSharedAdd(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::Value address, mlir::Value value);

/// Compares the i32 at `address` in block memory with `expected` and, if
/// they are equal, stores `desired`; returns what was there: PTX's
/// `atom.shared.cas.b32`.
mlir::Value emitSharedCompareSwap(mlir::OpBuilder &builder,
                                  mlir::Location loc, mlir::Value address,
                                  mlir::Value expected, mlir::Value desired);

//===----------------------------------------------------------------------===//
// hash(S, P, L): a table of a block
//===----------------------------------------------------------------------===//

/// The block memory of a table of `slots` slots for `destinations`, as the
/// workgroup attributions of the launch: the keys (i32, one a slot), then
/// the sums of each destination (its components a slot).
llvm::SmallVector<mlir::Type>
getHashAttributions(mlir::MLIRContext *context, int64_t slots,
                    llvm::ArrayRef<md_exec::scatter::Destination> destinations);

/// A table of a block in the kernel of a launch whose workgroup
/// attributions getHashAttributions gave (`shared`). `outs` are the
/// destinations on the device, in the order of `destinations`. The thread
/// `thread` of `threads` takes the slots thread, thread + threads, ... to
/// clear and flush. A search tries at most `probes` slots, from a
/// multiplicative hash of the index [Knuth1998, Section 6.4], and an index
/// it does not place goes to its destination with an atomic of its own.
///
/// The kernel calls `emitClear` first, `add` for each value, and
/// `emitFlush` last, every thread of the block reaching both, since they
/// hold barriers of the block.
class BlockHashTable {
public:
  BlockHashTable(mlir::OpBuilder &kernel, mlir::Location loc, int64_t slots,
                 int64_t probes, mlir::ValueRange shared,
                 llvm::ArrayRef<mlir::Value> outs,
                 llvm::ArrayRef<md_exec::scatter::Destination> destinations,
                 mlir::Value thread, mlir::Value threads);

  /// Every slot empty and its sums zero, then a barrier.
  void emitClear(mlir::OpBuilder &kernel);

  /// Adds `value`, a number or a vector of the components of a row, for the
  /// index `index` of the destination `out`: to the table where the index
  /// has or finds a slot, to the destination with an atomic otherwise. The
  /// slot of an index is looked up once in a block of the IR and reused by
  /// the adds of that index there.
  void add(mlir::OpBuilder &builder, mlir::Value out, mlir::Value index,
           mlir::Value value);

  /// A barrier, then each index of the table added to each destination
  /// once.
  void emitFlush(mlir::OpBuilder &kernel);

private:
  /// The slot of `index` and whether the search found one.
  std::pair<mlir::Value, mlir::Value> emitFind(mlir::OpBuilder &builder,
                                               mlir::Value index);

  mlir::Location loc;
  int64_t slots, probes;
  unsigned shift;
  mlir::Value keys;
  llvm::SmallVector<mlir::Value> sums;
  llvm::SmallVector<mlir::Value> outs;
  llvm::SmallVector<int64_t> components;
  llvm::SmallVector<mlir::Type> elements;
  mlir::Value thread, threads, numSlots, empty;
  llvm::DenseMap<mlir::Value, std::pair<mlir::Value, mlir::Value>> found;
};

} // namespace kernels
} // namespace mdir

#endif // MDIR_CONVERSION_SCATTEREMITTER_H
