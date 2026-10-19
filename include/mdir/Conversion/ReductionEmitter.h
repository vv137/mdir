// The emitters of reductions on a device: the butterfly of a warp and the
// reduction of a block (D[reduction-algebra], docs/device-primitives.md,
// Section 5).
//
// A warp combines a value of each lane by a butterfly of `shuffle xor`:
// at offsets 16, 8, 4, 2, 1 each lane combines its value with that of the
// lane whose number differs in that bit, so that after log2(32) rounds
// every lane holds the combination of all 32 (CUB's `WarpReduceShfl`
// [CUB]). The order of the combination is fixed by the lanes, not by the
// threads' timing: the butterfly has a fixed order, and so meets every
// algebra of md_exec::Algebra. A block reduces its warps' results through block memory
// in the order of the warps (D150), CUB's `BLOCK_REDUCE_WARP_REDUCTIONS`.
// The sites keep what is their own: what each thread sums before, and where
// the results go.

#ifndef MDIR_CONVERSION_REDUCTIONEMITTER_H
#define MDIR_CONVERSION_REDUCTIONEMITTER_H

#include "mlir/IR/Builders.h"

namespace mdir {
namespace kernels {

/// The operator of a reduction. A sum of floating-point numbers rounds by
/// its order; a sum of integers and a maximum of numbers do not (a maximum
/// of numbers among which one is not a number does: `OGT` with a NaN
/// selects the second operand).
enum class Combiner { Add, Max };

/// `value` from the lane of the warp whose number differs from this one's
/// by `offset` in its bits: a number of 32 bits in one `gpu.shuffle xor`,
/// one of 64 bits as two halves of 32, a vector element by element.
mlir::Value emitShuffleXor(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::Value value, int64_t offset);

/// `lhs` combined with `rhs`: `arith.addf` or `arith.addi` for a sum (of
/// numbers or vectors), the larger by `arith.cmpf ogt` and a select for a
/// maximum (numbers of floating point).
mlir::Value emitCombine(mlir::OpBuilder &builder, mlir::Location loc,
                        Combiner combiner, mlir::Value lhs, mlir::Value rhs);

/// The combination of `value` over groups of `lanes` adjacent lanes (a
/// power of two up to 32), in every lane of the group: the butterfly at
/// offsets lanes / 2, ..., 2, 1. Every lane of the warp reaches it.
mlir::Value emitWarpReduce(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::Value value, Combiner combiner,
                           int64_t lanes = 32);

/// The combinations of `values` (numbers or vectors of floating point) over
/// a block of `blockSize` threads, `thread` the number of this one: each
/// warp by emitWarpReduce, the first lane of each writes the warp's numbers
/// to `shared` (f64 in block memory, `blockSize / 32` times the count of
/// the numbers), and after one barrier every thread combines the warps in
/// their order (D150). Every thread of the block reaches the barrier.
llvm::SmallVector<mlir::Value>
emitBlockReduce(mlir::OpBuilder &builder, mlir::Location loc,
                mlir::ValueRange values, mlir::Value thread,
                mlir::Value shared, Combiner combiner, int64_t blockSize);

} // namespace kernels
} // namespace mdir

#endif // MDIR_CONVERSION_REDUCTIONEMITTER_H
