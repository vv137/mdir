// The emitter of the lanes of a row of a loop over pairs on the host
// (D402, include/mdir/Dialect/MDExec/HostPairs.h).
// Portability: docs/device-primitives.md, Section 6 (D340): the code is
// `vector`, `arith`, `math`, and `scf`, and one LLVM intrinsic (`powi`),
// with no intrinsic of a host written by hand.

#ifndef MDIR_CONVERSION_HOSTPAIREMITTER_H
#define MDIR_CONVERSION_HOSTPAIREMITTER_H

#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mlir/IR/Builders.h"

namespace mdir {
namespace kernels {

/// Emits what a loop over pairs with `host(lanes(lanes))` does for the
/// particle `central` on the host, in place of emitPairKernel: the row's
/// `n` candidates in floor(n / lanes) full groups read by one `vector.load`
/// of the row of the matrix, then, where `n` is not a multiple of
/// `lanes`, one tail read by `vector.maskedload`, whose padding lanes take
/// the row's last candidate (so that every lane evaluates the kernel at a
/// pair that the scalar row evaluates) and are masked. Each value of the
/// kernel is held as its components, each a `vector<lanes x T>`; a read
/// at a lane's index (the positions, the fields of `ins`, `md.lookup`, the
/// rows of a radial table) is one read a lane. Lane `l` sums the candidates
/// `k = l (mod lanes)` in increasing `k`, in the type of the scalar row's
/// sum; a sum that the kernel widens is widened in each lane, and the lanes
/// are combined by emitLaneReduce. The destinations are then written as by
/// the scalar row. Returns the contributions to the global sums, with
/// their weights applied.
///
/// `op` satisfies md_exec::hostpairs::checkLanes(op, lanes) and is in the
/// storage form; `counts`, `index`, `box`, and `inverse` are as for
/// emitPairKernel.
llvm::SmallVector<mlir::Value>
emitPairRowLanes(mlir::OpBuilder &builder, md_exec::PairForOp op,
                 mlir::Value counts, mlir::Value index, mlir::Value box,
                 mlir::Value inverse, mlir::Value central, int64_t lanes);

} // namespace kernels
} // namespace mdir

#endif // MDIR_CONVERSION_HOSTPAIREMITTER_H
