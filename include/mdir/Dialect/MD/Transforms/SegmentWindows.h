// Forces and strain of a segment window, and its per-site cutoffs
// (D[kind-pairs]).
#ifndef MDIR_DIALECT_MD_TRANSFORMS_SEGMENTWINDOWS_H
#define MDIR_DIALECT_MD_TRANSFORMS_SEGMENTWINDOWS_H
#include "mlir/IR/Builders.h"
namespace mlir { class IRMapping; }
namespace mdir::md {
class SumSegmentWindowOp;
/// First derivatives of the proved explicit site arithmetic. No alternate
/// placement or force formula is introduced by the physical lowering.
mlir::LogicalResult buildSegmentWindowDerivative(
    mlir::OpBuilder &, SumSegmentWindowOp, mlir::Value weight,
    bool virial, mlir::Value &result);
mlir::LogicalResult checkSegmentWindowActivity(
    SumSegmentWindowOp, mlir::Value positions);
/// Emits the actual per-site cutoff control flow. Only geometry is needed
/// outside each guard; expensive pair arithmetic executes inside it. A
/// gather with site forces returns the frame atoms' forces, after the
/// constant transpose of each molecule's sites.
llvm::SmallVector<mlir::Value> emitSegmentWindowBody(
    mlir::OpBuilder &, mlir::Operation *, mlir::IRMapping &);
}
#endif
