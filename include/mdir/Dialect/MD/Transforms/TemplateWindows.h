#ifndef MDIR_DIALECT_MD_TRANSFORMS_TEMPLATEWINDOWS_H
#define MDIR_DIALECT_MD_TRANSFORMS_TEMPLATEWINDOWS_H
#include "mlir/IR/Builders.h"
namespace mlir { class IRMapping; }
namespace mdir::md {
class SumTemplateWindowOp;
/// First derivatives of the proved explicit site arithmetic. No alternate
/// placement or force formula is introduced by the physical lowering.
mlir::LogicalResult buildTemplateWindowDerivative(
    mlir::OpBuilder &, SumTemplateWindowOp, mlir::Value weight,
    bool virial, mlir::Value &result);
mlir::LogicalResult checkTemplateWindowActivity(
    SumTemplateWindowOp, mlir::Value positions);
/// Emits the actual per-site cutoff control flow. Only geometry is needed
/// outside each guard; expensive pair arithmetic executes inside it.
llvm::SmallVector<mlir::Value> emitTemplateWindowBody(
    mlir::OpBuilder &, mlir::Operation *, mlir::IRMapping &);
}
#endif
