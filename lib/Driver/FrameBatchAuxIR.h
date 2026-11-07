// Auxiliary frame columns, in the order exposed by the native caller.
#ifndef MDIR_LIB_DRIVER_FRAMEBATCHAUXIR_H
#define MDIR_LIB_DRIVER_FRAMEBATCHAUXIR_H
#include "EntryEvaluationIR.h"
#include "llvm/Support/Error.h"
namespace mdir::driver { struct Program; }
namespace mdir::driver::builder {
struct SimpleFrameEntryIRConfig;
struct EntryFieldsIR;
struct FrameBatchEvaluationPoint;
llvm::Error buildFrameBatchAux(mlir::OpBuilder &, const Program &,
    const SimpleFrameEntryIRConfig &, const EntryFieldsIR &,
    const PotentialStateIR &, EnergyResults, const FrameBatchEvaluationPoint &,
    mlir::Value volume, mlir::Value volumeScale,
    mlir::Value dispersionVirial, mlir::Value coulombVirial);
}
#endif
