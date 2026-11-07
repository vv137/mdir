#ifndef MDIR_LIB_DRIVER_FRAMEBATCHEVALUATIONIR_H
#define MDIR_LIB_DRIVER_FRAMEBATCHEVALUATIONIR_H
#include "EntryEvaluationIR.h"
namespace mdir::driver { struct Program; }
namespace mdir::driver::builder {
struct SimpleFrameEntryIRConfig;
struct EntryFieldsIR;
struct FrameBatchEvaluationPoint {
  mlir::Value frame, energies, gradients, strain, status, corrections;
  mlir::ValueRange edges, asks, startValues;
  mlir::Value raw, rows, rowOffsets, sourceIndices, weights;
};
llvm::Error buildFrameBatchEvaluation(mlir::OpBuilder &, const Program &,
    const SimpleFrameEntryIRConfig &, const EntryFieldsIR &,
    const PotentialStateIR &, const FrameBatchEvaluationPoint &,
    llvm::function_ref<EntryCompletionIR(mlir::OpBuilder &, const PotentialStateIR &,
                                        EnergyResults, mlir::Value)> complete,
    mlir::Value scale);
}
#endif
