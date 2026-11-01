// Completed physical evaluation and ordered post-evaluation term energies.
#ifndef MDIR_LIB_DRIVER_ENTRYEVALUATIONIR_H
#define MDIR_LIB_DRIVER_ENTRYEVALUATIONIR_H
#include "PotentialIR.h"
#include "DynamicsIR.h"
#include "llvm/ADT/ArrayRef.h"
namespace mdir::driver::builder {
struct EntryCompletionIR {
  EnergyResults physical;
  llvm::SmallVector<mlir::Value> storedTerms;
};
mlir::Value buildEntrySites(mlir::OpBuilder &, const PotentialStateIR &,
                            llvm::ArrayRef<DescentSiteIR>);
EntryCompletionIR buildEntryEvaluation(mlir::OpBuilder &, const PotentialStateIR &,
    EnergyRequests, llvm::ArrayRef<DescentSiteIR>,
    llvm::function_ref<EntryCompletionIR(mlir::OpBuilder &, const PotentialStateIR &,
                                        EnergyResults, mlir::Value scale)> complete = {},
    mlir::Value scale = {});
}
#endif
