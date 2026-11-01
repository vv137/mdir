// Completed physical evaluation and ordered post-evaluation term energies.
#ifndef MDIR_LIB_DRIVER_ENTRYEVALUATIONIR_H
#define MDIR_LIB_DRIVER_ENTRYEVALUATIONIR_H
#include "PotentialIR.h"
namespace mdir::driver::builder {
struct EntryCompletionIR {
  EnergyResults physical;
  llvm::SmallVector<mlir::Value> storedTerms;
};
}
#endif
