// Shared typed core for native and transitional pull-coordinate output.
#ifndef MDIR_LIB_DRIVER_PULLOUTPUTIR_H
#define MDIR_LIB_DRIVER_PULLOUTPUTIR_H
#include "PotentialIR.h"
#include "llvm/ADT/ArrayRef.h"
#include <cstddef>
namespace mdir::driver::builder {
struct PullOutputTermIR {
  size_t index, groups;
  unsigned arity;
  mlir::Value relation;
  llvm::SmallVector<mlir::Value> weights;
};
void buildPullOutput(mlir::OpBuilder &, const PotentialStateIR &, mlir::Value step,
                     llvm::ArrayRef<PullOutputTermIR>);
}
#endif
