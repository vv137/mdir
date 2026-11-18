#ifndef MDIR_DIALECT_MDEXEC_PAIRKERNELBODY_H
#define MDIR_DIALECT_MDEXEC_PAIRKERNELBODY_H

#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/SmallVector.h"

namespace mdir::md_exec {
struct PairKernelBody {
  mlir::Region *region;
  mlir::Value squaredDistance;
  llvm::SmallVector<mlir::Value> outputs;
};
} // namespace mdir::md_exec

#include "mdir/Dialect/MDExec/PairKernelBody.h.inc"
#endif
