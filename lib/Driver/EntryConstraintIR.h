// Actual rigid-group fields at entry pressure and constraint-trace boundaries.
#ifndef MDIR_LIB_DRIVER_ENTRYCONSTRAINTIR_H
#define MDIR_LIB_DRIVER_ENTRYCONSTRAINTIR_H
#include "EntryFieldsIR.h"
#include "PotentialIR.h"
#include "CouplingIR.h"
namespace mdir::driver::builder {
struct EntryConstraintIR {
  unsigned relation, arity;
  llvm::SmallVector<unsigned> parameters;
  bool water;
};
llvm::SmallVector<RigidGroupIR> buildEntryGroups(mlir::ValueRange,
    llvm::ArrayRef<EntryConstraintIR>);
mlir::Value buildEntryConstraintTrace(mlir::OpBuilder &, const EntryFieldsIR &,
    const PotentialStateIR &, mlir::Value forces, mlir::Value trace,
    llvm::ArrayRef<EntryConstraintIR>, bool axes);
}
#endif
