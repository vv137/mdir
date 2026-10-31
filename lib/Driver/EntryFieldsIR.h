// Typed imports preserve storage precision and logical field types.
#ifndef MDIR_LIB_DRIVER_ENTRYFIELDSIR_H
#define MDIR_LIB_DRIVER_ENTRYFIELDSIR_H
#include "EntryBoundaryIR.h"
namespace mdir::driver::builder {
struct EntryFieldsIR {
  mlir::Value positions, velocities, masses, identities, forces;
  llvm::SmallVector<mlir::Value> fields, tables;
  llvm::SmallVector<EntryTupleBuffersIR> tuples;
};
mlir::Value buildEntryField(mlir::OpBuilder &, mlir::Value buffer,
    unsigned components, bool integer, llvm::StringRef set = "atoms");
EntryFieldsIR buildEntryFields(mlir::OpBuilder &, const Program &,
    const EntryArgumentsIR &, bool positions, bool givenForces);
}
#endif
