// Native entry arguments are physical buffers and semantic parameters.
#ifndef MDIR_LIB_DRIVER_ENTRYBOUNDARYIR_H
#define MDIR_LIB_DRIVER_ENTRYBOUNDARYIR_H
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/Error.h"
namespace mdir::driver { struct Program; }
namespace mdir::driver::builder {
struct EntryTupleBuffersIR {
  mlir::Value members;
  llvm::SmallVector<mlir::Value> fields;
};
struct EntryArgumentsIR {
  mlir::Value positions, velocities, forces, masses, identities;
  llvm::SmallVector<mlir::Value> fields, tables;
  llvm::SmallVector<EntryTupleBuffersIR> tuples;
  mlir::Value lengthX, lengthY, lengthZ, interval, start;
  mlir::Value firstSize, firstCall;
  llvm::SmallVector<mlir::Value> startValues;
};
struct EntryBoundaryConfigIR {
  bool segmentedMinimization = false, branchesStart = false;
};
llvm::Expected<mlir::func::FuncOp> buildEntryFunction(
    mlir::OpBuilder &, const Program &, const EntryBoundaryConfigIR &,
    llvm::function_ref<llvm::Error(mlir::OpBuilder &, const EntryArgumentsIR &)>);
}
#endif
