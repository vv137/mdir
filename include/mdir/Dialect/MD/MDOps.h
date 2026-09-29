// Ops of the md dialect.

#ifndef MDIR_DIALECT_MD_MDOPS_H
#define MDIR_DIALECT_MD_MDOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mdir/Dialect/MD/MDCoordinates.h"
#include "mdir/Dialect/MD/MDTypes.h"

#define GET_OP_CLASSES
#include "mdir/Dialect/MD/MDOps.h.inc"

#endif // MDIR_DIALECT_MD_MDOPS_H
