#ifndef MDIR_MDDIST_DIALECT_H
#define MDIR_MDDIST_DIALECT_H
#include "mdir/Dialect/MDRT/MDRTDialect.h"
#include "mdir/Dialect/MDRT/MDRTTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"

#include "mdir/Dialect/MDDist/MDDistDialect.h.inc"

#define GET_OP_CLASSES
#include "mdir/Dialect/MDDist/MDDistOps.h.inc"
#endif
