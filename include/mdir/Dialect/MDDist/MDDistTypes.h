// Types of the md_dist dialect.

#ifndef MDIR_DIALECT_MDDIST_MDDISTTYPES_H
#define MDIR_DIALECT_MDDIST_MDDISTTYPES_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

#include "mdir/Dialect/MDDist/MDDistEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "mdir/Dialect/MDDist/MDDistTypes.h.inc"

#endif // MDIR_DIALECT_MDDIST_MDDISTTYPES_H
