// The md dialect.

#include "mdir/Dialect/MD/MDDialect.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MD/MDTypes.h"

using namespace mlir;
using namespace mdir::md;

#include "mdir/Dialect/MD/MDDialect.cpp.inc"

void MDDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mdir/Dialect/MD/MDOps.cpp.inc"
      >();
  registerTypes();
}
