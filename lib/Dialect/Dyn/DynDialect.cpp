// The dyn dialect.

#include "mdir/Dialect/Dyn/DynDialect.h"
#include "mdir/Dialect/Dyn/DynOps.h"

using namespace mlir;
using namespace mdir::dyn;

#include "mdir/Dialect/Dyn/DynDialect.cpp.inc"

void DynDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mdir/Dialect/Dyn/DynOps.cpp.inc"
      >();
}
