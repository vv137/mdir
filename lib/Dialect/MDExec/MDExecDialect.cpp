// The md_exec dialect.

#include "mdir/Dialect/MDExec/MDExecDialect.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"

using namespace mlir;
using namespace mdir::md_exec;

#include "mdir/Dialect/MDExec/MDExecDialect.cpp.inc"

void MDExecDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mdir/Dialect/MDExec/MDExecOps.cpp.inc"
      >();
}
