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

LogicalResult MDDialect::verifyOperationAttribute(Operation *op,
                                                  NamedAttribute attr) {
  if (attr.getName() == kStageRowsAttrName)
    return verifyStageRowsAttribute(op, attr.getValue());
  if (attr.getName() == kStageBodyAttrName)
    return verifyStageBodyAttribute(op, attr.getValue());
  return success();
}
