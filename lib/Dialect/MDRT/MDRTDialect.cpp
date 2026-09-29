// The mdrt dialect, its types, and its ops.

#include "mdir/Dialect/MDRT/MDRTDialect.h"
#include "mdir/Dialect/MDRT/MDRTOps.h"
#include "mdir/Dialect/MDRT/MDRTTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mdir::mdrt;
using mdir::md::FieldType;

#include "mdir/Dialect/MDRT/MDRTDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mdir/Dialect/MDRT/MDRTTypes.cpp.inc"

#define GET_OP_CLASSES
#include "mdir/Dialect/MDRT/MDRTOps.cpp.inc"

void MDRTDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "mdir/Dialect/MDRT/MDRTTypes.cpp.inc"
      >();
}

void MDRTDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mdir/Dialect/MDRT/MDRTOps.cpp.inc"
      >();
  registerTypes();
}

MemRefType mdir::mdrt::getBufferType(FieldType field) {
  if (field.getNumComponents() == 1)
    return MemRefType::get({ShapedType::kDynamic}, field.getElementType());
  return MemRefType::get({ShapedType::kDynamic, field.getNumComponents()},
                         field.getElementType());
}

/// Verifies that `buffer` is the type of the buffer that holds `field`.
static LogicalResult verifyBufferType(Operation *op, Type buffer, Type field) {
  MemRefType expected = getBufferType(cast<FieldType>(field));
  if (buffer != expected)
    return op->emitOpError() << "expected the buffer of " << field
                             << " to have type " << expected << ", got "
                             << buffer;
  return success();
}

LogicalResult FromBufferOp::verify() {
  return verifyBufferType(getOperation(), getBuffer().getType(),
                          getResult().getType());
}

LogicalResult ToBufferOp::verify() {
  return verifyBufferType(getOperation(), getResult().getType(),
                          getField().getType());
}
