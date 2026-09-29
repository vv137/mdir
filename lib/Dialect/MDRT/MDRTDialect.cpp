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

/// Verifies that `buffer` is the type of a buffer that can hold `field`.
///
/// The element types may differ if both are floating-point types: the field
/// then has the values of the buffer in another precision, and the precision
/// policy decides which of the two types is stored.
static LogicalResult verifyBufferType(Operation *op, Type buffer, Type field) {
  auto fieldType = cast<FieldType>(field);
  MemRefType expected = getBufferType(fieldType);
  if (buffer == expected)
    return success();

  auto actual = dyn_cast<MemRefType>(buffer);
  if (actual && isa<FloatType>(fieldType.getElementType()) &&
      (actual.getElementType().isF32() || actual.getElementType().isF64()) &&
      actual == MemRefType::get(expected.getShape(), actual.getElementType()))
    return success();

  return op->emitOpError() << "expected the buffer of " << field
                           << " to have type " << expected << ", got "
                           << buffer;
}

LogicalResult FromBufferOp::verify() {
  return verifyBufferType(getOperation(), getBuffer().getType(),
                          getResult().getType());
}

LogicalResult ToBufferOp::verify() {
  return verifyBufferType(getOperation(), getResult().getType(),
                          getField().getType());
}
