// The mdrt dialect, its types, and its ops.

#include "mdir/Dialect/MDRT/MDRTDialect.h"
#include "mdir/Dialect/MDRT/MDRTOps.h"
#include "mdir/Dialect/MDRT/MDRTTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
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

MemRefType mdir::mdrt::getMembersType(md::RelationType relation) {
  return MemRefType::get(
      {ShapedType::kDynamic, static_cast<int64_t>(relation.getArity())},
      IntegerType::get(relation.getContext(), 32));
}

bool mdir::mdrt::canHold(Type buffer, FieldType field) {
  MemRefType expected = getBufferType(field);
  if (buffer == expected)
    return true;
  auto actual = dyn_cast<MemRefType>(buffer);
  return actual && isa<FloatType>(field.getElementType()) &&
         (actual.getElementType().isF32() ||
          actual.getElementType().isF64()) &&
         actual == MemRefType::get(expected.getShape(),
                                   actual.getElementType());
}

/// Verifies that `buffer` is the type of a buffer that can hold `field`.
static LogicalResult verifyBufferType(Operation *op, Type buffer, Type field) {
  auto fieldType = cast<FieldType>(field);
  if (canHold(buffer, fieldType))
    return success();
  return op->emitOpError() << "expected the buffer of " << field
                           << " to have type " << getBufferType(fieldType)
                           << ", got " << buffer;
}

LogicalResult FromBufferOp::verify() {
  auto relation = dyn_cast<md::RelationType>(getResult().getType());
  if (!relation)
    return verifyBufferType(getOperation(), getBuffer().getType(),
                            getResult().getType());

  if (!relation.getTupleSet())
    return emitOpError() << "expected the relation of a tuple set, got "
                         << relation
                         << ": the pairs of a neighborhood are found, not "
                            "given";
  if (getBuffer().getType() != getMembersType(relation))
    return emitOpError() << "expected the buffer of " << relation
                         << " to have type " << getMembersType(relation)
                         << ", got " << getBuffer().getType();
  return success();
}

LogicalResult ToBufferOp::verify() {
  return verifyBufferType(getOperation(), getResult().getType(),
                          getField().getType());
}

LogicalResult
HostCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto callee = symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(
      getOperation(), getCalleeAttr());
  if (!callee)
    return emitOpError() << "'" << getCallee()
                         << "' does not name a function";

  ArrayRef<Type> expected = callee.getArgumentTypes();
  if (expected.size() != getNumOperands())
    return emitOpError() << "'" << getCallee() << "' takes "
                         << expected.size() << " arguments, got "
                         << getNumOperands();
  if (!callee.getResultTypes().empty())
    return emitOpError() << "'" << getCallee()
                         << "' returns a value; a function of the host "
                            "that reads fields returns none";

  for (unsigned i = 0, e = expected.size(); i != e; ++i) {
    Type type = getOperand(i).getType();
    auto field = dyn_cast<FieldType>(type);
    bool fits = field ? canHold(expected[i], field) : type == expected[i];
    if (!fits)
      return emitOpError()
             << "argument " << i << " of '" << getCallee() << "' has type "
             << expected[i] << ", which does not take " << type;
  }
  return success();
}
