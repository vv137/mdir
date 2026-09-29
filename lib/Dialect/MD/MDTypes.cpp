// Types of the md dialect.

#include "mdir/Dialect/MD/MDTypes.h"

#include "mdir/Dialect/MD/MDDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mdir::md;

#include "mdir/Dialect/MD/MDEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// Field element: `f64` or `3 x f64`
//===----------------------------------------------------------------------===//

static ParseResult parseFieldElement(AsmParser &parser,
                                     unsigned &numComponents,
                                     Type &elementType) {
  numComponents = 1;
  OptionalParseResult count = parser.parseOptionalInteger(numComponents);
  if (count.has_value()) {
    if (count.value() || parser.parseKeyword("x"))
      return failure();
  }
  return parser.parseType(elementType);
}

static void printFieldElement(AsmPrinter &printer, unsigned numComponents,
                              Type elementType) {
  if (numComponents != 1)
    printer << numComponents << " x ";
  printer << elementType;
}

#define GET_TYPEDEF_CLASSES
#include "mdir/Dialect/MD/MDTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// FieldType
//===----------------------------------------------------------------------===//

LogicalResult FieldType::verify(function_ref<InFlightDiagnostic()> emitError,
                                FlatSymbolRefAttr particleSet,
                                unsigned numComponents, Type elementType) {
  if (!particleSet)
    return emitError() << "expected a particle set";
  if (numComponents != 1 && numComponents != 3)
    return emitError() << "expected 1 or 3 components, got " << numComponents;

  bool isReal = elementType.isF64();
  bool isInteger =
      elementType.isSignlessInteger(32) || elementType.isSignlessInteger(64);
  if (!isReal && !isInteger)
    return emitError() << "expected element type f64, i32, or i64, got "
                       << elementType;
  if (numComponents == 3 && !isReal)
    return emitError() << "a field with 3 components must have element type "
                          "f64, got "
                       << elementType;
  return success();
}

Type FieldType::getKernelValueType() const {
  if (getNumComponents() == 1)
    return getElementType();
  return VectorType::get({static_cast<int64_t>(getNumComponents())},
                         getElementType());
}

//===----------------------------------------------------------------------===//
// RelationType
//===----------------------------------------------------------------------===//

LogicalResult
RelationType::verify(function_ref<InFlightDiagnostic()> emitError,
                     FlatSymbolRefAttr particleSet, unsigned arity,
                     Orientation orientation) {
  if (!particleSet)
    return emitError() << "expected a particle set";
  if (arity < 2)
    return emitError() << "expected an arity of at least 2, got " << arity;
  return success();
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void MDDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "mdir/Dialect/MD/MDTypes.cpp.inc"
      >();
}
