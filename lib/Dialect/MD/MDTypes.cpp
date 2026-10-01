// Types of the md dialect.

#include "mdir/Dialect/MD/MDTypes.h"

#include "mdir/Dialect/MD/MDDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mdir::md;

#include "mdir/Dialect/MD/MDEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// Field element: `f64` or `3 x f64`
//===----------------------------------------------------------------------===//

/// Parses and prints `, symmetric`, which a table of rank 2 may have.
static ParseResult parseSymmetry(AsmParser &parser, bool &symmetric) {
  symmetric = succeeded(parser.parseOptionalComma());
  if (symmetric && parser.parseKeyword("symmetric"))
    return failure();
  return success();
}

static void printSymmetry(AsmPrinter &printer, bool symmetric) {
  if (symmetric)
    printer << ", symmetric";
}

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

  bool isReal = elementType.isF64() || elementType.isF32();
  bool isInteger =
      elementType.isSignlessInteger(32) || elementType.isSignlessInteger(64);
  if (!isReal && !isInteger)
    return emitError() << "expected element type f64, f32, i32, or i64, got "
                       << elementType;
  if (numComponents == 3 && !isReal)
    return emitError() << "a field with 3 components must have element type "
                          "f64 or f32, got "
                       << elementType;
  return success();
}

Type FieldType::getKernelValueType() const {
  if (getNumComponents() == 1)
    return getElementType();
  return VectorType::get({static_cast<int64_t>(getNumComponents())},
                         getElementType());
}

LogicalResult mdir::md::verifyReferencePrecision(Operation *op,
                                                 FunctionType type) {
  for (Type part : llvm::concat<const Type>(type.getInputs(),
                                            type.getResults())) {
    auto field = dyn_cast<FieldType>(part);
    if (field && field.getElementType().isF32())
      return op->emitOpError()
             << "expected fields at the reference precision, with elements "
                "of type f64, got "
             << part;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// RelationType
//===----------------------------------------------------------------------===//

LogicalResult TableType::verify(function_ref<InFlightDiagnostic()> emitError,
                                unsigned rank, Type elementType,
                                bool symmetric) {
  if (rank != 1 && rank != 2)
    return emitError() << "expected a table of rank 1 or 2, got " << rank;
  // A vector holds the values of several tables at one entry, which a
  // kernel reads with one load (`md-exec-fold-tables`).
  Type scalar = elementType;
  if (auto vector = dyn_cast<VectorType>(elementType))
    if (vector.getRank() == 1 && !vector.isScalable())
      scalar = vector.getElementType();
  if (!scalar.isF64() && !scalar.isF32())
    return emitError() << "expected element type f64 or f32, or a vector of "
                          "them, got "
                       << elementType;
  if (symmetric && rank != 2)
    return emitError() << "only a table of rank 2 can be symmetric";
  return success();
}

LogicalResult
RelationType::verify(function_ref<InFlightDiagnostic()> emitError,
                     FlatSymbolRefAttr particleSet, unsigned arity,
                     Orientation orientation, FlatSymbolRefAttr tupleSet) {
  if (!particleSet)
    return emitError() << "expected a particle set";
  // A tuple of a tuple set may have one member, as a restraint of a
  // particle has.
  unsigned least = tupleSet ? 1 : 2;
  if (arity < least)
    return emitError() << "expected an arity of at least " << least
                       << ", got " << arity;
  if (orientation == Orientation::Unordered && arity != 2)
    return emitError() << "expected the orientation 'unordered' with the "
                          "arity 2 only, got the arity "
                       << arity;
  if (orientation == Orientation::Reversal && arity < 2)
    return emitError() << "expected the orientation 'reversal' with an "
                          "arity of at least 2";
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
