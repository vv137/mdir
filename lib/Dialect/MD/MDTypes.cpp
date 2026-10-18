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

/// Parses `N x T`, `T`, or `irreps<M x l p + ...[, mul_ir | ir_mul]>, T`,
/// a block written as e3nn writes it, `4x1o`: the multiplicity, then a
/// keyword of `x`, the degree, and the parity.
static ParseResult parseFieldElement(AsmParser &parser,
                                     unsigned &numComponents,
                                     Type &elementType,
                                     SmallVector<int64_t> &irreps,
                                     bool &componentMajor) {
  numComponents = 1;
  componentMajor = false;
  if (succeeded(parser.parseOptionalKeyword("irreps"))) {
    if (parser.parseLess())
      return failure();
    int64_t total = 0;
    do {
      int64_t multiplicity;
      StringRef block;
      SMLoc loc = parser.getCurrentLocation();
      if (parser.parseInteger(multiplicity) || parser.parseKeyword(&block))
        return failure();
      int64_t degree;
      if (block.size() < 3 || block.front() != 'x' ||
          !(block.back() == 'e' || block.back() == 'o') ||
          block.drop_front().drop_back().getAsInteger(10, degree))
        return parser.emitError(loc)
               << "expected a block of the form '<M>x<l><e|o>', such as "
                  "'4x1o'";
      irreps.append({multiplicity, degree, block.back() == 'o' ? 1 : 0});
      total += multiplicity * (2 * degree + 1);
    } while (succeeded(parser.parseOptionalPlus()));
    if (succeeded(parser.parseOptionalComma())) {
      StringRef layout;
      SMLoc loc = parser.getCurrentLocation();
      if (parser.parseKeyword(&layout))
        return failure();
      if (layout != "mul_ir" && layout != "ir_mul")
        return parser.emitError(loc)
               << "expected the layout 'mul_ir' or 'ir_mul', got '" << layout
               << "'";
      componentMajor = layout == "ir_mul";
    }
    if (parser.parseGreater() || parser.parseComma())
      return failure();
    numComponents = total > 0 ? static_cast<unsigned>(total) : 0;
    return parser.parseType(elementType);
  }
  OptionalParseResult count = parser.parseOptionalInteger(numComponents);
  if (count.has_value()) {
    if (count.value() || parser.parseKeyword("x"))
      return failure();
  }
  return parser.parseType(elementType);
}

static void printFieldElement(AsmPrinter &printer, unsigned numComponents,
                              Type elementType, ArrayRef<int64_t> irreps,
                              bool componentMajor) {
  if (!irreps.empty()) {
    printer << "irreps<";
    for (size_t k = 0; k + 2 < irreps.size(); k += 3)
      printer << (k ? " + " : "") << irreps[k] << "x" << irreps[k + 1]
              << (irreps[k + 2] ? "o" : "e");
    if (componentMajor)
      printer << ", ir_mul";
    printer << ">, " << elementType;
    return;
  }
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
                                unsigned numComponents, Type elementType,
                                ArrayRef<int64_t> irreps,
                                bool componentMajor) {
  if (!particleSet)
    return emitError() << "expected a particle set";
  if (numComponents == 0)
    return emitError() << "expected at least 1 component";

  bool isReal = elementType.isF64() || elementType.isF32();
  bool isInteger =
      elementType.isSignlessInteger(32) || elementType.isSignlessInteger(64);
  if (!isReal && !isInteger)
    return emitError() << "expected element type f64, f32, i32, or i64, got "
                       << elementType;
  if (irreps.size() % 3 != 0)
    return emitError() << "expected the irreps as triples of numbers";
  int64_t total = 0;
  for (size_t k = 0; k < irreps.size(); k += 3) {
    if (irreps[k] < 1)
      return emitError() << "expected a positive multiplicity of each "
                            "irreducible representation, got "
                         << irreps[k];
    if (irreps[k + 1] < 0 || irreps[k + 1] > 8)
      return emitError() << "expected a degree l from 0 to 8, got "
                         << irreps[k + 1];
    if (irreps[k + 2] != 0 && irreps[k + 2] != 1)
      return emitError() << "expected a parity of 0 or 1";
    total += irreps[k] * (2 * irreps[k + 1] + 1);
  }
  if (!irreps.empty() && total != numComponents)
    return emitError() << "expected " << total
                       << " components for the irreps, got " << numComponents;
  if (irreps.empty() && componentMajor)
    return emitError() << "a layout is stated for irreps only";
  if (numComponents != 1 && !isReal)
    return emitError() << "a field with several components must have element type "
                          "f64 or f32, got "
                       << elementType;
  return success();
}

FieldType FieldType::withElementType(Type element) const {
  return FieldType::get(getContext(), getParticleSet(), getNumComponents(),
                        element, getIrreps(), getComponentMajor());
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
