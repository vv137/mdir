// The md_exec dialect.

#include "mdir/Dialect/MDExec/MDExecDialect.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mdir/Dialect/MDExec/Algebra.h"
#include "mdir/Dialect/MDExec/StorageSet.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mdir::md_exec;

#include "mdir/Dialect/MDExec/MDExecDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mdir/Dialect/MDExec/MDExecAttrs.cpp.inc"

void MDExecDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mdir/Dialect/MDExec/MDExecOps.cpp.inc"
      >();
  registerAttributes();
}

void MDExecDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mdir/Dialect/MDExec/MDExecAttrs.cpp.inc"
      >();
}

AffineMap SetLayoutAttr::getAffineMap() const {
  return AffineMap::getMultiDimIdentityMap(getRank(), getContext());
}

LogicalResult SetLayoutAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, FlatSymbolRefAttr set,
    int64_t rank) {
  if (!set)
    return emitError() << "expected a particle or tuple set";
  if (rank < 1)
    return emitError() << "expected a rank of at least one, got " << rank;
  return success();
}

//===----------------------------------------------------------------------===//
// #md_exec.scatter (D285)
//===----------------------------------------------------------------------===//

ScatterAttr ScatterAttr::getTile(MLIRContext *context, int64_t edge) {
  return get(context, ScatterStrategy::Tile, edge, 0, 0, 0);
}

ScatterAttr ScatterAttr::getHash(MLIRContext *context, int64_t slots,
                                 int64_t probes, int64_t least) {
  return get(context, ScatterStrategy::Hash, 0, slots, probes, least);
}

void ScatterAttr::printBody(llvm::raw_ostream &os) const {
  os << stringifyScatterStrategy(getStrategy());
  if (isTile())
    os << "(" << getEdge() << ")";
  else if (isHash())
    os << "(" << getSlots() << ", " << getProbes() << ", " << getLeast()
       << ")";
}

ScatterAttr ScatterAttr::parseBody(AsmParser &parser) {
  SMLoc loc = parser.getCurrentLocation();
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return {};
  std::optional<ScatterStrategy> strategy = symbolizeScatterStrategy(keyword);
  if (!strategy) {
    parser.emitError(loc) << "expected 'atomic', 'tile', 'hash', or "
                             "'sorted', got '"
                          << keyword << "'";
    return {};
  }
  int64_t edge = 0, slots = 0, probes = 0, least = 0;
  if (*strategy == ScatterStrategy::Tile) {
    if (parser.parseLParen() || parser.parseInteger(edge) ||
        parser.parseRParen())
      return {};
  } else if (*strategy == ScatterStrategy::Hash) {
    if (parser.parseLParen() || parser.parseInteger(slots) ||
        parser.parseComma() || parser.parseInteger(probes) ||
        parser.parseComma() || parser.parseInteger(least) ||
        parser.parseRParen())
      return {};
  }
  return getChecked([&] { return parser.emitError(loc); },
                    parser.getContext(), *strategy, edge, slots, probes,
                    least);
}

Attribute ScatterAttr::parse(AsmParser &parser, Type) {
  if (parser.parseLess())
    return {};
  ScatterAttr attr = parseBody(parser);
  if (!attr || parser.parseGreater())
    return {};
  return attr;
}

void ScatterAttr::print(AsmPrinter &printer) const {
  printer << "<";
  printBody(printer.getStream());
  printer << ">";
}

LogicalResult
ScatterAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                    ScatterStrategy strategy, int64_t edge, int64_t slots,
                    int64_t probes, int64_t least) {
  switch (strategy) {
  case ScatterStrategy::Atomic:
    break;
  case ScatterStrategy::Tile:
    if (edge < 1)
      return emitError() << "expected a tile of at least 1 point an edge, got "
                         << edge;
    return success();
  case ScatterStrategy::Hash:
    if (slots < 1 || !llvm::isPowerOf2_64(slots))
      return emitError() << "expected a table of a power of two slots, got "
                         << slots;
    if (probes < 1)
      return emitError() << "expected at least 1 slot tried, got " << probes;
    if (least < 0)
      return emitError() << "expected a least number of items of 0 or more, "
                            "got "
                         << least;
    return success();
  case ScatterStrategy::Sorted:
    return emitError() << "the strategy 'sorted' is reserved for #417 and not "
                          "implemented";
  }
  if (edge || slots || probes || least)
    return emitError() << "expected no parameters of 'atomic'";
  return success();
}

//===----------------------------------------------------------------------===//
// #md_exec.algebra (D301)
//===----------------------------------------------------------------------===//

Attribute AlgebraListAttr::parse(AsmParser &parser, Type) {
  SmallVector<Algebra> values;
  auto parseOne = [&]() -> ParseResult {
    SMLoc loc = parser.getCurrentLocation();
    StringRef keyword;
    if (parser.parseKeyword(&keyword))
      return failure();
    std::optional<Algebra> algebra = symbolizeAlgebra(keyword);
    if (!algebra)
      return parser.emitError(loc)
             << "expected 'any_order_exact', 'any_order', or 'fixed_order', got '"
             << keyword << "'";
    values.push_back(*algebra);
    return success();
  };
  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::LessGreater,
                                     parseOne))
    return {};
  return get(parser.getContext(), values);
}

void AlgebraListAttr::print(AsmPrinter &printer) const {
  printer << "<";
  llvm::interleaveComma(getValues(), printer.getStream(), [&](Algebra value) {
    printer << stringifyAlgebra(value);
  });
  printer << ">";
}

LogicalResult MDExecDialect::verifyOperationAttribute(Operation *op,
                                                      NamedAttribute attr) {
  if (attr.getName() == kInputSetBoundaryAttrName)
    return verifyInputSetBoundaryAttribute(op, attr.getValue());
  if (attr.getName() == "md_exec.typed_storage") {
    if (!isa<UnitAttr>(attr.getValue()) || !isa<ModuleOp>(op))
      return op->emitError()
             << "expects 'md_exec.typed_storage' on a module";
    return verifyStorageSetCasts(cast<ModuleOp>(op));
  }
  if (attr.getName() == algebra::kAlgebraAttrName)
    return algebra::verify(op, attr.getValue());
  if (attr.getName() == algebra::kDeterministicAttrName) {
    if (!isa<UnitAttr>(attr.getValue()) || !isa<ModuleOp>(op))
      return op->emitError() << "expects '" << algebra::kDeterministicAttrName
                             << "' to be a unit attribute of a module";
    return algebra::verifyDeterministic(op);
  }
  return success();
}
