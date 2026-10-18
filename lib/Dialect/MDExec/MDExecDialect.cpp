// The md_exec dialect.

#include "mdir/Dialect/MDExec/MDExecDialect.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"

#include "mlir/IR/Builders.h"
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
