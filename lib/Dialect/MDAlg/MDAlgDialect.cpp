// The md_alg dialect and the verifiers of its ops (D[md-alg]). The
// operand types are checked by the constraints of MDAlgOps.td; the rules
// here are those of the execution part and of the buffers that an op
// must not share.

#include "mdir/Dialect/MDAlg/MDAlgDialect.h"
#include "mdir/Dialect/MDAlg/MDAlgOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mdir::md_alg;

#include "mdir/Dialect/MDAlg/MDAlgDialect.cpp.inc"
#include "mdir/Dialect/MDAlg/MDAlgEnums.cpp.inc"

#define GET_OP_CLASSES
#include "mdir/Dialect/MDAlg/MDAlgOps.cpp.inc"

void MDAlgDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mdir/Dialect/MDAlg/MDAlgOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Verifiers
//===----------------------------------------------------------------------===//

/// The launch values of `op`: none, or `expected`.
static LogicalResult verifyLaunch(Operation *op, OperandRange launch,
                                  unsigned expected, StringRef what) {
  if (!launch.empty() && launch.size() != expected)
    return op->emitOpError() << "takes " << expected << " launch values ("
                             << what << "), not " << launch.size();
  return success();
}

LogicalResult ScanOp::verify() {
  std::optional<ScanStrategy> strategy = getStrategy();
  bool scratch = static_cast<bool>(getScratch());
  bool launch = !getLaunch().empty();
  if ((strategy.has_value() != scratch) || (scratch != launch))
    return emitOpError()
           << "takes its strategy, its scratch, and its launch together";
  if (!strategy)
    return success();
  Type element = cast<MemRefType>(getScratch().getType()).getElementType();
  bool wide = *strategy == ScanStrategy::SinglePass;
  if (element.isInteger(64) != wide)
    return emitOpError() << "with the strategy '"
                         << stringifyScanStrategy(*strategy)
                         << "' takes a scratch of " << (wide ? "i64" : "i32")
                         << ", not " << element;
  if (*strategy == ScanStrategy::Chunked) {
    if (!getCursor())
      return emitOpError()
             << "with the strategy 'chunked' requires a cursor, which it "
                "writes";
    return verifyLaunch(*this, getLaunch(), 5,
                        "chunks, chunk, grid, block, one");
  }
  return verifyLaunch(*this, getLaunch(), 3, "blocks, threads, one");
}

LogicalResult ScatterToBinsOp::verify() {
  if (getOut() == getKeys() || getOut() == getCursor())
    return emitOpError()
           << "writes its items into its keys or its cursor, which it reads";
  return verifyLaunch(*this, getLaunch(), 3, "grid, block, one");
}

LogicalResult SortBinsOp::verify() {
  std::optional<SortStrategy> strategy = getStrategy();
  if (strategy.has_value() != !getLaunch().empty())
    return emitOpError() << "takes its strategy and its launch together";
  if (Value out = getOut())
    if (out == getItems() || out == getStart() || out == getBy())
      return emitOpError()
             << "sorts into its items, its offsets, or its order, which it "
                "reads";
  if (getKeys() && !getOut())
    return emitOpError() << "takes the keys of the items only with 'into'";
  if (strategy == SortStrategy::Insertion && getOut())
    return emitOpError() << "with the strategy 'insertion' sorts in place; "
                            "it takes no 'into'";
  if (strategy == SortStrategy::StableRank) {
    if (!getOut() || !getKeys())
      return emitOpError() << "with the strategy 'stable_rank' requires "
                              "'into' and the keys of the items";
    if (getBy())
      return emitOpError()
             << "with the strategy 'stable_rank' orders by the items; it "
                "takes no 'by'";
  }
  return verifyLaunch(*this, getLaunch(), 3, "grid, block, one");
}

LogicalResult WrapOp::verify() {
  if (getSignedCompare() && !getType().isInteger(32))
    return emitOpError() << "compares signed only on i32";
  return success();
}
