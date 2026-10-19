// Ops of the md dialect.

#include "mdir/Dialect/MD/MDOps.h"

#include "mdir/Dialect/MD/MDDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/FunctionImplementation.h"

#include <cmath>

using namespace mlir;
using namespace mdir::md;

//===----------------------------------------------------------------------===//
// Custom directive: exchange(<kind>[, <basis>])
//===----------------------------------------------------------------------===//

static ParseResult parseExchange(OpAsmParser &parser, ExchangeAttr &exchange,
                                 ExchangeBasisAttr &basis) {
  MLIRContext *context = parser.getContext();
  StringRef keyword;
  SMLoc loc;
  if (parser.parseKeyword("exchange") || parser.parseLParen() ||
      parser.getCurrentLocation(&loc) || parser.parseKeyword(&keyword))
    return failure();
  std::optional<Exchange> kind = symbolizeExchange(keyword);
  if (!kind)
    return parser.emitError(loc)
           << "expected 'none', 'symmetric', or 'antisymmetric', got '"
           << keyword << "'";
  exchange = ExchangeAttr::get(context, *kind);

  if (succeeded(parser.parseOptionalComma())) {
    if (parser.getCurrentLocation(&loc) || parser.parseKeyword(&keyword))
      return failure();
    std::optional<ExchangeBasis> parsed = symbolizeExchangeBasis(keyword);
    if (!parsed)
      return parser.emitError(loc)
             << "expected 'proof', 'asserted', or 'derived', got '" << keyword
             << "'";
    basis = ExchangeBasisAttr::get(context, *parsed);
  }
  return parser.parseRParen();
}

static void printExchange(OpAsmPrinter &printer, Operation *,
                          ExchangeAttr exchange, ExchangeBasisAttr basis) {
  printer << "exchange(" << stringifyExchange(exchange.getValue());
  if (basis && basis.getValue() != ExchangeBasis::Proof)
    printer << ", " << stringifyExchangeBasis(basis.getValue());
  printer << ")";
}

//===----------------------------------------------------------------------===//
// Custom directive: truncation(<kind>[, from = <value>])
//===----------------------------------------------------------------------===//

static ParseResult parseTruncation(OpAsmParser &parser,
                                   TruncationAttr &truncation,
                                   FloatAttr &switchFrom) {
  if (failed(parser.parseOptionalKeyword("truncation")))
    return success();

  StringRef keyword;
  SMLoc loc;
  if (parser.parseLParen() || parser.getCurrentLocation(&loc) ||
      parser.parseKeyword(&keyword))
    return failure();
  std::optional<Truncation> kind = symbolizeTruncation(keyword);
  if (!kind)
    return parser.emitError(loc)
           << "expected 'none', 'shift', 'force_shift', 'switch', or "
              "'force_switch', got '"
           << keyword << "'";
  truncation = TruncationAttr::get(parser.getContext(), *kind);

  if (succeeded(parser.parseOptionalComma())) {
    double value;
    if (parser.parseKeyword("from") || parser.parseEqual() ||
        parser.parseFloat(value))
      return failure();
    switchFrom = parser.getBuilder().getF64FloatAttr(value);
  }
  return parser.parseRParen();
}

static void printTruncation(OpAsmPrinter &printer, Operation *,
                            TruncationAttr truncation, FloatAttr switchFrom) {
  if (!truncation || truncation.getValue() == Truncation::None)
    return;
  printer << "truncation(" << stringifyTruncation(truncation.getValue());
  if (switchFrom)
    printer << ", from = " << switchFrom.getValueAsDouble();
  printer << ")";
}

//===----------------------------------------------------------------------===//
// Custom directive: coordinates(<kind>(<members>), ...)
//===----------------------------------------------------------------------===//

static ParseResult parseCoordinates(OpAsmParser &parser,
                                    DenseI32ArrayAttr &kinds,
                                    DenseI64ArrayAttr &members) {
  return parseCoordinateList(parser, kinds, members);
}

static void printCoordinates(OpAsmPrinter &printer, Operation *,
                             DenseI32ArrayAttr kinds,
                             DenseI64ArrayAttr members) {
  printCoordinateList(printer, kinds, members);
}

//===----------------------------------------------------------------------===//
// Custom directive: [energy, forces, virial, derivative(<n>)]
//===----------------------------------------------------------------------===//

static ParseResult parseRequests(OpAsmParser &parser,
                                 DenseI32ArrayAttr &kinds,
                                 DenseI64ArrayAttr &arguments) {
  SmallVector<int32_t> parsedKinds;
  SmallVector<int64_t> parsedArguments;

  auto parseRequest = [&]() -> ParseResult {
    StringRef keyword;
    SMLoc loc;
    if (parser.getCurrentLocation(&loc) || parser.parseKeyword(&keyword))
      return failure();
    std::optional<Request> kind = symbolizeRequest(keyword);
    if (!kind)
      return parser.emitError(loc)
             << "expected 'energy', 'forces', 'virial', or 'derivative', "
                "got '"
             << keyword << "'";

    int64_t argument = -1;
    if (*kind == Request::Derivative) {
      if (parser.parseLParen() || parser.parseInteger(argument) ||
          parser.parseRParen())
        return failure();
    }
    parsedKinds.push_back(static_cast<int32_t>(*kind));
    parsedArguments.push_back(argument);
    return success();
  };

  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Square,
                                     parseRequest))
    return failure();
  kinds = parser.getBuilder().getDenseI32ArrayAttr(parsedKinds);
  arguments = parser.getBuilder().getDenseI64ArrayAttr(parsedArguments);
  return success();
}

static void printRequests(OpAsmPrinter &printer, Operation *,
                          DenseI32ArrayAttr kinds,
                          DenseI64ArrayAttr arguments) {
  printer << "[";
  for (unsigned i = 0, e = kinds.size(); i != e; ++i) {
    if (i != 0)
      printer << ", ";
    auto kind = static_cast<Request>(kinds[i]);
    printer << stringifyRequest(kind);
    if (kind == Request::Derivative)
      printer << "(" << arguments[i] << ")";
  }
  printer << "]";
}

#define GET_OP_CLASSES
#include "mdir/Dialect/MD/MDOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Function-like ops
//===----------------------------------------------------------------------===//

template <typename OpTy>
static ParseResult parseFunctionLike(OpAsmParser &parser,
                                     OperationState &result) {
  auto buildFunctionType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      OpTy::getFunctionTypeAttrName(result.name), buildFunctionType,
      OpTy::getArgAttrsAttrName(result.name),
      OpTy::getResAttrsAttrName(result.name));
}

template <typename OpTy>
static void printFunctionLike(OpAsmPrinter &printer, OpTy op) {
  function_interface_impl::printFunctionOp(
      printer, op, /*isVariadic=*/false, op.getFunctionTypeAttrName(),
      op.getArgAttrsAttrName(), op.getResAttrsAttrName());
}

ParseResult PotentialOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseFunctionLike<PotentialOp>(parser, result);
}

void PotentialOp::print(OpAsmPrinter &printer) {
  printFunctionLike(printer, *this);
}

ParseResult FunctionOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseFunctionLike<FunctionOp>(parser, result);
}

void FunctionOp::print(OpAsmPrinter &printer) {
  printFunctionLike(printer, *this);
}

/// Returns true if `type` is a position field: three components of f64.
static bool isPositionField(Type type) {
  auto field = dyn_cast<FieldType>(type);
  return field && field.getNumComponents() == 3 &&
         field.getElementType().isF64();
}

LogicalResult FunctionOp::verify() {
  return verifyReferencePrecision(getOperation(), getFunctionType());
}

LogicalResult PotentialOp::verify() {
  ArrayRef<Type> arguments = getArgumentTypes();
  ArrayRef<Type> results = getResultTypes();
  if (failed(verifyReferencePrecision(getOperation(), getFunctionType())))
    return failure();

  if (arguments.size() < 2)
    return emitOpError()
           << "expected at least 2 arguments, a position field and a cell";
  if (!isPositionField(arguments[0]))
    return emitOpError() << "expected argument 0 to be a position field with "
                            "3 components of f64, got "
                         << arguments[0];
  if (!isa<CellType>(arguments[1]))
    return emitOpError() << "expected argument 1 to be a cell, got "
                         << arguments[1];
  if (results.size() != 1 || !results[0].isF64())
    return emitOpError() << "expected exactly one result of type f64";
  return success();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  auto function = cast<FunctionOpInterface>((*this)->getParentOp());
  ArrayRef<Type> results = function.getResultTypes();

  if (getNumOperands() != results.size())
    return emitOpError() << "returns " << getNumOperands()
                         << " values, but the enclosing function has "
                         << results.size() << " results";
  for (unsigned i = 0, e = results.size(); i != e; ++i)
    if (getOperand(i).getType() != results[i])
      return emitOpError() << "type of return value " << i << " ("
                           << getOperand(i).getType()
                           << ") does not match the result type ("
                           << results[i] << ")";
  return success();
}

//===----------------------------------------------------------------------===//
// NeighborhoodOp
//===----------------------------------------------------------------------===//

LogicalResult NeighborhoodOp::verify() {
  if (!isPositionField(getPositions().getType()))
    return emitOpError() << "expected a position field with 3 components of "
                            "f64, got "
                         << getPositions().getType();

  double cutoff = getCutoff().convertToDouble();
  if (!(cutoff > 0.0))
    return emitOpError() << "expected a positive cutoff, got " << cutoff;

  auto positions = cast<FieldType>(getPositions().getType());
  auto relation = cast<RelationType>(getResult().getType());
  if (relation.getParticleSet() != positions.getParticleSet())
    return emitOpError() << "result is a relation on "
                         << relation.getParticleSet()
                         << ", but the positions belong to "
                         << positions.getParticleSet();
  if (relation.getArity() != 2 ||
      relation.getOrientation() != Orientation::Unordered)
    return emitOpError()
           << "expected the result to be an unordered relation of arity 2";

  if (Value excluded = getExcluded()) {
    auto pairs = cast<RelationType>(excluded.getType());
    if (!pairs.getTupleSet() || pairs.getArity() != 2 ||
        pairs.getOrientation() != Orientation::Unordered)
      return emitOpError() << "expected the excluded pairs to be the "
                              "unordered relation of arity 2 of a tuple set, "
                              "got "
                           << pairs;
    if (pairs.getParticleSet() != positions.getParticleSet())
      return emitOpError() << "the excluded pairs are on "
                           << pairs.getParticleSet()
                           << ", but the positions belong to "
                           << positions.getParticleSet();
  }
  return success();
}

LogicalResult TripletsOp::verify() {
  double cutoff = getCutoff().convertToDouble();
  if (!(cutoff > 0.0))
    return emitOpError() << "expected a positive cutoff, got " << cutoff;
  auto pairs = cast<RelationType>(getNeighbors().getType());
  if (pairs.getTupleSet() || pairs.getArity() != 2 ||
      pairs.getOrientation() != Orientation::Unordered)
    return emitOpError() << "expected the unordered relation of arity 2 of "
                            "a neighborhood, got "
                         << pairs;
  auto relation = cast<RelationType>(getResult().getType());
  if (relation.getTupleSet() || relation.getArity() != 3 ||
      relation.getOrientation() != Orientation::Reversal ||
      relation.getParticleSet() != pairs.getParticleSet())
    return emitOpError()
           << "expected the result to have type "
           << RelationType::get(getContext(), pairs.getParticleSet(), 3,
                                Orientation::Reversal, FlatSymbolRefAttr())
           << ", got " << relation;
  if (auto neighborhood = getNeighbors().getDefiningOp<NeighborhoodOp>()) {
    double reach = neighborhood.getCutoff().convertToDouble();
    if (cutoff > reach)
      return emitOpError() << "expected a cutoff of at most " << reach
                           << ", that of the neighborhood, got " << cutoff;
  }
  return success();
}

LogicalResult EdgesOp::verify() {
  double cutoff = getCutoff().convertToDouble();
  if (!(cutoff > 0.0))
    return emitOpError() << "expected a positive cutoff, got " << cutoff;
  auto pairs = cast<RelationType>(getNeighbors().getType());
  if (pairs.getTupleSet() || pairs.getArity() != 2 ||
      pairs.getOrientation() != Orientation::Unordered)
    return emitOpError() << "expected the unordered relation of arity 2 of "
                            "a neighborhood, got "
                         << pairs;
  if (!isPositionField(getPositions().getType()))
    return emitOpError() << "expected a position field with 3 components of "
                            "f64, got "
                         << getPositions().getType();
  auto positions = cast<FieldType>(getPositions().getType());
  auto edges = cast<EdgesType>(getResult().getType());
  if (positions.getParticleSet() != pairs.getParticleSet() ||
      edges.getParticleSet() != pairs.getParticleSet())
    return emitOpError() << "expected the positions and the result on "
                         << pairs.getParticleSet() << ", the set of the pairs";
  // The candidates are the neighbors of the neighborhood, at its positions
  // and in its cell, within its cutoff.
  if (auto neighborhood = getNeighbors().getDefiningOp<NeighborhoodOp>()) {
    double reach = neighborhood.getCutoff().convertToDouble();
    if (cutoff > reach)
      return emitOpError() << "expected a cutoff of at most " << reach
                           << ", that of the neighborhood, got " << cutoff;
    if (neighborhood.getPositions() != getPositions() ||
        neighborhood.getCell() != getCell())
      return emitOpError() << "expected the positions and the cell of the "
                              "neighborhood";
    if (neighborhood.getExcluded())
      return emitOpError() << "expected a neighborhood without excluded "
                              "pairs: the edges take every pair";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Ops with a pair kernel
//===----------------------------------------------------------------------===//

/// Verifies what `md.sum_relation` and `md.gather_relation` have in common:
/// the operands.
template <typename OpTy>
static LogicalResult verifyPairOperands(OpTy op) {
  auto relation = cast<RelationType>(op.getRelation().getType());
  if (relation.getArity() != 2)
    return op.emitOpError() << "expected a relation of arity 2, got arity "
                            << relation.getArity();

  if (!isPositionField(op.getPositions().getType()))
    return op.emitOpError() << "expected a position field with 3 components "
                               "of f64, got "
                            << op.getPositions().getType();
  auto positions = cast<FieldType>(op.getPositions().getType());
  if (positions.getParticleSet() != relation.getParticleSet())
    return op.emitOpError()
           << "the relation is on " << relation.getParticleSet()
           << ", but the positions belong to " << positions.getParticleSet();

  for (Value gathered : op.getGathered()) {
    auto field = cast<FieldType>(gathered.getType());
    if (field.getParticleSet() != relation.getParticleSet())
      return op.emitOpError()
             << "the relation is on " << relation.getParticleSet()
             << ", but a gathered field belongs to "
             << field.getParticleSet();
  }

  // A neighborhood is a function of the positions and the cell it was built
  // from. The kernel geometry must be evaluated at the same configuration.
  if (auto neighborhood =
          op.getRelation().template getDefiningOp<NeighborhoodOp>()) {
    if (neighborhood.getPositions() != op.getPositions())
      return op.emitOpError() << "expected the positions that the "
                                 "neighborhood was built from";
    if (neighborhood.getCell() != op.getCell())
      return op.emitOpError()
             << "expected the cell that the neighborhood was built from";
  }
  return success();
}

/// Verifies the kernel: its arguments and its terminator. `yieldTypes` are
/// the types the kernel must yield.
template <typename OpTy>
static LogicalResult verifyPairKernel(OpTy op, TypeRange yieldTypes) {
  Block &block = op.getKernel().front();
  Builder builder(op.getContext());

  SmallVector<Type> expected;
  expected.push_back(builder.getF64Type());
  expected.push_back(VectorType::get({3}, builder.getF64Type()));
  for (Value gathered : op.getGathered()) {
    Type value = cast<FieldType>(gathered.getType()).getKernelValueType();
    expected.push_back(value);
    expected.push_back(value);
  }

  if (block.getNumArguments() != expected.size())
    return op.emitOpError()
           << "expected the kernel to have " << expected.size()
           << " arguments (distance, displacement, and two per gathered "
              "field), got "
           << block.getNumArguments();
  for (unsigned i = 0, e = expected.size(); i != e; ++i)
    if (block.getArgument(i).getType() != expected[i])
      return op.emitOpError()
             << "expected kernel argument " << i << " to have type "
             << expected[i] << ", got " << block.getArgument(i).getType();

  auto yield = dyn_cast<YieldOp>(block.getTerminator());
  if (!yield)
    return op.emitOpError() << "expected the kernel to end with 'md.yield'";
  if (yield.getNumOperands() != yieldTypes.size())
    return yield.emitOpError()
           << "expected exactly " << yieldTypes.size() << " value"
           << (yieldTypes.size() == 1 ? "" : "s") << ", got "
           << yield.getNumOperands();
  for (auto [value, type] : llvm::zip_equal(yield.getOperands(), yieldTypes))
    if (value.getType() != type)
      return yield.emitOpError()
             << "expected a value of type " << type << ", got "
             << value.getType();
  return success();
}

/// Returns true if `type` is f64 or a fixed-size one-dimensional vector of
/// f64.
static bool isRealOrRealVector(Type type) {
  if (type.isF64())
    return true;
  auto vector = dyn_cast<VectorType>(type);
  return vector && !vector.isScalable() && vector.getRank() == 1 &&
         vector.getElementType().isF64();
}

//===----------------------------------------------------------------------===//
// SumRelationOp
//===----------------------------------------------------------------------===//

LogicalResult SumRelationOp::verify() {
  if (failed(verifyPairOperands(*this)))
    return failure();

  if (!isRealOrRealVector(getResult().getType()))
    return emitOpError()
           << "expected the result to be f64 or a fixed-size vector of f64, "
              "got "
           << getResult().getType();

  auto relation = cast<RelationType>(getRelation().getType());
  if (relation.getOrientation() == Orientation::Unordered &&
      getExchange() != Exchange::Symmetric)
    return emitOpError() << "a sum over an unordered relation requires "
                            "'exchange(symmetric)'";

  bool isSwitch = getTruncation() == Truncation::Switch ||
                  getTruncation() == Truncation::ForceSwitch;
  if (isSwitch != getSwitchFrom().has_value())
    return emitOpError() << (isSwitch
                                 ? "a switching truncation requires 'from'"
                                 : "'from' is allowed only with "
                                   "'truncation(switch)' and "
                                   "'truncation(force_switch)'");
  if (isSwitch) {
    double from = getSwitchFrom()->convertToDouble();
    if (!(from > 0.0))
      return emitOpError()
             << "expected a positive switching distance, got " << from;
    if (auto neighborhood = getRelation().getDefiningOp<NeighborhoodOp>()) {
      double cutoff = neighborhood.getCutoff().convertToDouble();
      if (!(from < cutoff))
        return emitOpError()
               << "expected the switching distance (" << from
               << ") to be smaller than the cutoff (" << cutoff << ")";
    }
  }
  return success();
}

LogicalResult SumRelationOp::verifyRegions() {
  return verifyPairKernel(*this, TypeRange(getResult().getType()));
}

//===----------------------------------------------------------------------===//
// GatherRelationOp
//===----------------------------------------------------------------------===//

LogicalResult GatherRelationOp::verify() {
  if (failed(verifyPairOperands(*this)))
    return failure();

  auto relation = cast<RelationType>(getRelation().getType());
  if (getResults().empty())
    return emitOpError() << "expected at least one result";
  for (Type type : getResultTypes()) {
    auto result = cast<FieldType>(type);
    if (result.getParticleSet() != relation.getParticleSet())
      return emitOpError() << "the relation is on "
                           << relation.getParticleSet()
                           << ", but the result belongs to "
                           << result.getParticleSet();
  }
  if (getNumResults() > 1 && getExchange() != Exchange::None)
    return emitOpError() << "a gather of several fields has no exchange "
                            "contract: expected 'exchange(none)'";
  return success();
}

LogicalResult GatherRelationOp::verifyRegions() {
  SmallVector<Type> values;
  for (Type type : getResultTypes())
    values.push_back(cast<FieldType>(type).getKernelValueType());
  return verifyPairKernel(*this, TypeRange(values));
}

//===----------------------------------------------------------------------===//
// Tables
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ReciprocalOp
//===----------------------------------------------------------------------===//

LogicalResult ReciprocalOp::verify() {
  auto positions = cast<FieldType>(getPositions().getType());
  auto charges = cast<FieldType>(getCharges().getType());
  if (positions.getParticleSet() != charges.getParticleSet())
    return emitOpError() << "the positions and the charges belong to "
                            "different particle sets";
  if (!isa<VectorType>(positions.getKernelValueType()))
    return emitOpError() << "expected positions of three numbers";
  if (isa<VectorType>(charges.getKernelValueType()))
    return emitOpError() << "expected one charge for each particle";
  if (getForces().getType() != getPositions().getType())
    return emitOpError() << "expected the forces to be of the type of the "
                            "positions";
  if (getPotential() && getPotential().getType() != getCharges().getType())
    return emitOpError() << "expected the potential at the particles to be "
                            "of the type of the charges";
  if (Value listed = getListed()) {
    auto list = cast<FieldType>(listed.getType());
    if (list.getParticleSet() != charges.getParticleSet() ||
        !list.getElementType().isInteger(32) || list.getNumComponents() != 1)
      return emitOpError() << "expected the listed particles as a field of "
                              "one i32 for each particle of the charges";
  } else if (getGatherAll()) {
    return emitOpError() << "expected a list of the particles with "
                            "'gather_all'";
  }
  if (getGrid().size() != 3 ||
      llvm::any_of(getGrid(), [](int64_t points) { return points < 2; }))
    return emitOpError() << "expected three numbers of points of at least 2";
  if (getOrder() < 3 || getOrder() > 8)
    return emitOpError() << "expected an order from 3 to 8";
  if (!getProlate() && !(getBeta().convertToDouble() > 0.0))
    return emitOpError() << "expected a positive beta";
  if (getProlate() && getDispersion())
    return emitOpError() << "the prolate kernel is for the Coulomb sum alone";
  if (getProlate() && (getOrder() < 5 || getOrder() > 8))
    return emitOpError() << "expected 5 to 8 points of the prolate window";
  if (cast<TableType>(getModuli().getType()).getRank() != 2)
    return emitOpError() << "expected a table of rank 2";
  return success();
}

LogicalResult LookupOp::verify() {
  Type type = getTable().getType();
  unsigned rank;
  if (auto table = dyn_cast<TableType>(type)) {
    rank = table.getRank();
  } else if (auto buffer = dyn_cast<MemRefType>(type)) {
    if (!isa<FloatType>(getElementTypeOrSelf(buffer.getElementType())))
      return emitOpError() << "expected a buffer of floating-point numbers, "
                              "got "
                           << type;
    rank = buffer.getRank();
  } else {
    return emitOpError() << "expected a table or the buffer that holds one, "
                            "got "
                         << type;
  }
  if (getIndices().size() != rank)
    return emitOpError() << "expected " << rank << " indices, one for each "
                         << "dimension of the table, got "
                         << getIndices().size();
  return success();
}

double mdir::md::evaluateChebyshev(llvm::ArrayRef<double> c, double y) {
  // b_k = c_k + 2y b_{k+1} − b_{k+2}; the sum is c_0 + y b_1 − b_2.
  double next = 0.0, after = 0.0;
  for (size_t k = c.size(); k-- > 1;) {
    double b = c[k] + 2.0 * y * next - after;
    after = next;
    next = b;
  }
  return c.empty() ? 0.0 : c[0] + y * next - after;
}

llvm::SmallVector<double>
mdir::md::getChebyshevDerivative(llvm::ArrayRef<double> c) {
  // T_k' in the T_j: d_{k−1} = d_{k+1} + 2k c_k from the top, d_0 halved.
  size_t n = c.size();
  if (n < 2)
    return {};
  llvm::SmallVector<double> d(n + 1, 0.0);
  for (size_t k = n - 1; k >= 1; --k)
    d[k - 1] = d[k + 1] + 2.0 * static_cast<double>(k) * c[k];
  d[0] *= 0.5;
  d.resize(n - 1);
  return d;
}

Value mdir::md::emitChebyshev(OpBuilder &builder, Location loc, Value y,
                              llvm::ArrayRef<double> c) {
  Type type = y.getType();
  auto constant = [&](double v) -> Value {
    return arith::ConstantOp::create(builder, loc, type,
                                     builder.getFloatAttr(type, v));
  };
  if (c.size() == 1)
    return constant(c[0]);
  Value twice = arith::AddFOp::create(builder, loc, y, y);
  Value next, after;
  for (size_t k = c.size(); k-- > 1;) {
    Value b = constant(c[k]);
    if (next)
      b = arith::AddFOp::create(builder, loc, b,
                                arith::MulFOp::create(builder, loc, twice,
                                                      next));
    if (after)
      b = arith::SubFOp::create(builder, loc, b, after);
    after = next;
    next = b;
  }
  Value sum = arith::AddFOp::create(builder, loc, constant(c[0]),
                                    arith::MulFOp::create(builder, loc, y,
                                                          next));
  if (after)
    sum = arith::SubFOp::create(builder, loc, sum, after);
  return sum;
}

llvm::SmallVector<double> ChebyshevOp::getSeries() {
  llvm::SmallVector<double> series;
  for (Attribute coefficient : getCoefficients())
    series.push_back(cast<FloatAttr>(coefficient).getValueAsDouble());
  return series;
}

LogicalResult ChebyshevOp::verify() {
  if (getCoefficients().empty())
    return emitOpError() << "expected at least one coefficient";
  if (!llvm::all_of(getSeries(), [](double c) { return std::isfinite(c); }))
    return emitOpError() << "expected finite coefficients";
  return success();
}

//===----------------------------------------------------------------------===//
// Tuples of a topology
//===----------------------------------------------------------------------===//

LogicalResult TupleSetOp::verify() {
  int64_t arity = getArity();
  if (arity < 1)
    return emitOpError() << "expected an arity of at least 1, got " << arity;
  if (getOrientation() == Orientation::Unordered && arity != 2)
    return emitOpError() << "expected the orientation 'unordered' with the "
                            "arity 2 only, got the arity "
                         << arity;
  if (getOrientation() == Orientation::Reversal && arity < 2)
    return emitOpError() << "expected the orientation 'reversal' with an "
                            "arity of at least 2";
  return success();
}

//===----------------------------------------------------------------------===//
// DisjointUnionOp
//===----------------------------------------------------------------------===//

LogicalResult
DisjointUnionOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  for (Attribute attr : getTupleSets()) {
    auto name = cast<FlatSymbolRefAttr>(attr);
    auto set = symbolTable.lookupNearestSymbolFrom<TupleSetOp>(*this, name);
    if (!set)
      return emitOpError() << "'" << name.getValue()
                           << "' does not name a tuple set";
    if (set.getParticleSetAttr() != getParticleSetAttr())
      return emitOpError() << "expected the tuple set '" << name.getValue()
                           << "' on '" << getParticleSet() << "', got '"
                           << set.getParticleSet() << "'";
    if (!set.getDisjoint())
      return emitOpError() << "expected the tuple set '" << name.getValue()
                           << "' to be 'disjoint'";
  }
  return success();
}

/// Verifies what `md.sum_tuples` and `md.gather_tuples` have in common:
/// the operands and the coordinates.
template <typename OpTy>
static LogicalResult verifyTupleOperands(OpTy op) {
  // A relation without a tuple set is accepted from 'md.triplets' only,
  // and then the tuples have no fields (D160).
  auto relation = cast<RelationType>(op.getRelation().getType());
  FlatSymbolRefAttr tupleSet = relation.getTupleSet();
  if (!tupleSet && !op.getRelation().template getDefiningOp<TripletsOp>())
    return op.emitOpError()
           << "expected the relation of a tuple set or of 'md.triplets', got "
           << relation;
  if (!tupleSet && !op.getParameters().empty())
    return op.emitOpError()
           << "takes no fields in 'tuple': the triplets have no tuple set";

  if (!isPositionField(op.getPositions().getType()))
    return op.emitOpError() << "expected a position field with 3 components "
                               "of f64, got "
                            << op.getPositions().getType();
  auto positions = cast<FieldType>(op.getPositions().getType());
  if (positions.getParticleSet() != relation.getParticleSet())
    return op.emitOpError()
           << "the relation is on " << relation.getParticleSet()
           << ", but the positions belong to " << positions.getParticleSet();

  for (Value gathered : op.getGathered()) {
    auto field = cast<FieldType>(gathered.getType());
    if (field.getParticleSet() != relation.getParticleSet())
      return op.emitOpError()
             << "the relation is on " << relation.getParticleSet()
             << ", but a gathered field belongs to "
             << field.getParticleSet();
  }
  for (Value parameter : op.getParameters()) {
    auto field = cast<FieldType>(parameter.getType());
    if (field.getParticleSet() != tupleSet)
      return op.emitOpError()
             << "the tuples are those of " << tupleSet
             << ", but a field in 'tuple' belongs to "
             << field.getParticleSet();
  }
  return verifyCoordinates(op.getOperation(), op.getCoordinateKinds(),
                           op.getCoordinateMembers(), relation.getArity());
}

/// Verifies the kernel: its arguments and its terminator. The kernel must
/// yield `numYields` values of the type `yieldType`, then, with
/// `extraFlag`, one i1.
template <typename OpTy>
static LogicalResult verifyTupleKernel(OpTy op, unsigned numYields,
                                       Type yieldType,
                                       bool extraFlag = false) {
  Block &block = op.getKernel().front();
  Builder builder(op.getContext());

  SmallVector<Type> expected;
  for (const Coordinate &coordinate : op.getCoordinates())
    expected.push_back(
        getCoordinateType(coordinate.kind, builder.getF64Type()));
  for (Value gathered : op.getGathered())
    expected.append(op.getArity(),
                    cast<FieldType>(gathered.getType()).getKernelValueType());
  for (Value parameter : op.getParameters())
    expected.push_back(
        cast<FieldType>(parameter.getType()).getKernelValueType());

  if (block.getNumArguments() != expected.size())
    return op.emitOpError()
           << "expected the kernel to have " << expected.size()
           << " arguments (one per coordinate, one per member for each "
              "gathered field, and one per field of the tuples), got "
           << block.getNumArguments();
  for (unsigned i = 0, e = expected.size(); i != e; ++i)
    if (block.getArgument(i).getType() != expected[i])
      return op.emitOpError()
             << "expected kernel argument " << i << " to have type "
             << expected[i] << ", got " << block.getArgument(i).getType();

  auto yield = dyn_cast<YieldOp>(block.getTerminator());
  if (!yield)
    return op.emitOpError() << "expected the kernel to end with 'md.yield'";
  if (yield.getNumOperands() != numYields + (extraFlag ? 1 : 0))
    return yield.emitOpError()
           << "expected " << numYields << " values"
           << (extraFlag ? " and an i1" : "") << ", got "
           << yield.getNumOperands();
  for (Value value : yield.getOperands().take_front(numYields))
    if (value.getType() != yieldType)
      return yield.emitOpError() << "expected values of type " << yieldType
                                 << ", got " << value.getType();
  if (extraFlag && !yield.getOperand(numYields).getType().isInteger(1))
    return yield.emitOpError() << "expected the last value to be an i1, got "
                               << yield.getOperand(numYields).getType();
  return success();
}

LogicalResult SumTuplesOp::verify() {
  if (failed(verifyTupleOperands(*this)))
    return failure();
  // An i1 tells whether the kernel yields true for some tuple
  // (D278).
  Type result = getResult().getType();
  if (!isRealOrRealVector(result) && !result.isInteger(1))
    return emitOpError()
           << "expected the result to be f64, a fixed-size vector of f64, "
              "or i1, got "
           << result;
  return success();
}

LogicalResult SumTuplesOp::verifyRegions() {
  return verifyTupleKernel(*this, 1, getResult().getType());
}

LogicalResult GatherTuplesOp::verify() {
  if (failed(verifyTupleOperands(*this)))
    return failure();
  auto relation = cast<RelationType>(getRelation().getType());
  auto result = cast<FieldType>(getResult().getType());
  if (result.getParticleSet() != relation.getParticleSet())
    return emitOpError() << "the relation is on " << relation.getParticleSet()
                         << ", but the result belongs to "
                         << result.getParticleSet();
  return success();
}

void GatherTuplesOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {}

LogicalResult GatherTuplesOp::verifyRegions() {
  auto result = cast<FieldType>(getResult().getType());
  return verifyTupleKernel(*this, getArity(), result.getKernelValueType(),
                           /*extraFlag=*/bool(getFailed()));
}

//===----------------------------------------------------------------------===//
// Ops with a particle kernel
//===----------------------------------------------------------------------===//

/// Verifies that the gathered fields exist and belong to one particle set.
/// Returns that particle set in `particleSet`.
template <typename OpTy>
static LogicalResult verifyParticleOperands(OpTy op,
                                            FlatSymbolRefAttr &particleSet) {
  if (op.getGathered().empty())
    return op.emitOpError() << "expected at least 1 gathered field";

  particleSet =
      cast<FieldType>(op.getGathered().front().getType()).getParticleSet();
  for (Value gathered : op.getGathered()) {
    auto field = cast<FieldType>(gathered.getType());
    if (field.getParticleSet() != particleSet)
      return op.emitOpError()
             << "expected all gathered fields to belong to " << particleSet
             << ", but one belongs to " << field.getParticleSet();
  }
  return success();
}

/// Verifies the kernel: its arguments and its terminator. `yieldType` is the
/// type the kernel must yield.
template <typename OpTy>
static LogicalResult verifyParticleKernel(OpTy op, Type yieldType) {
  Block &block = op.getKernel().front();

  SmallVector<Type> expected;
  for (Value gathered : op.getGathered())
    expected.push_back(
        cast<FieldType>(gathered.getType()).getKernelValueType());

  if (block.getNumArguments() != expected.size())
    return op.emitOpError()
           << "expected the kernel to have " << expected.size()
           << " arguments (one per gathered field), got "
           << block.getNumArguments();
  for (unsigned i = 0, e = expected.size(); i != e; ++i)
    if (block.getArgument(i).getType() != expected[i])
      return op.emitOpError()
             << "expected kernel argument " << i << " to have type "
             << expected[i] << ", got " << block.getArgument(i).getType();

  auto yield = dyn_cast<YieldOp>(block.getTerminator());
  if (!yield)
    return op.emitOpError() << "expected the kernel to end with 'md.yield'";
  if (yield.getNumOperands() != 1)
    return yield.emitOpError()
           << "expected exactly 1 value, got " << yield.getNumOperands();
  if (yield.getOperand(0).getType() != yieldType)
    return yield.emitOpError()
           << "expected a value of type " << yieldType << ", got "
           << yield.getOperand(0).getType();
  return success();
}

//===----------------------------------------------------------------------===//
// SumParticlesOp
//===----------------------------------------------------------------------===//

LogicalResult SumParticlesOp::verify() {
  FlatSymbolRefAttr particleSet;
  if (failed(verifyParticleOperands(*this, particleSet)))
    return failure();
  if (!isRealOrRealVector(getResult().getType()))
    return emitOpError()
           << "expected the result to be f64 or a fixed-size vector of f64, "
              "got "
           << getResult().getType();
  return success();
}

LogicalResult SumParticlesOp::verifyRegions() {
  return verifyParticleKernel(*this, getResult().getType());
}

//===----------------------------------------------------------------------===//
// MapParticlesOp
//===----------------------------------------------------------------------===//

LogicalResult MapParticlesOp::verify() {
  FlatSymbolRefAttr particleSet;
  if (failed(verifyParticleOperands(*this, particleSet)))
    return failure();
  auto result = cast<FieldType>(getResult().getType());
  if (result.getParticleSet() != particleSet)
    return emitOpError() << "the gathered fields belong to " << particleSet
                         << ", but the result belongs to "
                         << result.getParticleSet();
  return success();
}

LogicalResult MapParticlesOp::verifyRegions() {
  auto result = cast<FieldType>(getResult().getType());
  return verifyParticleKernel(*this, result.getKernelValueType());
}

//===----------------------------------------------------------------------===//
// Neural stages
//===----------------------------------------------------------------------===//

/// Whether `type` is the value of one element of a floating field: f64 or a
/// vector of it. The stages are declared at the reference precision.
static bool isStageReal(Type type) {
  if (auto vector = dyn_cast<VectorType>(type))
    return vector.getRank() == 1 && !vector.isScalable() &&
           vector.getElementType().isF64();
  return type.isF64();
}

static bool isStageInteger(Type type) {
  return type.isSignlessInteger(32) || type.isSignlessInteger(64);
}

bool StageOp::isDifferentiable(unsigned index) {
  return isStageReal(getFunctionType().getInput(index));
}

FunctionType StageOp::getVjpType() {
  FunctionType type = getFunctionType();
  SmallVector<Type> inputs(type.getInputs());
  llvm::append_range(inputs, type.getResults());
  SmallVector<Type> results;
  for (auto [index, input] : llvm::enumerate(type.getInputs()))
    if (isDifferentiable(index))
      results.push_back(input);
  return FunctionType::get(getContext(), inputs, results);
}

LogicalResult StageOp::verify() {
  FunctionType type = getFunctionType();
  for (Type input : type.getInputs())
    if (!isStageReal(input) && !isStageInteger(input))
      return emitOpError() << "expected each operand of f64, a vector of "
                              "f64, i32, or i64, got "
                           << input;
  if (type.getNumResults() == 0)
    return emitOpError() << "expected at least one result";
  for (Type result : type.getResults())
    if (!isStageReal(result))
      return emitOpError() << "expected each result of f64 or a vector of "
                              "f64, got "
                           << result;
  return success();
}

/// Whether `type`, the type of a function, is that of a batched body: its
/// values are tensors (D[ml4-batched-bodies]).
static bool isBatchedType(Type type) {
  auto function = dyn_cast<FunctionType>(type);
  return function && llvm::any_of(function.getInputs(), [](Type input) {
           return isa<RankedTensorType>(input);
         });
}

/// Checks that `actual` is the batched form of the value `declared` of one
/// element: `f64` as `tensor<T x E>`, `vector<N x f64>` as
/// `tensor<T x N x E>` with E f32 or f64 (a body may compute in a narrower
/// type than the declaration, as a model of float32 does), and an integer as
/// `tensor<T x iK>`. Sets `rows` to T, or checks it against T.
static bool isBatchedValue(Type declared, Type actual, int64_t &rows) {
  auto tensor = dyn_cast<RankedTensorType>(actual);
  if (!tensor || tensor.getRank() < 1 || ShapedType::isDynamic(tensor.getDimSize(0)) ||
      tensor.getDimSize(0) < 1 || !tensor.hasStaticShape())
    return false;
  if (rows && tensor.getDimSize(0) != rows)
    return false;
  rows = tensor.getDimSize(0);
  Type element = tensor.getElementType();
  if (declared.isSignlessInteger())
    return tensor.getRank() == 1 && element == declared;
  if (!element.isF32() && !element.isF64())
    return false;
  if (auto vector = dyn_cast<VectorType>(declared))
    return tensor.getRank() == 2 && tensor.getDimSize(1) == vector.getDimSize(0);
  return tensor.getRank() == 1;
}

/// Verifies that `name` names a body of the type `type` in the module: a
/// function with a body of ops without memory effects, of one element or
/// batched (D[ml4-batched-bodies]).
static LogicalResult verifyStageBody(StageOp stage, StringRef role,
                                     FlatSymbolRefAttr name, FunctionType type,
                                     SymbolTableCollection &symbolTable) {
  auto body = dyn_cast_or_null<FunctionOpInterface>(
      symbolTable.lookupNearestSymbolFrom(stage, name));
  if (!body)
    return stage.emitOpError()
           << "'" << name.getValue() << "', its " << role
           << " body, does not name a function";
  // A batched body that md-exec-lower-stage-bodies has lowered to buffers
  // was checked in its form of tensors.
  if (body->hasAttr("md.stage_body"))
    return success();
  if (isBatchedType(body.getFunctionType())) {
    auto actual = cast<FunctionType>(body.getFunctionType());
    int64_t rows = 0;
    bool fits = actual.getNumInputs() == type.getNumInputs() &&
                actual.getNumResults() == type.getNumResults();
    for (unsigned i = 0, e = fits ? type.getNumInputs() : 0; i != e; ++i)
      fits = fits && isBatchedValue(type.getInput(i), actual.getInput(i), rows);
    for (unsigned i = 0, e = fits ? type.getNumResults() : 0; i != e; ++i)
      fits =
          fits && isBatchedValue(type.getResult(i), actual.getResult(i), rows);
    if (!fits)
      return stage.emitOpError()
             << "expected its " << role << " body '" << name.getValue()
             << "' of the type " << type
             << " or its batched form, a tensor of a static number of rows "
                "for each value, got "
             << actual;
  } else if (body.getFunctionType() != type) {
    return stage.emitOpError()
           << "expected its " << role << " body '" << name.getValue()
           << "' of the type " << type << ", got " << body.getFunctionType();
  }
  if (body.isExternal())
    return stage.emitOpError()
           << "cannot bind the declaration '" << name.getValue()
           << "' as its " << role
           << " body: only a body in the module is lowered (an external "
              "body, such as an AOTInductor export, is step 4b of #377)";
  WalkResult walk = body.getFunctionBody().walk([&](Operation *op) {
    if (op->hasTrait<OpTrait::ReturnLike>() || isMemoryEffectFree(op))
      return WalkResult::advance();
    op->emitOpError() << "in the " << role << " body of the stage '"
                      << stage.getSymName()
                      << "', which must be pure: the op has memory effects";
    return WalkResult::interrupt();
  });
  return failure(walk.wasInterrupted());
}

LogicalResult StageOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  if (failed(verifyStageBody(*this, "forward", getForwardAttr(),
                             getFunctionType(), symbolTable)))
    return failure();
  if (FlatSymbolRefAttr vjp = getVjpAttr()) {
    if (failed(verifyStageBody(*this, "vjp", vjp, getVjpType(), symbolTable)))
      return failure();
    // A tile calls both bodies on the same rows.
    auto rowsOf = [&](FlatSymbolRefAttr name) -> int64_t {
      auto body = cast<FunctionOpInterface>(
          symbolTable.lookupNearestSymbolFrom(*this, name));
      if (auto rows = body->getAttrOfType<IntegerAttr>("md.stage_rows"))
        return rows.getInt();
      if (!isBatchedType(body.getFunctionType()))
        return 0;
      return cast<RankedTensorType>(
                 cast<FunctionType>(body.getFunctionType()).getInput(0))
          .getDimSize(0);
    };
    if (rowsOf(getForwardAttr()) != rowsOf(vjp))
      return emitOpError() << "expected its forward body and its "
                              "vector-Jacobian product of the same form and "
                              "the same number of rows";
  }
  return success();
}

int64_t StageOp::getRows() {
  auto body = dyn_cast_or_null<FunctionOpInterface>(
      SymbolTable::lookupNearestSymbolFrom(*this, getForwardAttr()));
  if (!body)
    return 0;
  // A body lowered to buffers keeps its rows (md-exec-lower-stage-bodies).
  if (auto rows = body->getAttrOfType<IntegerAttr>("md.stage_rows"))
    return rows.getInt();
  if (!isBatchedType(body.getFunctionType()))
    return 0;
  return cast<RankedTensorType>(
             cast<FunctionType>(body.getFunctionType()).getInput(0))
      .getDimSize(0);
}

LogicalResult ApplyStageOp::verify() {
  // A kernel of the md dialect or of md_dist holds the op: its operands are
  // the values of one element. A batched body stays in the kernel of the
  // md_exec loop until the loop is lowered to tiles (D[ml4-batched-bodies]).
  Operation *parent = (*this)->getParentOp();
  if (!parent || isa<FunctionOpInterface>(parent) ||
      !parent->getDialect() ||
      (parent->getDialect()->getNamespace() != "md" &&
       parent->getDialect()->getNamespace() != "md_dist" &&
       parent->getDialect()->getNamespace() != "md_exec"))
    return emitOpError() << "expected to be in a kernel of the md dialect";
  return success();
}

LogicalResult
ApplyStageOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto stage = symbolTable.lookupNearestSymbolFrom<StageOp>(*this,
                                                            getStageAttr());
  if (!stage)
    return emitOpError() << "'" << getStage() << "' does not name a stage";
  if (getVjp() && !stage.getVjpAttr())
    return emitOpError() << "the stage '" << getStage()
                         << "' has no vector-Jacobian product";
  FunctionType expected =
      getVjp() ? stage.getVjpType() : stage.getFunctionType();
  FunctionType actual = FunctionType::get(
      getContext(), getOperands().getTypes(), getResults().getTypes());
  if (actual != expected)
    return emitOpError() << "expected the type " << expected << " of "
                         << (getVjp() ? "the vector-Jacobian product of " : "")
                         << "the stage '" << getStage() << "', got "
                         << actual;
  return success();
}

//===----------------------------------------------------------------------===//
// EvaluateOp
//===----------------------------------------------------------------------===//

LogicalResult EvaluateOp::verify() {
  ArrayRef<int32_t> kinds = getRequestKinds();
  ArrayRef<int64_t> arguments = getRequestArguments();

  if (kinds.empty())
    return emitOpError() << "expected at least 1 request";
  if (kinds.size() != arguments.size())
    return emitOpError() << "expected one request argument per request";
  if (kinds.size() != getNumResults())
    return emitOpError() << "expected " << kinds.size()
                         << " results, one per request, got "
                         << getNumResults();
  if (getNumOperands() < 2)
    return emitOpError()
           << "expected at least 2 operands, a position field and a cell";

  Builder builder(getContext());
  for (unsigned i = 0, e = kinds.size(); i != e; ++i) {
    Type expected;
    switch (static_cast<Request>(kinds[i])) {
    case Request::Energy:
      expected = builder.getF64Type();
      break;
    case Request::Forces:
      expected = getOperand(0).getType();
      break;
    case Request::Virial:
      expected = VectorType::get({9}, builder.getF64Type());
      break;
    case Request::Derivative: {
      int64_t argument = arguments[i];
      if (argument < 2 || argument >= static_cast<int64_t>(getNumOperands()))
        return emitOpError()
               << "expected 'derivative' to name a parameter, an argument "
                  "index from 2 to "
               << getNumOperands() - 1 << ", got " << argument;
      // The derivative in a number is a number; that in a field of the
      // particles with one component of f64 is a field of its type, the
      // derivative in the value of each particle.
      Type type = getOperand(argument).getType();
      auto field = dyn_cast<FieldType>(type);
      bool isField = field && field.getNumComponents() == 1 &&
                     field.getElementType().isF64();
      if (!type.isF64() && !isField)
        return emitOpError()
               << "expected 'derivative' to name an argument of type f64 or "
                  "a field of f64 with one component, but argument "
               << argument << " has type " << type;
      expected = isField ? type : Type(builder.getF64Type());
      break;
    }
    }
    if (getResult(i).getType() != expected)
      return emitOpError()
             << "expected result " << i << " to have type " << expected
             << ", got " << getResult(i).getType();
  }
  return success();
}

/// Verifies that the operand types of `op` match the argument types of a
/// callee of type `type`.
static LogicalResult verifyCallOperands(Operation *op, FunctionType type) {
  if (type.getNumInputs() != op->getNumOperands())
    return op->emitOpError()
           << "expected " << type.getNumInputs() << " operands, got "
           << op->getNumOperands();
  for (unsigned i = 0, e = type.getNumInputs(); i != e; ++i)
    if (op->getOperand(i).getType() != type.getInput(i))
      return op->emitOpError()
             << "expected operand " << i << " to have type "
             << type.getInput(i) << ", got " << op->getOperand(i).getType();
  return success();
}

LogicalResult EvaluateOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto potential =
      symbolTable.lookupNearestSymbolFrom<PotentialOp>(*this, getCalleeAttr());
  if (!potential)
    return emitOpError() << "'" << getCallee()
                         << "' does not name a potential";
  return verifyCallOperands(getOperation(), potential.getFunctionType());
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  Operation *callee =
      symbolTable.lookupNearestSymbolFrom(*this, getCalleeAttr());
  if (!callee || !isa<FunctionOp, PotentialOp>(callee))
    return emitOpError() << "'" << getCallee()
                         << "' does not name a function or a potential";

  auto type =
      cast<FunctionType>(cast<FunctionOpInterface>(callee).getFunctionType());
  if (failed(verifyCallOperands(getOperation(), type)))
    return failure();

  if (type.getNumResults() != getNumResults())
    return emitOpError() << "expected " << type.getNumResults()
                         << " results, got " << getNumResults();
  for (unsigned i = 0, e = type.getNumResults(); i != e; ++i)
    if (getResult(i).getType() != type.getResult(i))
      return emitOpError()
             << "expected result " << i << " to have type "
             << type.getResult(i) << ", got " << getResult(i).getType();
  return success();
}
