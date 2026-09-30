// Ops of the md dialect.

#include "mdir/Dialect/MD/MDOps.h"

#include "mdir/Dialect/MD/MDDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/FunctionImplementation.h"

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

/// Verifies the kernel: its arguments and its terminator. `yieldType` is the
/// type the kernel must yield.
template <typename OpTy>
static LogicalResult verifyPairKernel(OpTy op, Type yieldType) {
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
  if (yield.getNumOperands() != 1)
    return yield.emitOpError()
           << "expected exactly 1 value, got " << yield.getNumOperands();
  if (yield.getOperand(0).getType() != yieldType)
    return yield.emitOpError()
           << "expected a value of type " << yieldType << ", got "
           << yield.getOperand(0).getType();
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
  return verifyPairKernel(*this, getResult().getType());
}

//===----------------------------------------------------------------------===//
// GatherRelationOp
//===----------------------------------------------------------------------===//

LogicalResult GatherRelationOp::verify() {
  if (failed(verifyPairOperands(*this)))
    return failure();

  auto relation = cast<RelationType>(getRelation().getType());
  auto result = cast<FieldType>(getResult().getType());
  if (result.getParticleSet() != relation.getParticleSet())
    return emitOpError() << "the relation is on " << relation.getParticleSet()
                         << ", but the result belongs to "
                         << result.getParticleSet();
  return success();
}

LogicalResult GatherRelationOp::verifyRegions() {
  auto result = cast<FieldType>(getResult().getType());
  return verifyPairKernel(*this, result.getKernelValueType());
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
  if (getGrid().size() != 3 ||
      llvm::any_of(getGrid(), [](int64_t points) { return points < 2; }))
    return emitOpError() << "expected three numbers of points of at least 2";
  if (getOrder() < 3 || getOrder() > 8)
    return emitOpError() << "expected an order from 3 to 8";
  if (!(getBeta().convertToDouble() > 0.0))
    return emitOpError() << "expected a positive beta";
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
    if (!isa<FloatType>(buffer.getElementType()))
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
  auto relation = cast<RelationType>(op.getRelation().getType());
  FlatSymbolRefAttr tupleSet = relation.getTupleSet();
  if (!tupleSet)
    return op.emitOpError()
           << "expected the relation of a tuple set, got " << relation;

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
/// yield `numYields` values of the type `yieldType`.
template <typename OpTy>
static LogicalResult verifyTupleKernel(OpTy op, unsigned numYields,
                                       Type yieldType) {
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
  if (yield.getNumOperands() != numYields)
    return yield.emitOpError() << "expected " << numYields << " values, got "
                               << yield.getNumOperands();
  for (Value value : yield.getOperands())
    if (value.getType() != yieldType)
      return yield.emitOpError() << "expected values of type " << yieldType
                                 << ", got " << value.getType();
  return success();
}

LogicalResult SumTuplesOp::verify() {
  if (failed(verifyTupleOperands(*this)))
    return failure();
  if (!isRealOrRealVector(getResult().getType()))
    return emitOpError()
           << "expected the result to be f64 or a fixed-size vector of f64, "
              "got "
           << getResult().getType();
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

LogicalResult GatherTuplesOp::verifyRegions() {
  auto result = cast<FieldType>(getResult().getType());
  return verifyTupleKernel(*this, getArity(), result.getKernelValueType());
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
      if (!getOperand(argument).getType().isF64())
        return emitOpError()
               << "expected 'derivative' to name an argument of type f64, "
                  "but argument "
               << argument << " has type " << getOperand(argument).getType();
      expected = builder.getF64Type();
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
