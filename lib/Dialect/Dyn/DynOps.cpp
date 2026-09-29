// Ops of the dyn dialect.

#include "mdir/Dialect/Dyn/DynOps.h"

#include "mdir/Dialect/Dyn/DynDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;
using namespace mdir::dyn;
using mdir::md::FieldType;

#define GET_OP_CLASSES
#include "mdir/Dialect/Dyn/DynOps.cpp.inc"

/// Returns true if `type` is a field with 3 components of f64.
static bool isVectorField(Type type) {
  auto field = dyn_cast<FieldType>(type);
  return field && field.getNumComponents() == 3 &&
         field.getElementType().isF64();
}

//===----------------------------------------------------------------------===//
// KickOp and DriftOp
//===----------------------------------------------------------------------===//

LogicalResult KickOp::verify() {
  if (!isVectorField(getVelocities().getType()))
    return emitOpError() << "expected a field with 3 components of f64, got "
                         << getVelocities().getType();
  return success();
}

LogicalResult DriftOp::verify() {
  if (!isVectorField(getPositions().getType()))
    return emitOpError() << "expected a field with 3 components of f64, got "
                         << getPositions().getType();
  return success();
}

//===----------------------------------------------------------------------===//
// ProgramOp
//===----------------------------------------------------------------------===//

ParseResult ProgramOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFunctionType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFunctionType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void ProgramOp::print(OpAsmPrinter &printer) {
  function_interface_impl::printFunctionOp(
      printer, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

/// Verifies that every name in `names` is one of `known`.
static LogicalResult verifyNames(Operation *op, StringRef attribute,
                                 std::optional<ArrayAttr> names,
                                 ArrayRef<StringRef> known) {
  if (!names)
    return success();
  for (Attribute name : *names) {
    StringRef value = cast<StringAttr>(name).getValue();
    if (llvm::is_contained(known, value))
      continue;
    InFlightDiagnostic diagnostic =
        op->emitOpError() << "unknown name '" << value << "' in '"
                          << attribute << "'";
    diagnostic.attachNote() << "known names: " << llvm::join(known, ", ");
    return diagnostic;
  }
  return success();
}

LogicalResult ProgramOp::verify() {
  if (failed(verifyNames(getOperation(), "requires", getRequires(),
                         {"temperature", "pressure"})))
    return failure();
  return verifyNames(getOperation(), "provides", getProvides(),
                     {"symplectic", "time_reversible", "thermostatting",
                      "barostatting"});
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  auto program = cast<ProgramOp>((*this)->getParentOp());
  ArrayRef<Type> results = program.getResultTypes();

  if (getNumOperands() != results.size())
    return emitOpError() << "returns " << getNumOperands()
                         << " values, but the enclosing program has "
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
// StepOp
//===----------------------------------------------------------------------===//

LogicalResult StepOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto program =
      symbolTable.lookupNearestSymbolFrom<ProgramOp>(*this, getCalleeAttr());
  if (!program)
    return emitOpError() << "'" << getCallee() << "' does not name a program";

  FunctionType type = program.getFunctionType();
  if (type.getNumInputs() != getNumOperands())
    return emitOpError() << "expected " << type.getNumInputs()
                         << " operands, got " << getNumOperands();
  for (unsigned i = 0, e = type.getNumInputs(); i != e; ++i)
    if (getOperand(i).getType() != type.getInput(i))
      return emitOpError()
             << "expected operand " << i << " to have type "
             << type.getInput(i) << ", got " << getOperand(i).getType();

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
