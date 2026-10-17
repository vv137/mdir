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

MemRefType mdir::mdrt::getTableBufferType(md::TableType table) {
  SmallVector<int64_t, 2> shape(table.getRank(), ShapedType::kDynamic);
  return MemRefType::get(shape, table.getElementType());
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
  if (auto table = dyn_cast<md::TableType>(getResult().getType())) {
    if (getBuffer().getType() != getTableBufferType(table))
      return emitOpError() << "expected the buffer of " << table
                           << " to have type " << getTableBufferType(table)
                           << ", got " << getBuffer().getType();
    return success();
  }
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

/// The type of the buffer of the host that takes `number`, a result of
/// mdrt.host_call: memref<1xf64> for an f64, memref<kxf64> for a vector of
/// k; null for any other type.
static MemRefType getNumberBufferType(Type number) {
  if (number.isF64())
    return MemRefType::get({1}, number);
  auto vector = dyn_cast<VectorType>(number);
  if (vector && vector.getRank() == 1 && !vector.isScalable() &&
      vector.getElementType().isF64())
    return MemRefType::get({vector.getDimSize(0)}, vector.getElementType());
  return MemRefType();
}

LogicalResult HostCallOp::verify() {
  unsigned numOuts = getOuts().size();
  if (getNumResults() < numOuts)
    return emitOpError() << "expected a result for each field in 'outs'";
  for (unsigned i = 0; i != numOuts; ++i)
    if (getResult(i).getType() != getOuts()[i].getType())
      return emitOpError() << "expected result " << i
                           << " to have the type of the field in 'outs', "
                           << getOuts()[i].getType();
  // The pairs of images are buffers of the device, which the host takes
  // only where they are (D277).
  for (Value input : getInputs())
    if (isa<mdir::md::EdgesType>(input.getType()) && !getInPlace())
      return emitOpError() << "takes the pairs of images, which a host call "
                              "takes only in place";
  for (Value number : getNumbers())
    if (!getNumberBufferType(number.getType()))
      return emitOpError() << "expected the results after the fields to be "
                              "f64 or vectors of f64, got "
                           << number.getType();
  return success();
}

LogicalResult
HostCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto callee = symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(
      getOperation(), getCalleeAttr());
  if (!callee)
    return emitOpError() << "'" << getCallee()
                         << "' does not name a function";

  // The operands, the buffers of the fields of `outs`, and the buffers of
  // the host that take the numbers. The pairs of images of
  // `md_exec.build_edges` are four arguments: its slots, counts, and
  // offsets, and the number of their build (D277).
  ArrayRef<Type> expected = callee.getArgumentTypes();
  MLIRContext *context = getContext();
  Type narrow = IntegerType::get(context, 32);
  SmallVector<Type, 4> edgeTypes = {
      MemRefType::get({ShapedType::kDynamic, 5}, narrow),
      MemRefType::get({ShapedType::kDynamic}, narrow),
      MemRefType::get({ShapedType::kDynamic}, narrow),
      IntegerType::get(context, 64)};
  unsigned numOperands = 0, numOuts = getOuts().size();
  for (Value input : getInputs())
    numOperands += isa<mdir::md::EdgesType>(input.getType()) ? edgeTypes.size() : 1;
  unsigned total = numOperands + numOuts + getNumbers().size();
  if (expected.size() != total)
    return emitOpError() << "'" << getCallee() << "' takes "
                         << expected.size() << " arguments, got " << total;
  if (!callee.getResultTypes().empty())
    return emitOpError() << "'" << getCallee()
                         << "' returns a value; a function of the host "
                            "that reads fields returns none";

  SmallVector<Type> given;
  for (Value input : getInputs()) {
    if (isa<mdir::md::EdgesType>(input.getType()))
      given.append(edgeTypes.begin(), edgeTypes.end());
    else
      given.push_back(input.getType());
  }
  for (Value out : getOuts())
    given.push_back(out.getType());
  for (unsigned i = 0, e = numOperands + numOuts; i != e; ++i) {
    Type type = given[i];
    auto field = dyn_cast<FieldType>(type);
    bool fits = field ? canHold(expected[i], field) : type == expected[i];
    if (!fits)
      return emitOpError()
             << "argument " << i << " of '" << getCallee() << "' has type "
             << expected[i] << ", which does not take " << type;
  }
  for (auto [index, number] : llvm::enumerate(getNumbers())) {
    unsigned i = numOperands + numOuts + index;
    if (expected[i] != getNumberBufferType(number.getType()))
      return emitOpError()
             << "argument " << i << " of '" << getCallee() << "' has type "
             << expected[i] << ", which does not take the result "
             << number.getType();
  }
  return success();
}

unsigned mdir::mdrt::getHostArgumentCount(Type input) {
  return isa<mdir::md::EdgesType>(input) ? 4 : 1;
}

unsigned mdir::mdrt::getHostArgumentIndex(HostCallOp call, unsigned index) {
  unsigned argument = 0;
  ValueRange inputs = call.getInputs();
  for (unsigned i = 0; i != index && i < inputs.size(); ++i)
    argument += getHostArgumentCount(inputs[i].getType());
  if (index > inputs.size())
    argument += index - inputs.size();
  return argument;
}

MemRefType mdir::mdrt::getHostNumberType(Type number) {
  return getNumberBufferType(number);
}

void HostCallOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // The state of the host: what the function writes or keeps (a file, a
  // model's energy on the device) is not seen by the compiler.
  effects.emplace_back(MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get());
  // Buffers, where the fields have them: those of `outs` are read and
  // written.
  for (OpOperand &operand : getInputsMutable())
    if (isa<MemRefType>(operand.get().getType()))
      effects.emplace_back(MemoryEffects::Read::get(), &operand);
  for (OpOperand &operand : getOutsMutable())
    if (isa<MemRefType>(operand.get().getType())) {
      effects.emplace_back(MemoryEffects::Read::get(), &operand);
      effects.emplace_back(MemoryEffects::Write::get(), &operand);
    }
}
