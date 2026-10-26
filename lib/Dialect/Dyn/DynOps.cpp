// Ops of the dyn dialect.

#include "mdir/Dialect/Dyn/DynOps.h"

#include "mdir/Dialect/Dyn/DynDialect.h"
#include "mdir/Dialect/MD/MDOps.h"
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
// Couplings
//===----------------------------------------------------------------------===//

/// Verifies that `value` is a field with 3 components of f64.
static LogicalResult verifyVectorField(Operation *op, Value value) {
  if (!isVectorField(value.getType()))
    return op->emitOpError() << "expected a field with 3 components of f64, "
                                "got "
                             << value.getType();
  return success();
}

LogicalResult MomentumOp::verify() {
  return verifyVectorField(getOperation(), getVelocities());
}

LogicalResult KineticOp::verify() {
  return verifyVectorField(getOperation(), getVelocities());
}

LogicalResult HalfStepExcessOp::verify() {
  return verifyVectorField(getOperation(), getForces());
}

LogicalResult ShiftOp::verify() {
  return verifyVectorField(getOperation(), getVelocities());
}

LogicalResult ScaleOp::verify() {
  return verifyVectorField(getOperation(), getVelocities());
}

LogicalResult NoiseOp::verify() {
  if (failed(verifyVectorField(getOperation(), getVelocities())))
    return failure();
  auto velocities = cast<FieldType>(getVelocities().getType());
  auto ids = cast<FieldType>(getIds().getType());
  if (ids.getParticleSet() != velocities.getParticleSet() ||
      ids.getNumComponents() != 1 || !ids.getElementType().isInteger(32))
    return emitOpError() << "expected the numbers of the particles as a "
                            "field of i32 on "
                         << velocities.getParticleSet() << ", got " << ids;
  if (getSpread().convertToDouble() < 0.0)
    return emitOpError() << "expected 'spread' >= 0, got "
                         << getSpread().convertToDouble();
  return success();
}

LogicalResult VelocityRescalingOp::verify() {
  double target = getTargetKinetic().convertToDouble();
  double freedom = getFreedom().convertToDouble();
  double decay = getDecay().convertToDouble();
  if (!(freedom > 0.0))
    return emitOpError() << "expected 'freedom' > 0, got " << freedom;
  if (!(target > 0.0))
    return emitOpError() << "expected 'target_kinetic' > 0, got " << target;
  if (!(decay > 0.0 && decay < 1.0))
    return emitOpError() << "expected 0 < 'decay' < 1, got " << decay;
  return success();
}

LogicalResult NoseHooverOp::verify() {
  if (getChainLength() < 1)
    return emitOpError() << "expected 'chain_length' >= 1, got "
                         << getChainLength();
  if (!(getTimeConstant().convertToDouble() > 0.0))
    return emitOpError() << "expected 'time_constant' > 0, got "
                         << getTimeConstant().convertToDouble();
  if (!(getTemperature().convertToDouble() > 0.0))
    return emitOpError() << "expected 'temperature' > 0, got "
                         << getTemperature().convertToDouble();
  if (!(getFreedom().convertToDouble() > 0.0))
    return emitOpError() << "expected 'freedom' > 0, got "
                         << getFreedom().convertToDouble();
  if (!(getPeriod().convertToDouble() > 0.0))
    return emitOpError() << "expected 'period' > 0, got "
                         << getPeriod().convertToDouble();
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

/// Returns true if `names` holds `name`.
static bool hasName(std::optional<ArrayAttr> names, StringRef name) {
  if (!names)
    return false;
  return llvm::any_of(*names, [&](Attribute attribute) {
    return cast<StringAttr>(attribute).getValue() == name;
  });
}

/// The first op of `body` that computes a global sum, a sum whose result
/// the host reads (docs/dynamics.md, Section 2.2.4), or null. The flags
/// of the constraints, an i1, are not one: they are exact in any order.
static Operation *findGlobalSum(Region &body) {
  Operation *found = nullptr;
  body.walk([&](Operation *op) {
    bool isSum = false;
    if (isa<MomentumOp, KineticOp, HalfStepExcessOp>(op)) {
      isSum = true;
    } else if (isa<mdir::md::SumParticlesOp, mdir::md::SumTuplesOp,
                   mdir::md::SumRelationOp>(op)) {
      isSum = llvm::any_of(op->getResultTypes(), [](Type type) {
        return isa<FloatType>(getElementTypeOrSelf(type));
      });
    } else if (auto evaluate = dyn_cast<mdir::md::EvaluateOp>(op)) {
      isSum = llvm::any_of(evaluate.getRequestKinds(), [](int32_t kind) {
        auto request = static_cast<mdir::md::Request>(kind);
        return request == mdir::md::Request::Energy ||
               request == mdir::md::Request::Virial ||
               request == mdir::md::Request::Derivative;
      });
    }
    if (!isSum)
      return WalkResult::advance();
    found = op;
    return WalkResult::interrupt();
  });
  return found;
}

LogicalResult ProgramOp::verify() {
  if (failed(mdir::md::verifyReferencePrecision(getOperation(),
                                                 getFunctionType())))
    return failure();
  if (failed(verifyNames(getOperation(), "requires", getRequires(),
                         {"temperature", "pressure"})))
    return failure();
  if (failed(verifyNames(getOperation(), "provides", getProvides(),
                         {"symplectic", "time_reversible", "thermostatting",
                          "barostatting", "no_global_sums"})))
    return failure();
  if (isExternal())
    return success();

  // The claims against the body (docs/dynamics.md, Section 2.2.4).
  Operation *noise = nullptr;
  getBody().walk([&](NoiseOp op) {
    noise = op;
    return WalkResult::interrupt();
  });
  bool temperature = hasName(getRequires(), "temperature");
  if (noise && !temperature)
    return emitOpError() << "has a 'dyn.noise' and does not require "
                            "'temperature'";
  if (noise && !hasName(getProvides(), "thermostatting"))
    return emitOpError() << "has a 'dyn.noise' and does not provide "
                            "'thermostatting'";
  for (StringRef name : {"symplectic", "time_reversible"})
    if (noise && hasName(getProvides(), name)) {
      InFlightDiagnostic diagnostic =
          emitOpError() << "provides '" << name
                        << "', which the friction and the noise of "
                           "'dyn.noise' are not";
      diagnostic.attachNote(noise->getLoc()) << "the noise";
      return diagnostic;
    }
  if (hasName(getProvides(), "no_global_sums"))
    if (Operation *sum = findGlobalSum(getBody())) {
      InFlightDiagnostic diagnostic =
          emitOpError() << "provides 'no_global_sums' and computes a "
                           "global sum";
      diagnostic.attachNote(sum->getLoc()) << "the sum";
      return diagnostic;
    }
  return success();
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

void StepOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  auto unknown = [&] {
    effects.emplace_back(MemoryEffects::Read::get());
    effects.emplace_back(MemoryEffects::Write::get());
  };
  auto program = SymbolTable::lookupNearestSymbolFrom<ProgramOp>(
      getOperation(), getCalleeAttr());
  if (!program || program.isExternal())
    return unknown();
  // The effects of the body, on the resources they name. An effect on a
  // value of the body is one on its resource, since the value is not
  // visible here.
  bool isUnknown = false;
  program.getBody().walk([&](Operation *op) {
    if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
      return WalkResult::advance();
    auto memory = dyn_cast<MemoryEffectOpInterface>(op);
    if (!memory) {
      isUnknown = true;
      return WalkResult::interrupt();
    }
    SmallVector<MemoryEffects::EffectInstance> nested;
    memory.getEffects(nested);
    for (const MemoryEffects::EffectInstance &effect : nested)
      effects.emplace_back(effect.getEffect(), effect.getResource());
    return WalkResult::advance();
  });
  if (isUnknown)
    unknown();
}
