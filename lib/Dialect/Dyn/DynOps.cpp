// Ops of the dyn dialect.

#include "mdir/Dialect/Dyn/DynOps.h"

#include "mdir/Dialect/Dyn/DynDialect.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
// The barostat
//===----------------------------------------------------------------------===//

/// Verifies that each relation of `groups` is a set of tuples of the
/// particle set of `field`.
static LogicalResult verifyGroups(Operation *op, Value field,
                                  ValueRange groups) {
  auto set = cast<FieldType>(field.getType()).getParticleSet();
  for (Value group : groups) {
    auto relation = cast<mdir::md::RelationType>(group.getType());
    if (relation.getParticleSet() != set || relation.getArity() < 2)
      return op->emitOpError()
             << "expected the groups as relations of two or more members "
                "of "
             << set << ", got " << relation;
  }
  return success();
}

LogicalResult GroupKineticOp::verify() {
  if (failed(verifyVectorField(getOperation(), getVelocities())))
    return failure();
  return verifyGroups(getOperation(), getVelocities(), getGroups());
}

LogicalResult ScalePositionsOp::verify() {
  if (failed(verifyVectorField(getOperation(), getPositions())))
    return failure();
  return verifyGroups(getOperation(), getPositions(), getGroups());
}

LogicalResult CellRescalingOp::verify() {
  StringRef coupling = getCoupling();
  size_t count = coupling == "isotropic"        ? 1
                 : coupling == "semi_isotropic" ? 2
                 : coupling == "anisotropic"    ? 3
                                                : 0;
  if (!count)
    return emitOpError() << "unknown coupling '" << coupling
                         << "'; expected 'isotropic', 'semi_isotropic', or "
                            "'anisotropic'";
  ArrayAttr compressibility = getCompressibility();
  if (compressibility.size() != count)
    return emitOpError() << "expected " << count
                         << " numbers in 'compressibility' for the coupling '"
                         << coupling << "', got " << compressibility.size();
  bool any = false;
  for (Attribute value : compressibility) {
    double beta = cast<FloatAttr>(value).getValueAsDouble();
    if (beta < 0.0)
      return emitOpError() << "expected no negative 'compressibility'";
    any |= beta > 0.0;
  }
  if (count == 3 && !any)
    return emitOpError() << "expected a 'compressibility' that is not 0";
  if (getTension() && coupling != "semi_isotropic")
    return emitOpError() << "'tension' needs the coupling 'semi_isotropic'";
  if (!(getRate().convertToDouble() > 0.0))
    return emitOpError() << "expected 'rate' > 0";
  if (!(getKT().convertToDouble() > 0.0))
    return emitOpError() << "expected 'kT' > 0";
  return success();
}

void CellRescalingOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // The strains of semi-isotropic and anisotropic coupling add to the
  // statistics of the pressures of the axes; step 3b of #541 moves them
  // into a call of their own.
  if (getCoupling() != "isotropic") {
    effects.emplace_back(MemoryEffects::Read::get(),
                         StatisticsResource::get());
    effects.emplace_back(MemoryEffects::Write::get(),
                         StatisticsResource::get());
  }
}

/// Verifies that `count` values of the cell are 3 or 6, and `memory`, if
/// given, holds as many.
static LogicalResult verifyCell(Operation *op, size_t count, Value memory) {
  if (count != 3 && count != 6)
    return op->emitOpError()
           << "expected the 3 edges, or with the 3 tilts 6 values, of the "
              "cell, got "
           << count;
  if (memory) {
    auto type = cast<MemRefType>(memory.getType());
    if (!type.hasStaticShape() ||
        type.getDimSize(0) != static_cast<int64_t>(count))
      return op->emitOpError() << "expected the memory of " << count
                               << " values of the cell, got " << type;
  }
  return success();
}

LogicalResult ScaleCellOp::verify() {
  if (getCell().size() != getResults().size())
    return emitOpError() << "expected as many results as values of the "
                            "cell";
  return verifyCell(getOperation(), getCell().size(), Value());
}

LogicalResult LoadCellOp::verify() {
  return verifyCell(getOperation(), getCell().size(), getMemory());
}

LogicalResult StoreCellOp::verify() {
  return verifyCell(getOperation(), getCell().size(), getMemory());
}

/// Verifies that `memory` holds the 9 values of the state of the pressure.
static LogicalResult verifyPressureState(Operation *op, Value memory) {
  auto type = cast<MemRefType>(memory.getType());
  if (!type.hasStaticShape() || type.getDimSize(0) != 9)
    return op->emitOpError() << "expected the memory of the 9 values of the "
                                "state of the pressure, got "
                             << type;
  return success();
}

LogicalResult LoadPressureStateOp::verify() {
  return verifyPressureState(getOperation(), getMemory());
}

LogicalResult StorePressureStateOp::verify() {
  return verifyPressureState(getOperation(), getMemory());
}

LogicalResult ScalingWorkOp::verify() {
  StringRef count = getCount();
  bool strain = static_cast<bool>(getStrain());
  bool logStrain = static_cast<bool>(getLogStrain());
  bool before = static_cast<bool>(getVirialBefore());
  bool after = static_cast<bool>(getVirialAfter());
  size_t energies = getEnergies().size(), volumes = getVolumes().size();
  // The operands of each count (docs/dynamics.md, Section 2.2.3).
  struct Rule {
    StringRef count;
    bool strain, logStrain, before, after;
    size_t energies, volumes;
  };
  static const Rule rules[] = {
      {"trotter", false, true, true, true, 0, 2},
      {"trotter_before", false, true, true, false, 0, 1},
      {"exact", true, false, false, false, 2, 2},
      {"first_order", true, false, true, false, 0, 1},
  };
  for (const Rule &rule : rules) {
    if (rule.count != count)
      continue;
    if (rule.strain != strain || rule.logStrain != logStrain ||
        rule.before != before || rule.after != after ||
        rule.energies != energies || rule.volumes != volumes)
      return emitOpError()
             << "the count '" << count << "' takes "
             << (rule.strain ? "strain" : "log_strain")
             << (rule.before ? ", virial_before" : "")
             << (rule.after ? ", virial_after" : "")
             << (rule.energies ? ", energies (2)" : "") << ", and "
             << rule.volumes << " volume" << (rule.volumes == 1 ? "" : "s");
    return success();
  }
  return emitOpError() << "unknown count '" << count
                       << "'; expected 'trotter', 'trotter_before', "
                          "'exact', or 'first_order'";
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
    if (isa<MomentumOp, KineticOp, HalfStepExcessOp, GroupKineticOp>(op)) {
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
  // A scaling of the positions within the step: the strain comes from the
  // pressure that the barostat computes (step_trotter).
  Operation *scaling = nullptr;
  getBody().walk([&](ScalePositionsOp op) {
    scaling = op;
    return WalkResult::interrupt();
  });
  if (scaling && !hasName(getRequires(), "pressure"))
    return emitOpError() << "has a 'dyn.scale_positions' and does not "
                            "require 'pressure'";
  if (scaling && !hasName(getProvides(), "barostatting"))
    return emitOpError() << "has a 'dyn.scale_positions' and does not "
                            "provide 'barostatting'";
  for (StringRef name : {"symplectic", "time_reversible"})
    if (scaling && hasName(getProvides(), name)) {
      InFlightDiagnostic diagnostic =
          emitOpError() << "provides '" << name
                        << "', which a scaling within the step is not";
      diagnostic.attachNote(scaling->getLoc()) << "the scaling";
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

//===----------------------------------------------------------------------===//
// Effects of a program (D355, #673)
//===----------------------------------------------------------------------===//

/// Adds the effects of the ops of `body` to `effects`, each on its
/// resource; returns the first op of unknown effects, or null. A call has
/// the effects that its function declares in `dyn.effects`.
static Operation *
collectEffects(Region &body, SymbolTableCollection &symbols,
               SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  Operation *unknown = nullptr;
  body.walk([&](Operation *op) {
    if (auto call = dyn_cast<func::CallOp>(op)) {
      auto function = symbols.lookupNearestSymbolFrom<func::FuncOp>(
          op, call.getCalleeAttr());
      auto declared = function ? function->getAttrOfType<DictionaryAttr>(
                                     "dyn.effects")
                               : DictionaryAttr();
      std::string error;
      if (!declared || failed(mdir::effects::parseEffectSummary(declared, effects, error))) {
        unknown = op;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }
    if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
      return WalkResult::advance();
    auto memory = dyn_cast<MemoryEffectOpInterface>(op);
    if (!memory) {
      unknown = op;
      return WalkResult::interrupt();
    }
    SmallVector<MemoryEffects::EffectInstance> nested;
    memory.getEffects(nested);
    for (const MemoryEffects::EffectInstance &effect : nested)
      effects.emplace_back(effect.getEffect(), effect.getResource());
    return WalkResult::advance();
  });
  return unknown;
}

LogicalResult
ProgramOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  DictionaryAttr summary = getEffectsAttr();
  if (!summary || isExternal())
    return success();
  SmallVector<MemoryEffects::EffectInstance> stated, body;
  std::string error;
  if (failed(mdir::effects::parseEffectSummary(summary, stated, error)))
    return emitOpError() << "in 'effects': " << error;
  if (Operation *unknown = collectEffects(getBody(), symbolTable, body)) {
    InFlightDiagnostic diagnostic =
        emitOpError() << "states its effects, and an op of its body has "
                         "unknown effects";
    diagnostic.attachNote(unknown->getLoc())
        << "the op; a call has the effects that its function declares in "
           "'dyn.effects'";
    return diagnostic;
  }
  for (const MemoryEffects::EffectInstance &effect : body) {
    bool covered = llvm::any_of(stated, [&](const auto &other) {
      return other.getEffect() == effect.getEffect() &&
             other.getResource() == effect.getResource();
    });
    if (covered)
      continue;
    StringRef kind = isa<MemoryEffects::Read>(effect.getEffect()) ? "read"
                     : isa<MemoryEffects::Write>(effect.getEffect())
                         ? "write"
                         : "allocate or free";
    return emitOpError() << "does not state the effect of its body: "
                         << kind << " of '" << effect.getResource()->getName()
                         << "'";
  }
  return success();
}

void StepOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // The summary of the program, which its verifier checked against the
  // body; without one, every resource.
  auto program = SymbolTable::lookupNearestSymbolFrom<ProgramOp>(
      getOperation(), getCalleeAttr());
  DictionaryAttr summary = program ? program.getEffectsAttr() : DictionaryAttr();
  SmallVector<MemoryEffects::EffectInstance> stated;
  std::string error;
  if (!summary || failed(mdir::effects::parseEffectSummary(summary, stated, error))) {
    effects.emplace_back(MemoryEffects::Read::get());
    effects.emplace_back(MemoryEffects::Write::get());
    return;
  }
  effects.append(stated.begin(), stated.end());
}
