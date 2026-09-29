// Semantic differentiation of potentials.
//
// See docs/ops-m0.md, Section 5.

#include "mdir/Dialect/MD/Transforms/Passes.h"
#include "mdir/Dialect/MD/Transforms/ScalarDerivative.h"
#include "mdir/Dialect/MD/Transforms/Truncation.h"

#include "mdir/Dialect/MD/MDDialect.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace mdir::md;

namespace {

/// Builds the function that computes requested quantities from a potential.
class DerivativeBuilder {
public:
  DerivativeBuilder(PotentialOp potential, StringRef name)
      : potential(potential), name(name), builder(potential.getContext()),
        loc(potential.getLoc()) {}

  /// Returns the generated function, or a null op after emitting a
  /// diagnostic.
  FunctionOp build(ArrayRef<int32_t> kinds, ArrayRef<int64_t> arguments);

private:
  LogicalResult checkPositionUses();

  /// Sets `weight` to the derivative of the energy with respect to the
  /// result of `sum`, or to null if the energy does not depend on it.
  LogicalResult getWeight(SumRelationOp sum, Value &weight);

  /// Creates a pair op named `opName` with the operands of `source` and a
  /// copy of its kernel. The kernel still yields the pair energy.
  Operation *createPairOp(StringRef opName, SumRelationOp source,
                          Exchange exchange, Type resultType);

  /// In the kernel of `op`, which still yields the pair energy `u`, sets
  /// `factor` to `−weight · u'(r) / r`, or to null if it is zero.
  LogicalResult emitRadialFactor(Operation *op, Value weight, OpBuilder &kernel,
                                 Value &factor);

  LogicalResult buildForces(Value &forces);
  LogicalResult buildVirial(Value &virial);
  LogicalResult buildParameterDerivative(int64_t argument, Value &result);

  PotentialOp potential;
  StringRef name;
  OpBuilder builder;
  Location loc;

  FunctionOp function;
  Block *body = nullptr;
  Value energy;
  SmallVector<SumRelationOp> sums;
};

} // namespace

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Replaces the value that the kernel in `block` yields, then removes what
/// the kernel no longer needs.
static void setYield(Block &block, Value value) {
  cast<YieldOp>(block.getTerminator()).setOperand(0, value);
  eraseDeadOps(block);
}

LogicalResult DerivativeBuilder::checkPositionUses() {
  Value positions = body->getArgument(0);
  for (OpOperand &use : positions.getUses()) {
    Operation *user = use.getOwner();
    unsigned index = use.getOperandNumber();
    bool known = (isa<NeighborhoodOp>(user) && index == 0) ||
                 (isa<SumRelationOp>(user) && index == 1);
    if (!known)
      return user->emitError()
             << "cannot differentiate with respect to the positions through "
                "this use";
  }
  for (SumRelationOp sum : sums) {
    Block &kernel = sum.getKernel().front();
    if (!kernel.getArgument(1).use_empty())
      return sum.emitOpError()
             << "cannot differentiate a kernel that uses the displacement; "
                "only kernels that depend on the distance are supported";
    if (!sum.getResult().getType().isF64())
      return sum.emitOpError()
             << "cannot differentiate a sum whose result is not f64";
  }
  return success();
}

LogicalResult DerivativeBuilder::getWeight(SumRelationOp sum, Value &weight) {
  // With respect to one sum, everything that does not come from a scalar op
  // is an independent input.
  auto leaf = [](Value value, Value &tangent) -> LogicalResult {
    tangent = Value();
    Operation *op = value.getDefiningOp();
    if (!op || isa<MDDialect>(op->getDialect()))
      return success();
    return op->emitError() << "no derivative rule for '" << op->getName()
                           << "'";
  };
  ScalarDerivative derivative(builder, sum.getResult(), leaf);
  return derivative.get(energy, weight);
}

Operation *DerivativeBuilder::createPairOp(StringRef opName,
                                           SumRelationOp source,
                                           Exchange exchange,
                                           Type resultType) {
  MLIRContext *context = builder.getContext();
  OperationState state(loc, opName);
  state.addOperands(
      {source.getRelation(), source.getPositions(), source.getCell()});
  state.addOperands(source.getGathered());
  state.addAttribute("exchange", ExchangeAttr::get(context, exchange));
  state.addAttribute("exchange_basis",
                     ExchangeBasisAttr::get(context, ExchangeBasis::Derived));
  state.addRegion();
  state.addTypes(resultType);

  Operation *op = builder.create(state);
  IRMapping mapping;
  source.getKernel().cloneInto(&op->getRegion(0), mapping);
  return op;
}

LogicalResult DerivativeBuilder::emitRadialFactor(Operation *op, Value weight,
                                                  OpBuilder &kernel,
                                                  Value &factor) {
  Block &block = op->getRegion(0).front();
  Value pairEnergy = cast<YieldOp>(block.getTerminator()).getOperand(0);
  Value r = block.getArgument(0);

  ScalarDerivative derivative(kernel, r);
  Value slope;
  if (failed(derivative.get(pairEnergy, slope)))
    return failure();

  ScalarEmitter emit(kernel, loc);
  factor = emit.neg(emit.div(emit.mul(weight, slope), r));
  return success();
}

//===----------------------------------------------------------------------===//
// Forces
//===----------------------------------------------------------------------===//

LogicalResult DerivativeBuilder::buildForces(Value &forces) {
  Type fieldType = body->getArgument(0).getType();
  Type vectorType = cast<FieldType>(fieldType).getKernelValueType();

  SmallVector<Value> terms;
  for (SumRelationOp sum : sums) {
    Value weight;
    if (failed(getWeight(sum, weight)))
      return failure();
    if (!weight)
      continue;

    // K(i, j) = −weight · u'(r) · d / r
    Operation *gather =
        createPairOp(GatherRelationOp::getOperationName(), sum,
                     Exchange::Antisymmetric, fieldType);
    Block &block = gather->getRegion(0).front();
    OpBuilder kernel(block.getTerminator());
    Value factor;
    if (failed(emitRadialFactor(gather, weight, kernel, factor)))
      return failure();

    ScalarEmitter emit(kernel, loc);
    Value contribution;
    if (factor) {
      Value broadcast =
          vector::BroadcastOp::create(kernel, loc, vectorType, factor);
      contribution = emit.mul(broadcast, block.getArgument(1));
    } else {
      contribution = emit.constant(0.0, vectorType);
    }
    setYield(block, contribution);
    terms.push_back(gather->getResult(0));
  }

  if (terms.empty())
    return potential.emitOpError()
           << "cannot compute forces: the energy does not depend on a sum "
              "over a relation";
  if (terms.size() == 1) {
    forces = terms.front();
    return success();
  }

  // Add the terms particle by particle.
  OperationState state(loc, MapParticlesOp::getOperationName());
  state.addOperands(terms);
  state.addRegion();
  state.addTypes(fieldType);
  Operation *map = builder.create(state);

  Block *block = new Block();
  map->getRegion(0).push_back(block);
  for (unsigned i = 0, e = terms.size(); i != e; ++i)
    block->addArgument(vectorType, loc);

  OpBuilder kernel = OpBuilder::atBlockEnd(block);
  ScalarEmitter emit(kernel, loc);
  Value total;
  for (BlockArgument argument : block->getArguments())
    total = emit.add(total, argument);
  YieldOp::create(kernel, loc, ValueRange{total});

  forces = map->getResult(0);
  return success();
}

//===----------------------------------------------------------------------===//
// Virial
//===----------------------------------------------------------------------===//

LogicalResult DerivativeBuilder::buildVirial(Value &virial) {
  Type virialType = VectorType::get({9}, builder.getF64Type());

  ScalarEmitter outer(builder, loc);
  Value total;
  for (SumRelationOp sum : sums) {
    Value weight;
    if (failed(getWeight(sum, weight)))
      return failure();
    if (!weight)
      continue;

    // W = Σ d ⊗ K(i, j), with K(i, j) = factor · d.
    Operation *term = createPairOp(SumRelationOp::getOperationName(), sum,
                                   Exchange::Symmetric, virialType);
    Block &block = term->getRegion(0).front();
    OpBuilder kernel(block.getTerminator());
    Value factor;
    if (failed(emitRadialFactor(term, weight, kernel, factor)))
      return failure();

    ScalarEmitter emit(kernel, loc);
    Value contribution;
    if (factor) {
      Value d = block.getArgument(1);
      SmallVector<Value, 3> components;
      for (int64_t a = 0; a < 3; ++a)
        components.push_back(vector::ExtractOp::create(kernel, loc, d, a));
      SmallVector<Value, 9> elements;
      for (int64_t a = 0; a < 3; ++a)
        for (int64_t b = 0; b < 3; ++b)
          elements.push_back(emit.mul(
              factor, emit.mul(components[a], components[b])));
      contribution =
          vector::FromElementsOp::create(kernel, loc, virialType, elements);
    } else {
      contribution = emit.constant(0.0, virialType);
    }
    setYield(block, contribution);
    total = outer.add(total, term->getResult(0));
  }

  virial = total ? total : outer.constant(0.0, virialType);
  return success();
}

//===----------------------------------------------------------------------===//
// Derivative with respect to a parameter
//===----------------------------------------------------------------------===//

LogicalResult DerivativeBuilder::buildParameterDerivative(int64_t argument,
                                                          Value &result) {
  Value parameter = body->getArgument(argument);

  // The derivative of a sum over a relation is the sum of the derivative of
  // its kernel.
  auto leaf = [&](Value value, Value &tangent) -> LogicalResult {
    tangent = Value();
    Operation *op = value.getDefiningOp();
    if (!op)
      return success();

    auto sum = dyn_cast<SumRelationOp>(op);
    if (!sum)
      return op->emitError() << "cannot differentiate '" << op->getName()
                             << "' with respect to a parameter";

    Operation *term = createPairOp(SumRelationOp::getOperationName(), sum,
                                   Exchange::Symmetric, value.getType());
    Block &block = term->getRegion(0).front();
    Value pairEnergy = cast<YieldOp>(block.getTerminator()).getOperand(0);

    OpBuilder kernel(block.getTerminator());
    ScalarDerivative derivative(kernel, parameter);
    Value slope;
    if (failed(derivative.get(pairEnergy, slope)))
      return failure();
    if (!slope) {
      term->erase();
      return success();
    }
    setYield(block, slope);
    tangent = term->getResult(0);
    return success();
  };

  ScalarDerivative derivative(builder, parameter, leaf);
  if (failed(derivative.get(energy, result)))
    return failure();
  if (!result) {
    ScalarEmitter emit(builder, loc);
    result = emit.constant(0.0, builder.getF64Type());
  }
  return success();
}

//===----------------------------------------------------------------------===//
// The function
//===----------------------------------------------------------------------===//

FunctionOp DerivativeBuilder::build(ArrayRef<int32_t> kinds,
                                    ArrayRef<int64_t> arguments) {
  if (potential.isExternal()) {
    potential.emitOpError() << "cannot differentiate a declaration";
    return FunctionOp();
  }
  if (!llvm::hasSingleElement(potential.getBody())) {
    potential.emitOpError()
        << "cannot differentiate a body with more than one block";
    return FunctionOp();
  }

  FunctionType potentialType = potential.getFunctionType();
  SmallVector<Type> resultTypes;
  for (int32_t kind : kinds) {
    switch (static_cast<Request>(kind)) {
    case Request::Energy:
    case Request::Derivative:
      resultTypes.push_back(builder.getF64Type());
      break;
    case Request::Forces:
      resultTypes.push_back(potentialType.getInput(0));
      break;
    case Request::Virial:
      resultTypes.push_back(VectorType::get({9}, builder.getF64Type()));
      break;
    }
  }

  OperationState state(loc, FunctionOp::getOperationName());
  state.addAttribute(FunctionOp::getSymNameAttrName(state.name),
                     builder.getStringAttr(name));
  state.addAttribute(
      FunctionOp::getFunctionTypeAttrName(state.name),
      TypeAttr::get(
          builder.getFunctionType(potentialType.getInputs(), resultTypes)));
  state.addRegion();

  // Place the function after the potential and after the functions that
  // were generated from it earlier, so that they appear in order of creation.
  Operation *last = potential;
  std::string prefix = (potential.getSymName() + ".").str();
  while (Operation *next = last->getNextNode()) {
    auto generated = dyn_cast<FunctionOp>(next);
    if (!generated || !generated.getSymName().starts_with(prefix))
      break;
    last = next;
  }
  builder.setInsertionPointAfter(last);
  function = cast<FunctionOp>(builder.create(state));

  IRMapping mapping;
  potential.getBody().cloneInto(&function.getBody(), mapping);
  body = &function.getBody().front();
  auto oldReturn = cast<ReturnOp>(body->getTerminator());
  energy = oldReturn.getOperand(0);

  auto fail = [&]() {
    function.erase();
    return FunctionOp();
  };

  for (Operation &op : *body) {
    auto sum = dyn_cast<SumRelationOp>(&op);
    if (!sum)
      continue;
    if (failed(expandTruncation(sum)))
      return fail();
    sums.push_back(sum);
  }

  bool needsPositions = llvm::any_of(kinds, [](int32_t kind) {
    auto request = static_cast<Request>(kind);
    return request == Request::Forces || request == Request::Virial;
  });
  if (needsPositions && failed(checkPositionUses()))
    return fail();

  builder.setInsertionPoint(oldReturn);
  SmallVector<Value> results;
  for (unsigned i = 0, e = kinds.size(); i != e; ++i) {
    Value result;
    LogicalResult status = success();
    switch (static_cast<Request>(kinds[i])) {
    case Request::Energy:
      result = energy;
      break;
    case Request::Forces:
      status = buildForces(result);
      break;
    case Request::Virial:
      status = buildVirial(result);
      break;
    case Request::Derivative:
      status = buildParameterDerivative(arguments[i], result);
      break;
    }
    if (failed(status))
      return fail();
    results.push_back(result);
  }

  ReturnOp::create(builder, loc, results);
  oldReturn.erase();
  eraseDeadOps(*body);
  return function;
}

/// The name of the function that computes `kinds` from the potential
/// `potential`: `lj.energy_forces`, for example.
static std::string getDerivativeName(StringRef potential,
                                     ArrayRef<int32_t> kinds,
                                     ArrayRef<int64_t> arguments) {
  std::string name = (potential + ".").str();
  for (unsigned i = 0, e = kinds.size(); i != e; ++i) {
    if (i != 0)
      name += "_";
    auto kind = static_cast<Request>(kinds[i]);
    name += stringifyRequest(kind).str();
    if (kind == Request::Derivative)
      name += std::to_string(arguments[i]);
  }
  return name;
}

namespace mdir {
namespace md {

#define GEN_PASS_DEF_DIFFERENTIATE
#include "mdir/Dialect/MD/Transforms/Passes.h.inc"

namespace {
class Differentiate : public impl::DifferentiateBase<Differentiate> {
public:
  using impl::DifferentiateBase<Differentiate>::DifferentiateBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    MLIRContext *context = module.getContext();

    SmallVector<EvaluateOp> evaluations;
    module->walk([&](EvaluateOp op) { evaluations.push_back(op); });

    for (EvaluateOp evaluate : evaluations) {
      auto potential = dyn_cast_or_null<PotentialOp>(
          SymbolTable::lookupNearestSymbolFrom(evaluate,
                                               evaluate.getCalleeAttr()));
      if (!potential) {
        evaluate.emitOpError() << "'" << evaluate.getCallee()
                               << "' does not name a potential";
        return signalPassFailure();
      }

      std::string name = getDerivativeName(potential.getSymName(),
                                           evaluate.getRequestKinds(),
                                           evaluate.getRequestArguments());
      if (Operation *existing = SymbolTable::lookupSymbolIn(module, name)) {
        if (!isa<FunctionOp>(existing)) {
          evaluate.emitOpError()
              << "the name '" << name
              << "' is needed for a generated function, but is taken";
          return signalPassFailure();
        }
      } else {
        DerivativeBuilder derivative(potential, name);
        if (!derivative.build(evaluate.getRequestKinds(),
                              evaluate.getRequestArguments()))
          return signalPassFailure();
      }

      OperationState state(evaluate.getLoc(), CallOp::getOperationName());
      state.addOperands(evaluate.getOperands());
      state.addAttribute("callee", FlatSymbolRefAttr::get(context, name));
      state.addTypes(evaluate.getResultTypes());
      OpBuilder builder(evaluate);
      Operation *call = builder.create(state);
      evaluate->replaceAllUsesWith(call);
      evaluate.erase();
    }
  }
};
} // namespace

} // namespace md
} // namespace mdir
