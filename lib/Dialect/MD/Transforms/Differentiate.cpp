// Semantic differentiation of potentials.
//
// See docs/ops-m0.md, Section 5, and docs/design-m1.md, Section 4.

#include "mdir/Dialect/MD/Transforms/Passes.h"
#include "mdir/Dialect/MD/Transforms/ScalarDerivative.h"
#include "mdir/Dialect/MD/Transforms/Truncation.h"

#include "mdir/Dialect/MD/MDDialect.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace mdir::md;

namespace {

/// Builds the function that computes requested quantities from a potential.
class DerivativeBuilder {
public:
  DerivativeBuilder(PotentialOp potential, StringRef name, bool remarks)
      : potential(potential), name(name), builder(potential.getContext()),
        loc(potential.getLoc()), remarks(remarks) {}

  /// Returns the generated function, or a null op after emitting a
  /// diagnostic.
  FunctionOp build(ArrayRef<int32_t> kinds, ArrayRef<int64_t> arguments);

private:
  LogicalResult checkPositionUses();

  /// Sets `weight` to the derivative of the energy with respect to `sum`,
  /// the result of a sum, or to null if the energy does not depend on it.
  LogicalResult getWeight(Value sum, Value &weight);

  /// Creates a pair op named `opName` with the operands of `source` and a
  /// copy of its kernel. The kernel still yields the pair energy.
  Operation *createPairOp(StringRef opName, SumRelationOp source,
                          Exchange exchange, Type resultType);

  /// In the kernel of `op`, which still yields the pair energy `u`, sets
  /// `factor` to `−weight · u'(r) / r`, or to null if it is zero.
  LogicalResult emitRadialFactor(Operation *op, Value weight, OpBuilder &kernel,
                                 Value &factor);

  /// Creates an op over tuples named `opName` with the operands and the
  /// coordinates of `source` and a copy of its kernel. The kernel still
  /// yields the energy of a tuple.
  Operation *createTupleOp(StringRef opName, SumTuplesOp source,
                           Type resultType);

  /// In the kernel of `op`, which still yields the energy `u` of a tuple,
  /// sets `forces` to `−weight · ∂u/∂x_m` for each member `m` of the tuple,
  /// or to null where it is zero. With `virial`, sets it to the sum of
  /// `d ⊗ F` over the members, or to null.
  LogicalResult emitTupleForces(Operation *op, Value weight, OpBuilder &kernel,
                                SmallVectorImpl<Value> &forces,
                                Value *virial = nullptr);

  /// `field` times the number `weight`, particle by particle, or `field`
  /// itself where `weight` is the constant 1.
  Value scaleField(Value field, Value weight);

  LogicalResult buildForces(Value &forces);
  LogicalResult buildVirial(Value &virial);

  //===--------------------------------------------------------------------===//
  // Intermediate fields
  //===--------------------------------------------------------------------===//
  //
  // A field that the potential computes from the positions, the result of a
  // gather over pairs or of a map over particles that reads one, enters the
  // energy through sums that read it. The derivative of the energy with
  // respect to such a field, its adjoint, is a field as well, which flows
  // back from the sums to the gathers, where it gives forces.

  /// Whether `field` depends on the positions.
  bool isPositional(Value field);
  /// Sets `adjoints` for every positional field the energy depends on.
  LogicalResult buildAdjoints();
  /// `fields` added particle by particle.
  Value addFields(ArrayRef<Value> fields);
  /// The forces, or with `virial` the virial, that the gather `gather`
  /// gives through the adjoint `adjoint` of its result.
  LogicalResult emitGatherTerm(GatherRelationOp gather, Value adjoint,
                               bool virial, Value &result);

  llvm::DenseMap<Value, bool> positional;
  llvm::DenseMap<Value, Value> adjoints;
  bool adjointsBuilt = false;
  LogicalResult buildParameterDerivative(int64_t argument, Value &result);

  PotentialOp potential;
  StringRef name;
  OpBuilder builder;
  Location loc;
  /// Whether a parameter derivative remarks on each op that it takes as
  /// independent of the parameter.
  bool remarks;

  FunctionOp function;
  Block *body = nullptr;
  Value energy;
  SmallVector<SumRelationOp> sums;
  SmallVector<SumTuplesOp> tupleSums;
  /// Sums over particles that read the positions themselves: terms of the
  /// absolute positions, which give forces but no virial.
  SmallVector<SumParticlesOp> particleSums;
  /// Reciprocal sums, which yield their own forces and virial.
  SmallVector<ReciprocalOp> reciprocals;
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
    bool known = (isa<NeighborhoodOp, ReciprocalOp>(user) && index == 0) ||
                 (isa<SumRelationOp, SumTuplesOp, GatherRelationOp>(user) &&
                  index == 1) ||
                 isa<SumParticlesOp>(user);
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
  for (Operation &op : *body) {
    auto gather = dyn_cast<GatherRelationOp>(&op);
    if (!gather || !isPositional(gather.getResult()))
      continue;
    if (!gather.getKernel().front().getArgument(1).use_empty())
      return gather.emitOpError()
             << "cannot differentiate a field gathered from a kernel that "
                "uses the displacement; only kernels that depend on the "
                "distance are supported";
    auto type = cast<FieldType>(gather.getResult().getType());
    if (!type.getKernelValueType().isF64())
      return gather.emitOpError()
             << "cannot differentiate a gathered field that is not of f64";
  }
  for (SumTuplesOp sum : tupleSums) {
    if (!sum.getResult().getType().isF64())
      return sum.emitOpError()
             << "cannot differentiate a sum whose result is not f64";
  }
  for (SumParticlesOp sum : particleSums) {
    if (!sum.getResult().getType().isF64())
      return sum.emitOpError()
             << "cannot differentiate a sum whose result is not f64";
    if (llvm::any_of(sum.getGathered(),
                     [&](Value input) { return isPositional(input); }))
      return sum.emitOpError()
             << "cannot differentiate a sum that reads both the positions "
                "and a field computed from them";
  }
  return success();
}

LogicalResult DerivativeBuilder::getWeight(Value sum, Value &weight) {
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
  ScalarDerivative derivative(builder, sum, leaf);
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
// Tuples
//===----------------------------------------------------------------------===//

Operation *DerivativeBuilder::createTupleOp(StringRef opName,
                                            SumTuplesOp source,
                                            Type resultType) {
  OperationState state(loc, opName);
  state.addOperands(source->getOperands());
  state.addAttributes(source->getAttrDictionary().getValue());
  state.addRegion();
  state.addTypes(resultType);

  Operation *op = builder.create(state);
  IRMapping mapping;
  source.getKernel().cloneInto(&op->getRegion(0), mapping);
  return op;
}

/// Returns the kernel arguments of `op`, an op over tuples, that hold the
/// displacements `wanted`. Those that the op does not take yet become
/// coordinates of it.
static SmallVector<Value, 3> getDisplacementArguments(Operation *op,
                                                      ArrayRef<Coordinate> wanted) {
  Block &block = op->getRegion(0).front();
  auto kinds = op->getAttrOfType<DenseI32ArrayAttr>("coordinate_kinds");
  auto members = op->getAttrOfType<DenseI64ArrayAttr>("coordinate_members");
  SmallVector<Coordinate, 2> coordinates =
      getCoordinates(kinds.asArrayRef(), members.asArrayRef());

  SmallVector<Value, 3> arguments;
  for (const Coordinate &displacement : wanted) {
    auto found = llvm::find_if(coordinates, [&](const Coordinate &known) {
      return known.kind == displacement.kind &&
             known.members == displacement.members;
    });
    unsigned index = std::distance(coordinates.begin(), found);
    if (found == coordinates.end()) {
      Builder types(op->getContext());
      coordinates.push_back(displacement);
      block.insertArgument(
          index, getCoordinateType(displacement.kind, types.getF64Type()),
          op->getLoc());
    }
    arguments.push_back(block.getArgument(index));
  }

  Builder attributes(op->getContext());
  getCoordinateAttrs(attributes, coordinates, kinds, members);
  op->setAttr("coordinate_kinds", kinds);
  op->setAttr("coordinate_members", members);
  return arguments;
}

LogicalResult DerivativeBuilder::emitTupleForces(Operation *op, Value weight,
                                                 OpBuilder &kernel,
                                                 SmallVectorImpl<Value> &forces,
                                                 Value *virial) {
  Block &block = op->getRegion(0).front();
  Value tupleEnergy = cast<YieldOp>(block.getTerminator()).getOperand(0);
  Type vectorType = VectorType::get({3}, kernel.getF64Type());

  unsigned arity =
      cast<RelationType>(op->getOperand(0).getType()).getArity();
  forces.assign(arity, Value());
  if (virial)
    *virial = Value();

  // The coordinates of the sum. The op receives more of them below.
  SmallVector<Coordinate, 2> coordinates = getCoordinates(
      op->getAttrOfType<DenseI32ArrayAttr>("coordinate_kinds").asArrayRef(),
      op->getAttrOfType<DenseI64ArrayAttr>("coordinate_members")
          .asArrayRef());

  ScalarEmitter emit(kernel, loc);
  SmallVector<Value, 9> elements(9, Value());
  // W += arm ⊗ force.
  auto addVirial = [&](Value arm, Value force) {
    if (!virial || !arm)
      return;
    SmallVector<Value, 3> armComponents, forceComponents;
    for (int64_t a = 0; a < 3; ++a) {
      armComponents.push_back(vector::ExtractOp::create(kernel, loc, arm, a));
      forceComponents.push_back(
          vector::ExtractOp::create(kernel, loc, force, a));
    }
    for (int64_t a = 0; a < 3; ++a)
      for (int64_t b = 0; b < 3; ++b)
        elements[3 * a + b] =
            emit.add(elements[3 * a + b],
                     emit.mul(armComponents[a], forceComponents[b]));
  };
  for (auto [index, coordinate] : llvm::enumerate(coordinates)) {
    if (coordinate.kind == CoordinateKind::Displacement) {
      // d = x_a − x_b: F_a = −weight · ∂u/∂d and F_b = −F_a, the gradient
      // a component at a time, each along its unit vector.
      Value d = block.getArgument(index);
      if (d.use_empty())
        continue;
      Type type = d.getType();
      Type element = cast<VectorType>(type).getElementType();
      SmallVector<Value, 3> components;
      bool any = false;
      for (int64_t c = 0; c < 3; ++c) {
        SmallVector<double, 3> unit(3, 0.0);
        unit[c] = 1.0;
        Value seed = arith::ConstantOp::create(
            kernel, loc, type,
            DenseElementsAttr::get(cast<ShapedType>(type),
                                   ArrayRef<double>(unit)));
        ScalarDerivative derivative(kernel, d, nullptr, seed);
        Value slope;
        if (failed(derivative.get(tupleEnergy, slope)))
          return failure();
        Value component = emit.neg(emit.mul(weight, slope));
        any |= static_cast<bool>(component);
        components.push_back(component ? component
                                       : emit.constant(0.0, element));
      }
      if (!any)
        continue;
      Value force = vector::FromElementsOp::create(kernel, loc, type,
                                                   components);
      int64_t a = coordinate.members[0], b = coordinate.members[1];
      forces[a] = emit.add(forces[a], force);
      forces[b] = emit.sub(forces[b], force);
      addVirial(d, force);
      continue;
    }

    // F_m = −weight · (∂u/∂q) (∂q/∂x_m)
    ScalarDerivative derivative(kernel, block.getArgument(index));
    Value slope;
    if (failed(derivative.get(tupleEnergy, slope)))
      return failure();
    Value factor = emit.neg(emit.mul(weight, slope));
    if (!factor)
      continue;

    SmallVector<Value, 3> displacements =
        getDisplacementArguments(op, getDisplacements(coordinate));
    CoordinateGradient gradient =
        emitCoordinateGradient(kernel, loc, coordinate.kind, displacements);
    Value broadcast =
        vector::BroadcastOp::create(kernel, loc, vectorType, factor);
    for (auto [place, member] : llvm::enumerate(coordinate.members)) {
      Value force = emit.mul(broadcast, gradient.members[place]);
      forces[member] = emit.add(forces[member], force);

      // W = Σ_m d_m ⊗ F_m, with the displacement of m from one of the
      // members. The sum of the forces of a coordinate is zero, so that
      // the member does not matter.
      addVirial(gradient.arms[place], force);
    }
  }

  if (virial && elements[0])
    *virial = vector::FromElementsOp::create(
        kernel, loc, VectorType::get({9}, kernel.getF64Type()), elements);
  return success();
}

//===----------------------------------------------------------------------===//
// Intermediate fields
//===----------------------------------------------------------------------===//

bool DerivativeBuilder::isPositional(Value field) {
  auto found = positional.find(field);
  if (found != positional.end())
    return found->second;
  bool result = false;
  Operation *op = field.getDefiningOp();
  if (auto gather = dyn_cast_or_null<GatherRelationOp>(op))
    result = gather.getPositions() == body->getArgument(0);
  else if (auto map = dyn_cast_or_null<MapParticlesOp>(op))
    result = llvm::any_of(map.getGathered(),
                          [&](Value input) { return isPositional(input); });
  positional[field] = result;
  return result;
}

Value DerivativeBuilder::addFields(ArrayRef<Value> fields) {
  if (fields.empty())
    return Value();
  if (fields.size() == 1)
    return fields.front();
  Type fieldType = fields.front().getType();
  Type valueType = cast<FieldType>(fieldType).getKernelValueType();
  OperationState state(loc, MapParticlesOp::getOperationName());
  state.addOperands(fields);
  state.addRegion();
  state.addTypes(fieldType);
  Operation *map = builder.create(state);
  Block *block = new Block();
  map->getRegion(0).push_back(block);
  for (size_t i = 0, e = fields.size(); i != e; ++i)
    block->addArgument(valueType, loc);
  OpBuilder kernel = OpBuilder::atBlockEnd(block);
  ScalarEmitter emit(kernel, loc);
  Value total;
  for (BlockArgument argument : block->getArguments())
    total = emit.add(total, argument);
  YieldOp::create(kernel, loc, ValueRange{total});
  return map->getResult(0);
}

/// Copies the ops of `source`, a kernel, to the end of `target` with
/// `mapping`, and returns the value that it yields there.
static Value inlineKernel(Block &source, Block &target, IRMapping &mapping) {
  OpBuilder builder = OpBuilder::atBlockEnd(&target);
  for (Operation &op : source.without_terminator())
    builder.clone(op, mapping);
  return mapping.lookupOrDefault(
      cast<YieldOp>(source.getTerminator()).getOperand(0));
}

LogicalResult DerivativeBuilder::buildAdjoints() {
  if (adjointsBuilt)
    return success();
  adjointsBuilt = true;
  // The fields in the reverse order of the body, so that the adjoint of a
  // map is there before those of the fields it reads.
  SmallVector<Operation *> producers;
  for (Operation &op : *body)
    if (isa<GatherRelationOp, MapParticlesOp>(op) &&
        isPositional(op.getResult(0)))
      producers.push_back(&op);
  for (Operation *producer : llvm::reverse(producers)) {
    Value field = producer->getResult(0);
    Type fieldType = field.getType();
    Type valueType = cast<FieldType>(fieldType).getKernelValueType();
    SmallVector<Value> parts;
    // The uses before the ops of the derivative, which read the field as
    // well, are added.
    SmallVector<std::pair<Operation *, unsigned>> uses;
    for (OpOperand &use : field.getUses())
      uses.push_back({use.getOwner(), use.getOperandNumber()});
    for (auto [user, operand] : uses) {
      if (auto sum = dyn_cast<SumRelationOp>(user)) {
        // ∂E/∂F_i = w Σ_j ∂u(i, j)/∂F_i, the kernel with i at its centre.
        unsigned place = operand - 3;
        Value weight;
        if (failed(getWeight(sum.getResult(), weight)))
          return failure();
        if (!weight)
          continue;
        Operation *op = createPairOp(GatherRelationOp::getOperationName(),
                                     sum, Exchange::None, fieldType);
        Block &block = op->getRegion(0).front();
        OpBuilder kernel(block.getTerminator());
        Value energy = cast<YieldOp>(block.getTerminator()).getOperand(0);
        ScalarDerivative derivative(kernel, block.getArgument(2 + 2 * place));
        Value slope;
        if (failed(derivative.get(energy, slope)))
          return failure();
        ScalarEmitter emit(kernel, loc);
        Value value = emit.mul(weight, slope);
        setYield(block, value ? value : emit.constant(0.0, valueType));
        parts.push_back(op->getResult(0));
        continue;
      }
      if (isa<SumParticlesOp, MapParticlesOp>(user)) {
        // ∂E/∂F_i = w ∂k/∂F_i for a sum over particles, and Ḡ_i ∂g/∂F_i for
        // a map G = g(F, ...).
        unsigned place = operand;
        Value weight;
        if (isa<SumParticlesOp>(user)) {
          if (failed(getWeight(user->getResult(0), weight)))
            return failure();
        } else {
          weight = adjoints.lookup(user->getResult(0));
        }
        if (!weight)
          continue;
        bool isField = isa<MapParticlesOp>(user);
        OperationState state(loc, MapParticlesOp::getOperationName());
        state.addOperands(user->getOperands());
        if (isField)
          state.addOperands(weight);
        state.addRegion();
        state.addTypes(fieldType);
        Operation *op = builder.create(state);
        Block *block = new Block();
        op->getRegion(0).push_back(block);
        Block &source = user->getRegion(0).front();
        IRMapping mapping;
        for (BlockArgument argument : source.getArguments())
          mapping.map(argument,
                      block->addArgument(argument.getType(), loc));
        Value adjoint;
        if (isField)
          adjoint = block->addArgument(valueType, loc);
        Value value = inlineKernel(source, *block, mapping);
        OpBuilder kernel = OpBuilder::atBlockEnd(block);
        ScalarDerivative derivative(kernel, block->getArgument(place));
        Value slope;
        if (failed(derivative.get(value, slope)))
          return failure();
        ScalarEmitter emit(kernel, loc);
        Value result = emit.mul(isField ? adjoint : weight, slope);
        YieldOp::create(kernel, loc,
                        ValueRange{result ? result
                                          : emit.constant(0.0, valueType)});
        eraseDeadOps(*block);
        parts.push_back(op->getResult(0));
        continue;
      }
      return user->emitError()
             << "cannot differentiate through this use of a field that "
                "depends on the positions";
    }
    if (Value total = addFields(parts))
      adjoints[field] = total;
  }
  return success();
}

LogicalResult DerivativeBuilder::emitGatherTerm(GatherRelationOp gather,
                                                Value adjoint, bool virial,
                                                Value &result) {
  // G_a = Σ_b k(a, b) with a at the centre gives the particle m the force
  // F_m = −Σ_b (Ḡ_m k'(m, b) + Ḡ_b k'(b, m)) d_mb / r, the kernel evaluated
  // from both ends of each pair; the factor of d is symmetric, so that the
  // force is antisymmetric, and the virial is Σ d ⊗ F over the pairs.
  Type fieldType = body->getArgument(0).getType();
  Type vectorType = cast<FieldType>(fieldType).getKernelValueType();
  Type virialType = VectorType::get({9}, builder.getF64Type());
  StringRef opName = virial ? SumRelationOp::getOperationName()
                            : GatherRelationOp::getOperationName();
  MLIRContext *context = builder.getContext();
  OperationState state(loc, opName);
  state.addOperands(
      {gather.getRelation(), gather.getPositions(), gather.getCell()});
  state.addOperands(gather.getGathered());
  state.addOperands(adjoint);
  state.addAttribute("exchange",
                     ExchangeAttr::get(context, virial ? Exchange::Symmetric
                                                       : Exchange::Antisymmetric));
  state.addAttribute("exchange_basis",
                     ExchangeBasisAttr::get(context, ExchangeBasis::Derived));
  state.addRegion();
  state.addTypes(virial ? virialType : fieldType);
  Operation *op = builder.create(state);
  Block *block = new Block();
  op->getRegion(0).push_back(block);
  Block &source = gather.getKernel().front();
  Value r = block->addArgument(source.getArgument(0).getType(), loc);
  Value d = block->addArgument(source.getArgument(1).getType(), loc);
  unsigned count = gather.getGathered().size();
  SmallVector<Value> ends;
  for (unsigned k = 0; k != 2 * count; ++k)
    ends.push_back(block->addArgument(source.getArgument(2 + k).getType(),
                                      loc));
  Type real = builder.getF64Type();
  Value adjointI = block->addArgument(real, loc);
  Value adjointJ = block->addArgument(real, loc);

  // The kernel with i at its centre, and with j.
  IRMapping forward, backward;
  forward.map(source.getArgument(0), r);
  backward.map(source.getArgument(0), r);
  forward.map(source.getArgument(1), d);
  backward.map(source.getArgument(1), d);
  for (unsigned k = 0; k != count; ++k) {
    forward.map(source.getArgument(2 + 2 * k), ends[2 * k]);
    forward.map(source.getArgument(3 + 2 * k), ends[2 * k + 1]);
    backward.map(source.getArgument(2 + 2 * k), ends[2 * k + 1]);
    backward.map(source.getArgument(3 + 2 * k), ends[2 * k]);
  }
  Value fromI = inlineKernel(source, *block, forward);
  Value fromJ = inlineKernel(source, *block, backward);
  OpBuilder kernel = OpBuilder::atBlockEnd(block);
  ScalarDerivative derivative(kernel, r);
  Value slopeI, slopeJ;
  if (failed(derivative.get(fromI, slopeI)) ||
      failed(derivative.get(fromJ, slopeJ)))
    return failure();
  ScalarEmitter emit(kernel, loc);
  Value both = emit.add(emit.mul(adjointI, slopeI), emit.mul(adjointJ, slopeJ));
  Value factor = both ? emit.neg(emit.div(both, r)) : Value();
  Value yielded;
  if (!factor) {
    yielded = emit.constant(0.0, virial ? virialType : vectorType);
  } else {
    Value force = emit.mul(
        vector::BroadcastOp::create(kernel, loc, vectorType, factor), d);
    if (!virial) {
      yielded = force;
    } else {
      SmallVector<Value, 9> elements;
      for (int64_t a = 0; a < 3; ++a)
        for (int64_t b = 0; b < 3; ++b)
          elements.push_back(
              emit.mul(vector::ExtractOp::create(kernel, loc, d, a),
                       vector::ExtractOp::create(kernel, loc, force, b)));
      yielded =
          vector::FromElementsOp::create(kernel, loc, virialType, elements);
    }
  }
  YieldOp::create(kernel, loc, ValueRange{yielded});
  eraseDeadOps(*block);
  result = op->getResult(0);
  return success();
}

//===----------------------------------------------------------------------===//
// Forces
//===----------------------------------------------------------------------===//

/// Whether `value` is the constant 1.
static bool isOne(Value value) {
  FloatAttr attr;
  return matchPattern(value, m_Constant(&attr)) &&
         attr.getValueAsDouble() == 1.0;
}

Value DerivativeBuilder::scaleField(Value field, Value weight) {
  if (isOne(weight))
    return field;
  Type vectorType = cast<FieldType>(field.getType()).getKernelValueType();
  OperationState state(loc, MapParticlesOp::getOperationName());
  state.addOperands(field);
  state.addRegion();
  state.addTypes(field.getType());
  Operation *map = builder.create(state);
  Block *block = new Block();
  map->getRegion(0).push_back(block);
  block->addArgument(vectorType, loc);
  OpBuilder kernel = OpBuilder::atBlockEnd(block);
  // The weight is a number outside the kernel; the kernel takes it as a
  // value that it captures.
  Value broadcast =
      vector::BroadcastOp::create(kernel, loc, vectorType, weight);
  Value product =
      arith::MulFOp::create(kernel, loc, broadcast, block->getArgument(0));
  YieldOp::create(kernel, loc, ValueRange{product});
  return map->getResult(0);
}

LogicalResult DerivativeBuilder::buildForces(Value &forces) {
  Type fieldType = body->getArgument(0).getType();
  Type vectorType = cast<FieldType>(fieldType).getKernelValueType();

  SmallVector<Value> terms;
  for (SumRelationOp sum : sums) {
    Value weight;
    if (failed(getWeight(sum.getResult(), weight)))
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

  for (SumTuplesOp sum : tupleSums) {
    Value weight;
    if (failed(getWeight(sum.getResult(), weight)))
      return failure();
    if (!weight)
      continue;

    // K(t)[m] = −weight · ∂u/∂x_m
    Operation *gather =
        createTupleOp(GatherTuplesOp::getOperationName(), sum, fieldType);
    Block &block = gather->getRegion(0).front();
    OpBuilder kernel(block.getTerminator());
    SmallVector<Value, 4> memberForces;
    if (failed(emitTupleForces(gather, weight, kernel, memberForces)))
      return failure();

    // A member that no coordinate names receives no force.
    ScalarEmitter emit(kernel, loc);
    Value zero;
    for (Value &force : memberForces) {
      if (force)
        continue;
      if (!zero)
        zero = emit.constant(0.0, vectorType);
      force = zero;
    }
    block.getTerminator()->setOperands(memberForces);
    eraseDeadOps(block);
    terms.push_back(gather->getResult(0));
  }

  // A reciprocal sum gives its forces, times the derivative of the energy
  // with respect to it.
  for (ReciprocalOp reciprocal : reciprocals) {
    Value weight;
    if (failed(getWeight(reciprocal.getEnergy(), weight)))
      return failure();
    if (!weight)
      continue;
    terms.push_back(scaleField(reciprocal.getForces(), weight));
  }

  // A sum S = Σ_i k(x_i, ...) of the positions themselves gives the
  // particle i the force −weight · ∇k(x_i), a component at a time along
  // its unit vector, from a map over the particles with the kernel of the
  // sum.
  for (SumParticlesOp sum : particleSums) {
    Value weight;
    if (failed(getWeight(sum.getResult(), weight)))
      return failure();
    if (!weight)
      continue;
    OperationState state(loc, MapParticlesOp::getOperationName());
    state.addOperands(sum.getGathered());
    state.addRegion();
    state.addTypes(fieldType);
    Operation *map = builder.create(state);
    Block *block = new Block();
    map->getRegion(0).push_back(block);
    Block &source = sum.getKernel().front();
    IRMapping mapping;
    for (BlockArgument argument : source.getArguments())
      mapping.map(argument, block->addArgument(argument.getType(), loc));
    Value value = inlineKernel(source, *block, mapping);
    OpBuilder kernel = OpBuilder::atBlockEnd(block);
    ScalarEmitter emit(kernel, loc);
    Value force;
    for (auto [index, input] : llvm::enumerate(sum.getGathered())) {
      if (input != body->getArgument(0))
        continue;
      Value x = block->getArgument(index);
      SmallVector<Value, 3> components;
      for (int64_t c = 0; c < 3; ++c) {
        SmallVector<double, 3> unit(3, 0.0);
        unit[c] = 1.0;
        Value seed = arith::ConstantOp::create(
            kernel, loc, vectorType,
            DenseElementsAttr::get(cast<ShapedType>(vectorType),
                                   ArrayRef<double>(unit)));
        ScalarDerivative derivative(kernel, x, nullptr, seed);
        Value slope;
        if (failed(derivative.get(value, slope)))
          return failure();
        Value component = emit.neg(emit.mul(weight, slope));
        components.push_back(component
                                 ? component
                                 : emit.constant(0.0, kernel.getF64Type()));
      }
      force = emit.add(force, vector::FromElementsOp::create(
                                  kernel, loc, vectorType, components));
    }
    YieldOp::create(kernel, loc,
                    ValueRange{force ? force
                                     : emit.constant(0.0, vectorType)});
    eraseDeadOps(*block);
    terms.push_back(map->getResult(0));
  }

  // The forces through the fields that depend on the positions.
  if (failed(buildAdjoints()))
    return failure();
  SmallVector<GatherRelationOp> gathers;
  for (Operation &op : *body)
    if (auto gather = dyn_cast<GatherRelationOp>(&op))
      if (isPositional(gather.getResult()) &&
          adjoints.count(gather.getResult()))
        gathers.push_back(gather);
  for (GatherRelationOp gather : gathers) {
    Value adjoint = adjoints.lookup(gather.getResult());
    if (!adjoint)
      continue;
    Value term;
    if (failed(emitGatherTerm(gather, adjoint, /*virial=*/false, term)))
      return failure();
    terms.push_back(term);
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
    if (failed(getWeight(sum.getResult(), weight)))
      return failure();
    if (!weight)
      continue;

    // W = Σ d ⊗ K(i, j), with K(i, j) = factor · d: the virial of pairs in
    // a periodic system [Louwerse2006].
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
      // d ⊗ (factor d): the vector factor d is that of the forces, which a
      // loop that fuses the two computes once; factor (d_a d_b) would have
      // the factor, a sum of terms, multiplied into each element.
      Value broadcast = vector::BroadcastOp::create(
          kernel, loc, d.getType(), factor);
      Value scaled = emit.mul(broadcast, d);
      SmallVector<Value, 9> elements;
      for (int64_t a = 0; a < 3; ++a)
        for (int64_t b = 0; b < 3; ++b)
          elements.push_back(emit.mul(
              components[a],
              vector::ExtractOp::create(kernel, loc, scaled, b)));
      contribution =
          vector::FromElementsOp::create(kernel, loc, virialType, elements);
    } else {
      contribution = emit.constant(0.0, virialType);
    }
    setYield(block, contribution);
    total = outer.add(total, term->getResult(0));
  }

  for (SumTuplesOp sum : tupleSums) {
    Value weight;
    if (failed(getWeight(sum.getResult(), weight)))
      return failure();
    if (!weight)
      continue;

    Operation *term =
        createTupleOp(SumTuplesOp::getOperationName(), sum, virialType);
    Block &block = term->getRegion(0).front();
    OpBuilder kernel(block.getTerminator());
    SmallVector<Value, 4> memberForces;
    Value contribution;
    if (failed(emitTupleForces(term, weight, kernel, memberForces,
                               &contribution)))
      return failure();

    ScalarEmitter emit(kernel, loc);
    if (!contribution)
      contribution = emit.constant(0.0, virialType);
    setYield(block, contribution);
    total = outer.add(total, term->getResult(0));
  }

  for (ReciprocalOp reciprocal : reciprocals) {
    Value weight;
    if (failed(getWeight(reciprocal.getEnergy(), weight)))
      return failure();
    if (!weight)
      continue;
    Value term = reciprocal.getVirial();
    if (!isOne(weight)) {
      Value broadcast =
          vector::BroadcastOp::create(builder, loc, virialType, weight);
      term = outer.mul(broadcast, term);
    }
    total = outer.add(total, term);
  }

  if (failed(buildAdjoints()))
    return failure();
  SmallVector<GatherRelationOp> gathers;
  for (Operation &op : *body)
    if (auto gather = dyn_cast<GatherRelationOp>(&op))
      if (isPositional(gather.getResult()) &&
          adjoints.count(gather.getResult()))
        gathers.push_back(gather);
  for (GatherRelationOp gather : gathers) {
    Value adjoint = adjoints.lookup(gather.getResult());
    if (!adjoint)
      continue;
    Value term;
    if (failed(emitGatherTerm(gather, adjoint, /*virial=*/true, term)))
      return failure();
    total = outer.add(total, term);
  }

  // A sum S = Σ_i k(x_i, L) of the positions themselves, and of the edges L
  // of the cell where the kernel takes them from md_exec.cell_edges: the
  // virial is −dU/dε when the positions and the cell scale by 1 + ε about
  // the origin, as a barostat scales them (D154), Σ_i x_i ⊗ F_i with
  // F_i = −weight · ∇k(x_i) and, on the diagonal, −weight Σ_i ∂k/∂L_a L_a.
  // A term whose frame scales with the cell, k(x ⊙ L_0 / L), has none.
  SmallVector<Value> cellEdges;
  for (Operation &op : *body)
    if (op.getName().getStringRef() == "md_exec.cell_edges" &&
        op.getOperand(0) == body->getArgument(1))
      cellEdges.push_back(op.getResult(0));
  for (SumParticlesOp sum : particleSums) {
    Value weight;
    if (failed(getWeight(sum.getResult(), weight)))
      return failure();
    if (!weight)
      continue;
    Type vectorType = VectorType::get({3}, builder.getF64Type());
    OperationState state(loc, SumParticlesOp::getOperationName());
    state.addOperands(sum.getGathered());
    state.addRegion();
    state.addTypes(virialType);
    Operation *term = builder.create(state);
    Block *block = new Block();
    term->getRegion(0).push_back(block);
    Block &source = sum.getKernel().front();
    IRMapping mapping;
    for (BlockArgument argument : source.getArguments())
      mapping.map(argument, block->addArgument(argument.getType(), loc));
    Value value = inlineKernel(source, *block, mapping);
    OpBuilder kernel = OpBuilder::atBlockEnd(block);
    ScalarEmitter emit(kernel, loc);
    auto unit = [&](int64_t c) {
      SmallVector<double, 3> components(3, 0.0);
      components[c] = 1.0;
      return arith::ConstantOp::create(
          kernel, loc, vectorType,
          DenseElementsAttr::get(cast<ShapedType>(vectorType),
                                 ArrayRef<double>(components)));
    };
    SmallVector<Value, 9> elements(9, Value());
    for (auto [index, input] : llvm::enumerate(sum.getGathered())) {
      if (input != body->getArgument(0))
        continue;
      Value x = block->getArgument(index);
      SmallVector<Value, 3> positions, forces;
      for (int64_t c = 0; c < 3; ++c) {
        ScalarDerivative derivative(kernel, x, nullptr, unit(c));
        Value slope;
        if (failed(derivative.get(value, slope)))
          return failure();
        positions.push_back(vector::ExtractOp::create(kernel, loc, x, c));
        forces.push_back(emit.neg(emit.mul(weight, slope)));
      }
      for (int64_t a = 0; a < 3; ++a)
        for (int64_t b = 0; b < 3; ++b)
          elements[3 * a + b] = emit.add(elements[3 * a + b],
                                         emit.mul(positions[a], forces[b]));
    }
    for (Value edges : cellEdges)
      for (int64_t a = 0; a < 3; ++a) {
        ScalarDerivative derivative(kernel, edges, nullptr, unit(a));
        Value slope;
        if (failed(derivative.get(value, slope)))
          return failure();
        Value edge = vector::ExtractOp::create(kernel, loc, edges, a);
        elements[4 * a] = emit.sub(elements[4 * a],
                                   emit.mul(weight, emit.mul(slope, edge)));
      }
    for (Value &element : elements)
      if (!element)
        element = emit.constant(0.0, kernel.getF64Type());
    Value contribution =
        vector::FromElementsOp::create(kernel, loc, virialType, elements);
    YieldOp::create(kernel, loc, ValueRange{contribution});
    eraseDeadOps(*block);
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

  // Each value is one of three: independent of the parameter, which only a
  // proof gives, then its derivative is exactly zero; dependent on it, then
  // a rule gives its derivative or the derivative fails; or undetermined,
  // which fails as well (D161). The proof follows every path by which the
  // parameter can reach a value: the operands of ops, the values that their
  // kernels take from outside, and the arguments of blocks whose meaning is
  // known: those of the body, the other arguments of the potential, and
  // those of a kernel, the values of its particles, tuples, or pairs, whose
  // fields are checked apart.
  enum class Dependence { Independent, Dependent, Undetermined };
  struct Verdict {
    Dependence dependence = Dependence::Independent;
    std::string reason;
  };
  auto combine = [](Verdict &into, const Verdict &from) {
    if (from.dependence == Dependence::Dependent ||
        into.dependence == Dependence::Dependent) {
      into = {Dependence::Dependent, ""};
      return;
    }
    if (from.dependence == Dependence::Undetermined &&
        into.dependence == Dependence::Independent)
      into = from;
  };
  llvm::DenseSet<Block *> knownBlocks = {body};
  llvm::DenseMap<Value, Verdict> verdicts;
  // Ops of a kernel whose meaning the pass knows: arithmetic without
  // effects, and the values of tables.
  auto isKernelOp = [](Operation *op) {
    if (isa<LookupOp, YieldOp>(op) ||
        op->getName().getStringRef() == "md_exec.cell_edges")
      return true;
    StringRef dialect = op->getName().getDialectNamespace();
    return (dialect == "arith" || dialect == "math" || dialect == "vector") &&
           op->getNumRegions() == 0 && isMemoryEffectFree(op);
  };
  // Ops over particles, tuples, or pairs whose kernels the pass knows.
  auto isSumOp = [](Operation *op) {
    return isa<SumRelationOp, GatherRelationOp, SumTuplesOp, GatherTuplesOp,
               SumParticlesOp, MapParticlesOp>(op);
  };
  std::function<Verdict(Value)> classify = [&](Value value) -> Verdict {
    if (value == parameter)
      return {Dependence::Dependent, ""};
    auto found = verdicts.find(value);
    if (found != verdicts.end())
      return found->second;
    Verdict verdict;
    Operation *op = value.getDefiningOp();
    if (!op) {
      Block *owner = cast<BlockArgument>(value).getOwner();
      if (!knownBlocks.contains(owner))
        verdict = {Dependence::Undetermined,
                   "it is an argument of a block whose meaning the pass does "
                   "not know"};
    } else {
      bool known = isKernelOp(op) || isSumOp(op) ||
                   (op->getName().getDialectNamespace() == "md" &&
                    op->getNumRegions() == 0 && isMemoryEffectFree(op));
      // An operand that depends on the parameter makes the op dependent,
      // known or not.
      for (Value operand : op->getOperands())
        combine(verdict, classify(operand));
      if (!known) {
        combine(verdict,
                {Dependence::Undetermined,
                 ("'" + op->getName().getStringRef() +
                  "' is not an op whose dependences the pass knows")
                     .str()});
      } else {
        // What the kernels take from outside.
        op->walk([&](Operation *inner) {
          if (inner == op)
            return WalkResult::advance();
          if (!isKernelOp(inner)) {
            combine(verdict,
                    {Dependence::Undetermined,
                     ("its kernel holds '" + inner->getName().getStringRef() +
                      "', whose dependences the pass does not know")
                         .str()});
            return WalkResult::advance();
          }
          for (Value operand : inner->getOperands()) {
            Operation *definition = operand.getDefiningOp();
            bool outside =
                definition ? !op->isAncestor(definition)
                           : !op->isAncestor(
                                 cast<BlockArgument>(operand).getOwner()
                                     ->getParentOp());
            if (outside)
              combine(verdict, classify(operand));
          }
          return WalkResult::advance();
        });
      }
    }
    verdicts[value] = verdict;
    return verdict;
  };

  auto describe = [&]() {
    return "argument " + std::to_string(argument) + " of '" +
           potential.getSymName().str() + "'";
  };
  // An error for a value that is not independent and has no rule here.
  auto refuse = [&](Value value, const Verdict &verdict) -> LogicalResult {
    Operation *op = value.getDefiningOp();
    Location at = op ? op->getLoc() : value.getLoc();
    if (verdict.dependence == Dependence::Undetermined)
      return emitError(at) << "cannot prove that this value does not depend on "
                           << describe() << ": " << verdict.reason;
    if (op)
      return emitError(at) << "'" << op->getName() << "' depends on "
                           << describe()
                           << " and has no rule for its derivative";
    return emitError(at) << "a block argument depends on " << describe()
                         << " and has no rule for its derivative";
  };
  auto takeAsZero = [&](Value value) {
    if (Operation *op = value.getDefiningOp(); op && remarks)
      op->emitRemark() << "independent of " << describe()
                       << ": its derivative is zero";
  };

  // Within a kernel: the arguments of the kernel and what it computes from
  // them are independent of the parameter; an op without a rule fails as
  // the leaf above says.
  ScalarDerivative::LeafHandler kernelLeaf = [&](Value value,
                                                 Value &tangent) {
    tangent = Value();
    Verdict verdict = classify(value);
    if (verdict.dependence == Dependence::Independent)
      return success();
    return refuse(value, verdict);
  };

  // A field that a map over particles computes, with its derivative with
  // respect to the parameter times `sign` added: the same map, whose
  // kernel yields k + sign · ∂k/∂θ.
  auto shiftField = [&](Value field, double sign, Value &shifted)
      -> LogicalResult {
    auto map = field.getDefiningOp<MapParticlesOp>();
    if (!map)
      return refuse(field, classify(field).dependence ==
                                   Dependence::Undetermined
                               ? classify(field)
                               : Verdict{Dependence::Dependent, ""});
    for (Value gathered : map.getGathered()) {
      Verdict verdict = classify(gathered);
      if (verdict.dependence != Dependence::Independent)
        return refuse(gathered, verdict);
    }
    Operation *copy = builder.clone(*map);
    Block &block = copy->getRegion(0).front();
    knownBlocks.insert(&block);
    Value value = cast<YieldOp>(block.getTerminator()).getOperand(0);
    OpBuilder kernel(block.getTerminator());
    ScalarDerivative derivative(kernel, parameter, kernelLeaf);
    Value slope;
    if (failed(derivative.get(value, slope)))
      return failure();
    ScalarEmitter emit(kernel, loc);
    setYield(block, emit.add(value, emit.scale(sign, slope)));
    shifted = copy->getResult(0);
    return success();
  };

  // The derivative of a sum over a relation, tuples, or particles is the
  // sum of the derivative of its kernel. That of a reciprocal sum, whose
  // energy E(c) = ½ cᵀ A c is a quadratic form of the charges, is
  // Δᵀ A c = (E(c + Δ) − E(c − Δ)) / 2 with Δ = ∂c/∂θ: two reciprocal
  // sums (D161).
  auto leaf = [&](Value value, Value &tangent) -> LogicalResult {
    tangent = Value();
    Verdict verdict = classify(value);
    if (verdict.dependence == Dependence::Independent) {
      takeAsZero(value);
      return success();
    }
    if (verdict.dependence == Dependence::Undetermined)
      return refuse(value, verdict);
    Operation *op = value.getDefiningOp();
    if (!op)
      return refuse(value, verdict);

    if (auto reciprocal = dyn_cast<ReciprocalOp>(op)) {
      if (value != reciprocal.getEnergy())
        return op->emitError() << "cannot differentiate the forces or the "
                                  "virial of a reciprocal sum with respect "
                                  "to a parameter";
      Verdict positions = classify(reciprocal.getPositions());
      if (positions.dependence != Dependence::Independent)
        return refuse(reciprocal.getPositions(), positions);
      Value energies[2];
      for (int k = 0; k != 2; ++k) {
        Value charges;
        if (failed(shiftField(reciprocal.getCharges(), k == 0 ? 1.0 : -1.0,
                              charges)))
          return failure();
        auto copy = cast<ReciprocalOp>(builder.clone(*reciprocal));
        copy.getChargesMutable().assign(charges);
        energies[k] = copy.getEnergy();
      }
      ScalarEmitter emit(builder, loc);
      tangent = emit.scale(0.5, emit.sub(energies[0], energies[1]));
      return success();
    }

    // The derivative of a sum takes its kernel's; the fields that the
    // kernel takes must not depend on the parameter, for which there is no
    // rule.
    if (!isa<SumRelationOp, SumTuplesOp, SumParticlesOp>(op))
      return refuse(value, verdict);
    for (Value operand : op->getOperands()) {
      if (operand == parameter)
        continue;
      Verdict field = classify(operand);
      if (field.dependence == Dependence::Dependent)
        return op->emitError()
               << "'" << op->getName() << "' takes a field that depends on "
               << describe() << ", and has no rule for that derivative";
      if (field.dependence == Dependence::Undetermined)
        return refuse(operand, field);
    }
    Operation *term;
    if (auto sum = dyn_cast<SumRelationOp>(op))
      term = createPairOp(SumRelationOp::getOperationName(), sum,
                          Exchange::Symmetric, value.getType());
    else if (auto tuples = dyn_cast<SumTuplesOp>(op))
      term = createTupleOp(SumTuplesOp::getOperationName(), tuples,
                           value.getType());
    else
      term = builder.clone(*op);
    Block &block = term->getRegion(0).front();
    knownBlocks.insert(&block);
    Value pairEnergy = cast<YieldOp>(block.getTerminator()).getOperand(0);

    OpBuilder kernel(block.getTerminator());
    ScalarDerivative derivative(kernel, parameter, kernelLeaf);
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
    if (auto tuples = dyn_cast<SumTuplesOp>(&op))
      tupleSums.push_back(tuples);
    if (auto reciprocal = dyn_cast<ReciprocalOp>(&op))
      reciprocals.push_back(reciprocal);
    if (auto particles = dyn_cast<SumParticlesOp>(&op))
      if (llvm::is_contained(particles.getGathered(), body->getArgument(0)))
        particleSums.push_back(particles);
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
  // The adjoints of the fields that depend on the positions, from the uses
  // of the potential alone, before the derivatives add their own.
  if (needsPositions && failed(buildAdjoints()))
    return fail();
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
        DerivativeBuilder derivative(potential, name, remarks);
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
