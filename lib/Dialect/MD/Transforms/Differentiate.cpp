// Semantic differentiation of potentials.
//
// See docs/ops-m0.md, Section 5, and docs/design-m1.md, Section 4.

#include "mdir/Dialect/MD/Transforms/Passes.h"
#include "mdir/Dialect/MD/Transforms/ScalarDerivative.h"
#include "mdir/Dialect/MD/Transforms/Activity.h"
#include "mdir/Dialect/MD/Transforms/Truncation.h"
#include "mdir/Dialect/MD/Transforms/AdjointInterface.h"
#include "mdir/Dialect/MDDist/MDDistDialect.h"
#include "mdir/Dialect/MDDist/MDDistOps.h"

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
namespace md_dist = mdir::md_dist;

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
  bool activityFailed = false;
  LogicalResult buildParameterDerivative(int64_t argument, Value &result);
  /// The derivative of the energy in the value of each particle of the
  /// field that is argument `argument` of the potential.
  LogicalResult buildFieldDerivative(int64_t argument, Value &result);

  //===--------------------------------------------------------------------===//
  // Potentials on domains (D[ml2-distribute])
  //===--------------------------------------------------------------------===//
  //
  // A potential that md-distribute rewrote, or one written in md_dist,
  // computes its energy from sums over the rows of domains and their
  // all_reduce. Its derivative in the positions is carried back from the
  // sums through the transfers by their declared transposes
  // (AdjointOpInterface): the adjoint of the ghost rows that a pair op
  // reads returns to the owners by G^T, and that of the owned rows to the
  // particle set by md_dist.collect.

  /// The positions are read only by md_dist.partition and distribute, and
  /// the sums are those whose derivatives are known.
  LogicalResult checkDistributedUses();
  /// Sets `weight` to the derivative of the energy in each domain's part of
  /// the partial sum `partial`, through its all_reduce.
  LogicalResult getPartialWeight(Value partial, Value &weight);
  /// A pair op over the relation and the rows of `source` with a copy of
  /// its kernel: a gather with partial results at owned and at ghost rows,
  /// or a sum of `valueType`.
  Operation *createDistributedPairOp(md_dist::SumPairsOp source, bool gather,
                                     Exchange exchange, Type valueType);
  /// The derivative of the energy in the field `input` of a sum over rows,
  /// times −weight: a map over the same rows.
  LogicalResult emitRowForces(md_dist::SumParticlesOp sum, unsigned place,
                              Value weight, Value &result);
  LogicalResult buildDistributedForces(Value &forces);
  LogicalResult buildDistributedVirial(Value &virial);

  // Fields on domains computed from the positions (D[ml2-eam]). The terms
  // of `distTerms` are negated adjoints, −∂U/∂v, of the values v on domains
  // and of the positions: forces where v is the positions, and the forces
  // on a field elsewhere. A term is complete, or one of the two partial
  // results of a traversal, which only md_dist.halo_sum adds.

  /// A gather of a field that the energy depends on: the negated adjoint
  /// of its result at the owned rows and its copy to the ghosts.
  struct Density {
    md_dist::GatherPairsOp gather;
    Value force;
    Value ghosts;
  };
  /// Fills `distTerms` and `densities`, in the reverse order of the body.
  /// With `wantForces`, the terms of the positions as well.
  LogicalResult buildDistributedTerms();
  /// Carries the terms of the result of `transfer` to its operands by its
  /// declared transpose.
  LogicalResult transposeTransfer(Operation *transfer);
  /// The complete terms of `list` added into one, the partial ones kept.
  void addCompleteTerms(SmallVectorImpl<Value> &list);
  /// `fields`, complete and of one type on domains, added row by row.
  Value addDistributedFields(ArrayRef<Value> fields);
  /// `scale · ∂value/∂variable` in a kernel, negated with `negate`; for a
  /// vector variable a component at a time, along its unit vectors.
  LogicalResult emitScaledSlope(OpBuilder &kernel, Value value,
                                Value variable, Value scale, bool negate,
                                Value &result);
  /// The forces on the field gathered at place `place` of the pair sum
  /// `sum`, a directed traversal, computed at the owners (Section 15.4).
  LogicalResult emitPairFieldTerm(md_dist::SumPairsOp sum, unsigned place,
                                  Value weight, Value &result);
  /// The forces on the field at place `place` of the map `map` from the
  /// forces `outer` on its result.
  LogicalResult emitMapTerm(md_dist::MapParticlesOp map, unsigned place,
                            Value outer, Value &result);
  /// The pair op of the forces (a two-ended gather) or of the virial (a sum
  /// of d ⊗ K) that the density `density` gives.
  LogicalResult emitDensityOp(const Density &density, bool virial,
                              Operation *&result);
  /// Whether `value` holds positions, or values moved from them by
  /// transfers alone: a term of the absolute positions, not of pairs.
  bool isPositionValued(Value value);

  bool distributed = false;
  bool wantForces = false;
  bool distTermsBuilt = false;
  llvm::DenseMap<Value, SmallVector<Value, 2>> distTerms;
  SmallVector<Density> densities;
  SmallVector<md_dist::SumPairsOp> pairSums;
  SmallVector<md_dist::SumParticlesOp> rowSums;

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
  /// The ops of the body of the potential, in order, before any derivative
  /// added its own. Every other op of the body was added by a derivative
  /// (the forces, the virial, the derivative in a number or in a field): a
  /// derivative in a field follows the uses of the field by the potential,
  /// not those by the derivatives requested before it, which are first
  /// derivatives of the same energy and enter none of its values.
  SmallVector<Operation *> potentialOps;
};

} // namespace

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Replaces the value that the kernel in `block` yields, then removes what
/// the kernel no longer needs.
static void setYield(Block &block, Value value) {
  // The terminator of a kernel of md or of md_dist.
  block.getTerminator()->setOperand(0, value);
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
                     [&](Value input) { return input != positions && isPositional(input); }))
      return sum.emitOpError()
             << "cannot differentiate a sum that reads both the positions "
                "and a field computed from them";
  }
  return success();
}

LogicalResult DerivativeBuilder::getWeight(Value sum, Value &weight) {
  // With respect to one sum, everything that does not come from a scalar op
  // is an independent input.
  ActivityAnalysis activity(sum);
  activity.addKnownBlock(body);
  auto leaf = [&](Value value, Value &tangent) -> LogicalResult {
    tangent = Value();
    auto verdict = activity.classify(value);
    if (verdict.dependence == Activity::Inactive)
      return success();
    Operation *op = value.getDefiningOp();
    Location at = verdict.unknownOperation ? verdict.unknownOperation->getLoc() : value.getLoc();
    return emitError(at) << "no derivative rule for '"
        << (op ? op->getName().getStringRef() : "block argument")
        << "' in potential '" << potential.getSymName() << "'"
        << (verdict.reason.empty() ? "" : ": " + verdict.reason);
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
  Value pairEnergy = block.getTerminator()->getOperand(0);
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
  if (found != positional.end()) return found->second;
  ActivityAnalysis activity(body->getArgument(0));
  activity.addKnownBlock(body);
  auto verdict = activity.classify(field);
  if (verdict.dependence == Activity::Unknown) {
    Location at = verdict.unknownOperation ? verdict.unknownOperation->getLoc() : field.getLoc();
    emitError(at) << "cannot prove positional inactivity in potential '"
                             << potential.getSymName() << "': " << verdict.reason;
    activityFailed = true;
  }
  bool result = verdict.dependence == Activity::Active;
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
  return mapping.lookupOrDefault(source.getTerminator()->getOperand(0));
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
// Potentials on domains (D[ml2-distribute])
//===----------------------------------------------------------------------===//

LogicalResult DerivativeBuilder::checkDistributedUses() {
  Value positions = body->getArgument(0);
  for (OpOperand &use : positions.getUses()) {
    Operation *user = use.getOwner();
    if (isa<md_dist::PartitionOp, md_dist::DistributeOp>(user) &&
        use.getOperandNumber() == 0)
      continue;
    return user->emitError()
           << "cannot differentiate with respect to the positions of a "
              "potential on domains through this use; only "
              "md_dist.partition and md_dist.distribute read them";
  }
  if (!sums.empty() || !tupleSums.empty() || !particleSums.empty() ||
      !reciprocals.empty())
    return potential.emitOpError()
           << "cannot differentiate a potential that sums both over the "
              "particle set and over domains";
  for (md_dist::SumPairsOp sum : pairSums) {
    Block &kernel = sum.getKernel().front();
    if (!kernel.getArgument(1).use_empty())
      return sum.emitOpError()
             << "cannot differentiate a kernel that uses the displacement; "
                "only kernels that depend on the distance are supported";
    if (!sum.getResult().getType().getValueType().isF64())
      return sum.emitOpError()
             << "cannot differentiate a sum whose result is not f64";
  }
  for (md_dist::SumParticlesOp sum : rowSums)
    if (!sum.getResult().getType().getValueType().isF64())
      return sum.emitOpError()
             << "cannot differentiate a sum whose result is not f64";
  return success();
}

LogicalResult DerivativeBuilder::getPartialWeight(Value partial,
                                                  Value &weight) {
  weight = Value();
  if (!partial.hasOneUse())
    return partial.getDefiningOp()->emitOpError()
           << "cannot differentiate a partial sum without its one "
              "md_dist.all_reduce";
  Operation *user = *partial.user_begin();
  auto reduce = dyn_cast<AdjointOpInterface>(user);
  if (!reduce || !isa<md_dist::AllReduceOp>(user))
    return user->emitOpError()
           << "cannot differentiate through this use of a partial sum";
  Value total;
  if (failed(getWeight(user->getResult(0), total)))
    return failure();
  if (!total)
    return success();
  // The rule of md_dist.all_reduce: each domain's part takes the cotangent
  // of the sum, once.
  weight = reduce.emitTranspose(builder, 0, total, Value());
  return success(static_cast<bool>(weight));
}

Operation *DerivativeBuilder::createDistributedPairOp(
    md_dist::SumPairsOp source, bool gather, Exchange exchange,
    Type valueType) {
  MLIRContext *context = builder.getContext();
  OperationState state(loc, gather
                                ? md_dist::GatherPairsOp::getOperationName()
                                : md_dist::SumPairsOp::getOperationName());
  state.addOperands(source->getOperands());
  state.addAttribute("exchange", ExchangeAttr::get(context, exchange));
  if (gather) {
    // The contributions at both particles of each pair: partial at owned
    // and at ghost rows, completed by the transpose of the copy of the
    // positions to the ghosts (G^T).
    auto rows = cast<md_dist::FieldType>(source.getPositions().getType());
    state.addTypes({rows.with(md_dist::Role::Owned, true),
                    rows.with(md_dist::Role::Ghost, true)});
  } else {
    state.addTypes(md_dist::PartialType::get(context, valueType));
  }
  state.addRegion();
  Operation *op = builder.create(state);
  IRMapping mapping;
  source.getKernel().cloneInto(&op->getRegion(0), mapping);
  return op;
}

LogicalResult DerivativeBuilder::emitScaledSlope(OpBuilder &kernel,
                                                Value value, Value variable,
                                                Value scale, bool negate,
                                                Value &result) {
  ScalarEmitter emit(kernel, loc);
  auto scaled = [&](Value slope) -> Value {
    Value product = emit.mul(scale, slope);
    return negate ? emit.neg(product) : product;
  };
  if (auto vector = dyn_cast<VectorType>(variable.getType())) {
    // A component at a time, along its unit vector.
    SmallVector<Value, 3> components;
    for (int64_t c = 0; c < vector.getNumElements(); ++c) {
      SmallVector<double, 3> unit(vector.getNumElements(), 0.0);
      unit[c] = 1.0;
      Value seed = arith::ConstantOp::create(
          kernel, loc, vector,
          DenseElementsAttr::get(cast<ShapedType>(vector),
                                 ArrayRef<double>(unit)));
      ScalarDerivative derivative(kernel, variable, nullptr, seed);
      Value slope;
      if (failed(derivative.get(value, slope)))
        return failure();
      Value component = scaled(slope);
      components.push_back(
          component ? component
                    : emit.constant(0.0, vector.getElementType()));
    }
    result = vector::FromElementsOp::create(kernel, loc, vector, components);
    return success();
  }
  ScalarDerivative derivative(kernel, variable);
  Value slope;
  if (failed(derivative.get(value, slope)))
    return failure();
  result = scaled(slope);
  if (!result)
    result = emit.constant(0.0, variable.getType());
  return success();
}

LogicalResult DerivativeBuilder::emitRowForces(md_dist::SumParticlesOp sum,
                                               unsigned place, Value weight,
                                               Value &result) {
  Value input = sum.getGathered()[place];
  auto type = cast<md_dist::FieldType>(input.getType());
  OperationState state(loc, md_dist::MapParticlesOp::getOperationName());
  state.addOperands(sum.getGathered());
  state.addRegion();
  state.addTypes(type.with(type.getRole(), false));
  Operation *map = builder.create(state);
  Block *block = new Block();
  map->getRegion(0).push_back(block);
  Block &source = sum.getKernel().front();
  IRMapping mapping;
  for (BlockArgument argument : source.getArguments())
    mapping.map(argument, block->addArgument(argument.getType(), loc));
  Value value = inlineKernel(source, *block, mapping);
  OpBuilder kernel = OpBuilder::atBlockEnd(block);
  Value force;
  if (failed(emitScaledSlope(kernel, value, block->getArgument(place), weight,
                             /*negate=*/true, force)))
    return failure();
  md_dist::YieldOp::create(kernel, loc, ValueRange{force});
  eraseDeadOps(*block);
  result = map->getResult(0);
  return success();
}

LogicalResult DerivativeBuilder::emitMapTerm(md_dist::MapParticlesOp map,
                                             unsigned place, Value outer,
                                             Value &result) {
  Value input = map.getGathered()[place];
  auto type = cast<md_dist::FieldType>(input.getType());
  if (!map.getResult().getType().getKernelValueType().isF64())
    return map.emitOpError()
           << "cannot differentiate on domains through a map whose value "
              "is not f64";
  // F_a = F_b ∂g/∂a for b = g(a, ...): the forces on the result, a field
  // the map reads beside its operands, carried back by the chain rule.
  OperationState state(loc, md_dist::MapParticlesOp::getOperationName());
  state.addOperands(map.getGathered());
  state.addOperands(outer);
  state.addRegion();
  state.addTypes(type.with(type.getRole(), false));
  Operation *op = builder.create(state);
  Block *block = new Block();
  op->getRegion(0).push_back(block);
  Block &source = map.getKernel().front();
  IRMapping mapping;
  for (BlockArgument argument : source.getArguments())
    mapping.map(argument, block->addArgument(argument.getType(), loc));
  Value scale = block->addArgument(
      cast<md_dist::FieldType>(outer.getType()).getKernelValueType(), loc);
  Value value = inlineKernel(source, *block, mapping);
  OpBuilder kernel = OpBuilder::atBlockEnd(block);
  Value term;
  if (failed(emitScaledSlope(kernel, value, block->getArgument(place), scale,
                             /*negate=*/false, term)))
    return failure();
  md_dist::YieldOp::create(kernel, loc, ValueRange{term});
  eraseDeadOps(*block);
  result = op->getResult(0);
  return success();
}

LogicalResult DerivativeBuilder::emitPairFieldTerm(md_dist::SumPairsOp sum,
                                                   unsigned place,
                                                   Value weight,
                                                   Value &result) {
  // Section 15.4: over a directed relation that covers its cutoff, the
  // derivative of a symmetric pair sum in the value a_i, through its own
  // row and every copy of it to the ghosts, is Σ_{k : (i,k) ∈ D} ∂₁k(i, k),
  // the gather at the owners with nothing returned. The premises, with the
  // code of the verifier.
  if (sum.getRelation().getType().getTraversal() !=
      md_dist::Traversal::Directed)
    return sum.emitOpError()
           << "cannot differentiate on the unique traversal a sum over pairs "
              "that reads a field computed from the positions: the "
              "derivative at the second particle of a pair has no exchange "
              "contract (D[ml2-eam], docs/distributed-ml2.md, Section 15.4); "
              "use the directed traversal";
  auto neighborhood =
      sum.getRelation().getDefiningOp<md_dist::NeighborhoodOp>();
  if (!neighborhood)
    return sum.emitOpError()
           << "cannot show that every pair is at both of its owners: the "
              "relation does not come from md_dist.neighborhood";
  if (failed(md_dist::verifyCoverage(sum, neighborhood)) ||
      failed(md_dist::proveContract(sum, sum.getKernel().front(),
                                    Exchange::Symmetric)))
    return failure();
  auto type = cast<md_dist::FieldType>(sum.getGathered()[place].getType());
  if (type.isGhost() || type.getPartial())
    return sum.emitOpError() << "expected the owned values of a field at an "
                                "even place of 'gather'";
  MLIRContext *context = builder.getContext();
  OperationState state(loc, md_dist::GatherPairsOp::getOperationName());
  state.addOperands(sum->getOperands());
  state.addAttribute("exchange", ExchangeAttr::get(context, Exchange::None));
  state.addTypes(type.with(md_dist::Role::Owned, false));
  state.addRegion();
  Operation *gather = builder.create(state);
  IRMapping mapping;
  sum.getKernel().cloneInto(&gather->getRegion(0), mapping);
  Block &block = gather->getRegion(0).front();
  OpBuilder kernel(block.getTerminator());
  Value value = block.getTerminator()->getOperand(0);
  // The center's value of the field: the kernel takes the distance, the
  // displacement, and two values of each gathered field.
  Value term;
  if (failed(emitScaledSlope(kernel, value, block.getArgument(2 + place),
                             weight, /*negate=*/true, term)))
    return failure();
  setYield(block, term);
  result = gather->getResult(0);
  return success();
}

LogicalResult DerivativeBuilder::emitDensityOp(const Density &density,
                                               bool virial,
                                               Operation *&result) {
  // ρ_i = Σ_k f(i, k) gives the particle i the force
  // K(i, k) = (F_i f'(i, k) + F_k f'(k, i)) d / r from each pair, with F the
  // forces on ρ, −ρ̄: the kernel evaluated from both ends, so that K is
  // antisymmetric by construction (Section 15.2). The virial is Σ d ⊗ K.
  md_dist::GatherPairsOp gather = density.gather;
  Type vectorType = VectorType::get({3}, builder.getF64Type());
  Type virialType = VectorType::get({9}, builder.getF64Type());
  MLIRContext *context = builder.getContext();
  OperationState state(loc, virial ? md_dist::SumPairsOp::getOperationName()
                                   : md_dist::GatherPairsOp::getOperationName());
  state.addOperands(
      {gather.getRelation(), gather.getPositions(), gather.getGhosts()});
  state.addOperands(gather.getGathered());
  state.addOperands({density.force, density.ghosts});
  state.addAttribute("exchange",
                     ExchangeAttr::get(context, virial ? Exchange::Symmetric
                                                       : Exchange::Antisymmetric));
  if (virial) {
    state.addTypes(md_dist::PartialType::get(context, virialType));
  } else {
    auto rows = cast<md_dist::FieldType>(gather.getPositions().getType());
    state.addTypes({rows.with(md_dist::Role::Owned, true),
                    rows.with(md_dist::Role::Ghost, true)});
  }
  state.addRegion();
  Operation *op = builder.create(state);
  Block *block = new Block();
  op->getRegion(0).push_back(block);
  Block &source = gather.getKernel().front();
  Value r = block->addArgument(source.getArgument(0).getType(), loc);
  Value d = block->addArgument(source.getArgument(1).getType(), loc);
  unsigned ends = source.getNumArguments() - 2;
  SmallVector<Value> values;
  for (unsigned k = 0; k != ends; ++k)
    values.push_back(
        block->addArgument(source.getArgument(2 + k).getType(), loc));
  Type real = builder.getF64Type();
  Value forceI = block->addArgument(real, loc);
  Value forceK = block->addArgument(real, loc);

  // The kernel with i at its center, and with k.
  IRMapping forward, backward;
  forward.map(source.getArgument(0), r);
  backward.map(source.getArgument(0), r);
  forward.map(source.getArgument(1), d);
  backward.map(source.getArgument(1), d);
  for (unsigned k = 0; k + 1 < ends; k += 2) {
    forward.map(source.getArgument(2 + k), values[k]);
    forward.map(source.getArgument(3 + k), values[k + 1]);
    backward.map(source.getArgument(2 + k), values[k + 1]);
    backward.map(source.getArgument(3 + k), values[k]);
  }
  Value fromI = inlineKernel(source, *block, forward);
  Value fromK = inlineKernel(source, *block, backward);
  OpBuilder kernel = OpBuilder::atBlockEnd(block);
  ScalarDerivative derivative(kernel, r);
  Value slopeI, slopeK;
  if (failed(derivative.get(fromI, slopeI)) ||
      failed(derivative.get(fromK, slopeK)))
    return failure();
  ScalarEmitter emit(kernel, loc);
  Value both = emit.add(emit.mul(forceI, slopeI), emit.mul(forceK, slopeK));
  Value factor = both ? emit.div(both, r) : Value();
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
  md_dist::YieldOp::create(kernel, loc, ValueRange{yielded});
  eraseDeadOps(*block);
  result = op;
  return success();
}

Value DerivativeBuilder::addDistributedFields(ArrayRef<Value> fields) {
  if (fields.empty())
    return Value();
  if (fields.size() == 1)
    return fields.front();
  auto type = cast<md_dist::FieldType>(fields.front().getType());
  Type valueType = type.getKernelValueType();
  OperationState state(loc, md_dist::MapParticlesOp::getOperationName());
  state.addOperands(fields);
  state.addRegion();
  state.addTypes(type);
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
  md_dist::YieldOp::create(kernel, loc, ValueRange{total});
  return map->getResult(0);
}

/// Whether `term` is one of the two partial results of a traversal.
static bool isPartialTerm(Value term) {
  auto type = dyn_cast<md_dist::FieldType>(term.getType());
  return type && type.getPartial();
}

void DerivativeBuilder::addCompleteTerms(SmallVectorImpl<Value> &list) {
  SmallVector<Value, 2> complete, kept;
  for (Value term : list)
    (isPartialTerm(term) ? kept : complete).push_back(term);
  if (complete.size() < 2)
    return;
  Value total = isa<md_dist::FieldType>(complete.front().getType())
                    ? addDistributedFields(complete)
                    : addFields(complete);
  kept.push_back(total);
  list.assign(kept.begin(), kept.end());
}

bool DerivativeBuilder::isPositionValued(Value value) {
  if (value == body->getArgument(0))
    return true;
  Operation *op = value.getDefiningOp();
  if (auto distribute = dyn_cast_or_null<md_dist::DistributeOp>(op))
    return isPositionValued(distribute.getField());
  if (auto copy = dyn_cast_or_null<md_dist::HaloCopyOp>(op))
    return isPositionValued(copy.getField());
  if (auto sum = dyn_cast_or_null<md_dist::HaloSumOp>(op))
    return isPositionValued(sum.getOwned()) ||
           isPositionValued(sum.getGhosts());
  if (auto collect = dyn_cast_or_null<md_dist::CollectOp>(op))
    return isPositionValued(collect.getField());
  return false;
}

LogicalResult DerivativeBuilder::transposeTransfer(Operation *op) {
  auto transfer = dyn_cast<AdjointOpInterface>(op);
  if (!transfer || op->getNumResults() != 1)
    return success();
  auto found = distTerms.find(op->getResult(0));
  if (found == distTerms.end())
    return success();
  // The complete terms travel as one; each partial one with its partner.
  addCompleteTerms(found->second);
  SmallVector<Value, 2> incoming = found->second;
  for (OpOperand &operand : op->getOpOperands()) {
    unsigned index = operand.getOperandNumber();
    Value input = operand.get();
    if (!transfer.isLinearOperand(index) || !isPositional(input))
      continue;
    for (Value term : incoming) {
      // The ghost contributions of a gather go back with the owned
      // contributions of the same gather, in one md_dist.halo_sum.
      Value partner;
      if (isPartialTerm(term)) {
        auto gather = term.getDefiningOp<md_dist::GatherPairsOp>();
        if (!isa<md_dist::HaloCopyOp>(op) || !gather ||
            gather.getResults().size() != 2 || term != gather.getResult(1))
          return op->emitOpError()
                 << "cannot carry a partial adjoint through this op";
        SmallVector<Value, 2> &owned = distTerms[input];
        auto place = llvm::find(owned, gather.getResult(0));
        if (place == owned.end())
          return op->emitOpError()
                 << "cannot find the owned contributions of the traversal "
                    "whose ghost contributions return to the owners";
        partner = *place;
        owned.erase(place);
      }
      Value next = transfer.emitTranspose(builder, index, term, partner);
      if (!next)
        return failure();
      distTerms[input].push_back(next);
    }
  }
  return success();
}

LogicalResult DerivativeBuilder::buildDistributedTerms() {
  if (distTermsBuilt)
    return success();
  distTermsBuilt = true;
  for (Operation *op : llvm::reverse(potentialOps)) {
    if (auto sum = dyn_cast<md_dist::SumPairsOp>(op)) {
      Value weight;
      if (failed(getPartialWeight(sum.getResult(), weight)))
        return failure();
      if (!weight)
        continue;
      // The fields the kernel reads at both particles (Section 15.4).
      for (unsigned place = 0; place < sum.getGathered().size(); place += 2) {
        Value field = sum.getGathered()[place];
        if (!isPositional(field))
          continue;
        Value term;
        if (failed(emitPairFieldTerm(sum, place, weight, term)))
          return failure();
        distTerms[field].push_back(term);
      }
      if (!wantForces)
        continue;
      // K(i, k) = −weight · u'(r) · d / r at i, and its value under
      // exchange, −K, at k: the forces of the distance.
      Operation *gather = createDistributedPairOp(
          sum, /*gather=*/true, Exchange::Antisymmetric, Type());
      Block &block = gather->getRegion(0).front();
      OpBuilder kernel(block.getTerminator());
      Value factor;
      if (failed(emitRadialFactor(gather, weight, kernel, factor)))
        return failure();
      Type vectorType = VectorType::get({3}, builder.getF64Type());
      ScalarEmitter emit(kernel, loc);
      Value contribution =
          factor ? emit.mul(vector::BroadcastOp::create(kernel, loc,
                                                        vectorType, factor),
                            block.getArgument(1))
                 : emit.constant(0.0, vectorType);
      setYield(block, contribution);
      distTerms[sum.getPositions()].push_back(gather->getResult(0));
      distTerms[sum.getGhosts()].push_back(gather->getResult(1));
      continue;
    }
    if (auto sum = dyn_cast<md_dist::SumParticlesOp>(op)) {
      Value weight;
      if (failed(getPartialWeight(sum.getResult(), weight)))
        return failure();
      if (!weight)
        continue;
      for (auto [place, input] : llvm::enumerate(sum.getGathered())) {
        if (!isPositional(input))
          continue;
        Value term;
        if (failed(emitRowForces(sum, place, weight, term)))
          return failure();
        distTerms[input].push_back(term);
      }
      continue;
    }
    if (auto map = dyn_cast<md_dist::MapParticlesOp>(op)) {
      auto found = distTerms.find(map.getResult());
      if (found == distTerms.end() || found->second.empty())
        continue;
      addCompleteTerms(found->second);
      Value outer = found->second.front();
      for (auto [place, input] : llvm::enumerate(map.getGathered())) {
        if (!isPositional(input))
          continue;
        Value term;
        if (failed(emitMapTerm(map, place, outer, term)))
          return failure();
        distTerms[input].push_back(term);
      }
      continue;
    }
    if (auto gather = dyn_cast<md_dist::GatherPairsOp>(op)) {
      bool twoEnded = gather.getResults().size() == 2;
      SmallVector<Value, 2> owned = distTerms.lookup(gather.getResult(0));
      SmallVector<Value, 2> ghost =
          twoEnded ? distTerms.lookup(gather.getResult(1))
                   : SmallVector<Value, 2>();
      if (owned.empty() && ghost.empty())
        continue;
      Block &kernel = gather.getKernel().front();
      if (!kernel.getArgument(1).use_empty())
        return gather.emitOpError()
               << "cannot differentiate a field gathered on domains from a "
                  "kernel that uses the displacement; only kernels that "
                  "depend on the distance are supported";
      if (!cast<md_dist::FieldType>(gather.getResult(0).getType())
               .getKernelValueType()
               .isF64())
        return gather.emitOpError()
               << "cannot differentiate a gathered field on domains that "
                  "is not of f64";
      for (Value field : gather.getGathered())
        if (isPositional(field))
          return gather.emitOpError()
                 << "cannot differentiate a field gathered on domains from "
                    "a kernel that reads a field computed from the "
                    "positions";
      auto neighborhood =
          gather.getRelation().getDefiningOp<md_dist::NeighborhoodOp>();
      Value halo = neighborhood.getHalo();
      Density density{gather, Value(), Value()};
      if (!twoEnded) {
        // Complete at the owners (directed): the forces on ρ, copied to the
        // ghosts, of exactly this version.
        addCompleteTerms(owned);
        density.force = owned.front();
        auto type = cast<md_dist::FieldType>(density.force.getType());
        density.ghosts = md_dist::HaloCopyOp::create(
            builder, loc, type.with(md_dist::Role::Ghost, false),
            density.force, halo, Value());
      } else {
        // The two partial results (unique), completed by halo_sum: its
        // transposes gave the forces on ρ and their copy.
        auto copy = ghost.size() == 1
                        ? ghost.front().getDefiningOp<md_dist::HaloCopyOp>()
                        : md_dist::HaloCopyOp();
        if (owned.size() != 1 || !copy || copy.getField() != owned.front() ||
            copy.getHalo() != halo || copy.getImage())
          return gather.emitOpError()
                 << "cannot differentiate a gather whose two results do not "
                    "return to the owners together through the halo of its "
                    "relation";
        density.force = owned.front();
        density.ghosts = copy.getResult();
      }
      densities.push_back(density);
      if (!wantForces)
        continue;
      Operation *forces;
      if (failed(emitDensityOp(density, /*virial=*/false, forces)))
        return failure();
      distTerms[gather.getPositions()].push_back(forces->getResult(0));
      distTerms[gather.getGhosts()].push_back(forces->getResult(1));
      continue;
    }
    if (failed(transposeTransfer(op)))
      return failure();
  }
  return success(!activityFailed);
}

LogicalResult DerivativeBuilder::buildDistributedForces(Value &forces) {
  if (failed(buildDistributedTerms()))
    return failure();
  Value positions = body->getArgument(0);
  SmallVector<Value> total;
  for (Value term : distTerms.lookup(positions))
    total.push_back(term);
  if (total.empty())
    return potential.emitOpError()
           << "cannot compute forces: the energy does not depend on a sum "
              "over a relation";
  forces = addFields(total);
  return success();
}

LogicalResult DerivativeBuilder::buildDistributedVirial(Value &virial) {
  Type virialType = VectorType::get({9}, builder.getF64Type());
  ScalarEmitter outer(builder, loc);
  // The virial of pairs, Σ d ⊗ K, is the derivative in a strain only where
  // the positions enter the energy through the distances of pairs.
  for (Operation *op : potentialOps)
    if (isa<md_dist::SumParticlesOp, md_dist::MapParticlesOp,
            md_dist::SumPairsOp, md_dist::GatherPairsOp>(op)) {
      ValueRange read = isa<md_dist::SumParticlesOp, md_dist::MapParticlesOp>(op)
                            ? op->getOperands()
                            : op->getOperands().drop_front(3);
      if (llvm::any_of(read, [&](Value input) {
            return isPositionValued(input);
          }))
        return op->emitOpError()
               << "cannot compute the virial of a term that reads the "
                  "positions on domains as a field";
    }
  if (failed(buildDistributedTerms()))
    return failure();
  Value total;
  for (md_dist::SumPairsOp sum : pairSums) {
    Value weight;
    if (failed(getPartialWeight(sum.getResult(), weight)))
      return failure();
    if (!weight)
      continue;
    // W = Σ d ⊗ K(i, k) over the pairs, each once: the sum over the
    // traversal of the energy, whose ghosts carry their images, so that d
    // is the displacement of the pair in the cell [Louwerse2006].
    Operation *term = createDistributedPairOp(
        sum, /*gather=*/false, Exchange::Symmetric, virialType);
    Block &block = term->getRegion(0).front();
    OpBuilder kernel(block.getTerminator());
    Value factor;
    if (failed(emitRadialFactor(term, weight, kernel, factor)))
      return failure();
    ScalarEmitter emit(kernel, loc);
    Value contribution;
    if (factor) {
      Value d = block.getArgument(1);
      Value scaled = emit.mul(
          vector::BroadcastOp::create(kernel, loc, d.getType(), factor), d);
      SmallVector<Value, 9> elements;
      for (int64_t a = 0; a < 3; ++a)
        for (int64_t b = 0; b < 3; ++b)
          elements.push_back(
              emit.mul(vector::ExtractOp::create(kernel, loc, d, a),
                       vector::ExtractOp::create(kernel, loc, scaled, b)));
      contribution =
          vector::FromElementsOp::create(kernel, loc, virialType, elements);
    } else {
      contribution = emit.constant(0.0, virialType);
    }
    setYield(block, contribution);
    Value reduced = md_dist::AllReduceOp::create(builder, loc, virialType,
                                                 term->getResult(0));
    total = outer.add(total, reduced);
  }
  // The virial of the forces of each density (Section 15.2).
  for (const Density &density : densities) {
    Operation *term;
    if (failed(emitDensityOp(density, /*virial=*/true, term)))
      return failure();
    Value reduced = md_dist::AllReduceOp::create(builder, loc, virialType,
                                                 term->getResult(0));
    total = outer.add(total, reduced);
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
  using Dependence = Activity;
  using Verdict = ActivityResult;
  ActivityAnalysis activity(parameter);
  activity.addKnownBlock(body);
  auto classify = [&](Value value) { return activity.classify(value); };

  auto describe = [&]() {
    return "argument " + std::to_string(argument) + " of '" +
           potential.getSymName().str() + "'";
  };
  // An error for a value that is not independent and has no rule here.
  auto refuse = [&](Value value, const Verdict &verdict) -> LogicalResult {
    Operation *op = value.getDefiningOp();
    Location at = verdict.unknownOperation ? verdict.unknownOperation->getLoc() :
                  op ? op->getLoc() : value.getLoc();
    if (verdict.dependence == Dependence::Unknown)
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
    if (verdict.dependence == Dependence::Inactive)
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
                                   Dependence::Unknown
                               ? classify(field)
                               : Verdict{Dependence::Active, ""});
    for (Value gathered : map.getGathered()) {
      Verdict verdict = classify(gathered);
      if (verdict.dependence != Dependence::Inactive)
        return refuse(gathered, verdict);
    }
    Operation *copy = builder.clone(*map);
    Block &block = copy->getRegion(0).front();
    activity.addKnownBlock(&block);
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
    if (verdict.dependence == Dependence::Inactive) {
      takeAsZero(value);
      return success();
    }
    if (verdict.dependence == Dependence::Unknown)
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
      if (positions.dependence != Dependence::Inactive)
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
      if (field.dependence == Dependence::Active)
        return op->emitError()
               << "'" << op->getName() << "' takes a field that depends on "
               << describe() << ", and has no rule for that derivative";
      if (field.dependence == Dependence::Unknown)
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
    activity.addKnownBlock(&block);
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
// Derivative with respect to a field of the particles
//===----------------------------------------------------------------------===//

LogicalResult DerivativeBuilder::buildFieldDerivative(int64_t argument,
                                                      Value &result) {
  // g_i = ∂U/∂a_i for the field a. The field enters the energy through
  // sums whose kernels take its values: a sum over a relation gives a
  // gather over it, whose kernel is the derivative of the pair's energy in
  // the value of the central particle (the kernel of the sum is symmetric
  // under exchange, so the derivative in the value of the other particle is
  // the same kernel with the particles swapped, which the directed
  // expansion of the gather visits); a sum over tuples gives a gather over
  // them that yields the derivative in the value of each member; a sum over
  // particles gives a map. As for a number (D161), the three outcomes: a
  // field that reaches no value of the energy by any use has the
  // derivative zero; a use of these kinds has a rule; any other use (a
  // reciprocal sum, a map that computes another field from it) is an
  // error that names the op.
  Value parameter = body->getArgument(argument);
  Type fieldType = parameter.getType();
  auto describe = [&]() {
    return "argument " + std::to_string(argument) + " of '" +
           potential.getSymName().str() + "'";
  };

  // The derivative of `value`, which the kernel in `block` computes, in the
  // argument `variable` of the kernel; everything else that the kernel
  // takes is independent of it by a proof, or the derivative fails.
  auto differentiate = [&](Block &block, Value value, Value variable,
                           OpBuilder &kernel, Value &slope) -> LogicalResult {
    ActivityAnalysis activity(variable);
    activity.addKnownBlock(&block);
    ScalarDerivative::LeafHandler leaf = [&](Value leafValue,
                                             Value &tangent) -> LogicalResult {
      tangent = Value();
      ActivityResult verdict = activity.classify(leafValue);
      if (verdict.dependence == Activity::Inactive)
        return success();
      Operation *op = leafValue.getDefiningOp();
      Location at = verdict.unknownOperation
                        ? verdict.unknownOperation->getLoc()
                        : op ? op->getLoc() : leafValue.getLoc();
      if (verdict.dependence == Activity::Unknown)
        return emitError(at)
               << "cannot prove that this value does not depend on "
               << describe() << ": " << verdict.reason;
      if (op)
        return emitError(at) << "'" << op->getName() << "' depends on "
                             << describe()
                             << " and has no rule for its derivative";
      return emitError(at) << "a block argument depends on " << describe()
                           << " and has no rule for its derivative";
    };
    ScalarDerivative derivative(kernel, variable, leaf);
    return derivative.get(value, slope);
  };

  SmallVector<Value> terms;
  SmallVector<Operation *> users;
  // In the order of the body, whatever the order of the uses.
  for (Operation *op : potentialOps)
    if (llvm::is_contained(op->getOperands(), parameter))
      users.push_back(op);
  // Every op of the potential that takes the field as an operand is among
  // `users`. Another op of the body that takes it was added by a derivative
  // requested before this one: the copy of a sum whose kernel yields its
  // derivative in a number takes the fields that the sum gathers, and so
  // does the gather of the forces of a sum. Those are not uses by the
  // energy. A use within the kernel of an op is one that no rule follows.
  for (OpOperand &use : parameter.getUses())
    if (use.getOwner()->getBlock() != body)
      return use.getOwner()->emitError()
             << "'" << use.getOwner()->getName() << "' takes " << describe()
             << ", a field, within another op, and has no rule for the "
                "derivative in it";
  for (Operation *user : users) {
    // The places of the field among the fields that the op gathers.
    SmallVector<unsigned, 2> places;
    ValueRange gathered;
    // A reciprocal sum of the charges: the derivative of its energy in the
    // charge of a particle is the potential of the grid at the particle,
    // which the op yields as a fourth result when asked. The op takes the
    // place of the one of the potential, which yields the same energy,
    // forces, and virial.
    if (auto reciprocal = dyn_cast<ReciprocalOp>(user)) {
      if (reciprocal.getCharges() != parameter ||
          reciprocal.getPositions() == parameter)
        return user->emitError()
               << "'" << user->getName() << "' takes " << describe()
               << " other than as its charges, and has no rule for the "
                  "derivative in it";
      Value weight;
      if (failed(getWeight(reciprocal.getEnergy(), weight)))
        return failure();
      if (!weight)
        continue;
      Value potential = reciprocal.getPotential();
      if (!potential) {
        OperationState state(reciprocal.getLoc(),
                             ReciprocalOp::getOperationName());
        state.addOperands(reciprocal->getOperands());
        state.addAttributes(reciprocal->getAttrs());
        state.addTypes(reciprocal->getResultTypes());
        state.addTypes(fieldType);
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(reciprocal);
        auto extended = cast<ReciprocalOp>(builder.create(state));
        reciprocal.getEnergy().replaceAllUsesWith(extended.getEnergy());
        reciprocal.getForces().replaceAllUsesWith(extended.getForces());
        reciprocal.getVirial().replaceAllUsesWith(extended.getVirial());
        if (energy == reciprocal.getEnergy())
          energy = extended.getEnergy();
        for (ReciprocalOp &known : reciprocals)
          if (known == reciprocal)
            known = extended;
        for (Operation *&known : potentialOps)
          if (known == reciprocal.getOperation())
            known = extended;
        reciprocal->erase();
        potential = extended.getPotential();
      }
      if (isOne(weight)) {
        terms.push_back(potential);
        continue;
      }
      OperationState state(loc, MapParticlesOp::getOperationName());
      state.addOperands(potential);
      state.addRegion();
      state.addTypes(fieldType);
      Operation *map = builder.create(state);
      Block *block = new Block();
      map->getRegion(0).push_back(block);
      block->addArgument(builder.getF64Type(), loc);
      OpBuilder kernel = OpBuilder::atBlockEnd(block);
      ScalarEmitter emit(kernel, loc);
      YieldOp::create(kernel, loc,
                      ValueRange{emit.mul(weight, block->getArgument(0))});
      terms.push_back(map->getResult(0));
      continue;
    }
    if (auto sum = dyn_cast<SumRelationOp>(user))
      gathered = sum.getGathered();
    else if (auto sum = dyn_cast<SumTuplesOp>(user))
      gathered = sum.getGathered();
    else if (auto sum = dyn_cast<SumParticlesOp>(user))
      gathered = sum.getGathered();
    else
      return user->emitError()
             << "'" << user->getName() << "' takes " << describe()
             << ", a field, and has no rule for the derivative in it";
    for (auto [place, field] : llvm::enumerate(gathered))
      if (field == parameter)
        places.push_back(place);
    unsigned uses = 0;
    for (Value operand : user->getOperands())
      uses += operand == parameter;
    if (uses != places.size())
      return user->emitError()
             << "'" << user->getName() << "' takes " << describe()
             << " other than as a field that its kernel gathers, and has no "
                "rule for the derivative in it";
    if (!user->getResult(0).getType().isF64())
      return user->emitError()
             << "cannot differentiate a sum whose result is not f64";

    Value weight;
    if (failed(getWeight(user->getResult(0), weight)))
      return failure();
    if (!weight)
      continue;

    if (auto sum = dyn_cast<SumRelationOp>(user)) {
      Operation *gather = createPairOp(GatherRelationOp::getOperationName(),
                                       sum, Exchange::None, fieldType);
      Block &block = gather->getRegion(0).front();
      Value pairEnergy = cast<YieldOp>(block.getTerminator()).getOperand(0);
      OpBuilder kernel(block.getTerminator());
      ScalarEmitter emit(kernel, loc);
      Value total;
      for (unsigned place : places) {
        Value slope;
        if (failed(differentiate(block, pairEnergy,
                                 block.getArgument(2 + 2 * place), kernel,
                                 slope)))
          return failure();
        total = emit.add(total, slope);
      }
      if (!total) {
        gather->erase();
        continue;
      }
      setYield(block, emit.mul(weight, total));
      terms.push_back(gather->getResult(0));
      continue;
    }

    if (auto sum = dyn_cast<SumTuplesOp>(user)) {
      unsigned arity = sum.getArity();
      unsigned first = sum.getCoordinates().size();
      Operation *gather =
          createTupleOp(GatherTuplesOp::getOperationName(), sum, fieldType);
      Block &block = gather->getRegion(0).front();
      Value tupleEnergy = cast<YieldOp>(block.getTerminator()).getOperand(0);
      OpBuilder kernel(block.getTerminator());
      ScalarEmitter emit(kernel, loc);
      SmallVector<Value, 4> members(arity);
      bool any = false;
      for (unsigned member = 0; member != arity; ++member) {
        Value total;
        for (unsigned place : places) {
          Value slope;
          if (failed(differentiate(
                  block, tupleEnergy,
                  block.getArgument(first + place * arity + member), kernel,
                  slope)))
            return failure();
          total = emit.add(total, slope);
        }
        any |= static_cast<bool>(total);
        members[member] = total ? emit.mul(weight, total)
                                : emit.constant(0.0, kernel.getF64Type());
      }
      if (!any) {
        gather->erase();
        continue;
      }
      block.getTerminator()->setOperands(members);
      eraseDeadOps(block);
      terms.push_back(gather->getResult(0));
      continue;
    }

    auto sum = cast<SumParticlesOp>(user);
    OperationState state(loc, MapParticlesOp::getOperationName());
    state.addOperands(sum.getGathered());
    state.addRegion();
    state.addTypes(fieldType);
    Operation *map = builder.create(state);
    Block *block = new Block();
    map->getRegion(0).push_back(block);
    Block &source = sum.getKernel().front();
    IRMapping mapping;
    for (BlockArgument blockArgument : source.getArguments())
      mapping.map(blockArgument,
                  block->addArgument(blockArgument.getType(), loc));
    Value value = inlineKernel(source, *block, mapping);
    OpBuilder kernel = OpBuilder::atBlockEnd(block);
    ScalarEmitter emit(kernel, loc);
    Value total;
    for (unsigned place : places) {
      Value slope;
      if (failed(differentiate(*block, value, block->getArgument(place),
                               kernel, slope)))
        return failure();
      total = emit.add(total, slope);
    }
    if (!total) {
      map->erase();
      continue;
    }
    YieldOp::create(kernel, loc, ValueRange{emit.mul(weight, total)});
    eraseDeadOps(*block);
    terms.push_back(map->getResult(0));
  }

  if (!terms.empty()) {
    result = addFields(terms);
    return success();
  }
  // No use of the field reaches the energy: its derivative is zero, a
  // field of zeros over its particles.
  if (remarks)
    potential.emitRemark() << "independent of " << describe()
                           << ": its derivative is zero";
  OperationState state(loc, MapParticlesOp::getOperationName());
  state.addOperands(parameter);
  state.addRegion();
  state.addTypes(fieldType);
  Operation *map = builder.create(state);
  Block *block = new Block();
  map->getRegion(0).push_back(block);
  block->addArgument(builder.getF64Type(), loc);
  OpBuilder kernel = OpBuilder::atBlockEnd(block);
  ScalarEmitter emit(kernel, loc);
  YieldOp::create(kernel, loc,
                  ValueRange{emit.constant(0.0, kernel.getF64Type())});
  result = map->getResult(0);
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
      resultTypes.push_back(builder.getF64Type());
      break;
    case Request::Derivative: {
      // A number for a number, a field for a field of the particles.
      Type type = potentialType.getInput(arguments[resultTypes.size()]);
      resultTypes.push_back(isa<FieldType>(type) ? type
                                                 : Type(builder.getF64Type()));
      break;
    }
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
    potentialOps.push_back(&op);
    if (isa_and_nonnull<md_dist::MDDistDialect>(op.getDialect()))
      distributed = true;
    if (auto pairs = dyn_cast<md_dist::SumPairsOp>(&op))
      pairSums.push_back(pairs);
    if (auto rows = dyn_cast<md_dist::SumParticlesOp>(&op))
      rowSums.push_back(rows);
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
  if (distributed) {
    wantForces = llvm::is_contained(kinds, static_cast<int32_t>(Request::Forces));
    for (int32_t kind : kinds)
      if (static_cast<Request>(kind) == Request::Derivative) {
        potential.emitOpError()
            << "cannot differentiate a potential on domains in a parameter "
               "or a field yet; energy, forces, and virial are supported "
               "(D[ml2-distribute])";
        return fail();
      }
    if (needsPositions && (failed(checkDistributedUses()) || activityFailed))
      return fail();
  } else if (needsPositions &&
             (failed(checkPositionUses()) || activityFailed)) {
    return fail();
  }

  builder.setInsertionPoint(oldReturn);
  // The adjoints of the fields that depend on the positions, from the uses
  // of the potential alone, before the derivatives add their own.
  if (needsPositions && !distributed && failed(buildAdjoints()))
    return fail();
  if (activityFailed) return fail();
  SmallVector<Value> results;
  for (unsigned i = 0, e = kinds.size(); i != e; ++i) {
    Value result;
    LogicalResult status = success();
    switch (static_cast<Request>(kinds[i])) {
    case Request::Energy:
      result = energy;
      break;
    case Request::Forces:
      status = distributed ? buildDistributedForces(result)
                           : buildForces(result);
      break;
    case Request::Virial:
      status = distributed ? buildDistributedVirial(result)
                           : buildVirial(result);
      break;
    case Request::Derivative:
      status = isa<FieldType>(body->getArgument(arguments[i]).getType())
                   ? buildFieldDerivative(arguments[i], result)
                   : buildParameterDerivative(arguments[i], result);
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
