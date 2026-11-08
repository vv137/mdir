// Forces and strain of exactly the same image-aware window arithmetic.
#include "mdir/Dialect/MD/Transforms/TemplateWindows.h"
#include "mdir/Dialect/MD/Transforms/ScalarDerivative.h"
#include "mdir/Dialect/MD/Transforms/Activity.h"
#include "mdir/Dialect/MD/MDTemplateWindow.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;
using namespace mdir::md;

namespace {
// Clone only a value's dependency closure. Mapping contains the explicit
// window coordinates, gathered parameters and any captured constants.
static Value materialize(OpBuilder &builder, Value value, Block &source,
                         IRMapping &mapping) {
  if (Value known = mapping.lookupOrNull(value)) return known;
  Operation *definition = value.getDefiningOp();
  if (!definition || definition->getBlock() != &source) return value;
  for (Value operand : definition->getOperands())
    materialize(builder, operand, source, mapping);
  builder.clone(*definition, mapping);
  return mapping.lookup(value);
}
} // namespace

SmallVector<Value> mdir::md::emitTemplateWindowBody(
    OpBuilder &builder, Operation *op, IRMapping &mapping) {
  Block &source = op->getRegion(0).front();
  auto yield = cast<YieldOp>(source.getTerminator());
  bool gather = isa<GatherTemplateWindowOp>(op);
  unsigned arity = cast<RelationType>(op->getOperand(0).getType()).getArity();
  unsigned stride = gather ? arity : 1;
  unsigned pairs = yield.getNumOperands() / (stride + 1);
  double cutoff = op->getAttrOfType<FloatAttr>("cutoff").getValueAsDouble();
  Location loc = op->getLoc();
  Value cutoff2 = arith::ConstantOp::create(builder, loc,
      builder.getF64FloatAttr(cutoff * cutoff));
  SmallVector<Value> results;
  for (unsigned pair = 0; pair != pairs; ++pair) {
    Value displacement = materialize(builder,
        yield.getOperand(stride * pairs + pair), source, mapping);
    Value squares = arith::MulFOp::create(builder, loc, displacement, displacement);
    Value r2 = vector::ReductionOp::create(builder, loc,
                                          vector::CombiningKind::ADD, squares);
    Value inside = arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::OLT,
                                         r2, cutoff2);
    SmallVector<Type> types;
    for (unsigned member = 0; member != stride; ++member)
      types.push_back(yield.getOperand(pair * stride + member).getType());
    auto guarded = scf::IfOp::create(builder, loc, inside,
        [&](OpBuilder &body, Location) {
          IRMapping local = mapping;
          SmallVector<Value> values;
          for (unsigned member = 0; member != stride; ++member)
            values.push_back(materialize(body,
                yield.getOperand(pair * stride + member), source, local));
          scf::YieldOp::create(body, loc, values);
        }, [&](OpBuilder &body, Location) {
          ScalarEmitter emit(body, loc);
          SmallVector<Value> zeros;
          for (Type type : types) zeros.push_back(emit.constant(0.0, type));
          scf::YieldOp::create(body, loc, zeros);
        });
    llvm::append_range(results, guarded.getResults());
  }
  return results;
}

LogicalResult mdir::md::checkTemplateWindowActivity(
    SumTemplateWindowOp op, Value positions) {
  if (!op.getResult().getType().isF64())
    return op.emitOpError("cannot differentiate a template-window sum whose result is not f64");
  ActivityAnalysis activity(positions);
  ActivityAnalysis cellActivity(op.getCell());
  for (Value field : op.getGathered())
    if (activity.classify(field).dependence != Activity::Inactive ||
        cellActivity.classify(field).dependence != Activity::Inactive)
      return op.emitOpError("template-window gathered parameters must be position- and cell-independent");
  Block &block = op.getKernel().front();
  for (Operation &inner : block)
    for (Value operand : inner.getOperands()) {
      bool local = operand.getDefiningOp()
                       ? op->isAncestor(operand.getDefiningOp())
                       : cast<BlockArgument>(operand).getOwner() == &block;
      if (local) continue;
      if (activity.classify(operand).dependence != Activity::Inactive ||
          cellActivity.classify(operand).dependence != Activity::Inactive)
        return op.emitOpError("template-window captures must not hide position or cell dependence");
    }
  return success();
}

LogicalResult mdir::md::buildTemplateWindowDerivative(
    OpBuilder &builder, SumTemplateWindowOp source, Value weight,
    bool virial, Value &result) {
  Location loc = source.getLoc();
  unsigned arity = cast<RelationType>(source.getRelation().getType()).getArity();
  auto coordinates = getTemplateWindowMembers(arity);
  Type vector = VectorType::get({3}, builder.getF64Type());
  Type tensor = VectorType::get({9}, builder.getF64Type());
  OperationState state(loc, virial ? SumTemplateWindowOp::getOperationName()
                                   : GatherTemplateWindowOp::getOperationName());
  state.addOperands(source->getOperands());
  state.addAttributes(source->getAttrs());
  state.addTypes(virial ? tensor : source.getPositions().getType());
  state.addRegion();
  Operation *derived = builder.create(state);
  IRMapping mapping;
  source.getKernel().cloneInto(&derived->getRegion(0), mapping);
  Block &block = derived->getRegion(0).front();
  auto yield = cast<YieldOp>(block.getTerminator());
  unsigned pairs = yield.getNumOperands() / 2;
  // Own the values: updating the terminator below invalidates its range.
  SmallVector<Value> energies(yield.getOperands().take_front(pairs));
  SmallVector<Value> displacements(yield.getOperands().drop_front(pairs));
  OpBuilder kernel(yield);
  ScalarEmitter emit(kernel, loc);
  SmallVector<Value> contributions;
  for (Value energy : energies) {
    SmallVector<Value> forces(arity);
    SmallVector<Value> elements(9);
    for (auto [index, members] : llvm::enumerate(coordinates)) {
      Value displacement = block.getArgument(index);
      SmallVector<Value, 3> components;
      for (unsigned axis = 0; axis != 3; ++axis) {
        SmallVector<double, 3> unit(3, 0.0);
        unit[axis] = 1.0;
        Value seed = arith::ConstantOp::create(
            kernel, loc, DenseElementsAttr::get(cast<VectorType>(vector),
                                                ArrayRef<double>(unit)));
        ScalarDerivative derivative(kernel, displacement, nullptr, seed);
        Value slope;
        if (failed(derivative.get(energy, slope))) {
          derived->erase();
          return failure();
        }
        Value component = emit.neg(emit.mul(weight, slope));
        components.push_back(component ? component : emit.constant(0.0, kernel.getF64Type()));
      }
      Value force = vector::FromElementsOp::create(kernel, loc, vector, components);
      // d = x_a - image(x_b): the image is fixed for the coordinate
      // derivative. Its cell shift is included when d strains below.
      forces[members[0]] = emit.add(forces[members[0]], force);
      forces[members[1]] = emit.sub(forces[members[1]], force);
      if (virial)
        for (unsigned a = 0; a != 3; ++a)
          for (unsigned b = 0; b != 3; ++b)
            elements[3 * a + b] = emit.add(
                elements[3 * a + b],
                emit.mul(vector::ExtractOp::create(kernel, loc, displacement, a),
                         vector::ExtractOp::create(kernel, loc, force, b)));
    }
    if (virial) {
      for (Value &element : elements)
        if (!element) element = emit.constant(0.0, kernel.getF64Type());
      contributions.push_back(vector::FromElementsOp::create(kernel, loc, tensor, elements));
    } else {
      for (Value force : forces)
        contributions.push_back(force ? force : emit.constant(0.0, vector));
    }
  }
  llvm::append_range(contributions, displacements);
  yield->setOperands(contributions);
  eraseDeadOps(block);
  result = derived->getResult(0);
  return success();
}
