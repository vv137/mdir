// Forces and strain of exactly the same image-aware window arithmetic
// (D[kind-pairs]; the construction of D[rigid-water-template]).
#include "mdir/Dialect/MD/Transforms/SegmentWindows.h"
#include "mdir/Dialect/MD/Transforms/ScalarDerivative.h"
#include "mdir/Dialect/MD/Transforms/Activity.h"
#include "mdir/Dialect/MD/MDSegmentWindow.h"
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

SmallVector<Value> mdir::md::emitSegmentWindowBody(
    OpBuilder &builder, Operation *op, IRMapping &mapping) {
  Block &source = op->getRegion(0).front();
  auto yield = cast<YieldOp>(source.getTerminator());
  auto gather = dyn_cast<GatherSegmentWindowOp>(op);
  bool siteForces = gather && gather.getSiteForces();
  SegmentShape shape = getSegmentShape(op);
  unsigned stride = gather ? shape.getNumSlots(siteForces) : 1;
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
    if (siteForces) {
      // The semantic reference scatters frame atoms. A native scheduler
      // may retain these site slots until its accumulated write boundary.
      auto values = guarded.getResults();
      unsigned first = shape.a.getSize();
      llvm::append_range(results, emitSiteForceTranspose(
          builder, loc, values.take_front(first), shape.a));
      llvm::append_range(results, emitSiteForceTranspose(
          builder, loc, values.drop_front(first), shape.b));
    } else llvm::append_range(results, guarded.getResults());
  }
  return results;
}

LogicalResult mdir::md::checkSegmentWindowActivity(
    SumSegmentWindowOp op, Value positions) {
  if (!op.getResult().getType().isF64())
    return op.emitOpError("cannot differentiate a segment-window sum whose result is not f64");
  ActivityAnalysis activity(positions);
  ActivityAnalysis cellActivity(op.getCell());
  for (Value field : op.getGathered())
    if (activity.classify(field).dependence != Activity::Inactive ||
        cellActivity.classify(field).dependence != Activity::Inactive)
      return op.emitOpError("segment-window gathered parameters must be position- and cell-independent");
  Block &block = op.getKernel().front();
  for (Operation &inner : block)
    for (Value operand : inner.getOperands()) {
      bool local = operand.getDefiningOp()
                       ? op->isAncestor(operand.getDefiningOp())
                       : cast<BlockArgument>(operand).getOwner() == &block;
      if (local) continue;
      if (activity.classify(operand).dependence != Activity::Inactive ||
          cellActivity.classify(operand).dependence != Activity::Inactive)
        return op.emitOpError("segment-window captures must not hide position or cell dependence");
    }
  return success();
}

LogicalResult mdir::md::buildSegmentWindowDerivative(
    OpBuilder &builder, SumSegmentWindowOp source, Value weight,
    bool virial, Value &result) {
  Location loc = source.getLoc();
  SegmentShape shape = getSegmentShape(source);
  unsigned arity = shape.getArity();
  auto coordinates = shape.getCoordinateMembers();
  Type vector = VectorType::get({3}, builder.getF64Type());
  Type tensor = VectorType::get({9}, builder.getF64Type());
  OperationState state(loc, virial ? SumSegmentWindowOp::getOperationName()
                                   : GatherSegmentWindowOp::getOperationName());
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
  SmallVector<Value> radialDistances;
  SmallVector<std::array<unsigned, 2>> places;
  bool radial = true;
  // Whether a site pair involves a massless site: only then do the slots
  // of the site layout differ from the frame atoms.
  bool formedSite = false;
  auto isSite = [](const SegmentSide &side, unsigned place) {
    return side.kind && !side.kind.getFrameIndex(place);
  };
  for (auto [energy, displacement] : llvm::zip_equal(energies, displacements)) {
    Value r2;
    for (Operation *user : displacement.getUsers())
      if (auto product = dyn_cast<arith::MulFOp>(user);
          product && product.getLhs() == displacement && product.getRhs() == displacement)
        for (Operation *use : product.getResult().getUsers())
          if (auto norm = dyn_cast<vector::ReductionOp>(use);
              norm && norm.getKind() == vector::CombiningKind::ADD)
            r2 = norm.getResult();
    llvm::DenseMap<Value, bool> proved;
    std::function<bool(Value)> onlyRadial = [&](Value value) -> bool {
      if (value == r2) return true;
      if (auto found = proved.find(value); found != proved.end()) return found->second;
      if (auto arg = dyn_cast<BlockArgument>(value))
        return arg.getOwner() != &block || arg.getArgNumber() >= coordinates.size();
      Operation *definition = value.getDefiningOp();
      if (!definition || definition->getBlock() != &block) return true;
      return proved[value] = llvm::all_of(definition->getOperands(), onlyRadial);
    };
    auto sites = getSegmentWindowSites(displacement, block, shape);
    radial &= bool(r2) && onlyRadial(energy) && sites.has_value();
    if (sites) {
      formedSite |= isSite(shape.a, (*sites)[0]) ||
                    isSite(shape.b, (*sites)[1] - shape.a.getSize());
      places.push_back(*sites);
    }
    radialDistances.push_back(r2);
  }
  bool siteLayout = radial && formedSite;
  if (siteLayout && !virial)
    derived->setAttr("site_forces", builder.getUnitAttr());
  // A slot of the frame layout: the frame index of each place.
  auto frameSlot = [&](unsigned slot) {
    unsigned na = shape.a.getSize();
    if (slot < na)
      return shape.a.kind ? *shape.a.kind.getFrameIndex(slot) : 0u;
    unsigned place = slot - na;
    return shape.a.getFrameSize() +
           (shape.b.kind ? *shape.b.kind.getFrameIndex(place) : 0u);
  };
  SmallVector<Value> contributions;
  for (auto [pair, energy] : llvm::enumerate(energies)) {
    if (radial) {
      Value slope;
      ScalarDerivative derivative(kernel, radialDistances[pair]);
      if (failed(derivative.get(energy, slope))) { derived->erase(); return failure(); }
      Value factor = emit.scale(-2.0, emit.mul(weight, slope));
      Value force = factor ? emit.mul(
          vector::BroadcastOp::create(kernel, loc, vector, factor), displacements[pair])
          : emit.constant(0.0, vector);
      if (virial) {
        SmallVector<Value> tensorValues;
        for (unsigned a = 0; a != 3; ++a)
          for (unsigned b = 0; b != 3; ++b)
            tensorValues.push_back(emit.mul(
                vector::ExtractOp::create(kernel, loc, displacements[pair], a),
                vector::ExtractOp::create(kernel, loc, force, b)));
        contributions.push_back(vector::FromElementsOp::create(kernel, loc, tensor, tensorValues));
      } else {
        std::array<unsigned, 2> sites = places[pair];
        unsigned slots = shape.getNumSlots(siteLayout);
        if (!siteLayout)
          for (unsigned &site : sites) site = frameSlot(site);
        for (unsigned site = 0; site != slots; ++site)
          contributions.push_back(site == sites[0] ? force : site == sites[1]
              ? emit.neg(force) : emit.constant(0.0, vector));
      }
      continue;
    }
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
