// md-distribute: a global potential on a grid of domains.
//
// See docs/distributed-ml2.md, Sections 11 and 14 (D276). The
// body of an md.potential is rewritten into md_dist: a layout from the
// positions, the owned rows of every field, one halo, a copy to the ghosts
// of every field that a pair kernel reads, and the pair ops on the chosen
// traversal. The potential keeps its signature, so that md.evaluate, and
// md-differentiate after it, see a potential of the particle set; the
// derivative then carries the adjoints of the transfers (G^T for G).

#include "mdir/Dialect/MD/Transforms/Passes.h"
#include "mdir/Dialect/MD/Transforms/Truncation.h"

#include "mdir/Dialect/Dyn/DynOps.h"
#include "mdir/Dialect/MD/MDExchange.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MDDist/Grid.h"
#include "mdir/Dialect/MDDist/MDDistDialect.h"
#include "mdir/Dialect/MDDist/MDDistOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseMap.h"

#include <cmath>
#include <limits>

using namespace mlir;
namespace md_dist = mdir::md_dist;

namespace mdir {
namespace md {

#define GEN_PASS_DEF_DISTRIBUTE
#include "mdir/Dialect/MD/Transforms/Passes.h.inc"

namespace {

class Distributor {
public:
  Distributor(FunctionOpInterface potential, ArrayRef<int64_t> grid,
              md_dist::Traversal traversal, double reach)
      : potential(potential), grid(grid), traversal(traversal),
        requestedReach(reach), builder(potential.getContext()) {}

  LogicalResult run();

private:
  LogicalResult check();
  LogicalResult declareDecomposition();
  void begin();

  /// The owned rows of `field`, a field of the particle set, or a null
  /// value after a diagnostic on `user`.
  Value ownedOf(Operation *user, Value field);
  /// The copy of the owned rows `owned` to the ghosts.
  Value ghostsOf(Value owned);
  /// The owned rows and their ghost copies of the fields that a pair op
  /// gathers, in pairs.
  LogicalResult gatheredPairs(Operation *user, ValueRange fields,
                              SmallVectorImpl<Value> &result);
  /// Copies the kernel of `source` into `target`, with md_dist.yield.
  void cloneKernel(Operation *source, Operation *target);
  /// Whether the exchange contract of a pair kernel is proven, or a
  /// diagnostic.
  LogicalResult proveKernel(Operation *op, Exchange exchange);

  md_dist::FieldType ownedType(Type type) {
    auto field = cast<FieldType>(type);
    return md_dist::FieldType::get(
        builder.getContext(), decompositionName, md_dist::Role::Owned,
        /*partial=*/false, field.getNumComponents(), field.getElementType());
  }

  LogicalResult rewrite(NeighborhoodOp op);
  LogicalResult rewrite(SumRelationOp op);
  LogicalResult rewrite(GatherRelationOp op);
  template <typename Op> LogicalResult rewriteTuple(Op op);
  LogicalResult rewrite(SumParticlesOp op);
  LogicalResult rewrite(MapParticlesOp op);

  FunctionOpInterface potential;
  ArrayRef<int64_t> grid;
  md_dist::Traversal traversal;
  double requestedReach;
  OpBuilder builder;

  Block *body = nullptr;
  Value positions, cell;
  FlatSymbolRefAttr decompositionName;
  double reach = 0.0;
  bool needsHalo = false;

  Value layout, ownedPositions, halo, ghostPositions;
  llvm::DenseMap<Value, Value> owned;
  llvm::DenseMap<Value, Value> ghosts;
  llvm::DenseMap<Value, Value> relations;
  SmallVector<Operation *> erased;
};

} // namespace

LogicalResult Distributor::check() {
  if (potential.isExternal())
    return potential.emitOpError() << "cannot distribute a declaration";
  if (!llvm::hasSingleElement(potential->getRegion(0)))
    return potential.emitOpError()
           << "cannot distribute a body of more than one block";
  body = &potential->getRegion(0).front();
  positions = body->getArgument(0);
  cell = body->getArgument(1);

  for (Operation &op : *body) {
    if (isa<ReturnOp>(op))
      continue;
    if (auto neighborhood = dyn_cast<NeighborhoodOp>(op)) {
      if (neighborhood.getExcluded() && traversal != md_dist::Traversal::Midpoint && !isa<FunctionOp>(potential.getOperation()))
        return op.emitOpError()
               << "cannot be distributed: exclusions are pairs of a tuple "
                  "set, which needs the topology across domains (stage 2 of "
                  "#31)";
      needsHalo = true;
      double cutoff = neighborhood.getCutoff().convertToDouble();
      reach = std::max(reach, traversal == md_dist::Traversal::Midpoint
                                  ? 0.5 * cutoff : cutoff);
      continue;
    }
    if (isa<SumRelationOp, GatherRelationOp, SumParticlesOp, MapParticlesOp>(
            op))
      continue;
    if (isa<SumTuplesOp, GatherTuplesOp>(op) && (traversal == md_dist::Traversal::Midpoint || isa<FunctionOp>(potential.getOperation()))) {
      if (!requestedReach)
        return op.emitOpError("midpoint topology tuples require an explicit halo reach for their extent");
      needsHalo = true;
      continue;
    }
    if (isa<SumTuplesOp, GatherTuplesOp>(op))
      return op.emitOpError()
             << "cannot be distributed: a sum over tuples needs the topology "
                "across domains (stage 2 of #31)";
    if (isa<ReciprocalOp>(op))
      return op.emitOpError()
             << "cannot be distributed: a reciprocal sum needs a "
                "decomposition of its grid (stage 3 of #31)";
    // Any other op computes numbers from numbers: the sums of the domains
    // after md_dist.all_reduce. An op of md, or one that takes or gives a
    // field or a relation, is not known to the pass.
    bool structured = op.getName().getDialectNamespace() == "md" ||
                      op.getNumRegions() != 0;
    for (Type type : llvm::concat<const Type>(op.getOperandTypes(),
                                              op.getResultTypes()))
      structured |= isa<FieldType, RelationType, CellType>(type);
    if (structured)
      return op.emitOpError() << "cannot be distributed by md-distribute";
  }
  if (requestedReach > 0.0) {
    if (requestedReach < reach && traversal == md_dist::Traversal::Midpoint)
      return potential.emitOpError() << "under-covered midpoint halo: reach " << requestedReach
                                     << " is below half the cutoff, " << reach;
    if (requestedReach < reach)
      return potential.emitOpError()
             << "the reach of the halo, " << requestedReach
             << ", is less than the cutoff, " << reach;
    reach = requestedReach;
  }
  return success();
}

LogicalResult Distributor::declareDecomposition() {
  auto positionType = cast<FieldType>(positions.getType());
  std::string name = (SymbolTable::getSymbolName(potential).getValue() + ".domains").str();
  if (isa<FunctionOp>(potential.getOperation()))
    name = (positionType.getParticleSet().getValue() + ".midpoint_domains").str();
  decompositionName = FlatSymbolRefAttr::get(builder.getContext(), name);
  auto module = potential->getParentOfType<ModuleOp>();
  if (Operation *existing = SymbolTable::lookupSymbolIn(module, name)) {
    auto decomposition = dyn_cast<md_dist::DecompositionOp>(existing);
    if (!decomposition ||
        decomposition.getGrid() != grid ||
        decomposition.getParticleSetAttr() != positionType.getParticleSet())
      return existing->emitOpError()
             << "the name '" << name
             << "' is needed for the decomposition of the potential, but is "
                "taken";
    return success();
  }
  builder.setInsertionPoint(potential);
  md_dist::DecompositionOp::create(builder, potential.getLoc(),
                                   builder.getStringAttr(name),
                                   positionType.getParticleSet(),
                                   builder.getDenseI64ArrayAttr(grid));
  return success();
}

void Distributor::begin() {
  Location loc = potential.getLoc();
  MLIRContext *context = builder.getContext();
  builder.setInsertionPointToStart(body);
  SmallVector<Value> groups;
  if (traversal == md_dist::Traversal::Midpoint || isa<FunctionOp>(potential.getOperation()))
    for (Value input : body->getArguments()) {
      auto type = dyn_cast<RelationType>(input.getType());
      if (type && type.getTupleSet()) {
        auto set = SymbolTable::lookupNearestSymbolFrom<TupleSetOp>(potential, type.getTupleSet());
        if (set && set->hasAttr("md_dist.constraint_group"))
          groups.push_back(input);
      }
    }
  layout = md_dist::PartitionOp::create(
      builder, loc, md_dist::LayoutType::get(context, decompositionName),
      positions, cell, groups);
  ownedPositions = md_dist::DistributeOp::create(
      builder, loc, ownedType(positions.getType()), positions, layout, cell);
  owned[positions] = ownedPositions;
  if (!needsHalo)
    return;
  auto haloType = md_dist::HaloType::get(context, decompositionName,
                                         APFloat(reach));
  halo = md_dist::HaloOp::create(builder, loc, haloType, layout,
                                 ownedPositions, cell);
  ghostPositions = md_dist::HaloCopyOp::create(
      builder, loc,
      ownedType(positions.getType()).with(md_dist::Role::Ghost, false),
      ownedPositions, halo, cell);
  ghosts[ownedPositions] = ghostPositions;
}

Value Distributor::ownedOf(Operation *user, Value field) {
  if (field == positions) {
    user->emitOpError()
        << "cannot be distributed: it reads the positions as a field, whose "
           "owned rows are wrapped into the cell, so that a term of the "
           "absolute positions would change";
    return Value();
  }
  if (Value found = owned.lookup(field))
    return found;
  auto argument = dyn_cast<BlockArgument>(field);
  if (!argument || argument.getOwner() != body) {
    user->emitOpError() << "cannot be distributed: a field it reads does not "
                           "come from an argument or from an op that the "
                           "pass distributes";
    return Value();
  }
  // A field argument: its owned rows, after the layout.
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfterValue(halo ? ghostPositions : ownedPositions);
  Value result = md_dist::DistributeOp::create(
      builder, potential.getLoc(), ownedType(field.getType()), field, layout,
      Value());
  owned[field] = result;
  return result;
}

Value Distributor::ghostsOf(Value ownedField) {
  if (Value found = ghosts.lookup(ownedField))
    return found;
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfterValue(ownedField);
  auto type = cast<md_dist::FieldType>(ownedField.getType());
  Value result = md_dist::HaloCopyOp::create(
      builder, ownedField.getLoc(), type.with(md_dist::Role::Ghost, false),
      ownedField, halo, Value());
  ghosts[ownedField] = result;
  return result;
}

LogicalResult Distributor::gatheredPairs(Operation *user, ValueRange fields,
                                         SmallVectorImpl<Value> &result) {
  for (Value field : fields) {
    Value own = ownedOf(user, field);
    if (!own)
      return failure();
    result.push_back(own);
    result.push_back(ghostsOf(own));
  }
  return success();
}

void Distributor::cloneKernel(Operation *source, Operation *target) {
  IRMapping mapping;
  source->getRegion(0).cloneInto(&target->getRegion(0), mapping);
  Block &block = target->getRegion(0).front();
  Operation *terminator = block.getTerminator();
  OpBuilder at(terminator);
  md_dist::YieldOp::create(at, terminator->getLoc(),
                           terminator->getOperands());
  terminator->erase();
}

LogicalResult Distributor::proveKernel(Operation *op, Exchange exchange) {
  if (exchange == Exchange::None)
    return success();
  Block &kernel = op->getRegion(0).front();
  int expected = exchange == Exchange::Symmetric ? 1 : -1;
  if (proveExchange(kernel, kernel.getTerminator()->getOperand(0)) ==
      expected)
    return success();
  return op->emitOpError()
         << "cannot be distributed: md_dist proves every exchange contract "
            "(V8), and the kernel cannot be shown to be "
         << stringifyExchange(exchange);
}

LogicalResult Distributor::rewrite(NeighborhoodOp op) {
  if (op.getPositions() != positions || op.getCell() != cell)
    return op.emitOpError() << "cannot be distributed: it is not of the "
                               "positions and the cell of the potential";
  builder.setInsertionPoint(op);
  auto type = md_dist::RelationType::get(builder.getContext(),
                                         decompositionName, traversal);
  relations[op.getResult()] = md_dist::NeighborhoodOp::create(
      builder, op.getLoc(), type, ownedPositions, ghostPositions, halo,
      op.getCutoffAttr(), op.getExcluded());
  return success();
}

LogicalResult Distributor::rewrite(SumRelationOp op) {
  Value relation = relations.lookup(op.getRelation());
  if (!relation)
    return op.emitOpError()
           << "cannot be distributed: its relation does not come from "
              "md.neighborhood";
  if (op.getPositions() != positions || op.getCell() != cell)
    return op.emitOpError() << "cannot be distributed: it is not of the "
                               "positions and the cell of the potential";
  if (failed(expandTruncation(op)) || failed(proveKernel(op, op.getExchange())))
    return failure();
  SmallVector<Value> gathered;
  if (failed(gatheredPairs(op, op.getGathered(), gathered)))
    return failure();
  builder.setInsertionPoint(op);
  Type valueType = op.getResult().getType();
  OperationState state(op.getLoc(), md_dist::SumPairsOp::getOperationName());
  state.addOperands({relation, ownedPositions, ghostPositions});
  state.addOperands(gathered);
  state.getOrAddProperties<md_dist::SumPairsOp::Properties>()
      .setOperandSegmentSizes(
          {1, 1, 1, 0, static_cast<int32_t>(gathered.size())});
  state.addAttribute("exchange", op.getExchangeAttr());
  state.addTypes(md_dist::PartialType::get(builder.getContext(), valueType));
  state.addRegion();
  Operation *sum = builder.create(state);
  cloneKernel(op, sum);
  Value total = md_dist::AllReduceOp::create(builder, op.getLoc(), valueType,
                                             sum->getResult(0));
  op.getResult().replaceAllUsesWith(total);
  return success();
}

LogicalResult Distributor::rewrite(GatherRelationOp op) {
  if (op->getNumResults() != 1)
    return op.emitOpError() << "cannot be distributed: it gathers several "
                               "fields (step 6 of #377)";
  Value relation = relations.lookup(op.getRelation());
  if (!relation)
    return op.emitOpError()
           << "cannot be distributed: its relation does not come from "
              "md.neighborhood";
  if (op.getPositions() != positions || op.getCell() != cell)
    return op.emitOpError() << "cannot be distributed: it is not of the "
                               "positions and the cell of the potential";
  if (traversal != md_dist::Traversal::Directed &&
      op.getExchange() == Exchange::None)
    return op.emitOpError()
           << "cannot be distributed on the unique traversal: each pair is "
              "computed once, and the value at its second particle needs an "
              "exchange contract";
  if (failed(proveKernel(op, op.getExchange())))
    return failure();
  SmallVector<Value> gathered;
  if (failed(gatheredPairs(op, op.getGathered(), gathered)))
    return failure();
  builder.setInsertionPoint(op);
  md_dist::FieldType type = ownedType(op.getResult().getType());
  OperationState state(op.getLoc(), md_dist::GatherPairsOp::getOperationName());
  state.addOperands({relation, ownedPositions, ghostPositions});
  state.addOperands(gathered);
  state.getOrAddProperties<md_dist::GatherPairsOp::Properties>()
      .setOperandSegmentSizes(
          {1, 1, 1, 0, static_cast<int32_t>(gathered.size())});
  state.addAttribute("exchange", op.getExchangeAttr());
  if (traversal == md_dist::Traversal::Directed) {
    state.addTypes(type);
  } else {
    state.addTypes(type.with(md_dist::Role::Owned, true));
    state.addTypes(type.with(md_dist::Role::Ghost, true));
  }
  state.addRegion();
  Operation *gather = builder.create(state);
  cloneKernel(op, gather);
  Value result = gather->getResult(0);
  if (traversal != md_dist::Traversal::Directed)
    result = md_dist::HaloSumOp::create(builder, op.getLoc(), type,
                                        gather->getResult(0),
                                        gather->getResult(1), halo);
  owned[op.getResult()] = result;
  return success();
}

template <typename Op>
LogicalResult Distributor::rewriteTuple(Op op) {
  if (op.getPositions() != positions || op.getCell() != cell || op->getNumResults() != 1)
    return op.emitOpError("expected one topology quantity at the current positions and cell");
  builder.setInsertionPoint(op);
  auto rowsType = MemRefType::get({ShapedType::kDynamic, 1 + int64_t(op.getArity())},
                                  builder.getI64Type());
  Value rows = md_dist::TupleRowsOp::create(builder, op.getLoc(), rowsType,
                                            op.getRelation(), ownedPositions, halo, false);
  SmallVector<Value> fields;
  if (failed(gatheredPairs(op, op.getGathered(), fields))) return failure();
  builder.setInsertionPoint(op);
  SmallVector<Type> outputs;
  Type result = op.getResult().getType();
  if constexpr (std::is_same_v<Op, SumTuplesOp>)
    outputs.push_back(md_dist::PartialType::get(builder.getContext(), result));
  else {
    auto type = ownedType(result);
    outputs = {type.with(md_dist::Role::Owned, true), type.with(md_dist::Role::Ghost, true)};
  }
  auto kernel = md_dist::TupleKernelOp::create(builder, op.getLoc(), outputs, rows, ownedPositions,
      ghostPositions, halo, fields, op.getParameters(), op.getCoordinateKindsAttr(), op.getCoordinateMembersAttr());
  cloneKernel(op, kernel);
  if constexpr (std::is_same_v<Op, SumTuplesOp>) {
    auto total = md_dist::AllReduceOp::create(builder, op.getLoc(), result, kernel.getResult(0));
    op.getResult().replaceAllUsesWith(total.getResult());
  } else {
    owned[op.getResult()] = md_dist::HaloSumOp::create(builder, op.getLoc(), ownedType(result),
        kernel.getResult(0), kernel.getResult(1), halo);
  }
  return success();
}

LogicalResult Distributor::rewrite(SumParticlesOp op) {
  SmallVector<Value> fields;
  for (Value field : op.getGathered()) {
    Value own = ownedOf(op, field);
    if (!own)
      return failure();
    fields.push_back(own);
  }
  builder.setInsertionPoint(op);
  Type valueType = op.getResult().getType();
  OperationState state(op.getLoc(),
                       md_dist::SumParticlesOp::getOperationName());
  state.addOperands(fields);
  state.addTypes(md_dist::PartialType::get(builder.getContext(), valueType));
  state.addRegion();
  Operation *sum = builder.create(state);
  cloneKernel(op, sum);
  Value total = md_dist::AllReduceOp::create(builder, op.getLoc(), valueType,
                                             sum->getResult(0));
  op.getResult().replaceAllUsesWith(total);
  return success();
}

LogicalResult Distributor::rewrite(MapParticlesOp op) {
  SmallVector<Value> fields;
  for (Value field : op.getGathered()) {
    Value own = ownedOf(op, field);
    if (!own)
      return failure();
    fields.push_back(own);
  }
  builder.setInsertionPoint(op);
  OperationState state(op.getLoc(),
                       md_dist::MapParticlesOp::getOperationName());
  state.addOperands(fields);
  state.addTypes(ownedType(op.getResult().getType()));
  state.addRegion();
  Operation *map = builder.create(state);
  cloneKernel(op, map);
  owned[op.getResult()] = map->getResult(0);
  return success();
}

LogicalResult Distributor::run() {
  if (failed(check()) || failed(declareDecomposition()))
    return failure();
  begin();
  SmallVector<Operation *> ops;
  for (Operation &op : *body)
    ops.push_back(&op);
  for (Operation *op : ops) {
    LogicalResult status = success();
    bool rewritten = true;
    if (auto o = dyn_cast<NeighborhoodOp>(op))
      status = rewrite(o);
    else if (auto o = dyn_cast<SumRelationOp>(op))
      status = rewrite(o);
    else if (auto o = dyn_cast<GatherRelationOp>(op))
      status = rewrite(o);
    else if (auto o = dyn_cast<SumTuplesOp>(op))
      status = rewriteTuple(o);
    else if (auto o = dyn_cast<GatherTuplesOp>(op))
      status = rewriteTuple(o);
    else if (auto o = dyn_cast<SumParticlesOp>(op))
      status = rewrite(o);
    else if (auto o = dyn_cast<MapParticlesOp>(op))
      status = rewrite(o);
    else
      rewritten = false;
    if (failed(status))
      return failure();
    if (rewritten)
      erased.push_back(op);
  }
  auto result = cast<ReturnOp>(body->getTerminator());
  builder.setInsertionPoint(result);
  for (OpOperand &operand : result->getOpOperands())
    if (Value field = owned.lookup(operand.get()))
      operand.set(md_dist::CollectOp::create(builder, result.getLoc(), operand.get().getType(), field, layout));
  // The ops of md go, in reverse order: a field or a relation that they
  // give is used only by the others, whose results are replaced.
  for (Operation *op : llvm::reverse(erased)) {
    if (!op->use_empty())
      return op->emitOpError()
             << "cannot be distributed: its result is used by an op that "
                "the pass does not distribute";
    op->erase();
  }
  return success();
}

namespace {
// Project disjoint constraint tuples at their common owner. The native
// integrator supplies the old and predicted rows, and consumes the complete
// correction and the agreed failure flag before its next kick.
static LogicalResult distributeGroups(ModuleOp module, ArrayRef<int64_t> grid,
                                      double reach, bool retainLayout) {
  for (auto program : module.getOps<dyn::ProgramOp>()) {
    SmallVector<Operation *> work;
    SmallVector<Value> groups;
    program.walk([&](Operation *op) {
      if (!isa<SumTuplesOp, GatherTuplesOp>(op)) return;
      Value relation = op->getOperand(0);
      auto type = cast<RelationType>(relation.getType());
      auto set = SymbolTable::lookupNearestSymbolFrom<TupleSetOp>(op, type.getTupleSet());
      if (set && set->hasAttr("md_dist.constraint_group")) {
        work.push_back(op);
        if (!llvm::is_contained(groups, relation)) groups.push_back(relation);
      }
    });
    if (work.empty()) continue;
    auto loc = program.getLoc();
    OpBuilder b(work.front());
    Value x = program.getBody().front().getArgument(0), cell = work.front()->getOperand(2);
    auto particleSet = cast<FieldType>(x.getType()).getParticleSet();
    auto name = FlatSymbolRefAttr::get(b.getContext(), (particleSet.getValue() + ".midpoint_domains").str());
    if (!module.lookupSymbol(name.getValue())) {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(module.getBody());
      md_dist::DecompositionOp::create(b, loc, name.getAttr(), particleSet, b.getDenseI64ArrayAttr(grid));
    }
    auto layoutType = md_dist::LayoutType::get(b.getContext(), name);
    auto haloType = md_dist::HaloType::get(b.getContext(), name, APFloat(reach));
    auto fieldType = [&](Value field) {
      auto t = cast<FieldType>(field.getType());
      return md_dist::FieldType::get(b.getContext(), name, md_dist::Role::Owned,
          false, t.getNumComponents(), t.getElementType());
    };
    Value original;
    if (!retainLayout)
      original = md_dist::PartitionOp::create(b, loc, layoutType, x, cell, groups);
    struct Epoch { Value layout, x, h, ghosts; };
    DenseMap<Value, Epoch> epochs;
    for (Operation *op : work) {
      b.setInsertionPoint(op);
      loc = op->getLoc();
      Value positions = op->getOperand(1);
      Epoch &e = epochs[positions];
      if (!e.layout) {
        if (retainLayout) {
          auto partition = md_dist::PartitionOp::create(b, loc, layoutType, positions, cell, groups);
          partition->setAttr("md_dist.projection_epoch", b.getUnitAttr());
          e.layout = partition;
          e.x = md_dist::DistributeOp::create(b, loc, fieldType(positions),
                                              positions, e.layout, cell);
        } else {
          Value moving = md_dist::DistributeOp::create(b, loc, fieldType(positions), positions, original, Value());
          e.layout = md_dist::MigrateOp::create(b, loc, layoutType, original, moving, cell);
          e.x = md_dist::RedistributeOp::create(b, loc, fieldType(positions), moving, original, e.layout, cell);
        }
        if (epochs.size() == 1) e.layout.getDefiningOp()->setAttr("md_dist.step", b.getUnitAttr());
        e.h = md_dist::HaloOp::create(b, loc, haloType, e.layout, e.x, cell);
        e.ghosts = md_dist::HaloCopyOp::create(b, loc,
            fieldType(positions).with(md_dist::Role::Ghost, false), e.x, e.h, cell);
      }
      auto convert = [&](auto tuple) {
        constexpr bool gather = std::is_same_v<decltype(tuple), GatherTuplesOp>;
        auto rowType = MemRefType::get({ShapedType::kDynamic, int64_t(tuple.getArity() + 1)}, b.getI64Type());
        Value rows = md_dist::TupleRowsOp::create(b, loc, rowType, tuple.getRelation(), e.x, e.h, true);
        SmallVector<Value> fields;
        for (Value field : tuple.getGathered()) {
          auto type = fieldType(field);
          Value owned = md_dist::DistributeOp::create(b, loc, type, field, e.layout, Value());
          fields.push_back(owned);
          fields.push_back(md_dist::HaloCopyOp::create(b, loc, type.with(md_dist::Role::Ghost, false), owned, e.h, Value()));
        }
        SmallVector<Type> resultTypes;
        if constexpr (gather) {
          resultTypes.push_back(fieldType(tuple.getResult()));
          if (tuple.getFailed()) resultTypes.push_back(b.getI1Type());
        } else
          resultTypes.push_back(md_dist::PartialType::get(b.getContext(), tuple.getResult().getType()));
        auto kernel = md_dist::TupleKernelOp::create(b, loc, resultTypes, rows, e.x, e.ghosts,
            e.h, fields, tuple.getParameters(), tuple.getCoordinateKindsAttr(), tuple.getCoordinateMembersAttr());
        IRMapping map;
        tuple.getKernel().cloneInto(&kernel.getKernel(), map);
        auto term = kernel.getKernel().front().getTerminator();
        OpBuilder inside(term);
        md_dist::YieldOp::create(inside, loc, term->getOperands());
        term->erase();
        Value result;
        if constexpr (gather) {
          result = md_dist::CollectOp::create(b, loc, tuple.getResult().getType(), kernel.getResult(0), e.layout);
          if (tuple.getFailed()) tuple.getFailed().replaceAllUsesWith(kernel.getResult(1));
        } else {
          result = md_dist::AllReduceOp::create(b, loc, tuple.getResult().getType(), kernel.getResult(0));
        }
        tuple.getResult().replaceAllUsesWith(result);
        tuple.erase();
      };
      if (auto gather = dyn_cast<GatherTuplesOp>(op)) convert(gather);
      else convert(cast<SumTuplesOp>(op));
    }
  }
  return success();
}

// A sum used only in the total potential can share its pair reduction
// with the other operand. Differentiation then forms one force and one
// virial tensor per pair. Independently observed terms keep their sums.
static void combinePairEnergiesIn(PotentialOp potential) {
  for (Operation &op : llvm::make_early_inc_range(potential.getBody().front())) {
    auto add = dyn_cast<arith::AddFOp>(op);
    if (!add || !add.getType().isF64())
      continue;
    auto left = add.getLhs().getDefiningOp<SumRelationOp>();
    auto right = add.getRhs().getDefiningOp<SumRelationOp>();
    if (!left || !right || !left->hasOneUse() || !right->hasOneUse() ||
        left.getRelation() != right.getRelation() ||
        left.getPositions() != right.getPositions() || left.getCell() != right.getCell() ||
        left.getExchange() != right.getExchange() ||
        left.getTruncation() != Truncation::None || right.getTruncation() != Truncation::None)
      continue;

    OpBuilder builder(add);
    auto joined = cast<SumRelationOp>(builder.clone(*left));
    joined.getGatheredMutable().append(right.getGathered());
    Block &kernel = joined.getKernel().front(), &source = right.getKernel().front();
    IRMapping map;
    for (BlockArgument arg : source.getArguments()) {
      unsigned i = arg.getArgNumber();
      map.map(arg, i < 2 ? kernel.getArgument(i) : kernel.addArgument(arg.getType(), arg.getLoc()));
    }
    Operation *yield = kernel.getTerminator();
    builder.setInsertionPoint(yield);
    for (Operation &nested : source.without_terminator())
      builder.clone(nested, map);
    Value rhs = map.lookupOrDefault(source.getTerminator()->getOperand(0));
    yield->setOperand(0, arith::AddFOp::create(builder, add.getLoc(), yield->getOperand(0), rhs));
    add.getResult().replaceAllUsesWith(joined.getResult());
    add.erase();
    right.erase();
    left.erase();
  }
}

class Distribute : public impl::DistributeBase<Distribute> {
public:
  using impl::DistributeBase<Distribute>::DistributeBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    if (grid.empty() == (domains == 0)) {
      module.emitError()
          << "md-distribute needs exactly one of 'grid=gx,gy,gz' or "
             "'domains=N box=Lx,Ly,Lz'";
      return signalPassFailure();
    }
    if (!grid.empty() &&
        (grid.size() != 3 ||
         llvm::any_of(grid, [](int64_t n) { return n < 1; }))) {
      module.emitError() << "md-distribute needs a grid of three positive "
                            "numbers of domains, 'grid=gx,gy,gz'";
      return signalPassFailure();
    }
    if (domains < 0 || (!grid.empty() && !box.empty()) ||
        (domains > 0 &&
         (box.size() != 3 || llvm::any_of(box, [](double edge) {
            return !(edge > 0.0 && std::isfinite(edge));
          })))) {
      module.emitError() << "md-distribute: 'domains=N' needs a positive N "
                            "and 'box=Lx,Ly,Lz' with three positive finite "
                            "lengths; 'box' is only for 'domains'";
      return signalPassFailure();
    }
    md_dist::Traversal kind;
    if (traversal == "directed")
      kind = md_dist::Traversal::Directed;
    else if (traversal == "unique")
      kind = md_dist::Traversal::Unique;
    else if (traversal == "midpoint")
      kind = md_dist::Traversal::Midpoint;
    else {
      module.emitError() << "md-distribute: the traversal is 'directed' or "
                            "'unique' or 'midpoint', not '"
                         << traversal << "'";
      return signalPassFailure();
    }
    if (!(reach >= 0.0)) {
      module.emitError() << "md-distribute: the reach is not negative";
      return signalPassFailure();
    }
    SmallVector<PotentialOp> selected;
    for (PotentialOp potential : module.getOps<PotentialOp>())
      if (potentials.empty() ||
          llvm::is_contained(potentials, potential.getSymName().str()))
        selected.push_back(potential);
    for (const std::string &name : potentials)
      if (!llvm::any_of(selected, [&](PotentialOp p) {
            return p.getSymName() == name;
          })) {
        module.emitError() << "md-distribute: no potential is named '" << name
                           << "'";
        return signalPassFailure();
      }
    SmallVector<int64_t, 3> divisions =
        domains ? md_dist::selectGrid(domains, box)
                : SmallVector<int64_t, 3>(grid.begin(), grid.end());
    if (derivatives) {
      for (FunctionOp fn : module.getOps<FunctionOp>()) {
        bool derived = llvm::any_of(selected, [&](PotentialOp p) {
          return fn.getSymName().starts_with((p.getSymName() + ".").str());
        });
        if (!derived || !llvm::any_of(fn.getBody().front(), [](Operation &op) {
              return isa<NeighborhoodOp, SumTuplesOp, GatherTuplesOp>(op);
            })) continue;
        if (failed(Distributor(fn, divisions, kind, reach).run()))
          return signalPassFailure();
      }
      if (failed(distributeGroups(module, divisions, reach, projectionEpochs))) return signalPassFailure();
      return;
    }
    for (PotentialOp potential : selected) {
      if (combinePairEnergies) combinePairEnergiesIn(potential);
      Distributor distributor(potential, divisions, kind, reach);
      if (failed(distributor.run()))
        return signalPassFailure();
    }
  }
};
} // namespace

} // namespace md
} // namespace mdir
