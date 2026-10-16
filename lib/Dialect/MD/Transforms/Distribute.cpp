// md-distribute: a global potential on a grid of domains.
//
// See docs/distributed-ml2.md, Sections 11 and 14 (D[ml2-distribute]). The
// body of an md.potential is rewritten into md_dist: a layout from the
// positions, the owned rows of every field, one halo, a copy to the ghosts
// of every field that a pair kernel reads, and the pair ops on the chosen
// traversal. The potential keeps its signature, so that md.evaluate, and
// md-differentiate after it, see a potential of the particle set; the
// derivative then carries the adjoints of the transfers (G^T for G).

#include "mdir/Dialect/MD/Transforms/Passes.h"
#include "mdir/Dialect/MD/Transforms/Truncation.h"

#include "mdir/Dialect/MD/MDExchange.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MDDist/MDDistDialect.h"
#include "mdir/Dialect/MDDist/MDDistOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;
namespace md_dist = mdir::md_dist;

namespace mdir {
namespace md {

#define GEN_PASS_DEF_DISTRIBUTE
#include "mdir/Dialect/MD/Transforms/Passes.h.inc"

namespace {

class Distributor {
public:
  Distributor(PotentialOp potential, ArrayRef<int64_t> grid,
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
  LogicalResult rewrite(SumParticlesOp op);
  LogicalResult rewrite(MapParticlesOp op);

  PotentialOp potential;
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
  if (!llvm::hasSingleElement(potential.getBody()))
    return potential.emitOpError()
           << "cannot distribute a body of more than one block";
  body = &potential.getBody().front();
  positions = body->getArgument(0);
  cell = body->getArgument(1);

  for (Operation &op : *body) {
    if (isa<ReturnOp>(op))
      continue;
    if (auto neighborhood = dyn_cast<NeighborhoodOp>(op)) {
      if (neighborhood.getExcluded())
        return op.emitOpError()
               << "cannot be distributed: exclusions are pairs of a tuple "
                  "set, which needs the topology across domains (stage 2 of "
                  "#31)";
      needsHalo = true;
      reach = std::max(reach, neighborhood.getCutoff().convertToDouble());
      continue;
    }
    if (isa<SumRelationOp, GatherRelationOp, SumParticlesOp, MapParticlesOp>(
            op))
      continue;
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
  std::string name = (potential.getSymName() + ".domains").str();
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
  layout = md_dist::PartitionOp::create(
      builder, loc, md_dist::LayoutType::get(context, decompositionName),
      positions, cell);
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
      op.getCutoffAttr());
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
  Value relation = relations.lookup(op.getRelation());
  if (!relation)
    return op.emitOpError()
           << "cannot be distributed: its relation does not come from "
              "md.neighborhood";
  if (op.getPositions() != positions || op.getCell() != cell)
    return op.emitOpError() << "cannot be distributed: it is not of the "
                               "positions and the cell of the potential";
  if (traversal == md_dist::Traversal::Unique &&
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
  if (traversal == md_dist::Traversal::Unique)
    result = md_dist::HaloSumOp::create(builder, op.getLoc(), type,
                                        gather->getResult(0),
                                        gather->getResult(1), halo);
  owned[op.getResult()] = result;
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
class Distribute : public impl::DistributeBase<Distribute> {
public:
  using impl::DistributeBase<Distribute>::DistributeBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    if (grid.size() != 3 || llvm::any_of(grid, [](int64_t n) { return n < 1; })) {
      module.emitError() << "md-distribute needs a grid of three positive "
                            "numbers of domains, 'grid=gx,gy,gz'";
      return signalPassFailure();
    }
    md_dist::Traversal kind;
    if (traversal == "directed")
      kind = md_dist::Traversal::Directed;
    else if (traversal == "unique")
      kind = md_dist::Traversal::Unique;
    else {
      module.emitError() << "md-distribute: the traversal is 'directed' or "
                            "'unique', not '"
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
    SmallVector<int64_t, 3> divisions(grid.begin(), grid.end());
    for (PotentialOp potential : selected) {
      Distributor distributor(potential, divisions, kind, reach);
      if (failed(distributor.run()))
        return signalPassFailure();
    }
  }
};
} // namespace

} // namespace md
} // namespace mdir
