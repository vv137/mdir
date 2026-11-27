// Carry the fixed-cell velocity Verlet state through the driver's nested
// output intervals. The differentiated kernels remain those of the input.
#include "mdir/Conversion/Passes.h"
#include "mdir/Dialect/Dyn/DynOps.h"
#include "mdir/Dialect/MDDist/MDDistOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/MapVector.h"

using namespace mlir;
namespace mdir {
#define GEN_PASS_DEF_MDISTRIBUTEDRIVERSTEPS
#include "mdir/Conversion/Passes.h.inc"

namespace {
namespace dd = md_dist;
struct Epoch {
  Value layout, halo, reference;
  SmallVector<Value> fields; // x, v, f, m, then the potential parameters.
};

class Segment {
public:
  LogicalResult run(scf::ForOp root);
private:
  Value pack(OpBuilder &b, const Epoch &e) {
    return dd::PackOp::create(b, loc, stateType, e.layout, e.halo, cell, e.fields);
  }
  Epoch unpack(OpBuilder &b, Value state) {
    auto u = dd::UnpackOp::create(b, loc, state, cell);
    return {u.getLayout(), u.getHalo(), u.getReference(), SmallVector<Value>(u.getFields())};
  }
  Value advance(OpBuilder &b, Value field, Value slope, Value dt, Value mass);
  Epoch rebuild(OpBuilder &b, Epoch e);
  Value groupLoop(OpBuilder &b, scf::ForOp source, Value state, IRMapping map);
  Value loop(OpBuilder &b, scf::ForOp source, Value state, IRMapping map);
  SmallVector<Value> parameters, groups;
  dd::StateType stateType;
  Value cell;
  Location loc;
public:
  double groupSkin;
  bool wrapDrift;
  Segment(Location loc, bool wrapDrift, double skin)
      : loc(loc), groupSkin(skin), wrapDrift(wrapDrift) {}
};

Value Segment::advance(OpBuilder &b, Value field, Value slope, Value dt, Value mass) {
  SmallVector<Value> inputs{field, slope};
  if (mass) inputs.push_back(mass);
  auto map = dd::MapParticlesOp::create(b, loc, field.getType(), inputs);
  map->setAttr("md_dist.integration", b.getUnitAttr());
  if (!wrapDrift)
    map->setAttr(mass ? "md_dist.verlet_kick" : "md_dist.verlet_drift", b.getUnitAttr());
  Block *body = new Block;
  map.getKernel().push_back(body);
  for (Value input : inputs)
    body->addArgument(cast<dd::FieldType>(input.getType()).getKernelValueType(), loc);
  OpBuilder k = OpBuilder::atBlockEnd(body);
  if (mass) {
    Value m = body->getArgument(2);
    Value zero = arith::ConstantOp::create(k, loc, k.getF64FloatAttr(0));
    Value massless = arith::CmpFOp::create(k, loc, arith::CmpFPredicate::OEQ, m, zero);
    Value factor = arith::DivFOp::create(k, loc, dt, m);
    dt = arith::SelectOp::create(k, loc, massless, zero, factor);
  }
  Value w = vector::BroadcastOp::create(k, loc, body->getArgument(0).getType(), dt);
  Value delta = arith::MulFOp::create(k, loc, w, body->getArgument(1));
  Value updated = arith::AddFOp::create(k, loc, body->getArgument(0), delta);
  if (!mass && wrapDrift) {
    auto box = cell.getDefiningOp<md::OrthorhombicCellOp>();
    Value edges = vector::FromElementsOp::create(k, loc,
        cast<VectorType>(updated.getType()), ValueRange{box.getLx(), box.getLy(), box.getLz()});
    Value image = math::FloorOp::create(k, loc, arith::DivFOp::create(k, loc, updated, edges));
    updated = arith::SubFOp::create(k, loc, updated, arith::MulFOp::create(k, loc, edges, image));
  }
  dd::YieldOp::create(k, loc, ValueRange{updated});
  return map;
}

Epoch Segment::rebuild(OpBuilder &b, Epoch e) {
  Value ok = dd::CoveredOp::create(b, loc, b.getI1Type(), e.halo, e.reference,
                                   e.fields[0], cell, b.getFloatAttr(b.getF64Type(), stateType.getCutoff()));
  // The rebuild branch replaces this epoch for every subsequent force.
  ok.getDefiningOp()->setAttr("md_dist.retire_matrix", b.getUnitAttr());
  auto branch = scf::IfOp::create(b, loc, ok, [&](OpBuilder &keep, Location) {
    scf::YieldOp::create(keep, loc, pack(keep, e));
  }, [&](OpBuilder &move, Location) {
    Epoch n;
    n.layout = dd::MigrateOp::create(move, loc, e.layout.getType(), e.layout, e.fields[0], cell);
    for (auto [i, f] : llvm::enumerate(e.fields))
      n.fields.push_back(dd::RedistributeOp::create(move, loc, f.getType(), f,
          e.layout, n.layout, i == 0 ? cell : Value{}));
    n.halo = dd::HaloOp::create(move, loc, e.halo.getType(), n.layout, n.fields[0], cell);
    scf::YieldOp::create(move, loc, pack(move, n));
  });
  return unpack(b, branch.getResult(0));
}

Value Segment::loop(OpBuilder &b, scf::ForOp source, Value state, IRMapping map) {
  auto result = scf::ForOp::create(b, loc, map.lookupOrDefault(source.getLowerBound()),
      map.lookupOrDefault(source.getUpperBound()), map.lookupOrDefault(source.getStep()),
      ValueRange{state}, [&](OpBuilder &body, Location, Value iv, ValueRange args) {
    map.map(source.getInductionVar(), iv);
    Epoch e = unpack(body, args[0]);
    // Map logical inputs to slots so a field used after a migration reads its
    // new rows. Observations materialize a global field only at the use.
    DenseMap<Value, unsigned> slot;
    for (auto [i, v] : llvm::enumerate(source.getRegionIterArgs())) slot[v] = i;
    for (auto [i, v] : llvm::enumerate(parameters)) slot[v] = i + 3;
    for (Operation &op : source.getBody()->without_terminator()) {
      if (auto nested = dyn_cast<scf::ForOp>(op)) {
        e = unpack(body, loop(body, nested, pack(body, e), map));
        for (auto [i, v] : llvm::enumerate(nested.getResults())) slot[v] = i;
      } else if (auto kick = dyn_cast<dyn::KickOp>(op)) {
        e.fields[1] = advance(body, e.fields[1], e.fields[2],
                             map.lookupOrDefault(kick.getDt()), e.fields[3]);
        slot[kick.getResult()] = 1;
      } else if (auto drift = dyn_cast<dyn::DriftOp>(op)) {
        e.fields[0] = advance(body, e.fields[0], e.fields[1],
                             map.lookupOrDefault(drift.getDt()), {});
        e = rebuild(body, e);
        slot[drift.getResult()] = 0;
      } else if (auto part = dyn_cast<dd::PartitionOp>(op)) {
        map.map(part.getLayout(), e.layout);
      } else if (auto field = dyn_cast<dd::DistributeOp>(op)) {
        map.map(field.getResult(), e.fields[slot.find(field.getField())->second]);
      } else if (auto halo = dyn_cast<dd::HaloOp>(op)) {
        map.map(halo.getResult(), e.halo);
      } else if (auto collect = dyn_cast<dd::CollectOp>(op)) {
        e.fields[2] = map.lookup(collect.getField());
        slot[collect.getResult()] = 2;
      } else if (auto sum = dyn_cast<md::SumParticlesOp>(op);
                 sum && !wrapDrift && llvm::all_of(sum.getGathered(),
                     [&](Value field) { return slot.contains(field); })) {
        // Reporting a sum needs only one partial per domain. Keep its
        // fields in owned-row order throughout the integration loop.
        SmallVector<Value> fields;
        for (Value field : sum.getGathered()) fields.push_back(e.fields[slot.lookup(field)]);
        Type type = sum.getResult().getType();
        auto local = dd::SumParticlesOp::create(body, loc,
            dd::PartialType::get(body.getContext(), type), fields);
        IRMapping kernelMap(map);
        sum.getKernel().cloneInto(&local.getKernel(), kernelMap);
        Operation *term = local.getKernel().front().getTerminator();
        OpBuilder inside(term);
        dd::YieldOp::create(inside, loc, term->getOperands());
        term->erase();
        auto total = dd::AllReduceOp::create(body, loc, type, local.getResult());
        map.map(sum.getResult(), total.getResult());
      } else {
        for (Value operand : op.getOperands()) {
          auto it = slot.find(operand);
          if (it != slot.end())
            map.map(operand, dd::CollectOp::create(body, loc, operand.getType(),
                e.fields[it->second], e.layout).getResult());
        }
        body.clone(op, map);
      }
    }
    scf::YieldOp::create(body, loc, pack(body, e));
  });
  return result.getResult(0);
}

// Carry the native constrained Verlet program on its owned rows. Scalar
// stop flags, bookkeeping and output calls retain their original order.
Value Segment::groupLoop(OpBuilder &b, scf::ForOp source, Value state,
                         IRMapping map) {
  auto result = scf::ForOp::create(
      b, loc, map.lookupOrDefault(source.getLowerBound()),
      map.lookupOrDefault(source.getUpperBound()),
      map.lookupOrDefault(source.getStep()), ValueRange{state},
      [&](OpBuilder &body, Location, Value iv, ValueRange args) {
        map.map(source.getInductionVar(), iv);
        Epoch e = unpack(body, args[0]);
        llvm::MapVector<Value, Value> fields;
        for (auto [i, v] : llvm::enumerate(source.getRegionIterArgs()))
          fields[v] = e.fields[i];
        for (auto [i, v] : llvm::enumerate(parameters))
          fields[v] = e.fields[3 + i];
        auto setState = [&](ValueRange xyz) {
          e.fields.clear();
          for (Value v : xyz)
            e.fields.push_back(fields.lookup(v));
          for (Value v : parameters)
            e.fields.push_back(fields.lookup(v));
        };
        auto mapKernel = [&](Operation *op, ValueRange gathered, bool sum) {
          SmallVector<Value> inputs;
          for (Value v : gathered)
            inputs.push_back(fields.lookup(v));
          Operation *target;
          if (sum) {
            Type type = op->getResult(0).getType();
            target = dd::SumParticlesOp::create(
                body, loc, dd::PartialType::get(body.getContext(), type),
                inputs);
          } else {
            auto type = cast<md::FieldType>(op->getResult(0).getType());
            auto local = dd::FieldType::get(
                body.getContext(), stateType.getDecomposition(),
                dd::Role::Owned, false, type.getNumComponents(),
                type.getElementType());
            target = dd::MapParticlesOp::create(body, loc, local, inputs);
          }
          IRMapping inner(map);
          op->getRegion(0).cloneInto(&target->getRegion(0), inner);
          Operation *term = target->getRegion(0).front().getTerminator();
          OpBuilder replace(term);
          dd::YieldOp::create(replace, loc, term->getOperands());
          term->erase();
          if (sum)
            map.map(op->getResult(0), dd::AllReduceOp::create(
                                          body, loc, op->getResult(0).getType(),
                                          target->getResult(0))
                                          .getResult());
          else
            fields[op->getResult(0)] = target->getResult(0);
        };
        for (Operation &op : source.getBody()->without_terminator()) {
          if (auto nested = dyn_cast<scf::ForOp>(op)) {
            setState(nested.getInitArgs());
            e = unpack(body, groupLoop(body, nested, pack(body, e), map));
            for (auto [i, v] : llvm::enumerate(nested.getResults()))
              fields[v] = e.fields[i];
            for (auto [i, v] : llvm::enumerate(parameters))
              fields[v] = e.fields[3 + i];
          } else if (auto kick = dyn_cast<dyn::KickOp>(op)) {
            fields[kick.getResult()] =
                advance(body, fields.lookup(kick->getOperand(0)),
                        fields.lookup(kick->getOperand(1)),
                        map.lookupOrDefault(kick.getDt()),
                        fields.lookup(kick.getMasses()));
          } else if (auto drift = dyn_cast<dyn::DriftOp>(op)) {
            fields[drift.getResult()] =
                advance(body, fields.lookup(drift->getOperand(0)),
                        fields.lookup(drift->getOperand(1)),
                        map.lookupOrDefault(drift.getDt()), {});
          } else if (auto part = dyn_cast<dd::PartitionOp>(op)) {
            // Only the force neighborhood consumes geometric ghosts. Rigid
            // projections before and after it require the common group owner.
            bool pairs =
                llvm::any_of(part.getLayout().getUsers(), [](Operation *user) {
                  auto h = dyn_cast<dd::HaloOp>(user);
                  return h && llvm::any_of(h.getResult().getUsers(),
                                           [](Operation *u) {
                                             return isa<dd::NeighborhoodOp>(u);
                                           });
                });
            if (pairs) {
              SmallVector<Value> keys{part.getPositions()};
              for (auto &item : fields) {
                Value key = item.first;
                if (key == keys.front())
                  continue;
                bool live = llvm::is_contained(parameters, key) ||
                            llvm::any_of(key.getUsers(), [&](Operation *user) {
                              auto use =
                                  op.getBlock()->findAncestorOpInBlock(*user);
                              return use && op.isBeforeInBlock(use);
                            });
                if (live)
                  keys.push_back(key);
              }
              SmallVector<Type> types;
              e.fields.clear();
              for (Value key : keys) {
                Value field = fields.lookup(key);
                types.push_back(field.getType());
                e.fields.push_back(field);
              }
              auto saved = stateType;
              stateType = dd::StateType::get(
                  body.getContext(), saved.getDecomposition(), saved.getReach(),
                  saved.getCutoff(), types);
              e = rebuild(body, e);
              stateType = saved;
              for (auto [key, field] : llvm::zip(keys, e.fields))
                fields[key] = field;
            }
            map.map(part.getLayout(), e.layout);
          } else if (auto distribute = dyn_cast<dd::DistributeOp>(op)) {
            map.map(distribute.getResult(),
                    fields.lookup(distribute.getField()));
          } else if (auto halo = dyn_cast<dd::HaloOp>(op)) {
            map.map(halo.getResult(), e.halo);
          } else if (auto collect = dyn_cast<dd::CollectOp>(op)) {
            fields[collect.getResult()] = map.lookup(collect.getField());
          } else if (auto sum = dyn_cast<md::SumTuplesOp>(op);
                     sum && llvm::is_contained(groups, sum.getRelation())) {
            Value x = fields.lookup(sum->getOperand(1));
            auto ghost = [&](Value owned, Value image = {}) {
              auto t = cast<dd::FieldType>(owned.getType())
                           .with(dd::Role::Ghost, false);
              return dd::HaloCopyOp::create(body, loc, t, owned, e.halo, image)
                  .getResult();
            };
            auto schedule = MemRefType::get(
                {ShapedType::kDynamic, int64_t(sum.getArity() + 1)},
                body.getI64Type());
            Value rows = dd::TupleRowsOp::create(
                body, loc, schedule, sum.getRelation(), x, e.halo, true);
            SmallVector<Value> inputs;
            for (Value field : sum.getGathered()) {
              Value owned = fields.lookup(field);
              inputs.append({owned, ghost(owned)});
            }
            auto type = sum.getResult().getType();
            auto target = dd::TupleKernelOp::create(
                body, loc,
                TypeRange{dd::PartialType::get(body.getContext(), type)}, rows,
                x, ghost(x, cell), e.halo, inputs, sum.getParameters(),
                sum.getCoordinateKindsAttr(), sum.getCoordinateMembersAttr());
            IRMapping inner(map);
            sum.getKernel().cloneInto(&target.getKernel(), inner);
            Operation *term = target.getKernel().front().getTerminator();
            OpBuilder replace(term);
            dd::YieldOp::create(replace, loc, term->getOperands());
            term->erase();
            map.map(sum.getResult(), dd::AllReduceOp::create(
                                         body, loc, type, target.getResult(0))
                                         .getResult());
          } else if (auto perParticle = dyn_cast<md::MapParticlesOp>(op)) {
            mapKernel(&op, perParticle.getGathered(), false);
          } else if (auto sum = dyn_cast<md::SumParticlesOp>(op);
                     sum && llvm::all_of(sum.getGathered(), [&](Value v) {
                       return fields.count(v);
                     })) {
            mapKernel(&op, sum.getGathered(), true);
          } else {
            for (Value v : op.getOperands())
              if (Value field = fields.lookup(v))
                map.map(v, dd::CollectOp::create(body, loc, v.getType(), field,
                                                 e.layout)
                               .getResult());
            body.clone(op, map);
          }
        }
        setState(source.getBody()->getTerminator()->getOperands());
        scf::YieldOp::create(body, loc, pack(body, e));
      });
  return result.getResult(0);
}

LogicalResult Segment::run(scf::ForOp root) {
  dd::PartitionOp partition;
  dd::HaloOp halo;
  dd::NeighborhoodOp neighbors;
  dyn::KickOp kick;
  root.walk([&](dd::PartitionOp op) { partition = op; });
  root.walk([&](dd::HaloOp op) { halo = op; });
  root.walk([&](dd::NeighborhoodOp op) { neighbors = op; });
  root.walk([&](dyn::KickOp op) { kick = op; });
  if (!kick || !partition || !neighbors || !halo)
    return root.emitOpError("distributed Verlet requires a finite-range pair force");
  llvm::append_range(groups, partition.getGroups());
  cell = partition.getCell();
  if (wrapDrift && !cell.getDefiningOp<md::OrthorhombicCellOp>())
    return root.emitOpError("deterministic distribution requires an orthorhombic cell");
  parameters.push_back(kick.getMasses());
  bool unsupported = false;
  root.walk([&](scf::ForOp step) {
    if (step.getNumRegionIterArgs() != 3 &&
        (!groupSkin || step->getParentOp() == root))
      unsupported = true;
  });
  root.walk([&](dd::DistributeOp op) {
    if (op.getImage()) return;
    if (!root.isDefinedOutsideOfLoop(op.getField())) {
      if (!groupSkin)
        unsupported = true;
      return;
    }
    if (!llvm::is_contained(parameters, op.getField())) parameters.push_back(op.getField());
  });
  if (unsupported || !root.isDefinedOutsideOfLoop(cell))
    return root.emitOpError("distributed Verlet requires position, velocity, force and fixed particle parameters");
  OpBuilder b(root);
  SmallVector<Value> inputs(root.getInitArgs());
  llvm::append_range(inputs, parameters);
  Epoch e;
  e.layout = dd::PartitionOp::create(b, loc, partition.getLayout().getType(), inputs[0], cell, partition.getGroups());
  auto decomposition = partition.getLayout().getType().getDecomposition();
  SmallVector<Type> types;
  for (auto [i, v] : llvm::enumerate(inputs)) {
    auto t = cast<md::FieldType>(v.getType());
    auto owned = dd::FieldType::get(b.getContext(), decomposition, dd::Role::Owned,
                                    false, t.getNumComponents(), t.getElementType());
    types.push_back(owned);
    e.fields.push_back(dd::DistributeOp::create(b, loc, owned, v, e.layout, i == 0 ? cell : Value{}));
  }
  auto ht = halo.getResult().getType();
  double radius = neighbors.getCutoff().convertToDouble();
  if (neighbors.getResult().getType().getTraversal() == dd::Traversal::Midpoint)
    radius *= 0.5;
  if (groupSkin)
    radius = ht.getReach().convertToDouble() - groupSkin;
  stateType = dd::StateType::get(b.getContext(), decomposition, ht.getReach(), APFloat(radius), types);
  e.halo = dd::HaloOp::create(b, loc, ht, e.layout, e.fields[0], cell);
  Value initial = pack(b, e);
  e = unpack(b, groupSkin ? groupLoop(b, root, initial, {})
                          : loop(b, root, initial, {}));
  for (auto [i, result] : llvm::enumerate(root.getResults()))
    if (!result.use_empty())
      result.replaceAllUsesWith(dd::CollectOp::create(b, loc, result.getType(), e.fields[i], e.layout));
  root.erase();
  return success();
}

class DriverSteps : public impl::MDistributeDriverStepsBase<DriverSteps> {
public:
  using MDistributeDriverStepsBase::MDistributeDriverStepsBase;
  void runOnOperation() final {
    SmallVector<scf::ForOp> roots;
    getOperation().walk([&](scf::ForOp op) {
      // A checkpoint segment may first reorder the particle set, carrying
      // its permutation and parameters as well. Build the domain state
      // inside that segment, after the new order has been selected.
      auto parent = op->getParentOfType<scf::ForOp>();
      if (op.getNumRegionIterArgs() == 3 &&
          (!parent || parent.getNumRegionIterArgs() != 3) &&
          op.walk([](dyn::DriftOp) { return WalkResult::interrupt(); }).wasInterrupted())
        roots.push_back(op);
    });
    if (roots.empty()) {
      getOperation().emitError("distributed dynamics requires a velocity Verlet step loop");
      return signalPassFailure();
    }
    for (auto root : roots)
      if (failed(Segment(root.getLoc(), wrapDrift, groupSkin).run(root)))
        return signalPassFailure();
  }
};
} // namespace
} // namespace mdir
