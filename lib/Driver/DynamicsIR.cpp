// Mass-metric descent with the existing scaled 16-norm and rejected trials.
#include "DynamicsIR.h"
#include "PotentialIR.h"
#include "BarostatIR.h"
#include "SiteIR.h"
#include "llvm/ADT/STLExtras.h"
#include "mdir/Dialect/Dyn/DynOps.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include <limits>
using namespace mlir;
using namespace mdir::driver::builder;
namespace {
class Kernel {
public:
  explicit Kernel(OpBuilder &builder)
      : builder(builder), loc(builder.getUnknownLoc()),
        vec(VectorType::get({3}, builder.getF64Type())) {}
  Value mul(Value a, Value b) { return arith::MulFOp::create(builder, loc, a, b); }
  Value div(Value a, Value b) { return arith::DivFOp::create(builder, loc, a, b); }
  Value sqrt(Value a) { return math::SqrtOp::create(builder, loc, a); }
  Value squareNorm(Value a) {
    Value square = mul(a, a);
    return vector::ReductionOp::create(builder, loc, vector::CombiningKind::ADD, square);
  }
  Value sum(Value gradient, Value mean = {}) {
    auto sum = mdir::md::SumParticlesOp::create(builder, loc, builder.getF64Type(), ValueRange{gradient});
    {
      OpBuilder::InsertionGuard guard(builder);
      auto block = builder.createBlock(&sum.getKernel(), {}, {vec}, {loc});
      Value result = squareNorm(block->getArgument(0));
      if (mean) {
        Value r2 = div(result, mean);
        Value r4 = mul(r2, r2), r8 = mul(r4, r4);
        result = mul(r8, r8);
      }
      mdir::md::YieldOp::create(builder, loc, ValueRange{result});
    }
    return sum.getResult();
  }
private:
  OpBuilder &builder;
  Location loc;
  VectorType vec;
};
} // namespace
Value mdir::driver::builder::buildDescentPositions(
    OpBuilder &builder, Value positions, Value gradient, Value distance,
    Value count, Value tiny) {
  auto loc = builder.getUnknownLoc();
  auto vec = VectorType::get({3}, builder.getF64Type());
  Kernel k(builder);
  Value square = k.sum(gradient);
  Value mean = k.div(square, count);
  Value safe = arith::MaximumFOp::create(builder, loc, mean, tiny);
  Value rms = k.sqrt(safe);
  Value power = k.sum(gradient, safe);
  Value root2 = k.sqrt(power), root4 = k.sqrt(root2);
  Value root8 = k.sqrt(root4), root16 = k.sqrt(root8);
  Value norm = k.mul(rms, root16);
  Value scale = k.div(distance, norm);
  auto map = mdir::md::MapParticlesOp::create(builder, loc, positions.getType(), ValueRange{positions, gradient});
  {
    OpBuilder::InsertionGuard guard(builder);
    auto args = builder.createBlock(&map.getKernel(), {}, {vec, vec}, {loc, loc})->getArguments();
    Value broadcast = vector::BroadcastOp::create(builder, loc, vec, scale);
    Value change = k.mul(broadcast, args[1]);
    Value moved = arith::AddFOp::create(builder, loc, args[0], change);
    mdir::md::YieldOp::create(builder, loc, ValueRange{moved});
  }
  return map.getResult();
}
ConstrainedTrialIR mdir::driver::builder::buildConstrainedTrial(
    OpBuilder &builder, Value candidate, Value previous, Value broken, Value zero) {
  auto loc = builder.getUnknownLoc();
  auto vec = VectorType::get({3}, builder.getF64Type());
  Value intact = arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::OEQ, broken, zero);
  auto map = mdir::md::MapParticlesOp::create(builder, loc, candidate.getType(), ValueRange{candidate, previous});
  {
    OpBuilder::InsertionGuard guard(builder);
    auto args = builder.createBlock(&map.getKernel(), {}, {vec, vec}, {loc, loc})->getArguments();
    Value localZero = arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(0.0));
    Value test = arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::OEQ, broken, localZero);
    Value chosen = arith::SelectOp::create(builder, loc, test, args[0], args[1]);
    mdir::md::YieldOp::create(builder, loc, ValueRange{chosen});
  }
  return {map.getResult(), intact};
}
Value mdir::driver::builder::buildTrialEnergy(OpBuilder &builder, Value energy, Value intact) {
  auto loc = builder.getUnknownLoc();
  Value never = arith::ConstantOp::create(builder, loc,
      builder.getF64FloatAttr(std::numeric_limits<double>::infinity()));
  return arith::SelectOp::create(builder, loc, intact, energy, never);
}

Value mdir::driver::builder::buildKick(
    OpBuilder &builder, Value velocities, Value forces, Value masses, Value interval) {
  return mdir::dyn::KickOp::create(builder, builder.getUnknownLoc(), velocities.getType(),
                                    velocities, forces, masses, interval);
}
Value mdir::driver::builder::buildDrift(
    OpBuilder &builder, Value positions, Value velocities, Value interval) {
  return mdir::dyn::DriftOp::create(builder, builder.getUnknownLoc(), positions.getType(),
                                     positions, velocities, interval);
}
DriftScaleIR mdir::driver::builder::buildDriftScale(OpBuilder &builder, Value scale) {
  auto loc = builder.getUnknownLoc();
  auto vec = VectorType::get({3}, builder.getF64Type());
  Value unit = arith::ConstantOp::create(builder, loc, DenseElementsAttr::get(vec, 1.0));
  Value oneMinus = arith::SubFOp::create(builder, loc, unit, scale);
  Value inverse = arith::DivFOp::create(builder, loc, unit, scale);
  return {oneMinus, inverse};
}
Value mdir::driver::builder::buildVelocityScaling(OpBuilder &builder, Value velocities, Value scale) {
  auto loc = builder.getUnknownLoc();
  auto vec = VectorType::get({3}, builder.getF64Type());
  auto map = mdir::md::MapParticlesOp::create(builder, loc, velocities.getType(), ValueRange{velocities});
  {
    OpBuilder::InsertionGuard guard(builder);
    auto args = builder.createBlock(&map.getKernel(), {}, {vec}, {loc})->getArguments();
    Value result = arith::MulFOp::create(builder, loc, scale, args[0]);
    mdir::md::YieldOp::create(builder, loc, ValueRange{result});
  }
  return map.getResult();
}
Value mdir::driver::builder::buildFieldDifference(OpBuilder &builder, Value first, Value second) {
  auto loc = builder.getUnknownLoc();
  auto vec = VectorType::get({3}, builder.getF64Type());
  auto map = mdir::md::MapParticlesOp::create(builder, loc, first.getType(), ValueRange{first, second});
  {
    OpBuilder::InsertionGuard guard(builder);
    auto args = builder.createBlock(&map.getKernel(), {}, {vec, vec}, {loc, loc})->getArguments();
    Value result = arith::SubFOp::create(builder, loc, args[0], args[1]);
    mdir::md::YieldOp::create(builder, loc, ValueRange{result});
  }
  return map.getResult();
}

Value mdir::driver::builder::buildDescentDirection(OpBuilder &builder, Value forces, Value masses) {
  using namespace mlir;
  auto loc = builder.getUnknownLoc();
  auto f64 = builder.getF64Type();
  auto vec = VectorType::get({3}, f64);
  auto map = mdir::md::MapParticlesOp::create(
      builder, loc, forces.getType(), ValueRange{forces, masses});
  {
    OpBuilder::InsertionGuard guard(builder);
    auto args = builder.createBlock(&map.getKernel(), {}, {vec, f64},
                                      {loc, loc})->getArguments();
    Value zero = arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(0.0));
    Value unit = arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(1.0));
    Value zeroVector = arith::ConstantOp::create(
        builder, loc, DenseElementsAttr::get(vec, 0.0));
    Value massive = arith::CmpFOp::create(builder, loc,
                                          arith::CmpFPredicate::OGT,
                                          args[1], zero);
    Value safe = arith::SelectOp::create(builder, loc, massive, args[1], unit);
    Value mass = vector::BroadcastOp::create(builder, loc, vec, safe);
    Value ratio = arith::DivFOp::create(builder, loc, args[0], mass);
    Value result = arith::SelectOp::create(builder, loc, massive, ratio,
                                           zeroVector);
    mdir::md::YieldOp::create(builder, loc, ValueRange{result});
  }
  return map.getResult();
}

mdir::dyn::ProgramOp mdir::driver::builder::buildNativeDescent(
    OpBuilder &builder, TypeRange parameters, double particles, bool timed,
    llvm::ArrayRef<DescentConstraintGroupIR> groups, llvm::ArrayRef<DescentSiteIR> sites, bool hosted,
    llvm::function_ref<EnergyResults(OpBuilder &, const PotentialStateIR &, EnergyResults)> complete) {
  auto loc = builder.getUnknownLoc();
  auto ctx = builder.getContext();
  auto atoms = FlatSymbolRefAttr::get(ctx, "atoms");
  auto f64 = builder.getF64Type();
  auto vec = mdir::md::FieldType::get(ctx, atoms, 3, f64);
  auto real = mdir::md::FieldType::get(ctx, atoms, 1, f64);
  SmallVector<Type> inputs{vec, vec, real, mdir::md::CellType::get(ctx), f64};
  llvm::append_range(inputs, parameters);
  auto function = builder.getFunctionType(inputs, {vec, vec, f64});
  auto program = mdir::dyn::ProgramOp::create(
      builder, loc, "descend", function, StringAttr{}, ArrayAttr{}, ArrayAttr{},
      FloatAttr{}, ArrayAttr{}, ArrayAttr{}, hosted
          ? builder.getDictionaryAttr({builder.getNamedAttr("read", builder.getStrArrayAttr({"<Default>"})),
                                       builder.getNamedAttr("write", builder.getStrArrayAttr({"<Default>"}))})
          : builder.getDictionaryAttr({}));
  {
    OpBuilder::InsertionGuard guard(builder);
    SmallVector<Location> locations(inputs.size(), loc);
    auto args = builder.createBlock(&program.getBody(), {}, inputs, locations)->getArguments();
    Value zero = arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(0.0));
    Value tiny = arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(1.0e-300));
    Value count = arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(particles));
    auto parameter = [&](unsigned slot) { return args[5 + slot]; };
    auto groupParameters = [&](const DescentConstraintGroupIR &group) {
      SmallVector<Value> fields;
      for (unsigned slot : group.parameters) fields.push_back(parameter(slot));
      return fields;
    };
    Value direction = buildDescentDirection(builder, args[1], args[2]);
    for (const auto &group : groups) {
      auto fields = groupParameters(group);
      ConstraintVelocityInputs input{parameter(group.relation), args[0], args[3],
                                      direction, args[2], fields, {}, {}};
      direction = group.water ? buildSettleVelocities(builder, input).velocities
                              : buildShakeVelocities(builder, input, group.arity).velocities;
    }
    Value positions = buildDescentPositions(builder, args[0], direction, args[4], count, tiny);
    for (const auto &group : groups) {
      auto fields = groupParameters(group);
      ConstraintPositionInputs input{parameter(group.relation), positions, args[3],
                                      args[0], args[2], fields};
      auto projected = group.settlePosition
                           ? buildSettlePositions(builder, input, false, group.position.tolerance)
                           : buildBondConstraints(builder, input, group.arity, group.bonds, group.position);
      positions = buildAddChange(builder, positions, projected.change);
    }
    Value intact;
    if (!groups.empty()) {
      SmallVector<Value> failures;
      for (const auto &group : groups) {
        auto fields = groupParameters(group);
        failures.push_back(buildConstraintTest(builder,
            {parameter(group.relation), positions, args[3], fields}, group.bonds,
            false, group.position.tolerance));
      }
      Value total;
      for (Value failed : failures)
        total = total ? arith::AddFOp::create(builder, loc, total, failed) : failed;
      Value none = arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(0.0));
      Value broken = arith::AddFOp::create(builder, loc, total, none);
      auto trial = buildConstrainedTrial(builder, positions, args[0], broken, zero);
      positions = trial.positions;
      intact = trial.intact;
    }
    for (const auto &site : sites)
      positions = buildPlacedSites(builder,
          {parameter(site.relation), positions, args[3], parameter(site.a), parameter(site.b)}, site.amber);
    auto evaluation = buildEnergyEvaluation(
        builder, {positions, args[3], ValueRange(args).drop_front(5), timed ? zero : Value{}},
        {true, true, false});
    Value forces = evaluation.forces;
    for (const auto &site : sites)
      forces = buildSiteSpread(builder,
          {parameter(site.relation), positions, args[3], parameter(site.a), parameter(site.b), forces, {}},
          site.amber, site.derived, site.potential).forces;
    evaluation.forces = forces;
    if (complete)
      evaluation = complete(builder,
          {positions, args[3], ValueRange(args).drop_front(5), timed ? zero : Value{}}, evaluation);
    forces = evaluation.forces;
    Value energy = groups.empty() ? evaluation.energy
                                  : buildTrialEnergy(builder, evaluation.energy, intact);
    mdir::dyn::ReturnOp::create(builder, loc, ValueRange{positions, forces, energy});
  }
  return program;
}

mdir::dyn::ProgramOp mdir::driver::builder::buildNativeVerletStep(
    OpBuilder &builder, StringRef name, TypeRange parameters,
    bool energy, bool leapfrog, bool timed, llvm::ArrayRef<DescentSiteIR> sites) {
  auto loc = builder.getUnknownLoc();
  auto ctx = builder.getContext();
  auto atoms = FlatSymbolRefAttr::get(ctx, "atoms");
  auto f64 = builder.getF64Type();
  auto vec = mdir::md::FieldType::get(ctx, atoms, 3, f64);
  auto real = mdir::md::FieldType::get(ctx, atoms, 1, f64);
  SmallVector<Type> inputs{vec, vec, vec, real, mdir::md::CellType::get(ctx), f64};
  llvm::append_range(inputs, parameters);
  if (timed) {
    inputs.push_back(builder.getI64Type());
    inputs.push_back(mdir::md::FieldType::get(ctx, atoms, 1, builder.getI32Type()));
  }
  SmallVector<Type> outputs{vec, vec, vec};
  if (energy) {
    outputs.push_back(f64);
    outputs.push_back(VectorType::get({9}, f64));
    outputs.push_back(VectorType::get({3}, f64));
    if (leapfrog) outputs.push_back(vec);
  }
  auto program = mdir::dyn::ProgramOp::create(builder, loc, name,
      builder.getFunctionType(inputs, outputs), StringAttr{}, ArrayAttr{}, ArrayAttr{},
      leapfrog ? builder.getF64FloatAttr(-0.5) : FloatAttr{}, ArrayAttr{},
      builder.getStrArrayAttr({"symplectic", "time_reversible"}), builder.getDictionaryAttr({}));
  OpBuilder::InsertionGuard guard(builder);
  SmallVector<Location> locations(inputs.size(), loc);
  auto args = builder.createBlock(&program.getBody(), {}, inputs, locations)->getArguments();
  Value half = arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(0.5));
  half = arith::MulFOp::create(builder, loc, half, args[5]);
  Value time = timed ? buildSimulationTime(builder, args[6 + parameters.size()], args[5]) : Value{};
  Value kicked = buildKick(builder, args[1], args[2], args[3], leapfrog ? args[5] : half);
  Value positions = buildDrift(builder, args[0], kicked, args[5]);
  auto parameter = [&](unsigned slot) { return args[6 + slot]; };
  for (const auto &site : sites)
    positions = buildPlacedSites(builder,
        {parameter(site.relation), positions, args[4], parameter(site.a), parameter(site.b)}, site.amber);
  auto evaluation = buildEnergyEvaluation(builder,
      {positions, args[4], ValueRange(args).slice(6, parameters.size()), time}, {energy, true, energy});
  for (const auto &site : sites) {
    auto spread = buildSiteSpread(builder,
        {parameter(site.relation), positions, args[4], parameter(site.a), parameter(site.b),
         evaluation.forces, evaluation.virial}, site.amber, site.derived, site.potential);
    evaluation.forces = spread.forces;
    evaluation.virial = spread.virial;
  }
  Value diagonal;
  if (energy) diagonal = buildVirialDiagonal(builder, evaluation.virial).vector;
  if (leapfrog && !energy) {
    mdir::dyn::ReturnOp::create(builder, loc, ValueRange{positions, kicked, evaluation.forces});
    return program;
  }
  Value current = buildKick(builder, kicked, evaluation.forces, args[3], half);
  SmallVector<Value> results{positions, leapfrog ? kicked : current, evaluation.forces};
  if (energy) {
    results.append({evaluation.energy, evaluation.virial, diagonal});
    if (leapfrog) results.push_back(current);
  }
  mdir::dyn::ReturnOp::create(builder, loc, results);
  return program;
}
