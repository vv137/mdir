// Each step consumes its loop's physical state through actual region arguments.
#include "SimpleStepEntryIR.h"
#include "EntryFieldsIR.h"
#include "EntryLoopIR.h"
#include "ReorderIR.h"
#include "DynamicsIR.h"
#include "BarostatIR.h"
#include "CouplingIR.h"
#include "KineticIR.h"
#include "PotentialIR.h"
#include "ScheduleIR.h"
#include "mdir/Driver/Builder.h"
#include "mdir/Dialect/MDRT/MDRTOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/ScopeExit.h"
#include <functional>
using namespace mlir;
using namespace mdir::driver;
using namespace mdir::driver::builder;

llvm::Expected<func::FuncOp> mdir::driver::builder::buildSimpleStepEntry(
    OpBuilder &builder, const Program &program, mdir::dyn::ProgramOp step,
    mdir::dyn::ProgramOp energyStep, const SimpleStepEntryIRConfig &config) {
  if (!step || (config.energyIntervals && !energyStep))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
        "a generated step has no program definition");
  return buildEntryFunction(builder, program, {},
      [&](OpBuilder &at, const EntryArgumentsIR &arguments) -> llvm::Error {
    auto loc = at.getUnknownLoc();
    Value zero = arith::ConstantIndexOp::create(at, loc, 0);
    Value one = arith::ConstantIndexOp::create(at, loc, 1);
    enum class Kind { Segment, Frame, Energy, Step };
    SmallVector<Kind> kinds;
    SmallVector<int64_t> counts;
    if (config.checkpointIntervals) {
      kinds.push_back(Kind::Segment);
      counts.push_back(*config.checkpointIntervals);
    }
    if (config.frameIntervals) {
      kinds.push_back(Kind::Frame);
      counts.push_back(*config.frameIntervals);
    }
    if (config.energyIntervals) {
      kinds.push_back(Kind::Energy);
      counts.push_back(*config.energyIntervals);
    }
    kinds.push_back(Kind::Step);
    counts.push_back(config.plainSteps);
    SmallVector<Value> extents, periods(kinds.size());
    for (int64_t count : counts) extents.push_back(arith::ConstantIndexOp::create(at, loc, count));
    int64_t taken = 1;
    for (size_t level = kinds.size(); level-- != 0;) {
      periods[level] = arith::ConstantIndexOp::create(at, loc, taken);
      taken *= counts[level];
      if (level && kinds[level - 1] == Kind::Energy) ++taken;
    }
    Value cell = buildCell(at, ValueRange{arguments.lengthX, arguments.lengthY, arguments.lengthZ});
    auto physical = buildEntryFields(at, program, arguments, true, false);
    SmallVector<Value> originalRelations;
    for (const auto &tuple : physical.tuples) originalRelations.push_back(tuple.members);
    physical = buildEntryReorder(at, program, std::move(physical), cell);
    auto fields = buildEntryPotentialFields(physical);
    PotentialStateIR state{physical.positions, cell, fields, {}};
    auto energy = buildEnergyEvaluation(at, state, {true, true, true});
    Value kinetic = buildKineticEnergy(at, physical.velocities, physical.masses);
    Value excess = buildKineticExcess(at, energy.forces, physical.masses, arguments.interval);
    Value correction = buildSum3(at, excess);
    auto diagonal = buildVirialDiagonal(at, energy.virial);
    Value trace = buildSum3(at, diagonal.vector);
    auto centers = buildGroupKinetic(at,
        {{}, physical.velocities, energy.forces, {}, physical.masses, arguments.interval, {}, {}});
    Value pressureKinetic = buildSum3(at, centers.half);
    func::CallOp::create(at, loc, "mdrtWriteEnergies", TypeRange{},
        ValueRange{arguments.start, energy.energy, kinetic, correction, trace, pressureKinetic});
    physical.forces = energy.forces;
    auto call = [&](OpBuilder &body, mdir::dyn::ProgramOp callee, const EntryFieldsIR &point) {
      SmallVector<Value> inputs{point.positions, point.velocities, point.forces, point.masses, cell, arguments.interval};
      llvm::append_range(inputs, buildEntryPotentialFields(point));
      return mdir::dyn::StepOp::create(body, loc, callee.getFunctionType().getResults(), callee.getSymName(), inputs);
    };
    auto bind = [](EntryFieldsIR point, ValueRange values, bool parameters) {
      point.positions = values[0];
      point.velocities = values[1];
      point.forces = values[2];
      if (parameters) {
        point.masses = values[3];
        for (auto [index, field] : llvm::enumerate(point.fields)) field = values[4 + index];
        point.identities = values.back();
      }
      return point;
    };
    auto pack = [](const EntryFieldsIR &point, bool parameters) {
      SmallVector<Value> values{point.positions, point.velocities, point.forces};
      if (parameters) {
        values.push_back(point.masses);
        llvm::append_range(values, point.fields);
        values.push_back(point.identities);
      }
      return values;
    };
    SmallVector<LoopIterationIR> iterations;
    auto number = [&](OpBuilder &body, unsigned level, Value index) {
      return buildStepNumber(body, iterations, index, {}, periods[level], one, arguments.start);
    };
    auto writeFrame = [&](OpBuilder &body, Value completed, const EntryFieldsIR &point) {
      mdir::mdrt::HostCallOp::create(body, loc, TypeRange{},
          FlatSymbolRefAttr::get(body.getContext(), "mdrtWriteFrame"),
          ValueRange{completed, point.positions, point.identities}, ValueRange{}, UnitAttr{});
    };
    std::function<llvm::Expected<scf::ForOp>(OpBuilder &, unsigned, EntryFieldsIR)> level;
    level = [&](OpBuilder &body, unsigned atLevel, EntryFieldsIR initial) -> llvm::Expected<scf::ForOp> {
      bool reorders = kinds[atLevel] == Kind::Segment && program.reorders;
      StringRef scope = kinds[atLevel] == Kind::Segment ? "checkpoint segments" : kinds[atLevel] == Kind::Frame ? "frame intervals" :
          kinds[atLevel] == Kind::Energy ? "energy intervals" : "plain steps";
      return buildEntryLoop(body, scope, zero, extents[atLevel], one, pack(initial, reorders),
          [&](OpBuilder &inside, Value index, ValueRange current) -> llvm::Expected<SmallVector<Value>> {
        iterations.push_back({index, extents[atLevel]});
        auto leave = llvm::make_scope_exit([&] { iterations.pop_back(); });
        auto point = bind(initial, current, reorders);
        if (reorders) {
          for (auto [index, relation] : llvm::enumerate(originalRelations)) point.tuples[index].members = relation;
          point = buildEntryReorder(inside, program, std::move(point), cell, true);
        }
        if (kinds[atLevel] == Kind::Step)
          return SmallVector<Value>(call(inside, step, point).getResults());
        auto before = level(inside, atLevel + 1, point);
        if (!before) return before.takeError();
        point = bind(std::move(point), before->getResults(), false);
        if (kinds[atLevel] == Kind::Segment) {
          Value completed = number(inside, atLevel, index);
          mdir::mdrt::HostCallOp::create(inside, loc, TypeRange{},
              FlatSymbolRefAttr::get(inside.getContext(), "mdrtWriteCheckpoint"),
              ValueRange{completed, point.positions, point.velocities, point.forces, point.identities}, ValueRange{}, UnitAttr{});
          return pack(point, reorders);
        }
        if (kinds[atLevel] == Kind::Frame) {
          Value completed = number(inside, atLevel, index);
          writeFrame(inside, completed, point);
          return SmallVector<Value>(before->getResults());
        }
        auto next = call(inside, energyStep, point);
        Value kinetic = buildKineticEnergy(inside, next.getResult(1), point.masses);
        Value excess = buildKineticExcess(inside, next.getResult(2), point.masses, arguments.interval);
        Value correction = buildSum3(inside, excess);
        Value trace = buildSum3(inside, next.getResult(5));
        auto centers = buildGroupKinetic(inside,
            {{}, next.getResult(1), next.getResult(2), {}, point.masses, arguments.interval, {}, {}});
        Value pressureKinetic = buildSum3(inside, centers.half);
        Value completed = number(inside, atLevel, index);
        func::CallOp::create(inside, loc, "mdrtWriteEnergies", TypeRange{},
            ValueRange{completed, next.getResult(3), kinetic, correction, trace, pressureKinetic});
        if (!config.periodic && !config.framesAtEnergies)
          mdir::mdrt::HostCallOp::create(inside, loc, TypeRange{},
              FlatSymbolRefAttr::get(inside.getContext(), "mdrtCheckSpread"),
              ValueRange{completed, next.getResult(0), point.identities}, ValueRange{}, UnitAttr{});
        if (config.framesAtEnergies) writeFrame(inside, completed, bind(point, next.getResults(), false));
        return SmallVector<Value>(next.getResults().take_front(3));
      });
    };
    auto loop = level(at, 0, physical);
    if (!loop) return loop.takeError();
    (*loop)->setAttr(config.checkpointIntervals ? "mdrt.segment" : "mdrt.fresh", at.getUnitAttr());
    auto final = bind(physical, loop->getResults(), config.checkpointIntervals && program.reorders);
    if (config.checkpointIntervals && program.reorders)
      for (Value relation : originalRelations) buildRenumber(at, relation, final.identities);
    mdir::mdrt::HostCallOp::create(at, loc, TypeRange{},
        FlatSymbolRefAttr::get(at.getContext(), "mdrtFinish"),
        ValueRange{final.positions, final.velocities, final.identities}, ValueRange{}, UnitAttr{});
    return llvm::Error::success();
  });
}
