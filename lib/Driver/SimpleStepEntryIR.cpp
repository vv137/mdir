// Each step consumes its loop's physical state through actual region arguments.
#include "SimpleStepEntryIR.h"
#include "EntryFieldsIR.h"
#include "EntryLoopIR.h"
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
    Value intervals;
    if (config.energyIntervals)
      intervals = arith::ConstantIndexOp::create(at, loc, *config.energyIntervals);
    Value extent = arith::ConstantIndexOp::create(at, loc, config.plainSteps);
    arith::ConstantIndexOp::create(at, loc, 1);
    Value period;
    if (config.energyIntervals)
      period = arith::ConstantIndexOp::create(at, loc, config.plainSteps + 1);
    Value cell = buildCell(at, ValueRange{arguments.lengthX, arguments.lengthY, arguments.lengthZ});
    auto physical = buildEntryFields(at, program, arguments, true, false);
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
    auto call = [&](OpBuilder &body, mdir::dyn::ProgramOp callee, ValueRange current) {
      SmallVector<Value> inputs{current[0], current[1], current[2], physical.masses, cell, arguments.interval};
      llvm::append_range(inputs, fields);
      return mdir::dyn::StepOp::create(body, loc, callee.getFunctionType().getResults(), callee.getSymName(), inputs);
    };
    auto plain = [&](OpBuilder &body, ValueRange initial) {
      return buildEntryLoop(body, "plain steps", zero, extent, one, initial,
          [&](OpBuilder &inside, Value, ValueRange current) -> llvm::Expected<SmallVector<Value>> {
        return SmallVector<Value>(call(inside, step, current).getResults());
      });
    };
    SmallVector<Value> initial{physical.positions, physical.velocities, energy.forces};
    auto loop = [&]() -> llvm::Expected<scf::ForOp> {
      if (!config.energyIntervals) return plain(at, initial);
      return buildEntryLoop(at, "energy intervals", zero, intervals, one, initial,
          [&](OpBuilder &body, Value index, ValueRange current) -> llvm::Expected<SmallVector<Value>> {
        auto before = plain(body, current);
        if (!before) return before.takeError();
        auto next = call(body, energyStep, before->getResults());
        Value kinetic = buildKineticEnergy(body, next.getResult(1), physical.masses);
        Value excess = buildKineticExcess(body, next.getResult(2), physical.masses, arguments.interval);
        Value correction = buildSum3(body, excess);
        Value trace = buildSum3(body, next.getResult(5));
        auto centers = buildGroupKinetic(body,
            {{}, next.getResult(1), next.getResult(2), {}, physical.masses, arguments.interval, {}, {}});
        Value pressureKinetic = buildSum3(body, centers.half);
        Value completed = buildStepNumber(body, {{index, {}}}, index, {}, period, one, arguments.start);
        func::CallOp::create(body, loc, "mdrtWriteEnergies", TypeRange{},
            ValueRange{completed, next.getResult(3), kinetic, correction, trace, pressureKinetic});
        if (!config.periodic)
          mdir::mdrt::HostCallOp::create(body, loc, TypeRange{},
              FlatSymbolRefAttr::get(body.getContext(), "mdrtCheckSpread"),
              ValueRange{completed, next.getResult(0), physical.identities}, ValueRange{}, UnitAttr{});
        return SmallVector<Value>(next.getResults().take_front(3));
      });
    }();
    if (!loop) return loop.takeError();
    (*loop)->setAttr("mdrt.fresh", at.getUnitAttr());
    mdir::mdrt::HostCallOp::create(at, loc, TypeRange{},
        FlatSymbolRefAttr::get(at.getContext(), "mdrtFinish"),
        ValueRange{loop->getResult(0), loop->getResult(1), physical.identities}, ValueRange{}, UnitAttr{});
    return llvm::Error::success();
  });
}
