// Each step consumes its loop's physical state through actual region arguments.
#include "SimpleStepEntryIR.h"
#include "EntryFieldsIR.h"
#include "EntryLoopIR.h"
#include "DynamicsIR.h"
#include "BarostatIR.h"
#include "CouplingIR.h"
#include "KineticIR.h"
#include "PotentialIR.h"
#include "mdir/Driver/Builder.h"
#include "mdir/Dialect/MDRT/MDRTOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
using namespace mlir;
using namespace mdir::driver;
using namespace mdir::driver::builder;

llvm::Expected<func::FuncOp> mdir::driver::builder::buildSimpleStepEntry(
    OpBuilder &builder, const Program &program, mdir::dyn::ProgramOp step, int64_t count) {
  if (!step)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
        "a generated step has no program definition");
  return buildEntryFunction(builder, program, {},
      [&](OpBuilder &at, const EntryArgumentsIR &arguments) -> llvm::Error {
    auto loc = at.getUnknownLoc();
    Value zero = arith::ConstantIndexOp::create(at, loc, 0);
    Value one = arith::ConstantIndexOp::create(at, loc, 1);
    Value extent = arith::ConstantIndexOp::create(at, loc, count);
    arith::ConstantIndexOp::create(at, loc, 1);
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
    auto loop = buildEntryLoop(at, "plain steps", zero, extent, one,
        ValueRange{physical.positions, physical.velocities, energy.forces},
        [&](OpBuilder &body, Value, ValueRange current) -> llvm::Expected<SmallVector<Value>> {
      SmallVector<Value> inputs{current[0], current[1], current[2], physical.masses, cell, arguments.interval};
      llvm::append_range(inputs, fields);
      auto next = mdir::dyn::StepOp::create(body, loc, step.getFunctionType().getResults(), step.getSymName(), inputs);
      return SmallVector<Value>(next.getResults());
    });
    if (!loop) return loop.takeError();
    (*loop)->setAttr("mdrt.fresh", at.getUnitAttr());
    mdir::mdrt::HostCallOp::create(at, loc, TypeRange{},
        FlatSymbolRefAttr::get(at.getContext(), "mdrtFinish"),
        ValueRange{loop->getResult(0), loop->getResult(1), physical.identities}, ValueRange{}, UnitAttr{});
    return llvm::Error::success();
  });
}
