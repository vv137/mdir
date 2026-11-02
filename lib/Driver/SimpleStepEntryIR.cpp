// Each step consumes its loop's physical state through actual region arguments.
#include "SimpleStepEntryIR.h"
#include "EntryFieldsIR.h"
#include "EntryLoopIR.h"
#include "EntryRuntimeIR.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "ReorderIR.h"
#include "DynamicsIR.h"
#include "BarostatIR.h"
#include "CouplingIR.h"
#include "KineticIR.h"
#include "PotentialIR.h"
#include "OutputIR.h"
#include "ScheduleIR.h"
#include "mdir/Driver/Builder.h"
#include "mdir/Dialect/MDRT/MDRTOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/ScopeExit.h"
#include <functional>
using namespace mlir;
using namespace mdir::driver;
using namespace mdir::driver::builder;

llvm::Expected<func::FuncOp> mdir::driver::builder::buildSimpleStepEntry(
    OpBuilder &builder, const Program &program, mdir::dyn::ProgramOp step,
    mdir::dyn::ProgramOp energyStep, const SimpleStepEntryIRConfig &config,
    llvm::function_ref<EntryCompletionIR(OpBuilder &, const PotentialStateIR &, EnergyResults, Value)> complete,
    llvm::function_ref<void(OpBuilder &, const PotentialStateIR &, Value)> coordinateOutput) {
  if (config.segments && config.restart)
    return llvm::createStringError(llvm::inconvertibleErrorCode(), segmentsStaticRestartError);
  if (config.needsCoordinateOutput && !coordinateOutput)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
        "native step entry requires the coordinate-output role");
  if (config.needsPostEvaluation && !complete)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
        "native step entry requires the post-evaluation role");
  if (config.needsHostedEvaluation && !complete)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
        "native step entry requires the hosted-evaluation role");
  if (config.segments && config.partCounts != 9)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
        "native step parts require nine host count slots");
  if (config.segments && (config.energyIntervals || config.frameIntervals || config.checkpointIntervals))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
        "native step parts require zero output periods");
  if (config.segments && !program.takesForces)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
        "native step parts require incoming force state");
  if (!step || ((config.energyIntervals || config.segments) && !energyStep))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
        "a generated step has no program definition");
  if (config.restart && !program.takesForces && !config.recomputeForces)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
        "native step restart has no force state");
  if (config.brownian && config.leapfrog)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
        "native step entry cannot combine Brownian and leapfrog");
  bool constrained = !config.constraints.empty();
  bool measured = measuresHalfSteps(constrained, config.brownian);
  auto couplingProgram = config.couplingStep;
  auto pressureProgram = config.pressureStep, scaledEnergyProgram = config.scaledEnergyStep;
  if (config.couplingIntervals && config.rescales && !config.bathParameters)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
        "native step entry thermostat requires bath parameters");
  if (config.barostat && (!config.couplingIntervals ||
      !config.barostatParameters || !config.barostatConstant || !config.bathParameters ||
      (config.exactBarostat && !config.barostatEnergyConstant)))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
        "native barostat requires coupling parameters");
  if (config.everyStep && config.segments)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
        "a program of segments with a barostat that scales the cell every step is not supported");
  if (config.everyStep && (!config.trotter || !config.countsAfterScaling))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
        "native period-one Trotter requires pressure state after scaling");
  bool measuredCoupling = measured && config.rescales && config.couplingIntervals.has_value();
  unsigned couplingResults = (config.trotter ? 4 + 2 * unsigned(config.countsAfterScaling) : config.barostat ? 6 : 3) +
      unsigned(config.leapfrog && (config.barostat || measuredCoupling)) + unsigned(measuredCoupling);
  if ((measuredCoupling || config.barostat) && (!couplingProgram ||
      couplingProgram.getFunctionType().getResults().size() != couplingResults))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
        "native step entry thermostat results do not match its integrator");
  bool solvent = measured && llvm::any_of(config.constraints, [](const auto &group) { return group.water; });
  bool needsCounter = config.timed || config.stepMemory || constrained;
  unsigned measuredSlot = 5 + unsigned(!config.brownian) + unsigned(config.leapfrog);
  unsigned reported = measuredSlot + unsigned(measured) + unsigned(solvent);
  if (step.getFunctionType().getResults().size() != 3 ||
      ((config.energyIntervals || config.segments) && energyStep.getFunctionType().getResults().size() != reported))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
        "native step entry dynamics results do not match its integrator (brownian=%d, leapfrog=%d, constrained=%d)",
        config.brownian, config.leapfrog, constrained);
  if (config.trotter && ((!config.everyStep && (!pressureProgram ||
      pressureProgram.getFunctionType().getResults().size() != 5 + unsigned(config.leapfrog))) ||
      !scaledEnergyProgram || scaledEnergyProgram.getFunctionType().getResults().size() != reported + 1))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
        "native Trotter entry requires pressure and scaled energy programs");
  SmallVector<unsigned> tiltSlots;
  if (config.triclinic)
    for (StringRef name : {"tilt_bx", "tilt_cx", "tilt_cy"}) {
      auto found = llvm::find_if(program.startValues,
          [&](const auto &value) { return value.name == name; });
      if (found == program.startValues.end())
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
            "native step entry is missing cell start value '%s'", name.str().c_str());
      tiltSlots.push_back(std::distance(program.startValues.begin(), found));
    }
  return buildEntryFunction(builder, program, {false, config.segments},
      [&](OpBuilder &at, const EntryArgumentsIR &arguments) -> llvm::Error {
    auto loc = at.getUnknownLoc();
    Value zero = arith::ConstantIndexOp::create(at, loc, 0);
    Value one = arith::ConstantIndexOp::create(at, loc, 1);
    Value firstCall;
    if (config.segments) {
      Value none = arith::ConstantIntOp::create(at, loc, 0, 64);
      firstCall = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ne, arguments.firstCall, none);
    }
    Value time, stepMemory, stepOne;
    if (config.timed) time = buildSimulationTime(at, arguments.start, arguments.interval);
    if (needsCounter) {
      stepMemory = memref::AllocaOp::create(at, loc, MemRefType::get({1}, at.getI64Type()));
      memref::StoreOp::create(at, loc, arguments.start, stepMemory, zero);
      stepOne = arith::ConstantIntOp::create(at, loc, 1, 64);
    }
    enum class Kind { Segment, Frame, Energy, Couple, Step };
    SmallVector<Kind> kinds;
    SmallVector<int64_t> counts;
    SmallVector<Value> extents, periods;
    if (!config.segments) {
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
    if (config.couplingIntervals) {
      kinds.push_back(Kind::Couple);
      counts.push_back(*config.couplingIntervals);
    }
    kinds.push_back(Kind::Step);
    counts.push_back(config.plainSteps);
    periods.resize(kinds.size());
    for (int64_t count : counts) extents.push_back(arith::ConstantIndexOp::create(at, loc, count));
    int64_t taken = 1;
    int64_t closingSteps = config.trotter && !config.everyStep ? 2 : 1;
    for (size_t level = kinds.size(); level-- != 0;) {
      periods[level] = arith::ConstantIndexOp::create(at, loc, taken);
      taken *= counts[level];
      if (level && kinds[level - 1] == Kind::Energy)
        taken += config.couplingIntervals ? config.plainSteps + closingSteps : 1;
      else if (level && kinds[level - 1] == Kind::Couple) taken += closingSteps;
    }
    }
    SmallVector<Value> edges{arguments.lengthX, arguments.lengthY, arguments.lengthZ};
    for (unsigned slot : tiltSlots) edges.push_back(arguments.startValues[slot]);
    Value boxMemory, barostatMemory, reference, scale, halfBack;
    SmallVector<Value> edgeIndices;
    if (config.barostat) {
      boxMemory = memref::AllocaOp::create(at, loc, MemRefType::get({int64_t(edges.size())}, at.getF64Type()));
      if (config.everyStep)
        barostatMemory = memref::AllocaOp::create(at, loc, MemRefType::get({9}, at.getF64Type()));
      for (auto [index, edge] : llvm::enumerate(edges)) {
        Value slot = arith::ConstantIndexOp::create(at, loc, index);
        memref::StoreOp::create(at, loc, edge, boxMemory, slot);
        edgeIndices.push_back(slot);
      }
    }
    if (config.referenceEdges) {
      SmallVector<Value> values;
      for (unsigned slot : *config.referenceEdges) values.push_back(arguments.startValues[slot]);
      reference = vector::FromElementsOp::create(at, loc, VectorType::get({3}, at.getF64Type()), values);
      scale = buildReferenceScale(at, edges, reference);
    }
    Value cell = buildCell(at, edges);
    Value entryCell = cell;
    auto readCell = [&](OpBuilder &body) {
      SmallVector<Value> current;
      for (Value index : edgeIndices) current.push_back(memref::LoadOp::create(body, loc, boxMemory, index));
      cell = buildCell(body, current);
      if (reference) scale = buildReferenceScale(body, current, reference);
    };
    bool givenForces = (config.restart || config.segments) && program.takesForces;
    auto physical = buildEntryFields(at, program, arguments, true, givenForces);
    SmallVector<Value> originalRelations;
    for (const auto &tuple : physical.tuples) originalRelations.push_back(tuple.members);
    Value inputMasses = physical.masses;
    auto originalParameters = buildEntryPotentialFields(physical);
    physical.positions = buildEntrySites(at,
        {physical.positions, cell, originalParameters, time}, config.sites);
    physical = buildEntryReorder(at, program, std::move(physical), cell, givenForces);
    auto fields = buildEntryPotentialFields(physical);
    CouplingMassIR coupling;
    Value seed, freedom, target, decay;
    BarostatParameterValues barostat;
    if (config.couplingIntervals) {
      coupling = buildCouplingMass(at, inputMasses);
      if (config.bathParameters) {
        if (config.barostat) barostat = buildBarostatParameters(at, *config.barostatParameters);
        seed = arith::ConstantIntOp::create(at, loc, config.seed, 64);
        freedom = arith::ConstantOp::create(at, loc, at.getF64FloatAttr(config.freedom));
        target = arith::ConstantOp::create(at, loc, at.getF64FloatAttr(config.targetKinetic));
        decay = arith::ConstantOp::create(at, loc, at.getF64FloatAttr(config.decay));
      }
    }
    if (config.barostat && config.leapfrog) {
      Value factor = arith::ConstantOp::create(at, loc, at.getF64FloatAttr(-0.5));
      halfBack = arith::MulFOp::create(at, loc, factor, arguments.interval);
    }
    Value pressureMass = config.removesMotion ? coupling.totalMass : Value{};
    auto storeInitialPressure = [&](OpBuilder &body, const EntryFieldsIR &point,
        const PotentialStateIR &state, EnergyResults energy, bool restartEstimate) {
      auto raw = buildVirialDiagonal(body, energy.virial);
      Value diagonal = raw.vector;
      if (constrained)
        diagonal = buildEntryConstraintTrace(body, point, state, energy.forces,
            diagonal, config.constraints, true);
      auto groups = buildEntryGroups(state.fields, config.constraints);
      Value kinetic;
      // The base retains distinct fresh and missing-state restart estimates.
      if (restartEstimate) {
        kinetic = buildKineticVector(body, point.velocities, point.masses);
        if (config.removesMotion) {
          Value momentum = buildMomentum(body, point.velocities, point.masses);
          Value velocity = arith::DivFOp::create(body, loc, momentum, coupling.totalMass);
          kinetic = buildAxesWithoutCenter(body, kinetic, momentum, velocity, barostat.halfVector);
        }
      } else kinetic = buildGroupKinetic(body,
          {point.positions, point.velocities, energy.forces, cell, point.masses,
           arguments.interval, pressureMass, groups}).half;
      Value molecular = buildMolecularDiagonal(body,
          {raw.vector, point.positions, energy.forces, cell, point.masses, groups});
      buildStoreBarostatState(body, barostatMemory, diagonal, molecular, kinetic);
    };
    auto initialize = [&](OpBuilder &at, EntryFieldsIR physical) -> llvm::Expected<EntryFieldsIR> {
    PotentialStateIR state{physical.positions, cell, fields, time};
    EntryCompletionIR evaluated{{}, {}};
    if (!config.restart || config.recomputeForces) {
      evaluated = buildEntryEvaluation(at, state, {true, true, true}, config.sites, complete, scale);
      if (evaluated.storedTerms.size() != config.storedTerms)
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
            "native step entry post-evaluation term count does not match its configuration");
      physical.forces = evaluated.physical.forces;
    }
    if (!config.restart) {
      auto energy = evaluated.physical;
      if (config.topology) {
        Value terms = buildTermEnergyBuffer(at, state, config.terms, config.terms.size() + config.storedTerms);
        for (auto [index, value] : llvm::enumerate(evaluated.storedTerms)) {
          Value slot = arith::ConstantIndexOp::create(at, loc, config.terms.size() + index);
          memref::StoreOp::create(at, loc, value, terms, slot);
        }
        buildWriteTerms(at, terms);
        auto diagonal = buildVirialDiagonal(at, energy.virial);
        func::CallOp::create(at, loc, "mdrtWriteVirial", TypeRange{},
            ValueRange{diagonal.xx, diagonal.yy, diagonal.zz});
      }
      Value kinetic = buildKineticEnergy(at, physical.velocities, physical.masses);
      Value correction, trace, pressureKinetic;
      if (config.brownian || constrained)
        correction = arith::ConstantOp::create(at, loc, at.getF64FloatAttr(0.0));
      else {
        Value excess = buildKineticExcess(at, energy.forces, physical.masses, arguments.interval);
        correction = buildSum3(at, excess);
      }
      if (config.brownian) {
        trace = buildVirialTrace(at, energy.virial);
        if (constrained)
          trace = buildEntryConstraintTrace(at, physical, state, energy.forces, trace, config.constraints, false);
        pressureKinetic = arith::AddFOp::create(at, loc, kinetic, correction);
      } else {
        auto diagonal = buildVirialDiagonal(at, energy.virial);
        auto groups = buildEntryGroups(fields, config.constraints);
        Value molecular = buildMolecularDiagonal(at,
            {diagonal.vector, physical.positions, energy.forces, cell, physical.masses, groups});
        trace = buildSum3(at, molecular);
        auto centers = buildGroupKinetic(at,
            {physical.positions, physical.velocities, energy.forces, cell, physical.masses,
             arguments.interval, pressureMass, groups});
        pressureKinetic = buildSum3(at, centers.half);
      }
      func::CallOp::create(at, loc, "mdrtWriteEnergies", TypeRange{},
          ValueRange{arguments.start, energy.energy, kinetic, correction, trace, pressureKinetic});
      if (coordinateOutput) coordinateOutput(at, state, arguments.start);
      buildEntryOutputs(at,
          {state, arguments.start, physical.identities, arguments.interval, zero, one,
           arguments.startValues, true, config.timed}, config.output);
      if (config.everyStep) storeInitialPressure(at, physical, state, energy, false);
    }
    if (config.leapfrog && !config.restart) {
      Value back = arith::ConstantOp::create(at, loc, at.getF64FloatAttr(-0.5));
      Value interval = arith::MulFOp::create(at, loc, back, arguments.interval);
      if (config.segments) {
        Value none = arith::ConstantOp::create(at, loc, at.getF64FloatAttr(0.0));
        Value first = arith::ConstantIntOp::create(at, loc, 1, 64);
        Value begins = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::eq, arguments.firstCall, first);
        interval = arith::SelectOp::create(at, loc, begins, interval, none);
      }
      physical.velocities = buildKick(at, physical.velocities, physical.forces, physical.masses, interval);
    }
    return physical;
    };
    if (config.segments) {
      Value trips = arith::SelectOp::create(at, loc, firstCall, one, zero);
      auto initial = buildEntryLoop(at, "step initial state", zero, trips, one,
          ValueRange{physical.forces, physical.velocities},
          [&](OpBuilder &body, Value, ValueRange current) -> llvm::Expected<SmallVector<Value>> {
        auto actual = physical;
        actual.forces = current[0];
        actual.velocities = current[1];
        auto result = initialize(body, actual);
        if (!result) return result.takeError();
        return SmallVector<Value>{result->forces, result->velocities};
      });
      if (!initial) return initial.takeError();
      (*initial)->setAttr("mdrt.segment", at.getUnitAttr());
      physical.forces = initial->getResult(0);
      physical.velocities = initial->getResult(1);
    } else {
      auto initial = initialize(at, physical);
      if (!initial) return initial.takeError();
      physical = std::move(*initial);
    }
    if (config.everyStep && config.restart) {
      if (config.barostatState) {
        SmallVector<Value> values;
        for (unsigned slot : *config.barostatState) values.push_back(arguments.startValues[slot]);
        auto vec = VectorType::get({3}, at.getF64Type());
        SmallVector<Value> vectors;
        for (unsigned axis = 0; axis != 3; ++axis)
          vectors.push_back(vector::FromElementsOp::create(at, loc, vec, ValueRange(values).slice(3 * axis, 3)));
        buildStoreBarostatState(at, barostatMemory, vectors[0], vectors[1], vectors[2]);
      } else {
        PotentialStateIR state{physical.positions, cell, fields, time};
        auto evaluated = buildEntryEvaluation(at, state, {true, true, true}, config.sites, complete, scale);
        auto point = physical;
        if (config.leapfrog) {
          Value half = arith::ConstantOp::create(at, loc, at.getF64FloatAttr(0.5));
          Value interval = arith::MulFOp::create(at, loc, half, arguments.interval);
          point.velocities = buildKick(at, point.velocities, evaluated.physical.forces, point.masses, interval);
        }
        storeInitialPressure(at, point, state, evaluated.physical, true);
      }
    }
    auto call = [&](OpBuilder &body, mdir::dyn::ProgramOp callee, const EntryFieldsIR &point, Value mu = {}) {
      SmallVector<Value> inputs{point.positions, point.velocities, point.forces, point.masses, cell, arguments.interval};
      if (mu) inputs.push_back(mu);
      if (scale) inputs.push_back(scale);
      llvm::append_range(inputs, buildEntryPotentialFields(point));
      if (needsCounter) {
        Value previous = memref::LoadOp::create(body, loc, stepMemory, zero);
        Value next = arith::AddIOp::create(body, loc, previous, stepOne);
        memref::StoreOp::create(body, loc, next, stepMemory, zero);
        inputs.push_back(next);
        inputs.push_back(point.identities);
      }
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
    struct ReportIR { Value kinetic, correction, step, diagonal; };
    auto report = [&](OpBuilder &inside, const EntryFieldsIR &point, mdir::dyn::StepOp next,
        llvm::function_ref<Value()> makeStep, bool plainRow, bool scaledRow = false, Value outputCell = {}) -> ReportIR {
        unsigned rowMeasuredSlot = measuredSlot + unsigned(scaledRow);
        if (plainRow && solvent) buildWriteSolvent(inside, next.getResult(rowMeasuredSlot + 1));
        Value full = next.getResult(config.leapfrog ? 6 : 1);
        Value kinetic = buildKineticEnergy(inside, full, point.masses);
        Value correction, trace, pressureKinetic;
        if (measured) {
          Value current = buildKineticVector(inside, full, point.masses);
          Value excess = arith::SubFOp::create(inside, loc, next.getResult(rowMeasuredSlot), current);
          correction = buildSum3(inside, excess);
        } else if (config.brownian)
          correction = arith::ConstantOp::create(inside, loc, inside.getF64FloatAttr(0.0));
        else {
          Value excess = buildKineticExcess(inside, next.getResult(2), point.masses, arguments.interval);
          correction = buildSum3(inside, excess);
        }
        if (!plainRow && solvent) buildWriteSolvent(inside, next.getResult(rowMeasuredSlot + 1));
        if (config.brownian) {
          trace = buildVirialTrace(inside, next.getResult(4));
          pressureKinetic = arith::AddFOp::create(inside, loc, kinetic, correction);
        } else {
          trace = buildSum3(inside, next.getResult(5));
          auto parameters = buildEntryPotentialFields(point);
          auto groups = buildEntryGroups(parameters, config.constraints);
          auto centers = buildGroupKinetic(inside,
              {next.getResult(0), full, next.getResult(2), cell, point.masses,
               arguments.interval, pressureMass, groups});
          pressureKinetic = buildSum3(inside, centers.half);
        }
        Value diagonal;
        if (config.barostat && !plainRow) diagonal = buildVirialDiagonal(inside, next.getResult(4)).vector;
        Value completed = makeStep();
        func::CallOp::create(inside, loc, "mdrtWriteEnergies", TypeRange{},
            ValueRange{completed, next.getResult(3), kinetic, correction, trace, pressureKinetic});
        if (!plainRow && coordinateOutput) {
          auto fields = buildEntryPotentialFields(point);
          Value time;
          if (config.timed && config.needsCoordinateOutput)
            time = buildSimulationTime(inside, completed, arguments.interval);
          coordinateOutput(inside, {next.getResult(0), outputCell ? outputCell : cell, fields, time}, completed);
        }
        auto outputFields = buildEntryPotentialFields(point);
        PotentialStateIR outputState{next.getResult(0), outputCell ? outputCell : cell, outputFields, {}};
        auto output = config.output;
        if (plainRow) output.freeEnergy.reset();
        buildEntryOutputs(inside,
            {outputState, completed, point.identities, arguments.interval, zero, one,
             arguments.startValues, false, config.timed}, output);
        if (!config.periodic && (plainRow || config.segments || !config.framesAtEnergies))
          mdir::mdrt::HostCallOp::create(inside, loc, TypeRange{},
              FlatSymbolRefAttr::get(inside.getContext(), "mdrtCheckSpread"),
              ValueRange{completed, next.getResult(0), point.identities}, ValueRange{}, UnitAttr{});
      return {kinetic, correction, completed, diagonal};
    };
    auto optimalEnergy = [&](OpBuilder &inside, Value kinetic, Value excess) -> Value {
      Value weight = arith::ConstantOp::create(inside, loc, inside.getF64FloatAttr(2.0 / 3.0));
      Value change = arith::MulFOp::create(inside, loc, weight, excess);
      return arith::AddFOp::create(inside, loc, kinetic, change);
    };
    struct TrotterIR {
      mdir::dyn::StepOp step;
      Value completed, cell, workBefore, volume, kineticHalf, groupsAfter;
      StrainIRResult strain;
    };
    auto trotterSteps = [&](OpBuilder &inside, const EntryFieldsIR &point, bool energy,
        llvm::function_ref<Value()> makeStep) -> TrotterIR {
      auto nextPoint = point;
      Value groups, kinetic, pressureFull;
      if (config.everyStep) {
        // Retain all three pressure vectors in the base's load/assembly order.
        SmallVector<Value> values;
        for (int64_t slot = 0; slot != 9; ++slot) {
          Value index = arith::ConstantIndexOp::create(inside, loc, slot);
          values.push_back(memref::LoadOp::create(inside, loc, barostatMemory, index));
        }
        auto vec = VectorType::get({3}, inside.getF64Type());
        SmallVector<Value> vectors;
        for (unsigned axis = 0; axis != 3; ++axis)
          vectors.push_back(vector::FromElementsOp::create(inside, loc, vec, ValueRange(values).slice(3 * axis, 3)));
        groups = vectors[1];
        kinetic = vectors[2];
      } else {
        auto pressure = call(inside, config.pressureStep, point);
        buildVirialDiagonal(inside, pressure.getResult(3));
        nextPoint = bind(point, pressure.getResults(), false);
        groups = pressure.getResult(4);
        pressureFull = pressure.getResult(config.leapfrog ? 5 : 1);
      }
      Value completed = makeStep();
      if (!config.everyStep) {
        auto parameters = buildEntryPotentialFields(point);
        auto rigid = buildEntryGroups(parameters, config.constraints);
        kinetic = buildGroupKinetic(inside,
            {nextPoint.positions, pressureFull, nextPoint.forces, cell, point.masses,
             arguments.interval, pressureMass, rigid}).half;
      }
      std::array<Value, 3> beta{barostat.compressibility, {}, {}};
      Value half, tension;
      if (config.strainCoupling == StrainCoupling::Anisotropic) beta = barostat.axisCompressibility;
      else if (config.strainCoupling == StrainCoupling::SemiIsotropic) {
        beta[2] = barostat.zCompressibility;
        half = coupling.half;
        tension = barostat.tension;
      }
      auto strain = buildBarostatStrain(inside,
          {kinetic, groups, boxMemory, edgeIndices,
           arguments.startValues[*config.barostatConstant], barostat.third, barostat.bar,
           barostat.unitVector, seed, completed, barostat.pressure, barostat.thermal,
           barostat.rate, beta, half, tension}, config.strainCoupling);
      Value work = arith::AddFOp::create(inside, loc, groups, strain.constantDiagonal);
      Value xy = arith::MulFOp::create(inside, loc, strain.edges[0], strain.edges[1]);
      Value volume = arith::MulFOp::create(inside, loc, xy, strain.edges[2]);
      Value nextCell = buildCell(inside, strain.edges);
      Value nextScale;
      if (reference) nextScale = buildReferenceScale(inside, strain.edges, reference);
      Value outsideCell = cell, outsideScale = scale;
      auto restore = llvm::make_scope_exit([&] { cell = outsideCell; scale = outsideScale; });
      cell = nextCell;
      scale = nextScale;
      auto next = call(inside, energy ? config.scaledEnergyStep : couplingProgram,
          nextPoint, strain.scale);
      bool virial = energy || config.countsAfterScaling;
      unsigned halfSlot = 3 + unsigned(energy) + 2 * unsigned(virial) + unsigned(config.leapfrog);
      return {next, completed, nextCell, work, volume, next.getResult(halfSlot),
          virial ? next.getResult(4 + unsigned(energy)) : Value{}, strain};
    };
    auto couple = [&](OpBuilder &inside, const EntryFieldsIR &current, Value completed, Value optimal, Value energy = {}, Value diagonal = {}, Value groups = {}, const TrotterIR *trotter = nullptr) {
      CenterMotionInputs motion;
      if (config.removesMotion)
        motion = {current.velocities, current.masses, coupling.totalMass, coupling.half};
      auto center = buildCenterMotion(inside, motion);
      Value bath = center.energy, factor, squared, axes, kinetic;
      if (config.barostat) {
        axes = buildKineticVector(inside, current.velocities, current.masses);
        kinetic = buildSum3(inside, axes);
        if (config.removesMotion)
          axes = buildAxesWithoutCenter(inside, axes, center.momentum, center.velocity, barostat.halfVector);
      }
      if (config.rescales) {
        if (!config.barostat) kinetic = buildKineticEnergy(inside, current.velocities, current.masses);
        auto thermal = buildThermostat(inside,
            {kinetic, config.removesMotion ? center.energy : Value{}, optimal, completed,
             seed, target, freedom, decay}, config.noseHoover);
        bath = thermal.heat;
        factor = thermal.factor;
        squared = thermal.factorSquared;
      }
      auto result = current;
      Value inverse;
      if (config.barostat && trotter) {
        GroupKineticResult centers;
        if (config.everyStep) {
          auto parameters = buildEntryPotentialFields(current);
          auto rigid = buildEntryGroups(parameters, config.constraints);
          centers = buildGroupKinetic(inside,
              {current.positions, current.velocities, current.forces, cell, current.masses,
               arguments.interval, pressureMass, rigid});
        }
        if (config.rescales) buildScaleKinetic(inside, squared, axes);
        if (config.everyStep) {
          Value stored = centers.half;
          if (config.rescales) stored = buildScaledGroupKinetic(inside, squared, centers.kinetic, centers.forceTerm);
          buildStoreBarostatState(inside, barostatMemory, diagonal, trotter->groupsAfter, stored);
        }
        Value work;
        if (config.countsAfterScaling)
          work = buildMeanVirialWork(inside,
              {arguments.startValues[*config.barostatConstant], trotter->volume, barostat.third,
               trotter->groupsAfter, trotter->workBefore, barostat.halfVector, trotter->strain.logScale});
        else work = buildVirialWork(inside, trotter->workBefore, trotter->strain.logScale);
        bath = buildTrotterBath(inside, trotter->strain.inverse, barostat.unitVector,
            trotter->kineticHalf, work, bath);
      } else if (config.barostat) {
        auto parameters = buildEntryPotentialFields(current);
        auto rigid = buildEntryGroups(parameters, config.constraints);
        auto centers = buildGroupKinetic(inside,
            {current.positions, current.velocities, current.forces, cell, current.masses,
             arguments.interval, pressureMass, rigid});
        std::array<Value, 3> beta{barostat.compressibility, {}, {}};
        Value half, tension;
        if (config.strainCoupling == StrainCoupling::Anisotropic) beta = barostat.axisCompressibility;
        else if (config.strainCoupling == StrainCoupling::SemiIsotropic) {
          beta[2] = barostat.zCompressibility;
          half = coupling.half;
          tension = barostat.tension;
        }
        auto strain = buildBarostatStrain(inside,
            {centers.half, groups, boxMemory, edgeIndices,
             arguments.startValues[*config.barostatConstant], barostat.third, barostat.bar,
             barostat.unitVector, seed, completed, barostat.pressure, barostat.thermal,
             barostat.rate, beta, half, tension}, config.strainCoupling);
        Value after = axes;
        if (config.rescales) after = buildScaleKinetic(inside, squared, axes);
        Value oneMinus = buildOneMinusScale(inside, barostat.unitVector, strain.scale);
        Value work;
        if (!config.exactBarostat)
          work = buildFirstOrderVirialWork(inside, groups, strain.constantTrace, barostat.third, oneMinus);
        Value change = buildKineticScaleChange(inside, strain.inverse, barostat.unitVector, after);
        if (!config.exactBarostat) bath = buildBathAfterWork(inside, work, change, bath);
        result.positions = buildGroupScaling(inside,
            {current.positions, strain.scale, oneMinus, cell, current.masses, rigid});
        if (config.exactBarostat) {
          Value newCell = buildCell(inside, strain.edges);
          Value newScale;
          if (reference) newScale = buildReferenceScale(inside, strain.edges, reference);
          // Preserve the base's site cell for EXACT trial placement/spreading (#722).
          Value trialSiteCell = entryCell;
          result.positions = buildEntrySites(inside, {result.positions, trialSiteCell, parameters, {}}, config.sites);
          Value time;
          if (config.timed) {
            Value currentStep = memref::LoadOp::create(inside, loc, stepMemory, zero);
            time = buildSimulationTime(inside, currentStep, arguments.interval);
          }
          auto evaluated = buildEntryEvaluation(inside,
              {result.positions, newCell, parameters, time}, {true, true, false}, config.sites, complete, newScale, trialSiteCell);
          result.forces = evaluated.physical.forces;
          bath = buildExactBath(inside,
              {ValueRange(strain.edges).take_front(3), barostat.unit, strain.volume,
               arguments.startValues[*config.barostatEnergyConstant], evaluated.physical.energy,
               energy, change, bath});
        }
        inverse = strain.inverse;
      }
      result.velocities = buildCoupledVelocities(inside, current.velocities,
          center.velocity, factor, inverse, bath);
      return result;
    };
    auto closeCoupling = [&](OpBuilder &inside, const EntryFieldsIR &point,
        llvm::function_ref<Value()> makeStep) -> EntryFieldsIR {
          if (config.trotter) {
            auto scaled = trotterSteps(inside, point, false, makeStep);
            auto next = scaled.step;
            Value diagonal;
            if (config.countsAfterScaling) diagonal = buildVirialDiagonal(inside, next.getResult(3)).vector;
            unsigned fullSlot = 3 + 2 * unsigned(config.countsAfterScaling);
            Value full = next.getResult(config.leapfrog ? fullSlot : 1), optimal;
            if (config.rescales) {
              Value excess;
              if (measuredCoupling) {
                Value current = buildKineticVector(inside, full, point.masses);
                excess = arith::SubFOp::create(inside, loc,
                    next.getResult(fullSlot + unsigned(config.leapfrog) + 1), current);
              } else excess = buildKineticExcess(inside, next.getResult(2), point.masses, arguments.interval);
              Value kinetic = buildKineticEnergy(inside, full, point.masses);
              optimal = optimalEnergy(inside, kinetic, buildSum3(inside, excess));
            }
            auto actual = bind(point, next.getResults(), false);
            if (config.leapfrog) actual.velocities = full;
            auto result = couple(inside, actual, scaled.completed, optimal, {}, diagonal, {}, &scaled);
            if (config.leapfrog)
              result.velocities = buildKick(inside, result.velocities, result.forces, result.masses, halfBack);
            return result;
          }
          auto next = call(inside, (measuredCoupling || config.barostat) ? couplingProgram : step, point);
          Value diagonal;
          if (config.barostat) diagonal = buildVirialDiagonal(inside, next.getResult(4)).vector;
          Value completed = makeStep();
          Value full = next.getResult(config.barostat && config.leapfrog ? 6 : 1), optimal;
          if (config.leapfrog && config.rescales && !config.barostat) {
            if (measuredCoupling) full = next.getResult(3);
            else {
              Value half = arith::ConstantOp::create(inside, loc, inside.getF64FloatAttr(0.5));
              Value interval = arith::MulFOp::create(inside, loc, half, arguments.interval);
              full = buildKick(inside, next.getResult(1), next.getResult(2), point.masses, interval);
            }
          }
          if (config.rescales && !config.brownian) {
            Value excess;
            if (measuredCoupling) {
              Value current = buildKineticVector(inside, full, point.masses);
              excess = arith::SubFOp::create(inside, loc, next.getResult((config.barostat ? 6 : 3) + unsigned(config.leapfrog)), current);
            } else excess = buildKineticExcess(inside, next.getResult(2), point.masses, arguments.interval);
            Value kinetic = buildKineticEnergy(inside, full, point.masses);
            Value scalarExcess = buildSum3(inside, excess);
            optimal = optimalEnergy(inside, kinetic, scalarExcess);
          }
          auto actual = bind(point, next.getResults(), false);
          if (config.barostat && config.leapfrog) actual.velocities = full;
          auto result = couple(inside, actual, completed, optimal,
              config.barostat ? next.getResult(3) : Value{}, diagonal,
              config.barostat ? next.getResult(5) : Value{});
          if (config.barostat && config.leapfrog)
            result.velocities = buildKick(inside, result.velocities, result.forces, result.masses, halfBack);
          return result;
    };
    if (config.segments) {
      auto frameIfDue = [&](OpBuilder &body, Value period, Value completed,
                            const EntryFieldsIR &point) -> llvm::Error {
        Value none = arith::ConstantIntOp::create(body, loc, 0, 64);
        Value single = arith::ConstantIntOp::create(body, loc, 1, 64);
        Value hasPeriod = arith::CmpIOp::create(body, loc, arith::CmpIPredicate::sgt, period, none);
        Value safe = arith::SelectOp::create(body, loc, hasPeriod, period, single);
        Value remainder = arith::RemSIOp::create(body, loc, completed, safe);
        Value exact = arith::CmpIOp::create(body, loc, arith::CmpIPredicate::eq, remainder, none);
        Value due = arith::AndIOp::create(body, loc, hasPeriod, exact);
        Value wide = arith::ExtUIOp::create(body, loc, body.getI64Type(), due);
        Value trips = arith::IndexCastOp::create(body, loc, body.getIndexType(), wide);
        auto write = buildEntryLoop(body, "part frame", zero, trips, one, {},
            [&](OpBuilder &at, Value, ValueRange) -> llvm::Expected<SmallVector<Value>> {
          mdir::mdrt::HostCallOp::create(at, loc, TypeRange{},
              FlatSymbolRefAttr::get(at.getContext(), "mdrtWriteFrame"),
              ValueRange{completed, point.positions, point.identities}, ValueRange{}, UnitAttr{});
          return SmallVector<Value>{};
        });
        if (!write) return write.takeError();
        return llvm::Error::success();
      };
      Value memory = memref::AllocaOp::create(at, loc, MemRefType::get({config.partCounts}, at.getI64Type()));
      Value end = arith::ConstantIndexOp::create(at, loc, INT64_MAX);
      auto parts = buildEntryLoop(at, "step parts", zero, end, one, pack(physical, false),
          [&](OpBuilder &part, Value, ValueRange current) -> llvm::Expected<SmallVector<Value>> {
        auto point = bind(physical, current, false);
        auto vec = VectorType::get({3}, part.getF64Type());
        auto finite = mdir::md::SumParticlesOp::create(part, loc, vec,
            ValueRange{point.positions, point.velocities, point.forces});
        {
          OpBuilder::InsertionGuard guard(part);
          auto args = part.createBlock(&finite.getKernel(), {}, {vec, vec, vec}, {loc, loc, loc})->getArguments();
          Value yes = arith::ConstantOp::create(part, loc, DenseElementsAttr::get(vec, 1.0));
          Value no = arith::ConstantOp::create(part, loc, DenseElementsAttr::get(vec, 0.0));
          SmallVector<Value> bad;
          for (Value field : args) {
            Value difference = arith::SubFOp::create(part, loc, field, field);
            Value good = arith::CmpFOp::create(part, loc, arith::CmpFPredicate::ORD, difference, difference);
            bad.push_back(arith::SelectOp::create(part, loc, good, no, yes));
          }
          Value first = arith::AddFOp::create(part, loc, bad[0], bad[1]);
          Value all = arith::AddFOp::create(part, loc, first, bad[2]);
          mdir::md::YieldOp::create(part, loc, ValueRange{all});
        }
        Value invalid = buildSum3(part, finite.getResult());
        buildPartBoundary(part, point.positions, point.velocities, point.forces, point.identities, invalid, memory);
        SmallVector<Value> counts;
        for (int64_t index = 0; index != config.partCounts; ++index) {
          Value slot = arith::ConstantIndexOp::create(part, loc, index);
          counts.push_back(memref::LoadOp::create(part, loc, memory, slot));
        }
        auto cast = [&](unsigned slot) -> Value {
          return arith::IndexCastOp::create(part, loc, part.getIndexType(), counts[slot]);
        };
        Value n0 = cast(1), inner, energyCount, energyCouples, energyInner, tailCount, plainCount;
        Value per0, perEnergy, perCouple;
        if (config.couplingIntervals) {
          inner = cast(2);
          energyCount = cast(5);
          energyCouples = cast(7);
          energyInner = cast(6);
          tailCount = cast(3);
          plainCount = cast(4);
          arith::ConstantIndexOp::create(part, loc, 1);
          Value last = arith::ConstantIndexOp::create(part, loc, config.trotter ? 2 : 1);
          per0 = arith::AddIOp::create(part, loc, inner, last);
          arith::ConstantIndexOp::create(part, loc, 1);
          perCouple = arith::AddIOp::create(part, loc, energyInner, last);
          Value periods = arith::AddIOp::create(part, loc, energyCouples, one);
          perEnergy = arith::MulIOp::create(part, loc, periods, perCouple);
        } else {
          energyCount = cast(5);
          energyInner = cast(6);
          tailCount = cast(3);
          plainCount = cast(4);
          per0 = arith::ConstantIndexOp::create(part, loc, 1);
          arith::ConstantIndexOp::create(part, loc, 1);
          perEnergy = arith::AddIOp::create(part, loc, energyInner, one);
        }
        auto plain = [&](OpBuilder &body, Value upper, const EntryFieldsIR &start, bool reload = true)
            -> llvm::Expected<scf::ForOp> {
          return buildEntryLoop(body, "part plain steps", zero, upper, one, pack(start, false),
              [&](OpBuilder &at, Value, ValueRange state) -> llvm::Expected<SmallVector<Value>> {
            Value outsideCell = cell, outsideScale = scale;
            auto restore = llvm::make_scope_exit([&] { cell = outsideCell; scale = outsideScale; });
            if (config.barostat && reload) readCell(at);
            return SmallVector<Value>(call(at, step, bind(start, state, false)).getResults());
          });
        };
        auto first = [&]() -> llvm::Expected<scf::ForOp> {
          if (!config.couplingIntervals) return plain(part, n0, point);
          return buildEntryLoop(part, "part coupling periods", zero, n0, one, pack(point, false),
              [&](OpBuilder &body, Value index, ValueRange state) -> llvm::Expected<SmallVector<Value>> {
            Value outsideCell = cell, outsideScale = scale;
            auto restore = llvm::make_scope_exit([&] { cell = outsideCell; scale = outsideScale; });
            if (config.barostat) readCell(body);
            auto actual = bind(point, state, false);
            auto before = plain(body, inner, actual);
            if (!before) return before.takeError();
            actual = bind(actual, before->getResults(), false);
            if (config.barostat) readCell(body);
            auto coupled = closeCoupling(body, actual, [&] {
              return buildStepNumber(body, {{index, {}}}, index, {}, per0, one, counts[0]);
            });
            return pack(coupled, false);
          });
        }();
        if (!first) return first.takeError();
        Value outsideCell = cell, outsideScale = scale;
        if (config.barostat) readCell(part);
        auto tail = plain(part, tailCount, bind(point, first->getResults(), false), false);
        if (!tail) return tail.takeError();
        Value plainStep = buildSegmentStep(part, n0, per0, tailCount, one, counts[0]);
        auto closing = buildEntryLoop(part, "part plain energy", zero, plainCount, one, tail->getResults(),
            [&](OpBuilder &body, Value, ValueRange state) -> llvm::Expected<SmallVector<Value>> {
          auto actual = bind(point, state, false);
          auto next = call(body, energyStep, actual);
          report(body, actual, next, [&] { return plainStep; }, true);
          auto completed = bind(actual, next.getResults(), false);
          if (auto error = frameIfDue(body, counts[8], plainStep, completed)) return std::move(error);
          return pack(completed, false);
        });
        if (!closing) return closing.takeError();
        cell = outsideCell;
        scale = outsideScale;
        Value start = buildSegmentStep(part, n0, per0, tailCount, plainCount, counts[0]);
        auto energies = buildEntryLoop(part, "part energy intervals", zero, energyCount, one, closing->getResults(),
            [&](OpBuilder &body, Value index, ValueRange state) -> llvm::Expected<SmallVector<Value>> {
          Value outsideCell = cell, outsideScale = scale;
          auto restore = llvm::make_scope_exit([&] { cell = outsideCell; scale = outsideScale; });
          if (config.barostat) readCell(body);
          auto actual = bind(point, state, false);
          if (config.couplingIntervals) {
            auto periods = buildEntryLoop(body, "part closing coupling periods", zero, energyCouples, one,
                pack(actual, false),
                [&](OpBuilder &period, Value within, ValueRange values) -> llvm::Expected<SmallVector<Value>> {
              Value outsideCell = cell, outsideScale = scale;
              auto restore = llvm::make_scope_exit([&] { cell = outsideCell; scale = outsideScale; });
              if (config.barostat) readCell(period);
              auto current = bind(actual, values, false);
              auto before = plain(period, energyInner, current);
              if (!before) return before.takeError();
              current = bind(current, before->getResults(), false);
              if (config.barostat) readCell(period);
              auto coupled = closeCoupling(period, current, [&] {
                return buildStepNumber(period, {{index, {}}}, within, perEnergy, perCouple, one, start);
              });
              return pack(coupled, false);
            });
            if (!periods) return periods.takeError();
            actual = bind(actual, periods->getResults(), false);
          }
          if (config.barostat) readCell(body);
          auto before = plain(body, energyInner, actual, false);
          if (!before) return before.takeError();
          actual = bind(actual, before->getResults(), false);
          auto makeStep = [&] {
            return buildStepNumber(body, {{index, {}}}, index, {}, perEnergy, one, start);
          };
          std::optional<TrotterIR> scaled;
          mdir::dyn::StepOp next;
          if (config.trotter) {
            scaled = trotterSteps(body, actual, true, makeStep);
            next = scaled->step;
          } else next = call(body, energyStep, actual);
          auto row = report(body, actual, next,
              [&] { return scaled ? scaled->completed : makeStep(); },
              false, bool(scaled), scaled ? scaled->cell : Value{});
          auto completed = bind(actual, next.getResults(), false);
          if (config.couplingIntervals) {
            Value optimal;
            if (config.rescales && !config.brownian)
              optimal = optimalEnergy(body, row.kinetic, row.correction);
            if (config.barostat && config.leapfrog) completed.velocities = next.getResult(6);
            completed = couple(body, completed, row.step, optimal,
                config.barostat ? next.getResult(3) : Value{}, row.diagonal,
                config.barostat ? next.getResult(5) : Value{}, scaled ? &*scaled : nullptr);
            if (config.barostat && config.leapfrog)
              completed.velocities = buildKick(body, completed.velocities, completed.forces, completed.masses, halfBack);
          }
          if (auto error = frameIfDue(body, counts[8], row.step, completed)) return std::move(error);
          return pack(completed, false);
        });
        if (!energies) return energies.takeError();
        return SmallVector<Value>(energies->getResults());
      });
      if (!parts) return parts.takeError();
      return llvm::Error::success();
    }
    SmallVector<LoopIterationIR> iterations;
    auto number = [&](OpBuilder &body, unsigned level, Value index) {
      bool split = kinds[level] == Kind::Couple && level && kinds[level - 1] == Kind::Energy;
      ArrayRef<LoopIterationIR> loops(iterations);
      return buildStepNumber(body, split ? loops.drop_back() : loops, index,
          split ? periods[level - 1] : Value{}, periods[level], one, arguments.start);
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
          kinds[atLevel] == Kind::Energy ? "energy intervals" : kinds[atLevel] == Kind::Couple ? "COM periods" : "plain steps";
      return buildEntryLoop(body, scope, zero, extents[atLevel], one, pack(initial, reorders),
          [&](OpBuilder &inside, Value index, ValueRange current) -> llvm::Expected<SmallVector<Value>> {
        iterations.push_back({index, extents[atLevel]});
        auto leave = llvm::make_scope_exit([&] { iterations.pop_back(); });
        Value outsideCell = cell, outsideScale = scale;
        auto restoreCell = llvm::make_scope_exit([&] { cell = outsideCell; scale = outsideScale; });
        if (config.barostat) readCell(inside);
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
        if (config.barostat) readCell(inside);
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
        if (kinds[atLevel] == Kind::Couple)
          return pack(closeCoupling(inside, point, [&] { return number(inside, atLevel, index); }), false);
        if (config.couplingIntervals) {
          auto last = buildEntryLoop(inside, "last COM period plain steps", zero, extents.back(), one,
              pack(point, false),
              [&](OpBuilder &at, Value, ValueRange carried) -> llvm::Expected<SmallVector<Value>> {
            auto actual = bind(point, carried, false);
            return SmallVector<Value>(call(at, step, actual).getResults());
          });
          if (!last) return last.takeError();
          point = bind(point, last->getResults(), false);
        }
        std::optional<TrotterIR> scaled;
        mdir::dyn::StepOp next;
        if (config.trotter) {
          scaled = trotterSteps(inside, point, true, [&] { return number(inside, atLevel, index); });
          next = scaled->step;
        } else next = call(inside, energyStep, point);
        auto row = report(inside, point, next,
            [&] { return scaled ? scaled->completed : number(inside, atLevel, index); },
            false, bool(scaled), scaled ? scaled->cell : Value{});
        Value kinetic = row.kinetic, correction = row.correction, completed = row.step;
        auto final = bind(point, next.getResults(), false);
        if (config.couplingIntervals) {
          Value optimal;
          if (config.rescales && !config.brownian) optimal = optimalEnergy(inside, kinetic, correction);
          if (config.barostat && config.leapfrog) final.velocities = next.getResult(6);
          final = couple(inside, final, completed, optimal, next.getResult(3), row.diagonal,
              config.barostat ? next.getResult(5) : Value{}, scaled ? &*scaled : nullptr);
          if (config.barostat && config.leapfrog)
            final.velocities = buildKick(inside, final.velocities, final.forces, final.masses, halfBack);
        }
        if (config.framesAtEnergies) writeFrame(inside, completed, final);
        return pack(final, false);
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
