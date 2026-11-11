// The schedule of @mdir_run: the levels of its loops, the branches of
// the start, the parts of a program of segments, the frames, and the
// minimization (docs/driver.md, Section 2.12).

#include "BuilderImpl.h"
#include "ScheduleIR.h"
#include "CouplingIR.h"
#include "EntryRuntimeIR.h"
#include "SimpleFrameEntryIR.h"
#include "SimpleStepEntryIR.h"
#include "MinimizationEntryIR.h"
#include "mdir/Dialect/Dyn/DynOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "PotentialIR.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MDRT/MDRTOps.h"
#include "BarostatIR.h"
#include "DynamicsIR.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"

using namespace mdir::driver;
using namespace mdir::driver::builder;
namespace prolate = mdir::prolate;
namespace influence = mdir::influence;

llvm::Error Builder::emitEntry() {
  collectStartValues();

  RuntimeDeclarationConfig declarations{
      program.state == Element::F64, program.force == Element::F64,
      program.tunableGradient, bool(control.frames), control.segments,
      static_cast<int64_t>(getPartCountsSize()), bool(control.minimize),
      stopsMinimization(), reportsSolvent(),
      hasConstraints() && !control.minimize, rescalesVelocities(),
      control.isNoseHoover(), control.getCouplingPeriod() > 0,
      bool(control.barostat), scalesEveryStep(), control.checkpointPeriod > 0,
      program.writesForces};
  auto coordinateOutput = [&](mlir::OpBuilder &at, const PotentialStateIR &state, mlir::Value step) {
    auto tuple = [&](StringRef name) {
      unsigned offset = program.fields.size() + program.tables.size();
      for (const auto &set : program.tupleSets) {
        if (set.name == name) return std::make_pair(&set, offset);
        offset += 1 + set.fields.size();
      }
      llvm_unreachable("output relation must exist");
    };
    auto relation = [&](StringRef name) { return state.fields[tuple(name).second]; };
    auto field = [&](StringRef set, StringRef name) {
      auto [metadata, offset] = tuple(set);
      auto found = llvm::find_if(metadata->fields,
          [&](const auto &field) { return field.name == name; });
      assert(found != metadata->fields.end() && "output parameter must exist");
      return state.fields[offset + 1 + std::distance(metadata->fields.begin(), found)];
    };
    TermNativeOutput point{state, step, relation, field};
    for (const TermBuilder *term : getTermBuilders())
      if (const auto *role = term->getRoles().coordinateOutput)
        role->buildOutput(*this, at, point);
  };
  auto complete = [&](mlir::OpBuilder &at, const PotentialStateIR &state, EnergyResults evaluation,
      mlir::Value scale) {
    bool minimizationVirial = control.minimize && !evaluation.virial;
    if (minimizationVirial && hasForceCorrections()) {
      auto vec = mlir::VectorType::get({9}, at.getF64Type());
      evaluation.virial = mlir::arith::ConstantOp::create(at, at.getUnknownLoc(),
          mlir::DenseElementsAttr::get(vec, 0.0));
    }
    llvm::SmallVector<mlir::Value> stored;
    auto field = [&](StringRef name) -> mlir::Value {
      auto found = llvm::find_if(program.fields,
          [&](const auto &field) { return field.name == name; });
      assert(found != program.fields.end() && "evaluation field must exist");
      return state.fields[std::distance(program.fields.begin(), found)];
    };
    for (const TermBuilder *term : getTermBuilders())
      if (const auto *role = term->getRoles().postEvaluation;
          role && role->hasForceCorrection(*this)) {
        auto result = role->build(*this, at,
            {state.positions, state.cell, evaluation.forces, evaluation.energy,
             evaluation.virial, scale, field});
        evaluation = {result.energy, result.forces, result.virial};
        llvm::append_range(stored, result.storedTerms);
      }
    if (minimizationVirial) evaluation.virial = {};
    for (const TermBuilder *term : getTermBuilders())
      if (const auto *role = term->getRoles().hostEvaluation;
          role && role->applies(*this)) {
        auto result = role->build(*this, at,
            {state.positions, state.cell, evaluation.forces, evaluation.energy,
             evaluation.virial, scale, field});
        evaluation = {result.energy, result.forces, result.virial};
      }
    return EntryCompletionIR{evaluation, std::move(stored)};
  };
  auto configureSites = [&](std::vector<DescentSiteIR> &sites) {
    for (const Program::TupleSet *selected : getSiteSets()) {
      unsigned slot = program.fields.size() + program.tables.size();
      for (const auto &set : program.tupleSets) {
        if (&set == selected) break;
        slot += 1 + set.fields.size();
      }
      auto fieldSlot = [&](StringRef name) {
        auto found = llvm::find_if(selected->fields,
            [&](const auto &field) { return field.name == name; });
        assert(found != selected->fields.end() && "site parameter must exist");
        return slot + 1 + unsigned(std::distance(selected->fields.begin(), found));
      };
      sites.push_back({slot, fieldSlot("a"), fieldSlot("b"),
          selected->name == "sites_amber", spreadsSitesDerived(), "spread_" + selected->name});
    }
  };
  auto configureConstraints = [&](std::vector<EntryConstraintIR> &constraints) {
    auto groups = getShakeSets();
    for (const auto &set : program.tupleSets)
      if (set.name == "settles") groups.insert(groups.begin(), &set);
    for (const Program::TupleSet *selected : groups) {
      unsigned slot = program.fields.size() + program.tables.size();
      for (const auto &set : program.tupleSets) {
        if (&set == selected) break;
        slot += 1 + set.fields.size();
      }
      auto fieldSlot = [&](StringRef name) {
        auto found = llvm::find_if(selected->fields,
            [&](const auto &field) { return field.name == name; });
        assert(found != selected->fields.end() && "constraint parameter must exist");
        return slot + 1 + unsigned(std::distance(selected->fields.begin(), found));
      };
      EntryConstraintIR group{slot, selected->arity, {}, selected->name == "settles"};
      if (group.water) {
        group.parameters = {fieldSlot("doh"), fieldSlot("dhh")};
      } else {
        for (unsigned member = 1; member != selected->arity; ++member)
          group.parameters.push_back(fieldSlot("d" + std::to_string(member)));
      }
      constraints.push_back(std::move(group));
    }
  };
  auto configureOutputs = [&](EntryOutputIRConfig &config, bool initial) {
    if (!control.freeEnergyFile.empty()) {
      config.freeEnergy.emplace(FreeEnergyOutputConfig{
          {}, static_cast<size_t>(control.freeEnergy.state), getNumPotentialArguments()});
      for (const auto &component : control.freeEnergy.lambdas)
        config.freeEnergy->lambdas.push_back(component.second);
    }
    config.observableCount = control.observables.size();
    config.observableBaseArgument = getNumPotentialArguments();
    const auto observedTerms = getObservedTerms();
    for (auto [index, term] : llvm::enumerate(observedTerms)) {
      ObservableOutputTermIR output;
      output.potential = "observe" + std::to_string(index);
      for (const auto &constant : term.constants) {
        bool tunable = control.isObservedTunable(term.name, constant);
        std::optional<unsigned> slot;
        if (tunable) {
          slot = config.observableTunables.size();
          std::string name = "obv_" + term.name + "_" + constant;
          auto found = llvm::find_if(program.startValues,
              [&](const auto &value) { return value.name == name; });
          assert(found != program.startValues.end() && "observed tunable must be an entry value");
          config.observableTunables.push_back(std::distance(program.startValues.begin(), found));
        }
        output.constants.push_back({tunable ? 0.0 : getObservedValue(term, constant), slot});
      }
      for (auto column : term.columns)
        output.columns.push_back({column.first, column.second});
      config.observables.push_back(std::move(output));
    }
    if (program.tunableGradient && initial) {
      auto &gradient = config.gradient.emplace();
      if (control.frames && control.frameProduct)
        // Requests 0/1 select forces/virial; result 0 holds positions.
        // The product entry takes g in request 2 and returns G's cotangent
        // as buffer 1, without adding a topology field.
        gradient.product = EntryGradientProductIR{2, 1, control.frameProductPositions,
                                                  control.frameProductDirection};
      auto startSlot = [&](StringRef name) -> unsigned {
        auto found = llvm::find_if(program.startValues,
            [&](const auto &value) { return value.name == name; });
        assert(found != program.startValues.end() && "gradient input must be an entry value");
        return std::distance(program.startValues.begin(), found);
      };
      gradient.request = startSlot("tunable_gradient");
      size_t constants = control.tunableConstants.size(), seeds = getNumGradientSeeds();
      for (size_t index = 0; index != constants; ++index)
        gradient.constants.push_back(startSlot("tg_c" + std::to_string(index)));
      gradient.seeds = seeds;
      if (seeds) {
        auto found = llvm::find_if(program.fields,
            [](const auto &field) { return field.name == "tg_zero"; });
        assert(found != program.fields.end() && "gradient seed field must exist");
        gradient.seedField = std::distance(program.fields.begin(), found);
      }
      unsigned base = getNumPotentialArguments(), chargeArgument = 2;
      for (const auto &field : program.fields) {
        if (field.name == "q") break;
        ++chargeArgument;
      }
      program.gradientArguments.clear();
      for (auto [index, slot] : llvm::enumerate(program.gradientSlots)) {
        unsigned argument = base + gradientColumns[index];
        gradient.output.scalarArguments.push_back(argument);
        program.gradientArguments.push_back({argument, slot.tunable});
      }
      for (auto [index, field] : llvm::enumerate(program.gradientFields)) {
        unsigned argument = index < seeds ? base + constants + index : chargeArgument;
        gradient.output.fieldArguments.push_back(argument);
        program.gradientArguments.push_back({argument, field.tunable});
      }
    }
  };
  if (control.minimize) {
    mlir::OpBuilder ir(&termIR.getContext());
    auto *body = termIR.getModule().getBody();
    ir.setInsertionPointToEnd(body);
    emitRuntimeDeclarations(ir, declarations, false);
    for (const TermBuilder *term : getTermBuilders())
      if (const auto *role = term->getRoles().hostEvaluation) role->buildDeclarations(*this, ir);
    emitRuntimeDeclarations(ir, declarations, true);
    MinimizationEntryIRConfig config;
    config.segments = control.segments;
    config.partCounts = getPartCountsSize();
    config.timed = control.usesTime;
    config.stepMemory = needsStepNumber();
    config.restart = isRestart();
    config.stops = stopsMinimization();
    config.topology = bool(system.topology);
    config.checkpointPeriod = control.checkpointPeriod;
    config.changesCell = changesCell();
    config.scalesEveryStep = scalesEveryStep();
    config.cellEdges = getCellSize();
    config.steps = control.numSteps;
    config.energyPeriod = control.energyPeriod;
    config.framePeriod = control.framePeriod;
    config.firstSize = control.minimizeStep * units::length;
    config.cap = Control::minimizeStepCap * units::length;
    if (scalesReference()) {
      std::array<unsigned, 3> slots;
      for (unsigned index = 0; index != 3; ++index) {
        auto found = llvm::find_if(program.startValues, [&](const auto &value) {
          return value.name == "rest_edge" + std::to_string(index);
        });
        assert(found != program.startValues.end() && "reference edge must exist");
        slots[index] = std::distance(program.startValues.begin(), found);
      }
      config.referenceEdges = slots;
    }
    configureSites(config.sites);
    std::vector<EntryConstraintIR> groups;
    configureConstraints(groups);
    ConstraintPositionConfig position{false, control.analyticBonds,
        getConstraintPositionTolerance(control.precision == Precision::Double),
        control.precision == Precision::Double ? 1.0e-12 : 1.0e-6};
    for (const auto &group : groups) {
      DescentConstraintGroupIR geometry{group.relation, group.arity,
          {group.parameters.begin(), group.parameters.end()}, {}, position, group.water,
          group.water && control.precision == Precision::Double};
      if (group.water) geometry.bonds = {{0, 1, 0}, {0, 2, 0}, {1, 2, 1}};
      else for (unsigned member = 1; member != group.arity; ++member)
        geometry.bonds.push_back({0, member, member - 1});
      config.constraints.push_back(std::move(geometry));
    }
    if (config.topology)
      for (const auto &definition : getTermDefinitions()) config.terms.push_back(definition.name);
    config.needsPostEvaluation = hasForceCorrections();
    config.needsHostedEvaluation = hasHostEvaluation();
    for (const TermBuilder *term : getTermBuilders())
      if (const auto *role = term->getRoles().postEvaluation)
        config.storedTerms += role->getStoredTermCount(*this);
    auto descend = termIR.getModule().lookupSymbol<mdir::dyn::ProgramOp>("descend");
    auto entry = buildMinimizationEntry(ir, program, descend, config, complete);
    if (!entry) return entry.takeError();

    return llvm::Error::success();
  }
  // Each selected entry is complete native IR, including every loop body.
  // Other physical families keep their existing entry until their scopes move.
  if (control.frames) {
    mlir::OpBuilder ir(&termIR.getContext());
    auto *moduleBody = termIR.getModule().getBody();
    ir.setInsertionPointToEnd(moduleBody);
    emitRuntimeDeclarations(ir, declarations, false);
    for (const TermBuilder *term : getTermBuilders())
      if (const auto *role = term->getRoles().hostEvaluation)
        role->buildDeclarations(*this, ir);
    emitRuntimeDeclarations(ir, declarations, true);
    SimpleFrameEntryIRConfig config;
    config.deviceBatch = control.frameMode == Control::FrameMode::DeviceBatch;
    config.batchOnDevice = control.target == Target::GPU;
    config.dependent = control.dependentTerms;
    if (config.deviceBatch && control.frameBatchTunables) {
      auto parameters = makeFrameBatchParameterPlan(program,
          *control.frameBatchTunables, control, system, coulombInternal);
      if (!parameters) return parameters.takeError();
      auto tails = makeFrameBatchTailPlan(*this, program, *control.frameBatchTunables,
          parameters->offsets);
      if (!tails) return tails.takeError();
      tails->resetCorrections = !parameters->lj.has_value();
      config.batchTails = std::move(*tails);
      config.batchParameters = std::move(*parameters);
    }
    config.branchesStart = branchesStart();
    config.givenForces = branchesStart() && program.takesForces;
    config.timed = control.usesTime;
    config.stepMemory = needsStepNumber();
    config.leapfrog = isLeapfrog();
    config.brownian = control.isBrownian();
    config.coupled = control.getCouplingPeriod() > 0;
    config.barostat = bool(control.barostat);
    config.changesCell = changesCell();
    config.scalesEveryStep = scalesEveryStep();
    config.removesMotion = control.comPeriod > 0;
    if (scalesReference()) {
      std::array<unsigned, 3> slots;
      for (unsigned index = 0; index != 3; ++index) {
        std::string name = "rest_edge" + std::to_string(index);
        auto found = llvm::find_if(program.startValues,
            [&](const auto &value) { return value.name == name; });
        assert(found != program.startValues.end() && "reference edge must be an entry value");
        slots[index] = std::distance(program.startValues.begin(), found);
      }
      config.referenceEdges = slots;
    }
    if (config.coupled && control.thermostat) {
      double freedom = system.getDegreesOfFreedom();
      double target = 0.5 * freedom * units::boltzmann * control.temperature;
      double decay = std::exp(-static_cast<double>(control.getCouplingPeriod()) *
          control.timestep / control.tauT);
      config.thermostat = FrameThermostatIR{static_cast<int64_t>(control.seed), freedom, target, decay};
      if (control.barostat) {
        double bar = 1.01325;
        config.barostatParameters = BarostatParameters{
            {control.compressibilities[0] / bar, control.compressibilities[1] / bar,
             control.compressibilities[2] / bar},
            control.pressure * bar, control.compressibility / bar,
            static_cast<double>(control.barostatPeriod) * control.timestep / control.tauP,
            units::boltzmann * control.temperature, control.compressibilityZ / bar,
            control.surfaceTension * control.surfaces * units::dynePerCmToBarNm};
      }
    }
    config.cellEdges = getCellSize();
    config.topology = bool(system.topology);
    for (const TermBuilder *term : getTermBuilders())
      if (const auto *role = term->getRoles().postEvaluation)
        config.storedTerms += role->getStoredTermCount(*this);
    configureSites(config.sites);
    configureConstraints(config.constraints);
    if (config.topology)
      for (const auto &definition : getTermDefinitions())
        config.terms.push_back(definition.name);
    configureOutputs(config.output, true);
    if (!control.segments) {
      for (const auto &level : levels) config.counts.push_back(level.count);
      int64_t steps = 1;
      for (unsigned index = levels.size(); index-- != 0;) {
        config.periods.push_back(steps);
        steps *= levels[index].count;
        if (index && levels[index - 1].name == "couple")
          steps += usesTrotter() && !scalesEveryStep() ? 2 : 1;
        else if (index && levels[index - 1].name == "energy")
          steps += levels[index].name == "couple" ? control.getCouplingPeriod() : 1;
      }
    }
    auto entry = buildSimpleFrameEntry(ir, program, config,
        complete, coordinateOutput);
    if (!entry) return entry.takeError();

    return llvm::Error::success();
  }
  {
    mlir::OpBuilder ir(&termIR.getContext());
    auto *body = termIR.getModule().getBody();
    ir.setInsertionPointToEnd(body);
    emitRuntimeDeclarations(ir, declarations, false);
    for (const TermBuilder *term : getTermBuilders())
      if (const auto *role = term->getRoles().hostEvaluation) role->buildDeclarations(*this, ir);
    emitRuntimeDeclarations(ir, declarations, true);
    auto step = termIR.getModule().lookupSymbol<mdir::dyn::ProgramOp>("step");
    SimpleStepEntryIRConfig config{levels.back().count};
    config.segments = control.segments;
    config.partCounts = getPartCountsSize();
    config.removesMotion = control.comPeriod > 0;
    config.rescales = rescalesVelocities();
    config.noseHoover = control.isNoseHoover();
    config.bathParameters = bool(control.thermostat);
    config.seed = static_cast<int64_t>(control.seed);
    config.freedom = system.getDegreesOfFreedom();
    config.targetKinetic = 0.5 * config.freedom * units::boltzmann * control.temperature;
    config.decay = std::exp(-static_cast<double>(control.getCouplingPeriod()) * control.timestep / control.tauT);
    config.barostat = bool(control.barostat);
    config.exactBarostat = control.barostatWork == BarostatWork::Exact;
    config.trotter = usesTrotter();
    config.countsAfterScaling = countsAfterScaling();
    config.everyStep = scalesEveryStep();
    if (config.everyStep && isRestart() && system.barostatState.size() == 9) {
      std::array<unsigned, 9> slots;
      for (unsigned slot = 0; slot != 9; ++slot) {
        auto found = llvm::find_if(program.startValues, [&](const auto &value) {
          return value.name == "bstate" + std::to_string(slot);
        });
        assert(found != program.startValues.end() && "barostat state role must exist");
        slots[slot] = std::distance(program.startValues.begin(), found);
      }
      config.barostatState = slots;
    }
    config.strainCoupling = control.anisotropic ? StrainCoupling::Anisotropic
        : control.semiIsotropic ? StrainCoupling::SemiIsotropic : StrainCoupling::Isotropic;
    if (config.barostat) {
      double bar = 1.01325;
      config.barostatParameters = BarostatParameters{
          {control.compressibilities[0] / bar, control.compressibilities[1] / bar,
           control.compressibilities[2] / bar},
          control.pressure * bar, control.compressibility / bar,
          static_cast<double>(control.barostatPeriod) * control.timestep / control.tauP,
          units::boltzmann * control.temperature, control.compressibilityZ / bar,
          control.surfaceTension * control.surfaces * units::dynePerCmToBarNm};
      auto constant = llvm::find_if(program.startValues,
          [](const auto &value) { return value.name == "baro_constant"; });
      assert(constant != program.startValues.end() && "barostat constant must exist");
      config.barostatConstant = std::distance(program.startValues.begin(), constant);
      if (config.exactBarostat) {
        auto energy = llvm::find_if(program.startValues,
            [](const auto &value) { return value.name == "baro_energy_constant"; });
        assert(energy != program.startValues.end() && "barostat energy constant must exist");
        config.barostatEnergyConstant = std::distance(program.startValues.begin(), energy);
      }
      config.couplingStep = termIR.getModule().lookupSymbol<mdir::dyn::ProgramOp>(
          config.trotter ? "step_trotter" : hasConstraints() ? "step_coupling" : "step_energy");
      if (config.trotter) {
        config.pressureStep = termIR.getModule().lookupSymbol<mdir::dyn::ProgramOp>("step_virial");
        config.scaledEnergyStep = termIR.getModule().lookupSymbol<mdir::dyn::ProgramOp>("step_trotter_energy");
      }
      if (scalesReference()) {
        std::array<unsigned, 3> slots;
        for (unsigned axis = 0; axis != 3; ++axis) {
          auto found = llvm::find_if(program.startValues, [&](const auto &value) {
            return value.name == "rest_edge" + std::to_string(axis);
          });
          assert(found != program.startValues.end() && "reference edge must exist");
          slots[axis] = std::distance(program.startValues.begin(), found);
        }
        config.referenceEdges = slots;
      }
    } else if (measuresCoupling())
      config.couplingStep = termIR.getModule().lookupSymbol<mdir::dyn::ProgramOp>("step_thermostat");
    config.periodic = control.periodic;
    config.restart = isRestart();
    config.recomputeForces = recomputes();
    config.timed = control.usesTime;
    config.stepMemory = needsStepNumber();
    config.leapfrog = isLeapfrog();
    config.brownian = control.isBrownian();
    config.triclinic = isTriclinic();
    config.topology = bool(system.topology);
    if (config.topology)
      for (const auto &definition : getTermDefinitions()) config.terms.push_back(definition.name);
    mdir::dyn::ProgramOp energy;
    config.framesAtEnergies = framesAtEnergies;
    for (const auto &level : levels) {
      if (level.name == "couple") config.couplingIntervals = level.count;
      if (level.name == "segment") config.checkpointIntervals = level.count;
      if (level.name == "frame") config.frameIntervals = level.count;
      if (level.name == "energy") {
        config.energyIntervals = level.count;
        energy = termIR.getModule().lookupSymbol<mdir::dyn::ProgramOp>("step_energy");
      }
    }
    if (config.segments) {
      config.energyIntervals.reset();
      energy = termIR.getModule().lookupSymbol<mdir::dyn::ProgramOp>("step_energy");
    }
    configureOutputs(config.output, !isRestart());
    configureSites(config.sites);
    configureConstraints(config.constraints);
    config.needsPostEvaluation = hasForceCorrections();
    config.needsHostedEvaluation = hasHostEvaluation();
    for (const TermBuilder *term : getTermBuilders())
      if (const auto *role = term->getRoles().postEvaluation)
        config.storedTerms += role->getStoredTermCount(*this);
    config.needsCoordinateOutput = llvm::any_of(getTermBuilders(), [&](const TermBuilder *term) {
      const auto *role = term->getRoles().coordinateOutput;
      return role && role->hasCoordinateOutput(*this);
    });
    auto entry = buildSimpleStepEntry(ir, program, step, energy, config, complete, coordinateOutput);
    if (!entry) return entry.takeError();

    return llvm::Error::success();
  }
}

void Builder::setSchedule() {
  // A period of zero means no output of that kind, and no loop for it.
  int64_t steps = control.numSteps;
  if (control.checkpointPeriod > 0) {
    levels.push_back({"segment", steps / control.checkpointPeriod});
    steps = control.checkpointPeriod;
  }
  // Frames more frequent than energies are written at the end of each
  // interval between energies, which then has their period.
  int64_t energyPeriod = control.getEnergyLoopPeriod();
  framesAtEnergies = energyPeriod != control.energyPeriod;
  if (control.framePeriod > 0 && !framesAtEnergies) {
    levels.push_back({"frame", steps / control.framePeriod});
    steps = control.framePeriod;
  }
  // The velocities are coupled at the end of the last step of a period of
  // coupling, which is taken after the loop over steps, as the last step
  // of an interval between energies is. The loop over the periods of an
  // interval between energies leaves the last period to the interval,
  // which ends it with its step of energy.
  // With the barostat of Trotter type the last two steps of a period are
  // taken after its loop: the step whose pressure gives the strain, and
  // the step that scales the cell within its drift (D92).
  int64_t coupling = control.getCouplingPeriod();
  int64_t taken = usesTrotter() && !scalesEveryStep() ? 2 : 1;
  if (energyPeriod > 0) {
    levels.push_back({"energy", steps / energyPeriod});
    steps = energyPeriod;
    if (coupling > 0) {
      levels.push_back({"couple", steps / coupling - 1});
      levels.push_back({"step", coupling - taken});
    } else {
      levels.push_back({"step", steps - 1});
    }
  } else if (coupling > 0) {
    levels.push_back({"couple", steps / coupling});
    levels.push_back({"step", coupling - taken});
  } else {
    levels.push_back({"step", steps});
  }
}
