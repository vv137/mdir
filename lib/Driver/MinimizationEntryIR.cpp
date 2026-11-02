// Controller state is carried by actual nested-loop arguments and results.
#include "MinimizationEntryIR.h"
#include "EntryConstraintIR.h"
#include "EntryFieldsIR.h"
#include "EntryLoopIR.h"
#include "EntryRuntimeIR.h"
#include "BarostatIR.h"
#include "ScheduleIR.h"
#include "SiteIR.h"
#include "OutputIR.h"
#include "mdir/Driver/Builder.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MDRT/MDRTOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
using namespace mlir;
using namespace mdir::driver;
using namespace mdir::driver::builder;

llvm::Expected<func::FuncOp> mdir::driver::builder::buildMinimizationEntry(
    OpBuilder &builder, const Program &program, mdir::dyn::ProgramOp descend,
    const MinimizationEntryIRConfig &config,
    llvm::function_ref<EntryCompletionIR(OpBuilder &, const PotentialStateIR &, EnergyResults, Value)> complete) {
  if (config.segments && config.partCounts < 2)
    return llvm::createStringError(llvm::inconvertibleErrorCode(), "native minimization part counts require start and step slots");
  if (!descend)
    return llvm::createStringError(llvm::inconvertibleErrorCode(), "native minimization has no descent program");
  if (config.needsPostEvaluation && !complete)
    return llvm::createStringError(llvm::inconvertibleErrorCode(), "native minimization requires the post-evaluation role");
  if (config.needsHostedEvaluation && !complete)
    return llvm::createStringError(llvm::inconvertibleErrorCode(), "native minimization requires the hosted-evaluation role");
  if (descend.getFunctionType().getResults().size() != 3 ||
      descend.getFunctionType().getResults()[2] != builder.getF64Type())
    return llvm::createStringError(llvm::inconvertibleErrorCode(), "native minimization descent results do not match its controller");
  if (config.segments && config.energyPeriod != 0)
    return llvm::createStringError(llvm::inconvertibleErrorCode(), "native minimization parts require a zero energy period");
  if (config.segments && config.framePeriod != 0)
    return llvm::createStringError(llvm::inconvertibleErrorCode(), "native minimization parts require a zero frame period");
  if (config.segments && config.checkpointPeriod != 0)
    return llvm::createStringError(llvm::inconvertibleErrorCode(), "native minimization parts require a zero checkpoint period");
  if (config.segments && config.stops)
    return llvm::createStringError(llvm::inconvertibleErrorCode(), "native minimization parts require host convergence checks");
  if (!config.segments && (config.steps <= 0 || config.energyPeriod <= 0))
    return llvm::createStringError(llvm::inconvertibleErrorCode(), "native minimization has invalid loop counts");
  SmallVector<unsigned> tilts;
  if (config.cellEdges == 6)
    for (StringRef name : {"tilt_bx", "tilt_cx", "tilt_cy"}) {
      auto found = llvm::find_if(program.startValues, [&](const auto &v) { return v.name == name; });
      if (found == program.startValues.end())
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
            "native minimization is missing cell start value '%s'", name.str().c_str());
      tilts.push_back(std::distance(program.startValues.begin(), found));
    }
  return buildEntryFunction(builder, program, {config.segments, config.segments},
      [&](OpBuilder &at, const EntryArgumentsIR &arguments) -> llvm::Error {
    auto loc = at.getUnknownLoc();
    auto constant = [&](double value) -> Value {
      return arith::ConstantOp::create(at, loc, at.getF64FloatAttr(value));
    };
    Value zero = arith::ConstantIndexOp::create(at, loc, 0);
    Value one = arith::ConstantIndexOp::create(at, loc, 1);
    Value firstCall, firstTrips;
    if (config.segments) {
      Value none = arith::ConstantIntOp::create(at, loc, 0, 64);
      firstCall = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ne, arguments.firstCall, none);
    }
    auto getFirstTrips = [&]() -> Value {
      if (!firstTrips) firstTrips = arith::SelectOp::create(at, loc, firstCall, one, zero);
      return firstTrips;
    };
    Value time;
    if (config.timed) time = constant(0.0);
    if (config.timed || config.stepMemory) {
      Value memory = memref::AllocaOp::create(at, loc, MemRefType::get({1}, at.getI64Type()));
      memref::StoreOp::create(at, loc, arguments.start, memory, zero);
      arith::ConstantIntOp::create(at, loc, 1, 64);
    }
    SmallVector<Value> edges{arguments.lengthX, arguments.lengthY, arguments.lengthZ};
    for (unsigned slot : tilts) edges.push_back(arguments.startValues[slot]);
    if (config.changesCell) {
      Value memory = memref::AllocaOp::create(at, loc, MemRefType::get({int64_t(config.cellEdges)}, at.getF64Type()));
      if (config.scalesEveryStep) memref::AllocaOp::create(at, loc, MemRefType::get({9}, at.getF64Type()));
      for (auto [index, edge] : llvm::enumerate(edges)) {
        Value slot = arith::ConstantIndexOp::create(at, loc, index);
        memref::StoreOp::create(at, loc, edge, memory, slot);
      }
    }
    Value scale;
    if (config.referenceEdges) {
      SmallVector<Value> reference;
      for (unsigned slot : *config.referenceEdges) reference.push_back(arguments.startValues[slot]);
      Value referenceVector = vector::FromElementsOp::create(at, loc, VectorType::get({3}, at.getF64Type()), reference);
      scale = buildReferenceScale(at, ValueRange(edges).take_front(3), referenceVector);
    }
    Value cell = buildCell(at, edges);
    auto physical = buildEntryFields(at, program, arguments, true, false);
    auto original = buildEntryPotentialFields(physical);
    physical.positions = buildEntrySites(at, {physical.positions, cell, original, time}, config.sites);
    physical = buildEntryReorder(at, program, std::move(physical), cell);
    auto fields = buildEntryPotentialFields(physical);
    auto lengths = [&](const DescentConstraintGroupIR &group) {
      SmallVector<Value> values;
      for (unsigned slot : group.parameters) values.push_back(fields[slot]);
      return values;
    };
    if (!config.restart && !config.constraints.empty()) {
      auto project = [&](OpBuilder &at, Value positions) {
        Value before = positions;
        for (const auto &group : config.constraints) {
          auto parameters = lengths(group);
          ConstraintPositionInputs input{fields[group.relation], positions, cell,
              before, physical.masses, parameters};
          auto projected = group.settlePosition
              ? buildSettlePositions(at, input, false, group.position.tolerance)
              : buildBondConstraints(at, input, group.arity, group.bonds, group.position);
          positions = buildAddChange(at, positions, projected.change);
        }
        return buildEntrySites(at, {positions, cell, fields, time}, config.sites);
      };
      if (config.segments) {
        Value trips = getFirstTrips();
        auto initial = buildEntryLoop(at, "minimization initial projection", zero, trips, one,
            ValueRange{physical.positions},
            [&](OpBuilder &body, Value, ValueRange state) -> llvm::Expected<SmallVector<Value>> {
          return SmallVector<Value>{project(body, state[0])};
        });
        if (!initial) return initial.takeError();
        physical.positions = initial->getResult(0);
      } else physical.positions = project(at, physical.positions);
    }
    PotentialStateIR state{physical.positions, cell, fields, time};
    auto evaluated = buildEntryEvaluation(at, state, {true, true, false}, config.sites, complete, scale);
    if (evaluated.storedTerms.size() != config.storedTerms)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
          "native minimization post-evaluation term count does not match its configuration");
    auto writeTerms = [&](OpBuilder &at) {
      Value terms = buildTermEnergyBuffer(at, state, config.terms, config.terms.size() + config.storedTerms);
      for (auto [index, value] : llvm::enumerate(evaluated.storedTerms)) {
        Value slot = arith::ConstantIndexOp::create(at, loc, config.terms.size() + index);
        memref::StoreOp::create(at, loc, value, terms, slot);
      }
      buildWriteTerms(at, terms);
    };
    if (config.segments) {
      // The writer omits an empty first-call branch when there are no terms.
      if (config.topology) {
        Value trips = getFirstTrips();
        auto terms = buildEntryLoop(at, "minimization initial terms", zero, trips, one, {},
            [&](OpBuilder &where, Value, ValueRange) -> llvm::Expected<SmallVector<Value>> {
          writeTerms(where);
          return SmallVector<Value>{};
        });
        if (!terms) return terms.takeError();
      }
    } else if (config.topology && !config.restart) writeTerms(at);
    Value first = config.segments ? arguments.firstSize : constant(config.firstSize);
    Value cap = constant(config.cap), grow = constant(1.2), shrink = constant(0.2);
    Value scalarOne = constant(1.0), scalarZero = constant(0.0);
    auto reported = [&](OpBuilder &body, Value positions, Value forces) {
      Value direction = buildDescentDirection(body, forces, physical.masses);
      for (const auto &group : config.constraints) {
        auto parameters = lengths(group);
        ConstraintVelocityInputs input{fields[group.relation], positions, cell, direction,
            physical.masses, parameters, {}, {}};
        direction = group.water ? buildSettleVelocities(body, input).velocities
            : buildShakeVelocities(body, input, group.arity).velocities;
      }
      auto map = mdir::md::MapParticlesOp::create(body, loc, direction.getType(), ValueRange{direction, physical.masses});
      {
        OpBuilder::InsertionGuard guard(body);
        auto vec = VectorType::get({3}, body.getF64Type());
        auto args = body.createBlock(&map.getKernel(), {}, {vec, body.getF64Type()}, {loc, loc})->getArguments();
        Value mass = vector::BroadcastOp::create(body, loc, vec, args[1]);
        Value force = arith::MulFOp::create(body, loc, mass, args[0]);
        mdir::md::YieldOp::create(body, loc, ValueRange{force});
      }
      return Value(map.getResult());
    };
    auto host = [&](OpBuilder &body, StringRef name, ValueRange inputs) {
      mdir::mdrt::HostCallOp::create(body, loc, TypeRange{},
          FlatSymbolRefAttr::get(body.getContext(), name), inputs, ValueRange{}, UnitAttr{});
    };
    Value initialForce = reported(at, physical.positions, evaluated.physical.forces);
    host(at, "mdrtWriteMinimization", {arguments.start, evaluated.physical.energy, first, initialForce, physical.identities});
    MinimizationStopIR stop;
    if (config.stops) stop = buildMinimizationStop(at, arguments.start, config.steps);
    auto run = [&](OpBuilder &at, Value start, ValueRange initial, Value partCount)
        -> llvm::Expected<SmallVector<Value>> {
    int64_t framePeriod = config.framePeriod > 0 ? config.framePeriod : config.steps;
    SmallVector<Value> counts;
    if (config.segments) {
      counts.push_back(arith::ConstantIndexOp::create(at, loc, 1));
      counts.push_back(arith::ConstantIndexOp::create(at, loc, 1));
      counts.push_back(arith::IndexCastOp::create(at, loc, at.getIndexType(), partCount));
      counts.push_back(arith::AddIOp::create(at, loc, counts[2], zero));
    } else for (int64_t count : {config.steps / framePeriod, framePeriod / config.energyPeriod, config.energyPeriod, framePeriod})
      counts.push_back(arith::ConstantIndexOp::create(at, loc, count));
    auto outer = buildEntryLoop(at, "minimization frames", zero, counts[0], one,
        initial,
        [&](OpBuilder &frame, Value frameIndex, ValueRange frameState) -> llvm::Expected<SmallVector<Value>> {
      auto rows = buildEntryLoop(frame, "minimization rows", zero, counts[1], one, frameState,
          [&](OpBuilder &row, Value rowIndex, ValueRange rowState) -> llvm::Expected<SmallVector<Value>> {
        MinimizationBoundsIR bounds{counts[2], one};
        if (config.stops) bounds = buildMinimizationBounds(row, stop, counts[2], zero, one);
        auto steps = buildEntryLoop(row, "minimization steps", zero, bounds.steps, one, rowState,
            [&](OpBuilder &step, Value, ValueRange current) -> llvm::Expected<SmallVector<Value>> {
          SmallVector<Value> inputs{current[0], current[1], physical.masses, cell, current[3]};
          llvm::append_range(inputs, fields);
          auto trial = mdir::dyn::StepOp::create(step, loc, descend.getFunctionType().getResults(), descend.getSymName(), inputs);
          Value lower = arith::CmpFOp::create(step, loc, arith::CmpFPredicate::OLT, trial.getResult(2), current[2]);
          Value longer = arith::MulFOp::create(step, loc, current[3], grow);
          Value capped = arith::MinimumFOp::create(step, loc, longer, cap);
          Value shorter = arith::MulFOp::create(step, loc, current[3], shrink);
          Value energy = arith::SelectOp::create(step, loc, lower, trial.getResult(2), current[2]);
          Value size = arith::SelectOp::create(step, loc, lower, capped, shorter);
          Value keep = arith::SelectOp::create(step, loc, lower, scalarOne, scalarZero);
          SmallVector<Value> chosen;
          for (unsigned index : {0u, 1u}) {
            auto map = mdir::md::MapParticlesOp::create(step, loc, current[index].getType(),
                ValueRange{trial.getResult(index), current[index]});
            {
              OpBuilder::InsertionGuard guard(step);
              auto vec = VectorType::get({3}, step.getF64Type());
              auto args = step.createBlock(&map.getKernel(), {}, {vec, vec}, {loc, loc})->getArguments();
              Value half = arith::ConstantOp::create(step, loc, step.getF64FloatAttr(0.5));
              Value taken = arith::CmpFOp::create(step, loc, arith::CmpFPredicate::OGT, keep, half);
              Value selected = arith::SelectOp::create(step, loc, taken, args[0], args[1]);
              mdir::md::YieldOp::create(step, loc, ValueRange{selected});
            }
            chosen.push_back(map.getResult());
          }
          return SmallVector<Value>{chosen[0], chosen[1], energy, size};
        });
        if (!steps) return steps.takeError();
        Value completed = buildStepNumber(row, {{frameIndex, {}}}, rowIndex, counts[3], counts[2], one, start);
        auto write = [&](OpBuilder &where) {
          Value force = reported(where, steps->getResult(0), steps->getResult(1));
          host(where, "mdrtWriteMinimization", {completed, steps->getResult(2), steps->getResult(3), force, physical.identities});
        };
        if (config.stops) {
          auto writeOnce = buildEntryLoop(row, "minimization report", zero, bounds.rows, one, {},
              [&](OpBuilder &where, Value, ValueRange) -> llvm::Expected<SmallVector<Value>> {
            write(where);
            host(where, "mdrtCheckMinimization", {stop.memory});
            return SmallVector<Value>{};
          });
          if (!writeOnce) return writeOnce.takeError();
        } else write(row);
        return SmallVector<Value>(steps->getResults());
      });
      if (!rows) return rows.takeError();
      if (config.framePeriod > 0) {
        Value completed = buildStepNumber(frame, {{frameIndex, {}}}, {}, {}, counts[3], one, start);
        if (config.stops) {
          Value trips = buildMinimizationFrameBound(frame, stop, completed, zero, one);
          auto write = buildEntryLoop(frame, "minimization frame", zero, trips, one, {},
              [&](OpBuilder &where, Value, ValueRange) -> llvm::Expected<SmallVector<Value>> {
            host(where, "mdrtWriteFrame", {completed, rows->getResult(0), physical.identities});
            return SmallVector<Value>{};
          });
          if (!write) return write.takeError();
        } else host(frame, "mdrtWriteFrame", {completed, rows->getResult(0), physical.identities});
      }
      return SmallVector<Value>(rows->getResults());
    });
    if (!outer) return outer.takeError();
    return SmallVector<Value>(outer->getResults());
    };
    SmallVector<Value> initial{physical.positions, evaluated.physical.forces, evaluated.physical.energy, first};
    if (config.segments) {
      Value memory = memref::AllocaOp::create(at, loc, MemRefType::get({config.partCounts}, at.getI64Type()));
      Value end = arith::ConstantIndexOp::create(at, loc, INT64_MAX);
      auto parts = buildEntryLoop(at, "minimization parts", zero, end, one, initial,
          [&](OpBuilder &part, Value, ValueRange carried) -> llvm::Expected<SmallVector<Value>> {
        auto vec = VectorType::get({3}, part.getF64Type());
        auto sum = mdir::md::SumParticlesOp::create(part, loc, vec,
            ValueRange{carried[0], physical.velocities, carried[1]});
        {
          OpBuilder::InsertionGuard guard(part);
          auto args = part.createBlock(&sum.getKernel(), {}, {vec, vec, vec}, {loc, loc, loc})->getArguments();
          Value yes = arith::ConstantOp::create(part, loc, DenseElementsAttr::get(vec, 1.0));
          Value no = arith::ConstantOp::create(part, loc, DenseElementsAttr::get(vec, 0.0));
          SmallVector<Value> bad;
          for (Value field : args) {
            Value difference = arith::SubFOp::create(part, loc, field, field);
            Value finite = arith::CmpFOp::create(part, loc, arith::CmpFPredicate::ORD, difference, difference);
            bad.push_back(arith::SelectOp::create(part, loc, finite, no, yes));
          }
          Value first = arith::AddFOp::create(part, loc, bad[0], bad[1]);
          Value all = arith::AddFOp::create(part, loc, first, bad[2]);
          mdir::md::YieldOp::create(part, loc, ValueRange{all});
        }
        Value invalid = buildSum3(part, sum.getResult());
        buildPartBoundary(part, carried[0], physical.velocities, carried[1], physical.identities, invalid, memory);
        SmallVector<Value> counts;
        for (int64_t index : {0, 1}) {
          Value slot = arith::ConstantIndexOp::create(part, loc, index);
          counts.push_back(memref::LoadOp::create(part, loc, memory, slot));
        }
        return run(part, counts[0], carried, counts[1]);
      });
      if (!parts) return parts.takeError();
      return llvm::Error::success();
    }
    auto outer = run(at, arguments.start, initial, {});
    if (!outer) return outer.takeError();
    if (config.checkpointPeriod > 0) {
      Value completed;
      if (config.stops) completed = memref::LoadOp::create(at, loc, stop.memory, stop.stepIndex);
      else {
        Value count = arith::ConstantIntOp::create(at, loc, config.steps, 64);
        completed = arith::AddIOp::create(at, loc, arguments.start, count);
      }
      host(at, "mdrtWriteCheckpoint", {completed, (*outer)[0], physical.velocities, physical.identities});
    }
    host(at, "mdrtFinish", {(*outer)[0], physical.velocities, physical.identities});
    return llvm::Error::success();
  });
}
