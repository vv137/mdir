// Frame positions and cell values are local to their native iteration.
#include "SimpleFrameEntryIR.h"
#include "EntryFieldsIR.h"
#include "EntryLoopIR.h"
#include "BarostatIR.h"
#include "KineticIR.h"
#include "PotentialIR.h"
#include "OutputIR.h"
#include "ReorderIR.h"
#include "SiteIR.h"
#include "CouplingIR.h"
#include "TermIR.h"
#include "mdir/Driver/Builder.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MDRT/MDRTOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include <limits>
using namespace mlir;
using namespace mdir::driver;
using namespace mdir::driver::builder;

namespace {
void writeFrameForces(OpBuilder &builder, Value forces, Value identities) {
  mdir::mdrt::HostCallOp::create(builder, builder.getUnknownLoc(), TypeRange{},
      FlatSymbolRefAttr::get(builder.getContext(), "mdrtWriteFrameForces"),
      ValueRange{forces, identities}, ValueRange{}, UnitAttr{});
}
}

llvm::Expected<func::FuncOp> mdir::driver::builder::buildSimpleFrameEntry(
    OpBuilder &builder, const Program &program, const SimpleFrameEntryIRConfig &config,
    llvm::function_ref<EntryCompletionIR(OpBuilder &, const PotentialStateIR &,
                                        EnergyResults, Value)> complete,
    llvm::function_ref<void(OpBuilder &, const PotentialStateIR &, Value)> coordinateOutput) {
  return buildEntryFunction(builder, program, {false, config.branchesStart},
      [&](OpBuilder &at, const EntryArgumentsIR &arguments) -> llvm::Error {
    auto loc = at.getUnknownLoc();
    Value zero = arith::ConstantIndexOp::create(at, loc, 0);
    Value one = arith::ConstantIndexOp::create(at, loc, 1);
    if (config.branchesStart) {
      Value none = arith::ConstantIntOp::create(at, loc, 0, 64);
      arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ne, arguments.firstCall, none);
    }
    Value time;
    if (config.timed) {
      time = buildSimulationTime(at, arguments.start, arguments.interval);
    }
    if (config.stepMemory || config.timed || !config.constraints.empty()) {
      Value memory = memref::AllocaOp::create(at, loc,
          MemRefType::get({1}, at.getI64Type()));
      memref::StoreOp::create(at, loc, arguments.start, memory, zero);
      arith::ConstantIntOp::create(at, loc, 1, 64);
    }
    for (int64_t count : config.counts) arith::ConstantIndexOp::create(at, loc, count);
    for (int64_t period : config.periods) arith::ConstantIndexOp::create(at, loc, period);
    auto startValue = [&](StringRef name) -> Value {
      auto found = llvm::find_if(program.startValues,
          [&](const auto &value) { return value.name == name; });
      assert(found != program.startValues.end() && "cell start value must exist");
      return arguments.startValues[std::distance(program.startValues.begin(), found)];
    };
    SmallVector<Value> replaced{arguments.positions, arguments.lengthX, arguments.lengthY, arguments.lengthZ};
    for (auto [metadata, value] : llvm::zip(program.startValues, arguments.startValues))
      if (StringRef(metadata.name).starts_with("tilt_")) replaced.push_back(value);
    Value boxMemory, trotterMemory, reference;
    if (config.changesCell) {
      boxMemory = memref::AllocaOp::create(at, loc,
          MemRefType::get({int64_t(config.cellEdges)}, at.getF64Type()));
      if (config.scalesEveryStep)
        trotterMemory = memref::AllocaOp::create(at, loc, MemRefType::get({9}, at.getF64Type()));
      SmallVector<Value> edges{arguments.lengthX, arguments.lengthY, arguments.lengthZ};
      if (config.cellEdges == 6)
        edges.append({startValue("tilt_bx"), startValue("tilt_cx"), startValue("tilt_cy")});
      for (auto [index, edge] : llvm::enumerate(edges)) {
        Value slot = arith::ConstantIndexOp::create(at, loc, index);
        memref::StoreOp::create(at, loc, edge, boxMemory, slot);
      }
      replaced.push_back(boxMemory);
    }
    if (config.referenceEdges) {
      SmallVector<Value> edges;
      for (unsigned slot : *config.referenceEdges) edges.push_back(arguments.startValues[slot]);
      reference = vector::FromElementsOp::create(at, loc, VectorType::get({3}, at.getF64Type()), edges);
    }
    auto imported = buildEntryFields(at, program, arguments, false, config.givenForces);
    CouplingMassIR massParameters;

    if (config.coupled) {
      massParameters = buildCouplingMass(at, imported.masses);
      if (config.thermostat) {
        const auto &thermostat = *config.thermostat;
        if (config.barostatParameters) buildBarostatParameters(at, *config.barostatParameters);
        arith::ConstantIntOp::create(at, loc, thermostat.seed, 64);
        for (double value : {thermostat.freedom, thermostat.target, thermostat.decay})
          arith::ConstantOp::create(at, loc, at.getF64FloatAttr(value));
      }
    }
    if (config.leapfrog && config.barostat) {
      Value back = arith::ConstantOp::create(at, loc, at.getF64FloatAttr(-0.5));
      arith::MulFOp::create(at, loc, back, arguments.interval);
    }
    Value cellMemory = memref::AllocaOp::create(at, loc,
        MemRefType::get({6}, at.getF64Type()));
    // The product program of a frame evaluator (D381, #726,
    // D380): its boundary also gives the cotangent of the energy, in
    // request 2, and that of the gradient in the positions, as returned
    // buffer 1; it evaluates the product alone. The program hands over the
    // derivatives of the energy and of its tangent apart, and the host
    // applies the cotangent of the energy with its chain rule, where what
    // it adds to the energy takes it too: request 2 is not read here.
    const std::optional<EntryGradientProductIR> product =
        config.output.gradient ? config.output.gradient->product : std::nullopt;
    Value requests = memref::AllocaOp::create(at, loc,
        MemRefType::get({product ? 3 : 2}, at.getF64Type()));
    auto storage = cast<MemRefType>(arguments.positions.getType());
    if (product) {
      // Linear sites have a constant Jacobian J: the product takes the
      // tangent J Gbar and spreads the forces with J^T
      // (D[force-matching-sites]). The sites of four-site water of Amber
      // follow the directions of the bonds, and their J does not stay.
      if (llvm::any_of(config.sites, [](const auto &site) { return site.amber; }))
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
            "the product of a frame is not implemented for a program with "
            "virtual sites placed by the directions of their bonds (the extra "
            "points of four-site water of Amber) yet; linear sites are (#771)");
      if (program.reorders)
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
            "the product program of frames keeps the order of the input; "
            "this is a defect of mdir");
      if (product->energyCotangentRequest != 2 || product->positionCotangentResult != 1)
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
            "the boundary of the product program takes the cotangent of the "
            "energy in request 2 and that of the positions as result 1");
      Operation *entry = at.getBlock()->getParentOp();
      while (entry && !isa<func::FuncOp>(entry)) entry = entry->getParentOp();
      auto module = entry->getParentOfType<ModuleOp>();
      if (!module.lookupSymbol("mdrtFrameProductBoundary")) {
        OpBuilder top(entry);
        auto declaration = func::FuncOp::create(top, loc, "mdrtFrameProductBoundary",
            top.getFunctionType({MemRefType::get({6}, top.getF64Type()),
                                 MemRefType::get({3}, top.getF64Type())},
                                {storage, storage}));
        declaration.setPrivate();
        declaration->setAttr("llvm.emit_c_interface", top.getUnitAttr());
      }
      // The forces of the energy and of its tangent (#771): the types of
      // those that mdrtWriteFrameForces takes.
      if (product->positions && !module.lookupSymbol("mdrtWriteFrameProductForces")) {
        auto forces = module.lookupSymbol<func::FuncOp>("mdrtWriteFrameForces");
        if (!forces)
          return llvm::createStringError(llvm::inconvertibleErrorCode(),
              "the product program of frames declares no mdrtWriteFrameForces; this is a "
              "defect of mdir");
        TypeRange inputs = forces.getFunctionType().getInputs();
        OpBuilder top(entry);
        auto declaration = func::FuncOp::create(top, loc, "mdrtWriteFrameProductForces",
            top.getFunctionType({inputs[0], inputs[0], inputs[1]}, {}));
        declaration.setPrivate();
        declaration->setAttr("llvm.emit_c_interface", top.getUnitAttr());
      }
    }
    SmallVector<Value> indices;
    for (int64_t index = 0; index != config.cellEdges; ++index)
      indices.push_back(arith::ConstantIndexOp::create(at, loc, index));
    Value end = arith::ConstantIndexOp::create(at, loc, std::numeric_limits<int64_t>::max());
    auto frames = buildEntryLoop(at, "frames", zero, end, one, {},
        [&](OpBuilder &frame, Value, ValueRange) -> llvm::Expected<SmallVector<Value>> {
      auto boundary = product
          ? func::CallOp::create(frame, loc, "mdrtFrameProductBoundary",
                TypeRange{storage, storage}, ValueRange{cellMemory, requests})
          : func::CallOp::create(frame, loc, "mdrtFrameBoundary",
                TypeRange{storage}, ValueRange{cellMemory, requests});
      SmallVector<Value> edges, asks;
      for (Value index : indices)
        edges.push_back(memref::LoadOp::create(frame, loc, cellMemory, index));
      for (unsigned index = 0; index != 2; ++index)
        asks.push_back(memref::LoadOp::create(frame, loc, requests, indices[index]));
      Value scale;
      if (config.referenceEdges) scale = buildReferenceScale(frame, ValueRange(edges).take_front(3), reference);
      Value cell = buildCell(frame, edges);
      auto physical = imported;
      physical.positions = buildEntryField(frame, boundary.getResult(0), 3, false);
      auto originalParameters = buildEntryPotentialFields(physical);
      physical.positions = buildEntrySites(frame,
          {physical.positions, cell, originalParameters, time}, config.sites);
      physical = buildEntryReorder(frame, program, std::move(physical), cell);
      auto parameters = buildEntryPotentialFields(physical);
      PotentialStateIR state{physical.positions, cell, parameters, time};
      if (product) {
        const auto &gradient = *config.output.gradient;
        Value tangent = buildEntryField(frame,
            boundary.getResult(product->positionCotangentResult), 3, false);
        // The tangent of the placement: J Gbar, in place of the entries of
        // the sites, whose own cotangent meets a gradient that is zero.
        for (const auto &site : config.sites)
          tangent = buildPlacedSiteTangents(frame,
              {originalParameters[site.relation], physical.positions, cell,
               originalParameters[site.a], originalParameters[site.b], tangent, {}});
        auto spread = [&](OpBuilder &at, Value forces) {
          for (const auto &site : config.sites)
            forces = buildSiteSpread(at,
                {state.fields[site.relation], state.positions, state.cell,
                 state.fields[site.a], state.fields[site.b], forces, {}},
                site.amber, site.derived, site.potential).forces;
          return forces;
        };
        auto iteration = buildEntryLoop(frame, "frame product", zero, one, one, {},
            [&](OpBuilder &body, Value, ValueRange) -> llvm::Expected<SmallVector<Value>> {
          SmallVector<Value> constants, seeds;
          for (unsigned slot : gradient.constants) constants.push_back(arguments.startValues[slot]);
          if (gradient.seeds) seeds.assign(gradient.seeds, state.fields[*gradient.seedField]);
          buildTunableProductOutput(body, state, tangent,
              physical.identities, constants, seeds, gradient.output, product->positions,
              config.sites.empty() ? llvm::function_ref<Value(OpBuilder &, Value)>() : spread);
          return SmallVector<Value>{};
        });
        if (!iteration) return iteration.takeError();
        (*iteration)->setAttr("mdrt.segment", frame.getUnitAttr());
        if (llvm::Error error = checkFrameEvaluation(*iteration->getOperation()))
          return std::move(error);
        return SmallVector<Value>{};
      }
      SmallVector<Value> storedTerms;
      auto evaluate = [&](OpBuilder &at, EnergyRequests requests) {
        auto result = buildEntryEvaluation(at, state, requests, config.sites, complete, scale);
        if (requests.energy) storedTerms = std::move(result.storedTerms);
        return result.physical;
      };
      auto iteration = buildEntryLoop(frame, "frame evaluation", zero, one, one, {},
          [&](OpBuilder &body, Value, ValueRange) -> llvm::Expected<SmallVector<Value>> {
        auto energy = evaluate(body, {true, true, true});
        if (storedTerms.size() != config.storedTerms)
          return llvm::createStringError(llvm::inconvertibleErrorCode(),
              "native frame evaluation returned an unexpected stored-term count");
        Value nil = arith::ConstantOp::create(body, loc, body.getF64FloatAttr(0.0));
        Value force = arith::CmpFOp::create(body, loc, arith::CmpFPredicate::ONE, asks[0], nil);
        Value virial = arith::CmpFOp::create(body, loc, arith::CmpFPredicate::ONE, asks[1], nil);
        Value forceCount = arith::SelectOp::create(body, loc, force, one, zero);
        Value virialCount = arith::SelectOp::create(body, loc, virial, one, zero);
        Value forceOnlyCount = arith::SelectOp::create(body, loc, virial, zero, forceCount);
        auto forceOnly = buildEntryLoop(body, "frame force request", zero, forceOnlyCount, one, {},
            [&](OpBuilder &requested, Value, ValueRange) -> llvm::Expected<SmallVector<Value>> {
          auto result = evaluate(requested, {false, true, false});
          writeFrameForces(requested, result.forces, physical.identities);
          return SmallVector<Value>{};
        });
        if (!forceOnly) return forceOnly.takeError();
        auto withVirial = buildEntryLoop(body, "frame virial request", zero, virialCount, one, {},
            [&](OpBuilder &requested, Value, ValueRange) -> llvm::Expected<SmallVector<Value>> {
          auto result = evaluate(requested, {false, true, true});
          auto forces = buildEntryLoop(requested, "frame shared force request", zero, forceCount, one, {},
              [&](OpBuilder &write, Value, ValueRange) -> llvm::Expected<SmallVector<Value>> {
            writeFrameForces(write, result.forces, physical.identities);
            return SmallVector<Value>{};
          });
          if (!forces) return forces.takeError();
          SmallVector<Value> entries;
          for (int index = 0; index != 9; ++index)
            entries.push_back(vector::ExtractOp::create(requested, loc, result.virial, index));
          func::CallOp::create(requested, loc, "mdrtWriteFrameVirial", TypeRange{}, entries);
          return SmallVector<Value>{};
        });
        if (!withVirial) return withVirial.takeError();
        if (config.topology) {
          Value terms = buildTermEnergyBuffer(body, state, config.terms,
              config.terms.size() + config.storedTerms);
          for (auto [index, value] : llvm::enumerate(storedTerms)) {
            Value slot = arith::ConstantIndexOp::create(body, loc, config.terms.size() + index);
            memref::StoreOp::create(body, loc, value, terms, slot);
          }
          buildWriteTerms(body, terms);
          auto diagonal = buildVirialDiagonal(body, energy.virial);
          func::CallOp::create(body, loc, "mdrtWriteVirial", TypeRange{},
              ValueRange{diagonal.xx, diagonal.yy, diagonal.zz});
        }
        Value kinetic = buildKineticEnergy(body, physical.velocities, physical.masses);
        Value correction;
        if (config.constraints.empty() && !config.brownian) {
          Value excess = buildKineticExcess(body, energy.forces, physical.masses, arguments.interval);
          correction = buildSum3(body, excess);
        } else {
          correction = arith::ConstantOp::create(body, loc, body.getF64FloatAttr(0.0));
        }
        Value trace = buildVirialTrace(body, energy.virial);
        if (!config.constraints.empty())
          trace = buildEntryConstraintTrace(body, physical, state, energy.forces, trace, config.constraints, false);
        Value pressureKinetic = arith::AddFOp::create(body, loc, kinetic, correction);
        func::CallOp::create(body, loc, "mdrtWriteEnergies", TypeRange{},
            ValueRange{arguments.start, energy.energy, kinetic, correction, trace, pressureKinetic});
        if (coordinateOutput) coordinateOutput(body, state, arguments.start);
        buildEntryOutputs(body,
            {state, arguments.start, physical.identities, arguments.interval, zero, one,
             arguments.startValues, true, config.timed}, config.output);
        if (config.scalesEveryStep) {
          auto virial = buildVirialDiagonal(body, energy.virial);
          Value diagonal = virial.vector;
          if (!config.constraints.empty())
            diagonal = buildEntryConstraintTrace(body, physical, state, energy.forces, diagonal, config.constraints, true);
          SmallVector<RigidGroupIR> groups;
          for (const auto &group : config.constraints)
            groups.push_back({parameters[group.relation], group.arity});
          Value molecular = virial.vector;
          if (!groups.empty())
            molecular = buildMolecularDiagonal(body,
                {virial.vector, physical.positions, energy.forces, cell, physical.masses, groups});
          auto kinetic = buildGroupKinetic(body,
              {groups.empty() ? Value{} : physical.positions, physical.velocities, energy.forces,
               groups.empty() ? Value{} : cell, physical.masses, arguments.interval,
               config.removesMotion ? massParameters.totalMass : Value{}, groups});
          buildStoreBarostatState(body, trotterMemory, diagonal, molecular, kinetic.half);
        }
        if (config.leapfrog) {
          Value back = arith::ConstantOp::create(body, loc, body.getF64FloatAttr(-0.5));
          Value interval = arith::MulFOp::create(body, loc, back, arguments.interval);
          if (config.branchesStart) {
            Value none = arith::ConstantOp::create(body, loc, body.getF64FloatAttr(0.0));
            Value first = arith::ConstantIntOp::create(body, loc, 1, 64);
            Value begins = arith::CmpIOp::create(body, loc, arith::CmpIPredicate::eq,
                arguments.firstCall, first);
            interval = arith::SelectOp::create(body, loc, begins, interval, none);
          }
          buildKick(body, physical.velocities, energy.forces, physical.masses, interval);
        }
        return SmallVector<Value>{};
      });
      if (!iteration) return iteration.takeError();
      (*iteration)->setAttr("mdrt.segment", frame.getUnitAttr());
      if (llvm::Error error = checkFrameEvaluation(*iteration->getOperation()))
        return std::move(error);
      return SmallVector<Value>{};
    });
    if (!frames) return frames.takeError();
    if (!arguments.positions.use_empty())
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
          "native frame iteration read entry positions or cell values");
    if (llvm::Error error = checkFrameInputs(*frames->getOperation(), replaced))
      return std::move(error);
    return llvm::Error::success();
  });
}
