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
#include "FrameBatchTunableIR.h"
#include "FrameBatchOutputIR.h"
#include "FrameBatchEvaluationIR.h"
#include "FrameBatchMixIR.h"
#include "FrameBatchLJSlopeIR.h"
#include "FrameBatchPMEIR.h"
#include "FrameBatchPositionIR.h"
#include "FrameBatchProductsIR.h"
#include "FrameBatchInputIR.h"
#include "mdir/Driver/Builder.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MDRT/MDRTOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
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
    Value frameStatus, frameCorrections;
    Value correctionSigma, correctionEpsilon, correctionCounts, correctionExcluded;
    Value correctionSigmaMap, correctionEpsilonMap;
    Value rawParameters, rowOffsets, sourceIndices, rowWeights;
    Value rowWork;
    Value safeTheta, initialTheta, thetaKinds;
    func::FuncOp parameterReader;
    Value rawKinds, rawVolume, rawFixed, typeOffsets, particlesByType;
    std::array<Value, 2> logicalTypeMaps, logicalTypeValues;
    Value fixedPairMask, orderedTypePairs;
    SmallVector<Value> seedWeights;
    Value logicalCharges, chargeMap, coulombCorrections;
    Value observableVolume, observableFixed;
    Value tailWorkspace;
    llvm::StringMap<Value> tailTables;
    if (config.batchTails && !config.batchTails->terms.empty()) {
      auto workspace = MemRefType::get({4001, 5}, at.getF64Type(),
          MemRefLayoutAttrInterface(), at.getI64IntegerAttr(1));
      tailWorkspace = gpu::AllocOp::create(at, loc, workspace, Type(),
          ValueRange(), ValueRange(), ValueRange()).getMemref();
      for (const auto &table : program.tables) {
        bool used = llvm::any_of(config.batchTails->terms, [&](const auto &term) {
          return llvm::is_contained(term.expression.getInputs({}).tables, table.name);
        });
        if (!used) continue;
        // Host quadrature uses the logical double coefficients even when
        // ordinary potential kernels store their tables in f32.
        auto coefficients = buildFrameBatchInitialValues(at, table.values);
        if (!coefficients) return coefficients.takeError();
        auto type = MemRefType::get({table.count, table.getColumns()}, at.getF64Type(),
            MemRefLayoutAttrInterface(), at.getI64IntegerAttr(1));
        SmallVector<OpFoldResult> sizes{at.getIndexAttr(table.count), at.getIndexAttr(table.getColumns())};
        SmallVector<OpFoldResult> strides{at.getIndexAttr(table.getColumns()), at.getIndexAttr(1)};
        tailTables[table.name] = memref::ReinterpretCastOp::create(at, loc, type,
            *coefficients, at.getIndexAttr(0), sizes, strides);
      }
    }
    if (config.deviceBatch) {
      Attribute space = config.batchOnDevice ? Attribute(at.getI64IntegerAttr(1)) : Attribute{};
      auto type = MemRefType::get({1}, at.getI64Type(), MemRefLayoutAttrInterface(), space);
      if (config.batchOnDevice)
        frameStatus = gpu::AllocOp::create(at, loc, type, Type(),
            ValueRange(), ValueRange(), ValueRange()).getMemref();
      else
        frameStatus = memref::AllocaOp::create(at, loc, type);
      if (config.batchOnDevice) {
        std::array<double, 5> values{};
        if (config.batchParameters) values = config.batchParameters->corrections;
        auto uploaded = buildFrameBatchInitialValues(at, values);
        if (!uploaded) return uploaded.takeError();
        frameCorrections = *uploaded;
        auto observedVolume = buildFrameBatchInitialValues(at, program.observableVolumeConstants);
        if (!observedVolume) return observedVolume.takeError();
        observableVolume = *observedVolume;
        auto observedFixed = buildFrameBatchInitialValues(at, program.observableFixedConstants);
        if (!observedFixed) return observedFixed.takeError();
        observableFixed = *observedFixed;
        if (config.batchParameters) {
          const auto &plan = *config.batchParameters;
          for (unsigned i = 0; i != 2; ++i) {
            auto map = buildFrameBatchParameterMap(at,
                i == 0 ? plan.sigmaMap : plan.epsilonMap, plan.parameters);
            if (!map) return map.takeError();
            logicalTypeMaps[i] = *map;
            auto values = buildFrameBatchInitialValues(at,
                i == 0 ? plan.typeSigma : plan.typeEpsilon);
            if (!values) return values.takeError();
            logicalTypeValues[i] = *values;
          }
          std::vector<int64_t> fixed(size_t(plan.types) * plan.types, 0);
          for (size_t i = 0; i != plan.fixedPairs.size(); ++i) fixed[i] = plan.fixedPairs[i];
          auto mask = buildFrameBatchParameterMap(at, fixed, 2);
          if (!mask) return mask.takeError();
          fixedPairMask = *mask;
          for (const auto &seed : plan.seeds) {
            auto weights = buildFrameBatchInitialValues(at, seed.weights);
            if (!weights) return weights.takeError();
            seedWeights.push_back(*weights);
          }
          if (plan.pme) {
            auto charges = buildFrameBatchInitialValues(at, plan.initialCharges);
            if (!charges) return charges.takeError();
            logicalCharges = *charges;
            auto map = buildFrameBatchParameterMap(at, plan.chargeMap, plan.parameters);
            if (!map) return map.takeError();
            chargeMap = *map;
            auto constants = buildFrameBatchInitialValues(at,
                std::array<double, 3>{program.coulombSelfEnergy,
                    program.coulombConstantEnergy, program.coulombConstantVirial});
            if (!constants) return constants.takeError();
            coulombCorrections = *constants;
          }
        }
        if (config.batchParameters && config.batchParameters->lj) {
          const auto &plan = *config.batchParameters;
          auto upload = [&](ArrayRef<double> values, Value &destination) -> llvm::Error {
            auto memory = buildFrameBatchInitialValues(at, values);
            if (!memory) return memory.takeError();
            destination = *memory;
            return llvm::Error::success();
          };
          if (auto error = upload(plan.initialSigma, correctionSigma)) return error;
          if (auto error = upload(plan.initialEpsilon, correctionEpsilon)) return error;
          if (auto error = upload(plan.typeCounts, correctionCounts)) return error;
          auto exclusions = buildFrameBatchParameterMap(at, plan.excludedTypes, plan.types);
          if (!exclusions) return exclusions.takeError();
          correctionExcluded = *exclusions;
          auto sigma = buildFrameBatchParameterMap(at, plan.sigmaPairs, plan.parameters);
          if (!sigma) return sigma.takeError();
          correctionSigmaMap = *sigma;
          auto epsilon = buildFrameBatchParameterMap(at, plan.epsilonPairs, plan.parameters);
          if (!epsilon) return epsilon.takeError();
          correctionEpsilonMap = *epsilon;
          auto pairs = buildFrameBatchInitialValues(at, plan.orderedTypePairs);
          if (!pairs) return pairs.takeError();
          orderedTypePairs = *pairs;
        }
      }
    }
    SmallVector<Value> parameterMaps, parameterDestinations;
    SmallVector<Type> parameterTypes;
    SmallVector<std::string> parameterUpdates;
    SmallVector<SmallVector<Value>> parameterExtraInputs;
    if (config.batchParameters) {
      if (!config.deviceBatch || !config.batchOnDevice)
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
            "device frame parameters require a GPU batch entry");
      const auto &plan = *config.batchParameters;
      auto initial = buildFrameBatchInitialValues(at, plan.initialParameters);
      if (!initial) return initial.takeError();
      initialTheta = *initial;
      auto kind = buildFrameBatchParameterMap(at, plan.parameterKinds, 3);
      if (!kind) return kind.takeError();
      thetaKinds = *kind;
      auto safeType = MemRefType::get({plan.parameters}, at.getF64Type(),
          MemRefLayoutAttrInterface(), at.getI64IntegerAttr(1));
      Value safe = gpu::AllocOp::create(at, loc, safeType, Type(),
          ValueRange(), ValueRange(), ValueRange()).getMemref();
      auto dynamic = MemRefType::get({ShapedType::kDynamic}, at.getF64Type(),
          MemRefLayoutAttrInterface(), at.getI64IntegerAttr(1));
      safeTheta = memref::CastOp::create(at, loc, dynamic, safe);
      auto reader = buildFrameBatchParameterReader(at);
      if (!reader) return reader.takeError();
      parameterReader = *reader;
      auto raw = buildFrameBatchInitialValues(at, std::vector<double>(plan.rawValues, 0.0));
      if (!raw) return raw.takeError();
      rawParameters = *raw;
      auto row = buildFrameBatchInitialValues(at, std::vector<double>(plan.parameters, 0.0));
      if (!row) return row.takeError();
      rowWork = *row;
      auto offsets = buildFrameBatchParameterMap(at, plan.rowOffsets,
          plan.sourceIndices.size() + 1);
      if (!offsets) return offsets.takeError();
      rowOffsets = *offsets;
      auto sources = buildFrameBatchParameterMap(at, plan.sourceIndices, plan.rawValues);
      if (!sources) return sources.takeError();
      sourceIndices = *sources;
      auto weights = buildFrameBatchInitialValues(at, plan.weights);
      if (!weights) return weights.takeError();
      rowWeights = *weights;
      auto kinds = buildFrameBatchParameterMap(at, plan.rawKinds, 3);
      if (!kinds) return kinds.takeError();
      rawKinds = *kinds;
      auto volume = buildFrameBatchInitialValues(at, plan.rawVolumeConstants);
      if (!volume) return volume.takeError();
      rawVolume = *volume;
      auto fixed = buildFrameBatchInitialValues(at, plan.rawFixedConstants);
      if (!fixed) return fixed.takeError();
      rawFixed = *fixed;
      auto types = buildFrameBatchParameterMap(at, plan.typeOffsets, plan.particlesByType.size() + 1);
      if (!types) return types.takeError();
      typeOffsets = *types;
      auto particles = buildFrameBatchParameterMap(at, plan.particlesByType, plan.particlesByType.size());
      if (!particles) return particles.takeError();
      particlesByType = *particles;
      for (const auto &binding : plan.mapped) {
        auto map = buildFrameBatchParameterMap(at, binding.map, plan.parameters);
        if (!map) return map.takeError();
        parameterMaps.push_back(*map);
        Value field;
        Type element;
        SmallVector<int64_t> shape;
        if (binding.kind == FrameBatchMappedResource::Field) {
          field = imported.fields[binding.resource];
          element = program.parameter == Element::F64 ? Type(at.getF64Type()) : Type(at.getF32Type());
          shape = {ShapedType::kDynamic};
        } else if (binding.kind == FrameBatchMappedResource::TupleField) {
          field = imported.tuples[binding.resource].fields[binding.field];
          element = at.getF64Type();
          shape = {ShapedType::kDynamic};
        } else {
          field = imported.tables[binding.resource];
          element = program.parameter == Element::F64 ? Type(at.getF64Type()) : Type(at.getF32Type());
          shape = {ShapedType::kDynamic, ShapedType::kDynamic};
        }
        auto type = MemRefType::get(shape, element, MemRefLayoutAttrInterface(),
            at.getI64IntegerAttr(1));
        Value memory = mdir::mdrt::ToBufferOp::create(at, loc, type, field);
        // Return the updated descriptor through an effectful call. A fresh
        // from_buffer below must depend on this call: from_buffer is Pure,
        // so importing an invariant root descriptor could otherwise hoist
        // the field and its derived values above the parameter update.
        auto entry = at.getInsertionBlock()->getParentOp();
        auto module = entry->getParentOfType<ModuleOp>();
        std::string name;
        unsigned serial = 0;
        do { name = "__mdir_frame_update_" + std::to_string(serial++); }
        while (module.lookupSymbol(name));
        OpBuilder top(entry);
        auto thetaType = MemRefType::get({ShapedType::kDynamic}, at.getF64Type(),
            MemRefLayoutAttrInterface(), at.getI64IntegerAttr(1));
        bool derived = binding.update != FrameBatchMappedResource::Direct;
        bool sigma = binding.update == FrameBatchMappedResource::MixSigma ||
                     binding.update == FrameBatchMappedResource::SeedSigma;
        bool seed = binding.update == FrameBatchMappedResource::SeedSigma ||
                    binding.update == FrameBatchMappedResource::SeedEpsilon;
        Value logical;
        SmallVector<Value> extra;
        if (derived) {
          extra = {logicalTypeMaps[sigma ? 0 : 1], logicalTypeValues[sigma ? 0 : 1], fixedPairMask};
          logical = seed ? seedWeights[binding.seed] :
              (sigma ? correctionSigma : correctionEpsilon);
          if (logical) extra.push_back(logical);
          extra.push_back(frameStatus);
        }
        SmallVector<Type> updateTypes{thetaType, map->getType(), type};
        for (Value input : extra) updateTypes.push_back(input.getType());
        auto update = func::FuncOp::create(top, loc, name,
            top.getFunctionType(updateTypes, {type}));
        update.setPrivate();
        OpBuilder body = OpBuilder::atBlockEnd(update.addEntryBlock());
        Value destination = update.getArgument(2);
        if (shape.size() == 2) {
          auto vector = MemRefType::get({ShapedType::kDynamic}, element,
              MemRefLayoutAttrInterface(), at.getI64IntegerAttr(1));
          destination = memref::CollapseShapeOp::create(body, loc, vector, destination,
              ArrayRef<ReassociationIndices>{{0, 1}});
        }
        if (derived) {
          bool geometric = seed ? plan.seeds[binding.seed].geometric :
              (sigma ? plan.geometricSigma : true);
          if (auto error = buildFrameBatchMixing(body, update.getArgument(0),
                  update.getArgument(3), update.getArgument(4), update.getArgument(1),
                  update.getArgument(5), destination,
                  logical ? update.getArgument(6) : Value{},
                  {plan.types, geometric, seed}, update.getArgument(logical ? 7 : 6))) return error;
        } else {
          if (llvm::Error error = buildFrameBatchMappedParameters(body,
                  update.getArgument(0), update.getArgument(1), destination))
            return std::move(error);
        }
        func::ReturnOp::create(body, loc, update.getArgument(2));
        parameterDestinations.push_back(memory);
        parameterTypes.push_back(field.getType());
        parameterUpdates.push_back(name);
        parameterExtraInputs.push_back(extra);
      }
    }
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
      // The product takes the tangent J Gbar of the placement, and the
      // variant spreads the forces with J^T (D384). The
      // J of the sites of four-site water of Amber, which follow the
      // directions of the bonds, depends on the positions: the derivative
      // in the positions would need its curvature as well.
      if (product->positions &&
          llvm::any_of(config.sites, [](const auto &site) { return site.amber; }))
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
            "the derivative in the positions of the product of a frame is not "
            "implemented for a program with virtual sites placed by the directions "
            "of their bonds (the extra points of four-site water of Amber) yet; "
            "linear sites are, and the product in theta takes both (#771)");
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
    // A loop's body is constructed before the loop is attached to its
    // parent. Capture the owning entry here rather than walking ancestors
    // from a still-detached batch loop body.
    auto batchEntry = cast<func::FuncOp>(at.getInsertionBlock()->getParentOp());
    bool acceptsF32 = config.deviceBatch && config.batchOnDevice &&
        program.state == Element::F64 && program.force == Element::F32;
    Value positionScratch, positionBaseline;
    func::FuncOp positionReader;
    if (config.deviceBatch && config.batchOnDevice) {
      Value particles = memref::DimOp::create(at, loc, arguments.masses, zero);
      Type state = program.state == Element::F64 ? Type(at.getF64Type()) : Type(at.getF32Type());
      Value width = arith::ConstantIndexOp::create(at, loc, state.isF64() ? 24 : 12);
      Value count = arith::MulIOp::create(at, loc, particles, width);
      auto bytes = MemRefType::get({ShapedType::kDynamic}, at.getI8Type(),
          MemRefLayoutAttrInterface(), at.getI64IntegerAttr(1));
      positionScratch = gpu::AllocOp::create(at, loc, bytes, Type(),
          ValueRange(), ValueRange{count}, ValueRange()).getMemref();
      auto initial = buildFrameBatchInitialValues(at, config.batchParameters
          ? ArrayRef<double>(config.batchParameters->initialPositions) : ArrayRef<double>());
      if (!initial) return initial.takeError();
      positionBaseline = *initial;
      auto reader = buildFrameBatchPositionReader(at, bytes, state, acceptsF32,
          config.batchParameters.has_value());
      if (!reader) return reader.takeError();
      positionReader = *reader;
    }
    if (config.deviceBatch) {
      auto module = batchEntry->getParentOfType<ModuleOp>();
      if (!module.lookupSymbol("mdrtFrameBatchFrame")) {
        OpBuilder top(batchEntry);
        auto declaration = func::FuncOp::create(top, loc, "mdrtFrameBatchFrame",
            top.getFunctionType({top.getI64Type()}, {}));
        declaration.setPrivate();
      }
    }
    auto emitFrame = [&](OpBuilder &frame, Value inputPositions,
                         ValueRange edges, ValueRange asks, Value productInput,
                         ValueRange startValues, const EntryFieldsIR &parameterFields,
                         const FrameBatchEvaluationPoint *batchPoint = nullptr)
        -> llvm::Expected<SmallVector<Value>> {
      Value scale;
      if (config.referenceEdges) scale = buildReferenceScale(frame, ValueRange(edges).take_front(3), reference);
      Value cell = buildCell(frame, edges);
      auto physical = parameterFields;
      physical.positions = buildEntryField(frame, inputPositions, 3, false);
      auto originalParameters = buildEntryPotentialFields(physical);
      physical.positions = buildEntrySites(frame,
          {physical.positions, cell, originalParameters, time}, config.sites);
      physical = buildEntryReorder(frame, program, std::move(physical), cell);
      auto parameters = buildEntryPotentialFields(physical);
      PotentialStateIR state{physical.positions, cell, parameters, time};
      if (batchPoint) {
        if (llvm::Error error = buildFrameBatchEvaluation(frame, program, config,
                physical, state, *batchPoint, complete, scale))
          return std::move(error);
        return SmallVector<Value>{};
      }
      if (product) {
        const auto &gradient = *config.output.gradient;
        Value tangent = buildEntryField(frame,
            productInput, 3, false);
        // The tangent of the placement: J Gbar, in place of the entries of
        // the sites, whose own cotangent meets a gradient that is zero.
        for (const auto &site : config.sites)
          tangent = buildPlacedSiteTangents(frame,
              {originalParameters[site.relation], physical.positions, cell,
               originalParameters[site.a], originalParameters[site.b], tangent, {}},
              site.amber);
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
          for (unsigned slot : gradient.constants) constants.push_back(startValues[slot]);
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
             startValues, true, config.timed}, config.output);
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
    };
    auto frames = buildEntryLoop(at, "frames", zero, end, one, {},
        [&](OpBuilder &frame, Value, ValueRange) -> llvm::Expected<SmallVector<Value>> {
      if (config.deviceBatch) {
        if (product)
          return llvm::createStringError(llvm::inconvertibleErrorCode(),
              "the device batch entry does not select the separate force product program");
        auto entry = batchEntry;
        entry->setAttr("mdir.frame_batch", frame.getUnitAttr());
        auto module = entry->getParentOfType<ModuleOp>();
        Attribute space = config.batchOnDevice ? Attribute(frame.getI64IntegerAttr(1)) : Attribute{};
        auto bytes = MemRefType::get({ShapedType::kDynamic}, frame.getI8Type(),
            MemRefLayoutAttrInterface(), space);
        auto cells = MemRefType::get({ShapedType::kDynamic, 6}, frame.getF64Type());
        auto parameters = MemRefType::get({ShapedType::kDynamic}, frame.getF64Type(),
            MemRefLayoutAttrInterface(), space);
        auto strain = MemRefType::get({ShapedType::kDynamic, 9}, frame.getF64Type(),
            MemRefLayoutAttrInterface(), space);
        SmallVector<Type> results{bytes, cells, parameters, parameters, bytes, strain, bytes,
                                  frame.getI64Type(), parameters, parameters, parameters, parameters, bytes,
                                  frame.getI64Type()};
        if (!module.lookupSymbol("mdrtFrameBatchBoundary")) {
          OpBuilder top(entry);
          auto declaration = func::FuncOp::create(top, loc, "mdrtFrameBatchBoundary",
              top.getFunctionType({MemRefType::get({2}, top.getF64Type()), frameStatus.getType()},
                  results));
          declaration.setPrivate();
          declaration->setAttr("llvm.emit_c_interface", top.getUnitAttr());
        }
        auto batch = func::CallOp::create(frame, loc, "mdrtFrameBatchBoundary",
            results, ValueRange{requests, frameStatus});
        if (config.batchOnDevice) {
          if (llvm::Error error = buildFrameBatchStatus(frame, frameStatus, batch.getResult(13)))
            return std::move(error);
          buildFrameBatchProductStart(frame, batch.getResult(11));
        } else {
          Value success = arith::ConstantIntOp::create(frame, loc,
              std::numeric_limits<int64_t>::max(), 64);
          memref::StoreOp::create(frame, loc, success, frameStatus, zero);
        }
        Value thetaValues = batch.getResult(2);
        if (parameterReader)
          thetaValues = func::CallOp::create(frame, loc, parameterReader.getSymName(),
              parameterReader.getResultTypes(), ValueRange{thetaValues, safeTheta,
                  initialTheta, thetaKinds, frameStatus}).getResult(0);
        auto parameterFields = imported;
        if (config.batchParameters)
          for (auto [index, binding] : llvm::enumerate(config.batchParameters->mapped)) {
            SmallVector<Value> inputs{thetaValues, parameterMaps[index], parameterDestinations[index]};
            llvm::append_range(inputs, parameterExtraInputs[index]);
            auto updated = func::CallOp::create(frame, loc, parameterUpdates[index],
                TypeRange{parameterDestinations[index].getType()},
                inputs);
            Value borrowed = mdir::mdrt::FromBufferOp::create(frame, loc,
                parameterTypes[index], updated.getResult(0));
            if (binding.kind == FrameBatchMappedResource::Field)
              parameterFields.fields[binding.resource] = borrowed;
            else if (binding.kind == FrameBatchMappedResource::TupleField)
              parameterFields.tuples[binding.resource].fields[binding.field] = borrowed;
            else
              parameterFields.tables[binding.resource] = borrowed;
          }
        SmallVector<Value> parameterValues(arguments.startValues);
        if (config.batchParameters && config.batchParameters->pme) {
          if (auto error = buildFrameBatchMappedParameters(frame, thetaValues,
                  chargeMap, logicalCharges)) return error;
          buildFrameBatchPMEConstants(frame, logicalCharges, coulombCorrections,
              rawVolume, rawFixed, *config.batchParameters->pme);
        }
        if (config.batchParameters && config.batchParameters->lj) {
          if (auto error = buildFrameBatchMappedParameters(frame, thetaValues,
                  correctionSigmaMap, correctionSigma)) return error;
          if (auto error = buildFrameBatchMappedParameters(frame, thetaValues,
                  correctionEpsilonMap, correctionEpsilon)) return error;
          if (auto error = buildFrameBatchLJConstants(frame, correctionSigma,
                  correctionEpsilon, correctionCounts, correctionExcluded,
                  frameCorrections, *config.batchParameters->lj)) return error;
          const auto &plan = *config.batchParameters;
          for (auto [i, seed] : llvm::enumerate(plan.seeds))
            if (seed.rawBase >= 0)
              buildFrameBatchLJSlopes(frame, correctionSigma, correctionEpsilon,
                  orderedTypePairs, seedWeights[i], rawVolume, rawFixed,
                  {plan.types, seed.rawBase, seed.sigma, plan.ljSlopeFactor, plan.ljFixedFactor});
        }
        if (tailWorkspace)
          if (auto error = buildFrameBatchTails(frame, *config.batchTails,
                  thetaValues, tailWorkspace, frameCorrections, rawVolume, rawFixed,
                  observableVolume, observableFixed, tailTables)) return error;
        if (config.batchParameters)
          for (const auto &scalar : config.batchParameters->scalars) {
            Value index = arith::ConstantIndexOp::create(frame, loc, scalar.parameter);
            auto value = memref::LoadOp::create(frame, loc, thetaValues, index);
            value->setAttr("mdir.frame_parameter", frame.getUnitAttr());
            parameterValues[scalar.start] = value;
          }
        SmallVector<Value> asks;
        for (unsigned index = 0; index != 2; ++index)
          asks.push_back(memref::LoadOp::create(frame, loc, requests, indices[index]));
        Value particles = memref::DimOp::create(frame, loc, arguments.masses, zero);
        auto rows = buildFrameBatchRows(frame, batch.getResult(0), batch.getResult(1),
            particles, storage.getElementType(), config.cellEdges,
            [&](OpBuilder &row, const FrameBatchRowIR &inputs) -> llvm::Error {
          Value index = arith::IndexCastOp::create(row, loc, row.getI64Type(), inputs.index);
          func::CallOp::create(row, loc, "mdrtFrameBatchFrame", TypeRange{}, ValueRange{index});
          Value inputPositions = inputs.positions;
          if (positionReader)
            inputPositions = func::CallOp::create(row, loc, positionReader.getSymName(),
                positionReader.getResultTypes(), ValueRange{batch.getResult(0), inputs.index,
                    particles, batch.getResult(7), positionScratch,
                    positionBaseline, frameStatus}).getResult(0);
          FrameBatchEvaluationPoint point;
          if (config.batchOnDevice)
            point = {inputs.index, batch.getResult(3), batch.getResult(4),
                batch.getResult(5), frameStatus, frameCorrections,
                inputs.edges, asks, parameterValues};
          if (config.batchParameters) {
            point.raw = rawParameters;
            point.rows = batch.getResult(6);
            point.rowOffsets = rowOffsets;
            point.sourceIndices = sourceIndices;
            point.weights = rowWeights;
            point.rowWork = rowWork;
            point.tangent = batch.getResult(8);
            point.cotangent = batch.getResult(9);
            point.jvp = batch.getResult(10);
            point.vjp = batch.getResult(11);
            point.rawKinds = rawKinds;
            point.rawVolume = rawVolume;
            point.rawFixed = rawFixed;
            point.typeOffsets = typeOffsets;
            point.particlesByType = particlesByType;
            point.coulombCorrections = coulombCorrections;
          }
          if (config.batchOnDevice) point.positionWidth = batch.getResult(7);
          if (config.batchOnDevice) {
            point.aux = batch.getResult(12);
            point.observableVolume = observableVolume;
            point.observableFixed = observableFixed;
          }
          auto result = emitFrame(row, inputPositions, inputs.edges, asks, {},
              parameterValues, parameterFields, config.batchOnDevice ? &point : nullptr);
          if (!result) return result.takeError();
          return llvm::Error::success();
        }, !positionReader);
        if (!rows) return rows.takeError();
        return SmallVector<Value>{};
      }
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
      return emitFrame(frame, boundary.getResult(0), edges, asks,
          product ? boundary.getResult(product->positionCotangentResult) : Value{}, arguments.startValues, imported);
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
