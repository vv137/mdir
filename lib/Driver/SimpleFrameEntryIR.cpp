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
SmallVector<Value> potentialFields(const EntryFieldsIR &fields) {
  SmallVector<Value> values;
  llvm::append_range(values, fields.fields);
  llvm::append_range(values, fields.tables);
  for (const auto &tuple : fields.tuples) {
    values.push_back(tuple.members);
    llvm::append_range(values, tuple.fields);
  }
  return values;
}
EntryFieldsIR reorderedFrame(OpBuilder &builder, const Program &program,
    EntryFieldsIR fields, Value cell) {
  if (!program.reorders) return fields;
  SmallVector<ReorderFieldIR> parameters;
  for (auto [index, field] : llvm::enumerate(program.fields)) {
    std::optional<unsigned> source;
    if (field.name == "pme_listed" || field.name == "ljpme_listed") {
      StringRef name = field.name == "pme_listed" ? "q" : "ljpme_c";
      auto found = llvm::find_if(program.fields,
          [&](const auto &candidate) { return candidate.name == name; });
      unsigned slot = std::distance(program.fields.begin(), found);
      assert(slot < index && "derived list must follow its source field");
      source = slot;
    }
    parameters.push_back({source ? Value{} : fields.fields[index], source});
  }
  SmallVector<Value> relations;
  ReorderIRConfig config{program.orderWidth, {}};
  for (auto [index, tuple] : llvm::enumerate(program.tupleSets)) {
    relations.push_back(fields.tuples[index].members);
    if (tuple.name == "anchors") config.anchorRelation = index;
  }
  ReorderIRInputs input{fields.positions, fields.velocities, {}, fields.masses,
      fields.identities, cell, parameters, relations};
  auto result = buildReorder(builder, input, config);
  fields.positions = result.positions;
  fields.velocities = result.velocities;
  fields.masses = result.masses;
  fields.identities = result.ids;
  fields.fields = std::move(result.fields);
  for (auto [index, relation] : llvm::enumerate(result.relations))
    fields.tuples[index].members = relation;
  return fields;
}
void writeFrameForces(OpBuilder &builder, Value forces, Value identities) {
  mdir::mdrt::HostCallOp::create(builder, builder.getUnknownLoc(), TypeRange{},
      FlatSymbolRefAttr::get(builder.getContext(), "mdrtWriteFrameForces"),
      ValueRange{forces, identities}, ValueRange{}, UnitAttr{});
}
}

llvm::Expected<func::FuncOp> mdir::driver::builder::buildSimpleFrameEntry(
    OpBuilder &builder, const Program &program, const SimpleFrameEntryIRConfig &config) {
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
      Value memory = memref::AllocaOp::create(at, loc,
          MemRefType::get({1}, at.getI64Type()));
      memref::StoreOp::create(at, loc, arguments.start, memory, zero);
      arith::ConstantIntOp::create(at, loc, 1, 64);
    }
    for (int64_t count : config.counts) arith::ConstantIndexOp::create(at, loc, count);
    for (int64_t period : config.periods) arith::ConstantIndexOp::create(at, loc, period);
    auto imported = buildEntryFields(at, program, arguments, false, config.givenForces);
    Value cellMemory = memref::AllocaOp::create(at, loc,
        MemRefType::get({6}, at.getF64Type()));
    Value requests = memref::AllocaOp::create(at, loc,
        MemRefType::get({2}, at.getF64Type()));
    SmallVector<Value> indices;
    for (int64_t index = 0; index != config.cellEdges; ++index)
      indices.push_back(arith::ConstantIndexOp::create(at, loc, index));
    Value end = arith::ConstantIndexOp::create(at, loc, std::numeric_limits<int64_t>::max());
    auto frames = buildEntryLoop(at, "frames", zero, end, one, {},
        [&](OpBuilder &frame, Value, ValueRange) -> llvm::Expected<SmallVector<Value>> {
      auto storage = cast<MemRefType>(arguments.positions.getType());
      auto boundary = func::CallOp::create(frame, loc, "mdrtFrameBoundary",
          TypeRange{storage}, ValueRange{cellMemory, requests});
      SmallVector<Value> edges, asks;
      for (Value index : indices)
        edges.push_back(memref::LoadOp::create(frame, loc, cellMemory, index));
      for (unsigned index = 0; index != 2; ++index)
        asks.push_back(memref::LoadOp::create(frame, loc, requests, indices[index]));
      Value cell = buildCell(frame, edges);
      auto physical = imported;
      physical.positions = buildEntryField(frame, boundary.getResult(0), 3, false);
      auto originalParameters = potentialFields(physical);
      for (const auto &site : config.sites)
        physical.positions = buildPlacedSites(frame,
            {originalParameters[site.relation], physical.positions, cell,
             originalParameters[site.a], originalParameters[site.b]}, site.amber);
      physical = reorderedFrame(frame, program, std::move(physical), cell);
      auto parameters = potentialFields(physical);
      PotentialStateIR state{physical.positions, cell, parameters, time};
      auto evaluate = [&](OpBuilder &at, EnergyRequests requests) {
        auto result = buildEnergyEvaluation(at, state, requests);
        for (const auto &site : config.sites) {
          auto spread = buildSiteSpread(at,
              {parameters[site.relation], physical.positions, cell,
               parameters[site.a], parameters[site.b], result.forces, result.virial},
              site.amber, site.derived, site.potential);
          result.forces = spread.forces;
          result.virial = spread.virial;
        }
        return result;
      };
      auto iteration = buildEntryLoop(frame, "frame evaluation", zero, one, one, {},
          [&](OpBuilder &body, Value, ValueRange) -> llvm::Expected<SmallVector<Value>> {
        auto energy = evaluate(body, {true, true, true});
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
          Value terms = buildTermEnergyBuffer(body, state, config.terms, config.terms.size());
          buildWriteTerms(body, terms);
          auto diagonal = buildVirialDiagonal(body, energy.virial);
          func::CallOp::create(body, loc, "mdrtWriteVirial", TypeRange{},
              ValueRange{diagonal.xx, diagonal.yy, diagonal.zz});
        }
        Value kinetic = buildKineticEnergy(body, physical.velocities, physical.masses);
        Value excess = buildKineticExcess(body, energy.forces, physical.masses, arguments.interval);
        Value correction = buildSum3(body, excess);
        Value trace = buildVirialTrace(body, energy.virial);
        Value pressureKinetic = arith::AddFOp::create(body, loc, kinetic, correction);
        func::CallOp::create(body, loc, "mdrtWriteEnergies", TypeRange{},
            ValueRange{arguments.start, energy.energy, kinetic, correction, trace, pressureKinetic});
        return SmallVector<Value>{};
      });
      if (!iteration) return iteration.takeError();
      (*iteration)->setAttr("mdrt.segment", frame.getUnitAttr());
      if (llvm::Error error = checkFrameEvaluation(*iteration->getOperation()))
        return std::move(error);
      return SmallVector<Value>{};
    });
    if (!frames) return frames.takeError();
    if (!arguments.positions.use_empty() || !arguments.lengthX.use_empty() ||
        !arguments.lengthY.use_empty() || !arguments.lengthZ.use_empty())
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
          "native frame iteration read entry positions or cell values");
    if (config.cellEdges == 6)
      for (auto [metadata, argument] : llvm::zip(program.startValues, arguments.startValues))
        if (StringRef(metadata.name).starts_with("tilt_") && !argument.use_empty())
          return llvm::createStringError(llvm::inconvertibleErrorCode(),
              "native frame iteration read entry tilt values");
    return llvm::Error::success();
  });
}
