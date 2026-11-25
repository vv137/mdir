#include "TowersSpecialization.h"
#include "TowersCell.h"
#include "mdir/Compiler/Compile.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mdir/Conversion/TowersForce.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Driver/RigidUnits.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/RegionUtils.h"
#include <cmath>

using namespace mlir;
using namespace mdir;
using namespace mdir::driver;

namespace {
struct PairModule {
  MLIRContext context{compiler::getRegistry(),
                      MLIRContext::Threading::DISABLED};
  OwningOpRef<ModuleOp> module;
  Program program;
};
bool lower(ModuleOp module, StringRef pipeline) {
  PassManager pm(module.getContext(), ModuleOp::getOperationName(),
                 PassManager::Nesting::Implicit);
  return succeeded(parsePassPipeline(pipeline, pm)) &&
         succeeded(pm.run(module));
}
// The production schedule fuses these operations into its particle loop.
void emitTrialInputs(ModuleOp module, kernels::TowersLayout g,
                     md_exec::PairForOp term) {
  auto device = module.lookupSymbol<gpu::GPUModuleOp>("towers");
  OpBuilder b = OpBuilder::atBlockEnd(device.getBody());
  auto loc = term.getLoc();
  auto narrow = b.getI32Type(), wide = b.getI64Type();
  auto dbl = b.getF64Type(), real = b.getF32Type();
  auto mem = [](ArrayRef<int64_t> dims, Type t) {
    return MemRefType::get(dims, t);
  };
  bool cartesian = !g.triclinic && g.cells() <= g.atoms;
  int edges = g.triclinic ? 6 : 3;
  SmallVector<Type> args{mem({g.atoms, 3}, dbl),
                         mem({edges}, dbl),
                         mem({3, g.atoms}, dbl),
                         mem({g.boxSize()}, dbl),
                         mem({3, g.rawColumns()}, dbl),
                         mem({1}, narrow),
                         mem({g.atoms}, real),
                         mem({g.atoms}, narrow),
                         mem({g.countWords()}, narrow),
                         mem({g.keyWords()}, narrow),
                         mem({g.atoms}, VectorType::get({4}, real)),
                         mem({g.atoms, 3}, wide),
                         mem({2}, narrow),
                         mem({1}, wide),
                         mem({1}, narrow)};
  llvm::append_range(args, term.getIns().getTypes());
  kernels::emitTowersEntry(
      b, loc, false, "towers_trial_inputs", args, g.atoms,
      [&](Value i, ValueRange a) {
        auto zero = kernels::createIndex(b, loc, 0);
        auto cell = vector::LoadOp::create(
            b, loc, VectorType::get({edges}, dbl), a[1], ValueRange{zero});
        kernels::TowersMovingRow row{g,     cartesian, a[0],  cell,  a[2], a[3],
                                     a[4],  a[5],      a[6],  a[7],  a[8], a[9],
                                     a[10], a[11],     a[12], a[13], a[14]};
        row.emit(b, loc, i, kernels::towersMovingBinGeometry(b, loc, g, cell),
                 a.drop_front(15));
      });
}
} // namespace

llvm::Expected<mdir::tool::TowersEmitter>
mdir::tool::prepareTowersSpecialization(const Control &control,
                                        const System &system, int reach,
                                        int capacity, int device) {
  // Construct the direct term through the normal force-field front end.
  // Its traversal is replaced after precision specialization.
  Control source = control;
  // Setup sees the input atom order. Its private rebuild packs the atoms.
  source.reorder = false;
  source.neighborStructure = NeighborStructure::Matrix;
  if (auto error = resolveRigidUnits(source, system))
    return std::move(error);
  auto program = buildProgram(source, system);
  if (!program)
    return program.takeError();
  auto state = std::make_shared<PairModule>();
  auto &context = state->context;
  compiler::shareThreadPool(context);
  context.loadAllAvailableDialects();
  std::string diagnostic;
  ScopedDiagnosticHandler handler(
      &context, [&](Diagnostic &d) { diagnostic += d.str() + "\n"; });
  auto fail = [&](StringRef why) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "towers_shape: %s: %s", why.str().c_str(),
                                   diagnostic.c_str());
  };
  auto module = parseSourceString<ModuleOp>(program->module, &context);
  if (!module ||
      !lower(
          *module,
          compiler::getPairRegionPipeline(source, *program) +
              "md-exec-assign-storage{memory=device tables=f32},canonicalize"))
    return fail("could not specialize direct pair region");
  md_exec::PairForOp term;
  int virialYield = -1;
  module->walk([&](md_exec::PairForOp op) {
    if (op.getOuts().size() != 1 ||
        op.getExchange(0) != md::Exchange::Antisymmetric)
      return WalkResult::advance();
    if (control.barostat) {
      for (auto [j, t] : llvm::enumerate(op.getResultTypes())) {
        auto v = dyn_cast<VectorType>(t);
        if (v && (v.getNumElements() == 3 || v.getNumElements() == 9)) {
          term = op;
          virialYield = j + 1;
          return WalkResult::interrupt();
        }
      }
    } else if (op.getReduce().empty()) {
      term = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (!term)
    return TowersEmitter{};
  compiler::useCudaToolkit();
  auto options = compiler::getGpuOptions(source, device);
  if (!options)
    return options.takeError();
  double rc = control.cutoffDistance * units::length;
  TowersCell geometry(system);
  kernels::TowersLayout layout{
      int(system.getNumParticles()),
      int(geometry.bins(0, reach, rc, control.barostat)),
      int(geometry.bins(1, reach, rc, control.barostat)),
      int(geometry.bins(2, reach, rc, control.barostat)),
      reach,
      capacity,
      false,
      true,
      geometry.triclinic,
      true};
  OpBuilder builder(term);
  int exclusions = system.topology ? system.topology->exclusions.size() : 0;
  // Bare-pointer kernel arguments need the static particle extent.
  // Keep the original fields for host input binding.
  SmallVector<Value> fields(term.getIns());
  auto specialize = [&](MutableOperandRange operands) {
    SmallVector<Value> fixed;
    for (OpOperand &operand : operands) {
      Value value = operand.get();
      auto type = cast<MemRefType>(value.getType());
      SmallVector<int64_t> dims(type.getShape());
      dims[0] = layout.atoms;
      auto sized = MemRefType::get(dims, type.getElementType(),
                                   type.getLayout(), type.getMemorySpace());
      fixed.push_back(type == sized ? value
                                    : memref::CastOp::create(
                                          builder, term.getLoc(), sized, value)
                                          .getResult());
    }
    operands.assign(fixed);
  };
  specialize(term.getInsMutable());
  specialize(term.getOutsMutable());
  md_exec::PairForOp correction;
  if (exclusions) {
    md_exec::TupleForOp excluded;
    for (auto tuple : term->getBlock()->getOps<md_exec::TupleForOp>()) {
      auto *inc = tuple.getIncidence().getDefiningOp();
      auto name =
          inc ? inc->getAttrOfType<FlatSymbolRefAttr>("md_exec.tuple_set")
              : FlatSymbolRefAttr();
      if (name && name.getValue() == "excluded" && tuple.getReduce().empty()) {
        if (excluded)
          return fail("multiple physical corrections for the excluded set");
        excluded = tuple;
      }
    }
    correction = cast<md_exec::PairForOp>(builder.clone(*term));
    Block &body = correction.getKernel().front();
    for (auto &op : llvm::make_early_inc_range(llvm::reverse(body)))
      op.erase();
    builder.setInsertionPointToEnd(&body);
    Value force;
    if (excluded) {
      auto coords = excluded.getCoordinates();
      if (coords.size() != 1 ||
          coords.front().members != ArrayRef<int64_t>({0, 1}) ||
          excluded.getArity() != 2 || !excluded.getParameters().empty() ||
          excluded.getOuts().size() != 1)
        return fail("unsupported physical exclusion correction");
      Block &old = excluded.getKernel().front();
      IRMapping map;
      map.map(old.getArgument(0), body.getArgument(1));
      for (auto [i, field] : llvm::enumerate(excluded.getIns())) {
        auto it = llvm::find(fields, field);
        if (it == fields.end())
          return fail("exclusion field is absent from the direct pair");
        auto j = it - fields.begin();
        for (int a : {0, 1})
          map.map(old.getArgument(1 + 2 * i + a),
                  body.getArgument(2 + 2 * j + a));
      }
      for (auto &op : old.without_terminator())
        builder.clone(op, map);
      force = map.lookup(old.getTerminator()->getOperand(0));
    } else {
      auto vector = body.getArgument(1).getType();
      force = arith::ConstantOp::create(builder, term.getLoc(), vector,
                                        builder.getZeroAttr(vector));
    }
    md_exec::YieldOp::create(builder, term.getLoc(), ValueRange{force});
  }
  // Prepare both bodies before collecting their bare-pointer kernel inputs.
  // Insert shared casts before the correction (the earlier of the two ops).
  builder.setInsertionPoint(correction ? correction : term);
  DenseMap<Value, Value> tables;
  for (auto region : {term, correction}) {
    if (!region)
      continue;
    auto prepared = region.walk([&](md::LookupOp lookup) {
      Value source = lookup.getTable();
      auto &table = tables[source];
      if (!table) {
        auto type = getTowersTableType(*program, source);
        if (!type)
          return WalkResult::interrupt();
        table = source;
        if (type != source.getType())
          table = memref::CastOp::create(builder, lookup.getLoc(), type, source);
      }
      lookup.getTableMutable().assign(table);
      return WalkResult::advance();
    });
    if (prepared.wasInterrupted())
      return fail("unsupported pair parameter table");
    kernels::lowerLookups(region);
    // Immutable radial tables are inputs shared by every lane.
    region.walk([&](memref::GetGlobalOp global) {
      global->moveBefore(builder.getInsertionBlock(), builder.getInsertionPoint());
    });
  }
  state->module = std::move(module);
  state->program = std::move(*program);
  return TowersEmitter(
      [state, term, correction, fields, virialYield, layout, exclusions,
       typeCount = system.topology ? system.topology->getNumTypes() : 0,
       options = std::move(*options)](
          const TowersShape &shape,
          TowersBuffers *trial) mutable -> llvm::Expected<std::string> {
        std::string diagnostic;
        ScopedDiagnosticHandler handler(&state->context, [&](Diagnostic &d) {
          diagnostic += d.str() + "\n";
        });
        if (trial) {
          llvm::SetVector<Value> captures;
          getUsedValuesDefinedAbove(term->getRegion(0), term->getRegion(0),
                                    captures);
          if (correction)
            getUsedValuesDefinedAbove(correction->getRegion(0),
                                      correction->getRegion(0), captures);
          auto buffers = bindTowersTrialInputs(state->program, fields,
                                               captures.getArrayRef());
          if (!buffers)
            return buffers.takeError();
          *trial = std::move(*buffers);
        }
        OwningOpRef<ModuleOp> image(ModuleOp::create(term.getLoc()));
        // The 4n force specialization uses a two-block compiler launch bound.
        // A higher control pin asks whether that compiled entry can reside
        // there.
        kernels::TowersForceShape launch{
            int(shape.panel),
            int(shape.stripe),
            int(shape.threads),
            int(shape.records),
            int(std::min<int64_t>(2, shape.blocksPerSM)),
            false,
            true,
            exclusions,
            int(shape.split)};
        launch.movingDispatch = true;
        launch.emitRetryCounter = true;
        launch.indexedExclusions = exclusions != 0;
        launch.typeCount = typeCount;
        launch.virialYield = virialYield;
        if (!layout.triclinic && layout.cells() <= layout.atoms)
          launch.cartesianPositions = MemRefType::get(
              {layout.atoms, 3}, Float64Type::get(&state->context));
        layout.widePack = shape.pack == "WIDE";
        if (failed(kernels::emitTowersBuild(*image, layout)))
          return llvm::createStringError(
              "towers_shape: could not compile rebuild: %s",
              diagnostic.c_str());
        emitTrialInputs(*image, layout, term);
        if (failed(kernels::emitTowersForce(*image, layout, term, launch, false,
                                            correction)) ||
            !lower(*image, "canonicalize,cse,mdir-gpu-lower-to-nvvm{" +
                               options +
                               " kernel-bare-ptr-calling-convention=true}"))
          return llvm::createStringError(
              "towers_shape: could not compile direct pair specialization: %s",
              diagnostic.c_str());
        auto binary = image->lookupSymbol<gpu::BinaryOp>("towers");
        if (!binary || binary.getObjects().empty())
          return llvm::createStringError(
              "towers_shape: missing compiled tower entry");
        return cast<gpu::ObjectAttr>(binary.getObjects()[0]).getObject().str();
      });
}
