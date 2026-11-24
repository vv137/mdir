// TOWERS runs on the simulation stream. Host observations transfer only
// completed scalar reductions. Positions and the force certificate stay on
// device.
#include "TowersDispatch.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mdir/Conversion/TowersForce.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/RegionUtils.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;

namespace {
struct Dispatch {
  md_exec::PairForOp op;
  OpBuilder b;
  Location loc;
  TowersLayout g;
  TowersForceShape shape;
  TowersRowLoop rows;
  TowersTransfer transfer;
  unsigned ordinal;
  ModuleOp module;
  func::FuncOp fn;
  gpu::GPUModuleOp image;
  Type narrow, wide, real, dbl;
  Value raw, box, counts, map, geometry, charge, types, keys, local, packed,
      packedTypes, ids, inverse, spans, status, edges, exclusionIndex;
  SmallVector<Value> fields, captures;
  Value idx(int64_t n) { return createIndex(b, loc, n); }
  MemRefType mem(ArrayRef<int64_t> dims, Type t) {
    return MemRefType::get(dims, t);
  }

  // All sites in an entry reuse the same scratch allocations. A segment
  // owns its scratch through its last launch, including on continuation.
  Value buffer(StringRef name, MemRefType type) {
    for (auto a : fn.front().getOps<gpu::AllocOp>())
      if (auto label = a->getAttrOfType<StringAttr>("mdir.towers_buffer"))
        if (label.getValue() == name && a.getMemref().getType() == type)
          return a.getMemref();
    OpBuilder top = OpBuilder::atBlockBegin(&fn.front());
    auto a = gpu::AllocOp::create(top, loc, type, Type(), ValueRange{},
                                  ValueRange{}, ValueRange{});
    a->setAttr("mdir.towers_buffer", top.getStringAttr(name));
    fn.walk([&](func::ReturnOp ret) {
      OpBuilder end(ret);
      gpu::DeallocOp::create(end, loc, Type(), ValueRange{}, a.getMemref());
    });
    return a.getMemref();
  }
  void launch(StringRef name, int64_t items, int block, ValueRange args,
              int shared = 0) {
    auto kernel = image.lookupSymbol<gpu::GPUFuncOp>(name);
    // The CUDA loader opts this specialization into its exact record storage.
    if (shared)
      kernel.setName((name + "_dynamic_" + Twine(shared)).str());
    Value one = idx(1);
    Value dynamic =
        shared ? Value(arith::ConstantIntOp::create(b, loc, shared, 32))
               : Value{};
    bool cooperative = name == "towers_finish";
    Value blocks = idx(cooperative ? towersFinishBlocks(items, block)
                                   : llvm::divideCeil(items, int64_t(block)));
    if (cooperative) {
      StringRef query = "mdrtCooperativeGridSize";
      if (!module.lookupSymbol(query)) {
        OpBuilder top = OpBuilder::atBlockBegin(module.getBody());
        func::FuncOp::create(
            top, loc, query,
            top.getFunctionType({b.getIndexType()}, {b.getIndexType()}))
            .setPrivate();
      }
      blocks = func::CallOp::create(b, loc, query, TypeRange{b.getIndexType()},
                                    ValueRange{blocks})
                   .getResult(0);
    }
    auto call = gpu::LaunchFuncOp::create(
        b, loc, kernel, gpu::KernelDim3{blocks, one, one},
        gpu::KernelDim3{idx(block), one, one}, dynamic, args);
    call.setCooperative(cooperative);
  }
  void fill(Value dest, Type element) {
    auto type = cast<MemRefType>(dest.getType());
    rows(b, loc, idx(type.getDimSize(0)), [&](OpBuilder &k, Value i) {
      Value zero = createZero(k, loc, element);
      if (type.getRank() == 1)
        memref::StoreOp::create(k, loc, zero, dest, ValueRange{i});
      else
        for (int j = 0; j < type.getDimSize(1); ++j)
          memref::StoreOp::create(k, loc, zero, dest,
                                  ValueRange{i, createIndex(k, loc, j)});
    });
  }
  void allocate() {
    int n = g.atoms, c = g.cells(), f = 5 + 2 * g.reach;
    if (!shape.cartesianPositions)
      raw = buffer("coordinates", mem({3, n}, dbl));
    box = buffer("box", mem({g.boxSize()}, dbl));
    geometry = buffer("geometry", mem({3, g.rawColumns()}, dbl));
    counts = buffer("counts", mem({g.countWords()}, narrow));
    map = buffer("map", mem({c, f}, narrow));
    spans = buffer("spans", mem({c, f, 2}, narrow));
    if (g.cells() > g.atoms)
      charge = buffer("charge", mem({n}, real));
    types = buffer("types", mem({n}, narrow));
    keys = buffer("keys", mem({g.keyWords()}, narrow));
    auto v4 = VectorType::get({4}, real);
    local = buffer("local", mem({n}, v4));
    packed = buffer("packed", mem({g.slots()}, v4));
    packedTypes = buffer("packed types", mem({g.slots()}, narrow));
    ids = buffer("atom ids", mem({g.slots()}, narrow));
    inverse = buffer("inverse", mem({g.inverseWords()}, narrow));
    status = buffer("rebuild status", mem({1}, narrow));
  }
  LogicalResult inputs(Value positions) {
    for (Value field : op.getIns()) {
      auto type = cast<MemRefType>(field.getType());
      if (type.getRank() != 1)
        return op.emitError("unsupported TOWERS per-atom field");
      fields.push_back(field);
    }
    if (shape.exclusions) {
      Value members;
      for (auto arg : fn.getArguments())
        if (fn.getArgAttr(arg.getArgNumber(), "mdir.towers_exclusions"))
          members = arg;
      if (!members)
        return op.emitError("missing input-order TOWERS exclusion pairs");
      edges = buffer("excluded", mem({shape.exclusions, 2}, narrow));
      exclusionIndex = buffer("exclusion index", mem({g.atoms, 2}, narrow));
      shape.indexedExclusions = true;
      // Tuple members are sorted by input identity. Physical tuple loops
      // still evaluate their topology corrections once.
      OpBuilder init(edges.getDefiningOp());
      init.setInsertionPointAfter(edges.getDefiningOp());
      if (!edges.getDefiningOp()->hasAttr("mdir.initialized")) {
        transfer(init, loc, edges, members);
        rows(init, loc, createIndex(init, loc, g.atoms),
             [&](OpBuilder &k, Value i) {
               for (int column = 0; column < 2; ++column)
                 memref::StoreOp::create(
                     k, loc, createZero(k, loc, narrow), exclusionIndex,
                     ValueRange{i, createIndex(k, loc, column)});
             });
        rows(init, loc, createIndex(init, loc, shape.exclusions),
             [&](OpBuilder &k, Value i) {
               Value zero = createIndex(k, loc, 0),
                     one = createIndex(k, loc, 1);
               Value first =
                   memref::LoadOp::create(k, loc, edges, ValueRange{i, zero});
               Value second =
                   memref::LoadOp::create(k, loc, edges, ValueRange{i, one});
               Value row =
                   arith::IndexCastOp::create(k, loc, k.getIndexType(), first);
               Value gap = arith::SubIOp::create(k, loc, second, first);
               auto word = [&](int n) {
                 return arith::ConstantIntOp::create(k, loc, n, 32);
               };
               Value near = arith::CmpIOp::create(
                   k, loc, arith::CmpIPredicate::ule, gap, word(32));
               Value shift = arith::AndIOp::create(
                   k, loc, arith::SubIOp::create(k, loc, gap, word(1)),
                   word(31));
               Value bit = arith::ShLIOp::create(k, loc, word(1), shift);
               Value mask = arith::SelectOp::create(k, loc, near, bit, word(0));
               memref::AtomicRMWOp::create(k, loc, arith::AtomicRMWKind::ori,
                                           mask, exclusionIndex,
                                           ValueRange{row, zero});
               memref::AtomicRMWOp::create(k, loc, arith::AtomicRMWKind::maxu,
                                           second, exclusionIndex,
                                           ValueRange{row, one});
             });
        edges.getDefiningOp()->setAttr("mdir.initialized", init.getUnitAttr());
      }
    }
    if (g.cells() <= g.atoms &&
        !counts.getDefiningOp()->hasAttr("mdir.initialized")) {
      OpBuilder init(counts.getDefiningOp());
      init.setInsertionPointAfter(counts.getDefiningOp());
      if (counts.getDefiningOp()->isBeforeInBlock(map.getDefiningOp()))
        init.setInsertionPointAfter(map.getDefiningOp());
      rows(init, loc, createIndex(init, loc, g.cells()),
           [&](OpBuilder &k, Value i) {
             emitTowersInitRow(k, loc, g, i, map, counts);
           });
      counts.getDefiningOp()->setAttr("mdir.initialized", init.getUnitAttr());
    }
    auto cell = op.getCellMutable().get();
    auto binsInfo = towersMovingBinGeometry(b, loc, g, cell);
    if (shape.cartesianPositions)
      raw = positions;
    Value initialWords, initialState, initialFailed, arrivals;
    if (!op.getOuts().empty()) {
      arrivals = buffer("certificate arrivals", mem({1}, narrow));
      initialWords = buffer("force words", mem({g.atoms, 3}, wide));
      initialState = buffer("certificate", mem({2}, narrow));
      initialFailed = buffer("failed conversions", mem({1}, wide));
    }
    TowersMovingRow input{g, bool(shape.cartesianPositions), positions, cell,
                          raw, box, geometry, status, charge, types, counts,
                          keys, local, initialWords, initialState, initialFailed,
                          arrivals};
    rows(b, loc, idx(g.atoms), [&](OpBuilder &k, Value i) {
      input.emit(k, loc, i, binsInfo, fields);
    });
    return success();
  }

  LogicalResult compile(unsigned channel) {
    OwningOpRef<ModuleOp> temp(ModuleOp::create(loc));
    if (failed(emitTowersBuild(*temp, g)) ||
        failed(emitTowersForce(*temp, g, op, shape, false, {})))
      return failure();
    image = temp->lookupSymbol<gpu::GPUModuleOp>("towers");
    image.setSymName(
        ("towers_site_" + Twine(ordinal) + "_" + Twine(channel)).str());
    image->moveBefore(module.getBody(), module.getBody()->end());
    return success();
  }

  LogicalResult run(Value positions);
};
} // namespace

LogicalResult Dispatch::run(Value positions) {
  shape.movingDispatch = true;
  if (!g.triclinic && g.cells() <= g.atoms)
    shape.cartesianPositions = positions.getType();
  allocate();
  if (g.cells() <= g.atoms && !module.lookupSymbol("__nv_fmod")) {
    OpBuilder top = OpBuilder::atBlockBegin(module.getBody());
    LLVM::LLVMFuncOp::create(top, loc, "__nv_fmod",
                             LLVM::LLVMFunctionType::get(dbl, {dbl, dbl}));
  }
  if (failed(inputs(positions)))
    return failure();
  // Radial globals are host constants here. Give kernels a device
  // allocation initialized once, just as for the particle type tables.
  SmallVector<memref::GetGlobalOp> globals;
  op.walk([&](memref::GetGlobalOp get) { globals.push_back(get); });
  for (auto get : globals) {
    auto name = get.getName();
    Value copy = buffer(name, get.getType());
    if (!copy.getDefiningOp()->hasAttr("mdir.initialized")) {
      OpBuilder init(copy.getDefiningOp());
      init.setInsertionPointAfter(copy.getDefiningOp());
      Value source = init.clone(*get)->getResult(0);
      transfer(init, loc, copy, source);
      copy.getDefiningOp()->setAttr("mdir.initialized", init.getUnitAttr());
    }
    // Each cubic radial row occupies one aligned 16-byte word in the
    // device allocation. Load its coefficients together for Horner evaluation.
    auto type = get.getType();
    if (type.getRank() == 2 && type.getDimSize(1) == 4 &&
        type.getElementType().isF32() && type.getLayout().isIdentity())
      for (Operation *use : get->getUsers())
        if (auto load = dyn_cast<vector::LoadOp>(use))
          if (load.getVectorType() == VectorType::get({4}, real) &&
              matchPattern(load.getIndices().back(), m_Zero()))
            load.setAlignmentAttr(b.getI64IntegerAttr(16));
    get.replaceAllUsesWith(copy);
    get.erase();
  }
  lowerLookups(op);
  llvm::SetVector<Value> external;
  getUsedValuesDefinedAbove(op.getKernel(), op.getKernel(), external);
  for (Value v : external)
    if (!matchPattern(v, m_Constant()))
      captures.push_back(v);
  int n = g.atoms;
  auto forceType = mem({n, 3}, wide);
  auto limbs = mem({2}, wide);
  Value state = buffer("certificate", mem({2}, narrow));
  Value failedConversions = buffer("failed conversions", mem({1}, wide));
  Value finishedForce, host;
  auto count = op.getKernel().front().getTerminator()->getNumOperands();
  auto components = [](Type type) -> int64_t {
    if (auto vector = dyn_cast<VectorType>(type))
      return vector.getNumElements();
    return 1;
  };
  int64_t stride = 3;
  for (Type type : op.getResultTypes())
    stride = std::max(stride, components(type));
  if (op.getNumResults()) {
    OpBuilder top = OpBuilder::atBlockBegin(&fn.front());
    host = memref::AllocaOp::create(
        top, loc, mem({int64_t(stride * op.getNumResults())}, dbl));
  }
  Value fusedVirial;
  for (auto [j, type] : llvm::enumerate(op.getResultTypes())) {
    if (op.getOuts().size() == 1 && isa<VectorType>(type) &&
        (components(type) == 3 || components(type) == 9))
      shape.virialYield = 1 + j;
  }
  Value observed;
  if (host)
    observed = buffer("observer", cast<MemRefType>(host.getType()));
  for (unsigned i = 0; i < count; ++i) {
    bool force = i < op.getOuts().size();
    unsigned reduction = i - op.getOuts().size();
    bool energy = !force && !isa<VectorType>(op.getResult(reduction).getType());
    int width = force ? 0 : components(op.getResult(reduction).getType());
    bool tensor = width == 9;
    shape.output = force    ? TowersOutput::Force
                   : energy ? TowersOutput::Energy
                   : tensor ? TowersOutput::TensorVirial
                            : TowersOutput::DiagonalVirial;
    shape.yieldIndex = i;
    if (failed(compile(i)))
      return failure();
    if (i == 0) {
      if (g.cells() > g.atoms)
        launch("towers_setup", g.cells(), 128,
               {map, counts, status, box, geometry});
      Value forward = arith::ConstantIntOp::create(b, loc, 0, 32);
      if (g.cells() > g.atoms)
        launch("towers_insert", n, 128,
               {raw, charge, geometry, counts, keys, local, forward});
      launch("towers_pack", 32LL * g.cells(), 128,
             {counts, keys, local, types, map, packed, packedTypes, ids,
              inverse, spans, status});
    }
    if (!force || i != 0) {
      fill(state, narrow);
      fill(failedConversions, wide);
    }
    Value words = force    ? buffer("force words", forceType)
                  : energy ? buffer("energy limbs", limbs)
                           : buffer("virial", mem({tensor ? 9 : 6}, dbl));
    bool reusedVirial = int(i) == shape.virialYield;
    if (reusedVirial)
      words = fusedVirial;
    else if (!force || i != 0)
      fill(words, force || energy ? wide : dbl);
    double scale = force ? 0x1p18 : energy ? 0x1p32 : 1.0;
    SmallVector<Value> args{spans,
                            packed,
                            ids,
                            geometry,
                            words,
                            state,
                            createReal(b, loc, real, scale),
                            failedConversions};
    if (shape.exclusions)
      args.append({edges, inverse, raw, box, exclusionIndex});
    llvm::append_range(args, fields);
    args.push_back(packedTypes);
    llvm::append_range(args, captures);
    int64_t blocks = towersForceBlocks(g.cells(), shape.split, shape.exclusions,
                                       shape.workers);
    int shared = 8 * shape.records;
    if (force && shape.virialYield >= 0) {
      bool full =
          components(op.getResult(shape.virialYield - 1).getType()) == 9;
      fusedVirial = buffer("fused virial", mem({full ? 9 : 6}, dbl));
      fill(fusedVirial, dbl);
      args.push_back(fusedVirial);
    }
    if (!reusedVirial)
      launch(force    ? (fusedVirial ? "towers_force_virial" : "towers_force")
             : energy ? "towers_energy"
                      : "towers_virial",
             blocks * shape.workers, shape.workers, args, shared);
    Value result;
    if (force) {
      if (fusedVirial)
        args.pop_back();
      Value lo = buffer("wide force low", forceType);
      args[4] = lo;
      Value hi = buffer("wide force high", forceType);
      args.push_back(hi);
      args.push_back(words);
      Value arrivals = buffer("certificate arrivals", mem({1}, narrow));
      if (i != 0)
        fill(arrivals, narrow);
      args.push_back(arrivals);
      if (op.getNumResults())
        finishedForce = buffer("published force", forceType);
      args.push_back(finishedForce ? finishedForce : op.getOuts().front());
      launch("towers_finish", n, shape.workers, args, shared);
      continue;
    } else if (energy) {
      launch("towers_certify_energy", 1, 1, {words, failedConversions, state});
      Value recovered = buffer("energy replay", limbs);
      args[4] = recovered;
      launch("towers_energy_recover_once", shape.workers, shape.workers, args,
             shared);
      launch("towers_check_recovered_energy", 1, 1, {recovered, state});
      result = buffer("published energy", mem({1}, wide));
      launch("towers_publish_energy", 1, 1, {state, words, recovered, result});
    } else {
      launch("towers_virial_check", tensor ? 9 : 6, 32, {words, state});
      result = words;
    }
    // A failed physical replay is fatal. No later launch in this context
    // can consume the stale contents of an unpublished destination.
    rows(b, loc, idx(1), [&](OpBuilder &k, Value zero) {
      Value code = memref::LoadOp::create(k, loc, state, ValueRange{zero});
      Value bad = arith::CmpIOp::create(k, loc, arith::CmpIPredicate::ne, code,
                                        createZero(k, loc, narrow));
      // The current face widths must still cover the cutoff after a box
      // change. Never integrate a force computed with an incomplete stencil.
      if (i == 0) {
        for (int a = 0; a < 3; ++a) {
          Value width = memref::LoadOp::create(
              k, loc, geometry,
              ValueRange{
                  createIndex(k, loc, a),
                  createIndex(k, loc, g.triclinic ? TowersLayout::slab : 1)});
          double minimum =
              op.getCutoff().convertToDouble() / (a == 2 ? g.reach : 1);
          Value shortFace =
              arith::CmpFOp::create(k, loc, arith::CmpFPredicate::ULT, width,
                                    createReal(k, loc, dbl, minimum));
          bad = arith::OrIOp::create(k, loc, bad, shortFace);
        }
      }
      scf::IfOp::create(k, loc, bad, [&](OpBuilder &stop, Location l) {
        LLVM::InlineAsmOp::create(stop, l, TypeRange{}, ValueRange{}, "trap;",
                                  "", true, false, LLVM::TailCallKind::None,
                                  LLVM::AsmDialectAttr(), ArrayAttr());
        scf::YieldOp::create(stop, l);
      });
      if (!force) {
        for (int j = 0; j < width; ++j) {
          Value value = memref::LoadOp::create(
              k, loc, result,
              ValueRange{createIndex(k, loc,
                                     width == 3 ? std::array{0, 3, 5}[j] : j)});
          if (energy) {
            value = arith::SIToFPOp::create(k, loc, dbl, value);
            value = arith::MulFOp::create(k, loc, value,
                                          createReal(k, loc, dbl, 1 / scale));
          }
          memref::StoreOp::create(
              k, loc, value, observed,
              ValueRange{createIndex(k, loc, stride * reduction + j)});
        }
      }
    });
  }
  if (finishedForce) {
    Value dest = op.getOuts().front();
    Type element = cast<MemRefType>(dest.getType()).getElementType();
    rows(b, loc, idx(n), [&](OpBuilder &k, Value atom) {
      for (int a = 0; a < 3; ++a) {
        SmallVector<Value> at{atom, createIndex(k, loc, a)};
        Value word = memref::LoadOp::create(k, loc, finishedForce, at);
        Value value = arith::SIToFPOp::create(k, loc, element, word);
        value = arith::MulFOp::create(k, loc, value,
                                      createReal(k, loc, element, 0x1p-18));
        if (!op.overwrites(0))
          value = arith::AddFOp::create(
              k, loc, value, memref::LoadOp::create(k, loc, dest, at));
        memref::StoreOp::create(k, loc, value, dest, at);
      }
    });
  }
  if (host) {
    transfer(b, loc, host, observed);
    for (auto [i, result] : llvm::enumerate(op.getResults())) {
      SmallVector<Value> parts;
      bool vector = isa<VectorType>(result.getType());
      for (int j = 0; j < components(result.getType()); ++j)
        parts.push_back(memref::LoadOp::create(
            b, loc, host, ValueRange{idx(stride * i + j)}));
      Value value = vector ? Value(vector::FromElementsOp::create(
                                 b, loc, result.getType(), parts))
                           : parts.front();
      result.replaceAllUsesWith(
          arith::AddFOp::create(b, loc, op.getReduce()[i], value));
    }
  }
  return success();
}

LogicalResult mdir::kernels::lowerTowersPair(md_exec::PairForOp pair,
                                             Value positions, bool masked,
                                             unsigned ordinal,
                                             TowersRowLoop rows,
                                             TowersTransfer transfer) {
  auto module = pair->getParentOfType<ModuleOp>();
  module->setAttr("gpu.container_module", UnitAttr::get(pair.getContext()));
  auto dims = module->getAttrOfType<DenseI64ArrayAttr>("mdir.towers_grid");
  auto tile = module->getAttrOfType<DenseI64ArrayAttr>("mdir.towers_launch");
  if (!dims || dims.size() != 7 || !tile || tile.size() != 6)
    return pair.emitError("invalid TOWERS grid or launch dimensions");
  auto grid = dims.asArrayRef();
  auto launch = tile.asArrayRef();
  OpBuilder b(pair);
  TowersLayout g{
      int(grid[0]), int(grid[1]), int(grid[2]),
      int(grid[3]), int(grid[4]), int(grid[5]),
      false,        true,         isTriclinic(pair.getCellMutable().get()),
      true};
  TowersForceShape shape{int(launch[2]),
                         int(launch[3]),
                         int(launch[0]),
                         int(launch[4]),
                         int(std::min<int64_t>(2, launch[1])),
                         false,
                         true,
                         masked ? int(grid[6]) : 0,
                         int(launch[5])};
  if (auto count = module->getAttrOfType<IntegerAttr>("mdir.towers_type_count"))
    shape.typeCount = count.getInt();
  if (pair.getOuts().size() > 1 || !positions ||
      !cast<MemRefType>(positions.getType()).getElementType().isF64())
    return pair.emitError("TOWERS needs the current binary64 positions and at "
                          "most one force destination");
  for (Type result : pair.getResultTypes())
    if (!result.isF64() && result != VectorType::get({3}, b.getF64Type()) &&
        result != VectorType::get({9}, b.getF64Type()))
      return pair.emitError("unsupported TOWERS reduction type ") << result;
  Dispatch dispatch{pair,
                    b,
                    pair.getLoc(),
                    g,
                    shape,
                    rows,
                    transfer,
                    ordinal,
                    module,
                    pair->getParentOfType<func::FuncOp>(),
                    {},
                    b.getI32Type(),
                    b.getI64Type(),
                    b.getF32Type(),
                    b.getF64Type()};
  return dispatch.run(positions);
}

void mdir::kernels::markTowersExclusions(ModuleOp module) {
  module.walk([&](memref::CastOp op) {
    auto layout = dyn_cast<md_exec::SetLayoutAttr>(
        cast<MemRefType>(op.getType()).getLayout());
    auto arg = dyn_cast<BlockArgument>(op.getSource());
    auto fn = op->getParentOfType<func::FuncOp>();
    if (layout && layout.getSet().getValue() == "excluded" && arg && fn &&
        arg.getOwner() == &fn.front())
      fn.setArgAttr(arg.getArgNumber(), "mdir.towers_exclusions",
                    UnitAttr::get(module.getContext()));
  });
}
