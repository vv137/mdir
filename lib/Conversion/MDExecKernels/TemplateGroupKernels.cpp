#include "mdir/Conversion/TemplateGroupKernels.h"
#include "mdir/Conversion/SpatialGroupEmitter.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mdir/Runtime/TemplatePairsABI.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace mlir;
using namespace mdir::kernels;

TemplateGroupBuffers mdir::kernels::emitPrepareTemplateGroups(OpBuilder &b,
    Location loc, ModuleOp module, Value handle, TemplateSpatialBuffers &v, Value cell) {
  auto ix = [&](OpBuilder &at, int64_t value) { return createIndex(at, loc, value); };
  auto integer = [&](int64_t value) -> Value {
    return arith::ConstantIntOp::create(b, loc, value, 64);
  };
  auto word = [&](OpBuilder &at, Value value) -> Value {
    return arith::IndexCastOp::create(at, loc, at.getI32Type(), value);
  };
  auto index = [&](OpBuilder &at, Value value) -> Value {
    return arith::IndexCastUIOp::create(at, loc, at.getIndexType(), value);
  };
  auto wide = [&](Value value) -> Value {
    if (value.getType().isIndex())
      return arith::IndexCastOp::create(b, loc, b.getI64Type(), value);
    if (value.getType().isInteger(1))
      return arith::ExtUIOp::create(b, loc, b.getI64Type(), value);
    return value;
  };
  auto device = MemRefType::get({ShapedType::kDynamic}, b.getI32Type(),
      MemRefLayoutAttrInterface(), b.getI64IntegerAttr(1));
  auto call = [&](StringRef name, ValueRange arguments, Type result, bool ciface) -> Value {
    SmallVector<Type, 1> results;
    if (result) results.push_back(result);
    auto type = b.getFunctionType(arguments.getTypes(), results);
    auto callee = module.lookupSymbol<func::FuncOp>(name);
    if (!callee) {
      OpBuilder symbols(module.getBodyRegion());
      callee = func::FuncOp::create(symbols, loc, name, type);
      callee.setPrivate();
    }
    assert(callee.getFunctionType() == type && "template group runtime type mismatch");
    if (ciface) callee->setAttr("llvm.emit_c_interface", b.getUnitAttr());
    auto operation = func::CallOp::create(b, loc, callee, arguments);
    return result ? operation.getResult(0) : Value{};
  };
  auto layout = [&](int64_t slot) -> Value {
    return index(b, call("mdrtTemplatePairsLayoutValue",
        ValueRange{handle, integer(slot)}, b.getI64Type(), false));
  };
  auto view = [&](StringRef name, int64_t slot) -> Value {
    return call(name, ValueRange{handle, integer(slot)}, device, true);
  };
  auto trace = [&](int64_t stage) {
    call("mdrtTemplatePairsTraceStage", ValueRange{integer(stage)}, Type{}, false);
  };
  Value chosen = call("mdrtTemplatePairsChooseGrid",
      ValueRange{handle, wide(v.columnsX), wide(v.columnsY), wide(v.waterActive), wide(v.atomActive)},
      b.getI64Type(), false);
  v.columnsX = index(b, chosen);
  v.columnsY = layout(TEMPLATE_LAYOUT_COLUMNS_Y);
  v.sortScratch = view("mdrtTemplatePairsPlanningView", TEMPLATE_PLAN_SORT);
  trace(1);
  emitTemplateSpatialOrder(b, loc, v);
  trace(0);

  trace(2);
  SpatialGroupPlanBuffers plan;
  plan.order = v.items; plan.columns = v.keys; plan.zKeys = v.zKeys;
  plan.size = arith::AddIOp::create(b, loc, v.waters, v.atoms);
  plan.firstClassSize = v.waters;
  plan.flags = view("mdrtTemplatePairsPlanningView", TEMPLATE_PLAN_FLAGS);
  plan.segments = view("mdrtTemplatePairsPlanningView", TEMPLATE_PLAN_SEGMENTS);
  plan.starts = view("mdrtTemplatePairsPlanningView", TEMPLATE_PLAN_STARTS);
  plan.lengths = view("mdrtTemplatePairsPlanningView", TEMPLATE_PLAN_LENGTHS);
  plan.counts = view("mdrtTemplatePairsPlanningView", TEMPLATE_PLAN_COUNTS);
  plan.offsets = view("mdrtTemplatePairsPlanningView", TEMPLATE_PLAN_OFFSETS);
  plan.control = view("mdrtTemplatePairsPlanningView", TEMPLATE_PLAN_CONTROL);
  plan.scanScratch = view("mdrtTemplatePairsPlanningView", TEMPLATE_PLAN_SCAN);
  plan.scanBlocks = arith::MaxUIOp::create(b, loc, ix(b, 1),
      arith::DivUIOp::create(b, loc,
          arith::AddIOp::create(b, loc, plan.size, ix(b, 1023)), ix(b, 1024)));
  Value widths = isTriclinic(cell) ? emitFaceWidths(b, loc, cell) : cell;
  Value zWidth = vector::ExtractOp::create(b, loc, widths, 2);
  auto quantized = [&](double length) -> Value {
    auto real = [&](double value) { return createReal(b, loc, b.getF64Type(), value); };
    Value scaled = arith::MulFOp::create(b, loc,
        arith::DivFOp::create(b, loc, real(length), zWidth), real(1 << 21));
    Value clipped = arith::MinimumFOp::create(b, loc, real((1 << 21) - 1),
        arith::MaximumFOp::create(b, loc, real(0), math::FloorOp::create(b, loc, scaled)));
    return arith::FPToUIOp::create(b, loc, b.getI32Type(), clipped);
  };
  // The prototype's water/solute gap and span defaults, in native nm.
  plan.gap = {quantized(0.4), quantized(0.3)};
  plan.span = {quantized(0.9), quantized(0.6)};
  auto result = emitSpatialGroupPlan(b, loc, plan);

  TemplateGroupBuffers groups;
  groups.starts = view("mdrtTemplatePairsGroupView", TEMPLATE_GROUP_STARTS);
  groups.lengths = view("mdrtTemplatePairsGroupView", TEMPLATE_GROUP_LENGTHS);
  groups.columns = view("mdrtTemplatePairsGroupView", TEMPLATE_GROUP_COLUMNS);
  groups.atStart = view("mdrtTemplatePairsGroupView", TEMPLATE_GROUP_AT_START);
  Value one = ix(b, 1), block = ix(b, 256);
  auto launch = [&](auto body) {
    Value grid = arith::MaxUIOp::create(b, loc, one,
        arith::DivUIOp::create(b, loc,
            arith::AddIOp::create(b, loc, plan.size, ix(b, 255)), block));
    auto kernel = gpu::LaunchOp::create(b, loc, grid, one, one, block, one, one);
    OpBuilder at = OpBuilder::atBlockEnd(&kernel.getBody().front());
    Value item = arith::AddIOp::create(at, loc,
        arith::MulIOp::create(at, loc, kernel.getBlockIds().x, block), kernel.getThreadIds().x);
    Value inside = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ult, item, plan.size);
    scf::IfOp::create(at, loc, inside, [&](OpBuilder &in, Location) {
      Value first = arith::CmpIOp::create(in, loc, arith::CmpIPredicate::ult, item, v.waters);
      Value active = arith::SelectOp::create(in, loc, first, v.waterActive, v.atomActive);
      scf::IfOp::create(in, loc, active, [&](OpBuilder &publish, Location) {
        body(publish, item, first);
        scf::YieldOp::create(publish, loc);
      });
      scf::YieldOp::create(in, loc);
    });
    gpu::TerminatorOp::create(at, loc);
  };
  launch([&](OpBuilder &at, Value item, Value) {
    Value zero = word(at, ix(at, 0));
    for (Value field : {groups.starts, groups.lengths, groups.columns, groups.atStart})
      memref::StoreOp::create(at, loc, zero, field, ValueRange{item});
  });
  launch([&](OpBuilder &at, Value item, Value first) {
    auto loadIndex = [&](Value array, Value position) -> Value {
      return index(at, memref::LoadOp::create(at, loc, array, ValueRange{position}));
    };
    Value segment = loadIndex(plan.segments, item);
    Value start = loadIndex(plan.starts, segment);
    Value length = loadIndex(plan.lengths, segment);
    Value local = arith::SubIOp::create(at, loc, item, start);
    Value begins = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::eq,
        arith::RemUIOp::create(at, loc, local, ix(at, 16)), ix(at, 0));
    scf::IfOp::create(at, loc, begins, [&](OpBuilder &store, Location) {
      Value base = index(store,
          memref::LoadOp::create(store, loc, plan.offsets, ValueRange{segment}));
      Value ordinal = arith::AddIOp::create(store, loc, base,
          arith::DivUIOp::create(store, loc, local, ix(store, 16)));
      Value group = arith::SelectOp::create(store, loc, first, ordinal,
          arith::AddIOp::create(store, loc, v.waters,
              arith::SubIOp::create(store, loc, ordinal, result.groups[0])));
      Value count = arith::MinUIOp::create(store, loc, ix(store, 16),
          arith::SubIOp::create(store, loc, length, local));
      Value unit = index(store,
          memref::LoadOp::create(store, loc, v.items, ValueRange{item}));
      Value column = memref::LoadOp::create(store, loc, v.keys, ValueRange{unit});
      memref::StoreOp::create(store, loc, word(store, item), groups.starts, ValueRange{group});
      memref::StoreOp::create(store, loc, word(store, count), groups.lengths, ValueRange{group});
      memref::StoreOp::create(store, loc, column, groups.columns, ValueRange{group});
      memref::StoreOp::create(store, loc, word(store, count), groups.atStart, ValueRange{item});
      scf::YieldOp::create(store, loc);
    });
  });
  Value rows = call("mdrtTemplatePairsCommitGroups",
      ValueRange{handle, wide(result.groups[0]), wide(result.groups[1]),
          wide(v.waterActive), wide(v.atomActive)}, b.getI64Type(), false);
  groups.rows = index(b, rows);
  groups.waterGroups = layout(TEMPLATE_LAYOUT_WATER_GROUPS);
  groups.atomGroups = layout(TEMPLATE_LAYOUT_ATOM_GROUPS);
  // Row growth preserves this plan's initialized spatial data but moves
  // its physical arena. All later kernels must use the reacquired views.
  v.keys = view("mdrtTemplatePairsArenaView", TEMPLATE_LAYOUT_SPATIAL_KEYS);
  v.zKeys = view("mdrtTemplatePairsArenaView", TEMPLATE_LAYOUT_SPATIAL_Z_KEYS);
  v.items = view("mdrtTemplatePairsArenaView", TEMPLATE_LAYOUT_SPATIAL_ITEMS);
  v.places = view("mdrtTemplatePairsArenaView", TEMPLATE_LAYOUT_SPATIAL_PLACES);
  v.counts = view("mdrtTemplatePairsArenaView", TEMPLATE_LAYOUT_SPATIAL_COUNTS);
  v.starts = view("mdrtTemplatePairsArenaView", TEMPLATE_LAYOUT_SPATIAL_STARTS);
  v.cursors = view("mdrtTemplatePairsArenaView", TEMPLATE_LAYOUT_SPATIAL_CURSORS);
  v.scanScratch = view("mdrtTemplatePairsArenaView", TEMPLATE_LAYOUT_BUILD_SCAN_SCRATCH);
  v.scanBlocks = layout(TEMPLATE_LAYOUT_SCAN_BLOCKS);
  trace(0);
  return groups;
}
