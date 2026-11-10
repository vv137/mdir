// Test-only workspace owner around the actual combined search emitter.
#include "mdir/Conversion/TemplateSearchKernels.h"
#include "mdir/Conversion/TemplateRefreshKernels.h"
#include "mdir/Conversion/SpatialGroupEmitter.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mdir/Runtime/TemplatePairsABI.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mdir::kernels;
namespace {
struct TestTemplateSearch : PassWrapper<TestTemplateSearch, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTemplateSearch)
  TestTemplateSearch() = default;
  TestTemplateSearch(const TestTemplateSearch &pass) : PassWrapper(pass) {}
  Option<bool> refresh{*this, "refresh", llvm::cl::desc("Use the shared cold refresh and automatic arena growth"),
                        llvm::cl::init(false)};
  Option<bool> forceGeneral{*this, "force-general",
      llvm::cl::desc("Force the retained general image iterator for exact comparisons"),
      llvm::cl::init(false)};
  Option<bool> reusePublished{*this, "reuse-published-groups",
      llvm::cl::desc("Replay the owner's committed descriptors without planning or sorting"),
      llvm::cl::init(false)};
  Option<unsigned> columnsX{*this, "columns-x", llvm::cl::init(3)};
  Option<unsigned> columnsY{*this, "columns-y", llvm::cl::init(2)};
  Option<double> baseReach{*this, "base-reach",
      llvm::cl::desc("Anchor-list base reach of the search fixture"),
      llvm::cl::init(99.0 / 128)};
  Option<double> extent{*this, "extent",
      llvm::cl::desc("Rigid-water anchor extent of the search fixture"),
      llvm::cl::init(18.0 / 128)};
  StringRef getArgument() const final { return "test-md-template-search"; }
  StringRef getDescription() const final { return "Outlines the combined periodic template bin search"; }
  void getDependentDialects(DialectRegistry &r) const override {
    r.insert<arith::ArithDialect, cf::ControlFlowDialect, func::FuncDialect, gpu::GPUDialect,
        math::MathDialect, memref::MemRefDialect, scf::SCFDialect, vector::VectorDialect>();
  }
  void runOnOperation() final {
    ModuleOp module = getOperation();
    auto function = module.lookupSymbol<func::FuncOp>("search_seed");
    if (!function || !function.isExternal() ||
        function.getNumArguments() != (reusePublished ? 7u : 6u) ||
        (reusePublished && (refresh || !forceGeneral || !columnsX || !columnsY))) {
      module.emitError("search fixture expects six arguments, or seven with published groups, positive grid, force-general and no refresh");
      return signalPassFailure();
    }
    Block *entry = function.addEntryBlock();
    OpBuilder b = OpBuilder::atBlockEnd(entry);
    Location loc = function.getLoc();
    Value handle = entry->getArgument(0);
    Type wide = b.getI64Type();
    auto device = MemRefType::get({ShapedType::kDynamic}, b.getI32Type(),
        MemRefLayoutAttrInterface(), b.getI64IntegerAttr(1));
    auto controls = MemRefType::get({4}, wide, MemRefLayoutAttrInterface(), b.getI64IntegerAttr(1));
    auto call = [&](StringRef name, ValueRange arguments, Type result, bool ciface) -> Value {
      auto type = b.getFunctionType(arguments.getTypes(), TypeRange{result});
      auto callee = module.lookupSymbol<func::FuncOp>(name);
      if (!callee) {
        OpBuilder symbols(module.getBodyRegion());
        callee = func::FuncOp::create(symbols, loc, name, type);
        callee.setPrivate();
      }
      assert(callee.getFunctionType() == type);
      if (ciface) callee->setAttr("llvm.emit_c_interface", b.getUnitAttr());
      return func::CallOp::create(b, loc, callee, arguments).getResult(0);
    };
    auto integer = [&](int64_t n) -> Value { return arith::ConstantIntOp::create(b, loc, n, 64); };
    auto layout = [&](int64_t slot) -> Value {
      Value value = call("mdrtTemplatePairsLayoutValue", ValueRange{handle, integer(slot)}, wide, false);
      return arith::IndexCastOp::create(b, loc, b.getIndexType(), value);
    };
    auto view = [&](int64_t slot) -> Value {
      return call("mdrtTemplatePairsArenaView", ValueRange{handle, integer(slot)}, device, true);
    };
    auto transfer = [&](Value to, Value from) {
      Type token = gpu::AsyncTokenType::get(b.getContext());
      Value ready = gpu::WaitOp::create(b, loc, token, ValueRange()).getAsyncToken();
      Value done = gpu::MemcpyOp::create(b, loc, token, ValueRange{ready}, to, from).getAsyncToken();
      gpu::WaitOp::create(b, loc, Type(), ValueRange{done});
    };
    TemplateSearchBuffers v;
    v.handle = handle;
    v.forcedImagePath = forceGeneral ? 2 : -1;
    v.spatial.points = entry->getArgument(1);
    auto boxes = MemRefType::get({ShapedType::kDynamic, 3}, b.getF64Type(),
        MemRefLayoutAttrInterface(), b.getI64IntegerAttr(1));
    auto searchBoxes = MemRefType::get({ShapedType::kDynamic, 3}, b.getF32Type(),
        MemRefLayoutAttrInterface(), b.getI64IntegerAttr(1));
    v.groupBoxes = call("mdrtTemplatePairsGroupBoxesF32",
        ValueRange{handle}, searchBoxes, true);
    v.fractionalBoxes = call("mdrtTemplatePairsFrameState",
        ValueRange{handle, integer(TEMPLATE_FRAME_FRACTIONAL_BOXES)}, boxes, true);
    v.spatial.places = view(TEMPLATE_LAYOUT_SPATIAL_PLACES);
    // Retain the fixture's observable inverse-map buffer, while testing
    // the same arena-owned workspace used by the production refresh.
    transfer(v.spatial.places, entry->getArgument(2));
    v.cell = entry->getArgument(3);
    v.spatial.waterActive = entry->getArgument(4);
    v.spatial.atomActive = entry->getArgument(5);
    v.spatial.waters = layout(TEMPLATE_LAYOUT_WATERS);
    v.spatial.atoms = layout(TEMPLATE_LAYOUT_ATOMS);
    v.spatial.columnsX = createIndex(b, loc, columnsX);
    v.spatial.columnsY = createIndex(b, loc, columnsY);
    if (refresh) {
      TemplateRefreshInputs input{handle, v.spatial.points, v.cell,
          createReal(b, loc, b.getF64Type(), baseReach.getValue()),
          createReal(b, loc, b.getF64Type(), extent.getValue()),
          v.spatial.columnsX, v.spatial.columnsY,
          v.spatial.waterActive, v.spatial.atomActive};
      input.forcedImagePath = forceGeneral ? 2 : -1;
      input.reaches = arith::ConstantOp::create(b, loc,
          DenseElementsAttr::get(VectorType::get({3}, b.getF64Type()),
              ArrayRef<double>{baseReach.getValue() + 2 * extent.getValue(),
                  baseReach.getValue() + extent.getValue(), baseReach.getValue()}));
      // The fixture supplies the already prepared hybrid references.
      // A real owner refreshes its requested reference epochs here.
      emitRefreshTemplatePairArena(b, loc, module, input,
          [](OpBuilder &, Value, Value) {});
      transfer(entry->getArgument(2), view(TEMPLATE_LAYOUT_SPATIAL_PLACES));
      func::ReturnOp::create(b, loc);
      return;
    }
    v.spatial.scanBlocks = layout(TEMPLATE_LAYOUT_SCAN_BLOCKS);
    v.spatial.order = call("mdrtTemplatePairsOrder", ValueRange{handle}, device, true);
    v.spatial.keys = view(TEMPLATE_LAYOUT_SPATIAL_KEYS);
    v.spatial.zKeys = view(TEMPLATE_LAYOUT_SPATIAL_Z_KEYS);
    v.spatial.items = view(TEMPLATE_LAYOUT_SPATIAL_ITEMS);
    v.spatial.counts = view(TEMPLATE_LAYOUT_SPATIAL_COUNTS);
    v.spatial.starts = view(TEMPLATE_LAYOUT_SPATIAL_STARTS);
    v.spatial.cursors = view(TEMPLATE_LAYOUT_SPATIAL_CURSORS);
    v.spatial.scanScratch = view(TEMPLATE_LAYOUT_BUILD_SCAN_SCRATCH);
    if (reusePublished) {
      v.spatial.sortScratch = call("mdrtTemplatePairsPlanningView",
          ValueRange{handle, integer(TEMPLATE_PLAN_SORT)}, device, true);
      auto groupView = [&](int64_t slot) {
        return call("mdrtTemplatePairsGroupView", ValueRange{handle, integer(slot)}, device, true);
      };
      v.build.groups = {groupView(TEMPLATE_GROUP_STARTS), groupView(TEMPLATE_GROUP_LENGTHS),
          groupView(TEMPLATE_GROUP_COLUMNS), groupView(TEMPLATE_GROUP_AT_START),
          layout(TEMPLATE_LAYOUT_WATER_GROUPS), layout(TEMPLATE_LAYOUT_ATOM_GROUPS),
          layout(TEMPLATE_LAYOUT_ROWS)};
      v.reaches = entry->getArgument(6);
    } else {
      v.build.groups = emitPrepareTemplateGroups(b, loc, module, handle, v.spatial, v.cell);
    }
    v.build.rows = v.build.groups.rows;
    v.build.arena = call("mdrtTemplatePairsBuffer", ValueRange{handle}, device, true);
    v.build.unitCapacity = layout(TEMPLATE_LAYOUT_UNITS);
    v.build.outerRows = view(TEMPLATE_LAYOUT_OUTER_ROWS);
    v.build.outerEntries = view(TEMPLATE_LAYOUT_OUTER_ENTRIES);
    v.build.stageUnits = view(TEMPLATE_LAYOUT_STAGE_UNITS);
    v.build.stageEntries = view(TEMPLATE_LAYOUT_STAGE_ENTRIES);
    v.build.counts = view(TEMPLATE_LAYOUT_BUILD_COUNTS);
    v.build.workCounts = view(TEMPLATE_LAYOUT_BUILD_WORK_COUNTS);
    v.build.entryOffsets = view(TEMPLATE_LAYOUT_BUILD_ENTRY_OFFSETS);
    v.build.workOffsets = view(TEMPLATE_LAYOUT_BUILD_WORK_OFFSETS);
    v.build.scanScratch = v.spatial.scanScratch;
    v.build.innerRows = view(TEMPLATE_LAYOUT_INNER_ROWS);
    v.build.outerUnits = view(TEMPLATE_LAYOUT_OUTER_UNITS);
    v.build.innerUnits = view(TEMPLATE_LAYOUT_INNER_UNITS);
    v.build.innerEntries = view(TEMPLATE_LAYOUT_INNER_ENTRIES);
    v.build.waters = v.spatial.waters;
    v.build.atoms = v.spatial.atoms;
    v.build.scanBlocks = v.spatial.scanBlocks;
    v.counters = call("mdrtTemplatePairsCounters", ValueRange{handle}, controls, true);
    v.excludedRows = call("mdrtTemplatePairsMetadata",
        ValueRange{handle, integer(TEMPLATE_SS_ROWS)}, device, true);
    v.excludedNeighbors = call("mdrtTemplatePairsMetadata",
        ValueRange{handle, integer(TEMPLATE_SS_NEIGHBORS)}, device, true);
    v.baseReach = createReal(b, loc, b.getF64Type(), baseReach.getValue());
    v.extent = createReal(b, loc, b.getF64Type(), extent.getValue());
    emitSearchTemplatePairs(b, loc, module, v);
    if (reusePublished) {
      auto hostType = MemRefType::get({4}, wide);
      Value host = memref::AllocOp::create(b, loc, hostType);
      transfer(host, v.counters);
      Value flags = memref::LoadOp::create(b, loc, host,
          ValueRange{createIndex(b, loc, TEMPLATE_COUNTER_FLAGS)});
      Value clean = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
          flags, integer(0));
      cf::AssertOp::create(b, loc, clean, "published replay search flags must be zero");
      memref::DeallocOp::create(b, loc, host);
      emitCompactTemplatePairs(b, loc, v.build);
    }
    transfer(entry->getArgument(2), v.spatial.places);
    func::ReturnOp::create(b, loc);
  }
};
}
namespace {
struct TestSpatialGroupPlan
    : PassWrapper<TestSpatialGroupPlan, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestSpatialGroupPlan)
  StringRef getArgument() const final { return "test-md-spatial-group-plan"; }
  StringRef getDescription() const final { return "Outlines the shared column/gap/span planner"; }
  void getDependentDialects(DialectRegistry &r) const override {
    r.insert<arith::ArithDialect, cf::ControlFlowDialect, func::FuncDialect,
        gpu::GPUDialect, memref::MemRefDialect, scf::SCFDialect>();
  }
  void runOnOperation() final {
    auto function = getOperation().lookupSymbol<func::FuncOp>("group_plan_probe");
    if (!function || !function.isExternal() || function.getNumArguments() != 17 ||
        function.getNumResults() != 3) {
      getOperation().emitError("group-plan fixture expects seventeen arguments and three index results");
      return signalPassFailure();
    }
    Block *entry = function.addEntryBlock();
    OpBuilder b = OpBuilder::atBlockEnd(entry);
    Location loc = function.getLoc();
    SpatialGroupPlanBuffers v;
    Value *fields[] = {&v.order, &v.columns, &v.zKeys, &v.flags, &v.segments,
        &v.starts, &v.lengths, &v.counts, &v.offsets, &v.control, &v.scanScratch,
        &v.size, &v.firstClassSize, &v.gap[0], &v.gap[1], &v.span[0], &v.span[1]};
    for (auto [i, value] : llvm::enumerate(fields)) *value = entry->getArgument(i);
    v.scanBlocks = arith::MaxUIOp::create(b, loc, createIndex(b, loc, 1),
        arith::DivUIOp::create(b, loc,
            arith::AddIOp::create(b, loc, v.size, createIndex(b, loc, 1023)),
            createIndex(b, loc, 1024)));
    auto plan = emitSpatialGroupPlan(b, loc, v);
    func::ReturnOp::create(b, loc, ValueRange{plan.groups[0], plan.groups[1], plan.passes});
  }
};
}
namespace mdir::test {
void registerTestTemplateSearch() {
  PassRegistration<TestTemplateSearch>();
  PassRegistration<TestSpatialGroupPlan>();
}
}
