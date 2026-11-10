// Test-only workspace owner around the actual combined search emitter.
#include "mdir/Conversion/TemplateSearchKernels.h"
#include "mdir/Conversion/OrderedRowWorkEmitter.h"
#include "mdir/Conversion/TemplateRefreshKernels.h"
#include "mdir/Conversion/SpatialGroupEmitter.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mdir/Runtime/TemplatePairsABI.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
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
  Option<bool> forceUnsplitSW{*this, "force-unsplit-sw",
      llvm::cl::desc("Use the original single-warp SW search for ordered reference"),
      llvm::cl::init(false)};
  StringRef getArgument() const final { return "test-md-template-search"; }
  StringRef getDescription() const final { return "Outlines the combined periodic template bin search"; }
  void getDependentDialects(DialectRegistry &r) const override {
    r.insert<arith::ArithDialect, cf::ControlFlowDialect, func::FuncDialect, gpu::GPUDialect,
        math::MathDialect, NVVM::NVVMDialect, memref::MemRefDialect, scf::SCFDialect, vector::VectorDialect>();
  }
  void runOnOperation() final {
    ModuleOp module = getOperation();
    if (refresh && forceUnsplitSW) {
      module.emitError("force-unsplit-sw is a direct-search reference, not a refresh override");
      return signalPassFailure();
    }
    auto function = module.lookupSymbol<func::FuncOp>("search_seed");
    if (!function || !function.isExternal() || function.getNumArguments() != 6) {
      module.emitError("search fixture expects the six-argument search_seed declaration");
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
    v.forceUnsplitSW = forceUnsplitSW;
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
    v.spatial.columnsX = createIndex(b, loc, 3);
    v.spatial.columnsY = createIndex(b, loc, 2);
    if (refresh) {
      TemplateRefreshInputs input{handle, v.spatial.points, v.cell,
          createReal(b, loc, b.getF64Type(), 99.0 / 128),
          createReal(b, loc, b.getF64Type(), 18.0 / 128),
          v.spatial.columnsX, v.spatial.columnsY,
          v.spatial.waterActive, v.spatial.atomActive};
      input.reaches = arith::ConstantOp::create(b, loc,
          DenseElementsAttr::get(VectorType::get({3}, b.getF64Type()),
              ArrayRef<double>{135.0 / 128, 117.0 / 128, 99.0 / 128}));
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
    v.build.groups = emitPrepareTemplateGroups(b, loc, module, handle, v.spatial, v.cell);
    v.build.rows = v.build.groups.rows;
    v.build.arena = call("mdrtTemplatePairsBuffer", ValueRange{handle}, device, true);
    v.build.unitCapacity = layout(TEMPLATE_LAYOUT_UNITS);
    v.build.outerRows = view(TEMPLATE_LAYOUT_OUTER_ROWS);
    v.build.outerEntries = view(TEMPLATE_LAYOUT_OUTER_ENTRIES);
    v.build.stageUnits = view(TEMPLATE_LAYOUT_STAGE_UNITS);
    v.build.stageEntries = view(TEMPLATE_LAYOUT_STAGE_ENTRIES);
    v.build.counts = view(TEMPLATE_LAYOUT_BUILD_COUNTS);
    v.build.workCounts = view(TEMPLATE_LAYOUT_BUILD_WORK_COUNTS);
    v.counters = call("mdrtTemplatePairsCounters", ValueRange{handle}, controls, true);
    v.excludedRows = call("mdrtTemplatePairsMetadata",
        ValueRange{handle, integer(TEMPLATE_SS_ROWS)}, device, true);
    v.excludedNeighbors = call("mdrtTemplatePairsMetadata",
        ValueRange{handle, integer(TEMPLATE_SS_NEIGHBORS)}, device, true);
    v.baseReach = createReal(b, loc, b.getF64Type(), 99.0 / 128);
    v.extent = createReal(b, loc, b.getF64Type(), 18.0 / 128);
    emitSearchTemplatePairs(b, loc, v);
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

namespace {
/// A direct primitive fixture: known bounded jobs isolate carry/window
/// boundaries from physical geometry. Search fixtures exercise both together.
struct TestOrderedRowWork
    : PassWrapper<TestOrderedRowWork, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestOrderedRowWork)
  StringRef getArgument() const final { return "test-md-ordered-row-work"; }
  StringRef getDescription() const final { return "Outlines bounded ordered row-work boundary probes"; }
  void getDependentDialects(DialectRegistry &r) const override {
    r.insert<arith::ArithDialect, func::FuncDialect, gpu::GPUDialect,
        math::MathDialect, NVVM::NVVMDialect, memref::MemRefDialect, scf::SCFDialect>();
  }
  void runOnOperation() final {
    auto function = getOperation().lookupSymbol<func::FuncOp>("ordered_row_probe");
    if (!function || !function.isExternal() || function.getNumArguments() != 8) {
      getOperation().emitError("ordered row probe expects eight arguments");
      return signalPassFailure();
    }
    Block *entry = function.addEntryBlock();
    OpBuilder b = OpBuilder::atBlockEnd(entry);
    Location loc = function.getLoc();
    auto ix = [&](OpBuilder &a, int64_t n) -> Value { return createIndex(a,loc,n); };
    auto word = [&](OpBuilder &a, int64_t n) -> Value { return arith::ConstantIntOp::create(a,loc,n,32); };
    auto wide = [&](OpBuilder &a, int64_t n) -> Value { return arith::ConstantIntOp::create(a,loc,n,64); };
    auto space = gpu::AddressSpaceAttr::get(b.getContext(),gpu::GPUDialect::getWorkgroupAddressSpace());
    SmallVector<Type> shared{
        MemRefType::get({128},b.getI32Type(),MemRefLayoutAttrInterface(),space),
        MemRefType::get({128,3},b.getI64Type(),MemRefLayoutAttrInterface(),space),
        MemRefType::get({128,16,4},b.getI32Type(),MemRefLayoutAttrInterface(),space),
        MemRefType::get({128,16},b.getI32Type(),MemRefLayoutAttrInterface(),space),
        MemRefType::get({2},b.getI64Type(),MemRefLayoutAttrInterface(),space)};
    Value one=ix(b,1);
    auto launch=gpu::LaunchOp::create(b,loc,one,one,one,ix(b,128),one,one,
        nullptr,nullptr,ValueRange{},shared);
    OpBuilder k=OpBuilder::atBlockEnd(&launch.getBody().front());
    auto args=launch.getWorkgroupAttributionBBArgs();
    Value thread=launch.getThreadIds().x;
    OrderedRowWorkBuffers jobs{args[0],args[1],args[2],args[3],args[4],thread};
    Value lane=arith::RemUIOp::create(k,loc,thread,ix(k,32));
    Value laneWord=arith::IndexCastOp::create(k,loc,k.getI32Type(),lane);
    TemplateStageBuffers stage{entry->getArgument(0),entry->getArgument(1),
        entry->getArgument(2),entry->getArgument(3),entry->getArgument(4),word(k,0),laneWord};
    Value njobs=entry->getArgument(5), wanted=entry->getArgument(6), frameBreak=entry->getArgument(7);
    auto worker=[&](OpBuilder &run,Value group,ValueRange images) {
      Value groupIndex=arith::IndexCastUIOp::create(run,loc,run.getIndexType(),group);
      Value first=arith::SubIOp::create(run,loc,njobs,wanted);
      Value selected=arith::CmpIOp::create(run,loc,arith::CmpIPredicate::uge,groupIndex,first);
      Value leader=arith::CmpIOp::create(run,loc,arith::CmpIPredicate::eq,lane,ix(run,0));
      Value keep=arith::AndIOp::create(run,loc,leader,selected);
      SmallVector<Value,4> fields{group};
      for(Value image:images)fields.push_back(arith::TruncIOp::create(run,loc,run.getI32Type(),image));
      return OrderedRowRecord{keep,fields};
    };
    auto append=[&](OpBuilder &take,TemplateStageState state,Value keep,ValueRange fields) {
      return emitAppendTemplateStage(take,loc,stage,state,keep,fields);
    };
    auto loop=scf::ForOp::create(k,loc,ix(k,0),njobs,ix(k,1),
        ValueRange{wide(k,0),wide(k,-1),ix(k,0)},
        [&](OpBuilder &at,Location,Value job,ValueRange carried) {
      OrderedRowWorkState state{{carried[0],carried[1]},carried[2]};
      Value boundary=arith::CmpIOp::create(at,loc,arith::CmpIPredicate::eq,job,frameBreak);
      auto before=scf::IfOp::create(at,loc,boundary,[&](OpBuilder &flush,Location) {
        auto next=emitFlushOrderedRowWork(flush,loc,jobs,state,worker,append);
        scf::YieldOp::create(flush,loc,ValueRange{next.stage.count,next.stage.chunk,next.queued});
      },[&](OpBuilder &skip,Location) {
        scf::YieldOp::create(skip,loc,ValueRange{state.stage.count,state.stage.chunk,state.queued});
      });
      Value image0=arith::IndexCastOp::create(at,loc,at.getI64Type(),job);
      Value image1=arith::AddIOp::create(at,loc,
          arith::MulIOp::create(at,loc,image0,wide(at,3)),wide(at,7));
      Value image2=arith::SubIOp::create(at,loc,wide(at,-11),image0);
      auto next=emitEnqueueOrderedRowWork(at,loc,jobs,
          {{before.getResult(0),before.getResult(1)},before.getResult(2)},
          arith::IndexCastOp::create(at,loc,at.getI32Type(),job),
          ValueRange{image0,image1,image2},worker,append);
      scf::YieldOp::create(at,loc,ValueRange{next.stage.count,next.stage.chunk,next.queued});
    });
    auto final=emitFlushOrderedRowWork(k,loc,jobs,
        {{loop.getResult(0),loop.getResult(1)},loop.getResult(2)},worker,append);
    Value owner=arith::CmpIOp::create(k,loc,arith::CmpIPredicate::ult,thread,ix(k,32));
    scf::IfOp::create(k,loc,owner,[&](OpBuilder &finish,Location) {
      emitFinishTemplateStage(finish,loc,stage,final.stage);
      scf::YieldOp::create(finish,loc);
    });
    gpu::TerminatorOp::create(k,loc);
    func::ReturnOp::create(b,loc);
  }
};
}

namespace mdir::test {
void registerTestTemplateSearch() {
  PassRegistration<TestTemplateSearch>();
  PassRegistration<TestSpatialGroupPlan>();
  PassRegistration<TestOrderedRowWork>();
}
}
