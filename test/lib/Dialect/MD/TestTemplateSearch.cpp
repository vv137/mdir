// Test-only workspace owner around the actual combined search emitter.
#include "mdir/Conversion/TemplateSearchKernels.h"
#include "mdir/Conversion/TemplateRefreshKernels.h"
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
  StringRef getArgument() const final { return "test-md-template-search"; }
  StringRef getDescription() const final { return "Outlines the combined periodic template bin search"; }
  void getDependentDialects(DialectRegistry &r) const override {
    r.insert<arith::ArithDialect, cf::ControlFlowDialect, func::FuncDialect, gpu::GPUDialect,
        math::MathDialect, memref::MemRefDialect, scf::SCFDialect, vector::VectorDialect>();
  }
  void runOnOperation() final {
    ModuleOp module = getOperation();
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
    v.spatial.points = entry->getArgument(1);
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
    v.build.rows = layout(TEMPLATE_LAYOUT_ROWS);
    v.build.unitCapacity = layout(TEMPLATE_LAYOUT_UNITS);
    v.build.outerRows = view(TEMPLATE_LAYOUT_OUTER_ROWS);
    v.build.outerEntries = view(TEMPLATE_LAYOUT_OUTER_ENTRIES);
    v.build.stageUnits = view(TEMPLATE_LAYOUT_STAGE_UNITS);
    v.build.stageEntries = view(TEMPLATE_LAYOUT_STAGE_ENTRIES);
    v.build.counts = view(TEMPLATE_LAYOUT_BUILD_COUNTS);
    v.counters = call("mdrtTemplatePairsCounters", ValueRange{handle}, controls, true);
    v.excludedRows = call("mdrtTemplatePairsMetadata",
        ValueRange{handle, integer(TEMPLATE_SS_ROWS)}, device, true);
    v.excludedNeighbors = call("mdrtTemplatePairsMetadata",
        ValueRange{handle, integer(TEMPLATE_SS_NEIGHBORS)}, device, true);
    v.baseReach = createReal(b, loc, b.getF64Type(), 99.0 / 128);
    v.extent = createReal(b, loc, b.getF64Type(), 18.0 / 128);
    emitTemplateSpatialOrder(b, loc, v.spatial);
    emitSearchTemplatePairs(b, loc, v);
    transfer(entry->getArgument(2), v.spatial.places);
    func::ReturnOp::create(b, loc);
  }
};
}
namespace mdir::test {
void registerTestTemplateSearch() { PassRegistration<TestTemplateSearch>(); }
}
