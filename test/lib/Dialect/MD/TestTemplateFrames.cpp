// Test owner for the actual outer-build frame snapshot emitter.
#include "mdir/Conversion/TemplateFrameKernels.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mdir/Runtime/TemplatePairsABI.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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
struct TestTemplateFrames : PassWrapper<TestTemplateFrames, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTemplateFrames)
  TestTemplateFrames() = default;
  TestTemplateFrames(const TestTemplateFrames &pass) : PassWrapper(pass) {}
  Option<bool> threeSite{*this, "three-site", llvm::cl::desc("Use three-site water"), llvm::cl::init(false)};
  Option<double> pointOffset{*this, "point-offset",
      llvm::cl::desc("The point offset of the production snapshot (1/2)"), llvm::cl::init(0.0)};
  StringRef getArgument() const final { return "test-md-template-frames"; }
  StringRef getDescription() const final { return "Outlines template reference-epoch snapshots"; }
  void getDependentDialects(DialectRegistry &r) const override {
    r.insert<arith::ArithDialect, func::FuncDialect, gpu::GPUDialect,
        math::MathDialect, memref::MemRefDialect, scf::SCFDialect, vector::VectorDialect>();
  }
  void runOnOperation() final {
    ModuleOp module = getOperation();
    auto function = module.lookupSymbol<func::FuncOp>("snapshot_seed");
    if (!function || !function.isExternal() || function.getNumArguments() != 5) {
      module.emitError("snapshot fixture expects the five-argument declaration");
      return signalPassFailure();
    }
    auto *entry = function.addEntryBlock();
    OpBuilder b = OpBuilder::atBlockEnd(entry);
    Location loc = function.getLoc();
    Value handle = entry->getArgument(0);
    Type wide = b.getI64Type();
    auto indices = MemRefType::get({ShapedType::kDynamic}, b.getI32Type(),
        MemRefLayoutAttrInterface(), b.getI64IntegerAttr(1));
    auto frames = MemRefType::get({ShapedType::kDynamic, 3}, b.getF32Type(),
        MemRefLayoutAttrInterface(), b.getI64IntegerAttr(1));
    auto state = MemRefType::get({ShapedType::kDynamic, 3}, b.getF64Type(),
        MemRefLayoutAttrInterface(), b.getI64IntegerAttr(1));
    auto controls = MemRefType::get({4}, wide, MemRefLayoutAttrInterface(), b.getI64IntegerAttr(1));
    auto integer = [&](int64_t n) -> Value { return arith::ConstantIntOp::create(b, loc, n, 64); };
    auto call = [&](StringRef name, ValueRange args, Type result, bool ciface) -> Value {
      auto type = b.getFunctionType(args.getTypes(), TypeRange{result});
      auto callee = module.lookupSymbol<func::FuncOp>(name);
      if (!callee) {
        OpBuilder symbols(module.getBodyRegion());
        callee = func::FuncOp::create(symbols, loc, name, type);
        callee.setPrivate();
      }
      assert(callee.getFunctionType() == type);
      if (ciface) callee->setAttr("llvm.emit_c_interface", b.getUnitAttr());
      return func::CallOp::create(b, loc, callee, args).getResult(0);
    };
    auto view = [&](int64_t slot) {
      return call("mdrtTemplatePairsFrameState", ValueRange{handle, integer(slot)}, state, true);
    };
    auto size = [&](int64_t slot) -> Value {
      return arith::IndexCastOp::create(b, loc, b.getIndexType(),
          call("mdrtTemplatePairsLayoutValue", ValueRange{handle, integer(slot)}, wide, false));
    };
    TemplateFrameBuffers v;
    v.positions = entry->getArgument(1); v.cell = entry->getArgument(2);
    v.inverse = createInverse(b, loc, v.cell);
    v.members = call("mdrtTemplatePairsMetadata", ValueRange{handle, integer(TEMPLATE_MEMBERS)}, indices, true);
    v.frames = call("mdrtTemplatePairsFrames", ValueRange{handle}, frames, true);
    v.outerReference = view(TEMPLATE_FRAME_OUTER_REFERENCE);
    v.packingCells = view(TEMPLATE_FRAME_PACKING_CELLS);
    v.currentPoints = view(TEMPLATE_FRAME_CURRENT_POINTS);
    v.counters = call("mdrtTemplatePairsCounters", ValueRange{handle}, controls, true);
    v.waters = size(TEMPLATE_LAYOUT_WATERS); v.atoms = size(TEMPLATE_LAYOUT_ATOMS);
    v.hasSite = !threeSite; v.alpha = .75; v.pointOffset = pointOffset;
    emitSnapshotTemplateFrames(b, loc, v, entry->getArgument(3), entry->getArgument(4));
    func::ReturnOp::create(b, loc);
  }
};
}
namespace mdir::test {
void registerTestTemplateFrames() { PassRegistration<TestTemplateFrames>(); }
}
