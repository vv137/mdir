// Exercise the production final-SSA packing/validity callback.
#include "mdir/Conversion/TemplateWriterKernels.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
using namespace mlir;
using namespace mdir::kernels;
namespace {
struct TestTemplateWriter : PassWrapper<TestTemplateWriter, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTemplateWriter)
  StringRef getArgument() const final { return "test-md-template-writer"; }
  StringRef getDescription() const final { return "Outlines final-SSA template packing and validity"; }
  void getDependentDialects(DialectRegistry &r) const override {
    r.insert<arith::ArithDialect, func::FuncDialect, gpu::GPUDialect,
        math::MathDialect, memref::MemRefDialect, scf::SCFDialect, vector::VectorDialect>();
  }
  void runOnOperation() final {
    auto f = getOperation().lookupSymbol<func::FuncOp>("writer_frame");
    if (!f || f.getNumArguments() != 12 || !f.isExternal()) {
      getOperation().emitError("expected a twelve-argument writer_frame declaration");
      return signalPassFailure();
    }
    Location loc = f.getLoc();
    OpBuilder b = OpBuilder::atBlockEnd(f.addEntryBlock());
    auto ix = [&](OpBuilder &at, int64_t n) { return createIndex(at, loc, n); };
    TemplateWriterBuffers v;
    v.frames = f.getArgument(1); v.packingCells = f.getArgument(2);
    v.currentPoints = f.getArgument(3); v.outerReference = f.getArgument(4);
    v.innerReference = f.getArgument(5); v.waters = f.getArgument(7);
    v.cell = f.getArgument(8); v.inverse = createInverse(b, loc, v.cell);
    v.built = f.getArgument(9); v.constrained = f.getArgument(10);
    v.hasSite = true; v.alpha = .75; v.extent = .25;
    for (int64_t g = 0; g != 2; ++g) {
      v.outerLimits[g] = vector::ExtractOp::create(b, loc, f.getArgument(11), g);
      v.innerLimits[g] = vector::ExtractOp::create(b, loc, f.getArgument(11), g + 2);
    }
    Value n = memref::DimOp::create(b, loc, v.currentPoints, ix(b, 0));
    Value one = ix(b, 1), block = ix(b, 128);
    auto launch = gpu::LaunchOp::create(b, loc,
        arith::CeilDivUIOp::create(b, loc, n, block), one, one, block, one, one);
    OpBuilder k = OpBuilder::atBlockEnd(&launch.getBody().front());
    Value unit = arith::AddIOp::create(k, loc, launch.getThreadIds().x,
        arith::MulIOp::create(k, loc, launch.getBlockIds().x, block));
    scf::IfOp::create(k, loc,
        arith::CmpIOp::create(k, loc, arith::CmpIPredicate::ult, unit, n),
        [&](OpBuilder &at, Location) {
      Value water = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ult, unit, v.waters);
      auto emit = [&](OpBuilder &in, bool isWater) {
        Value row = isWater ? Value(arith::MulIOp::create(in, loc, unit, ix(in, 3)))
            : Value(arith::AddIOp::create(in, loc, unit,
                arith::MulIOp::create(in, loc, v.waters, ix(in, 2))));
        SmallVector<TemplateParticleResult, 3> results;
        for (int64_t s = 0; s != (isWater ? 3 : 1); ++s) {
          TemplateParticleResult result;
          result.outputs.push_back(loadElement(in, loc, f.getArgument(0),
              arith::AddIOp::create(in, loc, row, ix(in, s))));
          results.push_back(result);
        }
        auto flags = emitUpdateTemplateFrame(in, loc, v,
            arith::IndexCastOp::create(in, loc, in.getI32Type(), unit), results);
        SmallVector<Value> values{flags.waterOuter, flags.atomOuter,
            flags.waterInner, flags.atomInner, flags.extent};
        for (auto [i, flag] : llvm::enumerate(values))
          memref::StoreOp::create(in, loc,
              arith::ExtUIOp::create(in, loc, in.getI32Type(), flag), f.getArgument(6),
              ValueRange{unit, ix(in, i)});
      };
      scf::IfOp::create(at, loc, water, [&](OpBuilder &in, Location) {
        emit(in, true); scf::YieldOp::create(in, loc);
      }, [&](OpBuilder &in, Location) {
        emit(in, false); scf::YieldOp::create(in, loc);
      });
      scf::YieldOp::create(at, loc);
    });
    gpu::TerminatorOp::create(k, loc);
    func::ReturnOp::create(b, loc);
  }
};
}
namespace mdir::test {
void registerTestTemplateWriter() { PassRegistration<TestTemplateWriter>(); }
}
