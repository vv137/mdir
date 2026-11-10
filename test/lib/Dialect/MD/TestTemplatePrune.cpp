#include "mdir/Conversion/TemplatePruneKernels.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
using namespace mlir;
using namespace mdir::kernels;
namespace {
struct TestTemplatePrune : PassWrapper<TestTemplatePrune, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTemplatePrune)
  StringRef getArgument() const final { return "test-md-template-prune"; }
  StringRef getDescription() const final { return "Outlines fixed-block fused template pruning"; }
  void getDependentDialects(DialectRegistry &r) const override {
    r.insert<arith::ArithDialect, func::FuncDialect, gpu::GPUDialect,
        NVVM::NVVMDialect, memref::MemRefDialect, scf::SCFDialect, vector::VectorDialect>();
  }
  void runOnOperation() final {
    auto f = getOperation().lookupSymbol<func::FuncOp>("prune_work");
    if (!f || f.getNumArguments() != 14 || !f.isExternal()) {
      getOperation().emitError("expected a fourteen-argument prune_work declaration");
      return signalPassFailure();
    }
    Location loc = f.getLoc();
    OpBuilder b = OpBuilder::atBlockEnd(f.addEntryBlock());
    auto ix = [&](OpBuilder &at, int64_t n) { return createIndex(at, loc, n); };
    TemplatePruneBuffers v{f.getArgument(0), f.getArgument(1), f.getArgument(2),
        f.getArgument(3), f.getArgument(4), f.getArgument(5), f.getArgument(6),
        f.getArgument(7), f.getArgument(8), f.getArgument(9), f.getArgument(10), f.getArgument(11)};
    v.groupLengths = f.getArgument(13);
    Value n = arith::DivUIOp::create(b, loc,
        memref::DimOp::create(b, loc, v.outerUnits, ix(b, 0)), ix(b, 4));
    Value one = ix(b, 1), block = ix(b, 128);
    Value threads = arith::MulIOp::create(b, loc, n, ix(b, 32));
    Value grid = arith::CeilDivUIOp::create(b, loc, threads, block);
    auto launch = gpu::LaunchOp::create(b, loc, grid, one, one, block, one, one);
    OpBuilder k = OpBuilder::atBlockEnd(&launch.getBody().front());
    Value thread = arith::AddIOp::create(k, loc, launch.getThreadIds().x,
        arith::MulIOp::create(k, loc, launch.getBlockIds().x, block));
    Value work = arith::DivUIOp::create(k, loc, thread, ix(k, 32));
    Value lane = arith::IndexCastOp::create(k, loc, k.getI32Type(),
        arith::RemUIOp::create(k, loc, thread, ix(k, 32)));
    scf::IfOp::create(k, loc,
        arith::CmpIOp::create(k, loc, arith::CmpIPredicate::ult, work, n),
        [&](OpBuilder &at, Location) {
      emitPruneTemplateWork(at, loc, v, work, lane);
      Value base = arith::MulIOp::create(at, loc, work, ix(at, 4));
      auto bound = [&](int64_t col) -> Value {
        return arith::IndexCastUIOp::create(at, loc, at.getIndexType(),
            memref::LoadOp::create(at, loc, v.innerUnits,
                ValueRange{arith::AddIOp::create(at, loc, base, ix(at, col))}));
      };
      Value begin = bound(2), end = bound(3);
      Value first = arith::AddIOp::create(at, loc, begin,
          arith::IndexCastUIOp::create(at, loc, at.getIndexType(), lane));
      // Consume the block here, in the same kernel immediately after
      // pruning, using different lanes from the compacted-store owners.
      scf::ForOp::create(at, loc, first, end, ix(at, 32), ValueRange(),
          [&](OpBuilder &in, Location, Value entry, ValueRange) {
        for (int64_t col = 0; col != 4; ++col) {
          Value address = arith::AddIOp::create(in, loc,
              arith::MulIOp::create(in, loc, entry, ix(in, 4)), ix(in, col));
          Value value = memref::LoadOp::create(in, loc, v.innerEntries, ValueRange{address});
          memref::StoreOp::create(in, loc, value, f.getArgument(12), ValueRange{address});
        }
        scf::YieldOp::create(in, loc);
      });
      scf::YieldOp::create(at, loc);
    });
    gpu::TerminatorOp::create(k, loc);
    func::ReturnOp::create(b, loc);
  }
};
}
namespace mdir::test {
void registerTestTemplatePrune() { PassRegistration<TestTemplatePrune>(); }
}
