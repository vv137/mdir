// Outliner for the complete-frame position writer, using its actual body.
#include "mdir/Conversion/TemplateParticleKernels.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
using namespace mlir;
using namespace mdir::kernels;
namespace {
struct TestTemplateParticles : PassWrapper<TestTemplateParticles, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTemplateParticles)
  StringRef getArgument() const final { return "test-md-template-particles"; }
  StringRef getDescription() const final { return "Outlines complete-frame template position writers"; }
  void getDependentDialects(DialectRegistry &r) const override {
    r.insert<arith::ArithDialect, func::FuncDialect, gpu::GPUDialect,
        memref::MemRefDialect, scf::SCFDialect, vector::VectorDialect>();
  }
  void runOnOperation() final {
    auto function = getOperation().lookupSymbol<func::FuncOp>("writer_source");
    SmallVector<mdir::md_exec::ParticleForOp> loops;
    if (function) function.walk([&](mdir::md_exec::ParticleForOp op) { loops.push_back(op); });
    if (!function || function.getNumArguments() != 9 || loops.size() != 1 ||
        loops[0].getOuts().size() != 2 || loops[0].getReduce().size() != 2) {
      getOperation().emitError("writer fixture expects nine arguments and one two-output/two-reduction loop");
      return signalPassFailure();
    }
    auto op = loops[0];
    Location loc = op.getLoc();
    OpBuilder b(op);
    auto ix = [&](OpBuilder &at, int64_t n) { return createIndex(at, loc, n); };
    Value slots = function.getArgument(3), members = function.getArgument(4);
    Value waters = function.getArgument(5), audit = function.getArgument(6);
    Value reductions = function.getArgument(7), visits = function.getArgument(8);
    Value n = memref::DimOp::create(b, loc, slots, ix(b, 0));
    Value one = ix(b, 1), block = ix(b, 128);
    Value grid = arith::CeilDivUIOp::create(b, loc, n, block);
    auto launch = gpu::LaunchOp::create(b, loc, grid, one, one, block, one, one);
    OpBuilder k = OpBuilder::atBlockEnd(&launch.getBody().front());
    Value particle = arith::AddIOp::create(k, loc, launch.getThreadIds().x,
        arith::MulIOp::create(k, loc, launch.getBlockIds().x, block));
    Value valid = arith::CmpIOp::create(k, loc, arith::CmpIPredicate::ult, particle, n);
    scf::IfOp::create(k, loc, valid, [&](OpBuilder &at, Location) {
      IRMapping captures;
      emitTemplateParticleKernel(at, loc, op, particle, slots, members, waters, captures,
          [&](OpBuilder &in, Value unit, ArrayRef<TemplateParticleResult> results) {
        for (const auto &result : results) {
          Value count = arith::ConstantIntOp::create(in, loc, 1, 32);
          memref::AtomicRMWOp::create(in, loc, arith::AtomicRMWKind::addi,
              count, visits, ValueRange{result.particle});
          Value flag = arith::UIToFPOp::create(in, loc, in.getF64Type(), result.reductions[1]);
          memref::StoreOp::create(in, loc, result.reductions[0], reductions,
                                  ValueRange{result.particle, ix(in, 0)});
          memref::StoreOp::create(in, loc, flag, reductions,
                                  ValueRange{result.particle, ix(in, 1)});
        }
        Value real = arith::CmpIOp::create(in, loc, arith::CmpIPredicate::sge, unit,
            arith::ConstantIntOp::create(in, loc, 0, 32));
        scf::IfOp::create(in, loc, real, [&](OpBuilder &out, Location) {
          Value base = arith::IndexCastOp::create(out, loc, out.getIndexType(), unit);
          if (results.size() == 3)
            base = arith::MulIOp::create(out, loc, base, ix(out, 3));
          else
            base = arith::AddIOp::create(out, loc, base,
                arith::MulIOp::create(out, loc, waters, ix(out, 2)));
          for (auto [site, result] : llvm::enumerate(results)) {
            Value row = arith::AddIOp::create(out, loc, base, ix(out, site));
            storeElement(out, loc, result.outputs[0], audit, row);
          }
          scf::YieldOp::create(out, loc);
        });
      });
      scf::YieldOp::create(at, loc);
    });
    gpu::TerminatorOp::create(k, loc);
    for (Value result : op.getResults())
      result.replaceAllUsesWith(createZero(b, loc, result.getType()));
    op.erase();
  }
};
}
namespace mdir::test {
void registerTestTemplateParticles() { PassRegistration<TestTemplateParticles>(); }
}
