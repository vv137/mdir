// A test pass that replaces calls of `@test.warp_*` with the code of the
// warp helpers of CompactionEmitter.h (D406), so that a test runs each
// helper by itself: since D406 the helpers have no name in the IR (the
// groups build calls them in C++, the matrix template writes the upstream
// ops), and the review of #862 asked for their own tests
// (D[md-alg-stages]).
//
//   @test.lanes_below(lane: i32) -> i32           emitLanesBelow
//   @test.warp_ballot(predicate: i1) -> i32       emitWarpBallot
//   @test.warp_rank(ballot: i32, below: i32) -> i32  emitWarpRank
//   @test.warp_count(ballot: i32) -> i32          emitWarpCount
//   @test.warp_scan(x, lane, zero, width: i32) -> i32 {lanes = N}
//                                                 emitWarpScan, with
//                                                 `unsigned` for an
//                                                 unsigned lane

#include "mdir/Conversion/CompactionEmitter.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mdir;

namespace {

struct TestWarpHelpers
    : public PassWrapper<TestWarpHelpers, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestWarpHelpers)

  StringRef getArgument() const final { return "test-md-alg-warp-helpers"; }
  StringRef getDescription() const final {
    return "Replaces calls of @test.warp_* with the warp helpers' code";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, gpu::GPUDialect, math::MathDialect>();
  }

  void runOnOperation() final {
    SmallVector<func::CallOp> calls;
    getOperation()->walk([&](func::CallOp call) {
      if (call.getCallee().starts_with("test."))
        calls.push_back(call);
    });
    for (func::CallOp call : calls) {
      OpBuilder b(call);
      Location loc = call.getLoc();
      OperandRange a = call.getOperands();
      StringRef callee = call.getCallee();
      Value result;
      if (callee == "test.lanes_below")
        result = kernels::emitLanesBelow(b, loc, a[0], Value());
      else if (callee == "test.warp_ballot")
        result = kernels::emitWarpBallot(b, loc, a[0]);
      else if (callee == "test.warp_rank")
        result = kernels::emitWarpRank(b, loc, a[0], a[1]);
      else if (callee == "test.warp_count")
        result = kernels::emitWarpCount(b, loc, a[0]);
      else if (callee == "test.warp_scan") {
        auto lanes = call->getAttrOfType<IntegerAttr>("lanes");
        result = kernels::emitWarpScan(b, loc, a[0], a[1], a[2], a[3],
                                       lanes ? lanes.getInt() : 32,
                                       call->hasAttr("unsigned"));
      } else {
        call.emitError() << "is no warp helper";
        return signalPassFailure();
      }
      call.getResult(0).replaceAllUsesWith(result);
      call.erase();
    }
    // The declarations, now without calls.
    SmallVector<func::FuncOp> declarations;
    for (auto f : getOperation().getOps<func::FuncOp>())
      if (f.isDeclaration() && f.getSymName().starts_with("test."))
        declarations.push_back(f);
    for (func::FuncOp f : declarations)
      f.erase();
  }
};

} // namespace

namespace mdir {
namespace test {
void registerTestWarpHelpers() { PassRegistration<TestWarpHelpers>(); }
} // namespace test
} // namespace mdir
