// Real-to-complex and complex-to-real transforms on the host [Reinecke2019].
// Partitioning by lines keeps each transform independent of the thread count.
#include "PMEFFTs.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;

namespace {
template <bool forward> void emitTransform(ModuleOp module) {
  auto fn = module.lookupSymbol<func::FuncOp>(
      forward ? "mdrt.pme_fft_forward" : "mdrt.pme_fft_backward");
  OpBuilder b(module.getContext());
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  Location loc = fn.getLoc();
  Value src = entry->getArgument(0), dst = entry->getArgument(1);
  Type grid = cast<MemRefType>(src.getType()).getElementType();
  Value spectrum = forward ? dst : src;
  Value planes = entry->getArgument(2);
  Type i64 = b.getI64Type();
  Value begin = arith::ConstantIndexOp::create(b, loc, 0);
  Value step = arith::ConstantIndexOp::create(b, loc, 1);
  Value zero = arith::ConstantIntOp::create(b, loc, i64, 0);
  Value one = arith::ConstantIntOp::create(b, loc, i64, 1);
  Value nx = arith::IndexCastOp::create(b, loc, i64, planes);
  Value ny = arith::IndexCastOp::create(b, loc, i64, entry->getArgument(3));
  Value nz = arith::IndexCastOp::create(b, loc, i64, entry->getArgument(4));
  func::CallOp::create(b, loc, "mdrtFFTPrepare", TypeRange{},
                       ValueRange{nx, ny, nz});
  auto rows = [&] {
    scf::ParallelOp::create(
        b, loc, ValueRange{begin}, ValueRange{planes}, ValueRange{step},
        [&](OpBuilder &body, Location, ValueRange iv) {
          Value plane = arith::IndexCastOp::create(body, loc, i64, iv[0]);
          Value first = arith::MulIOp::create(body, loc, plane, ny);
          Value last = arith::AddIOp::create(body, loc, first, ny);
          func::CallOp::create(
              body, loc,
              mdir::kernels::hostFFTName(grid, forward ? "RowsForward"
                                                       : "RowsBackward"),
              TypeRange{}, ValueRange{src, dst, nz, first, last});
        });
  };
  auto columns = [&](Value outer, Value axis) {
    Value blocks = func::CallOp::create(b, loc, "mdrtFFTColumnBlocks",
                                        TypeRange{i64}, ValueRange{outer, nz})
                       .getResult(0);
    Value count = arith::IndexCastOp::create(b, loc, b.getIndexType(), blocks);
    scf::ParallelOp::create(
        b, loc, ValueRange{begin}, ValueRange{count}, ValueRange{step},
        [&](OpBuilder &body, Location, ValueRange iv) {
          Value first = arith::IndexCastOp::create(body, loc, i64, iv[0]);
          Value last = arith::AddIOp::create(body, loc, first, one);
          func::CallOp::create(body, loc,
                               mdir::kernels::hostFFTName(grid, "Columns"),
                               TypeRange{},
                               ValueRange{spectrum, nx, ny, nz, axis,
                                          forward ? one : zero, first, last});
        });
  };
  if constexpr (forward)
    rows();
  columns(nx, one);
  columns(ny, zero);
  if constexpr (!forward)
    rows();
  func::ReturnOp::create(b, loc);
}
} // namespace

void mdir::kernels::emitHostPMEFFTs(ModuleOp module) {
  emitTransform<true>(module);
  emitTransform<false>(module);
}
