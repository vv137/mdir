// A native frame batch advances contiguous device views without copying data.
#include "SimpleFrameEntryIR.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
using namespace mlir;
using namespace mdir::driver::builder;

llvm::Expected<scf::ForOp> mdir::driver::builder::buildFrameBatchRows(
    OpBuilder &builder, Value positionBytes, Value cells, Value particles,
    Type stateType, unsigned cellEdges,
    llvm::function_ref<llvm::Error(OpBuilder &, const FrameBatchRowIR &)> body) {
  auto bytes = dyn_cast<MemRefType>(positionBytes.getType());
  auto boxes = dyn_cast<MemRefType>(cells.getType());
  if (!bytes || bytes.getRank() != 1 || !bytes.getElementType().isInteger(8) ||
      !bytes.getLayout().isIdentity() || !boxes || boxes.getRank() != 2 ||
      !boxes.getElementType().isF64() || boxes.getDimSize(1) != 6 ||
      boxes.getMemorySpace() || !particles.getType().isIndex() ||
      !(stateType.isF32() || stateType.isF64()) ||
      !(cellEdges == 3 || cellEdges == 6))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
        "native frame batch requires contiguous position bytes, host cells "
        "with six columns, an index particle count, f32/f64 state and 3/6 cell edges");
  auto loc = builder.getUnknownLoc();
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  Value count = memref::DimOp::create(builder, loc, cells, zero);
  Value stride = arith::ConstantIndexOp::create(builder, loc,
      3 * (stateType.isF64() ? 8 : 4));
  Value frameBytes = arith::MulIOp::create(builder, loc, particles, stride);
  auto positions = MemRefType::get({ShapedType::kDynamic, 3}, stateType,
      MemRefLayoutAttrInterface(), bytes.getMemorySpace());
  return buildEntryLoop(builder, "frame batch", zero, count, one, {},
      [&](OpBuilder &at, Value index, ValueRange)
          -> llvm::Expected<SmallVector<Value>> {
    Value offset = arith::MulIOp::create(at, loc, index, frameBytes);
    FrameBatchRowIR row;
    row.index = index;
    row.positions = memref::ViewOp::create(at, loc, positions,
        positionBytes, offset, ValueRange{particles});
    for (unsigned edge = 0; edge != cellEdges; ++edge) {
      Value column = arith::ConstantIndexOp::create(at, loc, edge);
      row.edges.push_back(memref::LoadOp::create(at, loc, cells,
          ValueRange{index, column}));
    }
    if (llvm::Error error = body(at, row))
      return std::move(error);
    return SmallVector<Value>{};
  });
}
