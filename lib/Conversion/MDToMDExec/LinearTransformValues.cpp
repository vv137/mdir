// Materialize the value-level FFT as an execution transform with owned buffers.

#include "mdir/Conversion/Passes.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MDExec/MDExecDialect.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;
using namespace mdir;

namespace mdir {
#define GEN_PASS_DEF_CONVERTLINEARTRANSFORMVALUES
#include "mdir/Conversion/Passes.h.inc"
} // namespace mdir

namespace {

Value index(OpBuilder &builder, Location loc, int64_t n) {
  return arith::ConstantIndexOp::create(builder, loc, n);
}

Value flat(OpBuilder &builder, Location loc, ValueRange xyz,
           ArrayRef<int64_t> shape) {
  Value row = arith::AddIOp::create(
      builder, loc,
      arith::MulIOp::create(builder, loc, xyz[0], index(builder, loc, shape[1])),
      xyz[1]);
  return arith::AddIOp::create(
      builder, loc,
      arith::MulIOp::create(builder, loc, row, index(builder, loc, shape[2])),
      xyz[2]);
}

void transfer(OpBuilder &builder, Location loc, Value destination,
              Value source) {
  Type token = gpu::AsyncTokenType::get(builder.getContext());
  Value begin =
      gpu::WaitOp::create(builder, loc, token, ValueRange()).getAsyncToken();
  Value copied = gpu::MemcpyOp::create(builder, loc, token,
                                       ValueRange{begin}, destination, source)
                     .getAsyncToken();
  gpu::WaitOp::create(builder, loc, Type(), ValueRange{copied});
}

class ConvertLinearTransformValues
    : public mdir::impl::ConvertLinearTransformValuesBase<ConvertLinearTransformValues> {
public:
  using mdir::impl::ConvertLinearTransformValuesBase<
      ConvertLinearTransformValues>::ConvertLinearTransformValuesBase;

  void runOnOperation() final {
    if (memory != "host" && memory != "device") {
      getOperation()->emitError()
          << "expected memory 'host' or 'device', got '" << memory << "'";
      signalPassFailure();
      return;
    }
    SmallVector<md::LinearTransformOp> operations;
    getOperation()->walk([&](md::LinearTransformOp op) { operations.push_back(op); });
    for (auto op : operations)
      convert(op);
  }

private:
  void convert(md::LinearTransformOp op) {
    OpBuilder builder(op);
    Location loc = op.getLoc();
    auto inputType = cast<RankedTensorType>(op.getInput().getType());
    auto outputType = cast<RankedTensorType>(op.getResult().getType());
    bool forward = op.getDirection() == "forward";
    auto realType = forward ? inputType : outputType;
    auto halfType = forward ? outputType : inputType;
    Type scalar = realType.getElementType();
    ArrayRef<int64_t> grid = realType.getShape();
    int64_t realCount = grid[0] * grid[1] * grid[2];
    int64_t halfCount = 2 * halfType.getNumElements();
    int64_t inputCount = forward ? realCount : halfCount;
    int64_t outputCount = forward ? halfCount : realCount;
    auto hostType = MemRefType::get({ShapedType::kDynamic}, scalar);
    Value hostIn = memref::AllocOp::create(builder, loc, hostType,
                                            ValueRange{index(builder, loc, inputCount)});
    Value hostOut = memref::AllocOp::create(builder, loc, hostType,
                                             ValueRange{index(builder, loc, outputCount)});

    // Every semantic tensor has its own packed source. In particular, a
    // backward execution call may overwrite this buffer without mutating a
    // value that another consumer reads.
    Value zero = index(builder, loc, 0);
    Value one = index(builder, loc, 1);
    Value two = index(builder, loc, 2);
    Value limit = index(builder, loc,
                        forward ? realCount : halfType.getNumElements());
    auto loop = scf::ForOp::create(builder, loc, zero, limit, one);
    OpBuilder body = OpBuilder::atBlockBegin(loop.getBody());
    Value i = loop.getInductionVar();
    ArrayRef<int64_t> shape = inputType.getShape();
    Value plane = index(body, loc, shape[1] * shape[2]);
    Value last = index(body, loc, shape[2]);
    Value x = arith::DivUIOp::create(body, loc, i, plane);
    Value rem = arith::RemUIOp::create(body, loc, i, plane);
    Value y = arith::DivUIOp::create(body, loc, rem, last);
    Value z = arith::RemUIOp::create(body, loc, rem, last);
    Value element = tensor::ExtractOp::create(body, loc, op.getInput(),
                                               ValueRange{x, y, z});
    if (forward) {
      memref::StoreOp::create(body, loc, element, hostIn, ValueRange{i});
    } else {
      Value offset = arith::MulIOp::create(body, loc, i, two);
      Value imagOffset = arith::AddIOp::create(body, loc, offset, one);
      Value real = complex::ReOp::create(body, loc, element);
      Value imag = complex::ImOp::create(body, loc, element);
      memref::StoreOp::create(body, loc, real, hostIn, ValueRange{offset});
      memref::StoreOp::create(body, loc, imag, hostIn,
                              ValueRange{imagOffset});
    }
    builder.setInsertionPointAfter(loop);

    Value executeIn = hostIn;
    Value executeOut = hostOut;
    if (memory == "device") {
      auto space = builder.getI64IntegerAttr(1);
      auto deviceType = MemRefType::get(
          {ShapedType::kDynamic}, scalar, MemRefLayoutAttrInterface(), space);
      executeIn = gpu::AllocOp::create(builder, loc, deviceType,
                                       Type(), ValueRange(),
                                       ValueRange{index(builder, loc, inputCount)},
                                       ValueRange()).getMemref();
      executeOut = gpu::AllocOp::create(builder, loc, deviceType,
                                        Type(), ValueRange(),
                                        ValueRange{index(builder, loc, outputCount)},
                                        ValueRange()).getMemref();
      transfer(builder, loc, executeIn, hostIn);
    }
    md_exec::LinearTransformOp::create(builder, loc, executeIn, executeOut,
                                        grid, op.getKind(), op.getDirection(),
                                        op.getAdjoint(), op.getPairing());
    if (memory == "device")
      transfer(builder, loc, hostOut, executeOut);

    // Reassemble a fresh tensor in row-major order. The generated value
    // depends on this transform alone; no unrelated stream is waited on.
    Value value = tensor::GenerateOp::create(
        builder, loc, outputType, ValueRange(),
        [&](OpBuilder &at, Location where, ValueRange xyz) {
          Value offset = flat(at, where, xyz, outputType.getShape());
          Value element;
          if (forward) {
            Value pair = arith::MulIOp::create(at, where, offset,
                                                index(at, where, 2));
            Value next = arith::AddIOp::create(at, where, pair,
                                                index(at, where, 1));
            Value re = memref::LoadOp::create(at, where, hostOut, pair);
            Value im = memref::LoadOp::create(at, where, hostOut, next);
            element = complex::CreateOp::create(at, where,
                                                  outputType.getElementType(),
                                                  re, im);
          } else {
            element = memref::LoadOp::create(at, where, hostOut, offset);
          }
          tensor::YieldOp::create(at, where, element);
        });
    op.getResult().replaceAllUsesWith(value);
    builder.setInsertionPointAfter(value.getDefiningOp());
    if (memory == "device") {
      Value plainIn = memref::MemorySpaceCastOp::create(
          builder, loc, hostType, executeIn);
      Value plainOut = memref::MemorySpaceCastOp::create(
          builder, loc, hostType, executeOut);
      gpu::DeallocOp::create(builder, loc, Type(), ValueRange(), plainIn);
      gpu::DeallocOp::create(builder, loc, Type(), ValueRange(), plainOut);
    }
    memref::DeallocOp::create(builder, loc, hostIn);
    memref::DeallocOp::create(builder, loc, hostOut);
    op.erase();
  }
};

} // namespace

