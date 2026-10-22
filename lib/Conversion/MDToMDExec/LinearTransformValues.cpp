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
#define GEN_PASS_DEF_PLACELINEARTRANSFORMDEVICE
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

    auto execute = md_exec::LinearTransformOp::create(
        builder, loc, hostIn, hostOut, grid, op.getKind(),
        op.getDirection(), op.getAdjoint(), op.getPairing());
    execute->setAttr("fft.value_bridge", builder.getUnitAttr());

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
    // Leave host-buffer lifetimes to the ownership-based deallocation pass
    // after tensor bufferization. An explicit memref.dealloc here prevents
    // that pass from seeing the complete tensor-to-buffer ownership graph.
    op.erase();
  }
};

class PlaceLinearTransformDevice
    : public mdir::impl::PlaceLinearTransformDeviceBase<PlaceLinearTransformDevice> {
public:
  using mdir::impl::PlaceLinearTransformDeviceBase<
      PlaceLinearTransformDevice>::PlaceLinearTransformDeviceBase;

  void runOnOperation() final {
    SmallVector<md_exec::LinearTransformOp> operations;
    getOperation()->walk([&](md_exec::LinearTransformOp op) {
      if (op->hasAttr("fft.value_bridge"))
        operations.push_back(op);
    });
    for (auto op : operations)
      if (failed(place(op))) {
        signalPassFailure();
        return;
      }
  }

private:
  LogicalResult place(md_exec::LinearTransformOp op) {
    Value input = op.getInput();
    Value output = op.getOut();
    auto allocation = [](Value value) -> Operation * {
      while (auto cast = value.getDefiningOp<memref::CastOp>())
        value = cast.getSource();
      return value.getDefiningOp<memref::AllocOp>();
    };
    Operation *inputAllocation = allocation(input);
    Operation *outputAllocation = allocation(output);
    auto inType = dyn_cast<MemRefType>(input.getType());
    auto outType = dyn_cast<MemRefType>(output.getType());
    if (!inType || !outType || inType.getMemorySpace() ||
        outType.getMemorySpace() || !inputAllocation || !outputAllocation ||
        inputAllocation == outputAllocation)
      return op.emitOpError()
             << "expected distinct owned host allocations from the value bridge";
    OpBuilder builder(op);
    Location loc = op.getLoc();
    auto space = builder.getI64IntegerAttr(1);
    auto deviceType = MemRefType::get(
        {ShapedType::kDynamic}, inType.getElementType(),
        MemRefLayoutAttrInterface(), space);
    Value deviceIn = gpu::AllocOp::create(
        builder, loc, deviceType, Type(), ValueRange(),
        ValueRange{memref::DimOp::create(builder, loc, input, 0)},
        ValueRange()).getMemref();
    Value deviceOut = gpu::AllocOp::create(
        builder, loc, deviceType, Type(), ValueRange(),
        ValueRange{memref::DimOp::create(builder, loc, output, 0)},
        ValueRange()).getMemref();
    transfer(builder, loc, deviceIn, input);
    md_exec::LinearTransformOp::create(
        builder, loc, deviceIn, deviceOut, op.getGrid(), op.getKind(),
        op.getDirection(), op.getAdjoint(), op.getPairing());
    transfer(builder, loc, output, deviceOut);
    auto plainType = MemRefType::get({ShapedType::kDynamic},
                                      inType.getElementType());
    Value plainIn = memref::MemorySpaceCastOp::create(
        builder, loc, plainType, deviceIn);
    Value plainOut = memref::MemorySpaceCastOp::create(
        builder, loc, plainType, deviceOut);
    gpu::DeallocOp::create(builder, loc, Type(), ValueRange(), plainIn);
    gpu::DeallocOp::create(builder, loc, Type(), ValueRange(), plainOut);
    op.erase();
    return success();
  }
};

} // namespace

