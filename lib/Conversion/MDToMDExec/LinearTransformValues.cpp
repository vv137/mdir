// Materialize the value-level FFT as an execution transform with owned buffers.

#include "mdir/Conversion/Passes.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MDExec/MDExecDialect.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mdir/Dialect/MDExec/Transfers.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/DenseSet.h"
#include <optional>
#include "mlir/Dialect/Utils/StaticValueUtils.h"
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
    if (memory != "host" && memory != "device") {
      getOperation().emitError("expected memory=host or memory=device");
      signalPassFailure();
      return;
    }
    SmallVector<md::LinearTransformOp> operations;
    getOperation()->walk([&](md::LinearTransformOp op) { operations.push_back(op); });
    for (auto op : operations)
      convert(op);
    for (auto value : llvm::reverse(generated))
      if (value.use_empty())
        value.erase();
  }

private:
  llvm::DenseMap<Value, Value> packed;
  SmallVector<tensor::GenerateOp> generated;

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

    // Explicit scalar packing is the layout-independent recovery. Connected
    // transforms share this packed representation, never native complex bytes.
    Value residentInput = memory == "device" ? packed.lookup(op.getInput())
                                               : Value();
    if (residentInput && residentInput.getParentBlock() != op->getBlock()) {
      op.emitRemark("FFT memory=device fallback: component crosses a block boundary");
      residentInput = {};
    }
    if (residentInput) {
      memref::CopyOp::create(builder, loc, residentInput, hostIn);
    } else {
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
    }

    auto execute = md_exec::LinearTransformOp::create(
        builder, loc, hostIn, hostOut, grid, op.getKind(),
        op.getDirection(), op.getAdjoint(), op.getPairing());
    execute->setAttr("fft.value_bridge", builder.getUnitAttr());
    if (memory == "device")
      execute->setAttr("fft.resident", builder.getUnitAttr());

    // Reassemble a fresh tensor in row-major order. Device placement stages
    // through host memory; its runtime copies currently synchronize streams.
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
    if (memory == "device") {
      packed[value] = hostOut;
      generated.push_back(value.getDefiningOp<tensor::GenerateOp>());
    }
    op.getResult().replaceAllUsesWith(value);
    // Leave host-buffer lifetimes to the ownership-based deallocation pass
    // after tensor bufferization. An explicit memref.dealloc here prevents
    // that pass from seeing the complete tensor-to-buffer ownership graph.
    op.erase();
  }
};

// A bounded component uses only owned, contiguous scalar buffers. Native
// complex storage is never reinterpreted: the value bridge packed it explicitly.
LogicalResult placeResidentBlock(Block *block,
                                ArrayRef<md_exec::LinearTransformOp> transforms,
                                bool &requiredFailure) {
  struct Storage {
    Value host, device;
    int64_t count = 0;
    bool initialized = false;
    bool hostRead = false;
  };
  llvm::MapVector<Value, Storage> storage;
  llvm::DenseMap<Value, Value> roots;
  auto rootOf = [](Value value) {
    while (auto cast = value.getDefiningOp<memref::CastOp>())
      value = cast.getSource();
    return value;
  };
  llvm::DenseSet<Operation *> selected;
  for (auto op : transforms)
    selected.insert(op);
  auto refuse = [&](StringRef reason) {
    transforms.front()->emitRemark()
        << "FFT memory=device fallback to host bridge: " << reason;
    return failure();
  };
  for (auto op : transforms) {
    auto grid = op.getGrid();
    int64_t real = grid[0] * grid[1] * grid[2];
    int64_t half = 2 * grid[0] * grid[1] * (grid[2] / 2 + 1);
    bool forward = op.getDirection() == "forward";
    for (auto [value, count] :
         {std::pair<Value, int64_t>{op.getInput(), forward ? real : half},
          {op.getOut(), forward ? half : real}}) {
      auto type = dyn_cast<MemRefType>(value.getType());
      Value root = rootOf(value);
      auto alloc = root.getDefiningOp<memref::AllocOp>();
      if (!type || type.getRank() != 1 || type.getMemorySpace() ||
          !alloc || alloc->getBlock() != block)
        return refuse("unknown allocation provenance or scalar layout");
      if (!type.getLayout().isIdentity() ||
          !cast<MemRefType>(root.getType()).getLayout().isIdentity()) {
        requiredFailure = true;
        return op.emitOpError("resident scalar layout must be contiguous with zero offset");
      }
      auto rootType = cast<MemRefType>(root.getType());
      std::optional<int64_t> extent = rootType.isDynamicDim(0)
          ? getConstantIntValue(alloc.getDynamicSizes().front())
          : std::optional<int64_t>(rootType.getDimSize(0));
      if (!extent)
        return refuse("unproved allocation extent");
      if (*extent != count) {
        requiredFailure = true;
        return op.emitOpError("resident allocation extent contradicts transform grid");
      }
      auto [it, inserted] = storage.try_emplace(root, Storage{root, {}, count});
      if (!inserted && it->second.count != count) {
        requiredFailure = true;
        return op.emitOpError("contradictory resident buffer extents");
      }
      roots[value] = root;
      roots[root] = root;
    }
    if (rootOf(op.getInput()) == rootOf(op.getOut())) {
      requiredFailure = true;
      return op.emitOpError("resident transform requires distinct owned buffers");
    }
  }
  // A destructive inverse input has a generation of its own, independent
  // of both a producing transform and another inverse consumer.
  for (auto op : transforms) {
    if (op.getDirection() != "backward")
      continue;
    Value input = rootOf(op.getInput());
    for (auto other : transforms)
      if (other != op && (rootOf(other.getInput()) == input ||
                          rootOf(other.getOut()) == input))
        return refuse("backward input lacks a private allocation");
  }
  // Every alias must stay inside this block's known load/store/copy graph.
  SmallVector<memref::CopyOp> copies;
  for (auto &[root, state] : storage) {
    SmallVector<Value> aliases{root};
    llvm::DenseSet<Value> seen;
    SmallVector<Operation *> uses, releases;
    while (!aliases.empty()) {
      Value alias = aliases.pop_back_val();
      if (!seen.insert(alias).second)
        continue;
      roots[alias] = root;
      for (Operation *user : alias.getUsers()) {
        Operation *owner = block->findAncestorOpInBlock(*user);
        if (!owner)
          return refuse("alias crosses a block boundary");
        uses.push_back(owner);
        if (isa<memref::DeallocOp>(user))
          releases.push_back(owner);
        if (auto cast = dyn_cast<memref::CastOp>(user)) {
          if (cast->getBlock() != block)
            return refuse("cast crosses a block boundary");
          aliases.push_back(cast.getResult());
          continue;
        }
        if (selected.contains(user) || isa<memref::DimOp, memref::DeallocOp>(user))
          continue;
        if (auto copy = dyn_cast<memref::CopyOp>(user)) {
          if (copy->getBlock() != block ||
              !storage.count(rootOf(copy.getSource())) ||
              !storage.count(rootOf(copy.getTarget())))
            return refuse("copy escapes the private storage graph");
          if (!llvm::is_contained(copies, copy))
            copies.push_back(copy);
          continue;
        }
        Operation *top = block->findAncestorOpInBlock(*user);
        if (!top)
          return refuse("alias crosses an unsupported boundary");
        if (isa<memref::LoadOp>(user)) {
          state.hostRead = true;
          continue;
        }
        if (isa<memref::StoreOp>(user)) {
          for (auto transform : transforms)
            if ((rootOf(transform.getInput()) == root ||
                 rootOf(transform.getOut()) == root) &&
                !top->isBeforeInBlock(transform))
              return refuse("host write after device use");
          continue;
        }
        return refuse("unknown alias use or release");
      }
    }
    for (Operation *release : releases)
      for (Operation *use : uses)
        if (release != use && release->isBeforeInBlock(use)) {
          requiredFailure = true;
          return release->emitOpError("resident storage is used after release");
        }
  }
  // Side-stream switches, opaque calls, and region control flow between
  // transforms are not evidence of a single producer queue.
  Operation *first = transforms.front();
  auto module = first->getParentOfType<ModuleOp>();
  Operation *last = transforms.back();
  for (Operation *op = first; op; op = op->getNextNode()) {
    bool boundary = false;
    op->walk([&](Operation *nested) {
      if (selected.contains(nested))
        return;
      StringRef dialect = nested->getName().getDialectNamespace();
      if (dialect != "arith" && dialect != "complex" &&
          dialect != "memref" && dialect != "scf")
        boundary = true;
    });
    if (boundary)
      return refuse("unknown producer queue at call boundary or operation");
    if (op == last)
      break;
  }
  requiredFailure = true;
  OpBuilder builder(first);
  Location loc = first->getLoc();
  for (auto &[root, state] : storage) {
    auto type = cast<MemRefType>(root.getType());
    auto deviceType = MemRefType::get({ShapedType::kDynamic},
        type.getElementType(), MemRefLayoutAttrInterface(),
        builder.getI64IntegerAttr(1));
    state.device = gpu::AllocOp::create(builder, loc, deviceType, Type(),
        ValueRange(), ValueRange{index(builder, loc, state.count)},
        ValueRange()).getMemref();
  }
  SmallVector<gpu::MemcpyOp> scoped;
  auto copy = [&](Value destination, Value source) {
    Type token = gpu::AsyncTokenType::get(builder.getContext());
    Value begin = gpu::WaitOp::create(builder, loc, token, ValueRange())
                      .getAsyncToken();
    auto copied = gpu::MemcpyOp::create(builder, loc, token,
        ValueRange{begin}, destination, source);
    gpu::WaitOp::create(builder, loc, Type(), ValueRange{copied.getAsyncToken()});
    scoped.push_back(copied);
  };
  auto initialize = [&](Storage &state) {
    if (!state.initialized) {
      copy(state.device, state.host);
      state.initialized = true;
    }
  };
  SmallVector<Operation *> ordered;
  for (auto op : transforms)
    ordered.push_back(op);
  for (auto op : copies)
    ordered.push_back(op);
  llvm::sort(ordered, [](Operation *a, Operation *b) {
    return a->isBeforeInBlock(b);
  });
  Operation *releaseAfter = nullptr;
  for (Operation *op : ordered) {
    builder.setInsertionPoint(op);
    if (auto move = dyn_cast<memref::CopyOp>(op)) {
      auto &source = storage.find(rootOf(move.getSource()))->second;
      auto &destination = storage.find(rootOf(move.getTarget()))->second;
      initialize(source);
      copy(destination.device, source.device);
      destination.initialized = true;
      releaseAfter = op->getPrevNode();
      move.erase();
      continue;
    }
    auto transform = cast<md_exec::LinearTransformOp>(op);
    auto &input = storage.find(rootOf(transform.getInput()))->second;
    auto &output = storage.find(rootOf(transform.getOut()))->second;
    initialize(input);
    md_exec::LinearTransformOp::create(builder, loc, input.device, output.device,
        transform.getGrid(), transform.getKind(), transform.getDirection(),
        transform.getAdjoint(), transform.getPairing());
    output.initialized = true;
    if (output.hostRead)
      copy(output.host, output.device);
    releaseAfter = op->getPrevNode();
    transform.erase();
  }
  builder.setInsertionPointAfter(releaseAfter);
  auto completed = module.lookupSymbol<func::FuncOp>("mdrtCudaWaitScoped");
  if (!completed) {
    OpBuilder at(module.getBodyRegion());
    completed = func::FuncOp::create(at, loc, "mdrtCudaWaitScoped",
        builder.getFunctionType({}, {}));
    completed.setPrivate();
  }
  func::CallOp::create(builder, loc, completed, ValueRange());
  for (auto &[root, state] : storage) {
    auto type = MemRefType::get({ShapedType::kDynamic},
        cast<MemRefType>(root.getType()).getElementType());
    Value plain = memref::MemorySpaceCastOp::create(builder, loc, type, state.device);
    gpu::DeallocOp::create(builder, loc, Type(), ValueRange(), plain);
  }
  // D358 sees the actual copy, borrow, and blocking wait sequence before
  // it is replaced by the backend adapter's equivalent scoped completion.
  if (failed(md_exec::verifyTransfers(module)))
    return failure();
  for (auto memcpy : scoped) {
    builder.setInsertionPoint(memcpy);
    auto wait = cast<gpu::WaitOp>(memcpy->getNextNode());
    auto sourceType = cast<MemRefType>(memcpy.getSrc().getType());
    auto root = roots.lookup(memcpy.getSrc());
    int64_t count = root ? storage.find(root)->second.count : 0;
    if (!count)
      for (auto &[host, state] : storage)
        if (state.device == memcpy.getSrc())
          count = state.count;
    assert(count > 0 && "verified storage extent");
    Type wide = builder.getI64Type();
    auto function = module.lookupSymbol<func::FuncOp>("mdrtCudaMemcpyScoped");
    if (!function) {
      OpBuilder at(module.getBodyRegion());
      function = func::FuncOp::create(at, loc, "mdrtCudaMemcpyScoped",
          builder.getFunctionType({wide, wide, wide}, {}));
      function.setPrivate();
    }
    auto pointer = [&](Value value) {
      Value address = memref::ExtractAlignedPointerAsIndexOp::create(builder, loc, value);
      return arith::IndexCastOp::create(builder, loc, wide, address).getResult();
    };
    Value size = arith::ConstantIntOp::create(builder, loc,
        count * (sourceType.getElementTypeBitWidth() / 8), 64);
    func::CallOp::create(builder, loc, function,
        ValueRange{pointer(memcpy.getDst()), pointer(memcpy.getSrc()), size});
    Value begin = memcpy.getAsyncDependencies().front();
    wait.erase();
    memcpy.erase();
    if (begin.use_empty())
      begin.getDefiningOp()->erase();
  }
  return success();
}

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
    llvm::MapVector<Block *, SmallVector<md_exec::LinearTransformOp>> resident;
    for (auto op : operations)
      if (op->hasAttr("fft.resident"))
        resident[op->getBlock()].push_back(op);
    llvm::DenseSet<Operation *> placed;
    for (auto &[block, group] : resident) {
      SmallVector<Operation *> originals;
      for (auto op : group)
        originals.push_back(op);
      bool requiredFailure = false;
      if (succeeded(placeResidentBlock(block, group, requiredFailure)))
        placed.insert(originals.begin(), originals.end());
      else if (requiredFailure) {
        signalPassFailure();
        return;
      }
    }
    for (auto op : operations)
      if (!placed.contains(op) && failed(place(op))) {
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
