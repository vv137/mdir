// Structured grid domains and the Euclidean/Hermitian real pairings (D319).
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <limits>

using namespace mlir;
using namespace mdir::md;

static bool realValue(Type type) {
  if (type.isF64())
    return true;
  auto vector = dyn_cast<VectorType>(type);
  return vector && vector.getRank() == 1 && !vector.isScalable() &&
         vector.getElementType().isF64();
}

static Type kernelType(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    type = tensor.getElementType();
  if (auto complex = dyn_cast<ComplexType>(type))
    return complex.getElementType().isF64()
        ? Type(VectorType::get({2}, complex.getElementType())) : Type();
  return realValue(type) ? type : Type();
}

static FailureOr<SmallVector<int64_t>> domain(Operation *op) {
  auto grid = op->getAttrOfType<DenseI64ArrayAttr>("grid").asArrayRef();
  if (grid.size() != 3) {
    op->emitOpError("expected three positive grid extents");
    return failure();
  }
  int64_t count = 1;
  for (int64_t n : grid) {
    if (n <= 0 || n > std::numeric_limits<int32_t>::max() ||
        count > std::numeric_limits<int64_t>::max() / n) {
      op->emitOpError("grid extents must fit i32 and their product signed i64");
      return failure();
    }
    count *= n;
  }
  SmallVector<int64_t> shape(grid);
  StringRef pairing = op->getAttrOfType<StringAttr>("pairing").getValue();
  if (pairing == "real_hermitian_half")
    shape[2] = grid[2] / 2 + 1;
  else if (pairing != "euclidean") {
    op->emitOpError("expected euclidean or real_hermitian_half pairing");
    return failure();
  }
  return shape;
}

static LogicalResult verifyGrid(Operation *op, bool sum) {
  auto shape = domain(op);
  if (failed(shape))
    return failure();
  if (!op->getNumOperands())
    return op->emitOpError("expected at least one explicit grid or uniform input");
  for (Value input : op->getOperands()) {
    if (!kernelType(input.getType()))
      return op->emitOpError("expected f64 scalar/vector inputs or static real/complex f64 grids");
    if (auto tensor = dyn_cast<RankedTensorType>(input.getType()))
      if (tensor.getShape() != ArrayRef<int64_t>(*shape))
        return op->emitOpError("tensor input must have the complete declared domain shape");
  }
  if (sum) {
    if (!realValue(op->getResult(0).getType()))
      return op->emitOpError("expected a real f64 scalar or vector reduction result");
  } else {
    auto tensor = cast<RankedTensorType>(op->getResult(0).getType());
    if (tensor.getShape() != ArrayRef<int64_t>(*shape) || !kernelType(tensor))
      return op->emitOpError("result must be a complete static real/complex f64 grid");
  }
  if (op->hasAttr("listed") || op->hasAttr("coverage") ||
      op->hasAttr("algebra") || op->hasAttr("reduction_tree"))
    return op->emitOpError("full coverage and ideal ordered sum cannot be overridden");
  return success();
}

static LogicalResult verifyKernel(Operation *op, bool sum) {
  Block &block = op->getRegion(0).front();
  Type index = VectorType::get({3}, IndexType::get(op->getContext()));
  if (block.getNumArguments() != op->getNumOperands() + 1 ||
      block.getArgument(0).getType() != index)
    return op->emitOpError("expected a vector<3xindex> and one kernel argument per input");
  for (auto [input, argument] : llvm::zip(op->getOperands(),
                                        block.getArguments().drop_front()))
    if (kernelType(input.getType()) != argument.getType())
      return op->emitOpError("kernel argument must match its input element, with complex as vector<2xf64>");
  auto yield = dyn_cast<YieldOp>(block.getTerminator());
  Type result = op->getResult(0).getType();
  Type yielded = sum ? result : kernelType(result);
  if (!yield || yield.getValues().size() != 1 ||
      yield.getValues()[0].getType() != yielded)
    return op->emitOpError("expected md.yield of exactly the kernel result type");
  for (Operation &inner : block) {
    if (inner.getNumRegions() || !isMemoryEffectFree(&inner))
      return op->emitOpError("grid kernel must be straight-line and effect-free");
    for (Value value : inner.getOperands()) {
      if (auto arg = dyn_cast<BlockArgument>(value)) {
        if (arg.getOwner() == &block)
          continue;
      } else if (value.getDefiningOp()->getBlock() == &block) {
        continue;
      }
      Attribute literal;
      if (matchPattern(value, m_Constant(&literal)) &&
          isMemoryEffectFree(value.getDefiningOp()))
        continue;
      return op->emitOpError("grid kernel cannot capture dynamic values; use explicit uniform inputs");
    }
  }
  return success();
}

LogicalResult MapGridOp::verify() { return verifyGrid(*this, false); }
LogicalResult SumGridOp::verify() { return verifyGrid(*this, true); }
LogicalResult MapGridOp::verifyRegions() { return verifyKernel(*this, false); }
LogicalResult SumGridOp::verifyRegions() { return verifyKernel(*this, true); }

static LogicalResult verifyRequest(Operation *op, bool reverse) {
  Operation *source = op->getOperand(0).getDefiningOp();
  if (!isa_and_nonnull<MapGridOp, SumGridOp>(source))
    return op->emitOpError("expected a structured grid map or sum result");
  int64_t operand = op->getAttrOfType<IntegerAttr>("operand").getInt();
  if (operand < 0 || operand >= source->getNumOperands())
    return op->emitOpError("grid derivative operand index is out of range");
  Type input = source->getOperand(operand).getType();
  Type output = source->getResult(0).getType();
  if (op->getOperand(1).getType() != (reverse ? output : input) ||
      op->getResult(0).getType() != (reverse ? input : output))
    return op->emitOpError("grid derivative seed or result has the wrong operand shape");
  return success();
}
LogicalResult GridJVPOp::verify() { return verifyRequest(*this, false); }
LogicalResult GridVJPOp::verify() { return verifyRequest(*this, true); }
