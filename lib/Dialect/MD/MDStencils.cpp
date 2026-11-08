// Structural assignment and interpolation contracts (Essmann et al., 1995).
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;
using namespace mdir::md;

LogicalResult BSplineWindowOp::verify() {
  if (getOrder() < 2 || getOrder() > 8)
    return emitOpError("expected B-spline order from 2 through 8");
  return success();
}

static LogicalResult verifyStencilApply(Operation *op, bool scatter) {
  auto relation = op->getOperand(0).getDefiningOp<GridStencilOp>();
  if (!relation || relation.getPositions() != op->getOperand(1) ||
      relation.getCell() != op->getOperand(2))
    return op->emitOpError(
        "expected the relation constructor's exact positions and cell operands");
  auto stencil = cast<GridStencilType>(op->getOperand(0).getType());
  auto grid = RankedTensorType::get(stencil.getShape().asArrayRef(),
                                    Float64Type::get(op->getContext()));
  auto field = FieldType::get(op->getContext(), stencil.getParticleSet(),
                             1, Float64Type::get(op->getContext()));
  if (op->getOperand(3).getType() != (scatter ? Type(field) : Type(grid)) ||
      op->getResult(0).getType() != (scatter ? Type(grid) : Type(field)))
    return op->emitOpError(
        "expected a scalar f64 particle field and the full stencil-shaped f64 grid");
  // These are semantic full sums, with no list, quantizer, or in-place alias.
  if (op->hasAttr("listed") || op->hasAttr("coverage") ||
      op->hasAttr("algebra"))
    return op->emitOpError(
        "full coverage and ideal additive algebra cannot be overridden");
  return success();
}

static LogicalResult verifyStencilKernel(Operation *op) {
  Block &body = op->getRegion(0).front();
  auto coordinate = VectorType::get({3}, Float64Type::get(op->getContext()));
  if (body.getNumArguments() != 1 ||
      body.getArgument(0).getType() != coordinate)
    return op->emitOpError("expected one vector<3xf64> support-distance argument");
  auto yield = dyn_cast<YieldOp>(body.getTerminator());
  if (!yield || yield.getValues().size() != 1 ||
      !yield.getValues()[0].getType().isF64())
    return op->emitOpError("expected one f64 window weight from md.yield");
  auto stencil = cast<GridStencilType>(op->getOperand(0).getType());
  for (Operation &inner : body) {
    if (inner.getNumRegions() || !isMemoryEffectFree(&inner))
      return op->emitOpError("window kernel must be straight-line and effect-free");
    for (Value operand : inner.getOperands()) {
      auto arg = dyn_cast<BlockArgument>(operand);
      if ((arg && arg.getOwner() != &body) ||
          (!arg && operand.getDefiningOp()->getBlock() != &body))
        return op->emitOpError(
            "window kernel cannot capture values; continuous inputs must be explicit");
    }
    if (auto window = dyn_cast<BSplineWindowOp>(inner))
      if (window.getOrder() != stencil.getOrder())
        return op->emitOpError("window and stencil support orders must agree");
  }
  return success();
}

LogicalResult ScatterStencilOp::verify() {
  return verifyStencilApply(*this, true);
}
LogicalResult GatherStencilOp::verify() {
  return verifyStencilApply(*this, false);
}
LogicalResult ScatterStencilOp::verifyRegions() {
  return verifyStencilKernel(*this);
}
LogicalResult GatherStencilOp::verifyRegions() {
  return verifyStencilKernel(*this);
}

static LogicalResult verifyRequest(Operation *op, bool reverse) {
  Operation *source = op->getOperand(0).getDefiningOp();
  if (!isa_and_nonnull<ScatterStencilOp, GatherStencilOp, BSplineWindowOp>(source))
    return op->emitOpError("expected a structured stencil or window result as source");
  int64_t index = op->getAttrOfType<IntegerAttr>("operand").getInt();
  if (index < 0 || index >= source->getNumOperands())
    return op->emitOpError("expected stencil operand number from 0 through 3");
  Type input = source->getOperand(index).getType();
  if (index == 2)
    input = VectorType::get({9}, Float64Type::get(op->getContext()));
  Type result = source->getResult(0).getType();
  if (op->getOperand(1).getType() != (reverse ? result : input) ||
      op->getResult(0).getType() != (reverse ? input : result))
    return op->emitOpError("seed or derivative result has the wrong operand shape");
  return success();
}
LogicalResult StencilJVPOp::verify() { return verifyRequest(*this, false); }
LogicalResult StencilVJPOp::verify() { return verifyRequest(*this, true); }
