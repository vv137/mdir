// Structural assignment and interpolation contracts (Essmann et al., 1995).
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;
using namespace mdir::md;

LogicalResult mdir::md::verifyStencilWindowActivityAttribute(
    Operation *op, Attribute value) {
  if (!isa<PotentialOp>(op) || !isa<LocationAttr>(value))
    return op->emitError(
        "expected stencil window diagnostic provenance as a location on md.potential");
  return success();
}

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
          (!arg && operand.getDefiningOp()->getBlock() != &body)) {
        Attribute literal;
        if (matchPattern(operand, m_Constant(&literal)) &&
            isMemoryEffectFree(operand.getDefiningOp()))
          continue;
        return op->emitOpError(
            "window kernel cannot capture values; continuous inputs must be explicit");
      }
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

static bool stencilValue(Type type) {
  if (type.isF64())
    return true;
  auto vector = dyn_cast<VectorType>(type);
  return vector && vector.getRank() == 1 && !vector.isScalable() &&
         vector.getElementType().isF64();
}

LogicalResult ReduceStencilOp::verify() {
  auto relation = getStencil().getDefiningOp<GridStencilOp>();
  if (!relation || relation.getPositions() != getPositions() ||
      relation.getCell() != getCell())
    return emitOpError("expected the relation constructor's exact positions and cell operands");
  auto stencil = cast<GridStencilType>(getStencil().getType());
  auto shape = stencil.getShape().asArrayRef();
  for (Value particle : getParticles()) {
    auto field = cast<FieldType>(particle.getType());
    if (field.getParticleSet() != stencil.getParticleSet() ||
        !field.getElementType().isF64())
      return emitOpError("particle inputs must be f64 fields of the stencil's particle set");
  }
  for (Value value : getGrids()) {
    auto grid = cast<RankedTensorType>(value.getType());
    if (grid.getShape() != shape || !stencilValue(grid.getElementType()))
      return emitOpError("grid inputs must have the complete real stencil shape");
  }
  for (Value value : getUniforms())
    if (!stencilValue(value.getType()))
      return emitOpError("uniform inputs must be explicit real f64 scalars or vectors");
  Type result = getResult().getType();
  if (getDestination() == "grid") {
    auto grid = dyn_cast<RankedTensorType>(result);
    if (!grid || grid.getShape() != shape || !stencilValue(grid.getElementType()))
      return emitOpError("grid destination requires a complete real stencil-shaped tensor");
  } else if (getDestination() == "particles") {
    auto field = dyn_cast<FieldType>(result);
    if (!field || field.getParticleSet() != stencil.getParticleSet() ||
        !field.getElementType().isF64())
      return emitOpError("particle destination requires a complete f64 field of the stencil's set");
  } else if (getDestination() == "sum") {
    if (!stencilValue(result))
      return emitOpError("sum destination requires a real f64 scalar or vector");
  } else {
    return emitOpError("expected destination grid, particles or sum");
  }
  if (getOperation()->hasAttr("listed") || getOperation()->hasAttr("coverage") ||
      getOperation()->hasAttr("algebra"))
    return emitOpError("full coverage and ideal additive algebra cannot be overridden");
  return success();
}

LogicalResult ReduceStencilOp::verifyRegions() {
  Block &block = getKernel().front();
  SmallVector<Type> arguments{
      VectorType::get({3}, Float64Type::get(getContext())),
      VectorType::get({3}, Float64Type::get(getContext())),
      VectorType::get({9}, Float64Type::get(getContext()))};
  for (Value field : getParticles())
    arguments.push_back(cast<FieldType>(field.getType()).getKernelValueType());
  for (Value grid : getGrids())
    arguments.push_back(cast<RankedTensorType>(grid.getType()).getElementType());
  for (Value uniform : getUniforms())
    arguments.push_back(uniform.getType());
  if (block.getArgumentTypes() != TypeRange(arguments))
    return emitOpError("expected distance, position, row-cell matrix and explicit input kernel arguments");
  Type result = getResult().getType();
  if (auto field = dyn_cast<FieldType>(result))
    result = field.getKernelValueType();
  else if (auto tensor = dyn_cast<RankedTensorType>(result))
    result = tensor.getElementType();
  auto yield = dyn_cast<YieldOp>(block.getTerminator());
  if (!yield || yield.getValues().size() != 1 ||
      yield.getValues()[0].getType() != result)
    return emitOpError("expected one kernel value matching the destination element");
  for (Operation &inner : block) {
    if (inner.getNumRegions() || !isMemoryEffectFree(&inner))
      return emitOpError("stencil reduction kernel must be straight-line and effect-free");
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
      return emitOpError("stencil reduction kernel cannot capture dynamic values");
    }
    if (auto window = dyn_cast<BSplineWindowOp>(inner))
      if (window.getOrder() != cast<GridStencilType>(getStencil().getType()).getOrder())
        return emitOpError("window and stencil support orders must agree");
  }
  return success();
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
