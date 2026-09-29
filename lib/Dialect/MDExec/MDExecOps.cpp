// Ops of the md_exec dialect.

#include "mdir/Dialect/MDExec/MDExecOps.h"

#include "mdir/Dialect/MDExec/MDExecDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mdir::md_exec;
using mdir::md::FieldType;
using mdir::mdrt::CellsType;
using mdir::mdrt::NeighborsType;
using mdir::mdrt::PermutationType;

#include "mdir/Dialect/MDExec/MDExecEnums.cpp.inc"

#define GET_OP_CLASSES
#include "mdir/Dialect/MDExec/MDExecOps.cpp.inc"

/// Returns true if `type` is a position field: three components of f64.
static bool isPositionField(Type type) {
  auto field = dyn_cast<FieldType>(type);
  return field && field.getNumComponents() == 3 &&
         field.getElementType().isF64();
}

/// Returns true if `type` is f64 or a fixed-size one-dimensional vector of
/// f64.
static bool isRealOrRealVector(Type type) {
  if (type.isF64())
    return true;
  auto vector = dyn_cast<VectorType>(type);
  return vector && !vector.isScalable() && vector.getRank() == 1 &&
         vector.getElementType().isF64();
}

/// Verifies that `positions` is a position field on `particleSet`.
static LogicalResult verifyPositions(Operation *op, Value positions,
                                     FlatSymbolRefAttr particleSet) {
  if (!isPositionField(positions.getType()))
    return op->emitOpError() << "expected a position field with 3 "
                                "components of f64, got "
                             << positions.getType();
  auto field = cast<FieldType>(positions.getType());
  if (field.getParticleSet() != particleSet)
    return op->emitOpError()
           << "the structure is on " << particleSet
           << ", but the positions belong to " << field.getParticleSet();
  return success();
}

//===----------------------------------------------------------------------===//
// Spatial structures
//===----------------------------------------------------------------------===//

LogicalResult BuildCellsOp::verify() {
  auto cells = cast<CellsType>(getResult().getType());
  if (failed(verifyPositions(getOperation(), getPositions(),
                             cells.getParticleSet())))
    return failure();
  double width = getWidth().convertToDouble();
  if (!(width > 0.0))
    return emitOpError() << "expected a positive width, got " << width;
  return success();
}

LogicalResult SpatialOrderOp::verify() {
  auto cells = cast<CellsType>(getCells().getType());
  auto order = cast<PermutationType>(getResult().getType());
  if (cells.getParticleSet() != order.getParticleSet())
    return emitOpError() << "the cells are on " << cells.getParticleSet()
                         << ", but the result is on "
                         << order.getParticleSet();
  return success();
}

LogicalResult PermuteOp::verify() {
  auto field = cast<FieldType>(getField().getType());
  auto order = cast<PermutationType>(getOrder().getType());
  if (field.getParticleSet() != order.getParticleSet())
    return emitOpError() << "the order is on " << order.getParticleSet()
                         << ", but the field belongs to "
                         << field.getParticleSet();
  return success();
}

LogicalResult BuildNeighborsOp::verify() {
  auto cells = cast<CellsType>(getCells().getType());
  auto neighbors = cast<NeighborsType>(getResult().getType());
  if (cells.getParticleSet() != neighbors.getParticleSet())
    return emitOpError() << "the cells are on " << cells.getParticleSet()
                         << ", but the result is on "
                         << neighbors.getParticleSet();
  if (failed(verifyPositions(getOperation(), getPositions(),
                             neighbors.getParticleSet())))
    return failure();

  double cutoff = getCutoff().convertToDouble();
  double skin = getSkin().convertToDouble();
  if (!(cutoff > 0.0))
    return emitOpError() << "expected a positive cutoff, got " << cutoff;
  if (!(skin >= 0.0))
    return emitOpError() << "expected a skin that is not negative, got "
                         << skin;
  if (getWidth() <= 0)
    return emitOpError() << "expected a positive width, got " << getWidth();

  // A cell that is narrower than the extended cutoff would hide neighbors
  // beyond the 27 surrounding cells.
  if (auto build = getCells().getDefiningOp<BuildCellsOp>()) {
    double width = build.getWidth().convertToDouble();
    if (width < cutoff + skin)
      return emitOpError()
             << "the cells are " << width
             << " wide, which is less than the cutoff plus the skin, "
             << cutoff + skin;
  }
  return success();
}

LogicalResult EmptyNeighborsOp::verify() {
  if (getWidth() <= 0)
    return emitOpError() << "expected a positive width, got " << getWidth();
  return success();
}

LogicalResult RefreshNeighborsOp::verify() {
  auto neighbors = cast<NeighborsType>(getNeighbors().getType());
  if (failed(verifyPositions(getOperation(), getPositions(),
                             neighbors.getParticleSet())))
    return failure();

  double cutoff = getCutoff().convertToDouble();
  double skin = getSkin().convertToDouble();
  double width = getCellWidth().convertToDouble();
  if (!(cutoff > 0.0))
    return emitOpError() << "expected a positive cutoff, got " << cutoff;
  if (!(skin >= 0.0))
    return emitOpError() << "expected a skin that is not negative, got "
                         << skin;
  if (width < cutoff + skin)
    return emitOpError()
           << "the cells are " << width
           << " wide, which is less than the cutoff plus the skin, "
           << cutoff + skin;
  return success();
}

//===----------------------------------------------------------------------===//
// Loops
//===----------------------------------------------------------------------===//

/// Verifies what the two loops have in common.
///
/// `leading` holds the types of the kernel arguments that come before those
/// of the fields in `ins`. `perField` is the number of kernel arguments for
/// each field in `ins`.
template <typename OpTy>
static LogicalResult verifyLoop(OpTy op, FlatSymbolRefAttr particleSet,
                                ArrayRef<Type> leading, unsigned perField) {
  // Fields.
  for (Value field : llvm::concat<Value>(op.getIns(), op.getOuts())) {
    auto type = cast<FieldType>(field.getType());
    if (type.getParticleSet() != particleSet)
      return op.emitOpError() << "expected all fields to belong to "
                              << particleSet << ", but one belongs to "
                              << type.getParticleSet();
  }
  for (Value value : op.getReduce())
    if (!isRealOrRealVector(value.getType()))
      return op.emitOpError() << "expected a value in 'reduce' to be f64 or "
                                 "a fixed-size vector of f64, got "
                              << value.getType();

  // Results: one per field in `outs`, then one per value in `reduce`.
  SmallVector<Type> expected;
  for (Value field : op.getOuts())
    expected.push_back(field.getType());
  for (Value value : op.getReduce())
    expected.push_back(value.getType());
  if (expected.empty())
    return op.emitOpError()
           << "expected at least 1 field in 'outs' or value in 'reduce'";
  if (op.getNumResults() != expected.size())
    return op.emitOpError()
           << "expected " << expected.size()
           << " results, one per field in 'outs' and value in 'reduce', got "
           << op.getNumResults();
  for (unsigned i = 0, e = expected.size(); i != e; ++i)
    if (op.getResult(i).getType() != expected[i])
      return op.emitOpError()
             << "expected result " << i << " to have type " << expected[i]
             << ", got " << op.getResult(i).getType();
  return success();
}

template <typename OpTy>
static LogicalResult verifyKernel(OpTy op, ArrayRef<Type> leading,
                                  unsigned perField) {
  Block &block = op.getKernel().front();

  SmallVector<Type> arguments(leading.begin(), leading.end());
  for (Value field : op.getIns()) {
    Type value = cast<FieldType>(field.getType()).getKernelValueType();
    arguments.append(perField, value);
  }
  if (block.getNumArguments() != arguments.size())
    return op.emitOpError() << "expected the kernel to have "
                            << arguments.size() << " arguments, got "
                            << block.getNumArguments();
  for (unsigned i = 0, e = arguments.size(); i != e; ++i)
    if (block.getArgument(i).getType() != arguments[i])
      return op.emitOpError()
             << "expected kernel argument " << i << " to have type "
             << arguments[i] << ", got " << block.getArgument(i).getType();

  SmallVector<Type> yielded;
  for (Value field : op.getOuts())
    yielded.push_back(
        cast<FieldType>(field.getType()).getKernelValueType());
  for (Value value : op.getReduce())
    yielded.push_back(value.getType());

  auto yield = dyn_cast<YieldOp>(block.getTerminator());
  if (!yield)
    return op.emitOpError()
           << "expected the kernel to end with 'md_exec.yield'";
  if (yield.getNumOperands() != yielded.size())
    return yield.emitOpError() << "expected " << yielded.size()
                               << " values, got " << yield.getNumOperands();
  for (unsigned i = 0, e = yielded.size(); i != e; ++i)
    if (yield.getOperand(i).getType() != yielded[i])
      return yield.emitOpError()
             << "expected value " << i << " to have type " << yielded[i]
             << ", got " << yield.getOperand(i).getType();
  return success();
}

/// The types of the kernel arguments that a pair loop provides itself: the
/// squared distance and the displacement.
static SmallVector<Type, 2> getPairGeometryTypes(MLIRContext *context) {
  Type real = Float64Type::get(context);
  return {real, VectorType::get({3}, real)};
}

LogicalResult PairForOp::verify() {
  auto neighbors = cast<NeighborsType>(getNeighbors().getType());
  if (failed(verifyPositions(getOperation(), getPositions(),
                             neighbors.getParticleSet())))
    return failure();

  double cutoff = getCutoff().convertToDouble();
  if (!(cutoff > 0.0))
    return emitOpError() << "expected a positive cutoff, got " << cutoff;

  if (auto weights = getWeights())
    if (weights->size() != getReduce().size())
      return emitOpError() << "expected " << getReduce().size()
                           << " weights, one per value in 'reduce', got "
                           << weights->size();

  if (getTraversal() != Traversal::Directed ||
      getConflict() != Conflict::OwnerOnly)
    return emitOpError()
           << "only the policy (directed, owner_only) is supported";

  // The list must contain every pair within the cutoff of this loop.
  std::optional<double> built;
  if (auto build = getNeighbors().getDefiningOp<BuildNeighborsOp>())
    built = build.getCutoff().convertToDouble();
  else if (auto refresh = getNeighbors().getDefiningOp<RefreshNeighborsOp>())
    built = refresh.getCutoff().convertToDouble();
  if (built && cutoff > *built)
    return emitOpError()
           << "the cutoff, " << cutoff
           << ", exceeds the cutoff that the neighbor structure was "
              "built with, "
           << *built;

  return verifyLoop(*this, neighbors.getParticleSet(),
                    getPairGeometryTypes(getContext()), 2);
}

LogicalResult PairForOp::verifyRegions() {
  return verifyKernel(*this, getPairGeometryTypes(getContext()), 2);
}

LogicalResult ParticleForOp::verify() {
  FlatSymbolRefAttr particleSet;
  if (!getIns().empty())
    particleSet =
        cast<FieldType>(getIns().front().getType()).getParticleSet();
  else if (!getOuts().empty())
    particleSet =
        cast<FieldType>(getOuts().front().getType()).getParticleSet();
  else
    return emitOpError() << "expected at least 1 field in 'ins' or 'outs'";
  return verifyLoop(*this, particleSet, {}, 1);
}

LogicalResult ParticleForOp::verifyRegions() {
  return verifyKernel(*this, {}, 1);
}
