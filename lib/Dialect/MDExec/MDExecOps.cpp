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

/// Returns true if `type` is a floating-point type of the execution level.
static bool isReal(Type type) { return type.isF32() || type.isF64(); }

bool mdir::md_exec::isBufferType(Type type) {
  auto buffer = dyn_cast<MemRefType>(type);
  if (!buffer || buffer.getRank() < 1 || buffer.getRank() > 2 ||
      !buffer.isDynamicDim(0))
    return false;
  Type element = buffer.getElementType();
  if (buffer.getRank() == 2)
    return buffer.getDimSize(1) == 3 && isReal(element);
  return isReal(element) || element.isSignlessInteger(32) ||
         element.isSignlessInteger(64);
}

Type mdir::md_exec::getKernelValueType(Type fieldOrBuffer) {
  if (auto field = dyn_cast<FieldType>(fieldOrBuffer))
    return field.getKernelValueType();
  auto buffer = cast<MemRefType>(fieldOrBuffer);
  if (buffer.getRank() == 1)
    return buffer.getElementType();
  return VectorType::get({buffer.getDimSize(1)}, buffer.getElementType());
}

/// Returns true if `type` is the type of positions: a field with three
/// components of f32 or f64, or a buffer that holds one.
static bool isPositionType(Type type) {
  if (auto buffer = dyn_cast<MemRefType>(type))
    return mdir::md_exec::isBufferType(type) && buffer.getRank() == 2;
  auto field = dyn_cast<FieldType>(type);
  return field && field.getNumComponents() == 3 &&
         isReal(field.getElementType());
}

/// The memory effect `effect` on `operand`.
template <typename EffectTy>
static void addEffect(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects,
    OpOperand &operand) {
  effects.emplace_back(EffectTy::get(), &operand, /*stage=*/0,
                       /*effectOnFullRegion=*/true,
                       SideEffects::DefaultResource::get());
}

/// Returns true if `type` is f32, f64, or a fixed-size one-dimensional
/// vector of one of them.
static bool isRealOrRealVector(Type type) {
  if (isReal(type))
    return true;
  auto vector = dyn_cast<VectorType>(type);
  return vector && !vector.isScalable() && vector.getRank() == 1 &&
         isReal(vector.getElementType());
}

/// Verifies that `positions` is a position field on `particleSet`, or a
/// buffer that holds a position field.
static LogicalResult verifyPositions(Operation *op, Value positions,
                                     FlatSymbolRefAttr particleSet) {
  if (!isPositionType(positions.getType()))
    return op->emitOpError() << "expected a position field with 3 "
                                "components of f32 or f64, got "
                             << positions.getType();
  auto field = dyn_cast<FieldType>(positions.getType());
  if (field && field.getParticleSet() != particleSet)
    return op->emitOpError()
           << "the structure is on " << particleSet
           << ", but the positions belong to " << field.getParticleSet();
  return success();
}

/// Verifies the buffers in `scratch`: none, or `perValue` for each of the
/// `types`, with one value of that type per particle.
static LogicalResult verifyScratch(Operation *op, bool isStorage,
                                   ValueRange scratch, ArrayRef<Type> types,
                                   unsigned perValue) {
  if (scratch.empty())
    return success();
  if (!isStorage)
    return op->emitOpError() << "'scratch' belongs to the storage form";
  if (scratch.size() != perValue * types.size())
    return op->emitOpError()
           << "expected no buffers in 'scratch' or "
           << perValue * types.size() << ", got " << scratch.size();
  for (unsigned i = 0, e = scratch.size(); i != e; ++i) {
    auto buffer = cast<MemRefType>(scratch[i].getType());
    Type expected = types[i / perValue];
    if (buffer.getRank() != 1 || buffer.getElementType() != expected)
      return op->emitOpError()
             << "expected buffer " << i << " in 'scratch' to hold one "
             << expected << " per particle, got " << buffer;
  }
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
  return success();
}

LogicalResult EmptyNeighborsOp::verify() {
  if (getWidth() <= 0)
    return emitOpError() << "expected a positive width, got " << getWidth();
  if (static_cast<bool>(getSize()) != static_cast<bool>(getPositions()))
    return emitOpError() << "expected 'size' and 'positions' together";
  if (getPositions() && !isPositionType(*getPositions()))
    return emitOpError() << "expected the type of a buffer that holds "
                            "positions, got "
                         << *getPositions();
  return success();
}

void EmptyNeighborsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (isStorageForm())
    effects.emplace_back(MemoryEffects::Allocate::get(),
                         getOperation()->getOpResult(0), /*stage=*/0,
                         /*effectOnFullRegion=*/true,
                         SideEffects::DefaultResource::get());
}

void RefreshNeighborsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (!isStorageForm())
    return;
  addEffect<MemoryEffects::Read>(effects, getNeighborsMutable());
  addEffect<MemoryEffects::Write>(effects, getNeighborsMutable());
  addEffect<MemoryEffects::Read>(effects, getPositionsMutable());
  for (OpOperand &operand : getScratchMutable())
    addEffect<MemoryEffects::Write>(effects, operand);
}

LogicalResult ReferencePositionsOp::verify() {
  auto neighbors = cast<NeighborsType>(getNeighbors().getType());
  return verifyPositions(getOperation(), getResult(),
                         neighbors.getParticleSet());
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
  if (!(width > 0.0))
    return emitOpError() << "expected cells of a positive width, got "
                         << width;

  if (getMoved()) {
    if (getPolicy() != RebuildPolicy::Check)
      return emitOpError()
             << "'moved' belongs to the policy 'check': with the policy "
                "'always' the structure is built whatever has moved";
    if (!getScratch().empty())
      return emitOpError()
             << "expected no buffers in 'scratch': with 'moved' the op "
                "does not test the displacements";
  }

  Type real = isStorageForm()
                  ? cast<MemRefType>(getPositions().getType())
                        .getElementType()
                  : Type();
  return verifyScratch(getOperation(), isStorageForm(), getScratch(), {real},
                       2);
}

//===----------------------------------------------------------------------===//
// Loops
//===----------------------------------------------------------------------===//

/// Verifies what the two loops have in common. `allowsFlags` tells whether
/// a value in `reduce` may have the type i1.
template <typename OpTy>
static LogicalResult verifyLoop(OpTy op, FlatSymbolRefAttr particleSet,
                                bool allowsFlags) {
  bool isStorage = op.isStorageForm();
  for (Value field : llvm::concat<Value>(op.getIns(), op.getOuts())) {
    if (isa<MemRefType>(field.getType()) != isStorage)
      return op.emitOpError()
             << "expected fields only, as in the value form, or buffers "
                "only, as in the storage form";
    auto type = dyn_cast<FieldType>(field.getType());
    if (type && type.getParticleSet() != particleSet)
      return op.emitOpError() << "expected all fields to belong to "
                              << particleSet << ", but one belongs to "
                              << type.getParticleSet();
  }
  for (Value value : op.getReduce()) {
    Type type = value.getType();
    if (isRealOrRealVector(type) || (allowsFlags && type.isInteger(1)))
      continue;
    return op.emitOpError()
           << "expected a value in 'reduce' to be f32, f64, "
           << (allowsFlags ? "" : "or ")
           << "a fixed-size vector of one of them"
           << (allowsFlags ? ", or i1" : "") << ", got " << type;
  }
  if (op.getOuts().empty() && op.getReduce().empty())
    return op.emitOpError()
           << "expected at least 1 field in 'outs' or value in 'reduce'";

  // Only a sum needs buffers.
  SmallVector<Type> sums;
  for (Value value : op.getReduce())
    if (!value.getType().isInteger(1))
      sums.push_back(value.getType());
  if (failed(verifyScratch(op.getOperation(), isStorage, op.getScratch(),
                           sums, 2)))
    return failure();

  // Results: one per field in `outs`, then one per value in `reduce`. A
  // buffer in `outs` is updated where it is and has no result.
  SmallVector<Type> expected;
  if (!isStorage)
    for (Value field : op.getOuts())
      expected.push_back(field.getType());
  for (Value value : op.getReduce())
    expected.push_back(value.getType());
  if (op.getNumResults() != expected.size())
    return op.emitOpError()
           << "expected " << expected.size() << " results, "
           << (isStorage ? "one per value in 'reduce'"
                         : "one per field in 'outs' and value in 'reduce'")
           << ", got " << op.getNumResults();
  for (unsigned i = 0, e = expected.size(); i != e; ++i)
    if (op.getResult(i).getType() != expected[i])
      return op.emitOpError()
             << "expected result " << i << " to have type " << expected[i]
             << ", got " << op.getResult(i).getType();
  return success();
}

/// The memory effects of a loop in the storage form. `readsOuts` tells
/// whether the loop reads what destination `index` holds.
template <typename OpTy>
static void getLoopEffects(
    OpTy op,
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects,
    function_ref<bool(unsigned)> readsOuts) {
  if (!op.isStorageForm())
    return;
  for (OpOperand &operand : op.getInsMutable())
    addEffect<MemoryEffects::Read>(effects, operand);
  for (auto [index, operand] : llvm::enumerate(op.getOutsMutable())) {
    if (readsOuts(index))
      addEffect<MemoryEffects::Read>(effects, operand);
    addEffect<MemoryEffects::Write>(effects, operand);
  }
  for (OpOperand &operand : op.getScratchMutable())
    addEffect<MemoryEffects::Write>(effects, operand);
}

/// Verifies the kernel of a loop.
///
/// `leading` holds the types of the kernel arguments that come before those
/// of the fields in `ins`. `perField` is the number of kernel arguments for
/// each field in `ins`.
template <typename OpTy>
static LogicalResult verifyKernel(OpTy op, ArrayRef<Type> leading,
                                  unsigned perField) {
  Block &block = op.getKernel().front();

  SmallVector<Type> arguments(leading.begin(), leading.end());
  for (Value field : op.getIns())
    arguments.append(perField,
                     mdir::md_exec::getKernelValueType(field.getType()));
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
    yielded.push_back(mdir::md_exec::getKernelValueType(field.getType()));
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
/// squared distance and the displacement. They have the type that the
/// kernel computes in, which the first argument states.
static SmallVector<Type, 2> getPairGeometryTypes(PairForOp op) {
  Type real = Float64Type::get(op.getContext());
  Block &block = op.getKernel().front();
  if (block.getNumArguments() != 0 && block.getArgument(0).getType().isF32())
    real = block.getArgument(0).getType();
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

  if (auto overwrite = getOverwrite()) {
    if (!isStorageForm())
      return emitOpError() << "'overwrite' belongs to the storage form; in "
                              "the value form the destination tells what "
                              "the loop adds to";
    if (overwrite->size() != getOuts().size())
      return emitOpError() << "expected " << getOuts().size()
                           << " flags in 'overwrite', one per buffer in "
                              "'outs', got "
                           << overwrite->size();
  }

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
                    /*allowsFlags=*/false);
}

LogicalResult PairForOp::verifyRegions() {
  return verifyKernel(*this, getPairGeometryTypes(*this), 2);
}

void PairForOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (!isStorageForm())
    return;
  addEffect<MemoryEffects::Read>(effects, getNeighborsMutable());
  addEffect<MemoryEffects::Read>(effects, getPositionsMutable());
  getLoopEffects(*this, effects,
                 [&](unsigned index) { return !overwrites(index); });
}

LogicalResult ParticleForOp::verify() {
  if (getIns().empty() && getOuts().empty())
    return emitOpError() << "expected at least 1 field in 'ins' or 'outs'";
  Value any = getIns().empty() ? getOuts().front() : getIns().front();
  FlatSymbolRefAttr particleSet;
  if (auto field = dyn_cast<FieldType>(any.getType()))
    particleSet = field.getParticleSet();
  return verifyLoop(*this, particleSet, /*allowsFlags=*/true);
}

LogicalResult ParticleForOp::verifyRegions() {
  return verifyKernel(*this, {}, 1);
}

void ParticleForOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getLoopEffects(*this, effects, [](unsigned) { return false; });
}
