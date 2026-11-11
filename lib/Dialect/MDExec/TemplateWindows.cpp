// Checked execution form of the proved, image-aware template reduction.
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mdir/Dialect/MD/MDTemplateWindow.h"
#include "mlir/IR/Builders.h"
#include <cmath>

using namespace mlir;
using namespace mdir::md_exec;

namespace {
static LogicalResult bounds(Operation *op) {
  for (StringRef name : {"cutoff", "extent", "alpha"}) {
    double value = op->getAttrOfType<FloatAttr>(name).getValueAsDouble();
    if (!std::isfinite(value) || value < 0.0 || (name != "alpha" && value == 0.0))
      return op->emitOpError("expected positive finite cutoff/extent and finite nonnegative alpha");
  }
  return success();
}
template <typename Op>
static LogicalResult anchors(Op op, bool storage) {
  if (bool(op.getAnchored()) != bool(op.getRadius()))
    return op.emitOpError("expected solute anchors together with their radius");
  if (!op.getAnchored()) return success();
  double radius = op.getRadius()->convertToDouble();
  if (!(radius > 0.0) || !std::isfinite(radius))
    return op.emitOpError("expected a positive finite anchor radius");
  if (storage != isa<MemRefType>(op.getAnchored().getType()))
    return op.emitOpError("expected fields/relations together or buffers together");
  if (storage) {
    auto type = cast<MemRefType>(op.getAnchored().getType());
    if (type.getRank() != 2 || type.getDimSize(1) != 2)
      return op.emitOpError("expected (member, anchor) rows");
  } else {
    auto type = cast<mdir::md::RelationType>(op.getAnchored().getType());
    if (type.getArity() != 2 || type.getOrientation() != mdir::md::Orientation::Ordered)
      return op.emitOpError("expected ordered (member, anchor) pairs");
  }
  return success();
}
template <typename Effect>
static void effect(SmallVectorImpl<MemoryEffects::EffectInstance> &effects,
                   OpOperand &operand) {
  effects.emplace_back(Effect::get(), &operand, 0, true,
                       SideEffects::DefaultResource::get());
}
} // namespace

LogicalResult BuildSegmentedPairsOp::verify() {
  if (failed(bounds(*this))) return failure();
  double skin = getSkin().convertToDouble();
  if (!std::isfinite(skin) || skin < 0.0)
    return emitOpError("expected a finite nonnegative skin");
  bool storage = isa<MemRefType>(getPositions().getType());
  if (storage != isa<MemRefType>(getUniverse().getType()) ||
      storage != isa<MemRefType>(getMolecules().getType()) ||
      (getExcluded() && storage != isa<MemRefType>(getExcluded().getType())))
    return emitOpError("expected fields/relations together or buffers together");
  unsigned arity;
  if (storage) arity = cast<MemRefType>(getMolecules().getType()).getDimSize(1);
  else {
    auto molecule = cast<mdir::md::RelationType>(getMolecules().getType());
    auto set = cast<mdir::mdrt::SegmentedPairsType>(getResult().getType()).getParticleSet();
    if (molecule.getParticleSet() != set ||
        cast<mdir::md::FieldType>(getPositions().getType()).getParticleSet() != set ||
        cast<mdir::md::FieldType>(getUniverse().getType()).getParticleSet() != set)
      return emitOpError("expected all template inputs on the list's particle set");
    arity = molecule.getArity();
  }
  if (arity != 3 && arity != 4)
    return emitOpError("expected three- or four-member template molecules");
  if (!getKernelValueType(getUniverse().getType()).isInteger(32))
    return emitOpError("expected a scalar i32 universe");
  if (failed(anchors(*this, storage))) return failure();
  auto positions = dyn_cast<VectorType>(getKernelValueType(getPositions().getType()));
  if (!positions || positions.getShape() != ArrayRef<int64_t>{3} ||
      (!positions.getElementType().isF32() && !positions.getElementType().isF64()))
    return emitOpError("expected three-component positions");
  return success();
}

LogicalResult EmptySegmentedPairsOp::verify() {
  if (failed(bounds(*this))) return failure();
  double skin = getSkin().convertToDouble();
  if (!std::isfinite(skin) || skin < 0.0)
    return emitOpError("expected a finite nonnegative skin");
  bool storage = isa<MemRefType>(getUniverse().getType());
  if (storage != isa<MemRefType>(getMolecules().getType()) ||
      (getExcluded() && storage != isa<MemRefType>(getExcluded().getType())))
    return emitOpError("expected fields/relations together or buffers together");
  unsigned arity;
  if (storage)
    arity = cast<MemRefType>(getMolecules().getType()).getDimSize(1);
  else {
    auto molecule = cast<mdir::md::RelationType>(getMolecules().getType());
    auto set = cast<mdir::mdrt::SegmentedPairsType>(getResult().getType()).getParticleSet();
    if (molecule.getParticleSet() != set ||
        cast<mdir::md::FieldType>(getUniverse().getType()).getParticleSet() != set)
      return emitOpError("expected membership on the list's particle set");
    arity = molecule.getArity();
  }
  if (arity != 3 && arity != 4)
    return emitOpError("expected three- or four-member template molecules");
  if (!getKernelValueType(getUniverse().getType()).isInteger(32))
    return emitOpError("expected a scalar i32 universe");
  return anchors(*this, storage);
}

LogicalResult RefreshSegmentedPairsOp::verify() {
  if (failed(bounds(*this))) return failure();
  double skin = getSkin().convertToDouble();
  double prune = getPruneSkin().convertToDouble();
  if (!std::isfinite(skin) || !std::isfinite(prune) ||
      skin < 0.0 || prune < 0.0 || prune > skin)
    return emitOpError("expected finite margins with zero <= prune_skin <= skin");
  auto positions = dyn_cast<VectorType>(getKernelValueType(getPositions().getType()));
  if (!positions || positions.getShape() != ArrayRef<int64_t>{3} ||
      (!positions.getElementType().isF32() && !positions.getElementType().isF64()))
    return emitOpError("expected three-component positions");
  if (auto field = dyn_cast<mdir::md::FieldType>(getPositions().getType()))
    if (field.getParticleSet() !=
        cast<mdir::mdrt::SegmentedPairsType>(getList().getType()).getParticleSet())
      return emitOpError("expected positions on the list's particle set");
  return success();
}

void EmptySegmentedPairsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  if (isa<MemRefType>(getUniverse().getType())) {
    effect<MemoryEffects::Read>(effects, getUniverseMutable());
    effect<MemoryEffects::Read>(effects, getMoleculesMutable());
    if (getExcluded()) effect<MemoryEffects::Read>(effects, getExcludedMutable()[0]);
    if (getAnchored()) effect<MemoryEffects::Read>(effects, getAnchoredMutable()[0]);
    effects.emplace_back(MemoryEffects::Allocate::get(),
                         getOperation()->getOpResults().front(), 0, true,
                         SideEffects::DefaultResource::get());
  }
}

void RefreshSegmentedPairsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  if (!isa<MemRefType>(getPositions().getType())) return;
  effect<MemoryEffects::Read>(effects, getListMutable());
  effect<MemoryEffects::Write>(effects, getListMutable());
  effect<MemoryEffects::Read>(effects, getPositionsMutable());
}

LogicalResult TemplateForOp::verify() {
  if (failed(bounds(*this))) return failure();
  unsigned columns = getOuts().size() + getReduce().size();
  if (!columns || getPairs().size() != 3 * columns)
    return emitOpError("expected three rows of pair counts, one column per destination/reduction");
  if (Attribute layout = (*this)->getAttr(mdir::md::kTemplateSiteForces)) {
    auto flags = dyn_cast<DenseI32ArrayAttr>(layout);
    if (!flags || flags.size() != 3 * getOuts().size() ||
        llvm::any_of(flags.asArrayRef(), [](int32_t flag) { return flag != 0 && flag != 1; }))
      return emitOpError("expected three rows of binary site-force flags, one per destination");
  }
  for (int32_t count : getPairs())
    if (count < 0) return emitOpError("expected nonnegative site-pair counts");
  for (Value value : getIns())
    if (isa<MemRefType>(value.getType()) != isStorageForm())
      return emitOpError("expected inputs in the positions' storage form");
  for (Value value : getOuts())
    if (isa<MemRefType>(value.getType()) != isStorageForm())
      return emitOpError("expected destinations in the positions' storage form");
  SmallVector<Type> results;
  if (!isStorageForm())
    for (Value value : getOuts()) results.push_back(value.getType());
  llvm::append_range(results, getReduce().getTypes());
  if (!llvm::equal(results, getOperation()->getResultTypes()))
    return emitOpError("expected destination results in value form, then reduction results");
  return success();
}

LogicalResult TemplateForOp::verifyRegions() {
  constexpr unsigned arities[] = {6, 4, 2};
  unsigned columns = getOuts().size() + getReduce().size();
  if (getPairs().size() != 3 * columns)
    return emitOpError("expected three rows of site-pair counts");
  for (unsigned tag = 0; tag != 3; ++tag) {
    Block &block = getOperation()->getRegion(tag).front();
    unsigned coordinateCount = mdir::md::getTemplateWindowMembers(arities[tag]).size();
    if (block.getNumArguments() != coordinateCount + arities[tag] * getIns().size())
      return emitOpError("expected class displacements then gathered member values");
    auto vector = dyn_cast<VectorType>(block.getArgument(0).getType());
    if (!vector || vector.getShape() != ArrayRef<int64_t>{3} ||
        (!vector.getElementType().isF32() && !vector.getElementType().isF64()))
      return emitOpError("expected three-component real class displacements");
    for (unsigned i = 0; i != coordinateCount; ++i)
      if (block.getArgument(i).getType() != vector)
        return emitOpError("expected class displacements of one real type");
    unsigned argument = coordinateCount;
    for (Value field : getIns())
      for (unsigned member = 0; member != arities[tag]; ++member)
        if (block.getArgument(argument++).getType() != getKernelValueType(field.getType()))
          return emitOpError("gathered member argument type disagrees with its field");
    SmallVector<Type> expected;
    for (unsigned column = 0; column != columns; ++column) {
      unsigned pairs = getPairs()[tag * columns + column];
      bool scatter = column < getOuts().size();
      Type type = scatter ? getKernelValueType(getOuts()[column].getType())
                          : getReduce()[column - getOuts().size()].getType();
      expected.append(pairs * (scatter ? mdir::md::getTemplateForceArity(
          *this, arities[tag], column, getOuts().size()) : 1), type);
      expected.append(pairs, vector);
    }
    auto yield = dyn_cast<YieldOp>(block.getTerminator());
    if (!yield || !llvm::equal(expected, yield.getOperandTypes()))
      return emitOpError("expected site contributions and displacements in destination/reduction order");
  }
  return success();
}

void TemplateForOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  if (!isStorageForm()) return;
  effect<MemoryEffects::Read>(effects, getListMutable());
  effect<MemoryEffects::Read>(effects, getPositionsMutable());
  for (OpOperand &operand : getInsMutable()) effect<MemoryEffects::Read>(effects, operand);
  for (OpOperand &operand : getOutsMutable()) {
    effect<MemoryEffects::Read>(effects, operand);
    effect<MemoryEffects::Write>(effects, operand);
  }
  for (OpOperand &operand : getScratchMutable()) effect<MemoryEffects::Write>(effects, operand);
}
