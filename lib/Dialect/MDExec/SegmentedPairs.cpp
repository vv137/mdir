// The lists of units in the segments of rigid kinds, and the loop over them
// (D417).

#include "mdir/Dialect/MDExec/Algebra.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mdir/Dialect/MDExec/NeighborContract.h"
#include "mdir/Dialect/MDRT/SegmentRegions.h"

#include "mlir/IR/Builders.h"
#include <cmath>

using namespace mlir;
using namespace mdir::md_exec;
using mdir::md::FieldType;
using mdir::md::RelationType;
using mdir::mdrt::SegmentedPairsType;
using Contract = mdir::mdrt::NeighborsContract;
namespace mdrt = mdir::mdrt;

Contract mdir::md_exec::getSegmentGuarantee() {
  Contract guarantee;
  guarantee.traversal = Contract::Traversal::Unique;
  guarantee.images = Contract::Images::Minimum;
  guarantee.exclusions = Contract::Exclusions::Removed;
  return Contract::withOrganization(guarantee,
                                    Contract::Organization::GroupedBy,
                                    Contract::Group::Group,
                                    Contract::Order::None);
}

SegmentedPairsType mdir::md_exec::getSegmentedBuildType(
    FlatSymbolRefAttr particleSet, ArrayAttr kinds) {
  unsigned k = kinds.size();
  SmallVector<Contract> segments((k + 1) * (k + 2) / 2,
                                 getSegmentGuarantee());
  return SegmentedPairsType::get(particleSet.getContext(), particleSet, kinds,
                                 segments);
}

Contract mdir::md_exec::getSegmentRequirement(Operation *loop) {
  Contract required;
  required.traversal = Contract::Traversal::Unique;
  required.images = Contract::Images::Minimum;
  required.exclusions = Contract::Exclusions::Removed;
  std::optional<SmallVector<Algebra>> algebras = algebra::getAlgebras(loop);
  if (algebras && !algebra::allowReorder(*algebras))
    required = Contract::withOrganization(required,
                                          Contract::Organization::Stable,
                                          Contract::Group::Group,
                                          Contract::Order::Search);
  return required;
}

namespace {
template <typename Effect>
void addEffect(SmallVectorImpl<MemoryEffects::EffectInstance> &effects,
               OpOperand &operand) {
  effects.emplace_back(Effect::get(), &operand, 0, true,
                       SideEffects::DefaultResource::get());
}

/// The particle set of a field or of a relation, or nothing for a buffer.
FlatSymbolRefAttr getSet(Type type) {
  if (auto field = dyn_cast<FieldType>(type))
    return field.getParticleSet();
  if (auto relation = dyn_cast<RelationType>(type))
    return relation.getParticleSet();
  return {};
}

/// The arity of a relation, or the width of its buffer of members.
int64_t getArity(Type type) {
  if (auto relation = dyn_cast<RelationType>(type))
    return relation.getArity();
  return cast<MemRefType>(type).getDimSize(1);
}

/// What the three ops of a list have in common: the reaches, the kinds
/// against the type, the anchors, one form, and the guarantee.
template <typename Op>
LogicalResult verifyList(Op op, Value positions) {
  double cutoff = op.getCutoff().convertToDouble();
  double skin = op.getSkin().convertToDouble();
  if (!std::isfinite(cutoff) || !(cutoff > 0.0))
    return op.emitOpError() << "expected a positive finite cutoff";
  if (!std::isfinite(skin) || skin < 0.0)
    return op.emitOpError() << "expected a finite nonnegative skin";
  auto type = cast<SegmentedPairsType>(op.getResult().getType());
  if (op.getKinds().size() != type.getNumKinds())
    return op.emitOpError() << "takes " << op.getKinds().size()
                            << " kinds, but its type has "
                            << type.getNumKinds();
  bool storage = isa<MemRefType>(op.getUniverse().getType());
  SmallVector<Value> values = {op.getUniverse()};
  llvm::append_range(values, op.getKinds());
  if (op.getExcluded())
    values.push_back(op.getExcluded());
  if (op.getAnchored())
    values.push_back(op.getAnchored());
  if (positions)
    values.push_back(positions);
  for (Value value : values) {
    if (isa<MemRefType>(value.getType()) != storage)
      return op.emitOpError() << "expected fields and relations only, as in "
                                 "the value form, or buffers only, as in the "
                                 "storage form";
    if (FlatSymbolRefAttr set = getSet(value.getType()))
      if (set != type.getParticleSet())
        return op.emitOpError() << "expected every operand on "
                                << type.getParticleSet() << ", got one on "
                                << set;
  }
  if (!getKernelValueType(op.getUniverse().getType()).isInteger(32))
    return op.emitOpError() << "expected a scalar i32 universe";
  for (auto [f, kind] : llvm::enumerate(op.getKinds())) {
    int64_t size = type.getKind(f).getSize();
    if (getArity(kind.getType()) != size)
      return op.emitOpError() << "takes kind " << f << " with "
                              << getArity(kind.getType())
                              << " members a copy, but its descriptor has "
                              << size << " places";
  }
  if (op.getExcluded() && getArity(op.getExcluded().getType()) != 2)
    return op.emitOpError() << "expected excluded pairs";
  if (bool(op.getAnchored()) != bool(op.getRadius()))
    return op.emitOpError() << "expected the anchors with their radius";
  if (op.getAnchored()) {
    double radius = op.getRadius()->convertToDouble();
    if (!std::isfinite(radius) || !(radius > 0.0))
      return op.emitOpError() << "expected a positive finite radius";
    if (getArity(op.getAnchored().getType()) != 2)
      return op.emitOpError() << "expected (member, anchor) pairs";
  }
  return verifyGuarantee(op, type, getSegmentGuarantee());
}
} // namespace

LogicalResult BuildSegmentedPairsOp::verify() {
  return verifyList(*this, getPositions());
}

LogicalResult EmptySegmentedPairsOp::verify() {
  return verifyList(*this, Value());
}

void EmptySegmentedPairsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  // The storage of a list is allocated where it begins (as
  // md_exec.empty_neighbors).
  if (!isa<MemRefType>(getUniverse().getType()))
    return;
  addEffect<MemoryEffects::Read>(effects, getUniverseMutable());
  for (OpOperand &operand : getKindsMutable())
    addEffect<MemoryEffects::Read>(effects, operand);
  for (OpOperand &operand : getExcludedMutable())
    addEffect<MemoryEffects::Read>(effects, operand);
  for (OpOperand &operand : getAnchoredMutable())
    addEffect<MemoryEffects::Read>(effects, operand);
  effects.emplace_back(MemoryEffects::Allocate::get(),
                       getOperation()->getOpResult(0), 0, true,
                       SideEffects::DefaultResource::get());
}

LogicalResult RefreshSegmentedPairsOp::verify() {
  double cutoff = getCutoff().convertToDouble();
  double skin = getSkin().convertToDouble();
  double prune = getPruneSkin().convertToDouble();
  if (!std::isfinite(cutoff) || !(cutoff > 0.0))
    return emitOpError() << "expected a positive finite cutoff";
  if (!std::isfinite(skin) || !std::isfinite(prune) || skin < 0.0 ||
      prune < 0.0 || prune > skin)
    return emitOpError() << "expected finite skins with 0 <= prune_skin <= "
                            "skin";
  Type positions = getPositions().getType();
  if (auto field = dyn_cast<FieldType>(positions))
    if (field.getParticleSet() != getList().getType().getParticleSet())
      return emitOpError() << "expected the positions on the list's set";
  auto vector = dyn_cast<VectorType>(getKernelValueType(positions));
  if (!vector || vector.getShape() != ArrayRef<int64_t>{3} ||
      !isa<FloatType>(vector.getElementType()))
    return emitOpError() << "expected positions of 3 components";
  if (getStale() && !getMoved())
    return emitOpError() << "takes 'stale' without 'moved'";
  if (Value moved = getMoved()) {
    // The test is the writer's: at this refresh's positions, cell, list, and
    // reaches (D422).
    auto writer = moved.getDefiningOp<WriteUnitFramesOp>();
    if (!writer || writer.getMoved() != moved ||
        (getStale() && getStale() != writer.getStale()))
      return emitOpError() << "takes 'moved' and 'stale' from the writer of "
                              "its list (md_exec.write_unit_frames) alone";
    if (writer.getList() != getList() ||
        writer.getPositions() != getPositions() ||
        writer.getCell() != getCell())
      return emitOpError() << "takes the test of a writer of another list, "
                              "positions, or cell";
    if (writer.getCutoffAttr() != getCutoffAttr() ||
        writer.getSkinAttr() != getSkinAttr() ||
        writer.getPruneSkinAttr() != getPruneSkinAttr())
      return emitOpError() << "takes the test of a writer of other reaches";
  }
  return success();
}

void RefreshSegmentedPairsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  if (!isa<MemRefType>(getPositions().getType()))
    return;
  addEffect<MemoryEffects::Read>(effects, getListMutable());
  addEffect<MemoryEffects::Write>(effects, getListMutable());
  addEffect<MemoryEffects::Read>(effects, getPositionsMutable());
}

LogicalResult WriteUnitFramesOp::verify() {
  double cutoff = getCutoff().convertToDouble();
  double skin = getSkin().convertToDouble();
  double prune = getPruneSkin().convertToDouble();
  if (!std::isfinite(cutoff) || !(cutoff > 0.0))
    return emitOpError() << "expected a positive finite cutoff";
  if (!std::isfinite(skin) || !std::isfinite(prune) || skin < 0.0 ||
      prune < 0.0 || prune > skin)
    return emitOpError() << "expected finite skins with 0 <= prune_skin <= "
                            "skin";
  Type positions = getPositions().getType();
  if (auto field = dyn_cast<FieldType>(positions))
    if (field.getParticleSet() != getList().getType().getParticleSet())
      return emitOpError() << "expected the positions on the list's set";
  auto vector = dyn_cast<VectorType>(getKernelValueType(positions));
  if (!vector || vector.getShape() != ArrayRef<int64_t>{3} ||
      !vector.getElementType().isF64())
    return emitOpError() << "expected positions of 3 components in f64, "
                            "from which the frames are formed";
  return success();
}

void WriteUnitFramesOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  // In the value form the frames are the list's own state, which only the
  // loops over the list read, after the refresh that takes the results.
  if (!isa<MemRefType>(getPositions().getType()))
    return;
  addEffect<MemoryEffects::Read>(effects, getListMutable());
  addEffect<MemoryEffects::Write>(effects, getListMutable());
  addEffect<MemoryEffects::Read>(effects, getPositionsMutable());
}

//===----------------------------------------------------------------------===//
// md_exec.segment_for
//===----------------------------------------------------------------------===//

std::pair<unsigned, unsigned> SegmentForOp::getSegmentKinds(unsigned index) {
  return mdrt::getSegmentKinds(getSegments(), index);
}

SmallVector<SmallVector<int64_t>> SegmentForOp::getSitePairs(unsigned index) {
  SmallVector<SmallVector<int64_t>> rows;
  for (const mdrt::SitePair &pair : mdrt::getSitePairs(getSites(), index)) {
    SmallVector<int64_t> row = {pair.p, pair.q};
    llvm::append_range(row, pair.values);
    rows.push_back(row);
  }
  return rows;
}

mdir::md::Exchange SegmentForOp::getExchange(unsigned index) {
  std::optional<ArrayAttr> list = getExchange();
  if (!list || index >= list->size())
    return mdir::md::Exchange::None;
  return cast<mdir::md::ExchangeAttr>((*list)[index]).getValue();
}

mdrt::SegmentSignature mdir::md_exec::getSegmentSignature(SegmentForOp loop) {
  mdrt::SegmentSignature signature;
  for (Value field : loop.getIns())
    signature.fields.push_back(getKernelValueType(field.getType()));
  for (Value out : loop.getOuts())
    signature.values.push_back(getKernelValueType(out.getType()));
  for (Value value : loop.getReduce())
    signature.values.push_back(value.getType());
  signature.terminator = YieldOp::getOperationName();
  return signature;
}

LogicalResult SegmentForOp::verify() {
  SegmentedPairsType type = getListType();
  bool storage = isStorageForm();
  if (auto field = dyn_cast<FieldType>(getPositions().getType()))
    if (field.getParticleSet() != type.getParticleSet())
      return emitOpError() << "expected the positions on the list's set";
  for (Value value : llvm::concat<Value>(getIns(), getOuts())) {
    if (isa<MemRefType>(value.getType()) != storage)
      return emitOpError() << "expected fields only, as in the value form, "
                              "or buffers only, as in the storage form";
    if (auto field = dyn_cast<FieldType>(value.getType()))
      if (field.getParticleSet() != type.getParticleSet())
        return emitOpError() << "expected every field on the list's set";
  }
  if (getOuts().empty() && getReduce().empty())
    return emitOpError()
           << "expected at least 1 field in 'outs' or value in 'reduce'";
  if (!scratchMatches())
    return failure();
  double cutoff = getCutoff().convertToDouble();
  if (!std::isfinite(cutoff) || !(cutoff > 0.0))
    return emitOpError() << "expected a positive finite cutoff";
  unsigned numOuts = getOuts().size();
  unsigned numValues = numOuts + getReduce().size();
  if (auto weights = getWeights())
    if (weights->size() != getReduce().size())
      return emitOpError() << "expected " << getReduce().size()
                           << " weights, one per value in 'reduce', got "
                           << weights->size();
  if (auto overwrite = getOverwrite()) {
    if (!storage)
      return emitOpError() << "'overwrite' belongs to the storage form";
    if (overwrite->size() != numOuts)
      return emitOpError() << "expected " << numOuts
                           << " flags in 'overwrite', one per buffer in "
                              "'outs', got "
                           << overwrite->size();
  }
  // Each pair once, added at both ends (D89, D415).
  if (!getExchange() || getExchange()->size() != numValues)
    return emitOpError() << "expected " << numValues
                         << " exchange contracts, one per value in 'outs' "
                            "and 'reduce'";
  for (unsigned i = 0; i != numValues; ++i) {
    mdir::md::Exchange exchange = getExchange(i);
    if (exchange == mdir::md::Exchange::Symmetric ||
        (i < numOuts && exchange == mdir::md::Exchange::Antisymmetric))
      continue;
    return emitOpError() << "adds at both ends of a pair, which needs every "
                            "destination symmetric or antisymmetric and "
                            "every sum symmetric; value "
                         << i << " is "
                         << mdir::md::stringifyExchange(exchange);
  }
  // Results: one per field in `outs`, then one per value in `reduce`.
  SmallVector<Type> expected;
  if (!storage)
    for (Value field : getOuts())
      expected.push_back(field.getType());
  for (Value value : getReduce())
    expected.push_back(value.getType());
  if (getResultTypes() != ArrayRef<Type>(expected))
    return emitOpError() << "expected the fields of 'outs' in the value "
                            "form, then the values of 'reduce', as results";
  // What each segment that the loop visits must guarantee (D413).
  Contract required = getSegmentRequirement(getOperation());
  for (unsigned index = 0, e = getSegments().size(); index != e; ++index) {
    auto segment = dyn_cast<DenseI64ArrayAttr>(getSegments()[index]);
    if (!segment || segment.size() != 2 || segment[0] < 0 || segment[1] < 0 ||
        segment[1] > segment[0] ||
        segment[0] > static_cast<int64_t>(type.getNumKinds()))
      continue; // the region verifier names it
    unsigned s = SegmentedPairsType::getSegmentIndex(segment[0], segment[1]);
    if (std::optional<std::string> reason =
            type.getSegments()[s].whyNot(required))
      return emitOpError() << "takes a list whose segment(" << segment[0]
                           << ", " << segment[1]
                           << ") does not meet its requirement: " << *reason;
  }
  return success();
}

bool SegmentForOp::scratchMatches() {
  if (getScratch().empty())
    return true;
  if (!isStorageForm()) {
    emitOpError() << "'scratch' belongs to the storage form";
    return false;
  }
  if (getScratch().size() != 2 * getReduce().size()) {
    emitOpError() << "expected 2 buffers in 'scratch' for each value in "
                     "'reduce'";
    return false;
  }
  return true;
}

LogicalResult SegmentForOp::verifyRegions() {
  return mdrt::verifySegmentRegions(getOperation(), getListType(),
                                    getSegments(), getSites(), getKernels(),
                                    getSegmentSignature(*this));
}

void SegmentForOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  if (!isStorageForm())
    return;
  addEffect<MemoryEffects::Read>(effects, getListMutable());
  addEffect<MemoryEffects::Read>(effects, getPositionsMutable());
  for (OpOperand &operand : getInsMutable())
    addEffect<MemoryEffects::Read>(effects, operand);
  for (auto [index, operand] : llvm::enumerate(getOutsMutable())) {
    if (!overwrites(index))
      addEffect<MemoryEffects::Read>(effects, operand);
    addEffect<MemoryEffects::Write>(effects, operand);
  }
  for (OpOperand &operand : getScratchMutable())
    addEffect<MemoryEffects::Write>(effects, operand);
}
