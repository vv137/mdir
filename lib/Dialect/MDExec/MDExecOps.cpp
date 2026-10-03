// Ops of the md_exec dialect.

#include "mdir/Dialect/MDExec/MDExecOps.h"

#include "mdir/Dialect/MDExec/MDExecDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mdir::md_exec;
using mdir::md::FieldType;
using mdir::md::RelationType;
using mdir::mdrt::CellsType;
using mdir::mdrt::IncidenceType;
using mdir::mdrt::NeighborsType;
using mdir::mdrt::PermutationType;

#include "mdir/Dialect/MDExec/MDExecEnums.cpp.inc"

// The custom directive `coordinates(<kind>(<members>), ...)`.
/// `exchange [<kind>, ...]`, one exchange contract a value, or nothing.
static ParseResult parseExchangeList(OpAsmParser &parser, ArrayAttr &list) {
  if (failed(parser.parseOptionalKeyword("exchange")))
    return success();
  MLIRContext *context = parser.getContext();
  SmallVector<Attribute> kinds;
  auto parseOne = [&]() -> ParseResult {
    StringRef keyword;
    SMLoc loc = parser.getCurrentLocation();
    if (parser.parseKeyword(&keyword))
      return failure();
    std::optional<mdir::md::Exchange> kind = mdir::md::symbolizeExchange(keyword);
    if (!kind)
      return parser.emitError(loc)
             << "expected 'none', 'symmetric', or 'antisymmetric', got '"
             << keyword << "'";
    kinds.push_back(mdir::md::ExchangeAttr::get(context, *kind));
    return success();
  };
  if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Square, parseOne))
    return failure();
  list = ArrayAttr::get(context, kinds);
  return success();
}

static void printExchangeList(OpAsmPrinter &printer, Operation *,
                              ArrayAttr list) {
  if (!list)
    return;
  printer << "exchange [";
  llvm::interleaveComma(list, printer, [&](Attribute kind) {
    printer << mdir::md::stringifyExchange(cast<mdir::md::ExchangeAttr>(kind).getValue());
  });
  printer << "] ";
}

static ParseResult parseCoordinates(OpAsmParser &parser,
                                    DenseI32ArrayAttr &kinds,
                                    DenseI64ArrayAttr &members) {
  return mdir::md::parseCoordinateList(parser, kinds, members);
}

static void printCoordinates(OpAsmPrinter &printer, Operation *,
                             DenseI32ArrayAttr kinds,
                             DenseI64ArrayAttr members) {
  mdir::md::printCoordinateList(printer, kinds, members);
}

#define GET_OP_CLASSES
#include "mdir/Dialect/MDExec/MDExecOps.cpp.inc"

/// Returns true if `type` is a floating-point type of the execution level.
static bool isReal(Type type) { return type.isF32() || type.isF64(); }

bool mdir::md_exec::isValueFormType(Type type) {
  return isa<mdir::md::FieldType, mdir::md::RelationType,
             mdir::mdrt::CellsType, mdir::mdrt::PermutationType,
             mdir::mdrt::IncidenceType, mdir::md::TableType>(type);
}

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

bool mdir::md_exec::isScratchType(Type type) {
  auto buffer = dyn_cast<MemRefType>(type);
  if (!buffer || buffer.getRank() < 1 || buffer.getRank() > 2 ||
      !buffer.isDynamicDim(0))
    return false;
  if (buffer.getRank() == 2 && buffer.isDynamicDim(1))
    return false;
  Type element = buffer.getElementType();
  return isReal(element) || element.isSignlessInteger(32) ||
         element.isSignlessInteger(64);
}

bool mdir::md_exec::isMembersType(Type type) {
  auto buffer = dyn_cast<MemRefType>(type);
  return buffer && buffer.getRank() == 2 && buffer.isDynamicDim(0) &&
         !buffer.isDynamicDim(1) && buffer.getDimSize(1) >= 1 &&
         buffer.getElementType().isSignlessInteger(32);
}

bool mdir::md_exec::isIncidenceBufferType(Type type) {
  auto buffer = dyn_cast<MemRefType>(type);
  return buffer && buffer.getRank() == 2 && buffer.isDynamicDim(0) &&
         buffer.isDynamicDim(1) &&
         buffer.getElementType().isSignlessInteger(32);
}

MemRefType mdir::md_exec::getIncidenceBufferType(MLIRContext *context,
                                                 Attribute memorySpace) {
  return MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic},
                         IntegerType::get(context, 32),
                         MemRefLayoutAttrInterface(), memorySpace);
}

MemRefType mdir::md_exec::getScratchType(Type value, MemRefType like) {
  SmallVector<int64_t, 2> shape = {ShapedType::kDynamic};
  Type element = value;
  if (auto vector = dyn_cast<VectorType>(value)) {
    shape.push_back(vector.getNumElements());
    element = vector.getElementType();
  }
  return MemRefType::get(shape, element, MemRefLayoutAttrInterface(),
                         like.getMemorySpace());
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
    if (buffer != mdir::md_exec::getScratchType(expected, buffer))
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

/// Returns true if `type` is a buffer with one `i32` per particle.
static bool isIndexBuffer(Type type) {
  auto buffer = dyn_cast<MemRefType>(type);
  return buffer && buffer.getRank() == 1 &&
         buffer.getElementType().isInteger(32);
}

/// Returns true if `type` is a field or a buffer with one `i32` per
/// particle.
static bool isIdsType(Type type) {
  if (auto field = dyn_cast<FieldType>(type))
    return field.getNumComponents() == 1 &&
           field.getElementType().isInteger(32);
  return isIndexBuffer(type);
}

LogicalResult SpatialOrderOp::verify() {
  bool isStorage = isStorageForm();
  if (isa<MemRefType>(getIds().getType()) != isStorage)
    return emitOpError()
           << "expected fields only, as in the value form, or buffers "
              "only, as in the storage form";
  if (static_cast<bool>(getOrder()) != isStorage ||
      static_cast<bool>(getResult()) == isStorage)
    return emitOpError() << "expected a result in the value form, and a "
                            "buffer in 'outs' in the storage form";

  FlatSymbolRefAttr particleSet;
  if (auto order = getResult())
    particleSet = cast<PermutationType>(order.getType()).getParticleSet();
  if (failed(verifyPositions(getOperation(), getPositions(), particleSet)))
    return failure();
  if (!isIdsType(getIds().getType()))
    return emitOpError() << "expected 'ids' to hold one i32 per particle, "
                            "got "
                         << getIds().getType();
  if (auto ids = dyn_cast<FieldType>(getIds().getType()))
    if (ids.getParticleSet() != particleSet)
      return emitOpError() << "the order is on " << particleSet
                           << ", but 'ids' belongs to "
                           << ids.getParticleSet();
  if (getOrder() && !isIndexBuffer(getOrder().getType()))
    return emitOpError() << "expected the buffer in 'outs' to hold one i32 "
                            "per particle, got "
                         << getOrder().getType();

  double width = getWidth().convertToDouble();
  if (!(width > 0.0))
    return emitOpError() << "expected a positive width, got " << width;
  return success();
}

void SpatialOrderOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (!isStorageForm())
    return;
  addEffect<MemoryEffects::Read>(effects, getPositionsMutable());
  addEffect<MemoryEffects::Read>(effects, getIdsMutable());
  for (OpOperand &operand : getOrderMutable())
    addEffect<MemoryEffects::Write>(effects, operand);
}

LogicalResult PermuteOp::verify() {
  bool isStorage = isStorageForm();
  if (isa<MemRefType>(getOrder().getType()) != isStorage)
    return emitOpError()
           << "expected a field and an order, as in the value form, or "
              "buffers only, as in the storage form";
  if (static_cast<bool>(getOut()) != isStorage ||
      static_cast<bool>(getResult()) == isStorage)
    return emitOpError() << "expected a result in the value form, and a "
                            "buffer in 'outs' in the storage form";

  if (isStorage) {
    if (!isIndexBuffer(getOrder().getType()))
      return emitOpError() << "expected the order to hold one i32 per "
                              "particle, got "
                           << getOrder().getType();
    if (getOut().getType() != getField().getType())
      return emitOpError()
             << "expected the buffer in 'outs' to have the type of the "
                "field, "
             << getField().getType() << ", got " << getOut().getType();
    if (getOut() == getField())
      return emitOpError() << "expected the buffer in 'outs' to be another "
                              "buffer than that of the field";
    return success();
  }

  auto field = cast<FieldType>(getField().getType());
  auto order = cast<PermutationType>(getOrder().getType());
  if (field.getParticleSet() != order.getParticleSet())
    return emitOpError() << "the order is on " << order.getParticleSet()
                         << ", but the field belongs to "
                         << field.getParticleSet();
  if (getResult().getType() != field)
    return emitOpError() << "expected the result to have the type of the "
                            "field, "
                         << field << ", got " << getResult().getType();
  return success();
}

void ReciprocalOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (!isStorageForm())
    return;
  addEffect<MemoryEffects::Read>(effects, getPositionsMutable());
  addEffect<MemoryEffects::Read>(effects, getChargesMutable());
  addEffect<MemoryEffects::Read>(effects, getModuliMutable());
  for (OpOperand &operand : getOutMutable())
    addEffect<MemoryEffects::Write>(effects, operand);
  for (OpOperand &operand : getScratchMutable()) {
    addEffect<MemoryEffects::Read>(effects, operand);
    addEffect<MemoryEffects::Write>(effects, operand);
  }
}

void PermuteOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (!isStorageForm())
    return;
  addEffect<MemoryEffects::Read>(effects, getFieldMutable());
  addEffect<MemoryEffects::Read>(effects, getOrderMutable());
  for (OpOperand &operand : getOutMutable())
    addEffect<MemoryEffects::Write>(effects, operand);
}

/// Verifies the incidence structure of the pairs that a neighbor structure
/// on `particleSet` leaves out, if there is one.
static LogicalResult verifyExcluded(Operation *op, Value excluded,
                                    FlatSymbolRefAttr particleSet) {
  if (!excluded)
    return success();
  if (auto incidence = dyn_cast<IncidenceType>(excluded.getType())) {
    if (incidence.getArity() != 2)
      return op->emitOpError()
             << "expected the excluded pairs to have 2 members, got "
             << incidence.getArity();
    if (incidence.getParticleSet() != particleSet)
      return op->emitOpError()
             << "the excluded pairs are on " << incidence.getParticleSet()
             << ", but the structure is on " << particleSet;
  }
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
  return verifyExcluded(getOperation(), getExcluded(),
                        neighbors.getParticleSet());
}

LogicalResult EmptyNeighborsOp::verify() {
  if (getWidth() <= 0)
    return emitOpError() << "expected a positive width, got " << getWidth();
  if (failed(verifyExcluded(
          getOperation(), getExcluded(),
          cast<NeighborsType>(getResult().getType()).getParticleSet())))
    return failure();
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

  if (getPolicy() == RebuildPolicy::Interval) {
    if (!getInterval() || *getInterval() < 1)
      return emitOpError()
             << "expected a positive 'interval' with the policy 'interval'";
  } else if (getInterval()) {
    return emitOpError()
           << "'interval' belongs to the policy 'interval'";
  }

  if (std::optional<APFloat> pruneSkin = getPruneSkin()) {
    double prune = pruneSkin->convertToDouble();
    if (!(prune > 0.0) || !(prune < skin))
      return emitOpError() << "expected a 'prune_skin' above 0 and below "
                              "the skin "
                           << skin << ", got " << prune;
    if (getPolicy() != RebuildPolicy::Check)
      return emitOpError() << "a dual list ('prune_skin') belongs to the "
                              "policy 'check'";
  }
  if (getStale() && !getPruneSkin())
    return emitOpError() << "'stale' belongs to a dual list ('prune_skin')";

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

LogicalResult NeighborViewOp::verify() {
  auto counts = dyn_cast<MemRefType>(getCounts().getType());
  auto entries = dyn_cast<MemRefType>(getEntries().getType());
  if (!counts || counts.getRank() != 1 || !counts.getElementType().isInteger(32) ||
      !entries || entries.getRank() != 2 || !entries.getElementType().isInteger(32))
    return emitOpError("requires rank-one i32 counts and rank-two i32 entries");
  if (counts.getMemorySpace() || entries.getMemorySpace())
    return emitOpError("requires host buffers");
  if (!counts.isDynamicDim(0) && !entries.isDynamicDim(0) &&
      counts.getDimSize(0) != entries.getDimSize(0))
    return emitOpError("counts and entries must have the same row count");
  return success();
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

  // Each pair once, with atomic additions to both particles, over a
  // structure of groups (D89): the value of the kernel for (j, i) follows
  // from that for (i, j) by the exchange contract of each destination, and
  // a sum over the pairs from the sum over each pair once if its kernel is
  // symmetric.
  bool unique = getTraversal() == Traversal::Unique &&
                getConflict() == Conflict::Atomic;
  if (!unique && (getTraversal() != Traversal::Directed ||
                  getConflict() != Conflict::OwnerOnly))
    return emitOpError() << "only the policies (directed, owner_only) and "
                            "(unique, atomic) are supported";
  if (unique) {
    unsigned numOuts = getOuts().size();
    for (unsigned i = 0, e = numOuts + getReduce().size(); i != e; ++i) {
      mdir::md::Exchange exchange = getExchange(i);
      if (exchange == mdir::md::Exchange::Symmetric ||
          (i < numOuts && exchange == mdir::md::Exchange::Antisymmetric))
        continue;
      return emitOpError()
             << "the policy (unique, atomic) needs every destination "
                "symmetric or antisymmetric and every sum symmetric; value "
             << i << " is " << mdir::md::stringifyExchange(exchange);
    }
  }

  if (auto exchange = getExchange()) {
    unsigned values = getOuts().size() + getReduce().size();
    if (exchange->size() != values)
      return emitOpError() << "expected " << values
                           << " exchange contracts, one per value in 'outs' "
                              "and 'reduce', got "
                           << exchange->size();
  }

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

LogicalResult TabulateOp::verify() {
  if (getTables().empty())
    return emitOpError() << "expected at least 1 table";
  auto result = cast<md::TableType>(getResult().getType());
  bool symmetric = true;
  for (Value table : getTables()) {
    auto type = cast<md::TableType>(table.getType());
    if (type.getRank() != result.getRank())
      return emitOpError() << "expected tables of rank " << result.getRank()
                           << ", got " << type;
    symmetric &= type.getSymmetric();
  }
  if (result.getSymmetric() && !symmetric)
    return emitOpError()
           << "the table is symmetric only if every table it comes from is";
  Type element = result.getElementType();
  if (!getElementTypeOrSelf(element).isF64())
    return emitOpError() << "expected a table of f64 or of vectors of f64, got "
                         << result;
  Block &kernel = getKernel().front();
  if (kernel.getNumArguments() != getTables().size())
    return emitOpError() << "expected " << getTables().size()
                         << " arguments of the kernel, one per table, got "
                         << kernel.getNumArguments();
  for (BlockArgument argument : kernel.getArguments())
    if (!argument.getType().isF64())
      return emitOpError() << "expected the arguments of the kernel in f64";
  Operation *yield = kernel.getTerminator();
  if (yield->getNumOperands() != 1 || yield->getOperand(0).getType() != element)
    return emitOpError() << "expected the kernel to yield one " << element;
  return success();
}

mdir::md::Exchange PairForOp::getExchange(unsigned index) {
  std::optional<ArrayAttr> list = getExchange();
  if (!list || index >= list->size())
    return mdir::md::Exchange::None;
  return cast<mdir::md::ExchangeAttr>((*list)[index]).getValue();
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

//===----------------------------------------------------------------------===//
// Tuples of a topology
//===----------------------------------------------------------------------===//

LogicalResult BuildIncidenceOp::verify() {
  bool isStorage = isStorageForm();
  if (isStorage != isa<MemRefType>(getResult().getType()))
    return emitOpError()
           << "expected a relation and a structure, as in the value form, or "
              "buffers only, as in the storage form";
  if (isStorage) {
    if (!getSize())
      return emitOpError() << "expected the number of particles in 'size': "
                              "the buffer of the members does not tell it";
    if (cast<MemRefType>(getRelation().getType()).getMemorySpace())
      return emitOpError()
             << "expected the buffer of the members on the host";
    return success();
  }

  if (getSize())
    return emitOpError() << "'size' belongs to the storage form";
  // A relation without a tuple set is one that is found, as the triplets
  // of 'md_exec.build_triplets' are (D160).
  auto relation = cast<RelationType>(getRelation().getType());
  auto incidence = cast<IncidenceType>(getResult().getType());
  if (relation.getParticleSet() != incidence.getParticleSet() ||
      relation.getTupleSet() != incidence.getTupleSet() ||
      relation.getArity() != incidence.getArity())
    return emitOpError()
           << "expected the result to have type "
           << IncidenceType::get(getContext(), relation.getParticleSet(),
                                 relation.getArity(), relation.getTupleSet())
           << ", got " << incidence;
  return success();
}

void BuildIncidenceOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (!isStorageForm())
    return;
  addEffect<MemoryEffects::Read>(effects, getRelationMutable());
  effects.emplace_back(MemoryEffects::Allocate::get(),
                       getOperation()->getOpResults().front(), /*stage=*/0,
                       /*effectOnFullRegion=*/true,
                       SideEffects::DefaultResource::get());
}

LogicalResult BuildTripletsOp::verify() {
  auto neighbors = cast<NeighborsType>(getNeighbors().getType());
  if (failed(verifyPositions(getOperation(), getPositions(),
                             neighbors.getParticleSet())))
    return failure();
  double cutoff = getCutoff().convertToDouble();
  if (!(cutoff > 0.0))
    return emitOpError() << "expected a positive cutoff, got " << cutoff;
  Type result = getResult().getType();
  if (isStorageForm()) {
    auto members = dyn_cast<MemRefType>(result);
    if (!members || members.getDimSize(1) != 3 || members.getMemorySpace())
      return emitOpError() << "expected the members of triplets, "
                              "memref<?x3xi32> on the host, got "
                           << result;
    return success();
  }
  auto relation = dyn_cast<RelationType>(result);
  if (!relation || relation.getTupleSet() || relation.getArity() != 3 ||
      relation.getOrientation() != md::Orientation::Reversal ||
      relation.getParticleSet() != neighbors.getParticleSet())
    return emitOpError()
           << "expected the result to have type "
           << RelationType::get(getContext(), neighbors.getParticleSet(), 3,
                                md::Orientation::Reversal,
                                FlatSymbolRefAttr())
           << ", got " << result;
  return success();
}

void BuildTripletsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (!isStorageForm())
    return;
  addEffect<MemoryEffects::Read>(effects, getNeighborsMutable());
  addEffect<MemoryEffects::Read>(effects, getPositionsMutable());
  effects.emplace_back(MemoryEffects::Allocate::get(),
                       getOperation()->getOpResults().front(), /*stage=*/0,
                       /*effectOnFullRegion=*/true,
                       SideEffects::DefaultResource::get());
}

/// The floating-point type that the kernel of a loop over tuples computes
/// in, which its first argument states.
static Type getTupleKernelReal(TupleForOp op) {
  Block &block = op.getKernel().front();
  if (block.getNumArguments() != 0)
    if (auto vector = dyn_cast<VectorType>(block.getArgument(0).getType()))
      if (vector.getElementType().isF32())
        return vector.getElementType();
  return Float64Type::get(op.getContext());
}

LogicalResult TupleForOp::verify() {
  int64_t arity = getArity();
  if (arity < 1)
    return emitOpError() << "expected an arity of at least 1, got " << arity;

  FlatSymbolRefAttr particleSet, tupleSet;
  if (auto incidence = dyn_cast<IncidenceType>(getIncidence().getType())) {
    particleSet = incidence.getParticleSet();
    tupleSet = incidence.getTupleSet();
    if (incidence.getArity() != arity)
      return emitOpError() << "the tuples have " << incidence.getArity()
                           << " members, but the arity is " << arity;
  }
  if (isStorageForm() != isa<MemRefType>(getIncidence().getType()))
    return emitOpError()
           << "expected a structure and fields, as in the value form, or "
              "buffers only, as in the storage form";
  if (failed(verifyPositions(getOperation(), getPositions(), particleSet)))
    return failure();

  for (Value parameter : getParameters()) {
    if (isa<MemRefType>(parameter.getType()) != isStorageForm())
      return emitOpError()
             << "expected fields only, as in the value form, or buffers "
                "only, as in the storage form";
    auto field = dyn_cast<FieldType>(parameter.getType());
    if (field && field.getParticleSet() != tupleSet)
      return emitOpError() << "the tuples are those of " << tupleSet
                           << ", but a field in 'tuple' belongs to "
                           << field.getParticleSet();
  }

  if (failed(mdir::md::verifyCoordinates(getOperation(),
                                         getCoordinateKinds(),
                                         getCoordinateMembers(), arity)))
    return failure();
  for (const mdir::md::Coordinate &coordinate : getCoordinates())
    if (coordinate.kind != mdir::md::CoordinateKind::Displacement)
      return emitOpError()
             << "expected displacements only, got '"
             << mdir::md::stringifyCoordinateKind(coordinate.kind)
             << "': the kernel computes the other coordinates from them";

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
  return verifyLoop(*this, particleSet, /*allowsFlags=*/false);
}

LogicalResult TupleForOp::verifyRegions() {
  Block &block = getKernel().front();
  unsigned arity = getArity();

  SmallVector<Type> arguments(getCoordinateKinds().size(),
                              VectorType::get({3}, getTupleKernelReal(*this)));
  for (Value field : getIns())
    arguments.append(arity, getKernelValueType(field.getType()));
  for (Value field : getParameters())
    arguments.push_back(getKernelValueType(field.getType()));
  if (block.getNumArguments() != arguments.size())
    return emitOpError()
           << "expected the kernel to have " << arguments.size()
           << " arguments (one per displacement, one per member for each "
              "field in 'ins', and one per field in 'tuple'), got "
           << block.getNumArguments();
  for (unsigned i = 0, e = arguments.size(); i != e; ++i)
    if (block.getArgument(i).getType() != arguments[i])
      return emitOpError()
             << "expected kernel argument " << i << " to have type "
             << arguments[i] << ", got " << block.getArgument(i).getType();

  SmallVector<Type> yielded;
  for (Value field : getOuts())
    yielded.append(arity, getKernelValueType(field.getType()));
  for (Value value : getReduce())
    yielded.push_back(value.getType());

  auto yield = dyn_cast<YieldOp>(block.getTerminator());
  if (!yield)
    return emitOpError() << "expected the kernel to end with 'md_exec.yield'";
  if (yield.getNumOperands() != yielded.size())
    return yield.emitOpError()
           << "expected " << yielded.size()
           << " values (one per member for each field in 'outs', and one "
              "per value in 'reduce'), got "
           << yield.getNumOperands();
  for (unsigned i = 0, e = yielded.size(); i != e; ++i)
    if (yield.getOperand(i).getType() != yielded[i])
      return yield.emitOpError()
             << "expected value " << i << " to have type " << yielded[i]
             << ", got " << yield.getOperand(i).getType();
  return success();
}

void TupleForOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (!isStorageForm())
    return;
  addEffect<MemoryEffects::Read>(effects, getIncidenceMutable());
  addEffect<MemoryEffects::Read>(effects, getPositionsMutable());
  for (OpOperand &operand : getParametersMutable())
    addEffect<MemoryEffects::Read>(effects, operand);
  getLoopEffects(*this, effects,
                 [&](unsigned index) { return !overwrites(index); });
}

LogicalResult RenumberOp::verify() {
  Type ids = getIds().getType();
  if (isStorageForm() != isa<MemRefType>(ids))
    return emitOpError()
           << "expected a relation and a field, as in the value form, or "
              "buffers only, as in the storage form";
  Type element = isa<MemRefType>(ids)
                     ? cast<MemRefType>(ids).getElementType()
                     : cast<FieldType>(ids).getElementType();
  if (!element.isSignlessInteger(32))
    return emitOpError() << "expected the numbers of the particles in i32, "
                            "got "
                         << ids;
  return success();
}

void RenumberOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (!isStorageForm())
    return;
  addEffect<MemoryEffects::Read>(effects, getMembersMutable());
  addEffect<MemoryEffects::Read>(effects, getIdsMutable());
  effects.emplace_back(MemoryEffects::Allocate::get(),
                       getOperation()->getOpResults().front(), /*stage=*/0,
                       /*effectOnFullRegion=*/true,
                       SideEffects::DefaultResource::get());
}
