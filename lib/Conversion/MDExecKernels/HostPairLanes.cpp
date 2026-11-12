// The lanes of a row of a loop over pairs on the host
// (D402); see include/mdir/Conversion/HostPairEmitter.h.
//
// Each value of the kernel is held as its components, one
// `vector<W x T>` a component (a number has one, a `vector<n x T>` has
// n): the component-major layout of the design. A value that is the same
// in every lane (a constant, a global, a value of the center) stays one
// scalar and is splat where an op of the lanes takes it.

#include "mdir/Conversion/HostPairEmitter.h"

#include "mdir/Conversion/MDExecKernels.h"
#include "mdir/Conversion/ReductionEmitter.h"
#include "mdir/Dialect/MDExec/HostPairs.h"

#include "mlir/Conversion/ArithCommon/AttrToLLVMConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;
using md_exec::hostpairs::LaneMapping;

namespace {

/// The number of components of a value of `type`.
int64_t getNumParts(Type type) {
  if (auto vector = dyn_cast<VectorType>(type))
    return vector.getNumElements();
  return 1;
}

/// The values of a kernel in lanes.
class Lanes {
public:
  Lanes(OpBuilder &builder, Location loc, int64_t width)
      : builder(builder), loc(loc), width(width) {}

  /// The type of a component of a value whose elements are `element`.
  VectorType getLaneType(Type element) const {
    return VectorType::get({width}, element);
  }

  /// `value` as the same scalar in every lane.
  void mapUniform(Value from, Value to) { uniform.map(from, to); }
  /// `value` as its components in lanes.
  void mapLanes(Value from, ArrayRef<Value> parts) {
    lanes[from].assign(parts.begin(), parts.end());
  }

  bool isUniform(Value value) const { return !lanes.count(value); }
  Value getUniform(Value value) const { return uniform.lookupOrDefault(value); }

  /// The components of `value` in lanes; a uniform value is splat.
  SmallVector<Value> getParts(Value value) {
    auto found = lanes.find(value);
    if (found != lanes.end())
      return found->second;
    Value scalar = getUniform(value);
    SmallVector<Value> parts;
    auto vector = dyn_cast<VectorType>(scalar.getType());
    if (!vector) {
      parts.push_back(splat(scalar));
      return parts;
    }
    for (int64_t c = 0, e = vector.getNumElements(); c != e; ++c)
      parts.push_back(
          splat(vector::ExtractOp::create(builder, loc, scalar, c)));
    return parts;
  }

  /// `scalar` in every lane.
  Value splat(Value scalar) {
    return vector::BroadcastOp::create(builder, loc,
                                       getLaneType(scalar.getType()), scalar);
  }

  /// The value of lane `lane` of `value`, of the type `type` of the kernel.
  Value getLane(Value value, Type type, int64_t lane) {
    if (isUniform(value))
      return getUniform(value);
    SmallVector<Value> &parts = lanes[value];
    if (!isa<VectorType>(type))
      return vector::ExtractOp::create(builder, loc, parts.front(), lane);
    SmallVector<Value> elements;
    for (Value part : parts)
      elements.push_back(vector::ExtractOp::create(builder, loc, part, lane));
    return vector::FromElementsOp::create(builder, loc, cast<VectorType>(type),
                                          elements);
  }

  /// The components of the values `perLane` (one of the type `type` a
  /// lane): the transpose of the reads of the lanes.
  SmallVector<Value> fromLanes(ArrayRef<Value> perLane, Type type) {
    Type element = getElementTypeOrSelf(type);
    SmallVector<Value> parts;
    for (int64_t c = 0, e = getNumParts(type); c != e; ++c) {
      SmallVector<Value> elements;
      for (Value value : perLane)
        elements.push_back(
            isa<VectorType>(type)
                ? vector::ExtractOp::create(builder, loc, value, c).getResult()
                : value);
      parts.push_back(vector::FromElementsOp::create(
          builder, loc, getLaneType(element), elements));
    }
    return parts;
  }

  /// Evaluates `op`, an op of the kernel other than its terminator.
  void emit(Operation &op) {
    LaneMapping mapping = md_exec::hostpairs::getLaneMapping(&op);
    bool allUniform = llvm::all_of(
        op.getOperands(), [&](Value value) { return isUniform(value); });
    if (mapping == LaneMapping::Uniform || allUniform) {
      Operation *clone = builder.clone(op, uniform);
      for (auto [from, to] : llvm::zip(op.getResults(), clone->getResults()))
        uniform.map(from, to);
      return;
    }
    switch (mapping) {
    case LaneMapping::Elementwise:
      return emitElementwise(op);
    case LaneMapping::PowI:
      return emitPowI(cast<math::FPowIOp>(op));
    case LaneMapping::Extract: {
      auto extract = cast<vector::ExtractOp>(op);
      int64_t place = extract.getStaticPosition().front();
      lanes[extract.getResult()] = {getParts(extract.getSource())[place]};
      return;
    }
    case LaneMapping::Broadcast: {
      auto broadcast = cast<vector::BroadcastOp>(op);
      Value part = getParts(broadcast.getSource()).front();
      lanes[broadcast.getResult()] = SmallVector<Value>(
          broadcast.getResultVectorType().getNumElements(), part);
      return;
    }
    case LaneMapping::FromElements: {
      auto elements = cast<vector::FromElementsOp>(op);
      SmallVector<Value> parts;
      for (Value element : elements.getElements())
        parts.push_back(getParts(element).front());
      lanes[elements.getResult()] = parts;
      return;
    }
    case LaneMapping::Gather:
      return emitGather(op);
    case LaneMapping::None:
    case LaneMapping::Uniform:
    case LaneMapping::Yield:
      break;
    }
    llvm_unreachable("an op without a lane mapping (checkLanes)");
  }

private:
  OpBuilder &builder;
  Location loc;
  int64_t width;
  IRMapping uniform;
  DenseMap<Value, SmallVector<Value>> lanes;

  /// The same op for each component, on vectors of the lanes.
  void emitElementwise(Operation &op) {
    SmallVector<SmallVector<Value>> operands;
    for (Value operand : op.getOperands())
      operands.push_back(getParts(operand));
    int64_t numParts = 1;
    for (Type type : op.getResultTypes())
      numParts = std::max(numParts, getNumParts(type));
    SmallVector<SmallVector<Value>> results(op.getNumResults());
    for (int64_t c = 0; c != numParts; ++c) {
      OperationState state(loc, op.getName());
      // An operand of one component (the condition of a select of
      // vectors) serves every component.
      for (SmallVector<Value> &parts : operands)
        state.addOperands(parts.size() == 1 ? parts.front() : parts[c]);
      for (Type type : op.getResultTypes())
        state.addTypes(getLaneType(getElementTypeOrSelf(type)));
      state.addAttributes(op.getAttrs());
      Operation *created = builder.create(state);
      for (unsigned r = 0, e = op.getNumResults(); r != e; ++r)
        results[r].push_back(created->getResult(r));
    }
    for (unsigned r = 0, e = op.getNumResults(); r != e; ++r)
      lanes[op.getResult(r)] = results[r];
  }

  /// `llvm.intr.powi` of the lanes with the scalar exponent: LLVM's code
  /// generator expands a constant exponent of a vector as it expands that
  /// of a number (repeated squaring), so that each lane rounds as the
  /// scalar row; MLIR's `math.fpowi` takes a vector of exponents, which its
  /// lowering to LLVM refuses.
  void emitPowI(math::FPowIOp op) {
    Value exponent = getUniform(op.getRhs());
    SmallVector<Value> parts;
    for (Value part : getParts(op.getLhs()))
      parts.push_back(LLVM::PowIOp::create(
          builder, loc, part.getType(), part, exponent,
          arith::convertArithFastMathAttrToLLVM(op.getFastmathAttr())));
    lanes[op.getResult()] = parts;
  }

  /// One read a lane, at the lane's indices, then the transpose into
  /// components.
  void emitGather(Operation &op) {
    SmallVector<SmallVector<Value>> perLane(op.getNumResults());
    for (int64_t lane = 0; lane != width; ++lane) {
      IRMapping inLane = uniform;
      for (Value operand : op.getOperands())
        if (!isUniform(operand))
          inLane.map(operand, getLane(operand, operand.getType(), lane));
      Operation *clone = builder.clone(op, inLane);
      for (unsigned r = 0, e = op.getNumResults(); r != e; ++r)
        perLane[r].push_back(clone->getResult(r));
    }
    for (unsigned r = 0, e = op.getNumResults(); r != e; ++r)
      lanes[op.getResult(r)] =
          fromLanes(perLane[r], op.getResult(r).getType());
  }
};

} // namespace

/// The minimum image of the components `d` of the displacements of the
/// lanes, as emitMinimumImage takes it of one displacement, op for op:
/// `box` and `inverse` are the components of the cell (3 or 6) and of the
/// inverse of its edges, splat.
static void emitLaneMinimumImage(OpBuilder &b, Location loc,
                                 MutableArrayRef<Value> d,
                                 ArrayRef<Value> box,
                                 ArrayRef<Value> inverse) {
  if (box.size() == 3) {
    for (int64_t k = 0; k != 3; ++k) {
      Value images = arith::MulFOp::create(b, loc, d[k], inverse[k]);
      Value nearest = math::RoundEvenOp::create(b, loc, images);
      Value shift = arith::MulFOp::create(b, loc, nearest, box[k]);
      d[k] = arith::SubFOp::create(b, loc, d[k], shift);
    }
    return;
  }
  Value ax = box[0], by = box[1], cz = box[2], bx = box[3], cx = box[4],
        cy = box[5];
  auto take = [&](Value along, Value inverseDiagonal,
                  ArrayRef<std::pair<int, Value>> row) {
    Value n = math::RoundEvenOp::create(
        b, loc, arith::MulFOp::create(b, loc, along, inverseDiagonal));
    for (auto [axis, component] : row)
      d[axis] = arith::SubFOp::create(
          b, loc, d[axis], arith::MulFOp::create(b, loc, n, component));
  };
  take(d[2], inverse[2], {{0, cx}, {1, cy}, {2, cz}});
  take(d[1], inverse[1], {{0, bx}, {1, by}});
  take(d[0], inverse[0], {{0, ax}});
}

SmallVector<Value> kernels::emitPairRowLanes(OpBuilder &builder,
                                             md_exec::PairForOp op,
                                             Value counts, Value index,
                                             Value box, Value inverse,
                                             Value central, int64_t width) {
  Location loc = op.getLoc();
  Value positions = op.getPositions();
  SmallVector<Value> insBuffers(op.getIns().begin(), op.getIns().end());
  Block &kernel = op.getKernel().front();
  Operation *yield = kernel.getTerminator();
  unsigned numIns = insBuffers.size();
  unsigned numOuts = op.getOuts().size();
  unsigned numYields = yield->getNumOperands();

  // As in the scalar row (emitPairKernel): the difference of the positions
  // in their type, then the minimum image and the squared length in the
  // type of the kernel.
  Type computed = kernel.getArgument(0).getType();
  Type indexElement = cast<MemRefType>(index.getType()).getElementType();
  Value cutoff2 =
      createCutoff2(builder, loc, computed, op.getCutoff().convertToDouble());
  Value boxComputed = convertReal(builder, loc, box, computed);
  Value inverseComputed = convertReal(builder, loc, inverse, computed);
  Value zero = createIndex(builder, loc, 0);
  Value one = createIndex(builder, loc, 1);
  Value lanesIndex = createIndex(builder, loc, width);

  // The values that every lane shares, splat once for the row.
  Lanes setup(builder, loc, width);
  auto splatComponents = [&](Value vector) {
    SmallVector<Value> parts;
    for (int64_t c = 0, e = getNumParts(vector.getType()); c != e; ++c)
      parts.push_back(setup.splat(
          vector::ExtractOp::create(builder, loc, vector, c)));
    return parts;
  };
  SmallVector<Value> boxParts = splatComponents(boxComputed);
  SmallVector<Value> inverseParts = splatComponents(inverseComputed);
  Value cutoff2Lanes = setup.splat(cutoff2);
  Value centralLanes = setup.splat(central);
  Value centralPosition = loadElement(builder, loc, positions, central);
  SmallVector<Value> centralParts = splatComponents(centralPosition);
  SmallVector<Value> centralValues;
  for (Value buffer : insBuffers)
    centralValues.push_back(loadElement(builder, loc, buffer, central));

  Value count =
      memref::LoadOp::create(builder, loc, counts, ValueRange{central});
  Value end =
      arith::IndexCastOp::create(builder, loc, builder.getIndexType(), count);
  Value rest = arith::RemUIOp::create(builder, loc, end, lanesIndex);
  Value full = arith::SubIOp::create(builder, loc, end, rest);

  // A contribution to a global sum that the kernel widens is summed in the
  // type it was computed in, as in the scalar row.
  auto getNarrow = [&](unsigned i) -> Value {
    if (i < numOuts)
      return Value();
    auto widen = yield->getOperand(i).getDefiningOp<arith::ExtFOp>();
    if (!widen || widen->getBlock() != &kernel)
      return Value();
    return widen.getIn();
  };
  // The sums of the lanes, flat: the components of each yielded value in
  // turn, from `first[i]`.
  SmallVector<Value> sums;
  SmallVector<unsigned> first;
  for (unsigned i = 0; i != numYields; ++i) {
    Value narrow = getNarrow(i);
    Type type = narrow ? narrow.getType() : yield->getOperand(i).getType();
    first.push_back(sums.size());
    for (int64_t c = 0, e = getNumParts(type); c != e; ++c)
      sums.push_back(
          createZero(builder, loc, setup.getLaneType(getElementTypeOrSelf(type))));
  }
  first.push_back(sums.size());

  // One group of `width` candidates whose entries are `entries`;
  // `valid`, where it is set, masks the lanes past the row.
  auto emitGroup = [&](OpBuilder &b, Value entries, Value valid,
                       ValueRange partial) -> SmallVector<Value> {
    Lanes values(b, loc, width);
    Value others = arith::IndexCastOp::create(
        b, loc, values.getLaneType(b.getIndexType()), entries);
    SmallVector<Value> other;
    for (int64_t lane = 0; lane != width; ++lane)
      other.push_back(vector::ExtractOp::create(b, loc, others, lane));
    auto gather = [&](Value buffer) {
      SmallVector<Value> perLane;
      for (Value at : other)
        perLane.push_back(loadElement(b, loc, buffer, at));
      return values.fromLanes(perLane, perLane.front().getType());
    };

    // The minimum-image displacement [AllenTildesley2017] and its squared
    // length, (d_x^2 + d_y^2) + d_z^2: the scalar row's `vector.reduction`
    // adds in this order from +0, and +0 + d_x^2 is d_x^2 (a square is not
    // -0).
    SmallVector<Value> otherPosition = gather(positions);
    SmallVector<Value> d;
    for (int64_t k = 0; k != 3; ++k)
      d.push_back(convertReal(
          b, loc,
          arith::SubFOp::create(b, loc, centralParts[k], otherPosition[k]),
          computed));
    emitLaneMinimumImage(b, loc, d, boxParts, inverseParts);
    Value r2;
    for (int64_t k = 0; k != 3; ++k) {
      Value square = arith::MulFOp::create(b, loc, d[k], d[k]);
      r2 = r2 ? arith::AddFOp::create(b, loc, r2, square).getResult()
              : square;
    }

    values.mapLanes(kernel.getArgument(0), r2);
    values.mapLanes(kernel.getArgument(1), d);
    for (unsigned i = 0; i != numIns; ++i) {
      values.mapUniform(kernel.getArgument(2 + 2 * i), centralValues[i]);
      values.mapLanes(kernel.getArgument(3 + 2 * i), gather(insBuffers[i]));
    }
    for (Operation &nested : kernel.without_terminator())
      values.emit(nested);

    // A pair beyond the cutoff contributes nothing, nor does an entry that
    // stands for the particle itself, nor a lane past the row.
    Value within = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLT,
                                         r2, cutoff2Lanes);
    Value distinct = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne,
                                           others, centralLanes);
    within = arith::AndIOp::create(b, loc, within, distinct);
    if (valid)
      within = arith::AndIOp::create(b, loc, within, valid);
    SmallVector<Value> updated;
    for (unsigned i = 0; i != numYields; ++i) {
      Value narrow = getNarrow(i);
      SmallVector<Value> parts =
          values.getParts(narrow ? narrow : yield->getOperand(i));
      for (auto [c, part] : llvm::enumerate(parts)) {
        Value nothing = createZero(b, loc, part.getType());
        Value masked = arith::SelectOp::create(b, loc, within, part, nothing);
        updated.push_back(
            arith::AddFOp::create(b, loc, partial[first[i] + c], masked));
      }
    }
    return updated;
  };

  VectorType entriesType = setup.getLaneType(indexElement);
  auto groups = scf::ForOp::create(
      builder, loc, zero, full, lanesIndex, sums,
      [&](OpBuilder &b, Location, Value k, ValueRange partial) {
        Value entries = vector::LoadOp::create(b, loc, entriesType, index,
                                               ValueRange{central, k});
        scf::YieldOp::create(b, loc, emitGroup(b, entries, Value(), partial));
      });
  // The tail: the lanes from `rest` on repeat the row's last candidate.
  Value hasTail =
      arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ne, rest, zero);
  auto tail = scf::IfOp::create(builder, loc, TypeRange(ValueRange(sums)),
                                hasTail, /*withElseRegion=*/true);
  {
    OpBuilder b = OpBuilder::atBlockEnd(tail.thenBlock());
    if (!tail.thenBlock()->empty())
      tail.thenBlock()->back().erase();
    Value lastEntry = arith::SubIOp::create(b, loc, end, one);
    Value last = memref::LoadOp::create(b, loc, index,
                                        ValueRange{central, lastEntry});
    Value padding = vector::BroadcastOp::create(b, loc, entriesType, last);
    Value valid = vector::CreateMaskOp::create(
        b, loc, setup.getLaneType(b.getI1Type()), ValueRange{rest});
    Value entries = vector::MaskedLoadOp::create(
        b, loc, entriesType, index, ValueRange{central, full}, valid,
        padding);
    scf::YieldOp::create(b, loc,
                         emitGroup(b, entries, valid, groups.getResults()));
  }
  {
    OpBuilder b = OpBuilder::atBlockEnd(tail.elseBlock());
    if (!tail.elseBlock()->empty())
      tail.elseBlock()->back().erase();
    scf::YieldOp::create(b, loc, groups.getResults());
  }

  // The lanes combined: a sum that the kernel widens is widened in each
  // lane first, as the device's lanes widen before their butterfly.
  SmallVector<Value> totals;
  for (unsigned i = 0; i != numYields; ++i) {
    Type type = yield->getOperand(i).getType();
    Type element = getElementTypeOrSelf(type);
    SmallVector<Value> parts;
    for (unsigned k = first[i]; k != first[i + 1]; ++k) {
      Value part = tail.getResult(k);
      if (getNarrow(i))
        part = arith::ExtFOp::create(builder, loc, setup.getLaneType(element),
                                     part);
      parts.push_back(emitLaneReduce(builder, loc, part, Combiner::Add));
    }
    totals.push_back(isa<VectorType>(type)
                         ? vector::FromElementsOp::create(
                               builder, loc, cast<VectorType>(type), parts)
                               .getResult()
                         : parts.front());
  }

  // The destinations, as the scalar row writes them.
  for (unsigned i = 0; i != numOuts; ++i) {
    Value destination = op.getOuts()[i];
    Value total = totals[i];
    if (!op.overwrites(i))
      total = arith::AddFOp::create(
          builder, loc, loadElement(builder, loc, destination, central),
          total);
    storeElement(builder, loc, total, destination, central);
  }

  SmallVector<Value> contributions;
  for (unsigned i = numOuts; i != numYields; ++i) {
    Value total = totals[i];
    if (auto weights = op.getWeights()) {
      double weight = (*weights)[i - numOuts];
      if (weight != 1.0) {
        Type type = total.getType();
        FloatAttr scalar =
            builder.getFloatAttr(getElementTypeOrSelf(type), weight);
        Value factor;
        if (auto vector = dyn_cast<VectorType>(type))
          factor = arith::ConstantOp::create(
              builder, loc, type,
              DenseElementsAttr::get(vector, scalar.getValue()));
        else
          factor = arith::ConstantOp::create(builder, loc, type, scalar);
        total = arith::MulFOp::create(builder, loc, factor, total);
      }
    }
    contributions.push_back(total);
  }
  return contributions;
}
