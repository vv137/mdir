// The proof of the site geometry of a segment window (D[kind-pairs]).
#include "mdir/Dialect/MD/MDSegmentWindow.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <cmath>

using namespace mlir;
using namespace mdir::md;

namespace {
// Symbolic affine proof, not a tolerance-based numerical classification.
// Only literal scalar weights and zero vector constants may enter it.
static bool coefficients(Value value, Block &block, unsigned count,
                         SmallVectorImpl<double> &out, unsigned depth = 0) {
  if (depth > 64) return false;
  out.assign(count, 0.0);
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (arg.getOwner() != &block || arg.getArgNumber() >= count) return false;
    out[arg.getArgNumber()] = 1.0;
    return true;
  }
  DenseFPElementsAttr zero;
  if (matchPattern(value, m_Constant(&zero)))
    return zero.isSplat() && zero.getSplatValue<APFloat>().isZero();
  Operation *op = value.getDefiningOp();
  if (!op) return false;
  if (isa<arith::AddFOp, arith::SubFOp>(op)) {
    SmallVector<double> a, b;
    if (!coefficients(op->getOperand(0), block, count, a, depth + 1) ||
        !coefficients(op->getOperand(1), block, count, b, depth + 1)) return false;
    double sign = isa<arith::SubFOp>(op) ? -1.0 : 1.0;
    for (unsigned i = 0; i != count; ++i) out[i] = a[i] + sign * b[i];
    return true;
  }
  if (auto mul = dyn_cast<arith::MulFOp>(op)) {
    for (unsigned i = 0; i != 2; ++i) {
      auto splat = mul->getOperand(i).getDefiningOp<vector::BroadcastOp>();
      FloatAttr scalar;
      DenseFPElementsAttr folded;
      double factor;
      if (splat && matchPattern(splat.getSource(), m_Constant(&scalar)))
        factor = scalar.getValueAsDouble();
      else if (matchPattern(mul->getOperand(i), m_Constant(&folded)) && folded.isSplat())
        factor = folded.getSplatValue<APFloat>().convertToDouble();
      else continue;
      if (!std::isfinite(factor) ||
          !coefficients(mul->getOperand(1 - i), block, count, out, depth + 1))
        return false;
      for (double &coefficient : out) coefficient *= factor;
      return true;
    }
  }
  return false;
}

// The first place of a side whose coefficients over its frame offsets are
// exactly these, compared as f64.
static std::optional<unsigned> identify(const SegmentSide &side,
                                        ArrayRef<double> found) {
  if (!side.kind) return 0u;
  for (unsigned place = 0; place != side.getSize(); ++place)
    if (llvm::equal(side.kind.getPlaceCoefficients(place), found))
      return place;
  return std::nullopt;
}

template <typename Op> static LogicalResult verifyOperands(Op op, bool gather) {
  auto pairs = op.getRelation().template getDefiningOp<KindPairsOp>();
  if (!pairs)
    return op.emitOpError("expected a segment of 'md.kind_pairs'");
  int64_t f = op.getF(), g = op.getG();
  int64_t kinds = pairs.getNumTakenKinds();
  if (f < 0 || g < 0 || f > kinds)
    return op.emitOpError("expected kinds 0 to K = ") << kinds << " in segment(f, g)";
  if (g > f)
    return op.emitOpError("expected g <= f in segment(")
           << f << ", " << g << ")";
  unsigned number = cast<OpResult>(op.getRelation()).getResultNumber();
  if (number != getSegmentNumber(f, g))
    return op.emitOpError("segment(")
           << f << ", " << g << ") is s(f, g) = " << getSegmentNumber(f, g)
           << ", not result " << number << " of 'md.kind_pairs'";
  auto relation = cast<RelationType>(op.getRelation().getType());
  auto positions = cast<FieldType>(op.getPositions().getType());
  if (positions.getParticleSet() != relation.getParticleSet() ||
      positions.getNumComponents() != 3 || !positions.getElementType().isF64())
    return op.emitOpError("expected positions of the segment, with 3 components of f64");
  for (Value gathered : op.getGathered())
    if (cast<FieldType>(gathered.getType()).getParticleSet() != relation.getParticleSet())
      return op.emitOpError("expected gathered fields of the segment");
  double cutoff = op.getCutoff().convertToDouble();
  if (!std::isfinite(cutoff) || cutoff <= 0.0)
    return op.emitOpError("expected a positive finite cutoff");
  Type result = op.getResult().getType();
  if (gather) {
    if (result != op.getPositions().getType())
      return op.emitOpError("expected a force field with the positions' type");
  } else {
    if (op.getSiteForces())
      return op.emitOpError("site forces are a layout of 'md.gather_segment_window' only");
    if (!result.isF64() &&
        result != VectorType::get({9}, Float64Type::get(op.getContext())))
      return op.emitOpError("expected f64 energy or vector<9xf64> strain");
  }
  return success();
}

template <typename Op> static LogicalResult verifyBody(Op op, bool gather) {
  SegmentShape shape = getSegmentShape(op);
  unsigned arity = shape.getArity(), count = shape.getNumCoordinates();
  Block &block = op.getKernel().front();
  Builder builder(op.getContext());
  Type vector = VectorType::get({3}, builder.getF64Type());
  SmallVector<Type> types(count, vector);
  for (Value field : op.getGathered())
    types.append(arity, cast<FieldType>(field.getType()).getKernelValueType());
  if (!llvm::equal(block.getArgumentTypes(), types))
    return op.emitOpError("expected the anchor displacement, ")
           << count - 1 << " frame offsets of the segment's kinds, then "
           << arity << " values of each gathered field";
  for (Operation &inner : block.without_terminator())
    if (inner.getNumRegions() || !isMemoryEffectFree(&inner))
      return op.emitOpError("expected a straight-line, effect-free window body");
  auto yield = dyn_cast<YieldOp>(block.getTerminator());
  unsigned stride = gather ? shape.getNumSlots(op.getSiteForces()) + 1 : 2;
  if (!yield || yield.getNumOperands() == 0 || yield.getNumOperands() % stride)
    return op.emitOpError("expected site contributions followed by their site displacements");
  unsigned pairs = yield.getNumOperands() / stride;
  Type contribution = gather ? vector : op.getResult().getType();
  unsigned contributions = pairs * (stride - 1);
  for (Value value : yield.getOperands().take_front(contributions))
    if (value.getType() != contribution)
      return op.emitOpError("site contribution type disagrees with the result");
  for (Value displacement : yield.getOperands().drop_front(contributions))
    if (displacement.getType() != vector ||
        !getSegmentWindowSites(displacement, block, shape))
      return op.emitOpError("cannot prove the site geometry of the segment: a "
                            "displacement is not the anchor displacement plus a "
                            "place of each kind, with the kinds' weights");
  return success();
}
} // namespace

std::optional<std::array<unsigned, 2>> mdir::md::getSegmentWindowSites(
    Value displacement, Block &block, const SegmentShape &shape) {
  SmallVector<double> c;
  if (!coefficients(displacement, block, shape.getNumCoordinates(), c) || c[0] != 1.0)
    return std::nullopt;
  unsigned ma = shape.a.getFrameSize();
  SmallVector<double> a(c.begin() + 1, c.begin() + ma), b;
  for (double coefficient : ArrayRef<double>(c).drop_front(ma)) b.push_back(-coefficient);
  auto p = identify(shape.a, a), q = identify(shape.b, b);
  if (!p || !q) return std::nullopt;
  return std::array<unsigned, 2>{*p, shape.a.getSize() + *q};
}

SmallVector<Value> mdir::md::emitSiteForceTranspose(
    OpBuilder &builder, Location loc, ValueRange values, const SegmentSide &side) {
  if (!side.kind || side.kind.getSites().empty()) return llvm::to_vector(values);
  RigidKindAttr kind = side.kind;
  auto vector = cast<VectorType>(values.front().getType());
  SmallVector<Value> real;
  for (auto [index, place] : llvm::enumerate(kind.getFrame().asArrayRef())) {
    Value force = values[place];
    for (int64_t site : kind.getSites().asArrayRef()) {
      Value weight = arith::ConstantOp::create(builder, loc,
          DenseElementsAttr::get(vector, FloatAttr::get(vector.getElementType(),
              kind.getSiteWeights(site)[index])));
      Value share = arith::MulFOp::create(builder, loc, values[site], weight);
      force = arith::AddFOp::create(builder, loc, force, share);
    }
    real.push_back(force);
  }
  return real;
}

LogicalResult SumSegmentWindowOp::verify() { return verifyOperands(*this, false); }
LogicalResult SumSegmentWindowOp::verifyRegions() { return verifyBody(*this, false); }
LogicalResult GatherSegmentWindowOp::verify() { return verifyOperands(*this, true); }
LogicalResult GatherSegmentWindowOp::verifyRegions() { return verifyBody(*this, true); }
