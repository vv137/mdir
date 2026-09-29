// Expansion of truncation attributes into kernels.
//
// See docs/ops-m0.md, Section 4.8.

#include "mdir/Dialect/MD/Transforms/Passes.h"
#include "mdir/Dialect/MD/Transforms/ScalarDerivative.h"
#include "mdir/Dialect/MD/Transforms/Truncation.h"

#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;
using namespace mdir::md;

/// Emits the switching function
///
///   S(r) = 1                        for r <= from
///   S(r) = 1 − 10t³ + 15t⁴ − 6t⁵    otherwise, t = (r − from) / (cutoff − from)
static Value emitSwitch(ScalarEmitter &emit, Value r, double from,
                        double cutoff) {
  OpBuilder &builder = emit.builder;
  Location loc = emit.loc;

  Value start = emit.constantLike(from, r);
  Value width = emit.constantLike(cutoff - from, r);
  Value t = emit.div(emit.sub(r, start), width);

  // 1 + t³ · (−10 + t · (15 − 6t)), in Horner form.
  Value inner = emit.sub(emit.constantLike(15.0, r), emit.scale(6.0, t));
  Value middle = emit.add(emit.constantLike(-10.0, r), emit.mul(t, inner));
  Value cube = emit.mul(emit.mul(t, t), t);
  Value polynomial =
      emit.add(emit.constantLike(1.0, r), emit.mul(cube, middle));

  Value below = arith::CmpFOp::create(builder, loc,
                                      arith::CmpFPredicate::OLE, r, start);
  return arith::SelectOp::create(builder, loc, below,
                                 emit.constantLike(1.0, r), polynomial);
}

/// Emits the force-switched energy
///
///   u(r) − C                                    for r <= from
///   u(r) − (A/3)(r − from)³ − (B/4)(r − from)⁴ − C    otherwise
///
/// A cubic polynomial in `r − from` is added to the force so that the force
/// and its derivative vanish at the cutoff. With F = −u'(cutoff),
/// F' = −u''(cutoff), and Δ = cutoff − from,
///
///   A = (F'Δ − 3F) / Δ²      B = (2F − F'Δ) / Δ³
///   C = u(cutoff) − (A/3)Δ³ − (B/4)Δ⁴
///
/// For a power law this is the force switch of GROMACS.
static Value emitForceSwitch(ScalarEmitter &emit, Value r, Value energy,
                             Value energyAtCutoff, Value slopeAtCutoff,
                             Value curvatureAtCutoff, double from,
                             double cutoff) {
  OpBuilder &builder = emit.builder;
  Location loc = emit.loc;
  double width = cutoff - from;

  Value force = emit.neg(slopeAtCutoff);
  Value forceSlope = emit.neg(curvatureAtCutoff);

  // A / 3 and B / 4.
  Value a = emit.scale(1.0 / (3.0 * width * width),
                       emit.sub(emit.scale(width, forceSlope),
                                emit.scale(3.0, force)));
  Value b = emit.scale(1.0 / (4.0 * width * width * width),
                       emit.sub(emit.scale(2.0, force),
                                emit.scale(width, forceSlope)));

  // (A/3) t³ + (B/4) t⁴ = t³ · (A/3 + (B/4) t)
  auto polynomial = [&](Value t) {
    Value cube = emit.mul(emit.mul(t, t), t);
    return emit.mul(cube, emit.add(a, emit.mul(b, t)));
  };

  Value atEnd = polynomial(emit.constantLike(width, r));
  Value offset = emit.sub(energyAtCutoff, atEnd);

  Value start = emit.constantLike(from, r);
  Value inside = polynomial(emit.sub(r, start));
  Value switched;
  if (inside) {
    Value below = arith::CmpFOp::create(builder, loc,
                                        arith::CmpFPredicate::OLE, r, start);
    switched = arith::SelectOp::create(builder, loc, below,
                                       emit.constantLike(0.0, r), inside);
  }
  return emit.sub(emit.sub(energy, switched), offset);
}

LogicalResult mdir::md::expandTruncation(SumRelationOp op) {
  Truncation truncation = op.getTruncation();
  if (truncation == Truncation::None)
    return success();

  auto neighborhood = op.getRelation().getDefiningOp<NeighborhoodOp>();
  if (!neighborhood)
    return op.emitOpError() << "a truncation needs a cutoff, but the relation "
                               "is not the result of 'md.neighborhood'";
  double cutoff = neighborhood.getCutoff().convertToDouble();

  Block &block = op.getKernel().front();
  auto yield = cast<YieldOp>(block.getTerminator());
  Value energy = yield.getOperand(0);
  if (!energy.getType().isF64())
    return op.emitOpError()
           << "a truncation requires a kernel that yields f64, got "
           << energy.getType();
  Value r = block.getArgument(0);

  // The ops of the kernel as it is now, before anything is added.
  SmallVector<Operation *> original;
  for (Operation &nested : block.without_terminator())
    original.push_back(&nested);

  OpBuilder builder(yield);
  ScalarEmitter emit(builder, op.getLoc());
  Value truncated;

  if (truncation == Truncation::Switch) {
    double from = op.getSwitchFrom()->convertToDouble();
    truncated = emit.mul(energy, emitSwitch(emit, r, from, cutoff));
  } else {
    // Evaluate the kernel a second time, at the cutoff.
    Value atCutoff = emit.constantLike(cutoff, r);
    IRMapping mapping;
    mapping.map(r, atCutoff);
    for (Operation *nested : original)
      builder.clone(*nested, mapping);
    Value energyAtCutoff = mapping.lookupOrDefault(energy);

    // The first and second derivatives at the cutoff.
    Value slope, curvature;
    if (truncation != Truncation::Shift) {
      ScalarDerivative first(builder, atCutoff);
      if (failed(first.get(energyAtCutoff, slope)))
        return failure();
    }
    if (truncation == Truncation::ForceSwitch && slope) {
      ScalarDerivative second(builder, atCutoff);
      if (failed(second.get(slope, curvature)))
        return failure();
    }

    if (truncation == Truncation::ForceSwitch) {
      double from = op.getSwitchFrom()->convertToDouble();
      truncated = emitForceSwitch(emit, r, energy, energyAtCutoff, slope,
                                  curvature, from, cutoff);
    } else {
      truncated = emit.sub(energy, energyAtCutoff);
      if (truncation == Truncation::ForceShift) {
        Value distance = emit.sub(r, atCutoff);
        truncated = emit.sub(truncated, emit.mul(distance, slope));
      }
    }
  }

  yield.setOperand(0, truncated);
  eraseDeadOps(block);

  op.setTruncation(Truncation::None);
  op.removeSwitchFromAttr();
  return success();
}

namespace mdir {
namespace md {

#define GEN_PASS_DEF_EXPANDTRUNCATION
#include "mdir/Dialect/MD/Transforms/Passes.h.inc"

namespace {
class ExpandTruncation : public impl::ExpandTruncationBase<ExpandTruncation> {
public:
  using impl::ExpandTruncationBase<ExpandTruncation>::ExpandTruncationBase;

  void runOnOperation() final {
    SmallVector<SumRelationOp> ops;
    getOperation()->walk([&](SumRelationOp op) { ops.push_back(op); });
    for (SumRelationOp op : ops)
      if (failed(expandTruncation(op)))
        return signalPassFailure();
  }
};
} // namespace

} // namespace md
} // namespace mdir
