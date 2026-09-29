// Conversion of md and dyn ops to md_exec loops.
//
// See docs/ops-m0.md, Section 9.

#include "mdir/Conversion/Passes.h"

#include "mdir/Dialect/Dyn/DynOps.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MDExec/MDExecDialect.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mdir/Dialect/MDRT/MDRTDialect.h"
#include "mdir/Dialect/MDRT/MDRTTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;
using namespace mdir;

namespace {

class Converter {
public:
  Converter(MLIRContext *context, double skin, int64_t width)
      : builder(context), skin(skin), width(width) {}

  LogicalResult convert(Operation *op);

  /// Erases the ops that have been converted.
  LogicalResult finish();

private:
  LogicalResult convertNeighborhood(md::NeighborhoodOp op);
  template <typename OpTy>
  LogicalResult convertPairOp(OpTy op, bool isSum);
  template <typename OpTy>
  LogicalResult convertParticleOp(OpTy op, bool isSum);
  LogicalResult convertKick(dyn::KickOp op);
  LogicalResult convertDrift(dyn::DriftOp op);

  /// A zero of type `type`, which is f64 or a vector of f64.
  Value createZero(Location loc, Type type) {
    return arith::ConstantOp::create(builder, loc, type,
                                     cast<TypedAttr>(builder.getZeroAttr(type)));
  }

  /// Gives `op`, which has no kernel yet, a block with arguments of the
  /// types `arguments`, and returns the block.
  Block *addKernel(Operation *op, ArrayRef<Type> arguments) {
    Block *block = new Block();
    op->getRegion(0).push_back(block);
    for (Type type : arguments)
      block->addArgument(type, op->getLoc());
    return block;
  }

  /// Copies the ops of `source` to the end of `target` and ends `target`
  /// with a yield of the values that `source` yields. `mapping` maps the
  /// arguments of `source` to values in `target`.
  void copyKernel(Block &source, Block *target, IRMapping &mapping) {
    OpBuilder kernel = OpBuilder::atBlockEnd(target);
    for (Operation &op : source.without_terminator())
      kernel.clone(op, mapping);
    SmallVector<Value> yielded;
    for (Value value : source.getTerminator()->getOperands())
      yielded.push_back(mapping.lookupOrDefault(value));
    md_exec::YieldOp::create(kernel, source.getTerminator()->getLoc(),
                             yielded);
  }

  OpBuilder builder;
  double skin;
  int64_t width;

  /// The neighbor structure that was built for a relation.
  llvm::DenseMap<Value, Value> neighbors;
  SmallVector<Operation *> converted;
};

} // namespace

//===----------------------------------------------------------------------===//
// Neighborhoods
//===----------------------------------------------------------------------===//

LogicalResult Converter::convertNeighborhood(md::NeighborhoodOp op) {
  Location loc = op.getLoc();
  MLIRContext *context = builder.getContext();
  auto relation = cast<md::RelationType>(op.getResult().getType());
  FlatSymbolRefAttr particleSet = relation.getParticleSet();
  double cutoff = op.getCutoff().convertToDouble();

  builder.setInsertionPoint(op);
  Value cells = md_exec::BuildCellsOp::create(
      builder, loc, mdrt::CellsType::get(context, particleSet),
      op.getPositions(), op.getCell(), APFloat(cutoff + skin));
  Value structure = md_exec::BuildNeighborsOp::create(
      builder, loc, mdrt::NeighborsType::get(context, particleSet), cells,
      op.getPositions(), op.getCell(), APFloat(cutoff), APFloat(skin),
      md_exec::NeighborKind::Matrix, static_cast<uint64_t>(width));

  neighbors[op.getResult()] = structure;
  converted.push_back(op);
  return success();
}

//===----------------------------------------------------------------------===//
// Loops over pairs
//===----------------------------------------------------------------------===//

template <typename OpTy>
LogicalResult Converter::convertPairOp(OpTy op, bool isSum) {
  Location loc = op.getLoc();

  auto neighborhood =
      op.getRelation().template getDefiningOp<md::NeighborhoodOp>();
  if (!neighborhood)
    return op.emitOpError() << "cannot convert a loop over a relation that "
                               "is not the result of 'md.neighborhood'";
  Value structure = neighbors.lookup(op.getRelation());
  assert(structure && "the neighborhood is converted before its users");
  double cutoff = neighborhood.getCutoff().convertToDouble();
  auto relation = cast<md::RelationType>(op.getRelation().getType());

  builder.setInsertionPoint(op);
  Type resultType = op.getResult().getType();
  SmallVector<Value, 1> outs, reduce;
  DenseF64ArrayAttr weights;
  if (isSum) {
    reduce.push_back(createZero(loc, resultType));
    // Each unordered pair is visited in both directions.
    if (relation.getOrientation() == md::Orientation::Unordered)
      weights = builder.getDenseF64ArrayAttr({0.5});
  } else {
    outs.push_back(md_exec::ZerosOp::create(builder, loc, resultType));
  }

  auto loop = md_exec::PairForOp::create(
      builder, loc, TypeRange(resultType), structure, op.getPositions(),
      op.getCell(), op.getGathered(), outs, reduce, /*scratch=*/ValueRange(),
      builder.getF64FloatAttr(cutoff), weights,
      /*overwrite=*/DenseBoolArrayAttr());

  // The kernel receives the squared distance; the semantic kernel is
  // written in terms of the distance.
  Block &source = op.getKernel().front();
  SmallVector<Type> arguments;
  for (BlockArgument argument : source.getArguments())
    arguments.push_back(argument.getType());
  Block *target = addKernel(loop, arguments);

  IRMapping mapping;
  OpBuilder kernel = OpBuilder::atBlockEnd(target);
  Value distance = target->getArgument(0);
  if (!source.getArgument(0).use_empty())
    distance = math::SqrtOp::create(kernel, loc, target->getArgument(0));
  mapping.map(source.getArgument(0), distance);
  for (unsigned i = 1, e = source.getNumArguments(); i != e; ++i)
    mapping.map(source.getArgument(i), target->getArgument(i));
  copyKernel(source, target, mapping);

  op.getResult().replaceAllUsesWith(loop.getResult(0));
  converted.push_back(op);
  return success();
}

//===----------------------------------------------------------------------===//
// Loops over particles
//===----------------------------------------------------------------------===//

template <typename OpTy>
LogicalResult Converter::convertParticleOp(OpTy op, bool isSum) {
  Location loc = op.getLoc();
  builder.setInsertionPoint(op);

  Type resultType = op.getResult().getType();
  SmallVector<Value, 1> outs, reduce;
  if (isSum)
    reduce.push_back(createZero(loc, resultType));
  else
    outs.push_back(md_exec::EmptyOp::create(builder, loc, resultType));

  auto loop = md_exec::ParticleForOp::create(
      builder, loc, TypeRange(resultType), op.getGathered(), outs, reduce,
      /*scratch=*/ValueRange());

  Block &source = op.getKernel().front();
  SmallVector<Type> arguments;
  for (BlockArgument argument : source.getArguments())
    arguments.push_back(argument.getType());
  Block *target = addKernel(loop, arguments);

  IRMapping mapping;
  for (unsigned i = 0, e = source.getNumArguments(); i != e; ++i)
    mapping.map(source.getArgument(i), target->getArgument(i));
  copyKernel(source, target, mapping);

  op.getResult().replaceAllUsesWith(loop.getResult(0));
  converted.push_back(op);
  return success();
}

/// v' = v + dt · f / m
LogicalResult Converter::convertKick(dyn::KickOp op) {
  Location loc = op.getLoc();
  builder.setInsertionPoint(op);

  auto field = cast<md::FieldType>(op.getResult().getType());
  Type vector = field.getKernelValueType();
  Type real = field.getElementType();

  Value destination = md_exec::EmptyOp::create(builder, loc, field);
  auto loop = md_exec::ParticleForOp::create(
      builder, loc, TypeRange(field),
      ValueRange{op.getVelocities(), op.getForces(), op.getMasses()},
      ValueRange{destination}, /*reduce=*/ValueRange(),
      /*scratch=*/ValueRange());

  Block *block = addKernel(loop, {vector, vector, real});
  OpBuilder kernel = OpBuilder::atBlockEnd(block);
  Value factor =
      arith::DivFOp::create(kernel, loc, op.getDt(), block->getArgument(2));
  Value broadcast = vector::BroadcastOp::create(kernel, loc, vector, factor);
  Value change =
      arith::MulFOp::create(kernel, loc, broadcast, block->getArgument(1));
  Value updated =
      arith::AddFOp::create(kernel, loc, block->getArgument(0), change);
  md_exec::YieldOp::create(kernel, loc, ValueRange{updated});

  op.getResult().replaceAllUsesWith(loop.getResult(0));
  converted.push_back(op);
  return success();
}

/// x' = x + dt · v
LogicalResult Converter::convertDrift(dyn::DriftOp op) {
  Location loc = op.getLoc();
  builder.setInsertionPoint(op);

  auto field = cast<md::FieldType>(op.getResult().getType());
  Type vector = field.getKernelValueType();

  Value destination = md_exec::EmptyOp::create(builder, loc, field);
  auto loop = md_exec::ParticleForOp::create(
      builder, loc, TypeRange(field),
      ValueRange{op.getPositions(), op.getVelocities()},
      ValueRange{destination}, /*reduce=*/ValueRange(),
      /*scratch=*/ValueRange());

  Block *block = addKernel(loop, {vector, vector});
  OpBuilder kernel = OpBuilder::atBlockEnd(block);
  Value broadcast =
      vector::BroadcastOp::create(kernel, loc, vector, op.getDt());
  Value change =
      arith::MulFOp::create(kernel, loc, broadcast, block->getArgument(1));
  Value updated =
      arith::AddFOp::create(kernel, loc, block->getArgument(0), change);
  md_exec::YieldOp::create(kernel, loc, ValueRange{updated});

  op.getResult().replaceAllUsesWith(loop.getResult(0));
  converted.push_back(op);
  return success();
}

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

LogicalResult Converter::convert(Operation *op) {
  if (auto evaluate = dyn_cast<md::EvaluateOp>(op))
    return evaluate.emitOpError()
           << "cannot be converted; run 'md-differentiate' first";

  if (auto neighborhood = dyn_cast<md::NeighborhoodOp>(op))
    return convertNeighborhood(neighborhood);

  if (auto sum = dyn_cast<md::SumRelationOp>(op)) {
    if (sum.getTruncation() != md::Truncation::None)
      return sum.emitOpError() << "cannot convert a sum with a truncation; "
                                  "run 'md-expand-truncation' first";
    return convertPairOp(sum, /*isSum=*/true);
  }
  if (auto gather = dyn_cast<md::GatherRelationOp>(op))
    return convertPairOp(gather, /*isSum=*/false);

  if (auto sum = dyn_cast<md::SumParticlesOp>(op))
    return convertParticleOp(sum, /*isSum=*/true);
  if (auto map = dyn_cast<md::MapParticlesOp>(op))
    return convertParticleOp(map, /*isSum=*/false);

  if (auto kick = dyn_cast<dyn::KickOp>(op))
    return convertKick(kick);
  if (auto drift = dyn_cast<dyn::DriftOp>(op))
    return convertDrift(drift);

  return success();
}

LogicalResult Converter::finish() {
  // Users come after what they use, so erase from the back.
  for (Operation *op : llvm::reverse(converted)) {
    if (!op->use_empty())
      return op->emitOpError()
             << "cannot be converted: its result is used by an op that was "
                "not converted";
    op->erase();
  }
  return success();
}

namespace mdir {

#define GEN_PASS_DEF_CONVERTMDTOMDEXEC
#include "mdir/Conversion/Passes.h.inc"

namespace {
class ConvertMDToMDExec
    : public impl::ConvertMDToMDExecBase<ConvertMDToMDExec> {
public:
  using impl::ConvertMDToMDExecBase<
      ConvertMDToMDExec>::ConvertMDToMDExecBase;

  void runOnOperation() final {
    if (skin < 0.0) {
      getOperation()->emitError() << "expected a skin that is not negative";
      return signalPassFailure();
    }
    if (width <= 0) {
      getOperation()->emitError() << "expected a positive width";
      return signalPassFailure();
    }

    SmallVector<Operation *> ops;
    getOperation()->walk<WalkOrder::PreOrder>(
        [&](Operation *op) { ops.push_back(op); });

    Converter converter(&getContext(), skin, width);
    for (Operation *op : ops)
      if (failed(converter.convert(op)))
        return signalPassFailure();
    if (failed(converter.finish()))
      signalPassFailure();
  }
};
} // namespace

} // namespace mdir
