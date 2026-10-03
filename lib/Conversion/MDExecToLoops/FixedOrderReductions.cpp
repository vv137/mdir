// Sums the reductions of parallel loops over fixed chunks in a fixed order,
// so that the threads decide neither the order nor the grouping of the
// operations (D[threaded-determinism]).

#include "mdir/Conversion/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

using namespace mlir;

namespace mdir {

#define GEN_PASS_DEF_FIXEDORDERREDUCTIONS
#include "mdir/Conversion/Passes.h.inc"

namespace {

/// Applies the combiner `region` of a reduction to `lhs` and `rhs`.
Value combine(OpBuilder &builder, Region &region, Value lhs, Value rhs) {
  Block &block = region.front();
  IRMapping mapping;
  mapping.map(block.getArgument(0), lhs);
  mapping.map(block.getArgument(1), rhs);
  for (Operation &op : block.without_terminator())
    builder.clone(op, mapping);
  auto yield = cast<scf::ReduceReturnOp>(block.getTerminator());
  return mapping.lookupOrDefault(yield.getResult());
}

class FixedOrderReductions
    : public impl::FixedOrderReductionsBase<FixedOrderReductions> {
public:
  using FixedOrderReductionsBase::FixedOrderReductionsBase;

  void runOnOperation() final {
    SmallVector<scf::ParallelOp> loops;
    // Innermost first: an outer loop clones the rewritten inner one.
    getOperation().walk([&](scf::ParallelOp loop) {
      if (loop.getNumResults() != 0)
        loops.push_back(loop);
    });
    for (scf::ParallelOp loop : loops)
      if (failed(rewrite(loop)))
        return signalPassFailure();
  }

private:
  LogicalResult rewrite(scf::ParallelOp loop);
};

LogicalResult FixedOrderReductions::rewrite(scf::ParallelOp loop) {
  if (loop.getNumLoops() != 1)
    return loop.emitOpError()
           << "has a reduction over more than one dimension, which "
              "fixed-order-reductions does not order";
  for (Value result : loop.getResults())
    if (!getElementTypeOrSelf(result.getType()).isIntOrIndexOrFloat() ||
        !MemRefType::isValidElementType(result.getType()))
      return loop.emitOpError()
             << "reduces a value of the type " << result.getType()
             << ", which fixed-order-reductions does not order";
  // The partial results live in one buffer at the entry of the function:
  // the iterations of an enclosing parallel loop would share it.
  if (loop->getParentOfType<scf::ParallelOp>())
    return loop.emitOpError()
           << "has a reduction inside another parallel loop, whose "
              "iterations would share the partial results of "
              "fixed-order-reductions";
  auto function = loop->getParentOfType<FunctionOpInterface>();
  if (!function)
    return loop.emitOpError() << "is not inside a function";

  Location loc = loop.getLoc();
  auto reduce = cast<scf::ReduceOp>(loop.getBody()->getTerminator());
  unsigned count = loop.getNumResults();

  // The partial results of the chunks, one buffer for each value, at the
  // entry of the function (D117).
  SmallVector<Value> partials;
  {
    Block &entry = function.getFunctionBody().front();
    OpBuilder builder(&entry, entry.begin());
    for (Value result : loop.getResults())
      partials.push_back(memref::AllocaOp::create(
          builder, loc, MemRefType::get({chunks}, result.getType())));
  }

  OpBuilder builder(loop);
  Value lower = loop.getLowerBound()[0];
  Value upper = loop.getUpperBound()[0];
  Value step = loop.getStep()[0];
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  Value limit = arith::ConstantIndexOp::create(builder, loc, chunks);

  // The number of iterations, and of chunks: at most one a chunk.
  Value span = arith::SubIOp::create(builder, loc, upper, lower);
  Value positive = arith::MaxSIOp::create(builder, loc, span, zero);
  Value iterations = arith::CeilDivSIOp::create(builder, loc, positive, step);
  Value parts = arith::MinSIOp::create(builder, loc, iterations, limit);

  scf::ParallelOp::create(
      builder, loc, ValueRange{zero}, ValueRange{parts}, ValueRange{one},
      [&](OpBuilder &body, Location, ValueRange chunk) {
        // Chunk c holds the iterations [c n / p, (c + 1) n / p): contiguous,
        // none empty, and set by n and p alone.
        Value next = arith::AddIOp::create(body, loc, chunk[0], one);
        Value first = arith::DivSIOp::create(
            body, loc,
            arith::MulIOp::create(body, loc, chunk[0], iterations), parts);
        Value last = arith::DivSIOp::create(
            body, loc, arith::MulIOp::create(body, loc, next, iterations),
            parts);

        SmallVector<Value> inits(loop.getInitVals());
        inits.push_back(arith::ConstantIntOp::create(body, loc, 1, 1));
        auto fold = scf::ForOp::create(
            body, loc, first, last, one, inits,
            [&](OpBuilder &inner, Location, Value k, ValueRange carried) {
              // The induction variable of the original loop.
              Value index = arith::AddIOp::create(
                  inner, loc, lower,
                  arith::MulIOp::create(inner, loc, k, step));
              IRMapping mapping;
              mapping.map(loop.getInductionVars()[0], index);
              for (Operation &op : loop.getBody()->without_terminator())
                inner.clone(op, mapping);
              Value isFirst = carried.back();
              SmallVector<Value> yielded;
              for (unsigned i = 0; i != count; ++i) {
                Value value = mapping.lookupOrDefault(reduce.getOperands()[i]);
                Value sum = combine(inner, reduce.getReductions()[i],
                                    carried[i], value);
                // The first iteration of the chunk starts its fold.
                yielded.push_back(arith::SelectOp::create(inner, loc, isFirst,
                                                          value, sum));
              }
              yielded.push_back(arith::ConstantIntOp::create(inner, loc, 0, 1));
              scf::YieldOp::create(inner, loc, yielded);
            });
        for (unsigned i = 0; i != count; ++i)
          memref::StoreOp::create(body, loc, fold.getResult(i), partials[i],
                                  chunk[0]);
        scf::ReduceOp::create(body, loc);
      });

  // The initial values, then the chunks in their order.
  auto total = scf::ForOp::create(
      builder, loc, zero, parts, one, loop.getInitVals(),
      [&](OpBuilder &serial, Location, Value chunk, ValueRange carried) {
        SmallVector<Value> yielded;
        for (unsigned i = 0; i != count; ++i) {
          Value partial =
              memref::LoadOp::create(serial, loc, partials[i], chunk);
          yielded.push_back(combine(serial, reduce.getReductions()[i],
                                    carried[i], partial));
        }
        scf::YieldOp::create(serial, loc, yielded);
      });

  loop.replaceAllUsesWith(total.getResults());
  loop.erase();
  return success();
}

} // namespace

} // namespace mdir
