// Pair-local periodic image choices for reduced cells [AllenTildesley2017].
#include "mdir/Conversion/ImageEnumerationEmitter.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include <cassert>

using namespace mlir;
using namespace mdir::kernels;

llvm::SmallVector<Value> mdir::kernels::emitPairImageChoices(
    OpBuilder &b, Location loc, Value d, Value cell, Value paddedReach,
    PairImageStrategy strategy, Value missingPayload,
    llvm::function_ref<Value(OpBuilder &, Value)> visit) {
  auto real = [&](double n) { return createReal(b, loc, b.getF64Type(), n); };
  auto add = [&](Value x, Value y) -> Value {
    return arith::AddFOp::create(b, loc, x, y);
  };
  auto sub = [&](Value x, Value y) -> Value {
    return arith::SubFOp::create(b, loc, x, y);
  };
  auto mul = [&](Value x, Value y) -> Value {
    return arith::MulFOp::create(b, loc, x, y);
  };
  auto elt = [&](Value x, unsigned i) -> Value {
    return vector::ExtractOp::create(b, loc, x, i);
  };
  auto both = [&](Value x, Value y) -> Value {
    return arith::AndIOp::create(b, loc, x, y);
  };
  Value yes = arith::ConstantIntOp::create(b, loc, 1, 1);
  Value ax = elt(cell, 0), by = elt(cell, 1), cz = elt(cell, 2);
  Value bx = isTriclinic(cell) ? elt(cell, 3) : real(0);
  Value cx = isTriclinic(cell) ? elt(cell, 4) : real(0);
  Value cy = isTriclinic(cell) ? elt(cell, 5) : real(0);
  Value inverse = createInverse(b, loc, cell);
  auto nearest = [&](Value x, unsigned axis) -> Value {
    return math::RoundEvenOp::create(b, loc, mul(x, elt(inverse, axis)));
  };
  auto alternate = [&](Value n, Value residue) -> Value {
    Value positive = arith::CmpFOp::create(
        b, loc, arith::CmpFPredicate::OGE, residue, real(0));
    return add(n, arith::SelectOp::create(b, loc, positive, real(1), real(-1)));
  };
  auto needed = [&](Value diagonal, Value residue) -> Value {
    return arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLE,
        sub(diagonal, math::AbsFOp::create(b, loc, residue)), paddedReach);
  };
  unsigned choices = strategy == PairImageStrategy::ConditionalNeighbors ? 2 : 1;
  llvm::SmallVector<Value> results;
  Value nc0 = nearest(elt(d, 2), 2);
  Value z0 = sub(elt(d, 2), mul(nc0, cz));
  for (unsigned ic = 0; ic < choices; ++ic) {
    Value nc = ic ? alternate(nc0, z0) : nc0;
    Value vc = ic ? needed(cz, z0) : yes;
    Value dy = sub(elt(d, 1), mul(nc, cy));
    Value nb0 = nearest(dy, 1), y0 = sub(dy, mul(nb0, by));
    for (unsigned ib = 0; ib < choices; ++ib) {
      Value nb = ib ? alternate(nb0, y0) : nb0;
      Value vb = ib ? both(vc, needed(by, y0)) : vc;
      Value dx = sub(sub(elt(d, 0), mul(nc, cx)), mul(nb, bx));
      Value na0 = nearest(dx, 0), x0 = sub(dx, mul(na0, ax));
      for (unsigned ia = 0; ia < choices; ++ia) {
        Value na = ia ? alternate(na0, x0) : na0;
        Value valid = ia ? both(vb, needed(ax, x0)) : vb;
        auto leaf = scf::IfOp::create(b, loc, valid,
            [&](OpBuilder &inside, Location) {
              Value query = vector::FromElementsOp::create(inside, loc,
                  d.getType(), ValueRange{na, nb, nc});
              Value payload = visit(inside, query);
              assert(payload.getType() == missingPayload.getType());
              scf::YieldOp::create(inside, loc, ValueRange{payload});
            },
            [&](OpBuilder &outside, Location) {
              scf::YieldOp::create(outside, loc, ValueRange{missingPayload});
            });
        results.push_back(leaf.getResult(0));
      }
    }
  }
  return results;
}
