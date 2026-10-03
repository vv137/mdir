// Restricted neighbor-lane vectorization for the CPU prototype.
#include "CPUVectorPairs.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/DenseMap.h"
using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;

static unsigned components(Type t) {
  if (auto v = dyn_cast<VectorType>(t))
    return v.getNumElements();
  return 1;
}
LogicalResult mdir::checkCPUSIMDKernel(md_exec::PairForOp op) {
  if (!op.getIns().empty())
    return op.emitOpError("CPU SIMD currently requires a kernel without ins");
  for (Operation &nested : op.getKernel().front())
    for (Type type : nested.getOperandTypes()) {
      if (auto vector = dyn_cast<VectorType>(type)) {
        if (vector.getRank() != 1 || vector.isScalable())
          return op.emitOpError(
              "CPU SIMD requires fixed one-dimensional vectors");
        type = vector.getElementType();
      }
      if (!type.isF32() && !type.isF64())
        return op.emitOpError("CPU SIMD requires f32 or f64 kernel values");
    }
  for (Operation &nested : op.getKernel().front().without_terminator()) {
    StringRef name = nested.getName().getStringRef();
    if (!isa<arith::ConstantOp, vector::BroadcastOp, vector::ExtractOp,
             vector::FromElementsOp>(nested) &&
        name != "arith.addf" && name != "arith.subf" && name != "arith.mulf" &&
        name != "arith.divf" && name != "arith.negf" && name != "arith.extf" &&
        name != "arith.truncf")
      return op.emitOpError("unsupported CPU SIMD kernel operation: ") << name;
    if (nested.getNumResults() != 1 || nested.getNumRegions())
      return op.emitOpError(
          "CPU SIMD requires single-result elementwise operations");
    if (auto ex = dyn_cast<vector::ExtractOp>(nested))
      if (!ex.getDynamicPosition().empty() ||
          ex.getStaticPosition().size() != 1)
        return op.emitOpError(
            "CPU SIMD requires a constant one-dimensional extract");
    if (auto c = dyn_cast<arith::ConstantOp>(nested))
      if (!isa<FloatAttr>(c.getValue()))
        return op.emitOpError(
            "CPU SIMD currently requires scalar floating constants");
    for (Value v : nested.getOperands())
      if (v.getParentBlock() != &op.getKernel().front())
        return op.emitOpError("CPU SIMD requires constants inside the kernel");
  }
  return success();
}

SmallVector<Value> mdir::emitCPUSIMDPairs(OpBuilder &b, md_exec::PairForOp op,
                                          Value counts, Value entries,
                                          Value box, Value inverse,
                                          Value central, int64_t width) {
  Location loc = op.getLoc();
  Block &kernel = op.getKernel().front();
  Operation *yield = kernel.getTerminator();
  Type real = kernel.getArgument(0).getType();
  auto laneType = [&](Type t) {
    return VectorType::get({width}, getElementTypeOrSelf(t));
  };
  Value zero = createIndex(b, loc, 0);
  Value step = createIndex(b, loc, width);
  Value count = memref::LoadOp::create(b, loc, counts, ValueRange{central});
  Value end = arith::IndexCastOp::create(b, loc, b.getIndexType(), count);
  Value center = loadElement(b, loc, op.getPositions(), central);
  box = convertReal(b, loc, box, real);
  inverse = convertReal(b, loc, inverse, real);
  SmallVector<Value> initial;
  for (Value v : yield->getOperands())
    for (unsigned c = 0; c < components(v.getType()); ++c)
      initial.push_back(createZero(b, loc, laneType(v.getType())));
  auto loop = scf::ForOp::create(
      b, loc, zero, end, step, initial,
      [&](OpBuilder &p, Location, Value base, ValueRange partial) {
        SmallVector<Value> rs, masks;
        SmallVector<Value> ds[3];
        for (int64_t lane = 0; lane < width; ++lane) {
          Value at =
              arith::AddIOp::create(p, loc, base, createIndex(p, loc, lane));
          Value valid =
              arith::CmpIOp::create(p, loc, arith::CmpIPredicate::ult, at, end);
          auto guarded = scf::IfOp::create(
              p, loc, valid,
              [&](OpBuilder &g, Location) {
                Value idx = memref::LoadOp::create(g, loc, entries,
                                                   ValueRange{central, at});
                Value other =
                    arith::IndexCastOp::create(g, loc, g.getIndexType(), idx);
                Value raw = arith::SubFOp::create(
                    g, loc, center,
                    loadElement(g, loc, op.getPositions(), other));
                Value d = emitMinimumImage(
                    g, loc, convertReal(g, loc, raw, real), box, inverse);
                Value sq = arith::MulFOp::create(g, loc, d, d);
                Value r2 = vector::ReductionOp::create(
                    g, loc, vector::CombiningKind::ADD, sq);
                double rc = op.getCutoff().convertToDouble();
                Value inside =
                    arith::CmpFOp::create(g, loc, arith::CmpFPredicate::OLT, r2,
                                          createCutoff2(g, loc, real, rc));
                Value distinct = arith::CmpIOp::create(
                    g, loc, arith::CmpIPredicate::ne, other, central);
                inside = arith::AndIOp::create(g, loc, inside, distinct);
                // Mask before division: inactive lanes never evaluate a zero
                // distance.
                Value safe = arith::SelectOp::create(
                    g, loc, inside, r2, createReal(g, loc, real, 1));
                scf::YieldOp::create(g, loc, ValueRange{safe, d, inside});
              },
              [&](OpBuilder &g, Location) {
                scf::YieldOp::create(
                    g, loc,
                    ValueRange{createReal(g, loc, real, 1),
                               createZero(g, loc, VectorType::get({3}, real)),
                               arith::ConstantIntOp::create(g, loc, 0, 1)});
              });
          rs.push_back(guarded.getResult(0));
          masks.push_back(guarded.getResult(2));
          for (int64_t c = 0; c < 3; ++c)
            ds[c].push_back(
                vector::ExtractOp::create(p, loc, guarded.getResult(1), c));
        }
        llvm::DenseMap<Value, SmallVector<Value>> map;
        map[kernel.getArgument(0)] = {
            vector::FromElementsOp::create(p, loc, laneType(real), rs)};
        for (auto &d : ds)
          map[kernel.getArgument(1)].push_back(
              vector::FromElementsOp::create(p, loc, laneType(real), d));
        for (Operation &n : kernel.without_terminator()) {
          SmallVector<Value> result;
          if (auto c = dyn_cast<arith::ConstantOp>(n)) {
            result.push_back(vector::BroadcastOp::create(
                p, loc, laneType(c.getType()), p.clone(n)->getResult(0)));
          } else if (auto ex = dyn_cast<vector::ExtractOp>(n)) {
            result.push_back(map[ex.getSource()][ex.getStaticPosition()[0]]);
          } else if (auto bc = dyn_cast<vector::BroadcastOp>(n)) {
            result.assign(components(bc.getType()), map[bc.getSource()][0]);
          } else if (isa<vector::FromElementsOp>(n)) {
            for (Value operand : n.getOperands())
              result.push_back(map[operand][0]);
          } else {
            for (unsigned c = 0; c < components(n.getResult(0).getType());
                 ++c) {
              OperationState state(loc, n.getName());
              for (Value operand : n.getOperands())
                state.addOperands(map[operand][c]);
              state.addTypes(laneType(n.getResult(0).getType()));
              state.addAttributes(n.getAttrs());
              result.push_back(p.create(state)->getResult(0));
            }
          }
          map[n.getResult(0)] = result;
        }
        Value mask = vector::FromElementsOp::create(
            p, loc, VectorType::get({width}, p.getI1Type()), masks);
        SmallVector<Value> next;
        unsigned i = 0;
        for (Value v : yield->getOperands())
          for (Value component : map[v]) {
            Value masked = arith::SelectOp::create(
                p, loc, mask, component,
                createZero(p, loc, component.getType()));
            next.push_back(arith::AddFOp::create(p, loc, partial[i++], masked));
          }
        scf::YieldOp::create(p, loc, next);
      });
  SmallVector<Value> totals;
  unsigned at = 0;
  for (Value v : yield->getOperands()) {
    SmallVector<Value> cs;
    for (unsigned c = 0; c < components(v.getType()); ++c)
      cs.push_back(vector::ReductionOp::create(
          b, loc, vector::CombiningKind::ADD, loop.getResult(at++)));
    totals.push_back(
        isa<VectorType>(v.getType())
            ? Value(vector::FromElementsOp::create(b, loc, v.getType(), cs))
            : cs[0]);
  }
  for (unsigned i = 0; i < op.getOuts().size(); ++i) {
    Value total = totals[i];
    if (!op.overwrites(i))
      total = arith::AddFOp::create(
          b, loc, total, loadElement(b, loc, op.getOuts()[i], central));
    storeElement(b, loc, total, op.getOuts()[i], central);
  }
  SmallVector<Value> reductions;
  for (unsigned i = op.getOuts().size(); i < totals.size(); ++i) {
    Value total = totals[i];
    if (auto weights = op.getWeights()) {
      double weight = (*weights)[i - op.getOuts().size()];
      Value scalar =
          createReal(b, loc, getElementTypeOrSelf(total.getType()), weight);
      Value factor = scalar;
      if (isa<VectorType>(total.getType()))
        factor = vector::BroadcastOp::create(b, loc, total.getType(), scalar);
      total = arith::MulFOp::create(b, loc, total, factor);
    }
    reductions.push_back(total);
  }
  return reductions;
}
