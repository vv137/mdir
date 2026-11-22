// Transpose the prolate series so adjacent lanes read adjacent coefficients
// (D266).
#include "DevicePME.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;

void DevicePME::emitCoefficients(StringRef symbol) {
  OpBuilder::InsertionGuard functionGuard(b);
  SmallVector<Type> inputs;
  inputs.push_back(
      globalMem({ShapedType::kDynamic, ShapedType::kDynamic}, f64Type));
  inputs.push_back(globalMem({ShapedType::kDynamic}, realType));
  inputs.append(3, idxType);
  auto fn = func::FuncOp::create(b, loc, symbol, b.getFunctionType(inputs, {}));
  fn.setPrivate();
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  unsigned arg = 0;
  Value moduli = entry->getArgument(arg++);
  Value tables = entry->getArgument(arg++);
  Value k1 = entry->getArgument(arg++);
  Value k2 = entry->getArgument(arg++);
  Value k3 = entry->getArgument(arg++);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  Value c256 = arith::ConstantIndexOp::create(b, loc, 256);
  Value c3 = arith::ConstantIndexOp::create(b, loc, 3);
  Value k12 = addi(k1, k2);
  Value k123 = addi(k12, k3);
  Value pe = muli(k123, c3);
  {
    auto launch = gpu::LaunchOp::create(b, loc, c1, c1, c1, c256, c1, c1);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value tx = launch.getThreadIds().x;
    Value i0 = arith::ConstantIndexOp::create(b, loc, 0);
    Value i1 = arith::ConstantIndexOp::create(b, loc, 1);
    Value i2 = arith::ConstantIndexOp::create(b, loc, 2);
    Value i3 = arith::ConstantIndexOp::create(b, loc, 3);
    Value i4 = arith::ConstantIndexOp::create(b, loc, 4);
    Value i256 = arith::ConstantIndexOp::create(b, loc, 256);
    Value points =
        arith::ConstantOp::create(b, loc, b.getIntegerAttr(idxType, order));
    Value df = load(moduli, {i4, i0});
    Value nf = load(moduli, {i4, i2});
    Value scale = load(moduli, {i4, i3});
    Value di = fptosi(i64Type, df);
    Value d = indexCast(idxType, di);
    Value ni = fptosi(i64Type, nf);
    Value n = indexCast(idxType, ni);
    Value n2 = addi(n, n);
    Value pd = muli(points, d);
    Value pd2 = addi(pd, pd);
    Value series = addi(i3, n2);
    Value count = addi(series, pd2);
    scf::ForOp::create(
        b, loc, tx, count, i256, ValueRange{},
        [&](OpBuilder &b, Location, Value t, ValueRange carried) {
          Value meta = cmpi(arith::CmpIPredicate::ult, t, i3);
          Value influence = cmpi(arith::CmpIPredicate::ult, t, series);
          auto valueOp =
              scf::IfOp::create(b, loc, TypeRange{f64Type}, meta, true, true);
          {
            OpBuilder::InsertionGuard guard(b);
            b.setInsertionPointToStart(valueOp.thenBlock());
            Value is_n = cmpi(arith::CmpIPredicate::eq, t, i0);
            Value is_scale = cmpi(arith::CmpIPredicate::eq, t, i1);
            Value v0 = select(is_scale, scale, df);
            Value v = select(is_n, nf, v0);
            scf::YieldOp::create(b, loc, ValueRange{v});
          }
          {
            OpBuilder::InsertionGuard guard(b);
            b.setInsertionPointToStart(valueOp.elseBlock());
            auto vOp = scf::IfOp::create(b, loc, TypeRange{f64Type}, influence,
                                         true, true);
            {
              OpBuilder::InsertionGuard guard(b);
              b.setInsertionPointToStart(vOp.thenBlock());
              Value u = subi(t, i3);
              Value col = addi(u, i4);
              Value a = load(moduli, {i4, col});
              scf::YieldOp::create(b, loc, ValueRange{a});
            }
            {
              OpBuilder::InsertionGuard guard(b);
              b.setInsertionPointToStart(vOp.elseBlock());
              Value u0 = subi(t, series);
              Value which = divui(u0, pd);
              Value r = remui(u0, pd);
              Value k = divui(r, points);
              Value j = remui(r, points);
              Value w0 = muli(which, pd);
              Value j0 = muli(j, d);
              Value w1 = addi(w0, j0);
              Value col = addi(w1, k);
              Value a = load(moduli, {i3, col});
              scf::YieldOp::create(b, loc, ValueRange{a});
            }
            Value v = vOp.getResult(0);
            scf::YieldOp::create(b, loc, ValueRange{v});
          }
          Value value = valueOp.getResult(0);
          Value vr = convert(realType, value);
          Value at = addi(pe, t);
          store(vr, tables, {at});
          scf::YieldOp::create(b, loc);
        });
    gpu::TerminatorOp::create(b, loc);
  }
  func::ReturnOp::create(b, loc);
}
