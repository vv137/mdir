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
  Value k12 = arith::AddIOp::create(b, loc, k1, k2);
  Value k123 = arith::AddIOp::create(b, loc, k12, k3);
  Value pe = arith::MulIOp::create(b, loc, k123, c3);
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
    Value di = arith::FPToSIOp::create(b, loc, i64Type, df);
    Value d = arith::IndexCastOp::create(b, loc, idxType, di);
    Value ni = arith::FPToSIOp::create(b, loc, i64Type, nf);
    Value n = arith::IndexCastOp::create(b, loc, idxType, ni);
    Value n2 = arith::AddIOp::create(b, loc, n, n);
    Value pd = arith::MulIOp::create(b, loc, points, d);
    Value pd2 = arith::AddIOp::create(b, loc, pd, pd);
    Value series = arith::AddIOp::create(b, loc, i3, n2);
    Value count = arith::AddIOp::create(b, loc, series, pd2);
    scf::ForOp::create(
        b, loc, tx, count, i256, ValueRange{},
        [&](OpBuilder &b, Location, Value t, ValueRange carried) {
          Value meta =
              arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, t, i3);
          Value influence = arith::CmpIOp::create(
              b, loc, arith::CmpIPredicate::ult, t, series);
          auto valueOp =
              scf::IfOp::create(b, loc, TypeRange{f64Type}, meta, true, true);
          {
            OpBuilder::InsertionGuard guard(b);
            b.setInsertionPointToStart(valueOp.thenBlock());
            Value is_n =
                arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, t, i0);
            Value is_scale =
                arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, t, i1);
            Value v0 = arith::SelectOp::create(b, loc, is_scale, scale, df);
            Value v = arith::SelectOp::create(b, loc, is_n, nf, v0);
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
              Value u = arith::SubIOp::create(b, loc, t, i3);
              Value col = arith::AddIOp::create(b, loc, u, i4);
              Value a = load(moduli, {i4, col});
              scf::YieldOp::create(b, loc, ValueRange{a});
            }
            {
              OpBuilder::InsertionGuard guard(b);
              b.setInsertionPointToStart(vOp.elseBlock());
              Value u0 = arith::SubIOp::create(b, loc, t, series);
              Value which = arith::DivUIOp::create(b, loc, u0, pd);
              Value r = arith::RemUIOp::create(b, loc, u0, pd);
              Value k = arith::DivUIOp::create(b, loc, r, points);
              Value j = arith::RemUIOp::create(b, loc, r, points);
              Value w0 = arith::MulIOp::create(b, loc, which, pd);
              Value j0 = arith::MulIOp::create(b, loc, j, d);
              Value w1 = arith::AddIOp::create(b, loc, w0, j0);
              Value col = arith::AddIOp::create(b, loc, w1, k);
              Value a = load(moduli, {i3, col});
              scf::YieldOp::create(b, loc, ValueRange{a});
            }
            Value v = vOp.getResult(0);
            scf::YieldOp::create(b, loc, ValueRange{v});
          }
          Value value = valueOp.getResult(0);
          Value vr = convert(realType, value);
          Value at = arith::AddIOp::create(b, loc, pe, t);
          store(vr, tables, {at});
          scf::YieldOp::create(b, loc);
        });
    gpu::TerminatorOp::create(b, loc);
  }
  func::ReturnOp::create(b, loc);
}
