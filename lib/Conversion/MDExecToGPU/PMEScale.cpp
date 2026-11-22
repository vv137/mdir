// Apply the same influence when the sum needs only the potential grid.
#include "DevicePME.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;

template <Window window, Splitting splitting, bool tilted>
void DevicePME::emitScale(StringRef symbol, bool storeFactors) {
  StringRef blockFunction = blockSymbol<window, tilted>();
  OpBuilder::InsertionGuard functionGuard(b);
  SmallVector<Type> inputs;
  inputs.push_back(globalMem({ShapedType::kDynamic}, realType));
  inputs.push_back(globalMem({ShapedType::kDynamic}, realType));
  if constexpr (!tilted) {
    inputs.push_back(VectorType::get({3}, f64Type));
  } else {
    inputs.push_back(VectorType::get({6}, f64Type));
  }
  inputs.push_back(f64Type);
  inputs.push_back(f64Type);
  inputs.append(3, idxType);
  auto fn = func::FuncOp::create(b, loc, symbol, b.getFunctionType(inputs, {}));
  fn.setPrivate();
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  unsigned arg = 0;
  Value c = entry->getArgument(arg++);
  Value tables = entry->getArgument(arg++);
  Value box = entry->getArgument(arg++);
  Value beta = entry->getArgument(arg++);
  Value coulomb = entry->getArgument(arg++);
  Value k1 = entry->getArgument(arg++);
  Value k2 = entry->getArgument(arg++);
  Value k3 = entry->getArgument(arg++);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  Value pe;
  if constexpr (splitting == Splitting::Radial) {
    Value pk12 = addi(k1, k2);
    Value pk123 = addi(pk12, k3);
    Value pc3 = arith::ConstantIndexOp::create(b, loc, 3);
    pe = muli(pk123, pc3);
  }
  Value c2 = arith::ConstantIndexOp::create(b, loc, 2);
  Value c128 = arith::ConstantIndexOp::create(b, loc, 128);
  Value pi2 = arith::ConstantOp::create(
      b, loc, b.getFloatAttr(f64Type, 9.869604401089358));
  Value lx = extract(box, 0);
  Value ly = extract(box, 1);
  Value lz = extract(box, 2);
  Value h00r;
  Value h10r;
  Value h11r;
  Value h20r;
  Value h21r;
  Value h22r;
  if constexpr (tilted) {
    auto [h00, h10, h11, h20, h21, h22] = inverseCell(box, lx, ly, lz);
    h00r = convert(realType, h00);
    h10r = convert(realType, h10);
    h11r = convert(realType, h11);
    h20r = convert(realType, h20);
    h21r = convert(realType, h21);
    h22r = convert(realType, h22);
  }
  Value beta2 = mulf(beta, beta);
  Value gauss = divf(pi2, beta2);
  Value pi = arith::ConstantOp::create(
      b, loc, b.getFloatAttr(f64Type, 3.141592653589793));
  Value lxy = mulf(lx, ly);
  Value volume = mulf(lxy, lz);
  Value piv = mulf(pi, volume);
  Value prefactor;
  if constexpr (splitting == Splitting::Gaussian) {
    prefactor = divf(coulomb, piv);
  }
  Value pibr;
  Value sqrtpir;
  Value sixgaussr;
  if constexpr (splitting == Splitting::Radial ||
                splitting == Splitting::Dispersion) {
    Value pi32 = arith::ConstantOp::create(
        b, loc, b.getFloatAttr(f64Type, 5.568327996831708));
    Value beta3 = mulf(beta2, beta);
    Value pb3 = mulf(pi32, beta3);
    Value three =
        arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 3.0));
    Value v3 = mulf(three, volume);
    Value pd0 = divf(pb3, v3);
    Value pdc = mulf(pd0, coulomb);
    prefactor = arith::NegFOp::create(b, loc, pdc);
    Value pib = divf(pi, beta);
    pibr = convert(realType, pib);
    Value sqrtpi = arith::ConstantOp::create(
        b, loc, b.getFloatAttr(f64Type, 1.7724538509055159));
    sqrtpir = convert(realType, sqrtpi);
    Value six = arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 6.0));
    Value sixgauss = mulf(six, gauss);
    sixgaussr = convert(realType, sixgauss);
  }
  Value gaussr = convert(realType, gauss);
  convert(realType, prefactor);
  Value rows = muli(k1, k2);
  Value c3t = arith::ConstantIndexOp::create(b, loc, 3);
  Value ty0 = muli(k1, c3t);
  Value k2_3 = muli(k2, c3t);
  Value tz0 = addi(ty0, k2_3);
  Value half = divui(k3, c2);
  Value depth = addi(half, c1);
  Value points = muli(rows, depth);
  Value depth32 = indexCast(i32Type, depth);
  Value k2_32 = indexCast(i32Type, k2);
  Value unit = arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 1.0));
  Value ilx = divf(unit, lx);
  Value ily = divf(unit, ly);
  Value ilz = divf(unit, lz);
  convert(realType, ilx);
  convert(realType, ily);
  convert(realType, ilz);
  Value blocks_points =
      func::CallOp::create(b, loc, blockFunction, TypeRange{idxType},
                           ValueRange{points})
          .getResult(0);
  {
    auto launch =
        gpu::LaunchOp::create(b, loc, blocks_points, c1, c1, c128, c1, c1);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;
    Value base = muli(bx, c128);
    Value item = addi(base, tx);
    Value inside = cmpi(arith::CmpIPredicate::ult, item, points);
    auto insideBranch =
        scf::IfOp::create(b, loc, TypeRange{}, inside, true, false);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(insideBranch.thenBlock());
      arith::ConstantIndexOp::create(b, loc, 0);
      Value cc1 = arith::ConstantIndexOp::create(b, loc, 1);
      Value cc2 = arith::ConstantIndexOp::create(b, loc, 2);
      arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 0.0));
      arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 1.0));
      Value item32 = indexCast(i32Type, item);
      Value row32 = divui(item32, depth32);
      Value z32 = remui(item32, depth32);
      Value a32 = divui(row32, k2_32);
      Value b32 = remui(row32, k2_32);
      Value z = indexCast(idxType, z32);
      Value a = indexCast(idxType, a32);
      Value accB = indexCast(idxType, b32);
      Value rzero =
          arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.0));
      Value rone =
          arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 1.0));
      Value m1s;
      Value t1;
      if constexpr (!tilted) {
        Value tbx1 = addi(a, k1);
        Value tbx2 = addi(tbx1, k1);
        load(tables, {a});
        m1s = load(tables, {tbx1});
        t1 = load(tables, {tbx2});
      }
      Value n1;
      if constexpr (tilted) {
        Value tbx2 = addi(a, k1);
        Value tbx2b = addi(tbx2, k1);
        n1 = load(tables, {a});
        t1 = load(tables, {tbx2b});
      }
      Value tby0 = addi(ty0, accB);
      Value tby1 = addi(tby0, k2);
      Value tby2 = addi(tby1, k2);
      Value m2s;
      if constexpr (!tilted) {
        load(tables, {tby0});
        m2s = load(tables, {tby1});
      }
      Value n2;
      if constexpr (tilted) {
        n2 = load(tables, {tby0});
      }
      Value t2 = load(tables, {tby2});
      Value tbz0 = addi(tz0, z);
      Value tbz1 = addi(tbz0, k3);
      Value tbz2 = addi(tbz1, k3);
      Value m3s;
      if constexpr (!tilted) {
        load(tables, {tbz0});
        m3s = load(tables, {tbz1});
      }
      Value n3;
      if constexpr (tilted) {
        n3 = load(tables, {tbz0});
      }
      Value t3 = load(tables, {tbz2});
      if constexpr (tilted) {
        Value m1 = mulf(h00r, n1);
        Value m2a = mulf(h10r, n1);
        Value m2b = mulf(h11r, n2);
        Value m2 = addf(m2a, m2b);
        Value m3a = mulf(h20r, n1);
        Value m3b = mulf(h21r, n2);
        Value m3c = mulf(h22r, n3);
        Value m3ab = addf(m3a, m3b);
        Value m3 = addf(m3ab, m3c);
        m1s = mulf(m1, m1);
        m2s = mulf(m2, m2);
        m3s = mulf(m3, m3);
      }
      Value m12 = addf(m1s, m2s);
      Value msqr = addf(m12, m3s);
      Influence influence;
      if constexpr (splitting == Splitting::Gaussian && !tilted) {
        influence = gaussianInfluence(msqr, rzero, rone, t1, t2, t3);
      }
      if constexpr (splitting == Splitting::Dispersion) {
        influence = dispersionInfluence(msqr, pibr, sqrtpir, rone, t1, t2, t3,
                                        sixgaussr);
      }
      if constexpr (splitting == Splitting::Gaussian && tilted) {
        influence =
            gaussianInfluenceTriclinic(msqr, rzero, rone, t1, t2, t3, gaussr);
      }
      if constexpr (splitting == Splitting::Radial) {
        influence = radialInfluence(tables, pe, cc1, msqr, rone, t1, t2, t3);
      }
      Value bc = influence.factor;
      if (storeFactors) {
        Value axes = muli(addi(addi(k1, k2), k3), c3t);
        store(bc, tables, {addi(axes, item)});
      } else {
        Value re_at = muli(item, cc2);
        Value im_at = addi(re_at, cc1);
        Value re = load(c, {re_at});
        Value im = load(c, {im_at});
        Value reb = mulf(re, bc);
        Value imb = mulf(im, bc);
        store(reb, c, {re_at});
        store(imb, c, {im_at});
      }
      scf::YieldOp::create(b, loc);
    }
    gpu::TerminatorOp::create(b, loc);
  }
  func::ReturnOp::create(b, loc);
}

template void
    DevicePME::emitScale<Window::Spline, Splitting::Gaussian, false>(StringRef, bool);
template void
    DevicePME::emitScale<Window::Spline, Splitting::Dispersion, false>(
        StringRef, bool);
template void
    DevicePME::emitScale<Window::Spline, Splitting::Gaussian, true>(StringRef, bool);
template void DevicePME::emitScale<Window::Spline, Splitting::Dispersion, true>(
    StringRef, bool);
template void
    DevicePME::emitScale<Window::Prolate, Splitting::Radial, false>(StringRef, bool);
template void
    DevicePME::emitScale<Window::Prolate, Splitting::Radial, true>(StringRef, bool);
