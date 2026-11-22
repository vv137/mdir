// Apply the same influence when the sum needs only the potential grid.
#include "DevicePME.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;

template <Splitting splitting, bool tilted>
void DevicePME::emitScale(StringRef symbol) {
  StringRef blockFunction = splitting == Splitting::Radial
                                ? "mdrt_gpu_pme_prolate_blocks"
                            : tilted ? "mdrt_gpu_pme_blocks_triclinic"
                                     : "mdrt_gpu_pme_blocks";
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
    Value pk12 = arith::AddIOp::create(b, loc, k1, k2);
    Value pk123 = arith::AddIOp::create(b, loc, pk12, k3);
    Value pc3 = arith::ConstantIndexOp::create(b, loc, 3);
    pe = arith::MulIOp::create(b, loc, pk123, pc3);
  }
  Value c2 = arith::ConstantIndexOp::create(b, loc, 2);
  Value c128 = arith::ConstantIndexOp::create(b, loc, 128);
  Value pi2 = arith::ConstantOp::create(
      b, loc, b.getFloatAttr(f64Type, 9.869604401089358));
  Value lx = vector::ExtractOp::create(b, loc, box, ArrayRef<int64_t>{0});
  Value ly = vector::ExtractOp::create(b, loc, box, ArrayRef<int64_t>{1});
  Value lz = vector::ExtractOp::create(b, loc, box, ArrayRef<int64_t>{2});
  Value h00r;
  Value h10r;
  Value h11r;
  Value h20r;
  Value h21r;
  Value h22r;
  if constexpr (tilted) {
    Value tbx = vector::ExtractOp::create(b, loc, box, ArrayRef<int64_t>{3});
    Value tcx = vector::ExtractOp::create(b, loc, box, ArrayRef<int64_t>{4});
    Value tcy = vector::ExtractOp::create(b, loc, box, ArrayRef<int64_t>{5});
    Value hunit =
        arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 1.0));
    Value h00 = arith::DivFOp::create(b, loc, hunit, lx);
    Value h11 = arith::DivFOp::create(b, loc, hunit, ly);
    Value h22 = arith::DivFOp::create(b, loc, hunit, lz);
    Value h10a = arith::MulFOp::create(b, loc, tbx, h00);
    Value h10b = arith::MulFOp::create(b, loc, h10a, h11);
    Value h10 = arith::NegFOp::create(b, loc, h10b);
    Value h21a = arith::MulFOp::create(b, loc, tcy, h11);
    Value h21b = arith::MulFOp::create(b, loc, h21a, h22);
    Value h21 = arith::NegFOp::create(b, loc, h21b);
    Value h20a = arith::MulFOp::create(b, loc, tbx, tcy);
    Value h20b = arith::MulFOp::create(b, loc, ly, tcx);
    Value h20c = arith::SubFOp::create(b, loc, h20a, h20b);
    Value h20d = arith::MulFOp::create(b, loc, h20c, h00);
    Value h20e = arith::MulFOp::create(b, loc, h20d, h11);
    Value h20 = arith::MulFOp::create(b, loc, h20e, h22);
    h00r = convert(realType, h00);
    h10r = convert(realType, h10);
    h11r = convert(realType, h11);
    h20r = convert(realType, h20);
    h21r = convert(realType, h21);
    h22r = convert(realType, h22);
  }
  Value beta2 = arith::MulFOp::create(b, loc, beta, beta);
  Value gauss = arith::DivFOp::create(b, loc, pi2, beta2);
  Value pi = arith::ConstantOp::create(
      b, loc, b.getFloatAttr(f64Type, 3.141592653589793));
  Value lxy = arith::MulFOp::create(b, loc, lx, ly);
  Value volume = arith::MulFOp::create(b, loc, lxy, lz);
  Value piv = arith::MulFOp::create(b, loc, pi, volume);
  Value prefactor;
  if constexpr (splitting == Splitting::Gaussian) {
    prefactor = arith::DivFOp::create(b, loc, coulomb, piv);
  }
  Value pibr;
  Value sqrtpir;
  Value sixgaussr;
  if constexpr (splitting == Splitting::Radial ||
                splitting == Splitting::Dispersion) {
    Value pi32 = arith::ConstantOp::create(
        b, loc, b.getFloatAttr(f64Type, 5.568327996831708));
    Value beta3 = arith::MulFOp::create(b, loc, beta2, beta);
    Value pb3 = arith::MulFOp::create(b, loc, pi32, beta3);
    Value three =
        arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 3.0));
    Value v3 = arith::MulFOp::create(b, loc, three, volume);
    Value pd0 = arith::DivFOp::create(b, loc, pb3, v3);
    Value pdc = arith::MulFOp::create(b, loc, pd0, coulomb);
    prefactor = arith::NegFOp::create(b, loc, pdc);
    Value pib = arith::DivFOp::create(b, loc, pi, beta);
    pibr = convert(realType, pib);
    Value sqrtpi = arith::ConstantOp::create(
        b, loc, b.getFloatAttr(f64Type, 1.7724538509055159));
    sqrtpir = convert(realType, sqrtpi);
    Value six = arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 6.0));
    Value sixgauss = arith::MulFOp::create(b, loc, six, gauss);
    sixgaussr = convert(realType, sixgauss);
  }
  Value gaussr = convert(realType, gauss);
  convert(realType, prefactor);
  Value rows = arith::MulIOp::create(b, loc, k1, k2);
  Value c3t = arith::ConstantIndexOp::create(b, loc, 3);
  Value ty0 = arith::MulIOp::create(b, loc, k1, c3t);
  Value k2_3 = arith::MulIOp::create(b, loc, k2, c3t);
  Value tz0 = arith::AddIOp::create(b, loc, ty0, k2_3);
  Value half = arith::DivUIOp::create(b, loc, k3, c2);
  Value depth = arith::AddIOp::create(b, loc, half, c1);
  Value points = arith::MulIOp::create(b, loc, rows, depth);
  Value depth32 = arith::IndexCastOp::create(b, loc, i32Type, depth);
  Value k2_32 = arith::IndexCastOp::create(b, loc, i32Type, k2);
  Value unit = arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 1.0));
  Value ilx = arith::DivFOp::create(b, loc, unit, lx);
  Value ily = arith::DivFOp::create(b, loc, unit, ly);
  Value ilz = arith::DivFOp::create(b, loc, unit, lz);
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
    Value base = arith::MulIOp::create(b, loc, bx, c128);
    Value item = arith::AddIOp::create(b, loc, base, tx);
    Value inside =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, item, points);
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
      Value item32 = arith::IndexCastOp::create(b, loc, i32Type, item);
      Value row32 = arith::DivUIOp::create(b, loc, item32, depth32);
      Value z32 = arith::RemUIOp::create(b, loc, item32, depth32);
      Value a32 = arith::DivUIOp::create(b, loc, row32, k2_32);
      Value b32 = arith::RemUIOp::create(b, loc, row32, k2_32);
      Value z = arith::IndexCastOp::create(b, loc, idxType, z32);
      Value a = arith::IndexCastOp::create(b, loc, idxType, a32);
      Value accB = arith::IndexCastOp::create(b, loc, idxType, b32);
      Value rzero =
          arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.0));
      Value rone =
          arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 1.0));
      Value m1s;
      Value t1;
      if constexpr (!tilted) {
        Value tbx1 = arith::AddIOp::create(b, loc, a, k1);
        Value tbx2 = arith::AddIOp::create(b, loc, tbx1, k1);
        load(tables, {a});
        m1s = load(tables, {tbx1});
        t1 = load(tables, {tbx2});
      }
      Value n1;
      if constexpr (tilted) {
        Value tbx2 = arith::AddIOp::create(b, loc, a, k1);
        Value tbx2b = arith::AddIOp::create(b, loc, tbx2, k1);
        n1 = load(tables, {a});
        t1 = load(tables, {tbx2b});
      }
      Value tby0 = arith::AddIOp::create(b, loc, ty0, accB);
      Value tby1 = arith::AddIOp::create(b, loc, tby0, k2);
      Value tby2 = arith::AddIOp::create(b, loc, tby1, k2);
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
      Value tbz0 = arith::AddIOp::create(b, loc, tz0, z);
      Value tbz1 = arith::AddIOp::create(b, loc, tbz0, k3);
      Value tbz2 = arith::AddIOp::create(b, loc, tbz1, k3);
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
        Value m1 = arith::MulFOp::create(b, loc, h00r, n1);
        Value m2a = arith::MulFOp::create(b, loc, h10r, n1);
        Value m2b = arith::MulFOp::create(b, loc, h11r, n2);
        Value m2 = arith::AddFOp::create(b, loc, m2a, m2b);
        Value m3a = arith::MulFOp::create(b, loc, h20r, n1);
        Value m3b = arith::MulFOp::create(b, loc, h21r, n2);
        Value m3c = arith::MulFOp::create(b, loc, h22r, n3);
        Value m3ab = arith::AddFOp::create(b, loc, m3a, m3b);
        Value m3 = arith::AddFOp::create(b, loc, m3ab, m3c);
        m1s = arith::MulFOp::create(b, loc, m1, m1);
        m2s = arith::MulFOp::create(b, loc, m2, m2);
        m3s = arith::MulFOp::create(b, loc, m3, m3);
      }
      Value m12 = arith::AddFOp::create(b, loc, m1s, m2s);
      Value msqr = arith::AddFOp::create(b, loc, m12, m3s);
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
      Value re_at = arith::MulIOp::create(b, loc, item, cc2);
      Value im_at = arith::AddIOp::create(b, loc, re_at, cc1);
      Value re = load(c, {re_at});
      Value im = load(c, {im_at});
      Value reb = arith::MulFOp::create(b, loc, re, bc);
      Value imb = arith::MulFOp::create(b, loc, im, bc);
      store(reb, c, {re_at});
      store(imb, c, {im_at});
      scf::YieldOp::create(b, loc);
    }
    gpu::TerminatorOp::create(b, loc);
  }
  func::ReturnOp::create(b, loc);
}

template void DevicePME::emitScale<Splitting::Gaussian, false>(StringRef);
template void DevicePME::emitScale<Splitting::Dispersion, false>(StringRef);
template void DevicePME::emitScale<Splitting::Gaussian, true>(StringRef);
template void DevicePME::emitScale<Splitting::Dispersion, true>(StringRef);
template void DevicePME::emitScale<Splitting::Radial, false>(StringRef);
template void DevicePME::emitScale<Splitting::Radial, true>(StringRef);
