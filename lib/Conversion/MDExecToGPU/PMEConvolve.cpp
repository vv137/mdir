// Multiply the half-complex grid by the influence and reduce energy and virial.
#include "DevicePME.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;

template <Splitting splitting, bool tilted>
void DevicePME::emitConvolve(StringRef symbol) {
  OpBuilder::InsertionGuard functionGuard(b);
  SmallVector<Type> inputs;
  inputs.push_back(globalMem({ShapedType::kDynamic}, realType));
  inputs.push_back(globalMem({ShapedType::kDynamic}, realType));
  inputs.push_back(globalMem({ShapedType::kDynamic}, f64Type));
  if constexpr (!tilted) {
    inputs.push_back(VectorType::get({3}, f64Type));
  } else {
    inputs.push_back(VectorType::get({6}, f64Type));
  }
  inputs.push_back(f64Type);
  inputs.push_back(f64Type);
  inputs.append(3, idxType);
  auto fn = func::FuncOp::create(
      b, loc, symbol,
      b.getFunctionType(inputs, {f64Type, VectorType::get({9}, f64Type)}));
  fn.setPrivate();
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  unsigned arg = 0;
  Value c = entry->getArgument(arg++);
  Value tables = entry->getArgument(arg++);
  Value rows = entry->getArgument(arg++);
  Value box = entry->getArgument(arg++);
  Value beta = entry->getArgument(arg++);
  Value coulomb = entry->getArgument(arg++);
  Value k1 = entry->getArgument(arg++);
  Value k2 = entry->getArgument(arg++);
  Value k3 = entry->getArgument(arg++);
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  Value pe;
  if constexpr (splitting == Splitting::Radial) {
    Value pk12 = addi(k1, k2);
    Value pk123 = addi(pk12, k3);
    Value pc3 = arith::ConstantIndexOp::create(b, loc, 3);
    pe = muli(pk123, pc3);
  }
  Value c2 = arith::ConstantIndexOp::create(b, loc, 2);
  Value c3t = arith::ConstantIndexOp::create(b, loc, 3);
  Value c10 = arith::ConstantIndexOp::create(b, loc, 10);
  Value c127 = arith::ConstantIndexOp::create(b, loc, 127);
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
  if constexpr (splitting == Splitting::Gaussian ||
                splitting == Splitting::P3M) {
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
  Value unit = arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 1.0));
  Value ilx = divf(unit, lx);
  Value ily = divf(unit, ly);
  Value ilz = divf(unit, lz);
  convert(realType, ilx);
  convert(realType, ily);
  convert(realType, ilz);
  Value count = muli(k1, k2);
  Value ty0 = muli(k1, c3t);
  Value k2_3 = muli(k2, c3t);
  Value tz0 = addi(ty0, k2_3);
  Value extra;
  Value xy0;
  Value xz;
  if constexpr (splitting == Splitting::P3M) {
    Value k12x = addi(k1, k2);
    Value k123x = addi(k12x, k3);
    extra = muli(k123x, c3t);
    Value c2x = arith::ConstantIndexOp::create(b, loc, 2);
    Value xy0a = muli(k1, c2x);
    xy0 = addi(extra, xy0a);
    Value xza = muli(k12x, c2x);
    xz = addi(extra, xza);
  }
  Value half = divui(k3, c2);
  Value depth = addi(half, c1);
  Value odd = remui(k3, c2);
  Value even = cmpi(arith::CmpIPredicate::eq, odd, c0);
  Value points = muli(count, depth);
  Value depth32 = indexCast(i32Type, depth);
  Value k2_32 = indexCast(i32Type, k2);
  Value depth127 = addi(depth, c127);
  Value per = divui(depth127, c128);
  Value span = muli(per, c128);
  Value points_up = addi(points, span);
  Value points_up1 = subi(points_up, c1);
  Value blocks = divui(points_up1, span);
  {
    auto launch = gpu::LaunchOp::create(b, loc, blocks, c1, c1, c128, c1, c1);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value bx = launch.getBlockIds().x;

    Value tx = launch.getThreadIds().x;

    Value cc0 = arith::ConstantIndexOp::create(b, loc, 0);
    Value cc1 = arith::ConstantIndexOp::create(b, loc, 1);
    Value cc2 = arith::ConstantIndexOp::create(b, loc, 2);
    Value rzero =
        arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.0));
    Value rone =
        arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 1.0));
    Value rhalf =
        arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.5));
    Value rtwo =
        arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 2.0));
    Value run = muli(bx, span);
    auto eOp = scf::ForOp::create(
        b, loc, cc0, per, cc1,
        ValueRange{rzero, rzero, rzero, rzero, rzero, rzero, rzero},
        [&](OpBuilder &b, Location, Value c_, ValueRange carried) {
          Value e0 = carried[0];
          Value a00 = carried[1];
          Value a01 = carried[2];
          Value a02 = carried[3];
          Value a11 = carried[4];
          Value a12 = carried[5];
          Value a22 = carried[6];
          Value c128_ = muli(c_, c128);
          Value run_c = addi(run, c128_);
          Value item = addi(run_c, tx);
          Value inside = cmpi(arith::CmpIPredicate::ult, item, points);
          auto sumsResult0Op = scf::IfOp::create(
              b, loc,
              TypeRange{realType, realType, realType, realType, realType,
                        realType, realType},
              inside, true, true);
          {
            OpBuilder::InsertionGuard guard(b);
            b.setInsertionPointToStart(sumsResult0Op.thenBlock());
            Value item32 = indexCast(i32Type, item);
            Value row32 = divui(item32, depth32);
            Value z32 = remui(item32, depth32);
            Value a32 = divui(row32, k2_32);
            Value b32 = remui(row32, k2_32);
            Value z = indexCast(idxType, z32);
            Value a = indexCast(idxType, a32);
            Value accB = indexCast(idxType, b32);
            Value m1;
            Value m1s;
            Value t1;
            if constexpr (!tilted) {
              Value tbx1 = addi(a, k1);
              Value tbx2 = addi(tbx1, k1);
              m1 = load(tables, {a});
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
            Value m2;
            Value m2s;
            if constexpr (!tilted) {
              m2 = load(tables, {tby0});
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
            Value m3;
            Value m3s;
            if constexpr (!tilted) {
              m3 = load(tables, {tbz0});
              m3s = load(tables, {tbz1});
            }
            Value n3;
            if constexpr (tilted) {
              n3 = load(tables, {tbz0});
            }
            Value t3 = load(tables, {tbz2});
            if constexpr (tilted) {
              m1 = mulf(h00r, n1);
              Value m2a = mulf(h10r, n1);
              Value m2b = mulf(h11r, n2);
              m2 = addf(m2a, m2b);
              Value m3a = mulf(h20r, n1);
              Value m3b = mulf(h21r, n2);
              Value m3c = mulf(h22r, n3);
              Value m3ab = addf(m3a, m3b);
              m3 = addf(m3ab, m3c);
              m1s = mulf(m1, m1);
              m2s = mulf(m2, m2);
              m3s = mulf(m3, m3);
            }
            Value m12 = addf(m1s, m2s);
            Value msqr = addf(m12, m3s);
            Influence influence;
            if constexpr (splitting == Splitting::P3M ||
                          (splitting == Splitting::Gaussian && !tilted)) {
              influence = gaussianInfluence(msqr, rzero, rone, t1, t2, t3);
            }
            if constexpr (splitting == Splitting::Dispersion) {
              influence = dispersionInfluence(msqr, pibr, sqrtpir, rone, t1, t2,
                                              t3, sixgaussr);
            }
            if constexpr (splitting == Splitting::Gaussian && tilted) {
              influence = gaussianInfluenceTriclinic(msqr, rzero, rone, t1, t2,
                                                     t3, gaussr);
            }
            if constexpr (splitting == Splitting::Radial) {
              influence =
                  radialInfluence(tables, pe, cc1, msqr, rone, t1, t2, t3);
            }
            Value bc = influence.factor;
            Value inverse;
            if constexpr (splitting == Splitting::Gaussian ||
                          splitting == Splitting::P3M) {
              inverse = influence.virialTerm;
            }
            Value vb;
            if constexpr (splitting == Splitting::Radial ||
                          splitting == Splitting::Dispersion) {
              vb = influence.virialTerm;
            }
            Value re_at = muli(item, cc2);
            Value im_at = addi(re_at, cc1);
            Value re = load(c, {re_at});
            Value im = load(c, {im_at});
            Value re2 = mulf(re, re);
            Value im2 = mulf(im, im);
            Value g2 = addf(re2, im2);
            Value first = cmpi(arith::CmpIPredicate::eq, z, cc0);
            Value middle = cmpi(arith::CmpIPredicate::eq, z, half);
            Value middle_even = andi(middle, even);
            Value single = ori(first, middle_even);
            Value hw = select(single, rhalf, rone);
            Value hwb = mulf(hw, bc);
            Value em = mulf(hwb, g2);
            Value ef;
            if constexpr (splitting == Splitting::Gaussian) {
              Value sum = addf(inverse, gaussr);
              Value factor0 = mulf(rtwo, sum);
              Value factor = select(influence.atOrigin, rzero, factor0);
              ef = mulf(em, factor);
            }
            if constexpr (splitting == Splitting::Radial ||
                          splitting == Splitting::Dispersion) {
              Value hwv = mulf(hw, vb);
              ef = mulf(hwv, g2);
            }
            Value t11;
            Value t12;
            Value t13;
            Value t22;
            Value t23;
            Value t33;
            if constexpr (tilted || splitting == Splitting::Radial ||
                          splitting == Splitting::Dispersion ||
                          splitting == Splitting::Gaussian) {
              Value p11 = mulf(m1, m1);
              Value p12 = mulf(m1, m2);
              Value p13 = mulf(m1, m3);
              Value p22 = mulf(m2, m2);
              Value p23 = mulf(m2, m3);
              Value p33 = mulf(m3, m3);
              t11 = mulf(ef, p11);
              t12 = mulf(ef, p12);
              t13 = mulf(ef, p13);
              t22 = mulf(ef, p22);
              t23 = mulf(ef, p23);
              t33 = mulf(ef, p33);
            }
            if constexpr (splitting == Splitting::P3M) {
              Value xa0 = addi(extra, a);
              Value xa1 = addi(xa0, k1);
              Value xb0 = addi(xy0, accB);
              Value xb1 = addi(xb0, k2);
              Value xz0 = addi(xz, z);
              Value xz1 = addi(xz0, k3);
              Value ga1 = load(tables, {xa0});
              Value gs1 = load(tables, {xa1});
              Value ga2 = load(tables, {xb0});
              Value gs2 = load(tables, {xb1});
              Value ga3 = load(tables, {xz0});
              Value gs3 = load(tables, {xz1});
              Value cga = mulf(rtwo, gaussr);
              Value cgb = mulf(rtwo, inverse);
              Value efa = mulf(em, cga);
              Value efb = mulf(em, cgb);
              Value qa12 = mulf(ga1, ga2);
              Value qa13 = mulf(ga1, ga3);
              Value qa23 = mulf(ga2, ga3);
              Value qb12 = mulf(m1, m2);
              Value qb13 = mulf(m1, m3);
              Value qb23 = mulf(m2, m3);
              Value ua11 = mulf(efa, gs1);
              Value ua12 = mulf(efa, qa12);
              Value ua13 = mulf(efa, qa13);
              Value ua22 = mulf(efa, gs2);
              Value ua23 = mulf(efa, qa23);
              Value ua33 = mulf(efa, gs3);
              Value ub11 = mulf(efb, m1s);
              Value ub12 = mulf(efb, qb12);
              Value ub13 = mulf(efb, qb13);
              Value ub22 = mulf(efb, m2s);
              Value ub23 = mulf(efb, qb23);
              Value ub33 = mulf(efb, m3s);
              t11 = addf(ua11, ub11);
              t12 = addf(ua12, ub12);
              t13 = addf(ua13, ub13);
              t22 = addf(ua22, ub22);
              t23 = addf(ua23, ub23);
              t33 = addf(ua33, ub33);
            }
            Value d11 = subf(em, t11);
            Value d22 = subf(em, t22);
            Value d33 = subf(em, t33);
            Value n00 = addf(a00, d11);
            Value n01 = subf(a01, t12);
            Value n02 = subf(a02, t13);
            Value n11 = addf(a11, d22);
            Value n12 = subf(a12, t23);
            Value n22 = addf(a22, d33);
            Value ne = addf(e0, em);
            Value reb = mulf(re, bc);
            Value imb = mulf(im, bc);
            store(reb, c, {re_at});
            store(imb, c, {im_at});
            scf::YieldOp::create(b, loc,
                                 ValueRange{ne, n00, n01, n02, n11, n12, n22});
          }
          {
            OpBuilder::InsertionGuard guard(b);
            b.setInsertionPointToStart(sumsResult0Op.elseBlock());
            scf::YieldOp::create(b, loc,
                                 ValueRange{e0, a00, a01, a02, a11, a12, a22});
          }
          Value sumsResult0 = sumsResult0Op.getResult(0);
          Value sumsResult1 = sumsResult0Op.getResult(1);
          Value sumsResult2 = sumsResult0Op.getResult(2);
          Value sumsResult3 = sumsResult0Op.getResult(3);
          Value sumsResult4 = sumsResult0Op.getResult(4);
          Value sumsResult5 = sumsResult0Op.getResult(5);
          Value sumsResult6 = sumsResult0Op.getResult(6);
          scf::YieldOp::create(b, loc,
                               ValueRange{sumsResult0, sumsResult1, sumsResult2,
                                          sumsResult3, sumsResult4, sumsResult5,
                                          sumsResult6});
        });
    Value e = eOp.getResult(0);
    Value v00 = eOp.getResult(1);
    Value v01 = eOp.getResult(2);
    Value v02 = eOp.getResult(3);
    Value v11 = eOp.getResult(4);
    Value v12 = eOp.getResult(5);
    Value v22 = eOp.getResult(6);
    Value out = muli(bx, c10);
    Value lead = cmpi(arith::CmpIPredicate::eq, tx, cc0);
    Value r0 = sumBlock(e);
    Value r0w = convert(f64Type, r0);
    Value r1 = sumBlock(v00);
    Value r1w = convert(f64Type, r1);
    Value r2 = sumBlock(v01);
    Value r2w = convert(f64Type, r2);
    Value r3 = sumBlock(v02);
    Value r3w = convert(f64Type, r3);
    Value r4 = sumBlock(v11);
    Value r4w = convert(f64Type, r4);
    Value r5 = sumBlock(v12);
    Value r5w = convert(f64Type, r5);
    Value r6 = sumBlock(v22);
    Value r6w = convert(f64Type, r6);
    auto leadBranch = scf::IfOp::create(b, loc, TypeRange{}, lead, true, false);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(leadBranch.thenBlock());
      Value oc0 = arith::ConstantIndexOp::create(b, loc, 0);
      Value o0 = addi(out, oc0);
      store(r0w, rows, {o0});
      Value oc1 = arith::ConstantIndexOp::create(b, loc, 1);
      Value o1 = addi(out, oc1);
      store(r1w, rows, {o1});
      Value oc2 = arith::ConstantIndexOp::create(b, loc, 2);
      Value o2 = addi(out, oc2);
      store(r2w, rows, {o2});
      Value oc3 = arith::ConstantIndexOp::create(b, loc, 3);
      Value o3 = addi(out, oc3);
      store(r3w, rows, {o3});
      Value oc4 = arith::ConstantIndexOp::create(b, loc, 4);
      Value o4 = addi(out, oc4);
      store(r4w, rows, {o4});
      Value oc5 = arith::ConstantIndexOp::create(b, loc, 5);
      Value o5 = addi(out, oc5);
      store(r5w, rows, {o5});
      Value oc6 = arith::ConstantIndexOp::create(b, loc, 6);
      Value o6 = addi(out, oc6);
      store(r6w, rows, {o6});
      scf::YieldOp::create(b, loc);
    }
    gpu::TerminatorOp::create(b, loc);
  }
  {
    auto launch = gpu::LaunchOp::create(b, loc, c1, c1, c1, c128, c1, c1);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value tx = launch.getThreadIds().x;
    Value zero_d =
        arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 0.0));
    Value cc0 = arith::ConstantIndexOp::create(b, loc, 0);
    auto sResult0Op = scf::ForOp::create(
        b, loc, tx, blocks, c128,
        ValueRange{zero_d, zero_d, zero_d, zero_d, zero_d, zero_d, zero_d},
        [&](OpBuilder &b, Location, Value blk, ValueRange carried) {
          Value q0 = carried[0];
          Value q1 = carried[1];
          Value q2 = carried[2];
          Value q3 = carried[3];
          Value q4 = carried[4];
          Value q5 = carried[5];
          Value q6 = carried[6];
          Value at = muli(blk, c10);
          Value tc0 = arith::ConstantIndexOp::create(b, loc, 0);
          Value ta0 = addi(at, tc0);
          Value tv0 = load(rows, {ta0});
          Value tn0 = addf(q0, tv0);
          Value tc1 = arith::ConstantIndexOp::create(b, loc, 1);
          Value ta1 = addi(at, tc1);
          Value tv1 = load(rows, {ta1});
          Value tn1 = addf(q1, tv1);
          Value tc2 = arith::ConstantIndexOp::create(b, loc, 2);
          Value ta2 = addi(at, tc2);
          Value tv2 = load(rows, {ta2});
          Value tn2 = addf(q2, tv2);
          Value tc3 = arith::ConstantIndexOp::create(b, loc, 3);
          Value ta3 = addi(at, tc3);
          Value tv3 = load(rows, {ta3});
          Value tn3 = addf(q3, tv3);
          Value tc4 = arith::ConstantIndexOp::create(b, loc, 4);
          Value ta4 = addi(at, tc4);
          Value tv4 = load(rows, {ta4});
          Value tn4 = addf(q4, tv4);
          Value tc5 = arith::ConstantIndexOp::create(b, loc, 5);
          Value ta5 = addi(at, tc5);
          Value tv5 = load(rows, {ta5});
          Value tn5 = addf(q5, tv5);
          Value tc6 = arith::ConstantIndexOp::create(b, loc, 6);
          Value ta6 = addi(at, tc6);
          Value tv6 = load(rows, {ta6});
          Value tn6 = addf(q6, tv6);
          scf::YieldOp::create(b, loc,
                               ValueRange{tn0, tn1, tn2, tn3, tn4, tn5, tn6});
        });
    Value sResult0 = sResult0Op.getResult(0);
    Value sResult1 = sResult0Op.getResult(1);
    Value sResult2 = sResult0Op.getResult(2);
    Value sResult3 = sResult0Op.getResult(3);
    Value sResult4 = sResult0Op.getResult(4);
    Value sResult5 = sResult0Op.getResult(5);
    Value sResult6 = sResult0Op.getResult(6);
    Value lead = cmpi(arith::CmpIPredicate::eq, tx, cc0);
    Value u0 = sumBlock(sResult0);
    Value u1 = sumBlock(sResult1);
    Value u2 = sumBlock(sResult2);
    Value u3 = sumBlock(sResult3);
    Value u4 = sumBlock(sResult4);
    Value u5 = sumBlock(sResult5);
    Value u6 = sumBlock(sResult6);
    auto leadBranch2 =
        scf::IfOp::create(b, loc, TypeRange{}, lead, true, false);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(leadBranch2.thenBlock());
      Value uc0 = arith::ConstantIndexOp::create(b, loc, 0);
      store(u0, rows, {uc0});
      Value uc1 = arith::ConstantIndexOp::create(b, loc, 1);
      store(u1, rows, {uc1});
      Value uc2 = arith::ConstantIndexOp::create(b, loc, 2);
      store(u2, rows, {uc2});
      Value uc3 = arith::ConstantIndexOp::create(b, loc, 3);
      store(u3, rows, {uc3});
      Value uc4 = arith::ConstantIndexOp::create(b, loc, 4);
      store(u4, rows, {uc4});
      Value uc5 = arith::ConstantIndexOp::create(b, loc, 5);
      store(u5, rows, {uc5});
      Value uc6 = arith::ConstantIndexOp::create(b, loc, 6);
      store(u6, rows, {uc6});
      scf::YieldOp::create(b, loc);
    }
    gpu::TerminatorOp::create(b, loc);
  }
  Value host = memref::AllocOp::create(b, loc, MemRefType::get({7}, f64Type));
  Value rows7 =
      memref::SubViewOp::create(b, loc, rows, ArrayRef<int64_t>{0},
                                ArrayRef<int64_t>{7}, ArrayRef<int64_t>{1});
  Value t0 =
      gpu::WaitOp::create(b, loc, tokenType, ValueRange{}).getAsyncToken();
  Value t1 =
      gpu::MemcpyOp::create(b, loc, tokenType, ValueRange{t0}, host, rows7)
          .getAsyncToken();
  gpu::WaitOp::create(b, loc, Type{}, ValueRange{t1});
  Value e = load(host, {c0});
  Value c3 = arith::ConstantIndexOp::create(b, loc, 3);
  Value c4 = arith::ConstantIndexOp::create(b, loc, 4);
  Value c5 = arith::ConstantIndexOp::create(b, loc, 5);
  Value c6 = arith::ConstantIndexOp::create(b, loc, 6);
  Value w00 = load(host, {c1});
  Value w01 = load(host, {c2});
  Value w02 = load(host, {c3});
  Value w11 = load(host, {c4});
  Value w12 = load(host, {c5});
  Value w22 = load(host, {c6});
  memref::DeallocOp::create(b, loc, host);
  Value virial = vector::FromElementsOp::create(
      b, loc, VectorType::get({9}, f64Type),
      ValueRange{w00, w01, w02, w01, w11, w12, w02, w12, w22});
  func::ReturnOp::create(b, loc, ValueRange{e, virial});
}

template void DevicePME::emitConvolve<Splitting::Gaussian, false>(StringRef);
template void DevicePME::emitConvolve<Splitting::Dispersion, false>(StringRef);
template void DevicePME::emitConvolve<Splitting::P3M, false>(StringRef);
template void DevicePME::emitConvolve<Splitting::Gaussian, true>(StringRef);
template void DevicePME::emitConvolve<Splitting::Dispersion, true>(StringRef);
template void DevicePME::emitConvolve<Splitting::Radial, false>(StringRef);
template void DevicePME::emitConvolve<Splitting::Radial, true>(StringRef);
