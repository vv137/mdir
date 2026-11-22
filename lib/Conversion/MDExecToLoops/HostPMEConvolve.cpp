#include "HostPME.h"

using namespace mlir;
using namespace mdir::kernels;

// The component of the wave vector of index `k` of `count` along an edge of
// the length `length`: k / L, or (k − count) / L beyond count / 2.
void HostPME::emitWave() {
  OpBuilder::InsertionGuard guard(b);
  auto fn = declare("mdrt.pme_wave",
                    b.getFunctionType({idxType, idxType, f64Type}, {f64Type}));
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  Value k = entry->getArgument(0);
  Value count = entry->getArgument(1);
  Value length = entry->getArgument(2);
  Value c2 = arith::ConstantIndexOp::create(b, loc, 2);
  Value half = arith::DivUIOp::create(b, loc, count, c2);
  Value ki = arith::IndexCastOp::create(b, loc, i64Type, k);
  Value ni = arith::IndexCastOp::create(b, loc, i64Type, count);
  Value beyond =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ugt, k, half);
  Value shifted = arith::SubIOp::create(b, loc, ki, ni);
  Value signedIndex = arith::SelectOp::create(b, loc, beyond, shifted, ki);
  Value sf = arith::SIToFPOp::create(b, loc, f64Type, signedIndex);
  Value m = arith::DivFOp::create(b, loc, sf, length);
  func::ReturnOp::create(b, loc, ValueRange{m});
}

// The factors of the influence function along the edges of a rectangular
// cell, as a device computes them (D104): for each point k of each edge a,
// at 3 (o_a + k) with o = (0, k1, k1 + k2), the wave number m_a, m_a², and
// exp(−π² m_a² / β²) times the factor B_a of the B-splines (row a of
// `moduli`), along x also times f / (π V); each computed in f64 and
// rounded once to the grid precision. The Gaussian influence function of a
// point is then the product of three of them over m².
//
// With the rows of influence "P3M" in `moduli` (more than five,
// D304), eq. 2.28 of [Ballenegger2012] in its form for the
// edges (docs/pme.md, Section 1.1): with the weights u_j of the aliases
// ν_j = n + j K of the point, μ_j = ν_j / L, g_j = exp(−π² μ_j² / β²),
// the three values are Σ u μ, Σ u μ², and Σ u g / S times the prefactor
// along x, so that the influence is again the product of the third over
// the sum of the second; after the 3 (k1 + k2 + k3) values, at
// 3 (k1 + k2 + k3) + 2 (o_a + k), Σ u g μ / Σ u g and Σ u g μ² / Σ u g,
// which the virial takes.
void HostPME::emitTables() {
  OpBuilder::InsertionGuard guard(b);
  auto fn =
      declare("mdrt.pme_tables",
              b.getFunctionType(
                  {MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic},
                                   f64Type),
                   MemRefType::get({ShapedType::kDynamic}, realType),
                   VectorType::get({3}, f64Type), f64Type, f64Type, idxType,
                   idxType, idxType},
                  {}));
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  Value moduli = entry->getArgument(0);
  Value tables = entry->getArgument(1);
  Value box = entry->getArgument(2);
  Value beta = entry->getArgument(3);
  Value coulomb = entry->getArgument(4);
  Value k1 = entry->getArgument(5);
  Value k2 = entry->getArgument(6);
  Value k3 = entry->getArgument(7);
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  Value c2 = arith::ConstantIndexOp::create(b, loc, 2);
  Value c3 = arith::ConstantIndexOp::create(b, loc, 3);
  Value pi2 = arith::ConstantOp::create(
      b, loc, b.getFloatAttr(f64Type, 9.869604401089358));
  Value pi = arith::ConstantOp::create(
      b, loc, b.getFloatAttr(f64Type, 3.141592653589793));
  Value lx = vector::ExtractOp::create(b, loc, box, ArrayRef<int64_t>{0});
  Value ly = vector::ExtractOp::create(b, loc, box, ArrayRef<int64_t>{1});
  Value lz = vector::ExtractOp::create(b, loc, box, ArrayRef<int64_t>{2});
  Value beta2 = arith::MulFOp::create(b, loc, beta, beta);
  Value gauss = arith::DivFOp::create(b, loc, pi2, beta2);
  Value lxy = arith::MulFOp::create(b, loc, lx, ly);
  Value volume = arith::MulFOp::create(b, loc, lxy, lz);
  Value piv = arith::MulFOp::create(b, loc, pi, volume);
  Value prefactor = arith::DivFOp::create(b, loc, coulomb, piv);
  Value k12 = arith::AddIOp::create(b, loc, k1, k2);
  Value c5 = arith::ConstantIndexOp::create(b, loc, 5);
  Value rows = memref::DimOp::create(b, loc, moduli, c0);
  Value p3m =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ugt, rows, c5);
  Value rows3 = arith::SubIOp::create(b, loc, rows, c3);
  Value span = arith::DivUIOp::create(b, loc, rows3, c3);
  Value span1 = arith::SubIOp::create(b, loc, span, c1);
  Value aliases = arith::DivUIOp::create(b, loc, span1, c2);
  Value entries0 = arith::AddIOp::create(b, loc, k12, k3);
  Value extra = arith::MulIOp::create(b, loc, entries0, c3);
  scf::ForOp::create(
      b, loc, c0, c3, c1, ValueRange{},
      [&](OpBuilder &b, Location, Value axis, ValueRange carried) {
        Value on_x =
            arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, axis, c0);
        Value on_y =
            arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, axis, c1);
        Value count0 = arith::SelectOp::create(b, loc, on_y, k2, k3);
        Value count = arith::SelectOp::create(b, loc, on_x, k1, count0);
        Value length0 = arith::SelectOp::create(b, loc, on_y, ly, lz);
        Value length = arith::SelectOp::create(b, loc, on_x, lx, length0);
        Value offset0 = arith::SelectOp::create(b, loc, on_y, k1, k12);
        Value offset = arith::SelectOp::create(b, loc, on_x, c0, offset0);
        scf::ForOp::create(
            b, loc, c0, count, c1, ValueRange{},
            [&](OpBuilder &b, Location, Value k, ValueRange carried) {
              Value point = arith::AddIOp::create(b, loc, offset, k);
              Value m = func::CallOp::create(b, loc, "mdrt.pme_wave",
                                             TypeRange{f64Type},
                                             ValueRange{k, count, length})
                            .getResult(0);
              auto v0Op = scf::IfOp::create(
                  b, loc, TypeRange{f64Type, f64Type, f64Type}, p3m, true,
                  true);
              {
                OpBuilder::InsertionGuard guard(b);
                b.setInsertionPointToStart(v0Op.thenBlock());
                Value zero = arith::ConstantOp::create(
                    b, loc, b.getFloatAttr(f64Type, 0.0));
                Value unit = arith::ConstantOp::create(
                    b, loc, b.getFloatAttr(f64Type, 1.0));
                Value n = func::CallOp::create(b, loc, "mdrt.pme_wave",
                                               TypeRange{f64Type},
                                               ValueRange{k, count, unit})
                              .getResult(0);
                Value ci = arith::IndexCastOp::create(b, loc, i64Type, count);
                Value cf = arith::SIToFPOp::create(b, loc, f64Type, ci);
                Value ai = arith::IndexCastOp::create(b, loc, i64Type, aliases);
                Value first = arith::MulIOp::create(b, loc, axis, span);
                Value row0 = arith::AddIOp::create(b, loc, first, c3);
                auto smOp = scf::ForOp::create(
                    b, loc, c0, span, c1,
                    ValueRange{zero, zero, zero, zero, zero},
                    [&](OpBuilder &b, Location, Value j, ValueRange carried) {
                      Value a0 = carried[0];
                      Value a1 = carried[1];
                      Value a2 = carried[2];
                      Value a3 = carried[3];
                      Value a4 = carried[4];
                      Value ji = arith::IndexCastOp::create(b, loc, i64Type, j);
                      Value js = arith::SubIOp::create(b, loc, ji, ai);
                      Value jf = arith::SIToFPOp::create(b, loc, f64Type, js);
                      Value jk = arith::MulFOp::create(b, loc, jf, cf);
                      Value nu = arith::AddFOp::create(b, loc, n, jk);
                      Value mu = arith::DivFOp::create(b, loc, nu, length);
                      Value mu2 = arith::MulFOp::create(b, loc, mu, mu);
                      Value gmu = arith::MulFOp::create(b, loc, gauss, mu2);
                      Value ngmu = arith::NegFOp::create(b, loc, gmu);
                      Value g = math::ExpOp::create(b, loc, ngmu);
                      Value row = arith::AddIOp::create(b, loc, row0, j);
                      Value u = load(moduli, {row, k});
                      Value umu = arith::MulFOp::create(b, loc, u, mu);
                      Value umu2 = arith::MulFOp::create(b, loc, u, mu2);
                      Value ug = arith::MulFOp::create(b, loc, u, g);
                      Value ugmu = arith::MulFOp::create(b, loc, ug, mu);
                      Value ugmu2 = arith::MulFOp::create(b, loc, ug, mu2);
                      Value b0 = arith::AddFOp::create(b, loc, a0, umu);
                      Value b1 = arith::AddFOp::create(b, loc, a1, umu2);
                      Value b2 = arith::AddFOp::create(b, loc, a2, ug);
                      Value b3 = arith::AddFOp::create(b, loc, a3, ugmu);
                      Value b4 = arith::AddFOp::create(b, loc, a4, ugmu2);
                      scf::YieldOp::create(b, loc,
                                           ValueRange{b0, b1, b2, b3, b4});
                    });
                Value sm = smOp.getResult(0);
                Value sq = smOp.getResult(1);
                Value sg = smOp.getResult(2);
                Value sgm = smOp.getResult(3);
                Value sgm2 = smOp.getResult(4);
                Value inverse = load(moduli, {axis, k});
                Value f0 = arith::MulFOp::create(b, loc, sg, inverse);
                Value some = arith::CmpFOp::create(
                    b, loc, arith::CmpFPredicate::OGT, sg, zero);
                Value sgs = arith::SelectOp::create(b, loc, some, sg, unit);
                Value gm0 = arith::DivFOp::create(b, loc, sgm, sgs);
                Value gq0 = arith::DivFOp::create(b, loc, sgm2, sgs);
                Value gm = arith::SelectOp::create(b, loc, some, gm0, zero);
                Value gq = arith::SelectOp::create(b, loc, some, gq0, zero);
                Value two = arith::ConstantIndexOp::create(b, loc, 2);
                Value at3a = arith::MulIOp::create(b, loc, point, two);
                Value at3 = arith::AddIOp::create(b, loc, extra, at3a);
                Value at4 = arith::AddIOp::create(b, loc, at3, c1);
                Value gmr = convert(realType, gm);
                Value gqr = convert(realType, gq);
                store(gmr, tables, {at3});
                store(gqr, tables, {at4});
                scf::YieldOp::create(b, loc, ValueRange{sm, sq, f0});
              }
              {
                OpBuilder::InsertionGuard guard(b);
                b.setInsertionPointToStart(v0Op.elseBlock());
                Value m2 = arith::MulFOp::create(b, loc, m, m);
                Value gm = arith::MulFOp::create(b, loc, gauss, m2);
                Value ngm = arith::NegFOp::create(b, loc, gm);
                Value ex = math::ExpOp::create(b, loc, ngm);
                Value mod = load(moduli, {axis, k});
                Value f0 = arith::MulFOp::create(b, loc, ex, mod);
                scf::YieldOp::create(b, loc, ValueRange{m, m2, f0});
              }
              Value v0 = v0Op.getResult(0);
              Value v1 = v0Op.getResult(1);
              Value factor0 = v0Op.getResult(2);
              Value scaled = arith::MulFOp::create(b, loc, factor0, prefactor);
              Value factor =
                  arith::SelectOp::create(b, loc, on_x, scaled, factor0);
              Value at0 = arith::MulIOp::create(b, loc, point, c3);
              Value at1 = arith::AddIOp::create(b, loc, at0, c1);
              Value at2 = arith::AddIOp::create(b, loc, at0, c2);
              Value mr = convert(realType, v0);
              Value m2r = convert(realType, v1);
              Value fr = convert(realType, factor);
              store(mr, tables, {at0});
              store(m2r, tables, {at1});
              store(fr, tables, {at2});
              scf::YieldOp::create(b, loc);
            });
        scf::YieldOp::create(b, loc);
      });
  func::ReturnOp::create(b, loc);
}

// The influence function of the point of the wave vector m, in f64, for
// the sums that are not separable along the edges: the Gaussian sum in a
// triclinic cell, the sum of the dispersion, and the prolate kernel.
// `mods` is the product of the factors of the B-splines of the three
// edges, `volume` that of the cell. Returns B C(m) and the parts of the
// virial of the point, E_m (δ − factor m m) − ½ w vd |G|² m m:
//
//   B C(m) = f / (π V) · exp(−π² m² / β²) / m² · B_1(k_1) B_2(k_2) B_3(k_3)
//   W_ab = Σ_m E_m (δ_ab − 2 (1 / m² + π² / β²) m_a m_b)
void HostPME::emitInfluence() {
  OpBuilder::InsertionGuard guard(b);
  auto fn =
      declare("mdrt.pme_influence",
              b.getFunctionType(
                  {f64Type, f64Type, f64Type, f64Type,
                   MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic},
                                   f64Type),
                   f64Type, f64Type, f64Type, boolType},
                  {f64Type, f64Type, f64Type}));
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  Value m1 = entry->getArgument(0);
  Value m2 = entry->getArgument(1);
  Value m3 = entry->getArgument(2);
  Value mods = entry->getArgument(3);
  Value moduli = entry->getArgument(4);
  Value volume = entry->getArgument(5);
  Value beta = entry->getArgument(6);
  Value coulomb = entry->getArgument(7);
  Value dispersion = entry->getArgument(8);
  Value zero = arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 0.0));
  Value one = arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 1.0));
  Value two = arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 2.0));
  Value pi2 = arith::ConstantOp::create(
      b, loc, b.getFloatAttr(f64Type, 9.869604401089358));
  Value pi = arith::ConstantOp::create(
      b, loc, b.getFloatAttr(f64Type, 3.141592653589793));
  Value beta2 = arith::MulFOp::create(b, loc, beta, beta);
  Value gauss = arith::DivFOp::create(b, loc, pi2, beta2);
  Value piv = arith::MulFOp::create(b, loc, pi, volume);
  Value prefactor = arith::DivFOp::create(b, loc, coulomb, piv);
  Value m1s = arith::MulFOp::create(b, loc, m1, m1);
  Value m2s = arith::MulFOp::create(b, loc, m2, m2);
  Value m3s = arith::MulFOp::create(b, loc, m3, m3);
  Value m12 = arith::AddFOp::create(b, loc, m1s, m2s);
  Value msq = arith::AddFOp::create(b, loc, m12, m3s);
  Value origin =
      arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OEQ, msq, zero);
  Value safe = arith::SelectOp::create(b, loc, origin, one, msq);
  Value inverse = arith::DivFOp::create(b, loc, one, safe);
  auto bcgOp = scf::IfOp::create(b, loc, TypeRange{f64Type, f64Type, f64Type},
                                 dispersion, true, true);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(bcgOp.thenBlock());
    Value pi32 = arith::ConstantOp::create(
        b, loc, b.getFloatAttr(f64Type, 5.568327996831708));
    Value beta3 = arith::MulFOp::create(b, loc, beta2, beta);
    Value pb3 = arith::MulFOp::create(b, loc, pi32, beta3);
    Value three =
        arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 3.0));
    Value v3 = arith::MulFOp::create(b, loc, three, volume);
    Value pd0 = arith::DivFOp::create(b, loc, pb3, v3);
    Value pdc = arith::MulFOp::create(b, loc, pd0, coulomb);
    Value dprefactor = arith::NegFOp::create(b, loc, pdc);
    Value pib = arith::DivFOp::create(b, loc, pi, beta);
    Value sqrtpi = arith::ConstantOp::create(
        b, loc, b.getFloatAttr(f64Type, 1.7724538509055159));
    Value six = arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 6.0));
    Value sixgauss = arith::MulFOp::create(b, loc, six, gauss);
    Value mabs = math::SqrtOp::create(b, loc, msq);
    Value bd = arith::MulFOp::create(b, loc, pib, mabs);
    Value bd2 = arith::MulFOp::create(b, loc, bd, bd);
    Value nbd2 = arith::NegFOp::create(b, loc, bd2);
    Value ebd = math::ExpOp::create(b, loc, nbd2);
    Value erfcb = math::ErfcOp::create(b, loc, bd);
    Value spb = arith::MulFOp::create(b, loc, sqrtpi, bd);
    Value sbd = arith::MulFOp::create(b, loc, spb, erfcb);
    Value twob2 = arith::MulFOp::create(b, loc, two, bd2);
    Value omt = arith::SubFOp::create(b, loc, one, twob2);
    Value fa = arith::MulFOp::create(b, loc, omt, ebd);
    Value fb2 = arith::MulFOp::create(b, loc, twob2, sbd);
    Value fd = arith::AddFOp::create(b, loc, fa, fb2);
    Value pmods = arith::MulFOp::create(b, loc, dprefactor, mods);
    Value dbc = arith::MulFOp::create(b, loc, pmods, fd);
    Value vd0 = arith::SubFOp::create(b, loc, ebd, sbd);
    Value vd1 = arith::MulFOp::create(b, loc, sixgauss, vd0);
    Value dvd = arith::MulFOp::create(b, loc, pmods, vd1);
    scf::YieldOp::create(b, loc, ValueRange{dbc, zero, dvd});
  }
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(bcgOp.elseBlock());
    Value gm = arith::MulFOp::create(b, loc, gauss, msq);
    Value ngm = arith::NegFOp::create(b, loc, gm);
    Value ex = math::ExpOp::create(b, loc, ngm);
    Value exm = arith::MulFOp::create(b, loc, ex, inverse);
    Value pexm = arith::MulFOp::create(b, loc, prefactor, exm);
    Value bc0 = arith::MulFOp::create(b, loc, pexm, mods);
    Value bcc = arith::SelectOp::create(b, loc, origin, zero, bc0);
    Value sum = arith::AddFOp::create(b, loc, inverse, gauss);
    Value factor0 = arith::MulFOp::create(b, loc, two, sum);
    Value factorc = arith::SelectOp::create(b, loc, origin, zero, factor0);
    scf::YieldOp::create(b, loc, ValueRange{bcc, factorc, zero});
  }
  Value bcg = bcgOp.getResult(0);
  Value factorg = bcgOp.getResult(1);
  Value vdg = bcgOp.getResult(2);
  Value p_r0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value p_r4 = arith::ConstantIndexOp::create(b, loc, 4);
  Value p_r5 = arith::ConstantIndexOp::create(b, loc, 5);
  Value p_rows = memref::DimOp::create(b, loc, moduli, p_r0);
  Value prolate =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, p_rows, p_r5);
  auto bcOp = scf::IfOp::create(b, loc, TypeRange{f64Type, f64Type, f64Type},
                                prolate, true, true);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(bcOp.thenBlock());
    Value p_c2 = arith::ConstantIndexOp::create(b, loc, 2);
    Value p_c3 = arith::ConstantIndexOp::create(b, loc, 3);
    Value p_tf = load(moduli, {p_r4, p_c2});
    Value p_ti = arith::FPToSIOp::create(b, loc, i64Type, p_tf);
    Value p_terms = arith::IndexCastOp::create(b, loc, idxType, p_ti);
    Value p_scale = load(moduli, {p_r4, p_c3});
    Value p_mabs = math::SqrtOp::create(b, loc, msq);
    Value p_x = arith::MulFOp::create(b, loc, p_scale, p_mabs);
    Value p_x2 = arith::MulFOp::create(b, loc, p_x, p_x);
    Value p_2x2 = arith::MulFOp::create(b, loc, two, p_x2);
    Value p_y = arith::SubFOp::create(b, loc, p_2x2, one);
    Value p_y2 = arith::MulFOp::create(b, loc, two, p_y);
    Value p_at = arith::AddIOp::create(b, loc, p_r4, p_terms);
    Value p_ch =
        func::CallOp::create(b, loc, "mdrt.pme_clenshaw", TypeRange{f64Type},
                             ValueRange{moduli, p_r4, p_r4, p_terms, p_y2, p_y})
            .getResult(0);
    Value p_gch =
        func::CallOp::create(b, loc, "mdrt.pme_clenshaw", TypeRange{f64Type},
                             ValueRange{moduli, p_r4, p_at, p_terms, p_y2, p_y})
            .getResult(0);
    Value p_pm = arith::MulFOp::create(b, loc, prefactor, mods);
    Value p_pmi = arith::MulFOp::create(b, loc, p_pm, inverse);
    Value p_bc0 = arith::MulFOp::create(b, loc, p_pmi, p_ch);
    Value p_f0 = arith::MulFOp::create(b, loc, two, inverse);
    Value p_vd1 = arith::MulFOp::create(b, loc, p_pmi, inverse);
    Value p_vd2 = arith::MulFOp::create(b, loc, p_vd1, p_gch);
    Value p_vd0 = arith::NegFOp::create(b, loc, p_vd2);
    Value p_band =
        arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLT, p_x, one);
    Value p_true = arith::ConstantOp::create(b, loc, b.getBoolAttr(true));
    Value p_nonzero = arith::XOrIOp::create(b, loc, origin, p_true);
    Value p_in = arith::AndIOp::create(b, loc, p_band, p_nonzero);
    Value p_bc = arith::SelectOp::create(b, loc, p_in, p_bc0, zero);
    Value p_f = arith::SelectOp::create(b, loc, p_in, p_f0, zero);
    Value p_vd = arith::SelectOp::create(b, loc, p_in, p_vd0, zero);
    scf::YieldOp::create(b, loc, ValueRange{p_bc, p_f, p_vd});
  }
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(bcOp.elseBlock());
    scf::YieldOp::create(b, loc, ValueRange{bcg, factorg, vdg});
  }
  Value bc = bcOp.getResult(0);
  Value factor = bcOp.getResult(1);
  Value vd = bcOp.getResult(2);
  func::ReturnOp::create(b, loc, ValueRange{bc, factor, vd});
}

// The energy and the virial of a point of the half-complex transform,
// in the grid precision, added to `e` and `w`, from its value `re` + i `im`,
// its wave vector, the influence `bc`, and the parts of the virial: with w(m) =
// 2 for the points whose conjugate is not stored,
//
//   E_m = ½ w(m) B C(m) |G(m)|²,  W_m = E_m (δ − factor m m) − ½ w vd |G|² m m.
void HostPME::emitPointSums() {
  OpBuilder::InsertionGuard guard(b);
  auto fn = declare(
      "mdrt.pme_point_sums",
      b.getFunctionType({realType, realType, realType, realType, realType,
                         realType, realType, realType, boolType, realType,
                         VectorType::get({9}, realType)},
                        {realType, VectorType::get({9}, realType)}));
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  Value re = entry->getArgument(0);
  Value im = entry->getArgument(1);
  Value m1 = entry->getArgument(2);
  Value m2 = entry->getArgument(3);
  Value m3 = entry->getArgument(4);
  Value bc = entry->getArgument(5);
  Value factor = entry->getArgument(6);
  Value vd = entry->getArgument(7);
  Value single = entry->getArgument(8);
  Value e = entry->getArgument(9);
  Value w = entry->getArgument(10);
  Value half = arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.5));
  Value one = arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 1.0));
  Value identity = arith::ConstantOp::create(
      b, loc,
      dense(VectorType::get({9}, realType),
            {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}));
  Value re2 = arith::MulFOp::create(b, loc, re, re);
  Value im2 = arith::MulFOp::create(b, loc, im, im);
  Value g2 = arith::AddFOp::create(b, loc, re2, im2);
  Value hw = arith::SelectOp::create(b, loc, single, half, one);
  Value hwb = arith::MulFOp::create(b, loc, hw, bc);
  Value em = arith::MulFOp::create(b, loc, hwb, g2);
  Value mv = vector::FromElementsOp::create(
      b, loc, VectorType::get({3}, realType), ValueRange{m1, m2, m3});
  Value mm = vector::OuterProductOp::create(
      b, loc, VectorType::get({3, 3}, realType), mv, mv, Value{});
  Value mmf =
      vector::ShapeCastOp::create(b, loc, VectorType::get({9}, realType), mm);
  Value fb = vector::BroadcastOp::create(b, loc, VectorType::get({9}, realType),
                                         factor);
  Value scaled = arith::MulFOp::create(b, loc, fb, mmf);
  Value diff = arith::SubFOp::create(b, loc, identity, scaled);
  Value eb9 =
      vector::BroadcastOp::create(b, loc, VectorType::get({9}, realType), em);
  Value wmc = arith::MulFOp::create(b, loc, eb9, diff);
  Value hvd = arith::MulFOp::create(b, loc, hw, vd);
  Value evd = arith::MulFOp::create(b, loc, hvd, g2);
  Value evb =
      vector::BroadcastOp::create(b, loc, VectorType::get({9}, realType), evd);
  Value evm = arith::MulFOp::create(b, loc, evb, mmf);
  Value wm = arith::SubFOp::create(b, loc, wmc, evm);
  Value wn = arith::AddFOp::create(b, loc, w, wm);
  Value en = arith::AddFOp::create(b, loc, e, em);
  func::ReturnOp::create(b, loc, ValueRange{en, wn});
}

// As pme_point_sums, for influence "P3M" (D304): the virial of
// the point is the derivative of its energy under a strain of the cell,
// through every alias in the sums of eq. 2.28 (docs/pme.md, Section 1.1),
//
//   W_ab = E_m (δ_ab − 2 (π²/β²) A_ab − 2 B_ab / D),
//
// with A_ab = ⟨μ_a⟩_g ⟨μ_b⟩_g off the diagonal and ⟨μ_a²⟩_g on it (`ga`,
// `gs`), B likewise of the plain means (`pa`, `ps`), D = Σ_a ⟨μ_a²⟩, and
// `inverse` 1 / D.
void HostPME::emitPointSumsP3m() {
  OpBuilder::InsertionGuard guard(b);
  auto fn = declare(
      "mdrt.pme_point_sums_p3m",
      b.getFunctionType(
          {realType, realType, VectorType::get({3}, realType),
           VectorType::get({3}, realType), VectorType::get({3}, realType),
           VectorType::get({3}, realType), realType, realType, realType,
           boolType, realType, VectorType::get({9}, realType)},
          {realType, VectorType::get({9}, realType)}));
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  Value re = entry->getArgument(0);
  Value im = entry->getArgument(1);
  Value pa = entry->getArgument(2);
  Value ps = entry->getArgument(3);
  Value ga = entry->getArgument(4);
  Value gs = entry->getArgument(5);
  Value bc = entry->getArgument(6);
  Value inverse = entry->getArgument(7);
  Value gauss = entry->getArgument(8);
  Value single = entry->getArgument(9);
  Value e = entry->getArgument(10);
  Value w = entry->getArgument(11);
  Value half = arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.5));
  Value one = arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 1.0));
  Value two = arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 2.0));
  Value identity = arith::ConstantOp::create(
      b, loc,
      dense(VectorType::get({9}, realType),
            {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}));
  Value re2 = arith::MulFOp::create(b, loc, re, re);
  Value im2 = arith::MulFOp::create(b, loc, im, im);
  Value g2 = arith::AddFOp::create(b, loc, re2, im2);
  Value hw = arith::SelectOp::create(b, loc, single, half, one);
  Value hwb = arith::MulFOp::create(b, loc, hw, bc);
  Value em = arith::MulFOp::create(b, loc, hwb, g2);
  Value aa = vector::OuterProductOp::create(
      b, loc, VectorType::get({3, 3}, realType), ga, ga, Value{});
  Value bb = vector::OuterProductOp::create(
      b, loc, VectorType::get({3, 3}, realType), pa, pa, Value{});
  Value gs0 = vector::ExtractOp::create(b, loc, gs, ArrayRef<int64_t>{0});
  Value gs1 = vector::ExtractOp::create(b, loc, gs, ArrayRef<int64_t>{1});
  Value gs2 = vector::ExtractOp::create(b, loc, gs, ArrayRef<int64_t>{2});
  Value ps0 = vector::ExtractOp::create(b, loc, ps, ArrayRef<int64_t>{0});
  Value ps1 = vector::ExtractOp::create(b, loc, ps, ArrayRef<int64_t>{1});
  Value ps2 = vector::ExtractOp::create(b, loc, ps, ArrayRef<int64_t>{2});
  Value aa0 =
      vector::InsertOp::create(b, loc, gs0, aa, ArrayRef<int64_t>{0, 0});
  Value aa1 =
      vector::InsertOp::create(b, loc, gs1, aa0, ArrayRef<int64_t>{1, 1});
  Value aa2 =
      vector::InsertOp::create(b, loc, gs2, aa1, ArrayRef<int64_t>{2, 2});
  Value bb0 =
      vector::InsertOp::create(b, loc, ps0, bb, ArrayRef<int64_t>{0, 0});
  Value bb1 =
      vector::InsertOp::create(b, loc, ps1, bb0, ArrayRef<int64_t>{1, 1});
  Value bb2 =
      vector::InsertOp::create(b, loc, ps2, bb1, ArrayRef<int64_t>{2, 2});
  Value af =
      vector::ShapeCastOp::create(b, loc, VectorType::get({9}, realType), aa2);
  Value bf =
      vector::ShapeCastOp::create(b, loc, VectorType::get({9}, realType), bb2);
  Value ca = arith::MulFOp::create(b, loc, two, gauss);
  Value cb = arith::MulFOp::create(b, loc, two, inverse);
  Value cav =
      vector::BroadcastOp::create(b, loc, VectorType::get({9}, realType), ca);
  Value cbv =
      vector::BroadcastOp::create(b, loc, VectorType::get({9}, realType), cb);
  Value ta = arith::MulFOp::create(b, loc, cav, af);
  Value tb = arith::MulFOp::create(b, loc, cbv, bf);
  Value t = arith::AddFOp::create(b, loc, ta, tb);
  Value diff = arith::SubFOp::create(b, loc, identity, t);
  Value eb9 =
      vector::BroadcastOp::create(b, loc, VectorType::get({9}, realType), em);
  Value wm = arith::MulFOp::create(b, loc, eb9, diff);
  Value wn = arith::AddFOp::create(b, loc, w, wm);
  Value en = arith::AddFOp::create(b, loc, e, em);
  func::ReturnOp::create(b, loc, ValueRange{en, wn});
}

// Multiplies the row `row` = a k2 + b of the half-complex transform `c` by
// the influence function, and returns its energy and virial (with `sums`),
// summed along the row in the grid precision and returned in f64. The Gaussian
// sum takes the factors of the edges from `tables` (pme_tables); the others
// compute the influence of each point in f64 (pme_influence), rounded once.
void HostPME::emitConvolveRow() {
  OpBuilder::InsertionGuard guard(b);
  auto fn =
      declare("mdrt.pme_convolve_row",
              b.getFunctionType(
                  {MemRefType::get({ShapedType::kDynamic}, realType),
                   MemRefType::get({ShapedType::kDynamic}, realType),
                   MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic},
                                   f64Type),
                   VectorType::get({3}, f64Type), f64Type, f64Type, boolType,
                   idxType, idxType, idxType, idxType, boolType},
                  {f64Type, VectorType::get({9}, f64Type)}));
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  Value c = entry->getArgument(0);
  Value tables = entry->getArgument(1);
  Value moduli = entry->getArgument(2);
  Value box = entry->getArgument(3);
  Value beta = entry->getArgument(4);
  Value coulomb = entry->getArgument(5);
  Value dispersion = entry->getArgument(6);
  Value k1 = entry->getArgument(7);
  Value k2 = entry->getArgument(8);
  Value k3 = entry->getArgument(9);
  Value row = entry->getArgument(10);
  Value sums = entry->getArgument(11);
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  Value c2 = arith::ConstantIndexOp::create(b, loc, 2);
  Value c3 = arith::ConstantIndexOp::create(b, loc, 3);
  Value c5 = arith::ConstantIndexOp::create(b, loc, 5);
  Value rzero =
      arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.0));
  Value rone = arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 1.0));
  Value rtwo = arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 2.0));
  Value rnone = arith::ConstantOp::create(
      b, loc, dense(VectorType::get({9}, realType), {0.0}));
  Value pi2 = arith::ConstantOp::create(
      b, loc, b.getFloatAttr(f64Type, 9.869604401089358));
  Value yes = arith::ConstantOp::create(b, loc, b.getBoolAttr(true));
  Value lx = vector::ExtractOp::create(b, loc, box, ArrayRef<int64_t>{0});
  Value ly = vector::ExtractOp::create(b, loc, box, ArrayRef<int64_t>{1});
  Value lz = vector::ExtractOp::create(b, loc, box, ArrayRef<int64_t>{2});
  Value lxy = arith::MulFOp::create(b, loc, lx, ly);
  Value volume = arith::MulFOp::create(b, loc, lxy, lz);
  Value beta2 = arith::MulFOp::create(b, loc, beta, beta);
  Value gauss = arith::DivFOp::create(b, loc, pi2, beta2);
  Value gaussr = convert(realType, gauss);
  Value rows = memref::DimOp::create(b, loc, moduli, c0);
  Value prolate =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, rows, c5);
  Value other = arith::OrIOp::create(b, loc, dispersion, prolate);
  Value separable = arith::XOrIOp::create(b, loc, other, yes);
  Value p3m =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ugt, rows, c5);
  Value k12e = arith::AddIOp::create(b, loc, k1, k2);
  Value entries = arith::AddIOp::create(b, loc, k12e, k3);
  Value extra = arith::MulIOp::create(b, loc, entries, c3);
  Value a = arith::DivUIOp::create(b, loc, row, k2);
  Value accB = arith::RemUIOp::create(b, loc, row, k2);
  Value h = arith::DivUIOp::create(b, loc, k3, c2);
  Value h1 = arith::AddIOp::create(b, loc, h, c1);
  Value odd = arith::RemUIOp::create(b, loc, k3, c2);
  Value even = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, odd, c0);
  Value ta = arith::MulIOp::create(b, loc, a, c3);
  Value ta1 = arith::AddIOp::create(b, loc, ta, c1);
  Value ta2 = arith::AddIOp::create(b, loc, ta, c2);
  Value m1 = load(tables, {ta});
  Value m1s = load(tables, {ta1});
  Value t1 = load(tables, {ta2});
  Value pb = arith::AddIOp::create(b, loc, k1, accB);
  Value tb = arith::MulIOp::create(b, loc, pb, c3);
  Value tb1 = arith::AddIOp::create(b, loc, tb, c1);
  Value tb2 = arith::AddIOp::create(b, loc, tb, c2);
  Value m2 = load(tables, {tb});
  Value m2s = load(tables, {tb1});
  Value t2 = load(tables, {tb2});
  Value t12 = arith::MulFOp::create(b, loc, t1, t2);
  Value m12s = arith::AddFOp::create(b, loc, m1s, m2s);
  Value k12 = arith::AddIOp::create(b, loc, k1, k2);
  Value m1d = func::CallOp::create(b, loc, "mdrt.pme_wave", TypeRange{f64Type},
                                   ValueRange{a, k1, lx})
                  .getResult(0);
  Value m2d = func::CallOp::create(b, loc, "mdrt.pme_wave", TypeRange{f64Type},
                                   ValueRange{accB, k2, ly})
                  .getResult(0);
  Value mod1 = load(moduli, {c0, a});
  Value mod2 = load(moduli, {c1, accB});
  Value mod12 = arith::MulFOp::create(b, loc, mod1, mod2);
  Value base = arith::MulIOp::create(b, loc, row, h1);
  auto eOp = scf::ForOp::create(
      b, loc, c0, h1, c1, ValueRange{rzero, rnone},
      [&](OpBuilder &b, Location, Value z, ValueRange carried) {
        Value e3 = carried[0];
        Value w3 = carried[1];
        Value pz = arith::AddIOp::create(b, loc, k12, z);
        Value tz = arith::MulIOp::create(b, loc, pz, c3);
        Value tz1 = arith::AddIOp::create(b, loc, tz, c1);
        Value tz2 = arith::AddIOp::create(b, loc, tz, c2);
        Value m3 = load(tables, {tz});
        Value m3s = load(tables, {tz1});
        Value t3 = load(tables, {tz2});
        auto bcOp =
            scf::IfOp::create(b, loc, TypeRange{realType, realType, realType},
                              separable, true, true);
        {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(bcOp.thenBlock());
          Value msq = arith::AddFOp::create(b, loc, m12s, m3s);
          Value origin = arith::CmpFOp::create(
              b, loc, arith::CmpFPredicate::OEQ, msq, rzero);
          Value safe = arith::SelectOp::create(b, loc, origin, rone, msq);
          Value inverse = arith::DivFOp::create(b, loc, rone, safe);
          Value t123 = arith::MulFOp::create(b, loc, t12, t3);
          Value bc0 = arith::MulFOp::create(b, loc, t123, inverse);
          Value bcs = arith::SelectOp::create(b, loc, origin, rzero, bc0);
          Value sum = arith::AddFOp::create(b, loc, inverse, gaussr);
          Value factor0 = arith::MulFOp::create(b, loc, rtwo, sum);
          Value factors =
              arith::SelectOp::create(b, loc, origin, rzero, factor0);
          scf::YieldOp::create(b, loc, ValueRange{bcs, factors, rzero});
        }
        {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(bcOp.elseBlock());
          Value zi = arith::IndexCastOp::create(b, loc, i64Type, z);
          Value zf = arith::SIToFPOp::create(b, loc, f64Type, zi);
          Value m3d = arith::DivFOp::create(b, loc, zf, lz);
          Value mod3 = load(moduli, {c2, z});
          Value mods = arith::MulFOp::create(b, loc, mod12, mod3);
          auto bcdCall = func::CallOp::create(
              b, loc, "mdrt.pme_influence",
              TypeRange{f64Type, f64Type, f64Type},
              ValueRange{m1d, m2d, m3d, mods, moduli, volume, beta, coulomb,
                         dispersion});
          Value bcd = bcdCall.getResult(0);
          Value fd = bcdCall.getResult(1);
          Value vdd = bcdCall.getResult(2);
          Value bcr = convert(realType, bcd);
          Value fr = convert(realType, fd);
          Value vdr = convert(realType, vdd);
          scf::YieldOp::create(b, loc, ValueRange{bcr, fr, vdr});
        }
        Value bc = bcOp.getResult(0);
        Value factor = bcOp.getResult(1);
        Value vd = bcOp.getResult(2);
        Value base2 = arith::AddIOp::create(b, loc, base, z);
        Value re_at = arith::MulIOp::create(b, loc, base2, c2);
        Value im_at = arith::AddIOp::create(b, loc, re_at, c1);
        Value re = load(c, {re_at});
        Value im = load(c, {im_at});
        Value first =
            arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, z, c0);
        Value middle =
            arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, z, h);
        Value middle_even = arith::AndIOp::create(b, loc, middle, even);
        Value single = arith::OrIOp::create(b, loc, first, middle_even);
        Value sums_p3m = arith::AndIOp::create(b, loc, sums, p3m);
        Value sums_plain = arith::XOrIOp::create(b, loc, p3m, yes);
        Value sums_other = arith::AndIOp::create(b, loc, sums, sums_plain);
        auto enOp = scf::IfOp::create(
            b, loc, TypeRange{realType, VectorType::get({9}, realType)},
            sums_other, true, true);
        {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(enOp.thenBlock());
          auto esCall = func::CallOp::create(
              b, loc, "mdrt.pme_point_sums",
              TypeRange{realType, VectorType::get({9}, realType)},
              ValueRange{re, im, m1, m2, m3, bc, factor, vd, single, e3, w3});
          Value es = esCall.getResult(0);
          Value ws = esCall.getResult(1);
          scf::YieldOp::create(b, loc, ValueRange{es, ws});
        }
        {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(enOp.elseBlock());
          auto epOp = scf::IfOp::create(
              b, loc, TypeRange{realType, VectorType::get({9}, realType)},
              sums_p3m, true, true);
          {
            OpBuilder::InsertionGuard guard(b);
            b.setInsertionPointToStart(epOp.thenBlock());
            Value c2p = arith::ConstantIndexOp::create(b, loc, 2);
            Value xa = arith::MulIOp::create(b, loc, a, c2p);
            Value xa0 = arith::AddIOp::create(b, loc, extra, xa);
            Value xa1 = arith::AddIOp::create(b, loc, xa0, c1);
            Value pbx = arith::AddIOp::create(b, loc, k1, accB);
            Value xb = arith::MulIOp::create(b, loc, pbx, c2p);
            Value xb0 = arith::AddIOp::create(b, loc, extra, xb);
            Value xb1 = arith::AddIOp::create(b, loc, xb0, c1);
            Value xz = arith::MulIOp::create(b, loc, pz, c2p);
            Value xz0 = arith::AddIOp::create(b, loc, extra, xz);
            Value xz1 = arith::AddIOp::create(b, loc, xz0, c1);
            Value ga1 = load(tables, {xa0});
            Value gs1 = load(tables, {xa1});
            Value ga2 = load(tables, {xb0});
            Value gs2 = load(tables, {xb1});
            Value ga3 = load(tables, {xz0});
            Value gs3 = load(tables, {xz1});
            Value pav = vector::FromElementsOp::create(
                b, loc, VectorType::get({3}, realType), ValueRange{m1, m2, m3});
            Value psv = vector::FromElementsOp::create(
                b, loc, VectorType::get({3}, realType),
                ValueRange{m1s, m2s, m3s});
            Value gav = vector::FromElementsOp::create(
                b, loc, VectorType::get({3}, realType),
                ValueRange{ga1, ga2, ga3});
            Value gsv = vector::FromElementsOp::create(
                b, loc, VectorType::get({3}, realType),
                ValueRange{gs1, gs2, gs3});
            Value msqp = arith::AddFOp::create(b, loc, m12s, m3s);
            Value originp = arith::CmpFOp::create(
                b, loc, arith::CmpFPredicate::OEQ, msqp, rzero);
            Value safep = arith::SelectOp::create(b, loc, originp, rone, msqp);
            Value inversep = arith::DivFOp::create(b, loc, rone, safep);
            auto esCall = func::CallOp::create(
                b, loc, "mdrt.pme_point_sums_p3m",
                TypeRange{realType, VectorType::get({9}, realType)},
                ValueRange{re, im, pav, psv, gav, gsv, bc, inversep, gaussr,
                           single, e3, w3});
            Value es = esCall.getResult(0);
            Value ws = esCall.getResult(1);
            scf::YieldOp::create(b, loc, ValueRange{es, ws});
          }
          {
            OpBuilder::InsertionGuard guard(b);
            b.setInsertionPointToStart(epOp.elseBlock());
            scf::YieldOp::create(b, loc, ValueRange{e3, w3});
          }
          Value ep = epOp.getResult(0);
          Value wp = epOp.getResult(1);
          scf::YieldOp::create(b, loc, ValueRange{ep, wp});
        }
        Value en = enOp.getResult(0);
        Value wn = enOp.getResult(1);
        Value reb = arith::MulFOp::create(b, loc, re, bc);
        Value imb = arith::MulFOp::create(b, loc, im, bc);
        store(reb, c, {re_at});
        store(imb, c, {im_at});
        scf::YieldOp::create(b, loc, ValueRange{en, wn});
      });
  Value e = eOp.getResult(0);
  Value w = eOp.getResult(1);
  Value e64 = convert(f64Type, e);
  Value w64 = convert(VectorType::get({9}, f64Type), w);
  func::ReturnOp::create(b, loc, ValueRange{e64, w64});
}

// As pme_convolve_row, in a triclinic cell, with the wave vectors
// k = H⁻¹ m, whose influence is not separable along the edges.
void HostPME::emitConvolveRowTriclinic() {
  OpBuilder::InsertionGuard guard(b);
  auto fn =
      declare("mdrt.pme_convolve_row_triclinic",
              b.getFunctionType(
                  {MemRefType::get({ShapedType::kDynamic}, realType),
                   MemRefType::get({ShapedType::kDynamic}, realType),
                   MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic},
                                   f64Type),
                   VectorType::get({6}, f64Type), f64Type, f64Type, boolType,
                   idxType, idxType, idxType, idxType, boolType},
                  {f64Type, VectorType::get({9}, f64Type)}));
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  Value c = entry->getArgument(0);
  Value tables = entry->getArgument(1);
  Value moduli = entry->getArgument(2);
  Value box = entry->getArgument(3);
  Value beta = entry->getArgument(4);
  Value coulomb = entry->getArgument(5);
  Value dispersion = entry->getArgument(6);
  Value k1 = entry->getArgument(7);
  Value k2 = entry->getArgument(8);
  Value k3 = entry->getArgument(9);
  Value row = entry->getArgument(10);
  Value sums = entry->getArgument(11);
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  Value c2 = arith::ConstantIndexOp::create(b, loc, 2);
  Value rzero =
      arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.0));
  Value rnone = arith::ConstantOp::create(
      b, loc, dense(VectorType::get({9}, realType), {0.0}));
  Value unit = arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 1.0));
  Value lx = vector::ExtractOp::create(b, loc, box, ArrayRef<int64_t>{0});
  Value ly = vector::ExtractOp::create(b, loc, box, ArrayRef<int64_t>{1});
  Value lz = vector::ExtractOp::create(b, loc, box, ArrayRef<int64_t>{2});
  Value lxy = arith::MulFOp::create(b, loc, lx, ly);
  Value volume = arith::MulFOp::create(b, loc, lxy, lz);
  auto h00Call = func::CallOp::create(
      b, loc, "mdrt.pme_inverse_cell",
      TypeRange{f64Type, f64Type, f64Type, f64Type, f64Type, f64Type},
      ValueRange{box});
  Value h00 = h00Call.getResult(0);
  Value h10 = h00Call.getResult(1);
  Value h11 = h00Call.getResult(2);
  Value h20 = h00Call.getResult(3);
  Value h21 = h00Call.getResult(4);
  Value h22 = h00Call.getResult(5);
  Value a = arith::DivUIOp::create(b, loc, row, k2);
  Value accB = arith::RemUIOp::create(b, loc, row, k2);
  Value h = arith::DivUIOp::create(b, loc, k3, c2);
  Value h1 = arith::AddIOp::create(b, loc, h, c1);
  Value odd = arith::RemUIOp::create(b, loc, k3, c2);
  Value even = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, odd, c0);
  Value i1 = func::CallOp::create(b, loc, "mdrt.pme_wave", TypeRange{f64Type},
                                  ValueRange{a, k1, unit})
                 .getResult(0);
  Value m1 = arith::MulFOp::create(b, loc, i1, h00);
  Value i1_h10 = arith::MulFOp::create(b, loc, i1, h10);
  Value i1_h20 = arith::MulFOp::create(b, loc, i1, h20);
  Value i2 = func::CallOp::create(b, loc, "mdrt.pme_wave", TypeRange{f64Type},
                                  ValueRange{accB, k2, unit})
                 .getResult(0);
  Value i2_h11 = arith::MulFOp::create(b, loc, i2, h11);
  Value m2 = arith::AddFOp::create(b, loc, i1_h10, i2_h11);
  Value i2_h21 = arith::MulFOp::create(b, loc, i2, h21);
  Value i12_z = arith::AddFOp::create(b, loc, i1_h20, i2_h21);
  Value m1r = convert(realType, m1);
  Value m2r = convert(realType, m2);
  Value mod1 = load(moduli, {c0, a});
  Value mod2 = load(moduli, {c1, accB});
  Value mod12 = arith::MulFOp::create(b, loc, mod1, mod2);
  Value base = arith::MulIOp::create(b, loc, row, h1);
  auto eOp = scf::ForOp::create(
      b, loc, c0, h1, c1, ValueRange{rzero, rnone},
      [&](OpBuilder &b, Location, Value z, ValueRange carried) {
        Value e3 = carried[0];
        Value w3 = carried[1];
        Value zi = arith::IndexCastOp::create(b, loc, i64Type, z);
        Value zf = arith::SIToFPOp::create(b, loc, f64Type, zi);
        Value i3_h22 = arith::MulFOp::create(b, loc, zf, h22);
        Value m3 = arith::AddFOp::create(b, loc, i12_z, i3_h22);
        Value m3r = convert(realType, m3);
        Value mod3 = load(moduli, {c2, z});
        Value mods = arith::MulFOp::create(b, loc, mod12, mod3);
        auto bcdCall = func::CallOp::create(
            b, loc, "mdrt.pme_influence", TypeRange{f64Type, f64Type, f64Type},
            ValueRange{m1, m2, m3, mods, moduli, volume, beta, coulomb,
                       dispersion});
        Value bcd = bcdCall.getResult(0);
        Value fd = bcdCall.getResult(1);
        Value vdd = bcdCall.getResult(2);
        Value bc = convert(realType, bcd);
        Value factor = convert(realType, fd);
        Value vd = convert(realType, vdd);
        Value base2 = arith::AddIOp::create(b, loc, base, z);
        Value re_at = arith::MulIOp::create(b, loc, base2, c2);
        Value im_at = arith::AddIOp::create(b, loc, re_at, c1);
        Value re = load(c, {re_at});
        Value im = load(c, {im_at});
        Value first =
            arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, z, c0);
        Value middle =
            arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, z, h);
        Value middle_even = arith::AndIOp::create(b, loc, middle, even);
        Value single = arith::OrIOp::create(b, loc, first, middle_even);
        auto enOp = scf::IfOp::create(
            b, loc, TypeRange{realType, VectorType::get({9}, realType)}, sums,
            true, true);
        {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(enOp.thenBlock());
          auto esCall = func::CallOp::create(
              b, loc, "mdrt.pme_point_sums",
              TypeRange{realType, VectorType::get({9}, realType)},
              ValueRange{re, im, m1r, m2r, m3r, bc, factor, vd, single, e3,
                         w3});
          Value es = esCall.getResult(0);
          Value ws = esCall.getResult(1);
          scf::YieldOp::create(b, loc, ValueRange{es, ws});
        }
        {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(enOp.elseBlock());
          scf::YieldOp::create(b, loc, ValueRange{e3, w3});
        }
        Value en = enOp.getResult(0);
        Value wn = enOp.getResult(1);
        Value reb = arith::MulFOp::create(b, loc, re, bc);
        Value imb = arith::MulFOp::create(b, loc, im, bc);
        store(reb, c, {re_at});
        store(imb, c, {im_at});
        scf::YieldOp::create(b, loc, ValueRange{en, wn});
      });
  Value e = eOp.getResult(0);
  Value w = eOp.getResult(1);
  Value e64 = convert(f64Type, e);
  Value w64 = convert(VectorType::get({9}, f64Type), w);
  func::ReturnOp::create(b, loc, ValueRange{e64, w64});
}

// Multiplies the half-complex transform `c` by the influence function
//
//   B C(m) = f / (π V) · exp(−π² m² / β²) / m² · B_1(k_1) B_2(k_2) B_3(k_3)
//
// with the factors B_a of each edge in the rows of `moduli`, and returns the
// energy and the virial
//
//   E = ½ Σ_m w(m) B C(m) |G(m)|²
//   W_ab = Σ_m E_m (δ_ab − 2 (1 / m² + π² / β²) m_a m_b)
//
// with w(m) = 2 for the points whose conjugate is not stored: a row of the
// grid an iteration of a parallel loop, the rows summed in f64 in the fixed
// order of D171.
void HostPME::emitConvolve() {
  OpBuilder::InsertionGuard guard(b);
  auto fn =
      declare("mdrt.pme_convolve",
              b.getFunctionType(
                  {MemRefType::get({ShapedType::kDynamic}, realType),
                   MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic},
                                   f64Type),
                   MemRefType::get({ShapedType::kDynamic}, realType),
                   VectorType::get({3}, f64Type), f64Type, f64Type, boolType,
                   idxType, idxType, idxType},
                  {f64Type, VectorType::get({9}, f64Type)}));
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  Value c = entry->getArgument(0);
  Value moduli = entry->getArgument(1);
  Value tables = entry->getArgument(2);
  Value box = entry->getArgument(3);
  Value beta = entry->getArgument(4);
  Value coulomb = entry->getArgument(5);
  Value dispersion = entry->getArgument(6);
  Value k1 = entry->getArgument(7);
  Value k2 = entry->getArgument(8);
  Value k3 = entry->getArgument(9);
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  Value zero = arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 0.0));
  Value none = arith::ConstantOp::create(
      b, loc, dense(VectorType::get({9}, f64Type), {0.0}));
  Value yes = arith::ConstantOp::create(b, loc, b.getBoolAttr(true));
  func::CallOp::create(
      b, loc, "mdrt.pme_tables", TypeRange{},
      ValueRange{moduli, tables, box, beta, coulomb, k1, k2, k3});
  Value rows = arith::MulIOp::create(b, loc, k1, k2);
  auto energyOp = scf::ParallelOp::create(
      b, loc, ValueRange{c0}, ValueRange{rows}, ValueRange{c1},
      ValueRange{zero, none},
      [&](OpBuilder &b, Location, ValueRange ivs, ValueRange) {
        Value r = ivs[0];
        auto erCall = func::CallOp::create(
            b, loc, "mdrt.pme_convolve_row",
            TypeRange{f64Type, VectorType::get({9}, f64Type)},
            ValueRange{c, tables, moduli, box, beta, coulomb, dispersion, k1,
                       k2, k3, r, yes});
        Value er = erCall.getResult(0);
        Value wr = erCall.getResult(1);
        sumReductions({er, wr});
      });
  Value energy = energyOp.getResult(0);
  Value virial = energyOp.getResult(1);
  func::ReturnOp::create(b, loc, ValueRange{energy, virial});
}

// As pme_convolve, for a step that writes neither the energy nor the
// virial: the product alone, as on a device.
void HostPME::emitScale() {
  OpBuilder::InsertionGuard guard(b);
  auto fn =
      declare("mdrt.pme_scale",
              b.getFunctionType(
                  {MemRefType::get({ShapedType::kDynamic}, realType),
                   MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic},
                                   f64Type),
                   MemRefType::get({ShapedType::kDynamic}, realType),
                   VectorType::get({3}, f64Type), f64Type, f64Type, boolType,
                   idxType, idxType, idxType},
                  {}));
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  Value c = entry->getArgument(0);
  Value moduli = entry->getArgument(1);
  Value tables = entry->getArgument(2);
  Value box = entry->getArgument(3);
  Value beta = entry->getArgument(4);
  Value coulomb = entry->getArgument(5);
  Value dispersion = entry->getArgument(6);
  Value k1 = entry->getArgument(7);
  Value k2 = entry->getArgument(8);
  Value k3 = entry->getArgument(9);
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  Value no = arith::ConstantOp::create(b, loc, b.getBoolAttr(false));
  func::CallOp::create(
      b, loc, "mdrt.pme_tables", TypeRange{},
      ValueRange{moduli, tables, box, beta, coulomb, k1, k2, k3});
  Value rows = arith::MulIOp::create(b, loc, k1, k2);
  scf::ParallelOp::create(
      b, loc, ValueRange{c0}, ValueRange{rows}, ValueRange{c1},
      [&](OpBuilder &b, Location, ValueRange ivs) {
        Value r = ivs[0];
        auto erCall = func::CallOp::create(
            b, loc, "mdrt.pme_convolve_row",
            TypeRange{f64Type, VectorType::get({9}, f64Type)},
            ValueRange{c, tables, moduli, box, beta, coulomb, dispersion, k1,
                       k2, k3, r, no});
        Value er = erCall.getResult(0);
        Value wr = erCall.getResult(1);
      });
  func::ReturnOp::create(b, loc);
}

// As pme_convolve, in a triclinic cell, with the wave vectors k = H⁻¹ m.
void HostPME::emitConvolveTriclinic() {
  OpBuilder::InsertionGuard guard(b);
  auto fn =
      declare("mdrt.pme_convolve_triclinic",
              b.getFunctionType(
                  {MemRefType::get({ShapedType::kDynamic}, realType),
                   MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic},
                                   f64Type),
                   MemRefType::get({ShapedType::kDynamic}, realType),
                   VectorType::get({6}, f64Type), f64Type, f64Type, boolType,
                   idxType, idxType, idxType},
                  {f64Type, VectorType::get({9}, f64Type)}));
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  Value c = entry->getArgument(0);
  Value moduli = entry->getArgument(1);
  Value tables = entry->getArgument(2);
  Value box = entry->getArgument(3);
  Value beta = entry->getArgument(4);
  Value coulomb = entry->getArgument(5);
  Value dispersion = entry->getArgument(6);
  Value k1 = entry->getArgument(7);
  Value k2 = entry->getArgument(8);
  Value k3 = entry->getArgument(9);
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  Value zero = arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 0.0));
  Value none = arith::ConstantOp::create(
      b, loc, dense(VectorType::get({9}, f64Type), {0.0}));
  Value yes = arith::ConstantOp::create(b, loc, b.getBoolAttr(true));
  Value rows = arith::MulIOp::create(b, loc, k1, k2);
  auto energyOp = scf::ParallelOp::create(
      b, loc, ValueRange{c0}, ValueRange{rows}, ValueRange{c1},
      ValueRange{zero, none},
      [&](OpBuilder &b, Location, ValueRange ivs, ValueRange) {
        Value r = ivs[0];
        auto erCall = func::CallOp::create(
            b, loc, "mdrt.pme_convolve_row_triclinic",
            TypeRange{f64Type, VectorType::get({9}, f64Type)},
            ValueRange{c, tables, moduli, box, beta, coulomb, dispersion, k1,
                       k2, k3, r, yes});
        Value er = erCall.getResult(0);
        Value wr = erCall.getResult(1);
        sumReductions({er, wr});
      });
  Value energy = energyOp.getResult(0);
  Value virial = energyOp.getResult(1);
  func::ReturnOp::create(b, loc, ValueRange{energy, virial});
}

// As pme_scale, in a triclinic cell.
void HostPME::emitScaleTriclinic() {
  OpBuilder::InsertionGuard guard(b);
  auto fn =
      declare("mdrt.pme_scale_triclinic",
              b.getFunctionType(
                  {MemRefType::get({ShapedType::kDynamic}, realType),
                   MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic},
                                   f64Type),
                   MemRefType::get({ShapedType::kDynamic}, realType),
                   VectorType::get({6}, f64Type), f64Type, f64Type, boolType,
                   idxType, idxType, idxType},
                  {}));
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  Value c = entry->getArgument(0);
  Value moduli = entry->getArgument(1);
  Value tables = entry->getArgument(2);
  Value box = entry->getArgument(3);
  Value beta = entry->getArgument(4);
  Value coulomb = entry->getArgument(5);
  Value dispersion = entry->getArgument(6);
  Value k1 = entry->getArgument(7);
  Value k2 = entry->getArgument(8);
  Value k3 = entry->getArgument(9);
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  Value no = arith::ConstantOp::create(b, loc, b.getBoolAttr(false));
  Value rows = arith::MulIOp::create(b, loc, k1, k2);
  scf::ParallelOp::create(
      b, loc, ValueRange{c0}, ValueRange{rows}, ValueRange{c1},
      [&](OpBuilder &b, Location, ValueRange ivs) {
        Value r = ivs[0];
        auto erCall = func::CallOp::create(
            b, loc, "mdrt.pme_convolve_row_triclinic",
            TypeRange{f64Type, VectorType::get({9}, f64Type)},
            ValueRange{c, tables, moduli, box, beta, coulomb, dispersion, k1,
                       k2, k3, r, no});
        Value er = erCall.getResult(0);
        Value wr = erCall.getResult(1);
      });
  func::ReturnOp::create(b, loc);
}
