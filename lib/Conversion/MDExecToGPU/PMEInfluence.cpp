#include "DevicePME.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;

// Radial influence and virial derivative from the two Chebyshev series.
DevicePME::Influence DevicePME::radialInfluence(Value tables, Value pe,
                                                Value cc1, Value msqr,
                                                Value rone, Value t1, Value t2,
                                                Value t3) {
  Value pmeta_n = load(tables, {pe});
  Value pe1 = arith::AddIOp::create(b, loc, pe, cc1);
  Value pscale = load(tables, {pe1});
  Value pn_i = arith::FPToSIOp::create(b, loc, i32Type, pmeta_n);
  Value pn = arith::IndexCastOp::create(b, loc, idxType, pn_i);
  Value pmabs = math::SqrtOp::create(b, loc, msqr);
  Value px = arith::MulFOp::create(b, loc, pscale, pmabs);
  Value px2 = arith::MulFOp::create(b, loc, px, px);
  Value ptwo = arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 2.0));
  Value p2x2 = arith::MulFOp::create(b, loc, ptwo, px2);
  Value py = arith::SubFOp::create(b, loc, p2x2, rone);
  Value py2 = arith::AddFOp::create(b, loc, py, py);
  Value pvb0 = arith::ConstantIndexOp::create(b, loc, 3);
  Value pvbase = arith::AddIOp::create(b, loc, pe, pvb0);
  Value psbase = arith::AddIOp::create(b, loc, pvbase, pn);
  Value pz = arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.0));
  auto pb1Op = scf::ForOp::create(
      b, loc, cc1, pn, cc1, ValueRange{pz, pz, pz, pz},
      [&](OpBuilder &b, Location, Value pr, ValueRange carried) {
        Value pp1 = carried[0];
        Value pp2 = carried[1];
        Value pq1 = carried[2];
        Value pq2 = carried[3];
        Value pk = arith::SubIOp::create(b, loc, pn, pr);
        Value pva = arith::AddIOp::create(b, loc, pvbase, pk);
        Value psa = arith::AddIOp::create(b, loc, psbase, pk);
        Value pa = load(tables, {pva});
        Value pas = load(tables, {psa});
        Value pm1 = arith::MulFOp::create(b, loc, py2, pp1);
        Value ps1 = arith::SubFOp::create(b, loc, pm1, pp2);
        Value pn1 = arith::AddFOp::create(b, loc, ps1, pa);
        Value pm2 = arith::MulFOp::create(b, loc, py2, pq1);
        Value ps2 = arith::SubFOp::create(b, loc, pm2, pq2);
        Value pn2 = arith::AddFOp::create(b, loc, ps2, pas);
        scf::YieldOp::create(b, loc, ValueRange{pn1, pp1, pn2, pq1});
      });
  Value pb1 = pb1Op.getResult(0);
  Value pb2 = pb1Op.getResult(1);
  Value pc1 = pb1Op.getResult(2);
  Value pc2 = pb1Op.getResult(3);
  Value pa0 = load(tables, {pvbase});
  Value pas0 = load(tables, {psbase});
  Value pm01 = arith::MulFOp::create(b, loc, py, pb1);
  Value ps01 = arith::SubFOp::create(b, loc, pm01, pb2);
  Value pchi = arith::AddFOp::create(b, loc, ps01, pa0);
  Value pm02 = arith::MulFOp::create(b, loc, py, pc1);
  Value ps02 = arith::SubFOp::create(b, loc, pm02, pc2);
  Value pgchi = arith::AddFOp::create(b, loc, ps02, pas0);
  Value pband =
      arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLT, px, rone);
  Value pnonzero =
      arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OGT, msqr, pz);
  Value pin = arith::AndIOp::create(b, loc, pband, pnonzero);
  Value psafe = arith::SelectOp::create(b, loc, pnonzero, msqr, rone);
  Value pinv = arith::DivFOp::create(b, loc, rone, psafe);
  Value tt12 = arith::MulFOp::create(b, loc, t1, t2);
  Value tt123 = arith::MulFOp::create(b, loc, tt12, t3);
  Value pbc0 = arith::MulFOp::create(b, loc, tt123, pchi);
  Value pbc1 = arith::MulFOp::create(b, loc, pbc0, pinv);
  Value bc = arith::SelectOp::create(b, loc, pin, pbc1, pz);
  Value p2chi = arith::AddFOp::create(b, loc, pchi, pchi);
  Value pvd0 = arith::SubFOp::create(b, loc, p2chi, pgchi);
  Value pvd1 = arith::MulFOp::create(b, loc, tt123, pvd0);
  Value pinv2 = arith::MulFOp::create(b, loc, pinv, pinv);
  Value pvd2 = arith::MulFOp::create(b, loc, pvd1, pinv2);
  Value vb = arith::SelectOp::create(b, loc, pin, pvd2, pz);
  return {bc, vb, {}};
}

// Dispersion influence and its virial factor, including the zero mode.
DevicePME::Influence DevicePME::dispersionInfluence(Value msqr, Value pibr,
                                                    Value sqrtpir, Value rone,
                                                    Value t1, Value t2,
                                                    Value t3, Value sixgaussr) {
  Value dmabs = math::SqrtOp::create(b, loc, msqr);
  Value dbd = arith::MulFOp::create(b, loc, pibr, dmabs);
  Value dbd2 = arith::MulFOp::create(b, loc, dbd, dbd);
  Value dnbd2 = arith::NegFOp::create(b, loc, dbd2);
  Value debd = math::ExpOp::create(b, loc, dnbd2);
  Value derfc = math::ErfcOp::create(b, loc, dbd);
  Value dspb = arith::MulFOp::create(b, loc, sqrtpir, dbd);
  Value dsbd = arith::MulFOp::create(b, loc, dspb, derfc);
  Value dtwo = arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 2.0));
  Value dtwob2 = arith::MulFOp::create(b, loc, dtwo, dbd2);
  Value domt = arith::SubFOp::create(b, loc, rone, dtwob2);
  Value dfa = arith::MulFOp::create(b, loc, domt, debd);
  Value dfb = arith::MulFOp::create(b, loc, dtwob2, dsbd);
  Value dfd = arith::AddFOp::create(b, loc, dfa, dfb);
  Value tt12 = arith::MulFOp::create(b, loc, t1, t2);
  Value tt123 = arith::MulFOp::create(b, loc, tt12, t3);
  Value bc = arith::MulFOp::create(b, loc, tt123, dfd);
  Value dvd0 = arith::SubFOp::create(b, loc, debd, dsbd);
  Value dvd1 = arith::MulFOp::create(b, loc, sixgaussr, dvd0);
  Value vb = arith::MulFOp::create(b, loc, tt123, dvd1);
  return {bc, vb, {}};
}

// Gaussian influence and inverse squared wave number for the virial.
DevicePME::Influence DevicePME::gaussianInfluence(Value msqr, Value rzero,
                                                  Value rone, Value t1,
                                                  Value t2, Value t3) {
  Value origin =
      arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OEQ, msqr, rzero);
  Value safe = arith::SelectOp::create(b, loc, origin, rone, msqr);
  Value inverse = arith::DivFOp::create(b, loc, rone, safe);
  Value tt12 = arith::MulFOp::create(b, loc, t1, t2);
  Value tt123 = arith::MulFOp::create(b, loc, tt12, t3);
  Value bc0 = arith::MulFOp::create(b, loc, tt123, inverse);
  Value bc = arith::SelectOp::create(b, loc, origin, rzero, bc0);
  return {bc, inverse, origin};
}

// Gaussian influence and inverse squared wave number for the virial.
DevicePME::Influence DevicePME::gaussianInfluenceTriclinic(Value msqr,
                                                           Value rzero,
                                                           Value rone, Value t1,
                                                           Value t2, Value t3,
                                                           Value gaussr) {
  Value origin =
      arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OEQ, msqr, rzero);
  Value safe = arith::SelectOp::create(b, loc, origin, rone, msqr);
  Value inverse = arith::DivFOp::create(b, loc, rone, safe);
  Value gk = arith::MulFOp::create(b, loc, gaussr, msqr);
  Value ngk = arith::NegFOp::create(b, loc, gk);
  Value ek = math::ExpOp::create(b, loc, ngk);
  Value tt12 = arith::MulFOp::create(b, loc, t1, t2);
  Value tt123 = arith::MulFOp::create(b, loc, tt12, t3);
  Value tte = arith::MulFOp::create(b, loc, tt123, ek);
  Value bc0 = arith::MulFOp::create(b, loc, tte, inverse);
  Value bc = arith::SelectOp::create(b, loc, origin, rzero, bc0);
  return {bc, inverse, origin};
}
