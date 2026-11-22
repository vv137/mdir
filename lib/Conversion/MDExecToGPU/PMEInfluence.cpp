#include "DevicePME.h"
#include "mdir/Conversion/PMEC2Spectrum.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;

// Radial influence and virial derivative from the two Chebyshev series.
DevicePME::Influence DevicePME::radialInfluence(Value tables, Value pe,
                                                Value cc1, Value msqr,
                                                Value rone, Value t1, Value t2,
                                                Value t3) {
  Value pmeta_n = load(tables, {pe});
  Value pe1 = addi(pe, cc1);
  Value pscale = load(tables, {pe1});
  Value pn_i = fptosi(i32Type, pmeta_n);
  Value pn = indexCast(idxType, pn_i);
  Value pmabs = math::SqrtOp::create(b, loc, msqr);
  Value px = mulf(pscale, pmabs);
  Value ptwo = arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 2.0));
  Value pz = arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.0));
  Value pchi, pgchi, pband;
  if (order == 4) {
    Value start = addi(pe, arith::ConstantIndexOp::create(b, loc, 3));
    Value k = arith::NegFOp::create(b, loc, px);
    std::tie(pchi, pgchi) =
        mdir::prolate::emitC2Spectrum(b, loc, k, [&](bool derivative, Value i) {
          Value at = addi(start, i);
          return load(tables, {derivative ? addi(at, pn) : at});
        });
    pband = arith::ConstantOp::create(b, loc, b.getBoolAttr(true));
  } else {
    Value px2 = mulf(px, px);
    Value p2x2 = mulf(ptwo, px2);
    Value py = subf(p2x2, rone);
    Value py2 = addf(py, py);
    Value pvb0 = arith::ConstantIndexOp::create(b, loc, 3);
    Value pvbase = addi(pe, pvb0);
    Value psbase = addi(pvbase, pn);
    auto pb1Op = scf::ForOp::create(
        b, loc, cc1, pn, cc1, ValueRange{pz, pz, pz, pz},
        [&](OpBuilder &b, Location, Value pr, ValueRange carried) {
          Value pp1 = carried[0];
          Value pp2 = carried[1];
          Value pq1 = carried[2];
          Value pq2 = carried[3];
          Value pk = subi(pn, pr);
          Value pva = addi(pvbase, pk);
          Value psa = addi(psbase, pk);
          Value pa = load(tables, {pva});
          Value pas = load(tables, {psa});
          Value pm1 = mulf(py2, pp1);
          Value ps1 = subf(pm1, pp2);
          Value pn1 = addf(ps1, pa);
          Value pm2 = mulf(py2, pq1);
          Value ps2 = subf(pm2, pq2);
          Value pn2 = addf(ps2, pas);
          scf::YieldOp::create(b, loc, ValueRange{pn1, pp1, pn2, pq1});
        });
    Value pb1 = pb1Op.getResult(0);
    Value pb2 = pb1Op.getResult(1);
    Value pc1 = pb1Op.getResult(2);
    Value pc2 = pb1Op.getResult(3);
    Value pa0 = load(tables, {pvbase});
    Value pas0 = load(tables, {psbase});
    Value pm01 = mulf(py, pb1);
    Value ps01 = subf(pm01, pb2);
    pchi = addf(ps01, pa0);
    Value pm02 = mulf(py, pc1);
    Value ps02 = subf(pm02, pc2);
    pgchi = addf(ps02, pas0);
    pband = cmpf(arith::CmpFPredicate::OLT, px, rone);
  }
  Value pnonzero = cmpf(arith::CmpFPredicate::OGT, msqr, pz);
  Value pin = andi(pband, pnonzero);
  Value psafe = select(pnonzero, msqr, rone);
  Value pinv = divf(rone, psafe);
  Value tt12 = mulf(t1, t2);
  Value tt123 = mulf(tt12, t3);
  Value pbc0 = mulf(tt123, pchi);
  Value pbc1 = mulf(pbc0, pinv);
  Value bc = select(pin, pbc1, pz);
  Value p2chi = addf(pchi, pchi);
  Value pvd0 = subf(p2chi, pgchi);
  Value pvd1 = mulf(tt123, pvd0);
  Value pinv2 = mulf(pinv, pinv);
  Value pvd2 = mulf(pvd1, pinv2);
  Value vb = select(pin, pvd2, pz);
  return {bc, vb, {}};
}

// Dispersion influence and its virial factor, including the zero mode.
DevicePME::Influence DevicePME::dispersionInfluence(Value msqr, Value pibr,
                                                    Value sqrtpir, Value rone,
                                                    Value t1, Value t2,
                                                    Value t3, Value sixgaussr) {
  Value dmabs = math::SqrtOp::create(b, loc, msqr);
  Value dbd = mulf(pibr, dmabs);
  Value dbd2 = mulf(dbd, dbd);
  Value dnbd2 = arith::NegFOp::create(b, loc, dbd2);
  Value debd = math::ExpOp::create(b, loc, dnbd2);
  Value derfc = math::ErfcOp::create(b, loc, dbd);
  Value dspb = mulf(sqrtpir, dbd);
  Value dsbd = mulf(dspb, derfc);
  Value dtwo = arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 2.0));
  Value dtwob2 = mulf(dtwo, dbd2);
  Value domt = subf(rone, dtwob2);
  Value dfa = mulf(domt, debd);
  Value dfb = mulf(dtwob2, dsbd);
  Value dfd = addf(dfa, dfb);
  Value tt12 = mulf(t1, t2);
  Value tt123 = mulf(tt12, t3);
  Value bc = mulf(tt123, dfd);
  Value dvd0 = subf(debd, dsbd);
  Value dvd1 = mulf(sixgaussr, dvd0);
  Value vb = mulf(tt123, dvd1);
  return {bc, vb, {}};
}

// Gaussian influence and inverse squared wave number for the virial.
DevicePME::Influence DevicePME::gaussianInfluence(Value msqr, Value rzero,
                                                  Value rone, Value t1,
                                                  Value t2, Value t3) {
  Value origin = cmpf(arith::CmpFPredicate::OEQ, msqr, rzero);
  Value safe = select(origin, rone, msqr);
  Value inverse = divf(rone, safe);
  Value tt12 = mulf(t1, t2);
  Value tt123 = mulf(tt12, t3);
  Value bc0 = mulf(tt123, inverse);
  Value bc = select(origin, rzero, bc0);
  return {bc, inverse, origin};
}

// Gaussian influence and inverse squared wave number for the virial.
DevicePME::Influence DevicePME::gaussianInfluenceTriclinic(Value msqr,
                                                           Value rzero,
                                                           Value rone, Value t1,
                                                           Value t2, Value t3,
                                                           Value gaussr) {
  Value origin = cmpf(arith::CmpFPredicate::OEQ, msqr, rzero);
  Value safe = select(origin, rone, msqr);
  Value inverse = divf(rone, safe);
  Value gk = mulf(gaussr, msqr);
  Value ngk = arith::NegFOp::create(b, loc, gk);
  Value ek = math::ExpOp::create(b, loc, ngk);
  Value tt12 = mulf(t1, t2);
  Value tt123 = mulf(tt12, t3);
  Value tte = mulf(tt123, ek);
  Value bc0 = mulf(tte, inverse);
  Value bc = select(origin, rzero, bc0);
  return {bc, inverse, origin};
}
