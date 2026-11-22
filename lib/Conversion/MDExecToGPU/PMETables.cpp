// Build the per-axis factors for deconvolution and reciprocal influence (D104).
#include "DevicePME.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;

template <Splitting splitting, bool tilted>
void DevicePME::emitTables(StringRef symbol) {
  StringRef blockFunction = splitting == Splitting::Radial
                                ? "mdrt_gpu_pme_prolate_blocks"
                            : tilted ? "mdrt_gpu_pme_blocks_triclinic"
                                     : "mdrt_gpu_pme_blocks";
  OpBuilder::InsertionGuard functionGuard(b);
  SmallVector<Type> inputs;
  inputs.push_back(
      globalMem({ShapedType::kDynamic, ShapedType::kDynamic}, f64Type));
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
  Value moduli = entry->getArgument(arg++);
  Value tables = entry->getArgument(arg++);
  Value box = entry->getArgument(arg++);
  Value beta = entry->getArgument(arg++);
  Value coulomb = entry->getArgument(arg++);
  Value k1 = entry->getArgument(arg++);
  Value k2 = entry->getArgument(arg++);
  Value k3 = entry->getArgument(arg++);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  Value c3 = arith::ConstantIndexOp::create(b, loc, 3);
  Value c8;
  if constexpr (splitting == Splitting::P3M) {
    c8 = arith::ConstantIndexOp::create(b, loc, 8);
  }
  Value c128 = arith::ConstantIndexOp::create(b, loc, 128);
  Value pi2 = arith::ConstantOp::create(
      b, loc, b.getFloatAttr(f64Type, 9.869604401089358));
  Value lx = vector::ExtractOp::create(b, loc, box, ArrayRef<int64_t>{0});
  Value ly = vector::ExtractOp::create(b, loc, box, ArrayRef<int64_t>{1});
  Value lz = vector::ExtractOp::create(b, loc, box, ArrayRef<int64_t>{2});
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
    convert(realType, h00);
    convert(realType, h10);
    convert(realType, h11);
    convert(realType, h20);
    convert(realType, h21);
    convert(realType, h22);
  }
  Value unit = arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 1.0));
  Value ilx = arith::DivFOp::create(b, loc, unit, lx);
  Value ily = arith::DivFOp::create(b, loc, unit, ly);
  Value ilz = arith::DivFOp::create(b, loc, unit, lz);
  Value beta2 = arith::MulFOp::create(b, loc, beta, beta);
  Value gauss = arith::DivFOp::create(b, loc, pi2, beta2);
  Value pi = arith::ConstantOp::create(
      b, loc, b.getFloatAttr(f64Type, 3.141592653589793));
  Value lxy = arith::MulFOp::create(b, loc, lx, ly);
  Value volume = arith::MulFOp::create(b, loc, lxy, lz);
  Value piv = arith::MulFOp::create(b, loc, pi, volume);
  Value prefactor;
  if constexpr (splitting == Splitting::Radial ||
                splitting == Splitting::Gaussian ||
                splitting == Splitting::P3M) {
    prefactor = arith::DivFOp::create(b, loc, coulomb, piv);
  }
  if constexpr (splitting == Splitting::Dispersion) {
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
  }
  if constexpr (splitting == Splitting::Radial ||
                splitting == Splitting::Dispersion) {
    Value pib = arith::DivFOp::create(b, loc, pi, beta);
    convert(realType, pib);
    Value sqrtpi = arith::ConstantOp::create(
        b, loc, b.getFloatAttr(f64Type, 1.7724538509055159));
    convert(realType, sqrtpi);
    Value six = arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 6.0));
    Value sixgauss = arith::MulFOp::create(b, loc, six, gauss);
    convert(realType, sixgauss);
  }
  Value k12a = arith::AddIOp::create(b, loc, k1, k2);
  Value entries = arith::AddIOp::create(b, loc, k12a, k3);
  Value blocks_entries;
  if constexpr (tilted || splitting == Splitting::Radial ||
                splitting == Splitting::Dispersion ||
                splitting == Splitting::Gaussian) {
    blocks_entries =
        func::CallOp::create(b, loc, blockFunction, TypeRange{idxType},
                             ValueRange{entries})
            .getResult(0);
  }
  Value blocks;
  if constexpr (splitting == Splitting::P3M) {
    Value threads = arith::MulIOp::create(b, loc, entries, c8);
    blocks = func::CallOp::create(b, loc, blockFunction, TypeRange{idxType},
                                  ValueRange{threads})
                 .getResult(0);
  }
  {
    gpu::LaunchOp launch;
    if constexpr (tilted || splitting == Splitting::Radial ||
                  splitting == Splitting::Dispersion ||
                  splitting == Splitting::Gaussian) {
      launch =
          gpu::LaunchOp::create(b, loc, blocks_entries, c1, c1, c128, c1, c1);
    }
    if constexpr (splitting == Splitting::P3M) {
      launch = gpu::LaunchOp::create(b, loc, blocks, c1, c1, c128, c1, c1);
    }
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;
    Value c0t;
    Value c1t;
    Value c2t;
    Value c5t;
    Value c8t;
    Value zero;
    if constexpr (splitting == Splitting::P3M) {
      c0t = arith::ConstantIndexOp::create(b, loc, 0);
      c1t = arith::ConstantIndexOp::create(b, loc, 1);
      c2t = arith::ConstantIndexOp::create(b, loc, 2);
      c5t = arith::ConstantIndexOp::create(b, loc, 5);
      c8t = arith::ConstantIndexOp::create(b, loc, 8);
      zero = arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 0.0));
    }
    Value base = arith::MulIOp::create(b, loc, bx, c128);
    Value item;
    scf::IfOp insideBranch;
    if constexpr (tilted || splitting == Splitting::Radial ||
                  splitting == Splitting::Dispersion ||
                  splitting == Splitting::Gaussian) {
      item = arith::AddIOp::create(b, loc, base, tx);
      Value inside = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult,
                                           item, entries);
      insideBranch =
          scf::IfOp::create(b, loc, TypeRange{}, inside, true, false);
    }
    Value axis;
    Value from;
    Value kk;
    Value k;
    Value s0;
    Value s1;
    Value s2;
    Value s3;
    Value s4;
    scf::IfOp writeResultBranch;
    if constexpr (splitting == Splitting::P3M) {
      Value thread = arith::AddIOp::create(b, loc, base, tx);
      Value item0 = arith::DivUIOp::create(b, loc, thread, c8t);
      Value lane = arith::RemUIOp::create(b, loc, thread, c8t);
      Value inside = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult,
                                           item0, entries);
      Value last = arith::SubIOp::create(b, loc, entries, c1t);
      item = arith::SelectOp::create(b, loc, inside, item0, last);
      Value k12 = arith::AddIOp::create(b, loc, k1, k2);
      Value on_y =
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::uge, item, k1);
      Value on_z =
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::uge, item, k12);
      Value axis0 = arith::SelectOp::create(b, loc, on_y, c1t, c0t);
      axis = arith::SelectOp::create(b, loc, on_z, c2t, axis0);
      Value from0 = arith::SelectOp::create(b, loc, on_y, k1, c0t);
      from = arith::SelectOp::create(b, loc, on_z, k12, from0);
      Value kk0 = arith::SelectOp::create(b, loc, on_y, k2, k1);
      kk = arith::SelectOp::create(b, loc, on_z, k3, kk0);
      Value l0 = arith::SelectOp::create(b, loc, on_y, ily, ilx);
      Value l = arith::SelectOp::create(b, loc, on_z, ilz, l0);
      k = arith::SubIOp::create(b, loc, item, from);
      Value half = arith::DivUIOp::create(b, loc, kk, c2t);
      Value ki = arith::IndexCastOp::create(b, loc, i64Type, k);
      Value ni = arith::IndexCastOp::create(b, loc, i64Type, kk);
      Value beyond =
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ugt, k, half);
      Value shifted = arith::SubIOp::create(b, loc, ki, ni);
      Value signedIndex = arith::SelectOp::create(b, loc, beyond, shifted, ki);
      Value sf = arith::SIToFPOp::create(b, loc, f64Type, signedIndex);
      Value alias =
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, lane, c5t);
      Value j = arith::SelectOp::create(b, loc, alias, lane, c0t);
      Value ji = arith::IndexCastOp::create(b, loc, i64Type, j);
      Value two64 =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i64Type, 2));
      Value js = arith::SubIOp::create(b, loc, ji, two64);
      Value jf = arith::SIToFPOp::create(b, loc, f64Type, js);
      Value cf = arith::SIToFPOp::create(b, loc, f64Type, ni);
      Value jk = arith::MulFOp::create(b, loc, jf, cf);
      Value nu = arith::AddFOp::create(b, loc, sf, jk);
      Value mu = arith::MulFOp::create(b, loc, nu, l);
      Value mu2 = arith::MulFOp::create(b, loc, mu, mu);
      Value gmu = arith::MulFOp::create(b, loc, gauss, mu2);
      Value ngmu = arith::NegFOp::create(b, loc, gmu);
      Value g = math::ExpOp::create(b, loc, ngmu);
      Value first = arith::MulIOp::create(b, loc, axis, c5t);
      Value row0 = arith::AddIOp::create(b, loc, first, c3);
      Value row = arith::AddIOp::create(b, loc, row0, j);
      Value u0 = load(moduli, {row, k});
      Value u = arith::SelectOp::create(b, loc, alias, u0, zero);
      Value t0 = arith::MulFOp::create(b, loc, u, mu);
      Value t1 = arith::MulFOp::create(b, loc, u, mu2);
      Value t2 = arith::MulFOp::create(b, loc, u, g);
      Value t3 = arith::MulFOp::create(b, loc, t2, mu);
      Value t4 = arith::MulFOp::create(b, loc, t2, mu2);
      Value width =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 32));
      Value o1 =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 1));
      Value o2 =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 2));
      Value o4 =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 4));
      auto t0aShuffle =
          gpu::ShuffleOp::create(b, loc, t0, o1, width, gpu::ShuffleMode::XOR);
      Value t0a = t0aShuffle.getShuffleResult();

      Value t0b = arith::AddFOp::create(b, loc, t0, t0a);
      auto t0cShuffle =
          gpu::ShuffleOp::create(b, loc, t0b, o2, width, gpu::ShuffleMode::XOR);
      Value t0c = t0cShuffle.getShuffleResult();

      Value t0d = arith::AddFOp::create(b, loc, t0b, t0c);
      auto t0eShuffle =
          gpu::ShuffleOp::create(b, loc, t0d, o4, width, gpu::ShuffleMode::XOR);
      Value t0e = t0eShuffle.getShuffleResult();

      s0 = arith::AddFOp::create(b, loc, t0d, t0e);
      auto t1aShuffle =
          gpu::ShuffleOp::create(b, loc, t1, o1, width, gpu::ShuffleMode::XOR);
      Value t1a = t1aShuffle.getShuffleResult();

      Value t1b = arith::AddFOp::create(b, loc, t1, t1a);
      auto t1cShuffle =
          gpu::ShuffleOp::create(b, loc, t1b, o2, width, gpu::ShuffleMode::XOR);
      Value t1c = t1cShuffle.getShuffleResult();

      Value t1d = arith::AddFOp::create(b, loc, t1b, t1c);
      auto t1eShuffle =
          gpu::ShuffleOp::create(b, loc, t1d, o4, width, gpu::ShuffleMode::XOR);
      Value t1e = t1eShuffle.getShuffleResult();

      s1 = arith::AddFOp::create(b, loc, t1d, t1e);
      auto t2aShuffle =
          gpu::ShuffleOp::create(b, loc, t2, o1, width, gpu::ShuffleMode::XOR);
      Value t2a = t2aShuffle.getShuffleResult();

      Value t2b = arith::AddFOp::create(b, loc, t2, t2a);
      auto t2cShuffle =
          gpu::ShuffleOp::create(b, loc, t2b, o2, width, gpu::ShuffleMode::XOR);
      Value t2c = t2cShuffle.getShuffleResult();

      Value t2d = arith::AddFOp::create(b, loc, t2b, t2c);
      auto t2eShuffle =
          gpu::ShuffleOp::create(b, loc, t2d, o4, width, gpu::ShuffleMode::XOR);
      Value t2e = t2eShuffle.getShuffleResult();

      s2 = arith::AddFOp::create(b, loc, t2d, t2e);
      auto t3aShuffle =
          gpu::ShuffleOp::create(b, loc, t3, o1, width, gpu::ShuffleMode::XOR);
      Value t3a = t3aShuffle.getShuffleResult();

      Value t3b = arith::AddFOp::create(b, loc, t3, t3a);
      auto t3cShuffle =
          gpu::ShuffleOp::create(b, loc, t3b, o2, width, gpu::ShuffleMode::XOR);
      Value t3c = t3cShuffle.getShuffleResult();

      Value t3d = arith::AddFOp::create(b, loc, t3b, t3c);
      auto t3eShuffle =
          gpu::ShuffleOp::create(b, loc, t3d, o4, width, gpu::ShuffleMode::XOR);
      Value t3e = t3eShuffle.getShuffleResult();

      s3 = arith::AddFOp::create(b, loc, t3d, t3e);
      auto t4aShuffle =
          gpu::ShuffleOp::create(b, loc, t4, o1, width, gpu::ShuffleMode::XOR);
      Value t4a = t4aShuffle.getShuffleResult();

      Value t4b = arith::AddFOp::create(b, loc, t4, t4a);
      auto t4cShuffle =
          gpu::ShuffleOp::create(b, loc, t4b, o2, width, gpu::ShuffleMode::XOR);
      Value t4c = t4cShuffle.getShuffleResult();

      Value t4d = arith::AddFOp::create(b, loc, t4b, t4c);
      auto t4eShuffle =
          gpu::ShuffleOp::create(b, loc, t4d, o4, width, gpu::ShuffleMode::XOR);
      Value t4e = t4eShuffle.getShuffleResult();

      s4 = arith::AddFOp::create(b, loc, t4d, t4e);
      Value leader =
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, lane, c0t);
      Value writeResult = arith::AndIOp::create(b, loc, inside, leader);
      writeResultBranch =
          scf::IfOp::create(b, loc, TypeRange{}, writeResult, true, false);
    }
    {
      OpBuilder::InsertionGuard guard(b);
      Value on_y;
      Value on_z;
      if constexpr (tilted || splitting == Splitting::Radial ||
                    splitting == Splitting::Dispersion ||
                    splitting == Splitting::Gaussian) {
        b.setInsertionPointToStart(insideBranch.thenBlock());
        c0t = arith::ConstantIndexOp::create(b, loc, 0);
        c1t = arith::ConstantIndexOp::create(b, loc, 1);
        c2t = arith::ConstantIndexOp::create(b, loc, 2);
        Value k12 = arith::AddIOp::create(b, loc, k1, k2);
        on_y =
            arith::CmpIOp::create(b, loc, arith::CmpIPredicate::uge, item, k1);
        on_z =
            arith::CmpIOp::create(b, loc, arith::CmpIPredicate::uge, item, k12);
        Value axis0 = arith::SelectOp::create(b, loc, on_y, c1t, c0t);
        axis = arith::SelectOp::create(b, loc, on_z, c2t, axis0);
        Value from0 = arith::SelectOp::create(b, loc, on_y, k1, c0t);
        from = arith::SelectOp::create(b, loc, on_z, k12, from0);
        Value kk0 = arith::SelectOp::create(b, loc, on_y, k2, k1);
        kk = arith::SelectOp::create(b, loc, on_z, k3, kk0);
      }
      Value l;
      if constexpr ((splitting == Splitting::Radial && !tilted) ||
                    (splitting == Splitting::Dispersion && !tilted) ||
                    (splitting == Splitting::Gaussian && !tilted)) {
        Value l0 = arith::SelectOp::create(b, loc, on_y, ily, ilx);
        l = arith::SelectOp::create(b, loc, on_z, ilz, l0);
      }
      if constexpr (tilted) {
        l = arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 1.0));
      }
      Value m;
      Value m2;
      if constexpr (tilted || splitting == Splitting::Radial ||
                    splitting == Splitting::Dispersion ||
                    splitting == Splitting::Gaussian) {
        k = arith::SubIOp::create(b, loc, item, from);
        m = waveNumber64(k, kk, l, c2t);
        m2 = arith::MulFOp::create(b, loc, m, m);
      }
      Value ex;
      if constexpr (splitting == Splitting::Gaussian && !tilted) {
        Value gm = arith::MulFOp::create(b, loc, gauss, m2);
        Value ngm = arith::NegFOp::create(b, loc, gm);
        ex = math::ExpOp::create(b, loc, ngm);
      }
      Value mod;
      if constexpr (tilted || splitting == Splitting::Radial ||
                    splitting == Splitting::Dispersion ||
                    splitting == Splitting::Gaussian) {
        mod = load(moduli, {axis, k});
      }
      Value factor0;
      if constexpr (splitting == Splitting::Gaussian && !tilted) {
        factor0 = arith::MulFOp::create(b, loc, ex, mod);
      }
      if constexpr (tilted || splitting == Splitting::Radial ||
                    splitting == Splitting::Dispersion) {
        Value fzero =
            arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 0.0));
        factor0 = arith::AddFOp::create(b, loc, mod, fzero);
      }
      Value gmean;
      Value gsquare;
      if constexpr (splitting == Splitting::P3M) {
        b.setInsertionPointToStart(writeResultBranch.thenBlock());
        Value unit1 =
            arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 1.0));
        Value inverse = load(moduli, {axis, k});
        factor0 = arith::MulFOp::create(b, loc, s2, inverse);
        Value some =
            arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OGT, s2, zero);
        Value sgs = arith::SelectOp::create(b, loc, some, s2, unit1);
        Value gm0 = arith::DivFOp::create(b, loc, s3, sgs);
        Value gq0 = arith::DivFOp::create(b, loc, s4, sgs);
        gmean = arith::SelectOp::create(b, loc, some, gm0, zero);
        gsquare = arith::SelectOp::create(b, loc, some, gq0, zero);
      }
      Value first_axis =
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, axis, c0t);
      Value scaled = arith::MulFOp::create(b, loc, factor0, prefactor);
      Value factor =
          arith::SelectOp::create(b, loc, first_axis, scaled, factor0);
      Value tstart = arith::MulIOp::create(b, loc, from, c3);
      Value at0 = arith::AddIOp::create(b, loc, tstart, k);
      Value at1 = arith::AddIOp::create(b, loc, at0, kk);
      Value at2 = arith::AddIOp::create(b, loc, at1, kk);
      Value mr;
      Value m2r;
      if constexpr (tilted || splitting == Splitting::Radial ||
                    splitting == Splitting::Dispersion ||
                    splitting == Splitting::Gaussian) {
        mr = convert(realType, m);
        m2r = convert(realType, m2);
      }
      if constexpr (splitting == Splitting::P3M) {
        mr = convert(realType, s0);
        m2r = convert(realType, s1);
      }
      Value fr = convert(realType, factor);
      store(mr, tables, {at0});
      store(m2r, tables, {at1});
      store(fr, tables, {at2});
      if constexpr (splitting == Splitting::P3M) {
        Value xstart0 = arith::MulIOp::create(b, loc, entries, c3);
        Value xfrom = arith::MulIOp::create(b, loc, from, c2t);
        Value xstart = arith::AddIOp::create(b, loc, xstart0, xfrom);
        Value xat0 = arith::AddIOp::create(b, loc, xstart, k);
        Value xat1 = arith::AddIOp::create(b, loc, xat0, kk);
        Value gmr = convert(realType, gmean);
        Value gqr = convert(realType, gsquare);
        store(gmr, tables, {xat0});
        store(gqr, tables, {xat1});
      }
      scf::YieldOp::create(b, loc);
    }
    gpu::TerminatorOp::create(b, loc);
  }
  func::ReturnOp::create(b, loc);
}

template void DevicePME::emitTables<Splitting::Gaussian, false>(StringRef);
template void DevicePME::emitTables<Splitting::Dispersion, false>(StringRef);
template void DevicePME::emitTables<Splitting::P3M, false>(StringRef);
template void DevicePME::emitTables<Splitting::Gaussian, true>(StringRef);
template void DevicePME::emitTables<Splitting::Dispersion, true>(StringRef);
template void DevicePME::emitTables<Splitting::Radial, false>(StringRef);
template void DevicePME::emitTables<Splitting::Radial, true>(StringRef);
