// Build the per-axis factors for deconvolution and reciprocal influence (D104).
#include "DevicePME.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;

template <Window window, Splitting splitting, bool tilted>
void DevicePME::emitTables(StringRef symbol, bool callbackInfo) {
  static_assert(!tilted || splitting != Splitting::P3M,
                "P3M requires a rectangular cell");
  StringRef blockFunction = blockSymbol<window, tilted>();
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
  if (callbackInfo)
    inputs.append({globalMem({ShapedType::kDynamic}, i64Type),
                   globalMem({ShapedType::kDynamic}, i32Type),
                   globalMem({ShapedType::kDynamic}, i64Type)});
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
  Value lx = extract(box, 0);
  Value ly = extract(box, 1);
  Value lz = extract(box, 2);
  if constexpr (tilted) {
    auto [h00, h10, h11, h20, h21, h22] = inverseCell(box, lx, ly, lz);
    convert(realType, h00);
    convert(realType, h10);
    convert(realType, h11);
    convert(realType, h20);
    convert(realType, h21);
    convert(realType, h22);
  }
  Value unit = arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 1.0));
  Value ilx = divf(unit, lx);
  Value ily = divf(unit, ly);
  Value ilz = divf(unit, lz);
  Value beta2 = mulf(beta, beta);
  Value gauss = divf(pi2, beta2);
  Value pi = arith::ConstantOp::create(
      b, loc, b.getFloatAttr(f64Type, 3.141592653589793));
  Value lxy = mulf(lx, ly);
  Value volume = mulf(lxy, lz);
  Value piv = mulf(pi, volume);
  Value prefactor;
  if constexpr (splitting == Splitting::Radial ||
                splitting == Splitting::Gaussian ||
                splitting == Splitting::P3M) {
    prefactor = divf(coulomb, piv);
  }
  if constexpr (splitting == Splitting::Dispersion) {
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
  }
  if constexpr (splitting == Splitting::Radial ||
                splitting == Splitting::Dispersion) {
    Value pib = divf(pi, beta);
    convert(realType, pib);
    Value sqrtpi = arith::ConstantOp::create(
        b, loc, b.getFloatAttr(f64Type, 1.7724538509055159));
    convert(realType, sqrtpi);
    Value six = arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 6.0));
    Value sixgauss = mulf(six, gauss);
    convert(realType, sixgauss);
  }
  Value k12a = addi(k1, k2);
  Value entries = addi(k12a, k3);
  Value blocks_entries;
  if constexpr (splitting != Splitting::P3M) {
    blocks_entries =
        func::CallOp::create(b, loc, blockFunction, TypeRange{idxType},
                             ValueRange{entries})
            .getResult(0);
  }
  Value blocks;
  if constexpr (splitting == Splitting::P3M) {
    Value threads = muli(entries, c8);
    blocks = func::CallOp::create(b, loc, blockFunction, TypeRange{idxType},
                                  ValueRange{threads})
                 .getResult(0);
  }
  {
    gpu::LaunchOp launch;
    if constexpr (splitting != Splitting::P3M) {
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
    Value base = muli(bx, c128);
    Value item;
    scf::IfOp insideBranch;
    if constexpr (splitting != Splitting::P3M) {
      item = addi(base, tx);
      Value inside = cmpi(arith::CmpIPredicate::ult, item, entries);
      insideBranch =
          scf::IfOp::create(b, loc, TypeRange{}, inside, true, false);
    }
    if (callbackInfo) {
      Value zero = arith::ConstantIndexOp::create(b, loc, 0);
      auto first = scf::IfOp::create(
          b, loc, cmpi(arith::CmpIPredicate::eq, item, zero), false);
      OpBuilder::InsertionGuard infoScope(b);
      b.setInsertionPointToStart(first.thenBlock());
      auto args = entry->getArguments();
      Value info = args[arg + 2];
      unsigned slot = 0;
      auto append = [&](Value word) {
        store(word, info, {arith::ConstantIndexOp::create(b, loc, slot++)});
      };
      // The ordered scale callback unflattens a 32-bit spectrum offset with
      // these two invariant reciprocals. cuFFT uses its plain plan for a
      // spectrum whose offsets need more bits.
      Value radix = arith::ConstantIntOp::create(b, loc, 1LL << 32, 64);
      Value two = arith::ConstantIndexOp::create(b, loc, 2);
      Value depth = addi(divui(k3, two), c1);
      auto reciprocal = [&](Value d) {
        return arith::CeilDivUIOp::create(b, loc, radix, indexCast(i64Type, d));
      };
      append(addi(muli(reciprocal(depth), radix), reciprocal(k2)));
      Value buffers[] = {args[arg], args[arg + 1], tables};
      for (Value buf : buffers) {
        auto ptr = memref::ExtractAlignedPointerAsIndexOp::create(b, loc, buf);
        append(indexCast(i64Type, ptr));
      }
      for (Value size : {k1, k2, k3})
        append(indexCast(i64Type, size));
      if constexpr (tilted) {
        auto [xx, yx, yy, zx, zy, zz] = inverseCell(box, lx, ly, lz);
        for (Value coeff : {xx, yx, yy, zx, zy, zz, gauss})
          append(arith::BitcastOp::create(b, loc, i64Type, coeff));
      }
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
      Value thread = addi(base, tx);
      Value item0 = divui(thread, c8t);
      Value lane = remui(thread, c8t);
      Value inside = cmpi(arith::CmpIPredicate::ult, item0, entries);
      Value last = subi(entries, c1t);
      item = select(inside, item0, last);
      Value k12 = addi(k1, k2);
      Value on_y = cmpi(arith::CmpIPredicate::uge, item, k1);
      Value on_z = cmpi(arith::CmpIPredicate::uge, item, k12);
      Value axis0 = select(on_y, c1t, c0t);
      axis = select(on_z, c2t, axis0);
      Value from0 = select(on_y, k1, c0t);
      from = select(on_z, k12, from0);
      Value kk0 = select(on_y, k2, k1);
      kk = select(on_z, k3, kk0);
      Value l0 = select(on_y, ily, ilx);
      Value l = select(on_z, ilz, l0);
      k = subi(item, from);
      Value half = divui(kk, c2t);
      Value ki = indexCast(i64Type, k);
      Value ni = indexCast(i64Type, kk);
      Value beyond = cmpi(arith::CmpIPredicate::ugt, k, half);
      Value shifted = subi(ki, ni);
      Value signedIndex = select(beyond, shifted, ki);
      Value sf = sitofp(f64Type, signedIndex);
      Value alias = cmpi(arith::CmpIPredicate::ult, lane, c5t);
      Value j = select(alias, lane, c0t);
      Value ji = indexCast(i64Type, j);
      Value two64 =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i64Type, 2));
      Value js = subi(ji, two64);
      Value jf = sitofp(f64Type, js);
      Value cf = sitofp(f64Type, ni);
      Value jk = mulf(jf, cf);
      Value nu = addf(sf, jk);
      Value mu = mulf(nu, l);
      Value mu2 = mulf(mu, mu);
      Value gmu = mulf(gauss, mu2);
      Value ngmu = arith::NegFOp::create(b, loc, gmu);
      Value g = math::ExpOp::create(b, loc, ngmu);
      Value first = muli(axis, c5t);
      Value row0 = addi(first, c3);
      Value row = addi(row0, j);
      Value u0 = load(moduli, {row, k});
      Value u = select(alias, u0, zero);
      Value t0 = mulf(u, mu);
      Value t1 = mulf(u, mu2);
      Value t2 = mulf(u, g);
      Value t3 = mulf(t2, mu);
      Value t4 = mulf(t2, mu2);
      Value width =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 32));
      Value o1 =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 1));
      Value o2 =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 2));
      Value o4 =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 4));
      auto sumAliases = [&](Value term) {
        for (Value offset : {o1, o2, o4}) {
          Value peer = laneValue(gpu::ShuffleMode::XOR, term, offset, width);
          term = addf(term, peer);
        }
        return term;
      };
      s0 = sumAliases(t0);
      s1 = sumAliases(t1);
      s2 = sumAliases(t2);
      s3 = sumAliases(t3);
      s4 = sumAliases(t4);
      Value leader = cmpi(arith::CmpIPredicate::eq, lane, c0t);
      Value writeResult = andi(inside, leader);
      writeResultBranch =
          scf::IfOp::create(b, loc, TypeRange{}, writeResult, true, false);
    }
    {
      OpBuilder::InsertionGuard guard(b);
      Value on_y;
      Value on_z;
      if constexpr (splitting != Splitting::P3M) {
        b.setInsertionPointToStart(insideBranch.thenBlock());
        c0t = arith::ConstantIndexOp::create(b, loc, 0);
        c1t = arith::ConstantIndexOp::create(b, loc, 1);
        c2t = arith::ConstantIndexOp::create(b, loc, 2);
        Value k12 = addi(k1, k2);
        on_y = cmpi(arith::CmpIPredicate::uge, item, k1);
        on_z = cmpi(arith::CmpIPredicate::uge, item, k12);
        Value axis0 = select(on_y, c1t, c0t);
        axis = select(on_z, c2t, axis0);
        Value from0 = select(on_y, k1, c0t);
        from = select(on_z, k12, from0);
        Value kk0 = select(on_y, k2, k1);
        kk = select(on_z, k3, kk0);
      }
      Value l;
      if constexpr (!tilted && splitting != Splitting::P3M) {
        Value l0 = select(on_y, ily, ilx);
        l = select(on_z, ilz, l0);
      }
      if constexpr (tilted) {
        l = arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 1.0));
      }
      Value m;
      Value m2;
      if constexpr (splitting != Splitting::P3M) {
        k = subi(item, from);
        m = waveNumber64(k, kk, l, c2t);
        m2 = mulf(m, m);
      }
      Value ex;
      if constexpr (splitting == Splitting::Gaussian && !tilted) {
        Value gm = mulf(gauss, m2);
        Value ngm = arith::NegFOp::create(b, loc, gm);
        ex = math::ExpOp::create(b, loc, ngm);
      }
      Value mod;
      if constexpr (splitting != Splitting::P3M) {
        // The window supplies its inverse squared transform on each axis.
        mod = load(moduli, {axis, k});
      }
      Value factor0;
      if constexpr (splitting == Splitting::Gaussian && !tilted) {
        factor0 = mulf(ex, mod);
      }
      if constexpr (tilted || splitting == Splitting::Radial ||
                    splitting == Splitting::Dispersion) {
        Value fzero =
            arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 0.0));
        factor0 = addf(mod, fzero);
      }
      Value gmean;
      Value gsquare;
      if constexpr (splitting == Splitting::P3M) {
        b.setInsertionPointToStart(writeResultBranch.thenBlock());
        Value unit1 =
            arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 1.0));
        Value inverse = load(moduli, {axis, k});
        factor0 = mulf(s2, inverse);
        Value some = cmpf(arith::CmpFPredicate::OGT, s2, zero);
        Value sgs = select(some, s2, unit1);
        Value gm0 = divf(s3, sgs);
        Value gq0 = divf(s4, sgs);
        gmean = select(some, gm0, zero);
        gsquare = select(some, gq0, zero);
      }
      Value first_axis = cmpi(arith::CmpIPredicate::eq, axis, c0t);
      Value scaled = mulf(factor0, prefactor);
      Value factor = select(first_axis, scaled, factor0);
      Value tstart = muli(from, c3);
      Value at0 = addi(tstart, k);
      Value at1 = addi(at0, kk);
      Value at2 = addi(at1, kk);
      Value mr;
      Value m2r;
      if constexpr (splitting != Splitting::P3M) {
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
        Value xstart0 = muli(entries, c3);
        Value xfrom = muli(from, c2t);
        Value xstart = addi(xstart0, xfrom);
        Value xat0 = addi(xstart, k);
        Value xat1 = addi(xat0, kk);
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

template void
DevicePME::emitTables<Window::Spline, Splitting::Gaussian, false>(StringRef,
                                                                  bool);
template void
DevicePME::emitTables<Window::Spline, Splitting::Dispersion, false>(StringRef,
                                                                    bool);
template void
DevicePME::emitTables<Window::Spline, Splitting::P3M, false>(StringRef, bool);
template void
DevicePME::emitTables<Window::Spline, Splitting::Gaussian, true>(StringRef,
                                                                 bool);
template void
DevicePME::emitTables<Window::Spline, Splitting::Dispersion, true>(StringRef,
                                                                   bool);
template void
DevicePME::emitTables<Window::Prolate, Splitting::Radial, false>(StringRef,
                                                                 bool);
template void
DevicePME::emitTables<Window::Prolate, Splitting::Radial, true>(StringRef,
                                                                bool);
