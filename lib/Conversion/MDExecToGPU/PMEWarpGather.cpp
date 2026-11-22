// Interpolate the prolate window with one warp per particle [Liang2026].
#include "DevicePME.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;

template <bool tilted, bool useList, bool potentialOnly>
void DevicePME::emitWarpGather(StringRef symbol) {
  OpBuilder::InsertionGuard functionGuard(b);
  SmallVector<Type> inputs;
  inputs.push_back(globalMem({ShapedType::kDynamic}, chargeType));
  if constexpr (useList) {
    inputs.push_back(globalMem({ShapedType::kDynamic}, i32Type));
  }
  inputs.push_back(globalMem({ShapedType::kDynamic}, realType));
  inputs.push_back(globalMem({ShapedType::kDynamic}, realType));
  inputs.push_back(globalMem({ShapedType::kDynamic}, realType));
  if constexpr (!tilted) {
    inputs.push_back(VectorType::get({3}, f64Type));
  } else {
    inputs.push_back(VectorType::get({6}, f64Type));
  }
  inputs.append(3, idxType);
  if constexpr (!potentialOnly) {
    inputs.push_back(globalMem({ShapedType::kDynamic, 3}, realType));
  } else {
    inputs.push_back(globalMem({ShapedType::kDynamic}, realType));
  }
  auto fn = func::FuncOp::create(b, loc, symbol, b.getFunctionType(inputs, {}));
  fn.setPrivate();
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  unsigned arg = 0;
  Value q = entry->getArgument(arg++);
  Value list;
  if constexpr (useList) {
    list = entry->getArgument(arg++);
  }
  Value places = entry->getArgument(arg++);
  Value tables = entry->getArgument(arg++);
  Value phi = entry->getArgument(arg++);
  Value box = entry->getArgument(arg++);
  Value k1 = entry->getArgument(arg++);
  Value k2 = entry->getArgument(arg++);
  Value k3 = entry->getArgument(arg++);
  Value f = entry->getArgument(arg++);
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  Value c3 = arith::ConstantIndexOp::create(b, loc, 3);
  Value c4 = arith::ConstantIndexOp::create(b, loc, 4);
  arith::ConstantIndexOp::create(b, loc, 16);
  arith::ConstantIndexOp::create(b, loc, 32);
  Value c128 = arith::ConstantIndexOp::create(b, loc, 128);
  Value places_length = memref::DimOp::create(b, loc, places, c0);
  Value c6 = arith::ConstantIndexOp::create(b, loc, 6);
  Value n_all = arith::DivUIOp::create(b, loc, places_length, c6);
  Value k12 = arith::AddIOp::create(b, loc, k1, k2);
  Value k123 = arith::AddIOp::create(b, loc, k12, k3);
  Value pe = arith::MulIOp::create(b, loc, k123, c3);
  Value warps = arith::AddIOp::create(b, loc, n_all, c3);
  Value blocks = arith::DivUIOp::create(b, loc, warps, c4);
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
  Value k1i = arith::IndexCastOp::create(b, loc, i64Type, k1);
  Value k2i = arith::IndexCastOp::create(b, loc, i64Type, k2);
  Value k3i = arith::IndexCastOp::create(b, loc, i64Type, k3);
  Value k1f = arith::SIToFPOp::create(b, loc, f64Type, k1i);
  Value k2f = arith::SIToFPOp::create(b, loc, f64Type, k2i);
  Value k3f = arith::SIToFPOp::create(b, loc, f64Type, k3i);
  Value rx = arith::DivFOp::create(b, loc, k1f, lx);
  Value ry = arith::DivFOp::create(b, loc, k2f, ly);
  Value rz = arith::DivFOp::create(b, loc, k3f, lz);
  Value rxr = convert(realType, rx);
  Value ryr = convert(realType, ry);
  Value rzr = convert(realType, rz);
  {
    auto launch = gpu::LaunchOp::create(b, loc, blocks, c1, c1, c128, c1, c1);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;
    Value c0w = arith::ConstantIndexOp::create(b, loc, 0);
    Value c32w = arith::ConstantIndexOp::create(b, loc, 32);
    Value two_i = arith::ConstantIndexOp::create(b, loc, 2);
    Value three_i = arith::ConstantIndexOp::create(b, loc, 3);
    Value four_w = arith::ConstantIndexOp::create(b, loc, 4);
    Value warp = arith::DivUIOp::create(b, loc, tx, c32w);
    Value lane_x = arith::RemUIOp::create(b, loc, tx, c32w);
    Value lane = arith::IndexCastOp::create(b, loc, i32Type, lane_x);
    Value wb = arith::MulIOp::create(b, loc, bx, four_w);
    Value i;
    Value inside;
    if constexpr (!useList) {
      i = arith::AddIOp::create(b, loc, wb, warp);
      inside =
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, i, n_all);
    }
    Value slot;
    if constexpr (useList) {
      slot = arith::AddIOp::create(b, loc, wb, warp);
      inside =
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, slot, n_all);
    }
    auto insideBranch =
        scf::IfOp::create(b, loc, TypeRange{}, inside, true, false);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(insideBranch.thenBlock());
      Value listed;
      scf::IfOp is_listedBranch;
      if constexpr (useList) {
        listed = load(list, {slot});
        Value listed_none =
            arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 0));
        Value is_listed = arith::CmpIOp::create(
            b, loc, arith::CmpIPredicate::sge, listed, listed_none);
        is_listedBranch =
            scf::IfOp::create(b, loc, TypeRange{}, is_listed, true, true);
        b.setInsertionPointToStart(is_listedBranch.thenBlock());
        i = arith::IndexCastOp::create(b, loc, idxType, listed);
      }
      Value points =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(idxType, order));
      Value one = arith::ConstantIndexOp::create(b, loc, 1);
      Value lane_i = arith::IndexCastOp::create(b, loc, idxType, lane);
      Value three_p = arith::ConstantIndexOp::create(b, loc, 3);
      Value tp = arith::MulIOp::create(b, loc, three_p, points);
      Value active =
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, lane_i, tp);
      Value lane_a = arith::SelectOp::create(b, loc, active, lane_i, c0w);
      Value axis = arith::DivUIOp::create(b, loc, lane_a, points);
      Value j = arith::RemUIOp::create(b, loc, lane_a, points);
      Value fat0 = arith::AddIOp::create(b, loc, axis, three_i);
      Value fat1 = arith::MulIOp::create(b, loc, fat0, n_all);
      Value fat = arith::AddIOp::create(b, loc, fat1, i);
      Value frac = load(places, {fat});
      Value twor =
          arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 2.0));
      Value oner =
          arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 1.0));
      Value frac2 = arith::MulFOp::create(b, loc, frac, twor);
      Value t = arith::SubFOp::create(b, loc, frac2, oner);
      Value meta_d = arith::AddIOp::create(b, loc, pe, two_i);
      Value dr = load(tables, {meta_d});
      Value meta_n = load(tables, {pe});
      Value terms_i = arith::FPToSIOp::create(b, loc, i32Type, dr);
      Value terms = arith::IndexCastOp::create(b, loc, idxType, terms_i);
      Value n_i = arith::FPToSIOp::create(b, loc, i32Type, meta_n);
      Value n_s = arith::IndexCastOp::create(b, loc, idxType, n_i);
      Value n_2 = arith::AddIOp::create(b, loc, n_s, n_s);
      Value wb0 = arith::AddIOp::create(b, loc, pe, three_i);
      Value wbase = arith::AddIOp::create(b, loc, wb0, n_2);
      Value wvalue0 = clenshaw(wbase, t, one, terms, points, j, tables);
      Value zero_w =
          arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.0));
      Value wvalue = arith::SelectOp::create(b, loc, active, wvalue0, zero_w);
      Value wslope;
      if constexpr (!potentialOnly) {
        Value pdw = arith::MulIOp::create(b, loc, points, terms);
        Value sbase = arith::AddIOp::create(b, loc, wbase, pdw);
        Value wslope0 = clenshaw(sbase, t, one, terms, points, j, tables);
        wslope = arith::SelectOp::create(b, loc, active, wslope0, zero_w);
      }
      Value xsat = arith::ConstantIndexOp::create(b, loc, 0);
      Value xsn = arith::MulIOp::create(b, loc, n_all, xsat);
      Value xst = arith::AddIOp::create(b, loc, xsn, i);
      Value xstartr = load(places, {xst});
      Value xstart = arith::FPToSIOp::create(b, loc, i32Type, xstartr);
      Value ysat = arith::ConstantIndexOp::create(b, loc, 1);
      Value ysn = arith::MulIOp::create(b, loc, n_all, ysat);
      Value yst = arith::AddIOp::create(b, loc, ysn, i);
      Value ystartr = load(places, {yst});
      Value ystart = arith::FPToSIOp::create(b, loc, i32Type, ystartr);
      Value zsat = arith::ConstantIndexOp::create(b, loc, 2);
      Value zsn = arith::MulIOp::create(b, loc, n_all, zsat);
      Value zst = arith::AddIOp::create(b, loc, zsn, i);
      Value zstartr = load(places, {zst});
      Value zstart = arith::FPToSIOp::create(b, loc, i32Type, zstartr);
      Value k1t = arith::IndexCastOp::create(b, loc, i32Type, k1);
      Value k2t = arith::IndexCastOp::create(b, loc, i32Type, k2);
      Value k3t = arith::IndexCastOp::create(b, loc, i32Type, k3);
      Value pt =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, order));
      Value ppt = arith::MulIOp::create(b, loc, pt, pt);
      Value pt2 = arith::AddIOp::create(b, loc, pt, pt);
      Value thirty_two =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 32));
      Value zero32 =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 0));
      Value width =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 32));
      Value zr =
          arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.0));
      Value cz0 =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 0));
      Value czl0 = arith::AddIOp::create(b, loc, pt2, cz0);
      auto wz0Shuffle = gpu::ShuffleOp::create(b, loc, wvalue, czl0, width,
                                               gpu::ShuffleMode::IDX);
      Value wz0 = wz0Shuffle.getShuffleResult();

      Value dz0;
      if constexpr (!potentialOnly) {
        auto dz0Shuffle = gpu::ShuffleOp::create(b, loc, wslope, czl0, width,
                                                 gpu::ShuffleMode::IDX);
        dz0 = dz0Shuffle.getShuffleResult();
      }
      Value cz1 =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 1));
      Value czl1 = arith::AddIOp::create(b, loc, pt2, cz1);
      auto wz1Shuffle = gpu::ShuffleOp::create(b, loc, wvalue, czl1, width,
                                               gpu::ShuffleMode::IDX);
      Value wz1 = wz1Shuffle.getShuffleResult();

      Value dz1;
      if constexpr (!potentialOnly) {
        auto dz1Shuffle = gpu::ShuffleOp::create(b, loc, wslope, czl1, width,
                                                 gpu::ShuffleMode::IDX);
        dz1 = dz1Shuffle.getShuffleResult();
      }
      Value cz2 =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 2));
      Value czl2 = arith::AddIOp::create(b, loc, pt2, cz2);
      auto wz2Shuffle = gpu::ShuffleOp::create(b, loc, wvalue, czl2, width,
                                               gpu::ShuffleMode::IDX);
      Value wz2 = wz2Shuffle.getShuffleResult();

      Value dz2;
      if constexpr (!potentialOnly) {
        auto dz2Shuffle = gpu::ShuffleOp::create(b, loc, wslope, czl2, width,
                                                 gpu::ShuffleMode::IDX);
        dz2 = dz2Shuffle.getShuffleResult();
      }
      Value cz3 =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 3));
      Value czl3 = arith::AddIOp::create(b, loc, pt2, cz3);
      auto wz3Shuffle = gpu::ShuffleOp::create(b, loc, wvalue, czl3, width,
                                               gpu::ShuffleMode::IDX);
      Value wz3 = wz3Shuffle.getShuffleResult();

      Value dz3;
      if constexpr (!potentialOnly) {
        auto dz3Shuffle = gpu::ShuffleOp::create(b, loc, wslope, czl3, width,
                                                 gpu::ShuffleMode::IDX);
        dz3 = dz3Shuffle.getShuffleResult();
      }
      Value cz4 =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 4));
      Value czl4 = arith::AddIOp::create(b, loc, pt2, cz4);
      auto wz4Shuffle = gpu::ShuffleOp::create(b, loc, wvalue, czl4, width,
                                               gpu::ShuffleMode::IDX);
      Value wz4 = wz4Shuffle.getShuffleResult();

      Value dz4;
      if constexpr (!potentialOnly) {
        auto dz4Shuffle = gpu::ShuffleOp::create(b, loc, wslope, czl4, width,
                                                 gpu::ShuffleMode::IDX);
        dz4 = dz4Shuffle.getShuffleResult();
      }
      Value cz5 =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 5));
      Value czl5 = arith::AddIOp::create(b, loc, pt2, cz5);
      auto wz5Shuffle = gpu::ShuffleOp::create(b, loc, wvalue, czl5, width,
                                               gpu::ShuffleMode::IDX);
      Value wz5 = wz5Shuffle.getShuffleResult();

      Value dz5;
      if constexpr (!potentialOnly) {
        auto dz5Shuffle = gpu::ShuffleOp::create(b, loc, wslope, czl5, width,
                                                 gpu::ShuffleMode::IDX);
        dz5 = dz5Shuffle.getShuffleResult();
      }
      Value cz6 =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 6));
      Value czl6 = arith::AddIOp::create(b, loc, pt2, cz6);
      auto wz6Shuffle = gpu::ShuffleOp::create(b, loc, wvalue, czl6, width,
                                               gpu::ShuffleMode::IDX);
      Value wz6 = wz6Shuffle.getShuffleResult();

      Value dz6;
      if constexpr (!potentialOnly) {
        auto dz6Shuffle = gpu::ShuffleOp::create(b, loc, wslope, czl6, width,
                                                 gpu::ShuffleMode::IDX);
        dz6 = dz6Shuffle.getShuffleResult();
      }
      Value cz7 =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 7));
      Value czl7 = arith::AddIOp::create(b, loc, pt2, cz7);
      auto wz7Shuffle = gpu::ShuffleOp::create(b, loc, wvalue, czl7, width,
                                               gpu::ShuffleMode::IDX);
      Value wz7 = wz7Shuffle.getShuffleResult();

      Value dz7;
      if constexpr (!potentialOnly) {
        auto dz7Shuffle = gpu::ShuffleOp::create(b, loc, wslope, czl7, width,
                                                 gpu::ShuffleMode::IDX);
        dz7 = dz7Shuffle.getShuffleResult();
      }
      Value passes_n =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(idxType, order));
      Value passes_n2 = arith::MulIOp::create(b, loc, passes_n, passes_n);
      Value c31p = arith::ConstantIndexOp::create(b, loc, 31);
      Value passes_u = arith::AddIOp::create(b, loc, passes_n2, c31p);
      Value passes = arith::DivUIOp::create(b, loc, passes_u, c32w);
      Value sumx;
      Value sumy;
      Value sumz;
      if constexpr (!potentialOnly) {
        auto sumxOp = scf::ForOp::create(
            b, loc, c0w, passes, one, ValueRange{zr, zr, zr},
            [&](OpBuilder &b, Location, Value pass, ValueRange carried) {
              Value ax = carried[0];
              Value ay = carried[1];
              Value az = carried[2];
              Value pass32 = arith::IndexCastOp::create(b, loc, i32Type, pass);
              Value pass_base =
                  arith::MulIOp::create(b, loc, pass32, thirty_two);
              Value ab = arith::AddIOp::create(b, loc, pass_base, lane);
              Value in = arith::CmpIOp::create(
                  b, loc, arith::CmpIPredicate::slt, ab, ppt);
              Value abc = arith::SelectOp::create(b, loc, in, ab, zero32);
              Value pa = arith::DivUIOp::create(b, loc, abc, pt);
              Value pb = arith::RemUIOp::create(b, loc, abc, pt);
              Value pbs = arith::AddIOp::create(b, loc, pb, pt);
              auto wxShuffle = gpu::ShuffleOp::create(b, loc, wvalue, pa, width,
                                                      gpu::ShuffleMode::IDX);
              Value wx = wxShuffle.getShuffleResult();

              auto wyShuffle = gpu::ShuffleOp::create(
                  b, loc, wvalue, pbs, width, gpu::ShuffleMode::IDX);
              Value wy = wyShuffle.getShuffleResult();

              auto dxShuffle = gpu::ShuffleOp::create(b, loc, wslope, pa, width,
                                                      gpu::ShuffleMode::IDX);
              Value dx = dxShuffle.getShuffleResult();

              auto dyShuffle = gpu::ShuffleOp::create(
                  b, loc, wslope, pbs, width, gpu::ShuffleMode::IDX);
              Value dy = dyShuffle.getShuffleResult();

              auto row_sumsResult0Op = scf::IfOp::create(
                  b, loc, TypeRange{realType, realType}, in, true, true);
              {
                OpBuilder::InsertionGuard guard(b);
                b.setInsertionPointToStart(row_sumsResult0Op.thenBlock());
                Value g1s = arith::AddIOp::create(b, loc, xstart, pa);
                Value g1w = arith::SubIOp::create(b, loc, g1s, k1t);
                Value g1over = arith::CmpIOp::create(
                    b, loc, arith::CmpIPredicate::sge, g1s, k1t);
                Value g1 = arith::SelectOp::create(b, loc, g1over, g1w, g1s);
                Value g2s = arith::AddIOp::create(b, loc, ystart, pb);
                Value g2w = arith::SubIOp::create(b, loc, g2s, k2t);
                Value g2over = arith::CmpIOp::create(
                    b, loc, arith::CmpIPredicate::sge, g2s, k2t);
                Value g2 = arith::SelectOp::create(b, loc, g2over, g2w, g2s);
                Value row1 = arith::MulIOp::create(b, loc, g1, k2t);
                Value row = arith::AddIOp::create(b, loc, row1, g2);
                Value rowk = arith::MulIOp::create(b, loc, row, k3t);
                Value use0 = arith::CmpIOp::create(
                    b, loc, arith::CmpIPredicate::slt, cz0, pt);
                Value g30 =
                    md_alg::WrapOp::create(b, loc, zstart, cz0, k3t, true);
                Value p0 =
                    md_alg::GridReadOp::create(b, loc, phi, rowk, g30, use0);
                Value pw0 = arith::MulFOp::create(b, loc, p0, wz0);
                Value pwu0 = arith::SelectOp::create(b, loc, use0, pw0, zr);
                Value sz0 = arith::AddFOp::create(b, loc, zr, pwu0);
                Value pd0 = arith::MulFOp::create(b, loc, p0, dz0);
                Value pdu0 = arith::SelectOp::create(b, loc, use0, pd0, zr);
                Value sd0 = arith::AddFOp::create(b, loc, zr, pdu0);
                Value use1 = arith::CmpIOp::create(
                    b, loc, arith::CmpIPredicate::slt, cz1, pt);
                Value g31 =
                    md_alg::WrapOp::create(b, loc, zstart, cz1, k3t, true);
                Value p1 =
                    md_alg::GridReadOp::create(b, loc, phi, rowk, g31, use1);
                Value pw1 = arith::MulFOp::create(b, loc, p1, wz1);
                Value pwu1 = arith::SelectOp::create(b, loc, use1, pw1, zr);
                Value sz1 = arith::AddFOp::create(b, loc, sz0, pwu1);
                Value pd1 = arith::MulFOp::create(b, loc, p1, dz1);
                Value pdu1 = arith::SelectOp::create(b, loc, use1, pd1, zr);
                Value sd1 = arith::AddFOp::create(b, loc, sd0, pdu1);
                Value use2 = arith::CmpIOp::create(
                    b, loc, arith::CmpIPredicate::slt, cz2, pt);
                Value g32 =
                    md_alg::WrapOp::create(b, loc, zstart, cz2, k3t, true);
                Value p2 =
                    md_alg::GridReadOp::create(b, loc, phi, rowk, g32, use2);
                Value pw2 = arith::MulFOp::create(b, loc, p2, wz2);
                Value pwu2 = arith::SelectOp::create(b, loc, use2, pw2, zr);
                Value sz2 = arith::AddFOp::create(b, loc, sz1, pwu2);
                Value pd2 = arith::MulFOp::create(b, loc, p2, dz2);
                Value pdu2 = arith::SelectOp::create(b, loc, use2, pd2, zr);
                Value sd2 = arith::AddFOp::create(b, loc, sd1, pdu2);
                Value use3 = arith::CmpIOp::create(
                    b, loc, arith::CmpIPredicate::slt, cz3, pt);
                Value g33 =
                    md_alg::WrapOp::create(b, loc, zstart, cz3, k3t, true);
                Value p3 =
                    md_alg::GridReadOp::create(b, loc, phi, rowk, g33, use3);
                Value pw3 = arith::MulFOp::create(b, loc, p3, wz3);
                Value pwu3 = arith::SelectOp::create(b, loc, use3, pw3, zr);
                Value sz3 = arith::AddFOp::create(b, loc, sz2, pwu3);
                Value pd3 = arith::MulFOp::create(b, loc, p3, dz3);
                Value pdu3 = arith::SelectOp::create(b, loc, use3, pd3, zr);
                Value sd3 = arith::AddFOp::create(b, loc, sd2, pdu3);
                Value use4 = arith::CmpIOp::create(
                    b, loc, arith::CmpIPredicate::slt, cz4, pt);
                Value g34 =
                    md_alg::WrapOp::create(b, loc, zstart, cz4, k3t, true);
                Value p4 =
                    md_alg::GridReadOp::create(b, loc, phi, rowk, g34, use4);
                Value pw4 = arith::MulFOp::create(b, loc, p4, wz4);
                Value pwu4 = arith::SelectOp::create(b, loc, use4, pw4, zr);
                Value sz4 = arith::AddFOp::create(b, loc, sz3, pwu4);
                Value pd4 = arith::MulFOp::create(b, loc, p4, dz4);
                Value pdu4 = arith::SelectOp::create(b, loc, use4, pd4, zr);
                Value sd4 = arith::AddFOp::create(b, loc, sd3, pdu4);
                Value use5 = arith::CmpIOp::create(
                    b, loc, arith::CmpIPredicate::slt, cz5, pt);
                Value g35 =
                    md_alg::WrapOp::create(b, loc, zstart, cz5, k3t, true);
                Value p5 =
                    md_alg::GridReadOp::create(b, loc, phi, rowk, g35, use5);
                Value pw5 = arith::MulFOp::create(b, loc, p5, wz5);
                Value pwu5 = arith::SelectOp::create(b, loc, use5, pw5, zr);
                Value sz5 = arith::AddFOp::create(b, loc, sz4, pwu5);
                Value pd5 = arith::MulFOp::create(b, loc, p5, dz5);
                Value pdu5 = arith::SelectOp::create(b, loc, use5, pd5, zr);
                Value sd5 = arith::AddFOp::create(b, loc, sd4, pdu5);
                Value use6 = arith::CmpIOp::create(
                    b, loc, arith::CmpIPredicate::slt, cz6, pt);
                Value g36 =
                    md_alg::WrapOp::create(b, loc, zstart, cz6, k3t, true);
                Value p6 =
                    md_alg::GridReadOp::create(b, loc, phi, rowk, g36, use6);
                Value pw6 = arith::MulFOp::create(b, loc, p6, wz6);
                Value pwu6 = arith::SelectOp::create(b, loc, use6, pw6, zr);
                Value sz6 = arith::AddFOp::create(b, loc, sz5, pwu6);
                Value pd6 = arith::MulFOp::create(b, loc, p6, dz6);
                Value pdu6 = arith::SelectOp::create(b, loc, use6, pd6, zr);
                Value sd6 = arith::AddFOp::create(b, loc, sd5, pdu6);
                Value use7 = arith::CmpIOp::create(
                    b, loc, arith::CmpIPredicate::slt, cz7, pt);
                Value g37 =
                    md_alg::WrapOp::create(b, loc, zstart, cz7, k3t, true);
                Value p7 =
                    md_alg::GridReadOp::create(b, loc, phi, rowk, g37, use7);
                Value pw7 = arith::MulFOp::create(b, loc, p7, wz7);
                Value pwu7 = arith::SelectOp::create(b, loc, use7, pw7, zr);
                Value sz7 = arith::AddFOp::create(b, loc, sz6, pwu7);
                Value pd7 = arith::MulFOp::create(b, loc, p7, dz7);
                Value pdu7 = arith::SelectOp::create(b, loc, use7, pd7, zr);
                Value sd7 = arith::AddFOp::create(b, loc, sd6, pdu7);
                scf::YieldOp::create(b, loc, ValueRange{sz7, sd7});
              }
              {
                OpBuilder::InsertionGuard guard(b);
                b.setInsertionPointToStart(row_sumsResult0Op.elseBlock());
                scf::YieldOp::create(b, loc, ValueRange{zr, zr});
              }
              Value row_sumsResult0 = row_sumsResult0Op.getResult(0);
              Value row_sumsResult1 = row_sumsResult0Op.getResult(1);
              Value dxwy = arith::MulFOp::create(b, loc, dx, wy);
              Value wxdy = arith::MulFOp::create(b, loc, wx, dy);
              Value wxwy = arith::MulFOp::create(b, loc, wx, wy);
              Value tx_ = arith::MulFOp::create(b, loc, dxwy, row_sumsResult0);
              Value ty_ = arith::MulFOp::create(b, loc, wxdy, row_sumsResult0);
              Value tz_ = arith::MulFOp::create(b, loc, wxwy, row_sumsResult1);
              Value nx = arith::AddFOp::create(b, loc, ax, tx_);
              Value ny = arith::AddFOp::create(b, loc, ay, ty_);
              Value nz = arith::AddFOp::create(b, loc, az, tz_);
              scf::YieldOp::create(b, loc, ValueRange{nx, ny, nz});
            });
        sumx = sumxOp.getResult(0);
        sumy = sumxOp.getResult(1);
        sumz = sumxOp.getResult(2);
      }
      Value sum;
      if constexpr (potentialOnly) {
        auto sumOp = scf::ForOp::create(
            b, loc, c0w, passes, one, ValueRange{zr},
            [&](OpBuilder &b, Location, Value pass, ValueRange carried) {
              Value acc = carried[0];
              Value pass32 = arith::IndexCastOp::create(b, loc, i32Type, pass);
              Value pass_base =
                  arith::MulIOp::create(b, loc, pass32, thirty_two);
              Value ab = arith::AddIOp::create(b, loc, pass_base, lane);
              Value in = arith::CmpIOp::create(
                  b, loc, arith::CmpIPredicate::slt, ab, ppt);
              Value abc = arith::SelectOp::create(b, loc, in, ab, zero32);
              Value pa = arith::DivUIOp::create(b, loc, abc, pt);
              Value pb = arith::RemUIOp::create(b, loc, abc, pt);
              Value pbs = arith::AddIOp::create(b, loc, pb, pt);
              auto wxShuffle = gpu::ShuffleOp::create(b, loc, wvalue, pa, width,
                                                      gpu::ShuffleMode::IDX);
              Value wx = wxShuffle.getShuffleResult();

              auto wyShuffle = gpu::ShuffleOp::create(
                  b, loc, wvalue, pbs, width, gpu::ShuffleMode::IDX);
              Value wy = wyShuffle.getShuffleResult();

              auto row_sumsResult0Op = scf::IfOp::create(
                  b, loc, TypeRange{realType, realType}, in, true, true);
              {
                OpBuilder::InsertionGuard guard(b);
                b.setInsertionPointToStart(row_sumsResult0Op.thenBlock());
                Value g1s = arith::AddIOp::create(b, loc, xstart, pa);
                Value g1w = arith::SubIOp::create(b, loc, g1s, k1t);
                Value g1over = arith::CmpIOp::create(
                    b, loc, arith::CmpIPredicate::sge, g1s, k1t);
                Value g1 = arith::SelectOp::create(b, loc, g1over, g1w, g1s);
                Value g2s = arith::AddIOp::create(b, loc, ystart, pb);
                Value g2w = arith::SubIOp::create(b, loc, g2s, k2t);
                Value g2over = arith::CmpIOp::create(
                    b, loc, arith::CmpIPredicate::sge, g2s, k2t);
                Value g2 = arith::SelectOp::create(b, loc, g2over, g2w, g2s);
                Value row1 = arith::MulIOp::create(b, loc, g1, k2t);
                Value row = arith::AddIOp::create(b, loc, row1, g2);
                Value rowk = arith::MulIOp::create(b, loc, row, k3t);
                Value use0 = arith::CmpIOp::create(
                    b, loc, arith::CmpIPredicate::slt, cz0, pt);
                Value g30 =
                    md_alg::WrapOp::create(b, loc, zstart, cz0, k3t, true);
                Value p0 =
                    md_alg::GridReadOp::create(b, loc, phi, rowk, g30, use0);
                Value pw0 = arith::MulFOp::create(b, loc, p0, wz0);
                Value pwu0 = arith::SelectOp::create(b, loc, use0, pw0, zr);
                Value sz0 = arith::AddFOp::create(b, loc, zr, pwu0);
                Value use1 = arith::CmpIOp::create(
                    b, loc, arith::CmpIPredicate::slt, cz1, pt);
                Value g31 =
                    md_alg::WrapOp::create(b, loc, zstart, cz1, k3t, true);
                Value p1 =
                    md_alg::GridReadOp::create(b, loc, phi, rowk, g31, use1);
                Value pw1 = arith::MulFOp::create(b, loc, p1, wz1);
                Value pwu1 = arith::SelectOp::create(b, loc, use1, pw1, zr);
                Value sz1 = arith::AddFOp::create(b, loc, sz0, pwu1);
                Value use2 = arith::CmpIOp::create(
                    b, loc, arith::CmpIPredicate::slt, cz2, pt);
                Value g32 =
                    md_alg::WrapOp::create(b, loc, zstart, cz2, k3t, true);
                Value p2 =
                    md_alg::GridReadOp::create(b, loc, phi, rowk, g32, use2);
                Value pw2 = arith::MulFOp::create(b, loc, p2, wz2);
                Value pwu2 = arith::SelectOp::create(b, loc, use2, pw2, zr);
                Value sz2 = arith::AddFOp::create(b, loc, sz1, pwu2);
                Value use3 = arith::CmpIOp::create(
                    b, loc, arith::CmpIPredicate::slt, cz3, pt);
                Value g33 =
                    md_alg::WrapOp::create(b, loc, zstart, cz3, k3t, true);
                Value p3 =
                    md_alg::GridReadOp::create(b, loc, phi, rowk, g33, use3);
                Value pw3 = arith::MulFOp::create(b, loc, p3, wz3);
                Value pwu3 = arith::SelectOp::create(b, loc, use3, pw3, zr);
                Value sz3 = arith::AddFOp::create(b, loc, sz2, pwu3);
                Value use4 = arith::CmpIOp::create(
                    b, loc, arith::CmpIPredicate::slt, cz4, pt);
                Value g34 =
                    md_alg::WrapOp::create(b, loc, zstart, cz4, k3t, true);
                Value p4 =
                    md_alg::GridReadOp::create(b, loc, phi, rowk, g34, use4);
                Value pw4 = arith::MulFOp::create(b, loc, p4, wz4);
                Value pwu4 = arith::SelectOp::create(b, loc, use4, pw4, zr);
                Value sz4 = arith::AddFOp::create(b, loc, sz3, pwu4);
                Value use5 = arith::CmpIOp::create(
                    b, loc, arith::CmpIPredicate::slt, cz5, pt);
                Value g35 =
                    md_alg::WrapOp::create(b, loc, zstart, cz5, k3t, true);
                Value p5 =
                    md_alg::GridReadOp::create(b, loc, phi, rowk, g35, use5);
                Value pw5 = arith::MulFOp::create(b, loc, p5, wz5);
                Value pwu5 = arith::SelectOp::create(b, loc, use5, pw5, zr);
                Value sz5 = arith::AddFOp::create(b, loc, sz4, pwu5);
                Value use6 = arith::CmpIOp::create(
                    b, loc, arith::CmpIPredicate::slt, cz6, pt);
                Value g36 =
                    md_alg::WrapOp::create(b, loc, zstart, cz6, k3t, true);
                Value p6 =
                    md_alg::GridReadOp::create(b, loc, phi, rowk, g36, use6);
                Value pw6 = arith::MulFOp::create(b, loc, p6, wz6);
                Value pwu6 = arith::SelectOp::create(b, loc, use6, pw6, zr);
                Value sz6 = arith::AddFOp::create(b, loc, sz5, pwu6);
                Value use7 = arith::CmpIOp::create(
                    b, loc, arith::CmpIPredicate::slt, cz7, pt);
                Value g37 =
                    md_alg::WrapOp::create(b, loc, zstart, cz7, k3t, true);
                Value p7 =
                    md_alg::GridReadOp::create(b, loc, phi, rowk, g37, use7);
                Value pw7 = arith::MulFOp::create(b, loc, p7, wz7);
                Value pwu7 = arith::SelectOp::create(b, loc, use7, pw7, zr);
                Value sz7 = arith::AddFOp::create(b, loc, sz6, pwu7);
                scf::YieldOp::create(b, loc, ValueRange{sz7, zr});
              }
              {
                OpBuilder::InsertionGuard guard(b);
                b.setInsertionPointToStart(row_sumsResult0Op.elseBlock());
                scf::YieldOp::create(b, loc, ValueRange{zr, zr});
              }
              Value row_sumsResult0 = row_sumsResult0Op.getResult(0);

              Value wxy = arith::MulFOp::create(b, loc, wx, wy);
              Value term = arith::MulFOp::create(b, loc, wxy, row_sumsResult0);
              Value next = arith::AddFOp::create(b, loc, acc, term);
              scf::YieldOp::create(b, loc, ValueRange{next});
            });
        sum = sumOp.getResult(0);
      }
      Value off16 =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 16));
      gpu::ShuffleOp r0_16Shuffle;
      if constexpr (!potentialOnly) {
        r0_16Shuffle = gpu::ShuffleOp::create(b, loc, sumx, off16, width,
                                              gpu::ShuffleMode::XOR);
      } else {
        r0_16Shuffle = gpu::ShuffleOp::create(b, loc, sum, off16, width,
                                              gpu::ShuffleMode::XOR);
      }
      Value r0_16 = r0_16Shuffle.getShuffleResult();

      Value a0_16;
      Value a1_16;
      Value a2_16;
      if constexpr (!potentialOnly) {
        a0_16 = arith::AddFOp::create(b, loc, sumx, r0_16);
        auto r1_16Shuffle = gpu::ShuffleOp::create(b, loc, sumy, off16, width,
                                                   gpu::ShuffleMode::XOR);
        Value r1_16 = r1_16Shuffle.getShuffleResult();

        a1_16 = arith::AddFOp::create(b, loc, sumy, r1_16);
        auto r2_16Shuffle = gpu::ShuffleOp::create(b, loc, sumz, off16, width,
                                                   gpu::ShuffleMode::XOR);
        Value r2_16 = r2_16Shuffle.getShuffleResult();

        a2_16 = arith::AddFOp::create(b, loc, sumz, r2_16);
      } else {
        a0_16 = arith::AddFOp::create(b, loc, sum, r0_16);
      }
      Value off8 =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 8));
      auto r0_8Shuffle = gpu::ShuffleOp::create(b, loc, a0_16, off8, width,
                                                gpu::ShuffleMode::XOR);
      Value r0_8 = r0_8Shuffle.getShuffleResult();

      Value a0_8 = arith::AddFOp::create(b, loc, a0_16, r0_8);
      Value a1_8;
      Value a2_8;
      if constexpr (!potentialOnly) {
        auto r1_8Shuffle = gpu::ShuffleOp::create(b, loc, a1_16, off8, width,
                                                  gpu::ShuffleMode::XOR);
        Value r1_8 = r1_8Shuffle.getShuffleResult();

        a1_8 = arith::AddFOp::create(b, loc, a1_16, r1_8);
        auto r2_8Shuffle = gpu::ShuffleOp::create(b, loc, a2_16, off8, width,
                                                  gpu::ShuffleMode::XOR);
        Value r2_8 = r2_8Shuffle.getShuffleResult();

        a2_8 = arith::AddFOp::create(b, loc, a2_16, r2_8);
      }
      Value off4 =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 4));
      auto r0_4Shuffle = gpu::ShuffleOp::create(b, loc, a0_8, off4, width,
                                                gpu::ShuffleMode::XOR);
      Value r0_4 = r0_4Shuffle.getShuffleResult();

      Value a0_4 = arith::AddFOp::create(b, loc, a0_8, r0_4);
      Value a1_4;
      Value a2_4;
      if constexpr (!potentialOnly) {
        auto r1_4Shuffle = gpu::ShuffleOp::create(b, loc, a1_8, off4, width,
                                                  gpu::ShuffleMode::XOR);
        Value r1_4 = r1_4Shuffle.getShuffleResult();

        a1_4 = arith::AddFOp::create(b, loc, a1_8, r1_4);
        auto r2_4Shuffle = gpu::ShuffleOp::create(b, loc, a2_8, off4, width,
                                                  gpu::ShuffleMode::XOR);
        Value r2_4 = r2_4Shuffle.getShuffleResult();

        a2_4 = arith::AddFOp::create(b, loc, a2_8, r2_4);
      }
      Value off2 =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 2));
      auto r0_2Shuffle = gpu::ShuffleOp::create(b, loc, a0_4, off2, width,
                                                gpu::ShuffleMode::XOR);
      Value r0_2 = r0_2Shuffle.getShuffleResult();

      Value a0_2 = arith::AddFOp::create(b, loc, a0_4, r0_2);
      Value a1_2;
      Value a2_2;
      if constexpr (!potentialOnly) {
        auto r1_2Shuffle = gpu::ShuffleOp::create(b, loc, a1_4, off2, width,
                                                  gpu::ShuffleMode::XOR);
        Value r1_2 = r1_2Shuffle.getShuffleResult();

        a1_2 = arith::AddFOp::create(b, loc, a1_4, r1_2);
        auto r2_2Shuffle = gpu::ShuffleOp::create(b, loc, a2_4, off2, width,
                                                  gpu::ShuffleMode::XOR);
        Value r2_2 = r2_2Shuffle.getShuffleResult();

        a2_2 = arith::AddFOp::create(b, loc, a2_4, r2_2);
      }
      Value off1 =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 1));
      auto r0_1Shuffle = gpu::ShuffleOp::create(b, loc, a0_2, off1, width,
                                                gpu::ShuffleMode::XOR);
      Value r0_1 = r0_1Shuffle.getShuffleResult();

      Value a0_1 = arith::AddFOp::create(b, loc, a0_2, r0_1);
      Value a1_1;
      Value a2_1;
      if constexpr (!potentialOnly) {
        auto r1_1Shuffle = gpu::ShuffleOp::create(b, loc, a1_2, off1, width,
                                                  gpu::ShuffleMode::XOR);
        Value r1_1 = r1_1Shuffle.getShuffleResult();

        a1_1 = arith::AddFOp::create(b, loc, a1_2, r1_1);
        auto r2_1Shuffle = gpu::ShuffleOp::create(b, loc, a2_2, off1, width,
                                                  gpu::ShuffleMode::XOR);
        Value r2_1 = r2_1Shuffle.getShuffleResult();

        a2_1 = arith::AddFOp::create(b, loc, a2_2, r2_1);
      }
      Value lane0 =
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, lane, zero32);
      auto lane0Branch =
          scf::IfOp::create(b, loc, TypeRange{}, lane0, true, false);
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(lane0Branch.thenBlock());
        Value mq;
        if constexpr (!potentialOnly) {
          Value qs = load(q, {i});
          Value qi = convert(realType, qs);
          mq = arith::NegFOp::create(b, loc, qi);
        }
        Value fxv;
        Value fyv;
        Value fzv;
        if constexpr (!tilted && !potentialOnly) {
          Value sxq = arith::MulFOp::create(b, loc, mq, rxr);
          Value syq = arith::MulFOp::create(b, loc, mq, ryr);
          Value szq = arith::MulFOp::create(b, loc, mq, rzr);
          fxv = arith::MulFOp::create(b, loc, sxq, a0_1);
          fyv = arith::MulFOp::create(b, loc, syq, a1_1);
          fzv = arith::MulFOp::create(b, loc, szq, a2_1);
        }
        if constexpr (tilted && !potentialOnly) {
          Value k1fr0 = arith::SIToFPOp::create(b, loc, realType, k1t);
          Value k2fr0 = arith::SIToFPOp::create(b, loc, realType, k2t);
          Value k3fr0 = arith::SIToFPOp::create(b, loc, realType, k3t);
          Value ka1 = arith::MulFOp::create(b, loc, k1fr0, a0_1);
          Value ka2 = arith::MulFOp::create(b, loc, k2fr0, a1_1);
          Value ka3 = arith::MulFOp::create(b, loc, k3fr0, a2_1);
          Value fxh = arith::MulFOp::create(b, loc, h00r, ka1);
          Value fyh1 = arith::MulFOp::create(b, loc, h10r, ka1);
          Value fyh2 = arith::MulFOp::create(b, loc, h11r, ka2);
          Value fyh = arith::AddFOp::create(b, loc, fyh1, fyh2);
          Value fzh1 = arith::MulFOp::create(b, loc, h20r, ka1);
          Value fzh2 = arith::MulFOp::create(b, loc, h21r, ka2);
          Value fzh3 = arith::MulFOp::create(b, loc, h22r, ka3);
          Value fzh12 = arith::AddFOp::create(b, loc, fzh1, fzh2);
          Value fzh = arith::AddFOp::create(b, loc, fzh12, fzh3);
          fxv = arith::MulFOp::create(b, loc, mq, fxh);
          fyv = arith::MulFOp::create(b, loc, mq, fyh);
          fzv = arith::MulFOp::create(b, loc, mq, fzh);
        }
        if constexpr (!potentialOnly) {
          Value fxs = convert(realType, fxv);
          Value fys = convert(realType, fyv);
          Value fzs = convert(realType, fzv);
          Value i0 = arith::ConstantIndexOp::create(b, loc, 0);
          Value i1 = arith::ConstantIndexOp::create(b, loc, 1);
          Value i2 = arith::ConstantIndexOp::create(b, loc, 2);
          store(fxs, f, {i, i0});
          store(fys, f, {i, i1});
          store(fzs, f, {i, i2});
        } else {
          Value stored = convert(realType, a0_1);
          store(stored, f, {i});
        }
        scf::YieldOp::create(b, loc);
      }
      scf::YieldOp::create(b, loc);
      if constexpr (useList) {
        b.setInsertionPointAfter(is_listedBranch);
        {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(is_listedBranch.elseBlock());
          Value zfirst = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
                                               lane_x, c0w);
          auto zfirstBranch =
              scf::IfOp::create(b, loc, TypeRange{}, zfirst, true, false);
          {
            OpBuilder::InsertionGuard guard(b);
            b.setInsertionPointToStart(zfirstBranch.thenBlock());
            Value zminus = arith::ConstantOp::create(
                b, loc, b.getIntegerAttr(i32Type, -1));
            Value zplace32 = arith::SubIOp::create(b, loc, zminus, listed);
            Value zplace =
                arith::IndexCastOp::create(b, loc, idxType, zplace32);
            Value zc1 = arith::ConstantIndexOp::create(b, loc, 1);
            Value zc2 = arith::ConstantIndexOp::create(b, loc, 2);
            Value zero_f = arith::ConstantOp::create(
                b, loc, b.getFloatAttr(realType, 0.0));
            store(zero_f, f, {zplace, c0w});
            store(zero_f, f, {zplace, zc1});
            store(zero_f, f, {zplace, zc2});
            scf::YieldOp::create(b, loc);
          }
          scf::YieldOp::create(b, loc);
        }
        scf::YieldOp::create(b, loc);
      }
    }
    gpu::TerminatorOp::create(b, loc);
  }
  func::ReturnOp::create(b, loc);
}

template void DevicePME::emitWarpGather<false, false, false>(StringRef);
template void DevicePME::emitWarpGather<false, false, true>(StringRef);
template void DevicePME::emitWarpGather<true, false, false>(StringRef);
template void DevicePME::emitWarpGather<true, false, true>(StringRef);
template void DevicePME::emitWarpGather<false, true, false>(StringRef);
template void DevicePME::emitWarpGather<true, true, false>(StringRef);
