// Interpolate from the potential grid, using derivatives for forces
// [Essmann1995].
#include "DevicePME.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;

template <bool tilted, bool useList, bool potentialOnly, bool cachedWeights>
void DevicePME::emitGather(StringRef symbol) {
  StringRef blockFunction =
      tilted ? "mdrt_gpu_pme_blocks_triclinic" : "mdrt_gpu_pme_blocks";
  OpBuilder::InsertionGuard functionGuard(b);
  SmallVector<Type> inputs;
  inputs.push_back(globalMem({ShapedType::kDynamic, 3}, positionType));
  inputs.push_back(globalMem({ShapedType::kDynamic}, chargeType));
  if constexpr (useList) {
    inputs.push_back(globalMem({ShapedType::kDynamic}, i32Type));
  }
  inputs.push_back(globalMem({ShapedType::kDynamic}, realType));
  if constexpr (!tilted) {
    inputs.push_back(VectorType::get({3}, f64Type));
  } else {
    inputs.push_back(VectorType::get({6}, f64Type));
  }
  if constexpr ((!potentialOnly && !cachedWeights) ||
                (!tilted && potentialOnly)) {
    inputs.push_back(idxType);
  }
  if constexpr (cachedWeights) {
    inputs.push_back(globalMem({ShapedType::kDynamic}, realType));
  }
  inputs.append(3, idxType);
  if constexpr (!potentialOnly) {
    inputs.push_back(globalMem({ShapedType::kDynamic, 3}, realType));
  }
  if constexpr (tilted && potentialOnly) {
    inputs.push_back(idxType);
  }
  if constexpr (potentialOnly) {
    inputs.push_back(globalMem({ShapedType::kDynamic}, realType));
  }
  auto fn = func::FuncOp::create(b, loc, symbol, b.getFunctionType(inputs, {}));
  fn.setPrivate();
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  unsigned arg = 0;
  Value x = entry->getArgument(arg++);
  Value q = entry->getArgument(arg++);
  Value list;
  if constexpr (useList) {
    list = entry->getArgument(arg++);
  }
  Value phi = entry->getArgument(arg++);
  Value box = entry->getArgument(arg++);
  Value weights;
  if constexpr (cachedWeights) {
    weights = entry->getArgument(arg++);
  }
  Value k1 = entry->getArgument(arg++);
  Value k2 = entry->getArgument(arg++);
  Value k3 = entry->getArgument(arg++);
  if constexpr (!cachedWeights) {
    ++arg;
  }
  Value f = entry->getArgument(arg++);
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  Value c128 = arith::ConstantIndexOp::create(b, loc, 128);
  Value lx = extract(box, 0);
  Value ly = extract(box, 1);
  Value lz = extract(box, 2);
  Value h00;
  Value h11;
  Value h22;
  Value h10;
  Value h21;
  Value h20;
  Value h00r;
  Value h10r;
  Value h11r;
  Value h20r;
  Value h21r;
  Value h22r;
  if constexpr (tilted) {
    std::tie(h00, h10, h11, h20, h21, h22) = inverseCell(box, lx, ly, lz);
    h00r = convert(realType, h00);
    h10r = convert(realType, h10);
    h11r = convert(realType, h11);
    h20r = convert(realType, h20);
    h21r = convert(realType, h21);
    h22r = convert(realType, h22);
  }
  Value k1i = indexCast(i64Type, k1);
  Value k2i = indexCast(i64Type, k2);
  Value k3i = indexCast(i64Type, k3);
  Value k1f = sitofp(f64Type, k1i);
  Value k2f = sitofp(f64Type, k2i);
  Value k3f = sitofp(f64Type, k3i);
  Value rx = divf(k1f, lx);
  Value ry = divf(k2f, ly);
  Value rz = divf(k3f, lz);
  Value rxr = convert(realType, rx);
  Value ryr = convert(realType, ry);
  Value rzr = convert(realType, rz);
  Value k1fr;
  Value k2fr;
  Value k3fr;
  if constexpr (tilted) {
    k1fr = convert(realType, k1f);
    k2fr = convert(realType, k2f);
    k3fr = convert(realType, k3f);
  }
  Value ilx;
  Value ily;
  Value ilz;
  if constexpr (!cachedWeights) {
    Value unit =
        arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 1.0));
    ilx = divf(unit, lx);
    ily = divf(unit, ly);
    ilz = divf(unit, lz);
  }
  Value count = memref::DimOp::create(b, loc, x, c0);
  Value lanes_all =
      arith::ConstantOp::create(b, loc, b.getIntegerAttr(idxType, lanes));
  Value points = muli(count, lanes_all);
  Value last = subi(count, c1);
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
    Value item0 = addi(base, tx);
    Value inside = cmpi(arith::CmpIPredicate::ult, item0, points);
    Value lanes_l =
        arith::ConstantOp::create(b, loc, b.getIntegerAttr(idxType, lanes));
    Value c4l = arith::ConstantIndexOp::create(b, loc, 4);
    Value lane_of = remui(item0, lanes_l);
    Value last_first = muli(last, lanes_l);
    Value last_item = addi(last_first, lane_of);
    Value item = select(inside, item0, last_item);
    Value laneCount =
        arith::ConstantOp::create(b, loc, b.getIntegerAttr(idxType, lanes));
    Value i;
    if constexpr (!useList) {
      i = divui(item, laneCount);
    }
    Value slot;
    if constexpr (useList) {
      slot = divui(item, laneCount);
    }
    Value j3 = remui(item, laneCount);
    Value listed;
    Value is_listed;
    scf::IfOp warp_activeBranch;
    if constexpr (useList) {
      listed = load(list, {slot});
      Value listed_none =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 0));
      is_listed = cmpi(arith::CmpIPredicate::sge, listed, listed_none);
      Value c32w = arith::ConstantIndexOp::create(b, loc, 32);
      Value lane32 = remui(item0, c32w);
      Value wfirst = subi(item0, lane32);
      Value wslot = divui(wfirst, lanes_l);
      Value wslot_in = cmpi(arith::CmpIPredicate::ult, wslot, count);
      Value wslot_c = select(wslot_in, wslot, c0);
      Value warp_listed = load(list, {wslot_c});
      Value wnone =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 0));
      Value wlisted = cmpi(arith::CmpIPredicate::sge, warp_listed, wnone);
      Value warp_active = andi(wslot_in, wlisted);
      warp_activeBranch =
          scf::IfOp::create(b, loc, TypeRange{}, warp_active, true, false);
      b.setInsertionPointToStart(warp_activeBranch.thenBlock());
      Value listed_or = select(is_listed, listed, warp_listed);
      i = indexCast(idxType, listed_or);
    }
    Value i0 = arith::ConstantIndexOp::create(b, loc, 0);
    Value i1 = arith::ConstantIndexOp::create(b, loc, 1);
    Value i2 = arith::ConstantIndexOp::create(b, loc, 2);
    Value pointCount;
    if constexpr (!cachedWeights) {
      pointCount =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(idxType, order));
    }
    if constexpr (cachedWeights) {
      pointCount = arith::ConstantIndexOp::create(b, loc, 4);
    }
    Value wx = memref::AllocaOp::create(b, loc, MemRefType::get({8}, realType));
    Value wy = memref::AllocaOp::create(b, loc, MemRefType::get({8}, realType));
    Value wz = memref::AllocaOp::create(b, loc, MemRefType::get({8}, realType));
    Value dx = memref::AllocaOp::create(b, loc, MemRefType::get({8}, realType));
    Value dy = memref::AllocaOp::create(b, loc, MemRefType::get({8}, realType));
    Value dz = memref::AllocaOp::create(b, loc, MemRefType::get({8}, realType));
    Value xs;
    Value ys;
    Value zs;
    if constexpr (!cachedWeights) {
      xs = load(x, {i, i0});
      ys = load(x, {i, i1});
      zs = load(x, {i, i2});
    }
    Value qs = load(q, {i});
    Value xi;
    Value yi;
    Value zi;
    if constexpr (!cachedWeights) {
      xi = convert(f64Type, xs);
      yi = convert(f64Type, ys);
      zi = convert(f64Type, zs);
    }
    Value qi = convert(realType, qs);
    std::pair<Value, Value> xPlacement;
    if constexpr (!tilted && !cachedWeights) {
      xPlacement = placeAxis(xi, ilx, k1, pointCount);
    }
    Value fr_one;
    Value fr_z;
    Value fr_y;
    if constexpr (tilted && !cachedWeights) {
      fr_one = arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 1.0));
      fr_z = mulf(zi, h22);
      Value fr_y1 = mulf(yi, h11);
      Value fr_y2 = mulf(zi, h21);
      fr_y = addf(fr_y1, fr_y2);
      Value fr_x1 = mulf(xi, h00);
      Value fr_x2 = mulf(yi, h10);
      Value fr_x3 = mulf(zi, h20);
      Value fr_x12 = addf(fr_x1, fr_x2);
      Value fr_x = addf(fr_x12, fr_x3);
      xPlacement = placeAxis(fr_x, fr_one, k1, pointCount);
    }
    Value px_start;
    if constexpr (!cachedWeights) {
      px_start = xPlacement.first;
      Value px_w = xPlacement.second;
      splineWeights(px_w, pointCount, wx, dx);
    }
    std::pair<Value, Value> yPlacement;
    if constexpr (!tilted && !cachedWeights) {
      yPlacement = placeAxis(yi, ily, k2, pointCount);
    }
    if constexpr (tilted && !cachedWeights) {
      yPlacement = placeAxis(fr_y, fr_one, k2, pointCount);
    }
    Value py_start;
    if constexpr (!cachedWeights) {
      py_start = yPlacement.first;
      Value py_w = yPlacement.second;
      splineWeights(py_w, pointCount, wy, dy);
    }
    std::pair<Value, Value> zPlacement;
    if constexpr (!tilted && !cachedWeights) {
      zPlacement = placeAxis(zi, ilz, k3, pointCount);
    }
    if constexpr (tilted && !cachedWeights) {
      zPlacement = placeAxis(fr_z, fr_one, k3, pointCount);
    }
    Value pz_start;
    if constexpr (!cachedWeights) {
      pz_start = zPlacement.first;
      Value pz_w = zPlacement.second;
      splineWeights(pz_w, pointCount, wz, dz);
    }
    Value xstart;
    Value ystart;
    Value zstart;
    if constexpr (cachedWeights) {
      Value xsc = arith::ConstantIndexOp::create(b, loc, 0);
      Value xsn = muli(count, xsc);
      Value xsat = addi(xsn, i);
      Value xstart_r = load(weights, {xsat});
      xstart = fptosi(i32Type, xstart_r);
      Value xfc = arith::ConstantIndexOp::create(b, loc, 3);
      Value xfn = muli(count, xfc);
      Value xfat = addi(xfn, i);
      Value xfrac = load(weights, {xfat});
      splineWeights(xfrac, pointCount, wx, dx);
      Value ysc = arith::ConstantIndexOp::create(b, loc, 1);
      Value ysn = muli(count, ysc);
      Value ysat = addi(ysn, i);
      Value ystart_r = load(weights, {ysat});
      ystart = fptosi(i32Type, ystart_r);
      Value yfc = arith::ConstantIndexOp::create(b, loc, 4);
      Value yfn = muli(count, yfc);
      Value yfat = addi(yfn, i);
      Value yfrac = load(weights, {yfat});
      splineWeights(yfrac, pointCount, wy, dy);
      Value zsc = arith::ConstantIndexOp::create(b, loc, 2);
      Value zsn = muli(count, zsc);
      Value zsat = addi(zsn, i);
      Value zstart_r = load(weights, {zsat});
      zstart = fptosi(i32Type, zstart_r);
      Value zfc = arith::ConstantIndexOp::create(b, loc, 5);
      Value zfn = muli(count, zfc);
      Value zfat = addi(zfn, i);
      Value zfrac = load(weights, {zfat});
      splineWeights(zfrac, pointCount, wz, dz);
    }
    Value w3p_value = pickSpline(wz, j3, pointCount);
    Value d3p_value = pickSpline(dz, j3, pointCount);
    Value gzero =
        arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.0));
    Value j0 = arith::ConstantIndexOp::create(b, loc, 0);
    Value j1c = arith::ConstantIndexOp::create(b, loc, 1);
    Value g3;
    if constexpr (!cachedWeights) {
      g3 = md_alg::WrapOp::create(b, loc, pz_start, j3, k3, false);
    }
    Value k1t;
    Value k2t;
    Value k3t;
    if constexpr (cachedWeights) {
      k1t = indexCast(i32Type, k1);
      k2t = indexCast(i32Type, k2);
      k3t = indexCast(i32Type, k3);
      Value j3t = indexCast(i32Type, j3);
      g3 = md_alg::WrapOp::create(b, loc, zstart, j3t, k3t, false);
    }
    auto sumxOp = scf::ForOp::create(
        b, loc, j0, pointCount, j1c, ValueRange{gzero, gzero, gzero},
        [&](OpBuilder &b, Location, Value j1, ValueRange carried) {
          Value ax = carried[0];
          Value ay = carried[1];
          Value az = carried[2];
          Value g1s;
          Value g1w;
          Value g1over;
          if constexpr (!cachedWeights) {
            g1s = addi(px_start, j1);
            g1w = subi(g1s, k1);
            g1over = cmpi(arith::CmpIPredicate::uge, g1s, k1);
          }
          if constexpr (cachedWeights) {
            Value j1t = indexCast(i32Type, j1);
            g1s = addi(xstart, j1t);
            g1w = subi(g1s, k1t);
            g1over = cmpi(arith::CmpIPredicate::uge, g1s, k1t);
          }
          Value g1 = select(g1over, g1w, g1s);
          Value row1;
          if constexpr (cachedWeights) {
            row1 = muli(g1, k2t);
          }
          Value w1 = load(wx, {j1});
          Value d1 = load(dx, {j1});
          auto bx2Op = scf::ForOp::create(
              b, loc, j0, pointCount, j1c, ValueRange{ax, ay, az},
              [&](OpBuilder &b, Location, Value j2, ValueRange carried) {
                Value cx = carried[0];
                Value cy = carried[1];
                Value cz = carried[2];
                Value g2s;
                Value g2w;
                Value g2over;
                if constexpr (!cachedWeights) {
                  g2s = addi(py_start, j2);
                  g2w = subi(g2s, k2);
                  g2over = cmpi(arith::CmpIPredicate::uge, g2s, k2);
                }
                if constexpr (cachedWeights) {
                  Value j2t = indexCast(i32Type, j2);
                  g2s = addi(ystart, j2t);
                  g2w = subi(g2s, k2t);
                  g2over = cmpi(arith::CmpIPredicate::uge, g2s, k2t);
                }
                Value g2 = select(g2over, g2w, g2s);
                Value w2 = load(wy, {j2});
                Value d2 = load(dy, {j2});
                if constexpr (!cachedWeights) {
                  row1 = muli(g1, k2);
                }
                Value row = addi(row1, g2);
                Value gbase;
                if constexpr (!cachedWeights) {
                  gbase = muli(row, k3);
                }
                if constexpr (cachedWeights) {
                  gbase = muli(row, k3t);
                }
                Value p =
                    md_alg::GridReadOp::create(b, loc, phi, gbase, g3, Value{});
                Value d1w2 = mulf(d1, w2);
                Value w1d2 = mulf(w1, d2);
                Value w1w2 = mulf(w1, w2);
                Value px;
                if constexpr (!potentialOnly) {
                  px = mulf(d1w2, w3p_value);
                } else {
                  px = mulf(w1w2, w3p_value);
                }
                Value py = mulf(w1d2, w3p_value);
                Value pz = mulf(w1w2, d3p_value);
                Value vx = mulf(p, px);
                Value vy = mulf(p, py);
                Value vz = mulf(p, pz);
                Value nx = addf(cx, vx);
                Value ny = addf(cy, vy);
                Value nz = addf(cz, vz);
                scf::YieldOp::create(b, loc, ValueRange{nx, ny, nz});
              });
          Value bx2 = bx2Op.getResult(0);
          Value by2 = bx2Op.getResult(1);
          Value bz2 = bx2Op.getResult(2);
          scf::YieldOp::create(b, loc, ValueRange{bx2, by2, bz2});
        });
    Value sumx = sumxOp.getResult(0);
    Value sumy = sumxOp.getResult(1);
    Value sumz = sumxOp.getResult(2);
    Value width =
        arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 32));
    Value one32 =
        arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 1));
    Value two32 =
        arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 2));
    Value sx1 = laneValue(gpu::ShuffleMode::XOR, sumx, one32, width);

    Value sy1 = laneValue(gpu::ShuffleMode::XOR, sumy, one32, width);

    Value sz1 = laneValue(gpu::ShuffleMode::XOR, sumz, one32, width);

    Value tx1 = addf(sumx, sx1);
    Value ty1 = addf(sumy, sy1);
    Value tz1 = addf(sumz, sz1);
    Value sx2 = laneValue(gpu::ShuffleMode::XOR, tx1, two32, width);

    Value sy2 = laneValue(gpu::ShuffleMode::XOR, ty1, two32, width);

    Value sz2 = laneValue(gpu::ShuffleMode::XOR, tz1, two32, width);

    Value ux = addf(tx1, sx2);
    Value uy = addf(ty1, sy2);
    Value uz = addf(tz1, sz2);
    Value four32 =
        arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 4));
    Value eight_lanes = cmpi(arith::CmpIPredicate::ugt, lanes_l, c4l);
    Value sx4 = laneValue(gpu::ShuffleMode::XOR, ux, four32, width);

    Value sy4 = laneValue(gpu::ShuffleMode::XOR, uy, four32, width);

    Value sz4 = laneValue(gpu::ShuffleMode::XOR, uz, four32, width);

    Value zero_r =
        arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.0));
    Value hx = select(eight_lanes, sx4, zero_r);
    Value hy = select(eight_lanes, sy4, zero_r);
    Value hz = select(eight_lanes, sz4, zero_r);
    Value allx = addf(ux, hx);
    Value ally = addf(uy, hy);
    Value allz = addf(uz, hz);
    Value first = cmpi(arith::CmpIPredicate::eq, j3, c0);
    Value stores;
    if constexpr (!useList) {
      stores = andi(inside, first);
    } else {
      Value stores_any = andi(inside, first);
      stores = andi(stores_any, is_listed);
    }
    auto storesBranch =
        scf::IfOp::create(b, loc, TypeRange{}, stores, true, false);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(storesBranch.thenBlock());
      Value mq;
      if constexpr (!potentialOnly) {
        mq = arith::NegFOp::create(b, loc, qi);
      }
      Value fx64;
      Value fy64;
      Value fz64;
      if constexpr (!tilted && !potentialOnly) {
        Value sxq = mulf(mq, rxr);
        Value syq = mulf(mq, ryr);
        Value szq = mulf(mq, rzr);
        fx64 = mulf(sxq, allx);
        fy64 = mulf(syq, ally);
        fz64 = mulf(szq, allz);
      }
      if constexpr (tilted && !potentialOnly) {
        Value ka1 = mulf(k1fr, allx);
        Value ka2 = mulf(k2fr, ally);
        Value ka3 = mulf(k3fr, allz);
        Value fxh = mulf(h00r, ka1);
        Value fyh1 = mulf(h10r, ka1);
        Value fyh2 = mulf(h11r, ka2);
        Value fyh = addf(fyh1, fyh2);
        Value fzh1 = mulf(h20r, ka1);
        Value fzh2 = mulf(h21r, ka2);
        Value fzh3 = mulf(h22r, ka3);
        Value fzh12 = addf(fzh1, fzh2);
        Value fzh = addf(fzh12, fzh3);
        fx64 = mulf(mq, fxh);
        fy64 = mulf(mq, fyh);
        fz64 = mulf(mq, fzh);
      }
      if constexpr (!potentialOnly) {
        Value fxs = convert(realType, fx64);
        Value fys = convert(realType, fy64);
        Value fzs = convert(realType, fz64);
        store(fxs, f, {i, i0});
        store(fys, f, {i, i1});
        store(fzs, f, {i, i2});
      } else {
        Value pots = convert(realType, allx);
        store(pots, f, {i});
      }
      scf::YieldOp::create(b, loc);
    }
    if constexpr (useList) {
      scf::YieldOp::create(b, loc);
      b.setInsertionPointAfter(warp_activeBranch);
      Value ztrue = arith::ConstantOp::create(b, loc, b.getBoolAttr(true));
      Value zunlisted = xori(is_listed, ztrue);
      Value zfirst = cmpi(arith::CmpIPredicate::eq, lane_of, c0);
      Value zin = andi(inside, zfirst);
      Value zstores = andi(zin, zunlisted);
      auto zstoresBranch =
          scf::IfOp::create(b, loc, TypeRange{}, zstores, true, false);
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(zstoresBranch.thenBlock());
        Value zminus =
            arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, -1));
        Value zplace32 = subi(zminus, listed);
        Value zplace = indexCast(idxType, zplace32);
        Value zc1 = arith::ConstantIndexOp::create(b, loc, 1);
        Value zc2 = arith::ConstantIndexOp::create(b, loc, 2);
        Value zero_f =
            arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.0));
        store(zero_f, f, {zplace, c0});
        store(zero_f, f, {zplace, zc1});
        store(zero_f, f, {zplace, zc2});
        scf::YieldOp::create(b, loc);
      }
    }
    gpu::TerminatorOp::create(b, loc);
  }
  func::ReturnOp::create(b, loc);
}

template void DevicePME::emitGather<false, false, false, false>(StringRef);
template void DevicePME::emitGather<false, false, false, true>(StringRef);
template void DevicePME::emitGather<false, false, true, false>(StringRef);
template void DevicePME::emitGather<false, true, false, false>(StringRef);
template void DevicePME::emitGather<false, true, false, true>(StringRef);
template void DevicePME::emitGather<true, false, false, false>(StringRef);
template void DevicePME::emitGather<true, false, false, true>(StringRef);
template void DevicePME::emitGather<true, false, true, false>(StringRef);
template void DevicePME::emitGather<true, true, false, false>(StringRef);
template void DevicePME::emitGather<true, true, false, true>(StringRef);
